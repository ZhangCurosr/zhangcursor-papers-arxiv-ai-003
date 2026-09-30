# XU-RS: Explaining Credal Width in Random-Set Language Models

David Achara, Maryam Sultana, Alexander D. Rast, Fabio Cuzzolin Institute for AI, Data Analysis and Systems (AIDAS) Oxford Brookes University

September 30, 2026

## Abstract

Uncertainty estimates tell us how unsure a model is, but not why. Without knowing which parts of an input influences a model’s uncertainty, we cannot tell whether that uncertainty score depends on input features that are relevant for the task. We study this problem in randomset classifiers built using pretrained language models. These classifiers assign probability to individual answers and to groups of answers, producing lower and upper probabilities for each answer; The diference between these probabilities, called credal width, is used to represent epistemic uncertainty about an answer arising from limited training data. We propose XU-RS, a framework that attributes an answer’s credal width to the input tokens (words or word pieces) supplied to a language model. XU-RS uses Expected Gradients (a standard feature attribution method) to estimate how input tokens contribute to credal width. The proposed framework is evaluated on a MedQA dataset using SmolLM3-3B and Llama-2-7B models, demonstrating that setting the embedding of a token ranked highly by XU-RS to zero (zero-masking) causes larger changes in credal width than zero-masking randomly selected tokens. In addition, we show that normalisation can cause other answer groups to influence an answer’s width, reveal how token attribution can mask numerical errors, and provide diagnostic checks to verify whether a token ranked highly by XU-RS meaningfully explains model uncertainty.

## 1 Introduction

Uncertainty quantification Kong et al. (2023); Cuzzolin (2021b) accompanies a model’s predictions with estimates of their uncertainty (Hüllermeier & Waegeman, 2021), which can be distinguished into either aleatoric or epistemic Swiler et al. (February 2009); Oren et al. (2022); Li et al. (2025) uncertainty Walters et al. (2023); Bickford Smith et al. (2025). These estimates do not, by themselves, explain which parts of the input influence them. Understanding this dependence is important when examining whether the features influencing uncertainty are actually meaningful for the task. We investigate this problem for random-set Matheron (1975); Molchanov (2005); Nguyen (2006); Cuzzolin (2023) classifiers built using pretrained language models.

![](images/76db5e1c8b70f648dff98aba9c1d70ce435821bf6a014d7852046668779da3e6.jpg)  
Figure 1: XU-RS predicts answer intervals and attributes a selected answer’s width to input tokens relative to reference prompts. The example assigns C interval [0.1, 0.9] and width 0.8.

Consider a medical question with four answer options: {A, B, C, D}. A conventional classifier assigns a probability to each option. This specifies how probability is divided between the options, but does not by itself express how certain the model is about that division. A random-set classifier Manchingal & Cuzzolin (2022) can instead assign probability jointly to a set of options Wang et al. (2026c), leaving its division among them unspecified (Manchingal et al., 2023, 2025d; Mubashar et al., 2025). This flexibility is intended to represent incomplete knowledge about the answer probabilities, and requires novel ways of assessing performance Manchingal et al. (2025c).

For example, suppose the random-set classifier assigns probability 0.1 to B alone, 0.1 to C alone, and 0.8 to the set containing B and C, with zero assigned to all other sets. For answer C, 0.1 is committed to that answer alone, while another 0.8 could belong to either B or C. C’s lower probability is therefore 0.1 and its upper probability is 0.9. The diference, 0.9 − 0.1 = 0.8, is called credal width. The random-set classifier measures the probability left uncommitted between C and its alternatives (Figure 1).

Two models can report the same width while responding to diferent features: one to conflicting evidence and another to the order of the answer options. We therefore ask: which parts of a question and its answer options influence the width reported for a selected answer? The explanation target is uncertainty, rather than the score used to choose an answer.

We introduce XU-RS to explain the widths produced by these classifiers. We evaluate XU-RS on four-option random set classifiers adapted from SmolLM3- 3B and Llama-2-7B (Hugging Face, 2025; Touvron et al., 2023). A trained output layer estimates lower probabilities for answer sets. These estimates are converted into probabilities assigned to the sets, from which the answer intervals are calculated. The models therefore classify answer options rather than generate free-form text.

XU-RS uses Expected Gradients (Erion et al., 2021), a standard featureattribution method, to assign scores to input tokens (words or word pieces). The method explains how an output changes from a specified starting input to the input being studied. This starting input is called a reference, or baseline: it defines the diference that the token contributions are intended to explain.

In our MedQA experiments, reference prompts contain other questions and answer options. We explain the diference between the selected answer’s width on the original input and its average width on the references supplied to the calculation. Expected Gradients gradually changes the references’ numerical token representations towards those of the original prompt, measuring how the selected answer’s width responds along the way. Section 3 explains how these references are prepared.

Our contributions are threefold. First, we develop and test a framework for attributing credal width to input tokens. On 100 MedQA questions per model (Jin et al., 2021), setting the embedding of tokens ranked highest by XU-RS (zero-masking) produces larger absolute width changes than zero-masking randomly selected tokens. Second, we derive how converting classifier outputs into valid probabilities makes an answer’s width depend on values assigned to sets excluding that answer. Third, we provide numerical and reference-preparation checks: controlled examples show that stable rankings can conceal inaccurate attribution values, and that preparing a reference can alter the width diference even when attribution accurately accounts for that altered diference.

## 2 Related work

Random-set and credal prediction. Random-set models, with the approach called ‘epistemic AI’ (Cuzzolin & Sultana (2024); Manchingal et al. (2024, 2025b)), allow probability to be assigned to sets of possible outcomes without specifying how it is divided among their members (Shafer, 1976; Cuzzolin, 2021a). Random-Set Neural Networks (RS-NN) implement this representation for classification, including text classification (Manchingal et al., 2025d). Random-Set Large Language Models (RS-LLM) extend it to language generation, predicting over sets of possible next tokens (Mubashar et al., 2025). Other approaches obtain lower and upper probabilities without predicting probabilities for answer sets, for instance those using credal sets, i.e., convex sets of distributions Levi (1980); Caprio et al. (2025). Credal Deep Ensembles combine networks trained to predict probability bounds (Wang et al., 2024, 2026b,a), while the Credal Wrapper constructs bounds from the predictions of Bayesian neural networks or deep ensembles (Wang et al., 2025b), using the intersection probability (Cuzzolin (2009, 2007)) as pointwise estimator. Credal INNs Wang et al. (2025d) combine the expressive power of interval neural networks with credal representations. For language models, Yang et al. (2026) investigate prompting models to report lower and upper probabilities for possible answers, as well as constructing intervals from multiple reported predictions. The credal approach has been recently extended to large language models Manchingal et al. (2026), generative adversarial networks (Mubashar & Cuzzolin (2026)) and to a wrapper applicable to Bayesian NN (Sultana et al. (2025)). Our classifiers follow the random-set approach of RS-NN and RS-LLM (Mubashar et al. (2025)), but predict over four answer options rather than vocabulary tokens. Second order probability representation has been applied in the past to pose estimation (Gong & Cuzzolin (2017)), tracking (Cuzzolin & Frezza (1999)), logistic regression (Cuzzolin (2018a)), clustering (Zhu et al. (2026)), feature fusion (Pang et al. (2025)), max-entropy classification (Cuzzolin (2018b)), statistical learning theory (Cuzzolin (2024); Caprio et al. (2024)) and a variety of other tasks (Liu et al. (2019); Faza et al. (2026); Manchingal et al. (2025a)). A recent review can be found in Wang et al. (2025c). We investigate how input tokens influence the resulting credal width, rather than proposing a new uncertainty representation.

Uncertainty attribution. Previous work explains uncertainty either by assigning contributions to input features or by finding changes to an input that reduce the model’s uncertainty. CLUE searches for similar inputs on which the model is less uncertain, using a generative model to keep the changes plausible (Antorán et al., 2021). Wang et al. (2025a) jointly learn which image regions to modify and how to modify them to reduce uncertainty. These approaches explain uncertainty through proposed changes to the input.

Other methods assign uncertainty contributions directly to input features. Perez et al. (2022) combine comparisons with less uncertain inputs and gradientbased attribution. Wang et al. (2023) trace uncertainty estimates in Bayesian neural networks back to individual image pixels. Watson et al. (2023) use Shapley values to allocate changes in conditional entropy (the uncertainty remaining when specified features are known) among input features. Iversen et al. (2025) apply existing attribution methods to predicted variance, targeting aleatoric uncertainty associated with variability in the data. XU-RS follows this general strategy of explaining an uncertainty output, specialising it to the credal width of a selected answer option. This difers from assessing uncertainty in an explanation: Marx et al. (2023) construct uncertainty sets for feature explanations, whereas our target is the uncertainty estimate reported by the classifier.

Gradient-based token attribution. Integrated Gradients (IG) is a featureattribution method that estimates how input features contribute to a diference in a model’s score (Sundararajan et al., 2017). It compares the original input with a chosen comparison input, called a reference or baseline. For text, this reference can be another prompt. For example, if a classifier reports an answer’s width as 0.6 on the original prompt and 0.2 on the reference prompt, the attribution explains the diference of 0.4.

To perform this comparison, the method works with the numerical vectors that represent the tokens, rather than changing the text directly. It evaluates intermediate vectors between the reference and the original input. Halfway between them, for example, each vector’s values are the averages of the corresponding reference and input values. At these intermediate points, IG measures how small changes to the vectors afect the chosen score. These sensitivities are called gradients. The method combines them with the diferences between the original and reference vectors to estimate each feature’s contribution.

Expected Gradients (EG) extends this approach by averaging contributions over sampled references and intermediate vectors (Erion et al., 2021). In XU-RS, the references are reference prompts/comparison inputs (other questions and their answer options), and the score being explained is the credal width of a selected answer option.

The reference matters because it determines what diference is being explained (Sturmfels et al., 2020). The intermediate vectors also require care: an average of two token vectors need not represent an actual word. Discretized Integrated Gradients therefore chooses intermediate vectors that remain closer to the representations of actual words (Sanyal & Ren, 2021).

Evaluation of attributions. An explanation that appears convincing to a reader might not describe the model’s full internal behaviour (Jacovi & Goldberg, 2020). For example, Adebayo et al. (2018) test whether explanations respond to changes in model parameters and training labels, showing why visual inspection alone is insuficient. Feature-replacement tests provide another form of evaluation: they modify highly ranked features and measure the resulting change in the model output. However, the modified input may difer substantially from inputs encountered during training. The measured change can therefore reflect the model’s response to an unfamiliar replacement, rather than the importance of the original feature under ordinary inputs (Hase et al., 2021).

These studies motivate testing whether an attribution reflects the model’s behaviour, rather than judging it only by whether it looks plausible. For XU-RS, we distinguish evidence that tokens afect credal width from evidence that they identify information meaningful for the task.

## 3 The XU-RS attribution framework

## 3.1 Random-set predictions and credal width

Let $\Theta = \{ A , B , C , D \}$ contain the answer options. A random-set prediction assigns probability, called mass, $m ( S )$ to each set $S \subseteq \Theta$ , with $m ( S ) \geq 0$ $m ( \emptyset ) = 0$ and $\Sigma _ { S } m ( S ) = 1$ . Mass on a set leaves its division among members unspecified. Lower probability counts masses assigned entirely within an event

E; upper probability counts masses on sets overlapping E:

$$
\operatorname { B e l } ( E ) = \sum _ { S \subseteq E } m ( S ) , \qquad \operatorname { P l } ( E ) = \sum _ { S \cap E \neq \emptyset } m ( S ) .\tag{1}
$$

These quantities are called belief and plausibility functions (Shafer, 1976; Cuzzolin, 2014, 2018c, 2021a, 2026). For an answer $c ,$ credal width and the pignistic answer probability are

$$
W _ { c } = \mathrm { P l } ( \{ c \} ) - \mathrm { B e l } ( \{ c \} ) = \sum _ { \stackrel { S \ni c } { | S | > 1 } } m ( S ) , \qquad \mathrm { B e t P } ( c ) = \sum _ { S \ni c } \frac { m ( S ) } { | S | } .\tag{2}
$$

Here |S| is the number of options in a set. Width lies in [0, 1] and counts probability assigned to sets containing c together with other options. A geometric approach to belief functions and other uncertainty measures can be found in Cuzzolin (2004, 2008b,a, 2020); their combinatorial analysis in Cuzzolin (2010), their algebraic properties in Cuzzolin (2001a,b); Zhou & Cuzzolin (2017). Selecting an answer requires a separate rule: the pignistic probability divides each set’s probability equally among its members and adds the shares for each option. This division is used to choose an answer; the original set probabilities still determine its width.

We select $c = \operatorname { a r g m a x } _ { k } \mathrm { B e t P } ( k )$ on the original prompt and explain that same option throughout attribution and zero-masking. If modifying the input changes the model’s preferred answer, we still measure the width of c. Otherwise, the measured diference would concern two diferent answer options.

## 3.2 Classifier architecture

A linear output layer reads the transformer’s representation at the last nonpadding token. Its 14 sigmoid outputs estimate lower probabilities for four single-option sets, six pairs and four triples, following the belief-output approach of RS-NN and RS-LLM (Manchingal et al., 2025d; Mubashar et al., 2025).

An estimated lower probability for a set is not the probability assigned to that set alone. For example, Bel $( \{ B , C \} ) = m ( \{ B \} ) + m ( \{ C \} ) + m ( \{ B , C \} )$ Recovering $m ( \{ B , C \} )$ therefore requires subtracting the probabilities already assigned to B and C individually. The implementation processes single-option sets, pairs and triples in that order, subtracting only the nonnegative parts of previously computed values for smaller sets. Appendix A gives the exact recursion.

Although the sigmoid places each predicted lower probability in [0, 1], it does not enforce the relationships between these predictions in Equation 1. The conversion can consequently produce negative values or a total diferent from one. A final correction sets negative values to zero, assigns any shortfall to the set of all four options and rescales the resulting values to sum to one. Section 4 derives how this operation afects width. The classifier uses the corrected probabilities during both training and attribution.

## 3.3 Reference prompts and Expected Gradients

Preparing reference prompts. The transformer represents each token by an embedding, a vector of d numerical values. Let $\bar { X ^ { \mathrm { ~ \bf ~ \in ~ } } } \mathbb { R } ^ { L \times d }$ collect these vectors for the original prompt, which has L token positions. MedQA reference prompts contain questions and answer options drawn from the training split. Attribution compares the original and reference embeddings position by position, so both must have the same dimensions. We truncate longer reference sequences and extend shorter ones with padding tokens to obtain $B \in \mathbb { R } ^ { L \times d }$

The model also receives an attention mask, which marks positions to use or ignore as padding, and information about token positions. We retain the original prompt’s mask and positions while changing the embeddings. Model weights are fixed and dropout is disabled. Write $F _ { c } ( Z )$ for the selected answer’s width at embeddings $Z$ under these settings. In particular, $F _ { c } ( B )$ evaluates the prepared reference embeddings under the original prompt’s settings. Processing the reference prompt normally uses its own sequence and settings and may give a diferent width. Our reference checks measure this diference.

Calculating attribution. Expected Gradients estimates contributions to the width diference between the original input and the prepared references. For a sampled reference B, it forms intermediate embeddings $Z _ { \alpha } = B + \alpha ( X - B )$ At $\alpha = 0$ these are the reference embeddings; at $\alpha = 1$ they are the original input embeddings; at $\alpha = 0 . 5$ each coordinate is halfway between the two.

At these intermediate inputs, the gradient measures how a small change in each embedding coordinate afects width. Multiplying these sensitivities by the corresponding input–reference diferences and averaging gives

$$
\begin{array} { r } { \mathrm { E G } ( F _ { c } ; X ) = \mathbb { E } _ { B , \alpha } \left[ ( X - B ) \odot \left. \nabla _ { Z } F _ { c } ( Z ) \right. _ { Z = B + \alpha ( X - B ) } \right] . } \end{array}\tag{3}
$$

Here B is sampled from the prepared reference collection, α is sampled uniformly from [0, 1], and ⊙ multiplies corresponding coordinates. The expectation E is estimated by averaging sampled contributions. We use Captum’s GradientShap with no added Gaussian noise. Gradients pass through the transformer, output layer and probability conversion, so attribution explains the final width reported by the classifier.

## 3.4 Answer-set attribution and token scores

Direct EG attributes $F _ { c }$ . The answer-set route attributes each corrected probability contributing to width, then adds its coordinate contributions. Linearity gives

$$
\operatorname { E G } ( F _ { c } ; X ) = \sum _ { \stackrel { S \ni c } { | S | > 1 } } \operatorname { E G } ( m ( S ) ; X ) .\tag{4}
$$

The equality follows because diferentiation and averaging both preserve addition. When checking agreement, both routes use the same sampled references and intermediate inputs, so sampling diferences do not obscure the result $( \mathrm { A p - }$ pendix $\mathrm { A } )$

The calculation produces a contribution $e _ { i j }$ for each embedding coordinate $j$ of token i. To rank tokens, we combine each token’s coordinate contributions into a nonnegative magnitude, $\textstyle a _ { i } = ( \sum _ { j } e _ { i j } ^ { 2 } ) ^ { 1 / 2 }$ . In the answer-set route, contributions are added across sets before taking this magnitude, allowing positive and negative contributions to cancel.

Numerical checks use a diferent summary: the signed sum $\textstyle \sum _ { i , j } e _ { i j }$ , which retains positive and negative contributions to the width diference. This sum is checked against the original input’s width minus the average width of the prepared references. The nonnegative token scores serve to rank tokens; they are not expected to sum to that diference.

## 4 Mass correction and credal width

The classifier converts predicted lower probabilities into intermediate answer-set values $\widetilde { m } _ { S }$ . To obtain valid probabilities, it removes negative values, assigns any shortfall to the full answer set and normalises. For non-empty proper subsets of Θ, define

$$
q _ { S } = \operatorname* { m a x } ( 0 , \widetilde { m } _ { S } ) , \qquad s = \sum _ { \emptyset \neq S \subset \Theta } q _ { S } , \qquad r = \operatorname* { m a x } ( 1 - s , 0 ) .\tag{5}
$$

The corrected masses are $m ( S ) = q _ { S } / ( s + r )$ and $m ( \Theta ) = r / ( s + r )$ . Their sum is one because $s + r = \operatorname* { m a x } ( s , 1 ) > 0$

Let $\begin{array} { r } { u _ { c } = \sum _ { S \subseteq \Theta , c \in S , | S | > 1 } q _ { S } } \end{array}$ be the surviving total on proper sets containing c with other options. Adding the full-set contribution gives the exact width

$$
W _ { c } = \frac { u _ { c } + r } { s + r } = \left\{ \begin{array} { l l } { u _ { c } + 1 - s , } & { s \leq 1 , } \\ { u _ { c } / s , } & { s > 1 . } \end{array} \right.\tag{6}
$$

For $s \leq 1$ , the shortfall contributes to every answer’s width. For $s > 1$ , all sets share the denominator $s ,$ so values assigned to sets excluding c can change its width. For example, suppose $q _ { \{ B , C \} } = q _ { \{ A \} } = 0 . 6$ and all other values are zero. Their total is 1.2, so $W _ { C } = 0 . \dot { 6 } / 1 . \dot { 2 } = \dot { 0 } . 5$ . Increasing only $q _ { \{ A \} }$ to $0 . 9$ raises the total to 1.5 and gives $W _ { C } = 0 . 6 / 1 . 5 = 0 . 4$ . The value assigned to the set containing C has not changed before normalisation, but its final probability and C’s width have both decreased.

Where derivatives exist and $s > 1$ , the input gradient is

$$
\nabla _ { X } W _ { c } = \frac { 1 } { s } \nabla _ { X } u _ { c } - \frac { u _ { c } } { s ^ { 2 } } \nabla _ { X } s .\tag{7}
$$

The first term describes changes in values assigned to sets containing $c$ and other options. The second describes changes in the total used to rescale them, including values assigned to sets excluding c. Although this total is shared by all options, its coeficient depends on $u _ { c } ,$ , so the efect need not be equal for every answer. Appendix A gives the full gradient, including the shortfall term and behaviour at boundaries. Both attribution routes diferentiate the corrected probabilities, so separating answer-set attributions does not remove this dependence. Correction is part of training and is reevaluated at every sampled intermediate embedding.

## 5 Experimental design

## 5.1 Dataset and classifiers

We use four-option MedQA US (Jin et al., 2021), with 10,178 training, 1,272 development and 1,273 test questions. Prompts contain the question, lettered options and an Answer: prefix, truncated to 512 tokens. SmolLM3-3B and Llama-2-7B are adapted using LoRA (Hu et al., 2022), which learns low-rank weight updates, and a trained classification output layer. The training loss combines the negative log pignistic probability of the correct answer with a binary cross-entropy term, weighted by 0.1. The latter indicates whether each answer set contains the correct answer. Checkpoints are selected by the lowest development-set answer negative log-likelihood. Appendix B gives the training settings.

We report classification accuracy on the full test set. To assess the uncertainty being explained, we also measure whether larger widths rank incorrect predictions above correct predictions, using error-detection AUROC. We compare this with ranking predictions by low probability of the selected answer.

## 5.2 Token attribution and zero-masking

The main study uses direct attribution on the first 100 test questions per model, keeping the originally predicted answer fixed. SmolLM3 uses 256 training reference prompts and 512 EG samples. Llama averages coordinate-level attributions from three collections, each containing 1,024 reference prompts and using 1,024 samples. Appendix B reports precision settings and implementation checks.

Eligible tokens belong to the question or answer-option text and contain a letter or digit. Special tokens, padding, formatting labels, punctuation-only tokens and the answer prefix are excluded. Let $R _ { i } X$ set token i’s embedding to zero, preserving sequence length and the attention mask. We measure

$$
D _ { i } ( x ) = | F _ { c } ( R _ { i } X ) - F _ { c } ( X ) | , \qquad A ( x ) = D _ { i ^ { * } } ( x ) - { \frac { 1 } { 2 0 } } \sum _ { b = 1 } ^ { 2 0 } D _ { j b } ( x ) ,\tag{8}
$$

Here $i ^ { * }$ is the highest-ranked eligible token and the $j _ { b }$ are 20 uniformly sampled eligible positions. A positive $A ( x )$ means that zero-masking the highest-ranked token changes width more than the average change from zero-masking these randomly selected tokens. Both changes concern the same selected answer.

Each random draw uses the same eligible positions: positions can repeat, and the highest-ranked token can also be selected. Random selection is not restricted to tokens with a similar meaning or position, or to the same question or option section. Controls therefore match the zero-masking operation and number of tokens, not these other properties. We report the mean advantage and a 95% bootstrap interval, obtained by resampling questions and recalculating the mean 10,000 times. Each question retains its highest-ranked and random-token results together during resampling.

## 5.3 Probability correction

We evaluate all test questions under six configurations: output-layer-only, partial and LoRA adaptation of SmolLM3, and Llama LoRA seeds 7, 17 and 23. We record negative intermediate values, the surviving total s, the added shortfall r, and the total absolute change from intermediate values to final masses. Rescaling is active when $s > 1 + 1 0 ^ { - 6 }$

## 5.4 Controlled attribution checks

Synthetic questions describe fictional conditions in paired versions with and without a distinguishing test result. The 960 examples are split into 640 training, 160 development and 160 test examples, keeping condition families within one split. A softmax layer on frozen SmolLM3 representations predicts probabilities for the 15 non-empty answer sets, without correction. A later dataset audit found an answer-order shortcut and unjustified target asymmetry. We use this model to test attribution calculations; Appendix D describes the data limitations and revision.

Reference construction. For four test pairs, we explain the prompt containing the result using three alternatives: training reference prompts, a zeroembedding baseline, and the paired reference prompt without the result. Adding the test result can change the number of tokens and shift later text to diferent positions. For the paired reference prompt, we align matching token sequences so that unchanged text occupies the original prompt’s positions. Fitting the remaining sequences may omit or repeat reference tokens; unmatched positions receive padding. Attribution retains the original prompt’s attention mask and token positions. This alignment is additional to the ordinary truncation and padding used for MedQA reference prompts.

We evaluate the paired reference prompt normally and then evaluate its prepared embeddings under the attribution settings. If these widths difer, the attribution calculation no longer explains the width diference between the two prompts as normally processed.

Numerical accuracy. We repeat attribution calculations using diferent sample counts and random seeds. Spearman correlation measures agreement in token order, while overlap measures how many highly ranked tokens are shared.

Table 1: Mean advantage in absolute width change from zero-masking the highest-ranked rather than random eligible tokens (Equation $8 ) ;$ 100 MedQA questions per model. Intervals are question-level 95% bootstrap intervals.
<table><tr><td>Model</td><td>Questions</td><td>Mean advantage</td><td>95% CI</td></tr><tr><td>SmolLM3-3B</td><td>0.0381</td><td>[0.0243, 0.0532]</td><td></td></tr><tr><td>Llama-2-7B</td><td>0.0450</td><td>[0.0295, 0.0617]</td><td></td></tr></table>

We also compare the full attribution vectors, retaining their positive and negative entries. Cosine similarity measures agreement in their direction, and relative Euclidean distance measures diferences in their values relative to vector size.

A separate check asks whether the contributions add up to the width diference being explained. The completeness residual is the absolute gap between their signed sum and the original input’s width minus the average width of the prepared references:

$$
R = \left| \sum _ { i , j } \widehat { e } _ { i j } - \left( F _ { c } ( X ) - \frac { 1 } { M } \sum _ { b = 1 } ^ { M } F _ { c } ( B _ { b } ) \right) \right| ,\tag{9}
$$

Here $B _ { b }$ are sampled prepared reference embeddings, evaluated with the same mask and positions used for attribution. The sum retains positive and negative coordinate contributions rather than the nonnegative token scores used for ranking. Initial median and maximum residual thresholds are 0.01 and 0.05; Appendix B gives the vector and ranking criteria.

For further checks, we use Integrated Gradients (IG) in FP32 and increase the number of intermediate points between each original input and prepared reference. Unlike EG’s sampling over references and intermediate points together, this evaluates the full transition for each chosen reference separately. We refine all four aligned-reference calculations and, for one case, calculations using 128 training reference prompts. Each reference’s residual is checked before averaging its contributions with those from other references. Appendix C reports the one-case clue analysis.

## 6 Results

## 6.1 Zero-masking on MedQA

Zero-masking the highest-ranked token produces a larger absolute width change than random zero-masking on both models (Table 1). Mean advantages are 0.0381 for SmolLM3 and 0.0450 for Llama on the [0, 1] width scale; both 95% intervals exclude zero. Full-test accuracies are 49.3% and 45.7%, respectively. Width ranks errors with AUROC 0.634 and 0.598, below the 0.680 and 0.671 obtained using low predicted-answer probability.

Table 2: Numerical checks on four synthetic cases. Entries are medians. Rank correlation and relative vector diference compare sampling budgets; R is the completeness residual (Equation 9).
<table><tr><td>Reference</td><td>Rank ρ</td><td>R</td><td>Vector difference</td></tr><tr><td>Training prompts</td><td>0.9951</td><td>0.0652</td><td>0.1563</td></tr><tr><td>Aligned paired prompt</td><td>1.0000</td><td>0.0144</td><td>0.0384</td></tr><tr><td>Zero-embedding baseline</td><td>0.9840</td><td>0.6428</td><td>0.3375</td></tr></table>

## 6.2 Frequency of probability correction

The surviving total exceeds one on 95.84–99.92% of test questions across the six configurations (Appendix B, Table 8). Thus width is usually $u _ { c } / s$ at the original prompt. Every tested prediction contains negative intermediate values; the mean total absolute correction ranges from 1.23 to 2.81. Equation 7 shows how the shared total afects input sensitivity. Its frequency at test prompts does not measure its contribution along the intermediate embeddings used for attribution.

## 6.3 Numerical accuracy

On the four synthetic cases, increasing the sample count leaves training-reference token rankings nearly unchanged: median correlation is 0.9951. However, median residual is 0.0652 and relative vector diference is 0.1563, exceeding tolerances of 0.01 and 0.10 (Table 2). Aligned-reference and zero-baseline calculations also exceed the residual threshold despite high ranking agreement.

The ranking and numerical checks answer diferent questions. Token order can remain unchanged even when contribution values change substantially. Here, the high ranking agreement coexists with a sum of contributions that misses the measured width diference by more than the specified tolerance. This motivates refining the calculation before interpreting its values.

Denser FP32 IG passes the residual and attribution-agreement checks for all four aligned references, with maximum residual $1 . 1 0 5 \times 1 0 ^ { - 5 }$ . For one case, refining five of 128 training-reference integrals gives mean and maximum residuals of 0.00401 and 0.02454. This refinement evaluates complete IG integrals, not a rerun of stochastic EG.

## 6.4 Efects of reference preparation

Let $w _ { \mathrm { w i t h } }$ and $w _ { \mathrm { w i t h o u t } }$ be widths for the paired prompts processed normally, and $w _ { \mathrm { a l i g n e d } }$ the width of the prepared reference under the attribution settings. Then

$$
\Delta _ { \mathrm { n a t u r a l } } = w _ { \mathrm { w i t h } } - w _ { \mathrm { w i t h o u t } } , \qquad \Delta _ { \mathrm { a l i g n e d } } = w _ { \mathrm { w i t h } } - w _ { \mathrm { a l i g n e d } } .\tag{10}
$$

Table 3: Width diferences using the original and aligned paired reference prompts (Equation 10). Shift is $\lvert \Delta _ { \mathrm { n a t u r a l } } - \Delta _ { \mathrm { a l i g n e d } } \rvert ;$ ; residuals refer to IG with aligned references. All evaluations use the same FP32 model.
<table><tr><td>Case</td><td> $\Delta _ { \mathrm { n a t u r a l } }$ </td><td> $\Delta _ { \mathrm { a l i g n e d } }$ </td><td>Absolute shift</td><td>IG residual</td></tr><tr><td>family007_case00</td><td>-0.478592</td><td>+0.002640</td><td>0.481233</td><td> $1 . 1 0 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>family094_case02</td><td>-0.538493</td><td>-0.164970</td><td>0.373522</td><td> $4 . 6 1 \times 1 0 ^ { - 7 }$ </td></tr><tr><td>family098_case00</td><td>-0.629888</td><td>-0.427342</td><td>0.202545</td><td> $7 . 3 9 \times 1 0 ^ { - 6 }$ </td></tr><tr><td>family017_case00</td><td>-0.619253</td><td>-0.082523</td><td>0.536730</td><td> $6 . 5 1 \times 1 0 ^ { - 6 }$ </td></tr></table>

Alignment shifts the width diferences by 0.203–0.537 (Table 3). In the first case, supplying the test result reduces width by 0.478592 when both prompts are processed normally. Using the aligned reference instead gives a small increase of 0.002640. The signed IG contributions sum to +0.002651, closely accounting for the aligned-reference diference rather than the original decrease.

The two checks therefore identify diferent properties: the small residual shows that the contributions account for the diference actually supplied to IG, while evaluating the reference before and after preparation shows that this diference has changed. A direct evaluation of both reference versions detects this shift without requiring an attribution calculation.

## 7 Discussion and limitations

XU-RS identifies tokens that influence reported credal width under zero-masking. The correction analysis specifies the function being explained, while the numerical and reference checks address how that function is attributed to tokens.

Before interpreting highlighted tokens, we evaluate the reference prompt normally and under the prepared attribution settings to establish which width diference the explanation concerns. We then check whether the signed contributions account for that diference and whether their values agree when more samples or intermediate points are used. Both checks matter: a small completeness residual constrains the sum but can conceal errors in individual contributions that cancel. Finally, zero-masking tests how selected tokens afect width under an explicit input modification. Appendix E gives the procedure.

The evaluation covers fixed 100-question MedQA panels and two four-option classifiers with diferent attribution budgets. Zero-masking changes embeddings rather than deleting or rewording text. These experiments establish sensitivity to that operation, not clinical evidence recovery or an independent validation of width as epistemic uncertainty. The numerical and reference findings concern four synthetic pairs. Their dataset permits shortcuts, so attribution to the constructed clue cannot be assessed as though the model necessarily relies on it. Some follow-up checks were developed after inspecting test results.

Further evaluation should use fresh questions, controlled evidence edits and meaning-preserving rewordings. Comparing width attribution with entropy or answer-score attribution under the same references and computational budgets would test the additional value of explaining the random-set output. Clinical interpretation would additionally require domain-expert evaluation.

## 8 Conclusion

XU-RS attributes an answer’s credal width to input tokens. On MedQA, its highest-ranked tokens produce larger zero-masking efects than random controls. The correction analysis explains the width being attributed, while controlled tests show why numerical accuracy and preservation of the reference prompt require separate checks. Together, these provide a framework for inspecting the input dependence of random-set uncertainty estimates.

## References

Julius Adebayo, Justin Gilmer, Michael Muelly, Ian Goodfellow, Moritz Hardt, and Been Kim. Sanity checks for saliency maps. In Advances in Neural Information Processing Systems, volume 31, 2018.

Javier Antorán, Umang Bhatt, Tameem Adel, Adrian Weller, and José Miguel Hernández-Lobato. Getting a CLUE: A method for explaining uncertainty estimates. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id=XSLF1XFq5h.

Freddie Bickford Smith, Jannik Kossen, Eleanor Trollope, Mark Van Der Wilk, Adam Foster, and Tom Rainforth. Rethinking aleatoric and epistemic uncertainty. In Aarti Singh, Maryam Fazel, Daniel Hsu, Simon Lacoste-Julien, Felix Berkenkamp, Tegan Maharaj, Kiri Wagstaf, and Jerry Zhu (eds.), Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pp. 4345–4359. PMLR, 2025. URL https://proceedings.mlr.press/v267/bickford-smith25a.html.

Michele Caprio, Maryam Sultana, Eleni G Elia, and Fabio Cuzzolin. Credal learning theory. Advances in Neural Information Processing Systems, 37: 38665–38694, 2024.

Michele Caprio, Shireen K Manchingal, and Fabio Cuzzolin. Credal and interval deep evidential classifications. arXiv preprint arXiv:2512.05526, 2025.

F Cuzzolin. On the properties of the intersection probability. submitted to the Annals of Mathematics and AI, 2007.

Fabio Cuzzolin. Lattice modularity and linear independence. In Proceedings of the 18th British Combinatorial Conference (BCC’01), 2001a.

Fabio Cuzzolin. Lattice structure of the families of compatible frames of discernment. In Isipta, 2001b.

Fabio Cuzzolin. Geometry of dempster’s rule of combination. IEEE Transactions on Systems, Man, and Cybernetics, Part B (Cybernetics), 34(2):961– 977, 2004.

Fabio Cuzzolin. On the credal structure of consistent probabilities. In European Workshop on Logics in Artificial Intelligence, pp. 126–139. Springer, 2008a.

Fabio Cuzzolin. A geometric approach to the theory of evidence. IEEE Transactions on Systems, Man, and Cybernetics, Part C (Applications and Reviews), 38(4):522–534, 2008b.

Fabio Cuzzolin. Credal semantics of bayesian transformations in terms of probability intervals. IEEE Transactions on Systems, Man, and Cybernetics, Part B (Cybernetics), 40(2):421–432, 2009.

Fabio Cuzzolin. Three alternative combinatorial formulations of the theory of evidence. Intelligent Data Analysis, 14(4):439–464, 2010.

Fabio Cuzzolin. Belief functions: theory and applications. Springer, 2014.

Fabio Cuzzolin. Belief likelihood function for generalised logistic regression. arXiv preprint arXiv:1808.02560, 2018a.

Fabio Cuzzolin. Generalised max entropy classifiers. In International Conference on Belief Functions, pp. 39–47. Springer, 2018b.

Fabio Cuzzolin. Visions of a generalized probability theory. arXiv preprint arXiv:1810.10341, 2018c.

Fabio Cuzzolin. Geometry of dempster’s rule. In The Geometry of Uncertainty: The Geometry of Imprecise Probabilities, pp. 351–388. Springer, 2020.

Fabio Cuzzolin. The Geometry of Uncertainty: The Geometry of Imprecise Probabilities. Springer, 2021a. doi: 10.1007/978-3-030-63153-6.

Fabio Cuzzolin. Uncertainty measures: The big picture. arXiv preprint arXiv:2104.06839, 2021b.

Fabio Cuzzolin. Reasoning with random sets: An agenda for the future. arXiv preprint arXiv:2401.09435, 2023.

Fabio Cuzzolin. Generalising realisability in statistical learning theory under epistemic uncertainty. arXiv preprint arXiv:2402.14759, 2024.

Fabio Cuzzolin. Statistical inference with belief functions: A survey. arXiv preprint arXiv:2605.07908, 2026.

Fabio Cuzzolin and Ruggero Frezza. Evidential reasoning framework for object tracking. In Telemanipulator and Telepresence Technologies VI, volume 3840, pp. 13–24. SPIE, 1999.

Fabio Cuzzolin and Maryam Sultana. Epistemic Uncertainty in Artificial Intelligence. Springer, 2024.

Gabriel Erion, Joseph D. Janizek, Pascal Sturmfels, Scott M. Lundberg, and Su-In Lee. Improving performance of deep learning models with axiomatic attribution priors and expected gradients. Nature Machine Intelligence, 3: 620–631, 2021. doi: 10.1038/s42256-021-00343-w.

Ghifari Adam Faza, Jolan Wauters, Fabio Cuzzolin, Hans Hallez, and David Moens. Direct interval propagation methods using neural-network surrogates for uncertainty quantification in physical systems surrogate model. Knowledge-Based Systems, pp. 115824, 2026.

Wenjuan Gong and Fabio Cuzzolin. A belief-theoretical approach to examplebased pose estimation. IEEE Transactions on Fuzzy Systems, 26(2):598–611, 2017.

Peter Hase, Harry Xie, and Mohit Bansal. The out-of-distribution problem in explainability and search methods for feature importance explanations. In Advances in Neural Information Processing Systems, volume 34, 2021. URL https://papers.nips.cc/paper/2021/hash/ 1def1713ebf17722cbe300cfc1c88558-Abstract.html.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=nZeVKeeFYf9.

Hugging Face. SmolLM3-3B model card, 2025. URL https: //huggingface.co/HuggingFaceTB/SmolLM3-3B. Model identifier: HuggingFaceTB/SmolLM3-3B.

Eyke Hüllermeier and Willem Waegeman. Aleatoric and epistemic uncertainty in machine learning: an introduction to concepts and methods. Machine Learning, 110(3):457–506, 2021. doi: 10.1007/s10994-021-05946-3. URL https://link.springer.com/article/10.1007/s10994-021-05946-3.

Pascal Iversen, Simon Witzke, Katharina Baum, and Bernhard Y. Renard. Identifying drivers of predictive aleatoric uncertainty. In Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence, pp. 5453–5462. International Joint Conferences on Artificial Intelligence Organization, 2025. doi: 10.24963/ijcai.2025/607. URL https://www.ijcai.org/ proceedings/2025/607.

Alon Jacovi and Yoav Goldberg. Towards faithfully interpretable NLP systems: How should we define and evaluate faithfulness? In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pp. 4198–4205. Association for Computational Linguistics, 2020. doi: 10.18653/v1/2020.acl-main.386. URL https://aclanthology.org/2020. acl-main.386/.

Di Jin, Eileen Pan, Nassim Oufattole, Wei-Hung Weng, Hanyi Fang, and Peter Szolovits. What disease does this patient have? a large-scale open domain question answering dataset from medical exams. Applied Sciences, 11(14): 6421, 2021. doi: 10.3390/app11146421.

Lingkai Kong, Harshavardhan Kamarthi, Peng Chen, B Aditya Prakash, and Chao Zhang. Uncertainty quantification in deep learning. In Proceedings of the 29th ACM SIGKDD conference on knowledge discovery and data mining, pp. 5809–5810, 2023.

Isaac Levi. The enterprise of knowledge: An essay on knowledge, credal probability, and chance. MIT press, 1980.

Mingda Li, Xinyu Li, Weinan Zhang, and Longxuan Ma. ESI: Epistemic uncertainty quantification via semantic-preserving intervention for large language models. 2025. URL https://arxiv.org/abs/2510.13103.

Zhun-Ga Liu, Yu Liu, Jean Dezert, and Fabio Cuzzolin. Evidence combination based on credal belief redistribution for pattern classification. IEEE Transactions on Fuzzy Systems, 28(4):618–631, 2019.

Shireen Kudukkil Manchingal and Fabio Cuzzolin. Epistemic deep learning. arXiv preprint arXiv:2206.07609, 2022.

Shireen Kudukkil Manchingal, Muhammad Mubashar, Kaizheng Wang, Keivan Shariatmadar, and Fabio Cuzzolin. Random-set convolutional neural network (rs-cnn) for epistemic deep learning. arXiv preprint arXiv:2307.05772, 2023.

Shireen Kudukkil Manchingal, Muhammad Mubashar, Maryam Sultana, Salman Khan, and Fabio Cuzzolin. Epistemic artificial intelligence: Using random sets to quantify uncertainty in machine learning. Manuscript, 2024.

Shireen Kudukkil Manchingal, Armand Amaritei, Mihir Gohad, Maryam Sultana, Julian FP Kooij, Fabio Cuzzolin, and Andrew Bradley. Uncertaintyaware autonomous vehicles: Predicting the road ahead. arXiv preprint arXiv:2510.22680, 2025a.

Shireen Kudukkil Manchingal, Andrew Bradley, Julian FP Kooij, Keivan Shariatmadar, Neil Yorke-Smith, and Fabio Cuzzolin. Epistemic artificial intelligence is essential for machine learning models to truly’know when they do not know’. arXiv preprint arXiv:2505.04950, 2025b.

Shireen Kudukkil Manchingal, Muhammad Mubashar, Kaizheng Wang, and Fabio Cuzzolin. A unified evaluation framework for epistemic predictions, 2025c. URL https://arxiv.org/abs/2501.16912.

Shireen Kudukkil Manchingal, Muhammad Mubashar, Kaizheng Wang, Keivan Shariatmadar, and Fabio Cuzzolin. Random-set neural networks. In International Conference on Learning Representations, 2025d. URL https://proceedings.iclr.cc/paper\_files/paper/2025/ file/260110399ec17409228521208086c113-Paper-Conference.pdf.

Shireen Kudukkil Manchingal, Sofiia Nikolenko, and Fabio Cuzzolin. Credal large language models for semantic commitment under uncertainty. arXiv preprint arXiv:2608.23244, 2026.

Charles Marx, Youngsuk Park, Hilaf Hasson, Yuyang Wang, Stefano Ermon, and Luke Huan. But are you sure? an uncertainty-aware perspective on explainable AI. In Proceedings of the 26th International Conference on Artificial Intelligence and Statistics, volume 206 of Proceedings of Machine Learning Research, pp. 7375–7391. PMLR, 2023. URL https://proceedings.mlr. press/v206/marx23a.html.

Georges Matheron. Random sets and integral geometry. (No Title), 1975.

Ilya Molchanov. Theory of random sets. Springer, 2005.

Muhammad Mubashar and Fabio Cuzzolin. Epistemic generative adversarial networks. arXiv preprint arXiv:2603.18348, 2026.

Muhammad Mubashar, Shireen Kudukkil Manchingal, and Fabio Cuzzolin. Random-set large language models, 2025. URL https://arxiv.org/abs/ 2504.18085.

Hung T Nguyen. An introduction to random sets. CRC press, 2006.

Yaniv Oren, Matthijs TJ Spaan, and Wendelin Böhmer. E-mcts: Deep exploration in model-based reinforcement learning by planning with epistemic uncertainty. arXiv preprint arXiv:2210.13455, 2022.

Yanwei Pang, Fahad Shahbaz Khan, Xin Lu, and Fabio Cuzzolin. Guest editorial special issue on efective feature fusion in deep neural networks. IEEE Transactions on Neural Networks and Learning Systems, 36(3):3886–3889, 2025.

Iker Perez, Piotr Skalski, Alec Barns-Graham, Jason Wong, and David Sutton. Attribution of predictive uncertainties in classification models. In Proceedings of the Thirty-Eighth Conference on Uncertainty in Artificial Intelligence, volume 180 of Proceedings of Machine Learning Research, pp. 1582–1591, 2022. URL https://proceedings.mlr.press/v180/perez22a.html.

Soumya Sanyal and Xiang Ren. Discretized integrated gradients for explaining language models. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pp. 10285–10299. Association for Computational Linguistics, 2021. doi: 10.18653/v1/2021.emnlp-main.805. URL https://aclanthology.org/2021.emnlp-main.805/.

Glenn Shafer. A Mathematical Theory of Evidence. Princeton University Press, 1976.

Pascal Sturmfels, Scott Lundberg, and Su-In Lee. Visualizing the impact of feature attribution baselines. Distill, 2020. doi: 10.23915/distill.00022. URL https://distill.pub/2020/attribution-baselines/.

Maryam Sultana, Neil Yorke-Smith, Kaizheng Wang, Shireen Kudukkil Manchingal, Muhammad Mubashar, and Fabio Cuzzolin. Epistemic wrapping for uncertainty quantification. arXiv preprint arXiv:2505.02277, 2025.

Mukund Sundararajan, Ankur Taly, and Qiqi Yan. Axiomatic attribution for deep networks. In Proceedings of the 34th International Conference on Machine Learning, volume 70 of Proceedings of Machine Learning Research, pp. 3319–3328, 2017. URL https://proceedings.mlr.press/v70/ sundararajan17a.html.

Laura P. Swiler, Thomas L. Paez, and Randall L. Mayes. Epistemic uncertainty quantification tutorial. In Proceedings of the 27th International Modal Analysis Conference (IMAC-XXVII), February 2009.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models, 2023. URL https://arxiv.org/abs/2307.09288.

Daniel J Walters, Gülden Ülkümen, David Tannenbaum, Carsten Erner, and Craig R Fox. Investor behavior under epistemic vs. aleatory uncertainty. Management Science, 69(5):2761–2777, 2023.

Hanjing Wang, Dhiraj Joshi, Shiqiang Wang, and Qiang Ji. Gradient-based uncertainty attribution for explainable Bayesian deep learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 12044–12053, 2023. URL https://openaccess.thecvf.com/content/ CVPR2023/html/Wang\_Gradient-Based\_Uncertainty\_Attribution\_for\_ Explainable\_Bayesian\_Deep\_Learning\_CVPR\_2023\_paper.html.

Hanjing Wang, Bashirul Azam Biswas, and Qiang Ji. Optimization-based uncertainty attribution via learning informative perturbations. In Computer Vision – ECCV 2024, volume 15136 of Lecture Notes in Computer Science, pp. 237– 253. Springer Nature Switzerland, 2025a. doi: 10.1007/978-3-031-73229-4 14. URL https://www.ecva.net/papers/eccv\_2024/papers\_ECCV/html/ 10054\_ECCV\_2024\_paper.php.

Kaizheng Wang, Fabio Cuzzolin, Shireen Kudukkil Manchingal, Keivan Shariatmadar, David Moens, and Hans Hallez. Credal deep ensembles for uncertainty quantification. In Advances in Neural Information Processing Systems, volume 37, 2024. doi: 10.52202/079017-2525. URL https://proceedings.neurips.cc/paper\_files/paper/2024/hash/ 911fc798523e7d4c2e9587129fcf88fc-Abstract-Conference.html.

Kaizheng Wang, Fabio Cuzzolin, Keivan Shariatmadar, David Moens, and Hans Hallez. Credal wrapper of model averaging for uncertainty estimation in classification. In International Conference on Learning Representations, 2025b. URL https://proceedings.iclr.cc/paper\_files/paper/ 2025/file/af9ac087ed9123957bb3a45dca56b9d4-Paper-Conference.pdf.

Kaizheng Wang, Fabio Cuzzolin, Keivan Shariatmadar, David Moens, and Hans Hallez. A review of uncertainty representation and quantification in neural networks. IEEE transactions on pattern analysis and machine intelligence, 2025c.

Kaizheng Wang, Keivan Shariatmadar, Shireen Kudukkil Manchingal, Fabio Cuzzolin, David Moens, and Hans Hallez. Creinns: Credal-set interval neural networks for uncertainty estimation in classification tasks. Neural Networks, pp. 107198, 2025d.

Kaizheng Wang, Fabio Cuzzolin, David Moens, and Hans Hallez. Credal ensemble distillation for uncertainty quantification. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pp. 26319–26327, 2026a.

Kaizheng Wang, Ghifari Adam Faza, Fabio Cuzzolin, Siu Lun Chau, David Moens, and Hans Hallez. Learning credal ensembles via distributionally robust optimization. arXiv preprint arXiv:2602.08470, 2026b.

Kaizheng Wang, Yunjia Wang, Fabio Cuzzolin, David Moens, Hans Hallez, and Siu Lun Chau. Set-based vs distribution-based representations of epistemic uncertainty: A comparative study. arXiv preprint arXiv:2602.22747, 2026c.

David S. Watson, Joshua O’Hara, Niek Tax, Richard Mudd, and Ido Guy. Explaining predictive uncertainty with information theoretic shapley values. In Advances in Neural Information Processing Systems, volume 36, 2023. URL https://proceedings.neurips.cc/paper\_files/paper/2023/hash/ 16e4be78e61a3897665fa01504e9f452-Abstract-Conference.html.

Anita Yang, Krikamol Muandet, Michele Caprio, Siu Lun Chau, and Masaki Adachi. Verbalizing LLM’s higher-order uncertainty via imprecise probabilities, 2026. URL https://arxiv.org/abs/2603.10396. arXiv preprint arXiv:2603.10396.

Chunlai Zhou and Fabio Cuzzolin. The total belief theorem. In UAI, 2017.

Yuchen Zhu, Kuang Zhou, and Fabio Cuzzolin. Tdcc: A trustworthy deep credal clustering method for uncertain data. IEEE Transactions on Cybernetics, 2026.

## A Mathematical details

This appendix explains the conversion from predicted lower probabilities to answer-set masses, derives the corrected width, and establishes the relationship between the two attribution routes.

## A.1 Conversion from lower probabilities to masses

Let Θ contain the four answer options. For each non-empty proper subset S of Θ, the output layer produces a sigmoid value $b _ { S }$ , intended to estimate the lower probability of S. There are 14 such subsets.

A lower probability includes the masses assigned to all subsets of S. Recovering the mass assigned to S therefore involves subtracting the amounts already assigned to its smaller subsets. The implementation processes singletons first, followed by pairs and then triples:

$$
\widetilde { m } _ { \{ c \} } = b _ { \{ c \} } , \qquad \widetilde { m } _ { S } = b _ { S } - \sum _ { \scriptscriptstyle { \mathscr { Q } \neq T \subsetneq S } } \operatorname* { m a x } ( \widetilde { m } _ { T } , 0 ) , \qquad 2 \leq | S | \leq 3 .
$$

Here $\widetilde { m } _ { S }$ is an intermediate value, not necessarily a valid probability. All strict subsets of S have been processed before this expression is evaluated.

The positive-part operation inside the sum is important. Ordinary Möbius inversion, the exact algebraic inverse of the belief-to-mass relationship, would subtract the smaller-set values with their signs retained. The implemented conversion instead subtracts only their nonnegative parts. The subsequent correction removes any remaining negative values and produces masses summing to one. The corrected-width derivation below concerns this final correction and does not assume that the intermediate values are valid probabilities.

## A.2 Corrected width and its gradient

After conversion, define

$$
q _ { S } = \operatorname* { m a x } ( \widetilde { m } _ { S } , 0 ) , \qquad s = \sum _ { \substack { \mathcal { O } \neq S \subseteq \Theta } } q _ { S } , \qquad r = \operatorname* { m a x } ( 1 - s , 0 ) .
$$

Thus $q _ { S }$ is the surviving nonnegative value, s is their total, and $r$ fills any shortfall below one. The corrected masses are

$$
m _ { S } = \frac { q _ { S } } { s + r } \quad ( \alpha \neq S \subsetneq \Theta ) , \qquad m _ { \Theta } = \frac { r } { s + r } .
$$

For a selected answer $c ,$ let

$$
u _ { c } = \sum _ { \stackrel { S \subseteq \Theta } { c \in S , \stackrel { \ldots } { | S | } > 1 } } q _ { S } .
$$

Only non-singleton sets containing c contribute to its width. The proper subsets contribute $u _ { c } / ( s + r )$ , and the full answer set contributes $r / ( s + r )$ . Consequently,

$$
W _ { c } = { \frac { u _ { c } + r } { s + r } } .
$$

When $s \leq 1$ , the added amount is $r = 1 - s$ and the denominator is one. When $s > 1$ , no additional mass is needed and the denominator is s. Hence

$$
W _ { c } = \left\{ \begin{array} { l l } { u _ { c } + 1 - s , } & { s \leq 1 , } \\ { u _ { c } / s , } & { s > 1 . } \end{array} \right.
$$

To diferentiate this expression with respect to the input embeddings $X$ write $n = u _ { c } + r$ and $d = s + r$ . The quotient rule gives

$$
\begin{array} { r } { \nabla _ { X } W _ { c } = \frac { \nabla _ { X } u _ { c } + \nabla _ { X } r } { d } - \frac { n } { d ^ { 2 } } \big ( \nabla _ { X } s + \nabla _ { X } r \big ) } \\ { = \frac { 1 } { d } \nabla _ { X } u _ { c } + \frac { s - u _ { c } } { d ^ { 2 } } \nabla _ { X } r - \frac { n } { d ^ { 2 } } \nabla _ { X } s . } \end{array}
$$

The three terms describe diferent contributions to the derivative of the same width, not separate uncertainty measures.

The width formula remains valid when an intermediate value is zero or when $s = 1$ . At such boundaries, an ordinary derivative may not exist; the implementation follows the derivative rules of its automatic diferentiation operations. During attribution, s and r must be evaluated at each intermediate embedding between the original prompt’s embeddings and the prepared reference embeddings. Rescaling at the original input does not imply that rescaling is active throughout that calculation.

## A.3 Equivalence of the attribution routes

For fixed model weights and a fixed answer $c ,$ width is a finite sum of corrected masses:

$$
F _ { c } ( X ) = \sum _ { \stackrel { S \subseteq \Theta } { c \in S , \mid S \mid > 1 } } m _ { S } ( X ) .
$$

Diferentiating this sum gives the sum of the derivatives. Expected Gradients then multiplies by the same input–reference embedding diference and averages over references and intermediate points. These operations preserve addition, giving

$$
\operatorname { E G } ( F _ { c } ; X ) = \sum _ { \stackrel { S \subseteq \Theta } { c \in S , \mid S \mid > 1 } } \operatorname { E G } ( m _ { S } ; X ) ,
$$

provided the expectations exist.

With the same sampled references and intermediate points, the finite estimates should therefore agree up to numerical rounding. Diferent samples can introduce diferences even when both implementations are correct.

This equality applies to the attribution values before they are converted into token magnitudes. Contributions from diferent answer sets must first be added with their positive and negative signs retained. Adding their magnitudes instead would remove cancellation between contributions and would generally give a diferent result.

## A.4 Answer probabilities do not determine credal width

Consider two valid random-set predictions over four answers. The first assigns all probability to the full answer set: $m ( \Theta ) = 1$ . Each answer then has lower probability zero, upper probability one, and width one.

The second assigns probability $1 / 4$ separately to each singleton answer set. Each answer then has lower and upper probability $1 / 4 .$ , giving width zero.

Both predictions produce the same pignistic answer probabilities: $1 / 4$ for every option. A loss that depends only on those answer probabilities cannot distinguish these two assignments at that input. This example does not establish that answer-label training cannot learn useful uncertainty. In particular, our training objective also contains an answer-set membership loss, described below, so it is not determined solely by pignistic probabilities.

## B Model training and attribution settings

## B.1 Classifier training

The pretrained models are HuggingFaceTB/SmolLM3-3B and meta-llama/Llama-2-7b-hf. The LoRA runs begin with previously trained classification output layers, rather than newly initialised output layers. These adaptation runs should therefore not be interpreted as independent training runs from scratch.

The training objective encourages the correct answer to receive high probability and the predicted lower probabilities to agree with answer-set membership. For a correct answer $y ,$ define $t _ { S } = 1$ when $y \in S$ and zero otherwise. The auxiliary loss is

$$
\ell _ { \mathrm { s e t } } = - \frac { 1 } { 1 4 } \sum _ { \substack { \boldsymbol { \sigma } \neq \boldsymbol { S } \subsetneq \boldsymbol { \Theta } } } \left[ t _ { S } \log b _ { S } + ( 1 - t _ { S } ) \log ( 1 - b _ { S } ) \right] .
$$

The complete objective is

$$
\begin{array} { r } { \mathcal { L } = - \log \mathrm { B e t P } ( y ) + 0 . 1 \ell _ { \mathrm { s e t } } . } \end{array}
$$

The answer probability Bet $\mathrm { P } ( y )$ is calculated from the corrected masses. The conversion and correction operations therefore participate in training.

LoRA updates the attention query, key, value, and output projections, together with the feed-forward gate, up, and down projections. Training uses eight workers, each processing one example at a time and accumulating gradients over four batches before an update. This gives an efective batch size of 32. The gradient norm is clipped at one.

Table 4: Training settings for the main LoRA classifiers. Patience is the number of epochs without improvement allowed before stopping. Checkpoints are selected using development-set answer negative log-likelihood.
<table><tr><td>Setting</td><td>SmolLM3-3B</td><td>Llama-2-7B</td></tr><tr><td>LoRA rank</td><td>16</td><td>8</td></tr><tr><td>LoRA scaling parameter</td><td>32</td><td>16</td></tr><tr><td>LoRA dropout</td><td>0.05</td><td>0.05</td></tr><tr><td>LoRA learning rate</td><td>10⁻⁴</td><td>10⁻⁴</td></tr><tr><td>Output-layer learning rate</td><td> $5 \times 1 0 ^ { - 5 }$ </td><td>10⁻⁴</td></tr><tr><td>Weight decay</td><td>0.01</td><td>0.1</td></tr><tr><td>Warmup fraction</td><td>0.1</td><td>0.05</td></tr><tr><td>Maximum epochs</td><td>5</td><td>5</td></tr><tr><td>Early-stopping patience</td><td>2</td><td>3</td></tr><tr><td>Primary training seed</td><td>7</td><td>7</td></tr><tr><td>Additional correction-study seeds</td><td></td><td>17, 23</td></tr></table>

## B.2 Expected Gradients settings

An EG sample selects a reference prompt and an intermediate point between its prepared embeddings and the original prompt’s embeddings. Table 5 distinguishes the number of available reference prompts from the number of samples used in the calculation.

Table 5: Settings for the main MedQA attribution runs. FP16 and BF16 are diferent 16-bit floating-point formats. The samples-per-call setting controls how many samples are processed together.
<table><tr><td>Setting</td><td>SmolLM3-3B</td><td>Llama-2-7B</td></tr><tr><td>Reference collections</td><td>1</td><td>3</td></tr><tr><td>Prompts per collection</td><td>256</td><td>1,024</td></tr><tr><td>EG samples per collection</td><td>512</td><td>1,024</td></tr><tr><td>Attribution base seed</td><td>11</td><td>11</td></tr><tr><td>Samples processed per call</td><td>8</td><td>2</td></tr><tr><td>Model arithmetic</td><td>FP16</td><td>BF16</td></tr><tr><td>Averaging across collections</td><td>Not applicable</td><td>FP32</td></tr></table>

SmolLM3 uses the first 256 training prompts as references. For Llama, training examples are placed in a reproducible order using seeded SHA-256 hash ranking with seed 20260804. The first 3,072 examples in this order form three consecutive, non-overlapping collections of 1,024 references. The three attribution vectors are averaged with equal weights in FP32 before token magnitudes are calculated.

## B.3 Checks of Llama ranking agreement

The final Llama settings are checked on 20 selected questions from the 100- question panel. Attributions for the predicted answer and correct answer are assessed separately. We compare the average from all three reference collections with averages omitting each collection in turn, and repeat the three-collection calculation with attribution seeds 17 and 23. The comparisons that omit one collection reuse calculations from the full average; they are not independent replications.

For section-level comparisons, token magnitudes are summed within the question, each of the four options, and the Answer: prefix. Each sum is expressed as a share of total magnitude. We then compare the ordering of these sections and the diferences between their shares.

Table 6: Criteria used for the final Llama ranking checks. These assess agreement between repeated calculations, not whether the attributions explain meaningful evidence.
<table><tr><td>Measurement</td><td>Required value</td></tr><tr><td>Median Spearman correlation between section rankings</td><td>≥ 0.90</td></tr><tr><td>Fraction agreeing on the highest-attributed option</td><td>≥ 0.80</td></tr><tr><td>Mean overlap among the 20 highest-ranked tokens</td><td> $\geq 0 . 7 5$ </td></tr><tr><td>Median mean absolute difference between section shares Maximum difference in the selected answer&#x27;s width</td><td> $\leq 0 . 0 2$   $\leq 1 0 ^ { - 6 }$ </td></tr></table>

The highest-attributed option is the option whose text receives the largest attribution share, not necessarily the option predicted by the classifier. Token overlap is the fraction of selected token positions shared by the two rankings. These ranking checks are separate from the completeness checks below.

## B.4 Checks of attribution values

For the four synthetic examples, we compare both token rankings and attribution values. The latter comparisons retain the positive and negative contributions for individual embedding coordinates, before conversion into token magnitudes.

For two nonzero attribution vectors e and $e ^ { \prime } ,$ cosine similarity measures agreement in their pattern of contributions, allowing an overall scale diference. Relative Euclidean distance measures the size of their diference relative to the second vector:

$$
\mathrm { c o s i n e } ( e , e ^ { \prime } ) = \frac { e ^ { \top } e ^ { \prime } } { \| e \| _ { 2 } \| e ^ { \prime } \| _ { 2 } } , \qquad \mathrm { r e l a t i v e ~ d i s t a n c e } ( e , e ^ { \prime } ) = \frac { \| e - e ^ { \prime } \| _ { 2 } } { \| e ^ { \prime } \| _ { 2 } } .
$$

Completeness is assessed using the residual R in Equation 9.

The later FP32 Integrated Gradients calculations use a mean per-reference residual threshold of 0.01 and a maximum of 0.05. Agreement between integration resolutions requires cosine similarity of at least 0.99, relative Euclidean distance of at most 0.10, token-ranking correlation of at least 0.95, and topthree overlap of at least 0.80. For the aligned paired references, these ranking comparisons concern the nine positions changed by alignment.

Table 7: Criteria for the initial four-example numerical study. Agreement is assessed between attribution calculations using diferent sample counts and random seeds for the same original prompt and reference collection.
<table><tr><td>Measurement Required value</td></tr><tr><td>Median completeness residual ≤ 0.01</td></tr><tr><td>Maximum completeness residual ≤ 0.05</td></tr><tr><td>Median Spearman correlation between token rankings ≥ 0.95</td></tr><tr><td>Median overlap among the 20 highest-ranked tokens ≥ 0.80</td></tr><tr><td>Median cosine similarity between attribution vectors ≥ 0.99</td></tr><tr><td>Median relative Euclidean distance between vectors ≤ 0.10</td></tr></table>

For complete IG integrals, residuals are checked separately for each prepared reference before averaging, preventing errors of opposite signs from cancelling unnoticed. A single EG sample evaluates only one intermediate point and is not required to account for the full input–reference width diference.

These thresholds are operational choices for the reported studies, not universal guarantees of attribution quality.

## B.5 Computing environment

Early experiments used an NVIDIA T4. Later experiments used Quadro RTX 6000 and NVIDIA A30 devices with 24 GiB of memory per GPU. Independent attribution jobs were distributed across devices; their memory was not combined to process a single explanation.

Hardware and software environments difered across runs. The environment used for the later controlled-model study should not be treated as the environment for every MedQA run. We do not report a controlled comparison of end-to-end runtime or energy consumption.

## B.6 Supplementary implementation and correction results

In a constructed model with one deliberately influential token, both attribution routes rank that token first. For a separate SmolLM3 classifier with frozen transformer weights, direct and answer-set attributions have minimum signed-vector cosine similarity 0.999986 and maximum relative Euclidean distance 0.00522. The Llama check compares the scalar width with the sum of its contributing answer-set probabilities, with maximum diference $2 . 9 8 \times 1 0 ^ { - 8 }$ ; it is not a comparison of attribution vectors.

One Llama attribution calculation gives non-finite values in FP16 and finite values in BF16. A single collection of 1,024 reference prompts with 1,024 EG samples fails the ranking checks. Averaging coordinate-level attributions across three collections meets the recorded criteria. Leave-one-collection-out averages share components, so these checks are not independent ensemble replications.

Table 8: Correction on 1,273 test questions per configuration. Rescaling means $s > 1 + 1 0 ^ { - 6 }$ . Adjustment is the mean total absolute change from intermediate values to final masses. Negative values occur in every tested prediction.
<table><tr><td>Configuration Rescaling (%) Mean total adjustment</td></tr><tr><td>SmolLM3: head only 95.84 1.2311</td></tr><tr><td>SmolLM3: partial tuning 99.84 2.1623</td></tr><tr><td>SmolLM3: LoRA 99.92 1.7221</td></tr><tr><td>Llama-2: LoRA, seed 7 99.14 2.2889</td></tr><tr><td>Llama-2: LoRA, seed 17 99.61 2.4980</td></tr><tr><td>Llama-2: LoRA, seed 23 99.92 2.8108</td></tr></table>

Table 8 reports correction measurements for every training configuration. The intermediate vectors can contain negative values and are not probability distributions; their total absolute adjustment is therefore not a distance between two valid probability distributions.

## C Attribution to the distinguishing test result

This analysis examines one synthetic question, family007\_case00\_resolved, whose text includes a test result distinguishing between two plausible answers. The clue occupies eight tokens. We ask how much attribution magnitude it receives relative to other text of the same length.

## C.1 Reference prompts and measurements

We use two IG calculations that passed the numerical checks. The first averages attributions over 128 training reference prompts. The second uses the paired reference prompt without the distinguishing test result, after alignment. They explain width diferences relative to diferent reference embeddings.

For each calculation, we examine two regions: all 205 active, non-special text tokens, and the 29 tokens within the patient case description. The full-text region includes the instructions, fictional diagnostic rules, and answer options. It is not restricted by the token-eligibility rule used in the MedQA zero-masking experiment.

The clue’s magnitude share is the sum of its token magnitudes divided by the sum for the selected region. Control windows contain the same number of tokens as the clue, are contiguous within that region, and exclude clue tokens. We compare the clue with the mean magnitude of these windows.

With training prompts as references, the clue receives 2.07% of magnitude across the full text, or 22.97% when the denominator includes only the case description. Its full-text magnitude is at approximately the 40th percentile of the equal-length control windows. The windows overlap, so they are not independent observations and this percentile is descriptive.

Table 9: Attribution to the distinguishing clue in one example. “Diference” is the clue’s magnitude share minus the mean control-window share, expressed in percentage points. “Best rank” is the rank of the highest-ranked clue token within the selected region; rank one is highest.
<table><tr><td>IG reference</td><td>Region</td><td>Clue share (%)</td><td>Difference (pp)</td><td>Best rank</td></tr><tr><td>Training prompts</td><td>All text</td><td>2.07</td><td>-1.33</td><td>44</td></tr><tr><td>Training prompts</td><td>Case text</td><td>22.97</td><td>-3.40</td><td>5</td></tr><tr><td>Aligned paired prompt</td><td>All text</td><td>52.59</td><td>+50.89</td><td>1</td></tr><tr><td>Aligned paired prompt</td><td>Case text</td><td>52.59</td><td>+33.02</td><td>1</td></tr></table>

IG using the aligned paired reference assigns 52.59% of magnitude to the clue and ranks one of its tokens first. This reference changes the width diference (Table 3) and gives zero attribution to tokens with embeddings identical to the original input. The larger share therefore reflects a diferent attribution task, not an isolated improvement in clue recovery.

Magnitude shares are not signed fractions of the width change. The clue is also known from the question’s construction, not from evidence that the trained model relies on it. The dataset limitations in Appendix D therefore restrict the interpretation of this example.

## C.2 Number of integration points

Integrated Gradients approximates its integral by evaluating gradients at intermediate points between the original prompt’s embeddings and the prepared reference embeddings. The final training-reference calculation uses 512 points for 123 references and 2,048 points for the remaining five:

$$
1 2 3 \times 5 1 2 + 5 \times 2 0 4 8 = 7 3 . 2 1 6
$$

gradient evaluation points.

Using 2,048 points for every reference would require

$$
1 2 8 \times 2 0 4 8 = 2 6 2 , 1 4 4
$$

points. The final calculation therefore contains 27.9% of that nominal total. However, the uniform 2,048-point calculation was not run, so its accuracy is unknown.

These totals count only the points contributing to the final estimate. Earlier calculations, repeated evaluations, and diagnostic checks required additional computation. The comparison describes how integration points were allocated; it is not a measured end-to-end speedup.

## D Synthetic dataset revisions and model evaluation

This appendix explains why the original synthetic data can support checks of attribution calculations but cannot establish reliable recovery of meaningful evidence. It also describes a revised dataset and the limits of the models trained on it. We refer to the original and revised datasets as V1 and V2.

## D.1 Limitations of the original dataset

Against its generated labels, the original controlled model achieves approximately 91.7% mean test accuracy. The mean Jensen–Shannon divergence between predicted and prescribed answer-set probabilities is 0.0208; this measures their disagreement, with zero indicating equality. The mean absolute width error is 0.0502. Across three training seeds, every evaluated question pair has lower width after the distinguishing result is supplied.

Subsequent inspection found two problems. First, the diagnostic test named first in the case description identified the labelled answer with 100% accuracy. This made it possible to predict the label without correctly interpreting which result belonged to which test. Second, questions without distinguishing results assigned unequal singleton probabilities to the two plausible answers, favouring the labelled answer without justification in the text.

We did not establish whether the trained model actually used the ordering shortcut. Its availability, together with the unsupported target preference, is enough to prevent treating high accuracy as evidence that the model learned the intended reasoning. The model remains usable for checking whether attribution values account for a specified output diference.

## D.2 Construction of the revised dataset

V2 addresses the identified problems in several ways:

• The order of diagnostic tests is varied independently of the correct answer.

• Each example appears in four cyclic arrangements of the answer options, with the probability targets remapped to preserve the underlying answer identities.

• When the text does not distinguish the two plausible answers, their singleton target probabilities are equal.

• Some paired inputs change only a detail that does not afect the prescribed answer-set probabilities, such as whether the visit occurs in the morning or evening.

• Fictional condition families are separated across training, development, and test data. Development and test data also include wording templates absent from training.

The revised dataset contains 7,680 training, 3,840 development, and 3,840 test records, drawn from 80, 20, and 20 condition families respectively. Diferent option arrangements and wording variants of a case are related observations, not independent examples.

A rule-based reader checks that each generated prompt expresses the intended test results and matches its prescribed target. We also check that paired prompts can be tokenised with the same length and attention mask without deleting, duplicating, or aligning tokens. These are checks of the generated data and its representation, not tests of whether a trained model understands it.

Additional simple predictors examine whether labels can be predicted from test order or from text with the test results hidden. They are fitted using training data only. These diagnostics have no pass/fail threshold and do not prove that every possible shortcut has been removed.

## D.3 Evaluation before attribution

Before explaining a V2 model, we require it to reproduce the prescribed probabilities and respond appropriately to changes in the text. For probability assignments mb and $m _ { ; }$ , the total absolute error is

$$
E _ { m } = \sum _ { \delta \neq S \subseteq \Theta } | { \widehat { m } } _ { S } - m _ { S } | .
$$

Width error is the mean absolute diference between predicted and prescribed widths across the four answers.

The evaluation checks the following:

• Mean total probability error is at most 0.20, assessed separately for questions with and without a distinguishing result.

• Mean width error is at most 0.05, and accuracy on questions with a distinguishing result is at least 0.80.

• The selected answer’s width is lower for the prompt containing the distinguishing result than for its paired prompt without that result in at least 90% of pairs.

• Changing only an irrelevant detail produces mean total probability change at most 0.05 and mean absolute width change at most 0.02.

• Rearranging answer options produces mean total probability change at most 0.10 after predictions are mapped back to the same answer identities.

Predictions must also be nonnegative and sum to one within $1 0 ^ { - 6 }$ . The behavioural criteria are applied separately to familiar and previously unseen wording templates. Results are averaged within condition families before aggregation, so repeated presentations are not treated as independent cases.

## D.4 Training results on the revised dataset

We fit linear output layers and nonlinear output layers with 128 hidden units to frozen SmolLM3 representations, using three training seeds. The nonlinear model completes its 100-epoch schedule by continuing from the saved training state.

Table 10 shows the final total probability errors for that model. Training errors are substantially smaller than development errors on previously unseen wording.

Table 10: Mean total absolute error in the predicted answer-set probabilities after the nonlinear model’s 100-epoch schedule. Development values concern wording templates not used during training.
<table><tr><td>Training seed</td><td>Training error</td><td>Development error</td></tr><tr><td>7</td><td>0.1221</td><td>0.4263</td></tr><tr><td>17</td><td>0.1176</td><td>0.4261</td></tr><tr><td>23</td><td>0.1173</td><td>0.4359</td></tr></table>

No evaluated development checkpoint meets all the required criteria. We therefore report no V2 attribution experiment. This result concerns the tested models and training procedure; it does not establish that random-set uncertainty cannot be learned.

## E Reference preparation and numerical checks.

The following procedure separates changes caused by reference preparation from errors in the attribution calculation.

1. Define the width diference. Choose the answer option and fix the model weights and precision. Evaluate the original prompt and intended reference prompt with their normal tokenisation, masks and positions; record both widths.

2. Evaluate the prepared reference. Record the prepared reference embeddings and their token identities. Evaluate them using the exact mask and positions used for attribution.

3. Check the efect of preparation. Calculate the original prompt’s width minus each of the two reference widths. Report their diference and signs, as in Equation 10. Inspect token changes as well: equal widths alone do not establish that prompts convey the same information.

4. Check numerical accuracy. Compare the signed attribution sum with the width diference for the prepared reference. Repeat with more EG samples or IG integration points, checking attribution vectors as well as rankings. For complete IG integrals, inspect residuals for individual references before averaging; a single EG sample need not satisfy completeness.

5. Evaluate highlighted evidence. State the eligible token region and any clue known from the task’s construction. Compare its attribution magnitude with that of equal-length control windows. Record positions forced to zero by identical input and reference embeddings, and account for related versions of the same question when aggregating results.

The procedure checks the reference and numerical calculation. Whether the learned width represents epistemic uncertainty and whether highlighted tokens identify meaningful evidence require separate model and task evaluations.