# RISR: RESIDUAL-INFORMED SCIENTIFIC EQUATION DISCOVERY WITH LARGE LANGUAGE MODELS

Haobo Li<sup>1</sup>, Wenshuo Zhang<sup>2</sup>, Wenxiao Zhao<sup>3</sup>, Eunseo Jung<sup>2</sup>, Rui Sheng<sup>2</sup>, Yushi Sun<sup>2</sup>, Peiqin Zhuang<sup>1</sup>, Hao Chen<sup>1</sup>, Fenghua Ling<sup>1</sup>

<sup>1</sup>Shanghai AI Laboratory <sup>2</sup>The Hong Kong University of Science and Technology <sup>3</sup>University of California, Los Angeles

## ABSTRACT

Symbolic regression combines structural search with numerical fitting, but aggregate fit scores do not describe how the remaining error varies across inputs. We introduce RISR, a residual-informed method that uses these error patterns to guide formula discovery and learn which corrections are worth fitting. A residual encoder compresses aligned inputs, targets, current predictions, and residuals into continuous tokens that condition a language model to propose formulas. For subsequent refinement, a dual-view relational encoder uses additive and regularized multiplicative residuals to predict the post-fit utility of candidate corrections. We evaluate RISR on scientific tasks from the LLM-SRBench. RISR achieves 63.57% and 38.50% ID accuracy at the 1% and 0.1% pointwise relative-error tolerances, respectively. The corresponding OOD accuracies are 56.07% and 38.24%. RISR outperforms the reported baselines using the same backbone. The results show that our residual-informed approach can improve numerical equation recovery.

## 1 INTRODUCTION

Symbolic regression (SR) discovers explicit mathematical relationships from observations, providing interpretable models for scientific analysis (Schmidt & Lipson, 2009; Udrescu & Tegmark, 2020; Cranmer, 2023). These expressions make dependencies between variables explicit and allow proposed relationships to be examined and compared. Discovering an equation requires both discrete search over expression structures and continuous estimation of numerical coefficients. The two are closely coupled: the usefulness of a structural term depends on its fitted coefficients and its combination with the rest of the equation. A discovery procedure must therefore explore candidate structures while accounting for their numerical behavior on the observations.

Existing methods address this coupled problem through different representations and search strategies. Sparse identification selects a small set of terms from a candidate function library (Brunton et al., 2016). Neural expression search learns a sampling policy through risk-seeking policy gradients (Petersen et al., 2021), while hybrid frameworks integrate neural search, genetic programming, and other complementary strategies (Landajuela et al., 2022). Another line of work trains expression generators on synthetic equation–observation pairs, learning to map numerical observations to symbolic expressions (Biggio et al., 2021; Kamienny et al., 2022). Together, these approaches provide mechanisms for proposing structures and organizing exploration of the expression space.

Recent work increasingly uses numerical evaluation to guide large language models (LLMs) during search. TPSR incorporates fitting accuracy and expression complexity into Transformer decoding through Monte Carlo tree search (Shojaee et al., 2023). For LLM-based program generation, FunSearch combines a pretrained language model with an evaluator in an evolutionary search procedure (Romera-Paredes et al., 2024). LLM-based equation-discovery systems use scientific knowledge and executable programs to construct and revise candidate relationships (Grayeli et al., 2024; Shojaee et al., 2025a; Xia et al., 2026). LLM-SR scores fitted candidates by their mean squared error and uses a score-organized experience buffer to construct subsequent prompts (Shojaee et al., 2025a). SR-Scientist further allows the model to analyze data and residual errors through code execution (Xia et al., 2026).

The form of numerical feedback determines what guidance is available for the next proposal. Aggregate fit scores rank candidate equations, but do not preserve the location or variable dependence of their errors. Equations with comparable overall losses can therefore require different corrections. Pointwise residuals expose how the remaining error varies across inputs, including nonlinear patterns, variable interactions, and localized discrepancies. Turning this information into useful search guidance requires accounting for the current equation and the effects of numerical fitting. A correction suggested by a residual pattern may produce different improvements depending on its parameterization and how it is combined with the current equation. We study how to encode these numerical states for formula generation and how to learn which corrections are worth fitting.

We introduce Residual-Informed Symbolic Regression (RISR), a method for scientific equation discovery. Its search encoder compresses aligned inputs, targets, predictions, and residuals into continuous tokens that condition LLM proposals (Li & Liang, 2021). These proposals are complemented by a learned structural-edit generator. Candidate formulas are fitted and evaluated to obtain an initial equation. For subsequent refinement, a dual-view encoder captures variable-specific pat terns and interactions from additive and regularized multiplicative residuals. It models relationships between variables through attention, following the principle of attention-based set encoding (Lee et al., 2019). A scoring head uses these representations to predict the utility of candidate corrections after coefficient fitting. Corrections are fitted numerically and accepted when the training loss improves, after which the residuals and action ranking are refreshed.

The encoders are trained separately using synthetic data constructed from randomly transformed and composed elementary functions, independently of benchmark equations and observations. The search encoder and its projection are trained from scratch on residual-to-formula examples while the LLM remains frozen. The refinement encoder and scoring head learn from paired current equations with shared observations and candidate actions. Corrections are fitted on one subset of observations, and their outcomes on a disjoint subset provide utility supervision.

We evaluate RISR on scientific problems in the released LLM-SRBench data, covering materials science, chemistry, biology, and physics (Shojaee et al., 2025b). With Qwen3-Coder-30B-A3B-Instruct as the LLM backbone, RISR achieves 63.57% and 38.50% accuracy at the 1% and 0.1% relative-error tolerances. Its performance exceeds that of existing non-LLM or LLM-based methods with the same LLM as backbone and remains competitive with substantially larger backbones. Our contributions are summarized as follows:

• We introduce RISR, a residual-informed method that guides both formula search and subsequent refinement. A search encoder conditions LLM proposals on numerical error patterns, while a dual-view refinement encoder supports correction selection.

• We construct complementary synthetic datasets for training RISR. Random transformations and compositions generate residual-to-formula examples, while paired current equations with shared observations and candidate actions provide fitted-utility supervision on observations held out from coefficient fitting.

• We evaluate RISR on scientific problems. Our method improves overall accuracy over the reported 30B baselines and achieves the highest overall ID accuracy at the 1% tolerance among the baselines, including those using larger backbones.

## 2 RELATED WORK

Symbolic equation discovery Symbolic regression combines expression search with numerical evaluation to recover mathematical relationships from data (Schmidt & Lipson, 2009). To make this search more efficient, physics-inspired methods such as AI Feynman exploit properties including symmetry and separability to simplify the search (Udrescu & Tegmark, 2020). PySR uses a multi-population evolutionary procedure that alternates expression evolution, simplification, and coefficient optimization (Cranmer, 2023). Moving beyond explicit search procedures, neural approaches learn from generated equations and observations: NeSymReS pretrains a Transformer to predict symbolic expressions, while end-to-end symbolic regression predicts expressions and constants jointly before numerical refinement (Biggio et al., 2021; Kamienny et al., 2022). TPSR incorporates fitting accuracy and expression complexity into Transformer decoding through Monte Carlo tree search (Shojaee et al., 2023).

The score of a fitted equation does not determine which structural correction will improve it. The outcome also depends on the current residual pattern, the correction’s parameterization, and its composition with the equation. RISR addresses this decision by learning correction utility from fitted outcomes. Its dual-view encoder represents additive and regularized multiplicative correction signals, while a shared scorer uses these representations to rank corrections before fitting.

LLM-based Symbolic Regression LLM-based methods combine program generation with executable evaluation. FunSearch places a pretrained LLM and an evaluator within an evolutionary program-search loop (Romera-Paredes et al., 2024). LLM-SR combines language-model proposals, parameter fitting, and evolutionary search for SR (Shojaee et al., 2025a). LaSR learns a textual concept library from high-performing hypotheses and uses it to guide subsequent search (Grayeli et al., 2024). Alternatively, SR-Scientist leverages agents for data analysis, equation evaluation, and iterative optimization, including residual inspection through code execution (Xia et al., 2026).

Beyond making residuals available as agents’ input, a central question is how to represent the current numerical state for subsequent formula proposals. RISR learns this representation from synthetic residual-to-formula examples. It jointly processes aligned inputs, targets, current predictions, and residuals, compressing them into fixed-size continuous tokens that condition a frozen LLM alongside the textual search context.

## 3 RISR

Given a set of training observations $\mathcal { D } = \{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { n }$ , where $x _ { i } \in \mathbb { R } ^ { d }$ contains the input variables of observation i and $y _ { i } \in \mathbb { R }$ is its observed response, RISR seeks an explicit equation that predicts the response from the inputs. As shown in Figure 1, RISR proceeds in two stages. In stage A, residualinformed formula search uses a search residual encoder to supply pointwise numerical feedback to a frozen LLM through continuous tokens. LLM proposals and learned structural edits are numerically fitted and evaluated to obtain an initial equation $p _ { 0 }$ . In stage B, residual-guided refinement then uses a dual-view refinement encoder and utility head to rank correction actions for the current equation. Selected corrections are fitted and accepted when they improve the training loss.

![](images/72deb8c570783823cfc8d044c1df3886c8eb5ef6d2d5be442945062fec3114d2.jpg)  
Figure 1: Overview of RISR. (A) Residual-informed formula search conditions a frozen LLM on textual context and numerical tokens from the search residual encoder. LLM proposals and learned structural edits are fitted and evaluated to select an initial equation $p _ { 0 }$ . (B) Residual-guided refinement encodes additive and regularized multiplicative residual views and predicts the fitted utility of unary and product corrections.

## 3.1 RESIDUAL-INFORMED FORMULA SEARCH

Inspired by recent work on LLM-guided program search and symbolic regression (Romera-Paredes et al., 2024; Grayeli et al., 2024; Shojaee et al., 2025a), we iteratively generate and evaluate equation programs. The LLM receives two complementary inputs: textual context describing previously evaluated equations and numerical tokens representing a fitted parent’s residual state.

Textual context. Each prompt contains equation programs sampled from the candidate pool. The highest-scoring equation included in the prompt is the parent equation used for numerical conditioning. The textual context provides the equation-completion instruction, variable descriptions, and sampled programs.

Numerical residual. The search residual encoder processes aligned tuples $( x _ { i } , y _ { i } , p ( x _ { i } ) , y _ { i } \ -$ $p ( x _ { i } ) )$ . A point-attention path aggregates these observations, while a spectral-field path represents coordinate–residual interactions using frequency features and residual-region pooling. Learned gates fuse the two paths, and a projection maps their outputs to 16 continuous tokens in the LLM embedding space. These tokens accompany the textual context, conditioning executable formula proposals on the parent’s input-dependent error pattern. A learnable structural-edit generator also supplies candidates. Given the observations and current formula, the generator combines its own residual encoder with an encoding of the formula’s abstract syntax tree (AST) to predict an addition, replacement, deletion, or stop action. The resulting equation $p _ { 0 }$ , with the lowest mean squared error (MSE), initializes downstream refinement. Appendix A gives more details of the encoder architecture and proposal procedures.

## 3.2 RESIDUAL-GUIDED REFINEMENT

Stage B constructs residual views, scores candidate actions, and fits the selected correction.

## 3.2.1 DUAL-VIEW RESIDUAL ENCODING

For a current equation $p ,$ let $p _ { i } = p ( x _ { i } )$ be its prediction and $\begin{array} { r } { r _ { i } = y _ { i } - p _ { i } } \end{array}$ its residual. We consider two ways to correct its predictions. Let $h ( x )$ denote a candidate correction. 1) An additive update, $p + h ,$ adds an output increment, so matching the observations requires $h ( x _ { i } ) \approx r _ { i } . 2 ,$ ) $\mathbf { A }$ multiplicative update, $p ( 1 + h )$ , applies a relative adjustment to the current prediction. Because its output increment is $p _ { i } h ( x _ { i } )$ , matching the observations requires $p _ { i } h ( x _ { i } ) \approx r _ { i }$ . The two update modes therefore motivate different correction signals for the same residual state.

The additive view uses $r _ { i }$ directly. For the multiplicative view, direct division by $p _ { i }$ would be unstable near zero, so we use the regularized signal:

$$
u _ { i } = \frac { p _ { i } r _ { i } } { p _ { i } ^ { 2 } + \lambda ^ { 2 } } .\tag{1}
$$

Here $\lambda$ is $5 \%$ of the prediction root mean square (RMS), with a positive floor. The dimensionless quantity $u _ { i }$ approximates the fractional adjustment $r _ { i } / p _ { i }$ when $| p _ { i } |$ is large relative to $\lambda ,$ and suppresses unstable ratios near zero. Appendix B gives more details.

The two residual views shown combine the same input coordinates, targets, and predictions with their respective correction signals, r and u. Each view summarizes correlations between transformed coordinates and observation signals. Pairwise statistics capture relationships between variables and the alignment of product corrections with each view’s signal.

The two views share an embedding network and two relation-attention blocks that encode the variables. Pair features contribute to both attention biases and values. For each variable, we fuse its representations from the two views with a pooled global representation, producing $\mathbf { h } _ { j }$ for variable $j .$ This set-structured design shares parameters across variables and their relationships (Zaheer et al., 2017; Lee et al., 2019). Padding and masks support inputs with up to eight variables.

## 3.2.2 CORRECTION UTILITY PREDICTION

Each correction action selects an additive or multiplicative update, one variable or two distinct variables, and an operator for each selected variable. We use eight operator families:

$$
\Phi = \left\{ z , z ^ { 2 } , z ^ { 3 } , \sin z , \operatorname { t a n h } z , \exp z , \log ( 1 + z ^ { 2 } ) , ( 1 + z ^ { 2 } ) ^ { - 1 } \right\}\tag{2}
$$

where z denotes a dimensionless scalar argument. A unary action applies one operator to one variable, while a product action combines two transformed variables. Masks exclude actions involving inactive variables.

A shared query network combines each variable representation $\mathbf { h } _ { j }$ with an operator embedding $\mathbf { e } _ { k }$ to form a variable–operator query. For product actions, the two queries are combined symmetrically and augmented with their product correlations from both residual views. A shared scoring head combines this representation with the update mode and action type to predict the fitted utility $Q _ { \theta } ( \mathcal { D } , p , a )$ . Appendix B specifies the action encoding and network architecture.

## 3.2.3 COEFFICIENT FITTING AND ADAPTIVE UPDATES

Coefficient fitting. For the selected action, numerical fitting determines the coefficients of its correction function. For a subset S of observation indices, we define the relative squared-error objective

$$
L _ { S } ( f ) = \frac { 1 } { | S | } \sum _ { i \in S } \left( \frac { f ( x _ { i } ) - y _ { i } } { s _ { i , S } } \right) ^ { 2 } ,\tag{3}
$$

where $s _ { i , S }$ is the target magnitude with a positive floor scaled to the target RMS on $S .$ During refinement, fitting and acceptance use all training observations.

An action with selected variables J and operators $\phi _ { k _ { j } }$ defines the correction

$$
h _ { a } ( x ) = \alpha \prod _ { j \in J } \phi _ { k _ { j } } ( b _ { j } z _ { j } + c _ { j } ) + \beta ,\tag{4}
$$

where $z _ { j }$ is the standardized value of input variable $j$ and $| J | \in \{ 1 , 2 \}$ . The parameters $b _ { j } , c _ { j }$ control each operator’s input scale and shift, while $\alpha , \beta$ control the correction’s amplitude and offset. The selected update mode gives either $p + h _ { a }$ or $p ( 1 + h _ { a } )$ . We fit the correction by variable projection: for each setting of the nonlinear parameters $b _ { j } , c _ { j }$ , weighted least squares determines the amplitude α and offset $\beta$ (Golub & Pereyra, 1973; O’Leary & Rust, 2013).

Adaptive action selection. At each attempt, the frozen scorer selects the highest-scoring untried action. The fitted correction is accepted if it reduces the training relative squared error by more than 0.1%. After acceptance, the updated equation determines new residual views and a refreshed action ranking. Rejected actions leave the equation unchanged, and every attempted action is marked as tried. Refinement uses at most four action-fitting attempts, including rejected attempts, and makes no additional LLM calls. Appendix B gives the point-selection protocol, numerical fitting settings, and early-stopping criterion.

## 3.3 SYNTHETIC DATA AND MODEL TRAINING

We train the search residual encoder, structural-edit generator, and refinement model separately using synthetic data generated from elementary functions with sampled transformations, coefficients, and input coordinates, independently of the benchmark equations and observations. All three components are trained offline, while the pretrained LLM parameters remain frozen. Appendix C provides more details.

## 3.3.1 SEARCH RESIDUAL ENCODER DATA AND TRAINING

Target formulas combine randomly transformed elementary functions into compositional products, smooth regime transitions, or condition-gated responses. Current equations omit components or use affine approximations. Each example pairs a residual state from 256 observations over one to six variables with its target formula. Validation rejects nonfinite, nearly constant, or excessively large targets and checks numerical alignment and residual consistency. All target programs pass parsing and compilation. Audits confirm no cross-split overlap in example identities or exact normalized numerical inputs. The dataset contains 6,000 training and 768 validation examples. We train the search residual encoder and projector from scratch by minimizing the LLM loss on target-formula tokens, with the pretrained LLM parameters frozen.

## 3.3.2 STRUCTURAL-EDIT DATA AND TRAINING

For structural-edit generator training, we generate 20,000 training and 2,000 validation examples, each with 256 observations over one to six variables. Current formulas omit a target component, contain an extra or incorrect component, or already match the target, providing supervision for addition, deletion, replacement, and stopping. Numerical evaluation of candidate edits supplies outcome regression and ranking targets. We train the program encoder and editing modules using supervised edit losses, parameter and outcome regression, and candidate ranking.

## 3.3.3 REFINEMENT-ENCODER DATA AND TRAINING

Refinement-encoder dataset. Targets combine three to six unary or product components. For each target, we construct paired current equations with different residual patterns, covering both coarse and near-target states. Each pair shares 256 observations and eight candidate actions selected by frozen policies and random sampling. Sharing observations and actions allows supervision to capture how an action’s utility changes with the current equation.

The splits contain 8,192 training pairs, 1,024 validation pairs, and three test sets of 1,024 pairs covering new observation clouds, held-out product actions, and eight-variable inputs. Together, they provide 24,576 states and 196,608 action-utility labels. Validation checks action alignment, variable eligibility, and acceptance decisions.

Fitted-action utility. For each synthetic problem, we divide the 256 observations into 192 fitting rows $F$ and 64 utility rows H. Rows in $F$ supply the encoder inputs, coefficient fits, and acceptance decisions; rows in $\dot { H }$ measure the resulting utility. The losses $L _ { F }$ and $L _ { H }$ apply Eq. 3 to these respective subsets.

For each current equation and candidate action, we fit the correction on F and accept it if it reduces $L _ { F }$ by more than 0.1%. Let $p _ { a } ^ { \prime }$ denote the accepted equation, or the unchanged parent when the action is rejected. Its supervision target is the clipped log reduction in error on ${ \bar { H } } :$

$$
U ( p , a ) = \mathrm { c l i p } _ { [ - 3 , 3 ] } \left[ \log _ { 1 0 } \frac { L _ { H } ( p ) + \varepsilon } { L _ { H } ( p _ { a } ^ { \prime } ) + \varepsilon } \right] ,\tag{5}
$$

where $\varepsilon > 0$ stabilizes the logarithm. Rejected actions have zero utility, while accepted actions receive positive or negative utility according to their effect on H. The target therefore measures the outcome of fitting a correction beyond the rows used to estimate its coefficients.

Training objective. We jointly train the relational encoder and action scorer. The loss is:

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { v a l u e } } + 0 . 5 \mathcal { L } _ { \mathrm { r a n k } } + 0 . 5 \mathcal { L } _ { \Delta } .\tag{6}
$$

The value term ${ \mathcal { L } } _ { \mathrm { v a l u e } }$ regresses measured utilities. The ranking term $\mathcal { L } _ { \mathrm { r a n k } }$ learns their ordering within each state (Burges et al., 2005). The paired-state term $\mathcal { L } _ { \Delta }$ matches utility differences for the same action across the two current equations. We train with AdamW (Loshchilov & Hutter, 2019) for 16 epochs, corresponding to 2,048 updates, at a learning rate of $3 \times 1 0 ^ { - 4 }$

## 4 EVALUATION

## 4.1 EVALUATION SETTING

Dataset As many datasets may have been included in LLM pretraining corpora, LLM-SRBench (Shojaee et al., 2025b) combines known terms in the underlying equations with novel, synthetically generated terms to create problems that go beyond memorization. To ensure scientific rigor, two subject-matter experts verified all equations and visualizations of the corresponding generated data. We evaluate our approach on LSR-Synth, following Xia et al. (Xia et al., 2026). It contains 25 problems in materials science, 36 in chemistry, 24 in biology, and 44 in physics. Each problem contains a training set for SR methods, along with separate in-domain (ID) and out-ofdomain (OOD) test sets.

Evaluation Metrics We report accuracy-to-tolerance following the released SR-Scientist (Xia et al., 2026; Kamienny et al., 2022; Biggio et al., 2021). Point i passes when $| \widehat { y } _ { i } - y _ { i } | / | y _ { i } | \le \tau$ Writing this pointwise indicator as $b _ { i } ( \tau )$ , task success is

$$
\mathrm { P a s s } _ { \tau } ( f ) = { \bf 1 } \left[ \frac { 1 } { N } \sum _ { i = 1 } ^ { N } b _ { i } ( \tau ) \geq 0 . 9 5 \right] .\tag{7}
$$

Table 1: Scientific ID accuracy (%), measured by $\mathrm { A c c } _ { 1 \% }$ and $\mathrm { A c c } _ { 0 . 1 \% }$ . The best and second-best results are highlighted in bold and underlined, respectively.
<table><tr><td rowspan="2">Method</td><td colspan="2">Overall</td><td colspan="2">Materials</td><td colspan="2">Chemistry</td><td colspan="2">Biology</td><td colspan="2">Physics</td></tr><tr><td> $\mathrm { A c c } _ { 1 \% }$ </td><td> $\mathrm { A c c } _ { 0 . 1 \% }$ </td><td> $\mathrm { A c c _ { 1 \% } }$ </td><td> $\mathrm { A c c } _ { 0 . 1 \% }$ </td><td> $\mathrm { A c c } _ { 1 \% }$ </td><td> $\mathrm { A c c _ { 0 . 1 \% } }$ </td><td> $\mathrm { A c c } _ { 1 \% }$ </td><td> $\mathrm { A c c } _ { 0 . 1 \% }$ </td><td> $\operatorname { A c c } _ { 1 \% }$ </td><td> $\mathrm { A c c } _ { 0 . 1 \% }$ </td></tr><tr><td colspan="10">Without LLMs</td></tr><tr><td>E2E</td><td>0.26</td><td>0.00</td><td>1.33</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>NeSymReS</td><td>3.10</td><td>0.78</td><td>8.00</td><td>4.00</td><td>2.78</td><td>0.00</td><td>4.17</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>DSR</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>uDSR</td><td>29.46</td><td>12.40</td><td>36.00</td><td>8.00</td><td>50.00</td><td>25.00</td><td>29.17</td><td>8.33</td><td>9.09</td><td>6.82</td></tr><tr><td>PySR</td><td>29.46</td><td>14.47</td><td>53.33</td><td>22.67</td><td>25.93</td><td>11.11</td><td>16.67</td><td>6.95</td><td>25.76</td><td>16.67</td></tr><tr><td colspan="10">GLM-4.5-Air (Zeng et al., 2025)</td></tr><tr><td>LaSR</td><td>14.21</td><td>8.53</td><td>18.67</td><td>12.00</td><td>16.67</td><td>10.18</td><td>11.11</td><td>5.56</td><td>11.36</td><td>6.82</td></tr><tr><td>LLM-SR</td><td>35.92</td><td>14.47</td><td>61.33</td><td>38.67</td><td>30.56</td><td>3.71</td><td>18.06</td><td>4.17</td><td>35.61</td><td>15.15</td></tr><tr><td>SR-Scientist</td><td>48.32</td><td>25.06</td><td>81.33</td><td>70.67</td><td>45.37</td><td>11.11</td><td>40.28</td><td>16.66</td><td>36.37</td><td>15.15</td></tr><tr><td colspan="10">Qwen3-Coder-480B-A35B-Instruct ( (Yang et al., 2025)</td></tr><tr><td>LaSR</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LLM-SR</td><td>11.89 41.08</td><td>7.49 18.09</td><td>13.33 80.00</td><td>6.67 52.00</td><td>16.67 36.11</td><td>9.26 9.26</td><td>6.95 30.56</td><td>6.95 18.06</td><td>9.85 28.79</td><td>6.82 6.06</td></tr><tr><td>SR-Scientist</td><td>49.09</td><td>24.55</td><td>86.67</td><td>69.33</td><td>40.74</td><td>5.56</td><td>50.00</td><td>26.39</td><td>34.09</td><td>13.64</td></tr><tr><td colspan="10">GPT-OSS-120B (Agarwal et al., 2025)</td></tr><tr><td>LaSR</td><td>16.02</td><td>10.08</td><td>20.00</td><td>12.00</td><td>18.52</td><td>12.96</td><td>9.72</td><td>5.56</td><td>15.15</td><td>9.09</td></tr><tr><td>LLM-SR</td><td>28.16</td><td>11.37</td><td>64.00</td><td>38.67</td><td>22.22</td><td>3.71</td><td>11.11</td><td>2.78</td><td>21.97</td><td>6.82</td></tr><tr><td>SR-Scientist</td><td>63.57</td><td>49.35</td><td>74.67</td><td>60.00</td><td>81.48</td><td>64.81</td><td>66.67</td><td>43.05</td><td>40.91</td><td>34.09</td></tr><tr><td colspan="10">Qwen3-Coder-30B-A3B-Instruct (Yang et al., 2025)</td></tr><tr><td>LaSR</td><td>12.66</td><td>8.27</td><td>20.00</td><td>10.67</td><td>17.59</td><td>12.04</td><td>8.33</td><td>6.95</td><td>6.82</td><td>4.55</td></tr><tr><td>LLM-SR</td><td>24.55</td><td>7.24</td><td>29.33</td><td>6.67</td><td>30.55</td><td>10.18</td><td>15.28</td><td>5.55</td><td>21.97</td><td>6.06</td></tr><tr><td>SR-Scientist</td><td>32.30</td><td>16.02</td><td>81.33</td><td>52.00</td><td>22.22</td><td>5.56</td><td>22.22</td><td>8.33</td><td>18.18</td><td>8.33</td></tr><tr><td>SR-Scientist + RL</td><td>40.92</td><td>20.69</td><td>85.33</td><td>65.33</td><td>37.38</td><td>7.46</td><td>29.17</td><td>11.11</td><td>25.00</td><td>11.37</td></tr><tr><td>RISR</td><td>63.57</td><td>38.50</td><td>89.33</td><td>58.67</td><td>62.04</td><td>30.56</td><td>50.00</td><td>25.00</td><td>57.58</td><td>40.91</td></tr></table>

Accuracy is the percentage of tasks that pass, reported at $\tau = 0 . 0 1$ and 0.001.

![](images/a4e272fa58dfdbcb6578c97bd7ea513dfb24f6dc0e174377dc6b93807f66babe.jpg)  
Figure 2: ID and OOD accuracy at a 1% relative-error tolerance. Colors denote methods. Filled and hatched bars represent ID and OOD results, respectively. All LLM-based methods use Qwen3- Coder-30B-A3B-Instruct.

Baselines We compare RISR against both non-LLM and LLM-based symbolic regression methods. The non-LLM baselines include E2E (Kamienny et al., 2022), NeSymReS (Biggio et al., 2021), DSR (Petersen et al., 2021), uDSR (Landajuela et al., 2022), and PySR (Cranmer, 2023), while the LLM-based baselines comprise LaSR (Grayeli et al., 2024), LLM-SR (Shojaee et al., 2025a), and SR-Scientist (Xia et al., 2026). We use the published results reported by (Xia et al., 2026).

Table 2: The ablation results with one seed setting. Accuracy is reported at the 1% / 0.1% tolerances. For Stage B, all rows share the same Stage-A initial equations and numerical fitting protocol.
<table><tr><td>Configuration</td><td>ID accuracy ↑</td><td>OOD accuracy ↑</td></tr><tr><td>LLM-only formula search</td><td>7.75 / 0.00</td><td>9.30 / 3.10</td></tr><tr><td>+ residual-informed formula search (Stage A)</td><td>36.43 / 23.26</td><td>40.31 / 30.23</td></tr><tr><td>+ coefficient fitting</td><td>59.69 / 34.11</td><td>53.49 / 38.76</td></tr><tr><td>+ random corrections</td><td>60.47 / 34.88</td><td>52.71 / 36.43</td></tr><tr><td>+ learned-scored corrections (RISR)</td><td>63.57 / 37.21</td><td>55.04 / 38.76</td></tr></table>

Implementation and Protocol We use Qwen3-Coder-30B-A3B-Instruct as the search backbone. Its pretrained parameters remain frozen during encoder training and evaluation. Each search has 64 outer candidate slots, including initialization. LLM proposals and learned structural edits share this budget. Stage B first fits the selected equation’s coefficients and then allows at most four new correction-fitting attempts, including rejections, without further LLM calls. Within-run equation selection and fitting use training observations, leaving the test set for evaluation.

## 4.2 MAIN RESULTS

ID Performance Table 1 reports the ID performance of RISR, averaged across results of three seeds. More details can be found in Appendix D. Among methods using the same backbone, RISR achieves 63.57% and 38.50% accuracy at the 1% and 0.1% tolerances, respectively, exceeding SR-Scientist+RL by 22.65 and 17.81 percentage points. The gains span all four domains at the 1% tolerance, with particularly large improvements in chemistry and biology: 24.66 and 20.83 percentage points, respectively. This breadth indicates that the overall improvement is shared across scientific domains rather than concentrated in a single problem category.

The 30B model results are also competitive with systems built on substantially larger backbones. RISR exceeds the strongest reported Qwen3-Coder-480B result in seven of the ten dimensions. At the 1% tolerance, across all baselines, it matches the highest reported overall accuracy, 63.57% for SR-Scientist with GPT-OSS-120B. At the stricter 0.1% tolerance, its 38.50% accuracy ranks second overall, ahead of all reported GLM-4.5-Air and Qwen3-Coder-480B configurations. These results demonstrate the effectiveness of the complete RISR pipeline at the 30B scale and show that leading accuracy at the 1% tolerance can be achieved without adopting a larger LLM backbone.

As for domains, the strongest cross-backbone results occur in materials science and physics. RISR achieves the highest materials accuracy at the 1% tolerance, reaching 89.33%. In physics, it lead at both tolerances with 57.58% and 40.91%, exceeding the strongest competing results of 120B model by 16.67 and 6.82 percentage points, respectively. In chemistry, RISR ranks second at both tolerances. In biology, it ties for second place at the 1% tolerance.

OOD performance RISR achieves the highest overall OOD accuracy among the compared methods at the same 30B model scale (Figure 2). At the 1% tolerance, its accuracy reaches 56.07%, compared with 39.50% for SR-Scientist + RL, a gain of 16.57 percentage points. At 1%, OOD accuracy is 84.00% in materials science, 48.15% in chemistry, 48.61% in biology, and 50.76% in physics. Physics retains 40.91% accuracy at the stricter 0.1% tolerance. These results show that the advantage of RISR extends beyond ID fitting to prediction outside the training domain using the same discovered equations.

## 4.3 ABLATION STUDY

Table 2 isolates the contributions of coefficient fitting and the learned correction-selection policy. The complete residual-informed search raises accuracy at the 1% tolerance from 7.75% to 36.43% on ID data and from 9.30% to 40.31% on OOD data compared with the LLM-only method. Residualguided refinement provides additional gains, reaching 63.57% ID and 55.04% OOD accuracy at the 1% tolerance. Compared with random corrections, learned-scored corrections improve ID and OOD accuracy at the 1% tolerance by 3.10 and 2.33 percentage points. The results show a positive contribution from residual-informed formula search and learned action selection under the fixed correction budget.

![](images/b8c81f2e377fb992234a208c4c8bc8db0e2acb147e1804cd1954508851c5410b.jpg)

![](images/4f9abfd4e96b34ed9667b2d70c6b9cfaa5a18b37f2aa335c9e1aa77c1f375e3b.jpg)  
Figure 3: Case study on PO30. RISR retains the target components, removes an extraneous term through structural editing, and calibrates the coefficients during refinement.

## 4.4 SYMBOLIC ACCURACY

Symbolic accuracy (SA) evaluates whether a discovered equation can represent the target relationship. We follow the symbolic-assessment criterion of SR-Scientist (Xia et al., 2026): A candidate is counted as positive when an assignment of its adjustable constants can represent the target expression. GPT-6-Astra serves as our judge, and non-unanimous cases are reserved for human review. 17 problems receive ten Yes votes, and 112 receive ten No votes, leaving no voting disagreements. We additionally manually reviewed all 17 cases classified as positive by the LLM judge. RISR obtains an SA of 13.18% (17/129), exceeding existing methods (Table 3). SA and numerical accuracy are complementary: predictive agreement on held-out observations does not necessarily imply structural recovery under the constant-assignment criterion. Appendix E documents more details.

Table 3: SA (%) on one seed setting.
<table><tr><td>Method</td><td>LLM backbone</td><td>SA↑</td></tr><tr><td>uDSR</td><td></td><td>0.77</td></tr><tr><td>PySR</td><td></td><td>4.65</td></tr><tr><td>LLM-SR</td><td>GLM-4.5-Air</td><td>5.43</td></tr><tr><td>LLM-SR LLM-SR</td><td>GPT-OSS-120B Qwen3-Coder-480B</td><td>4.65</td></tr><tr><td>SR-Scientist</td><td>GLM-4.5-Air</td><td>5.43 7.75</td></tr><tr><td>SR-Scientist</td><td>GPT-OSS-120B</td><td></td></tr><tr><td>SR-Scientist</td><td></td><td>7.00</td></tr><tr><td></td><td>Qwen3-Coder-480B</td><td>7.00</td></tr><tr><td>RISR</td><td>Qwen3-Coder-30B</td><td>13.18</td></tr></table>

## 4.5 CASE STUDY

PO30 illustrates how RISR contributes to equation recovery (Figure 3). The target describes a nonlinear oscillator, with position x and velocity v. During Stage A, the formula proposal contains all four target components, but also includes an extra term c ∗ v. Then the learned structural-edit generator replaces this extra term with a constant and submits the edited formula for coefficient fitting and scoring. At Stage B, coefficient fitting calibrates the selected equation without changing its structure. The remaining constant becomes negligible, and the two independently fitted sinusoidal coefficients closely match the shared coefficient in the target.

We examine more cases (Appendix F), including those in which numerical error is substantially reduced despite incomplete recovery of the underlying formula. These cases may inform future research aimed not only at reducing numerical error but also at recovering the scientific formulas.

## 5 CONCLUSION

We present RISR, a two-stage method that uses the pointwise residual information of a current equation to guide scientific equation discovery. Residual-conditioned LLM proposals and structuraledit generators support formula search, while a dual-view encoder predicts which corrections are likely to improve the equation after fitting. For the evaluation, RISR reaches 63.57% and 38.50% ID accuracy at the 1% and 0.1% tolerances. The case study illustrates how structural editing and coefficient calibration contribute at different points in the discovery process. The evaluation results show the effectiveness of our method in the task of scientific equation discovery.

## AI USE STATEMENT

In this work, we used generative AI tools to assist in the writing of proofs, implement methods, assist with translation, and formulate mathematical claims. We have not used generative AI tools to generate synthetic data sets, propose or refine hypotheses, design or provide feedback on research methodology or experiments, clean and reformat datasets, support qualitative and thematic data analysis, or interpret results. Providing critical ingredients for proving mathematical claims and helping develop theoretical models or conceptual frameworks is not applicable to this work. We have reviewed all AI-assisted work. We take responsibility for the final content of this work, including text, claims, or artifacts produced with the aid of generative AI.

## ETHICS STATEMENT

RISR is developed and evaluated using publicly available scientific benchmark data and synthetic numerical datasets generated through transformations and compositions of elementary mathematical functions. This work focuses on methodological research in scientific equation discovery. We have not identified specific ethical concerns associated with the data and experiments presented in this study.

## REPRODUCIBILITY STATEMENT

The main paper and appendices describe the model architectures, synthetic data generation procedures, training settings, and evaluation protocols used in our experiments. To facilitate reproducibility, we will publicly release the synthetic datasets and the code for data generation, model training, equation search and refinement, and evaluation.

## REFERENCES

Sandhini Agarwal, Lama Ahmad, Jason Ai, Sam Altman, Andy Applebaum, Edwin Arbus, Rahul K Arora, Yu Bai, Bowen Baker, Haiming Bao, et al. gpt-oss-120b & gpt-oss-20b model card. arXiv preprint arXiv:2508.10925, 2025.

Luca Biggio, Tommaso Bendinelli, Alexander Neitz, Aurelien Lucchi, and Giambattista Parascandolo. Neural symbolic regression that scales. In International conference on machine learning, pp. 936–945. Pmlr, 2021.

Steven L. Brunton, Joshua L. Proctor, and J. Nathan Kutz. Discovering governing equations from data by sparse identification of nonlinear dynamical systems. Proceedings of the National Academy of Sciences, 113(15):3932–3937, 2016. doi: 10.1073/pnas.1517384113. URL https://www.pnas.org/doi/abs/10.1073/pnas.1517384113.

Chris Burges, Tal Shaked, Erin Renshaw, Ari Lazier, Matt Deeds, Nicole Hamilton, and Greg Hullender. Learning to rank using gradient descent. In Proceedings of the 22nd International Conference on Machine Learning, ICML ’05, pp. 89–96, New York, NY, USA, 2005. Association for Computing Machinery. ISBN 1595931805. doi: 10.1145/1102351.1102363. URL https://doi.org/10.1145/1102351.1102363.

Miles Cranmer. Interpretable machine learning for science with pysr and symbolicregression.jl. arXiv preprint arXiv:2305.01582, 2023. doi: 10.48550/arXiv.2305.01582.

G. H. Golub and V. Pereyra. The differentiation of pseudo-inverses and nonlinear least squares problems whose variables separate. SIAM Journal on Numerical Analysis, 10(2):413–432, 1973. doi: 10.1137/0710036. URL https://doi.org/10.1137/0710036.

Arya Grayeli, Atharva Sehgal, Omar Costilla-Reyes, Miles Cranmer, and Swarat Chaudhuri. Symbolic regression with a learned concept library. In Proceedings of the 38th International Conference on Neural Information Processing Systems, NIPS ’24, Red Hook, NY, USA, 2024. Curran Associates Inc. ISBN 9798331314385.

Pierre-Alexandre Kamienny, Stephane d’Ascoli, Guillaume Lample, and Franc¸ois Charton. End-to-´ end symbolic regression with transformers. In Proceedings of the 36th International Conference on Neural Information Processing Systems, NIPS ’22, Red Hook, NY, USA, 2022. Curran Associates Inc. ISBN 9781713871088.

Mikel Landajuela, Chak Shing Lee, Jiachen Yang, Ruben Glatt, Claudio P Santiago, Ignacio Aravena, Terrell Mundhenk, Garrett Mulcahy, and Brenden K Petersen. A unified framework for deep symbolic regression. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh (eds.), Advances in Neural Information Processing Systems, volume 35, pp. 33985–33998. Curran Associates, Inc., 2022. doi: 10.52202/ 068431-2463. URL https://proceedings.neurips.cc/paper\_files/paper/ 2022/file/dbca58f35bddc6e4003b2dd80e42f838-Paper-Conference.pdf.

Juho Lee, Yoonho Lee, Jungtaek Kim, Adam Kosiorek, Seungjin Choi, and Yee Whye Teh. Set transformer: A framework for attention-based permutation-invariant neural networks. In Kamalika Chaudhuri and Ruslan Salakhutdinov (eds.), Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research, pp. 3744–3753. PMLR, 09–15 Jun 2019. URL https://proceedings.mlr.press/v97/ lee19d.html.

Xiang Lisa Li and Percy Liang. Prefix-tuning: Optimizing continuous prompts for generation. In Chengqing Zong, Fei Xia, Wenjie Li, and Roberto Navigli (eds.), Proceedings of the 59th Annual Meeting ofthe Associationfor Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pp. 4582–4597, Online, August 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.acl-long.353. URL https://aclanthology.org/2021.acl-long.353/.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In 7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019. OpenReview.net, 2019. URL https://openreview.net/forum?id=Bkg6RiCqY7.

Dianne P. O’Leary and Bert W. Rust. Variable projection for nonlinear least squares problems. Comput. Optim. Appl., 54(3):579–593, April 2013. ISSN 0926-6003. doi: 10.1007/ s10589-012-9492-9. URL https://doi.org/10.1007/s10589-012-9492-9.

Brenden K. Petersen, Mikel Landajuela, T. Nathan Mundhenk, Claudio Prata Santiago, Sookyung´ Kim, and Joanne Taery Kim. Deep symbolic regression: Recovering mathematical expressions from data via risk-seeking policy gradients. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net, 2021. URL https://openreview.net/forum?id=m5Qsh0kBQG.

Bernardino Romera-Paredes, Mohammadamin Barekatain, Alexander Novikov, Matej Balog, M Pawan Kumar, Emilien Dupont, Francisco JR Ruiz, Jordan S Ellenberg, Pengming Wang, Omar Fawzi, et al. Mathematical discoveries from program search with large language models. Nature, 625(7995):468–475, 2024. doi: https://doi.org/10.1038/s41586-023-06924-6.

Michael Schmidt and Hod Lipson. Distilling free-form natural laws from experimental data. Science, 324(5923):81–85, 2009. doi: 10.1126/science.1165893. URL https://www.science. org/doi/abs/10.1126/science.1165893.

Parshin Shojaee, Kazem Meidani, Amir Barati Farimani, and Chandan Reddy. Transformerbased planning for symbolic regression. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine (eds.), Advances in Neural Information Processing Systems, volume 36, pp. 45907–45919. Curran Associates, Inc., 2023. doi: 10.52202/ 075280-1990. URL https://proceedings.neurips.cc/paper\_files/paper/ 2023/file/8ffb4e3118280a66b192b6f06e0e2596-Paper-Conference.pdf.

Parshin Shojaee, Kazem Meidani, Shashank Gupta, Amir Barati Farimani, and Chandan K. Reddy. LLM-SR: scientific equation discovery via programming with large language models. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net, 2025a. URL https://openreview.net/forum?id= m2nmp8P5in.

Parshin Shojaee, Ngoc-Hieu Nguyen, Kazem Meidani, Amir Barati Farimani, Khoa D Doan, and Chandan K. Reddy. LLM-SRBench: A new benchmark for scientific equation discovery with large language models. In Aarti Singh, Maryam Fazel, Daniel Hsu, Simon Lacoste-Julien, Felix Berkenkamp, Tegan Maharaj, Kiri Wagstaff, and Jerry Zhu (eds.), Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings ofMachine Learning Research, pp. 55325–55359. PMLR, 13–19 Jul 2025b. URL https://proceedings.mlr. press/v267/shojaee25a.html.

Silviu-Marian Udrescu and Max Tegmark. Ai feynman: A physics-inspired method for symbolic regression. Science Advances, 6(16):eaay2631, 2020. doi: 10.1126/sciadv.aay2631. URL https://www.science.org/doi/abs/10.1126/sciadv.aay2631.

Shijie Xia, Yuhan Sun, and Pengfei Liu. Sr-scientist: Scientific equation discovery with agentic ai. In International Conference on Learning Representations, volume 2026, pp. 75787–75811, 2026.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

Manzil Zaheer, Satwik Kottur, Siamak Ravanbhakhsh, Barnabas P´ oczos, Ruslan Salakhutdinov, and´ Alexander J Smola. Deep sets. In Proceedings of the 31st International Conference on Neural Information Processing Systems, NIPS’17, pp. 3394–3404, Red Hook, NY, USA, 2017. Curran Associates Inc. ISBN 9781510860964.

Aohan Zeng, Xin Lv, Qinkai Zheng, Zhenyu Hou, Bin Chen, Chengxing Xie, Cunxiang Wang, Da Yin, Hao Zeng, Jiajie Zhang, et al. Glm-4.5: Agentic, reasoning, and coding (arc) foundation models. arXiv preprint arXiv:2508.06471, 2025.

## A DETAILS OF RESIDUAL-INFORMED FORMULA SEARCH

This section describes the configuration used for RISR. Stage A combines residual-conditioned LLM proposals and learned structural edits.

## A.1 FORMULA SEARCH AND NUMERICAL INPUTS

The search extends the evaluator–buffer–sampler design of LLM-SR (Shojaee et al., 2025a). Executable candidates are fitted to the training observations and stored with their training MSE, predictions, and residuals. The buffer maintains ten islands and samples two equation programs from one island to form each prompt. The highest-scoring equation included in that prompt supplies the parent predictions for numerical conditioning. The textual input contains an equation-completion instruction, variable descriptions, and the sampled code context. Pointwise observations are supplied through continuous embeddings rather than a natural-language account of the residuals.

The residual encoder receives aligned coordinates $x _ { i } ,$ targets $y _ { i } ,$ parent predictions $p _ { i } ,$ , and residuals $\boldsymbol { r } _ { i } = y _ { i } - p _ { i }$ . It has six coordinate slots. For a problem with more than six variables, each coordinate is scored by its largest absolute correlation with the residual, considering raw, absolute-value, and squared transformations on both sides. The six highest-scoring coordinates are retained in their original order, with coordinate-index tie breaking. Routing affects only this encoder: the equation signature and numerical fitting retain all original variables. The search configuration allows up to 4,000 encoding rows. If subsampling is required, rows are sorted by residual and one aligned row is sampled from each equal-count stratum.

Normalization statistics are computed on the retained rows. Coordinate $j$ is centered by its median and divided by the larger of its interquartile range and standard deviation. A scale at most $1 0 ^ { - 8 }$ is replaced by one. Coordinates are clipped to [−12, 12] and padded to six slots. Let $\mu _ { y }$ be the target mean and $\sigma _ { y } = \operatorname* { m a x } \{ \mathrm { s t d } ( y ) , 1 0 ^ { - 8 } \}$ . Targets and predictions are centered by $\mu _ { y }$ and divided by $\sigma _ { y } .$ Residuals are divided by $\sigma _ { y }$ without centering. These three channels are clipped to [−20, 20]. The normalized residual’s magnitude and signed log(1 + |r|) complete five observation channels. Each row therefore has eleven entries, with a separate mask for active coordinate slots.

## A.2 POINT-ATTENTION AND SPECTRAL-FIELD PATHS

Figure 4 shows how the residual encoder maps aligned numerical observations to continuous LLM tokens through point-attention and spectral-field paths.

Point-attention path. A two-layer 11 → 256 → 256 MLP embeds each row. Two eight-head self-attention blocks mix point features, followed by layer normalization. For large observation sets, attention operates within chunks after sorting by normalized residual plus a small deterministic projection of the full row. Search-time chunks contain at most 256 points. Sixteen learned queries aggregate the valid points through two cross-attention blocks. Their shared context combines an MLP of feature means, standard deviations, and the variable mask with an MLP of the mask alone. A final layer normalization yields sixteen vectors of width 256.

Spectral-field path. The second path adds an explicit feature lift to the point representations. It contains 24 coordinate channels (each coordinate alone and multiplied by the normalized residual, its magnitude, or its square), 21 coordinate-pair products multiplied by the residual, 72 sine/cosine channels, and the five observation channels. Thus, each point supplies $2 4 + 2 1 + 7 2 + 5 = 1 2 2$ features. The coordinate-pair channels include squares. Six positive frequencies, shared across coordinates, are learned in log space and initialized on a logarithmic grid from one to eight. Their sinusoidal arguments include a factor of π. Features associated with padded coordinates are masked. An MLP maps the lift to width 256 and adds it to the point features. These are learned frequency features, not an FFT of the observations.

Seven learned residual centers, initialized uniformly from −3 to 3, define additional summaries. For each center, Gaussian-shaped weights depend on normalized residual distance and a shared learned positive precision. Weights sum to one over valid points. A weighted field-feature average and a learned region embedding form each summary. Sixteen field queries attend to both point features and the seven summaries. Two cross-attention/latent-self-attention pairs, followed by layer normalization, produce sixteen width-256 vectors.

Fusion and projection. One learned sigmoid gate per token interpolates between the two paths. The gates are shared across examples. A two-layer GELU projector with widths 256 → 2048 → 2048, layer normalization, and a learned positive output scale maps the fused vectors to the Qwen embedding space. The scale starts at 0.02. One active marker embedding is replaced by the sixteen continuous vectors, and the sequence mask is updated. Inserted positions do not receive targetformula token labels. The LLM is frozen during encoder training, and both networks operate in evaluation mode during equation search.

## A.3 AUXILIARY LEARNED STRUCTURAL-EDIT GENERATOR

Figure 5 shows the structural editor’s numerical and program encoders. The editor takes the training observations and the current formula. Its own residual encoder summarizes coordinates, targets, predictions, and residuals into sixteen latent vectors, projected to the editor’s hidden width. A program encoder represents the formula’s abstract syntax tree, including node types, variables, constants, parent relations, and depths. Cross-attention combines the numerical and program representations to predict an edit operation, target node, function family, variable support, and continuous parameters. The available operations are addition, replacement, deletion, and stopping. The input program is limited to 96 nodes and numerical encoding uses at most 512 observations.

Predicted edits are converted into executable formulas and evaluated with lightweight training-data fitting. Their resulting residuals inform learned outcome scores used for reranking. A selected edit is then submitted to the common outer evaluator, which fits its coefficients and scores its training MSE. The editor’s predicted extrapolation-related score is learned from synthetic data. It does not evaluate the benchmark OOD split during search. All editor weights are frozen at deployment. Its pretraining and edit supervision are described in Appendix C.2.

![](images/326972bbe1a3b8f39b5877bbd1c57c1b17b0be848cf29f0a28ce9ce3aaac73f6.jpg)  
Figure 4: Residual encoder architecture.

![](images/dab601298a66d96ad0227bfbeff52c2f8972ebd4e6a7b8d8c825ebc2d01d27d5.jpg)  
Figure 5: Numerical and program encoders within the structural edit generator.

## A.4 STAGE-A LLM CALLING

LLM decoding uses temperature 0.8, top-k = 30, top-p = 0.9, repetition penalty 1.0, and at most 256 new tokens per completion. The buffer cluster temperature is 0.1. Outer coefficient fitting uses BFGS with one configured restart and a 120-second evaluation timeout. Among executable logged

candidates that reproduce their recorded training fit, the equation with the lowest training MSE is selected, retaining the earlier candidate on ties. This selected equation is the Stage-A output $p _ { 0 }$

## B DETAILS OF RESIDUAL-GUIDED REFINEMENT

## B.1 DUAL-VIEW CONSTRUCTION AND RELATIONAL STATISTICS

Figure 6 details the dual-view refinement encoder and its relation-attention blocks. The encoder maps additive and multiplicative residual statistics to variable representations used by the correction utility scorer.

For predictions $p _ { i }$ and residuals $\begin{array} { r } { r _ { i } = y _ { i } - p _ { i } } \end{array}$ , the regularized multiplicative coordinate solves

$$
u _ { i } = \arg \operatorname* { m i n } _ { u \in \mathbb { R } } \{ ( r _ { i } - p _ { i } u ) ^ { 2 } + \lambda ^ { 2 } u ^ { 2 } \} = \frac { p _ { i } r _ { i } } { p _ { i } ^ { 2 } + \lambda ^ { 2 } } , \qquad \lambda = 0 . 0 5 \operatorname* { m a x } \{ \mathrm { R M S } ( p ) , 1 0 ^ { - 8 } \} .\tag{8}
$$

Prediction RMS is computed on the encoding rows. The positive floor makes $u _ { i }$ defined even when all predictions vanish. The additive view uses $r _ { i } / \sigma _ { y } ;$ the multiplicative view uses $u _ { i } / \operatorname* { m a x } \{ \mathrm { s t d } ( u ) , 1 0 ^ { - 8 } \}$ , without centering the signal. Both signals are clipped to [−20, 20].

Coordinates use median/IQR-or-standard-deviation normalization and clipping as in the residual encoder, but are padded to eight slots. Each view also contains target and prediction channels normalized by the target mean and standard deviation, its correction signal, the signal magnitude, and its signed logarithm. Squaring and cubing the signal yields seven observation features. For a normalized coordinate $z ,$ the fourteen coordinate transforms are

$$
\begin{array} { c } { { \Psi ( z ) = \left[ z , z ^ { 2 } , z ^ { 3 } , | z | , \log ( 1 + z ^ { 2 } ) , e ^ { - z ^ { 2 } } , \right. } } \\ { { \left. \left\{ \sin ( \pi f z ) , \cos ( \pi f z ) \right\} _ { f \in \{ 0 . 5 , 1 , 2 , 4 \} } \right] . } } \end{array}\tag{9}
$$

Cross-correlations of the fourteen transforms with seven signals, plus their means and log standard deviations, provide $1 4 \times 7 + 2 \times 1 4 + 2 \times 7 = 1 4 0$ features per variable. Moment standardization floors the variance at $1 0 ^ { - 8 }$

For each directed variable pair $( j , k )$ , correlations of $z _ { j } z _ { k } , z _ { j } ^ { 2 } z _ { k }$ , and $z _ { j } z _ { k } ^ { 2 }$ with the correction signal, its magnitude, and its square give nine edge features. Separately, each unordered pair has 64 operator-product correlations in each view. These use the eight correction operators evaluated on normalized coordinates, standardize each operator response, and correlate their products with the view signal. Operator-response and signal standard deviations are floored at $1 0 ^ { - 4 }$ ; product variances are floored at $\bar { 1 0 } ^ { - 8 }$ , and correlations are clipped to $[ - 1 , 1 ]$ . Exponentials in this statistical probe use arguments clipped to $[ - 5 , 5 ]$ ]. This probe clipping does not redefine the exponential operator in fitted correction formulas.

A shared $1 4 0 \to 1 2 8 \to 1 2 8$ embedding and two four-head relation-attention blocks encode each view. Linear maps of the nine edge features contribute attention biases and values. Each block has residual connections and a width-256 feed-forward hidden layer. Masked mean pooling of the two views supplies a global width-128 context. Concatenating this context with each variable’s two representations gives a width-384 vector, which a 384 →128 fusion layer maps to $\mathbf { h } _ { j }$

## B.2 ACTION VOCABULARY AND SHARED UTILITY HEAD

Each action specifies an additive or multiplicative update, one variable or two distinct variables, and one operator per variable. The operator order is

$$
\Phi = \{ z , z ^ { 2 } , z ^ { 3 } , \sin z , \operatorname { t a n h } z , \exp z , \log ( 1 + z ^ { 2 } ) , ( 1 + z ^ { 2 } ) ^ { - 1 } \} .\tag{10}
$$

There are $1 6 d + 1 2 8 \binom { d } { 2 }$ valid actions at dimension $d ,$ or 3,712 for eight variables. Product factors are stored in canonical variable order, and actions involving padded variables are masked. A shared network maps the concatenation of $\mathbf { h } _ { j }$ and a 16-dimensional operator embedding to a 64-dimensional query $\mathbf { q } _ { j , k }$ . The action representation is

$$
\begin{array} { r l } & { { \mathbf { v } } _ { a } = \big [ { \mathbf { q } } _ { L } + { \mathbf { q } } _ { R } , \ { \mathbf { q } } _ { L } \odot { \mathbf { q } } _ { R } , \ | { \mathbf { q } } _ { L } - { \mathbf { q } } _ { R } | , } \\ & { \quad \quad \quad { \mathbf { e } } _ { m } , \ I _ { \mathrm { p r o d u c t } } , \ c _ { a } ^ { ( + ) } , \ c _ { a } ^ { ( \times ) } \big ] , } \end{array}\tag{11}
$$

![](images/6298b5558048139041b622f45ca3489e7f6042ab1430de42473352dbe634a316.jpg)  
Figure 6: Dual-view refinement encoder. The two views share the embedding network and relationattention weights. The lower panel shows how pair features modify attention logits and values.

where $\mathbf { e } _ { m }$ is a 16-dimensional mode embedding and the last two scalars are the action’s product correlations. Unary actions set $\mathbf { q } _ { R } = 0 , I _ { \mathrm { p r o d u c t } } = 0$ , and both correlations to zero. A shared $2 1 1  1 2 8  1$ head predicts fitted utility. The refinement encoder, queries, and utility head together contain 389,297 trainable parameters.

## B.3 CORRECTION COEFFICIENT FITTING

For an observation subset S, the fitting loss is

$$
L _ { S } ( f ) = \frac { 1 } { | S | } \sum _ { i \in S } \left( \frac { f ( x _ { i } ) - y _ { i } } { s _ { i , S } } \right) ^ { 2 } , \qquad s _ { i , S } = \operatorname* { m a x } \{ | y _ { i } | , 0 . 0 1 \mathrm { R M S } ( y _ { S } ) , \epsilon _ { 6 4 } \} ,\tag{12}
$$

where $\epsilon _ { 6 4 }$ is double-precision machine epsilon. This weighted squared-error objective is different from the test-time accuracy-to-tolerance criterion.

At the start of Stage B, all adjustable coefficients of $\dot { p } _ { 0 }$ are fitted on the full training set with its structure fixed. The trust-region reflective least-squares solver starts from the saved coefficients, uses a two-point numerical Jacobian, at most 500 function evaluations, and $1 0 ^ { - 1 0 }$ for each termination tolerance. Parameter scales are $\operatorname* { m a x } \{ | c _ { j } | , 1 \}$ . A finite strict improvement in $L _ { S }$ is retained; otherwise the original coefficients are kept. This initial calibration is separate from the four-attempt correction budget and does not use the 0.1% minimum-improvement threshold required for accepting a new correction.

An action defines $h _ { a } \ = \ \alpha g _ { a } \ + \beta$ , where $\begin{array} { r } { g _ { a } ( x ) \ = \ \prod _ { j \in J } \phi _ { k _ { j } } ( b _ { j } z _ { j } \ + \ : c _ { j } ) } \end{array}$ and $z _ { j } ~ = ~ ( x _ { j } ~ -$ $\mu _ { j } ) / \operatorname* { m a x } \{ \sigma _ { j } , 1 0 ^ { - 8 } \}$ uses fitting-row means and standard deviations. For fixed nonlinear parameters, the two linear design columns are $( g _ { a } , 1 )$ for an additive update and $( p g _ { a } , p )$ for a multiplicative update, with response $y - p .$ Dividing the design and response by $s _ { i , S }$ gives the weighted least-squares problem for $\alpha , \beta$ . The current equation’s existing coefficients remain fixed during every new correction fit.

The nonlinear parameters use three deterministic starts. Unary starts for $( b , c )$ are $( 1 , 0 ) , ( - 1 , 0 )$ and (2, 0.5); product starts for $( b _ { 1 } , c _ { 1 } , b _ { 2 } , c _ { 2 } )$ are $( 1 , 0 , 1 , 0 ) , ( - 1 , 0 , 1 , 0 )$ , and $( 2 , 0 . 5 , 0 . 5 , - 0 . 5 )$ Slopes are bounded by $[ - 2 0 , 2 0 ]$ and shifts by $[ - 1 0 , 1 0 ]$ . Each start uses trust-region reflective least squares, a two-point Jacobian, and tolerances of $1 0 ^ { - 9 }$ . Each action has a 15-second overall deadline across its starts. The best finite fit completed within the budget is retained. An attempt with no finite candidate leaves the equation unchanged; solver convergence alone is not the acceptance criterion.

## B.4 ENCODING ROWS, ACTION SELECTION, AND STOPPING

The current refinement configuration encodes 256 training rows. When more rows are available, a deterministic sample without replacement is drawn using seed 20260831 and its indices are sorted. The same indices are used after accepted updates; their predictions, residuals, and normalization statistics are recomputed. This fixed encoding-row seed is distinct from the Stage-A search seed. Every coefficient fit and acceptance decision uses the full training set. During synthetic utility training, encoding and fitting instead use the designated 192 fitting rows.

At each attempt, the frozen model ranks dimension-valid actions and selects the highest-scoring untried action, using a stable order for ties. For the fitted equation $\widetilde { p } _ { a } .$ , acceptance requires finite training predictions and $L _ { S } ( \widetilde { p } _ { a } ) < 0 . 9 9 9 L _ { S } ( p )$ . An accepted action updates p and its residual state; a rejection retains $p .$ Every attempted action is marked as tried, and all attempts count toward the cap of four. Refinement stops before a new attempt when training NMSE is at most $1 0 ^ { - 1 2 }$ . Thus, a task may require zero new correction fits even though its initial coefficients were refitted. No Stage-B operation calls the LLM or uses test observations for acceptance.

## C SYNTHETIC DATA AND OFFLINE TRAINING

The two main training corpora supervise residual-to-formula generation and fitted correction utility. The auxiliary structural editor has additional residual-pretraining and edit-supervision datasets. All are generated from elementary functions, sampled transformations, coefficients, and coordinates. They do not use benchmark equations or observation arrays as generation inputs. Table 4 separates their sizes and learning roles.

Table 4: Synthetic training collections. Counts are examples except for the paired utility corpus, whose counts are pairs. The three utility test sets are additional to its training and validation splits.
<table><tr><td>Collection</td><td>Train</td><td>Validation</td><td>Supervision</td></tr><tr><td>Search residual-to-formula</td><td>6,000</td><td>768</td><td>Target formula tokens</td></tr><tr><td>Editor residual pretraining</td><td>50,000</td><td>5,000</td><td>Families, variables, gains</td></tr><tr><td>Structural edits</td><td>20,000</td><td>2,000</td><td>Edits and fitted outcomes</td></tr><tr><td>Paired correction utility</td><td>8,192</td><td>1,024</td><td>Eight actions per state</td></tr></table>

## C.1 SEARCH ENCODER DATA AND TRAINING

Three families contribute 2,000 training and 256 validation examples each. Compositional examples multiply two to six transformed factors with a global amplitude and at most ten target-program parameters; the current equation omits a factor or uses an affine approximation. Smooth-regime examples combine coordinate and condition responses through sigmoid or tanh gates; the current equation omits the late regime or uses an affine approximation. Factorized responses combine onset, hardening or softening, condition-dependent factors, and a smooth damage gate; current equations omit a condition factor or damage gate, or use an affine approximation.

Each example has 256 observations and one to six variables. Compositional coordinates are sampled uniformly from randomly chosen intervals among [−3, 3], [0.05, 5], [0, 1], and [0, 40]. The other families use two randomly assigned active coordinates in [0, 1], with additional coordinates in [−1, 1]. Elementary transformations include shifts, rescaling, powers, saturation, exponential decay or onset, squared sine, and hinges. Generation rejects nonfinite targets, target standard deviation at most $1 0 ^ { - 4 }$ , and absolute outputs exceeding $1 0 ^ { 5 }$ for compositional examples or $1 0 ^ { 4 }$ for the other families. Validation checks row alignment, finite inputs, and $\boldsymbol { r } = \boldsymbol { y } - \boldsymbol { p }$ . All 6,768 target programs pass parsing and compilation; this is a syntactic validation and is not a proof of identifiability from observations.

The residual encoder and projector are randomly initialized and trained through target-formula token cross-entropy, with all LLM weights frozen and no LoRA adapter. Training uses AdamW, one epoch over 6,000 examples, an effective batch size of eight, and 750 updates. Learning rates are $2 \times \mathrm { 1 0 ^ { - 5 } }$ for the encoder trunk and $1 0 ^ { - 4 }$ for the projector, with weight decay 0.01.

## C.2 STRUCTURAL-EDITOR DATA AND TRAINING

The editor’s residual encoder is separately pretrained on 50,000 training and 5,000 validation examples. A sampled sum of elementary terms forms a current equation; additional sampled terms form the clean target. The library includes constants, powers, trigonometric and saturating functions, bounded exponentials, protected inverses and ratios, and variable products. Labels identify function families, active variables, and numerical fitting gains. This encoder is frozen during subsequent edit training.

Edit training uses 20,000 training and 2,000 validation examples, each with 256 observations and one to six variables. Current programs omit a target component, contain an extra or incorrect component, or already match the target. These constructions provide addition, deletion, replacement, and stop labels. Up to four valid candidate slots provide measured fitting gains, gains on separately sampled wider-range synthetic inputs, and changes in expression complexity. Gaussian observation noise has standard deviation equal to 0.5% of the clean-target standard deviation, with a positive floor. Structural-record hashes separate training and validation.

The program encoder and editing modules learn supervised operation, location, family, and variable predictions, continuous parameters, outcome regressions, and candidate rankings. AdamW training uses ten epochs, batch size sixteen, learning rate $2 \times 1 0 ^ { - 4 }$ , and weight decay 0.01. Selection uses validation loss. The synthetic wider-range observations provide offline supervision; they are not the benchmark OOD arrays.

## C.3 PAIRED-STATE REFINEMENT DATA

Targets contain three to six unary or two-variable product components using the eight correction operators. Affine input slopes are sampled in [0.6, 2.2] and shifts in $[ - 1 , 1 ]$ . Components are combined by addition, multiplication by one plus a component, saturation through tanh, or division by one plus a component’s square. An affine output transformation varies scale from $\mathrm { i 0 ^ { - 3 } }$ to $1 0 ^ { 3 }$ Each problem has 256 points drawn from uniform, clipped-normal, or standardized-lognormal distributions, followed by coordinate rescaling and translation. Training and validation use two to six variables; the dimensional test uses eight.

Two current equations share each observation cloud and noisy target. Coarse states are formed by removing an additive component, dividing out a multiplicative component, fitting a subset of components, or combining these operations. One quarter of pairs remain coarse. For the others, each prediction is rescaled toward the clean target,

$$
p ^ { ( s ) } = f _ { \mathrm { c l e a n } } + \alpha _ { s } ( p _ { \mathrm { c o a r s e } } ^ { ( s ) } - f _ { \mathrm { c l e a n } } ) , \qquad s \in \{ 0 , 1 \} ,\tag{13}
$$

placing its clean-error contribution, measured on the first 192 rows, into one of eight logarithmic bins spanning $1 0 ^ { - 8 } \mathrm { t o } 1$ . Observation noise is unchanged. The encoder receives the numerical state, not the target expression.

Each pair shares eight dimension-valid actions. Frozen earlier residual-based policies propose top actions for both current states, and a frozen target-only policy supplies additional candidates. After deduplication, random valid actions fill the set to eight, retaining at least two random candidates.

Table 5: Paired-state utility splits. Each pair has two states and eight actions per state. Only the training split contributes labels to gradient updates.
<table><tr><td>Split</td><td>Pairs States</td><td>Utility labels</td></tr><tr><td>Training</td><td>16,384</td><td>131,072</td></tr><tr><td>Validation</td><td>1,024 2,048</td><td>16,384</td></tr><tr><td>New observation clouds</td><td>2,048</td><td>16,384</td></tr><tr><td>Held-out product actions</td><td>1,024 1,024 2,048</td><td>16,384</td></tr><tr><td>Eight-variable inputs</td><td>1,024 2,048</td><td>16,384</td></tr><tr><td>Total</td><td>12,288 24,576</td><td>196,608</td></tr></table>

Splits use separate random streams and keep both states of a pair together. In the held-out-product test, operator-index pairs with $\left( o _ { 1 } + o _ { 2 } \right)$ mod $4 = 0$ are reserved for candidate actions and excluded from candidates in the other splits. Target-expression construction can still use all operator combinations. This tests held-out correction actions rather than disjoint target-function families. Validation checks shared-action alignment, variable eligibility, acceptance decisions, and utility arithmetic. Positive, zero, and negative labels are retained.

## C.4 UTILITY TARGETS, LOSSES, AND OPTIMIZATION

The first 192 rows form F and supply encoder inputs, coefficient fits, and acceptance decisions. The remaining 64 rows form H and measure fitted utility. Some target-generation normalizations use the full synthetic cloud before this assignment; H is therefore held out from correction fitting, not from every operation used to construct the synthetic problem. Neither subset contains benchmark test observations. After fitting on $F ,$ , define

$$
p _ { a } ^ { \prime } = \binom { \widetilde { p } _ { a } , \quad L _ { F } ( \widetilde { p } _ { a } ) < 0 . 9 9 9 L _ { F } ( p ) \mathrm { a n d t h e f i t i s f n i t e } , } { p , \quad \mathrm { o t h e r w i s e } . }\tag{14}
$$

The supervision target is

$$
U ( p , a ) = \mathrm { c l i p } _ { [ - 3 , 3 ] } \left[ \log _ { 1 0 } \frac { L _ { H } ( p ) + 1 0 ^ { - 8 } } { L _ { H } ( p _ { a } ^ { \prime } ) + 1 0 ^ { - 8 } } \right] .\tag{15}
$$

The same loss in Eq. 12 is evaluated separately on F and H. Rejection gives zero utility; nonfinite utility-row predictions receive −3. An accepted action may have negative utility if it worsens prediction on H.

Let $\widehat { U } _ { s , a }$ be the predicted utility and ℓ the smooth- $L _ { 1 }$ loss with transition at absolute error one. The value term averages $\ell ( \widehat { U } _ { s , a } - U _ { s , a } )$ . For actions a, b in the same state, define $d _ { s , a b } = U _ { s , a } - U _ { s , b } $ $\widehat { d } _ { s , a b } = \widehat { U } _ { s , a } - \widehat { U } _ { s , b }$ , and $w _ { s , a b } = \operatorname* { m i n } \{ | d _ { s , a b } | , 3 \} \mathbf { 1 } [ | d _ { s , a b } | > 0 . 0 5 ]$ . The ranking loss is

$$
\mathcal { L } _ { \mathrm { r a n k } } = \frac { \sum _ { s , a , b } w _ { s , a b } \log ( 1 + \exp [ - \mathrm { s i g n } ( d _ { s , a b } ) \widehat { d } _ { s , a b } ] ) } { \operatorname* { m a x } \{ 1 , \sum _ { s , a , b } w _ { s , a b } \} } .\tag{16}
$$

For each paired current state and shared action, the paired term averages

$$
\ell \big ( ( \widehat { U } _ { 0 , a } - \widehat { U } _ { 1 , a } ) - ( U _ { 0 , a } - U _ { 1 , a } ) \big ) .\tag{17}
$$

The total is $\mathcal { L } _ { \mathrm { v a l u e } } + 0 . 5 \mathcal { L } _ { \mathrm { r a n k } } + 0 . 5 \mathcal { L } _ { \Delta }$

The refinement encoder, action-query network, and utility head are trained jointly from random initialization. AdamW uses learning rate $3 \times 1 0 ^ { - 4 }$ , weight decay 0.01, and gradient-norm clipping at one. A batch contains 64 pairs, or 128 states. Sixteen epochs over 8,192 pairs give 2,048 updates. The selected model remains frozen during refinement.

## D RESULT STATISTICS

The main table and ID/OOD plot average three complete results with seeds 0, 42, and 20260831 (Table 6). We first compute the success rate for each seed, then take the mean of the three rates. Standard deviations are sample standard deviations with two degrees of freedom. Domain results use the corresponding domain’s task count.

Table 6: Detailed results for the three seeds.
<table><tr><td>Result set</td><td>ID, 1%</td><td>ID, 0.1%</td><td>0OD, 1%</td><td>0OD, 0.1%</td></tr><tr><td>Seed 0</td><td>65.89 (85)</td><td>42.64 (55)</td><td>57.36 (74)</td><td>41.09 (53)</td></tr><tr><td>Seed 42</td><td>61.24 (79)</td><td>35.66 (46)</td><td>55.81 (72)</td><td>34.88 (45)</td></tr><tr><td>20260831</td><td>63.57 (82)</td><td>37.21 (48)</td><td>55.04 (71)</td><td>38.76 (50)</td></tr><tr><td>Mean ± SD</td><td>63.57 ± 2.33 38.50 ± 3.66</td><td></td><td>56.07 ± 1.18</td><td>38.24 ± 3.13</td></tr></table>

## E SYMBOLIC ACCURACY: PROTOCOL AND DETAILED RESULTS

SA uses the constant-assignment criterion of SR-Scientist (Xia et al., 2026): whether some assignment of the candidate’s adjustable constants makes it represent the target relationship. The original program parameters and the parameters of accepted corrections remain adjustable for this assessment. Coordinate normalization, operator choices, powers, clipping guards, and fixed literals remain unchanged.

SA uses all 129 final programs and includes every accepted correction. With their saved coefficients, the exported programs reproduce stored predictions in all 387 training/ID/OOD replay checks.

GPT-6-Astra evaluates each candidate–target pair in ten separate requests. The recorded requests specify high reasoning effort and a limit of 8,192 completion tokens, without an explicit temperature or sampling seed. The prompt asks about the existence of suitable constant assignments and requests a structured Yes/No answer. Unanimous Yes and No outcomes are accepted; non-unanimous outcomes are reserved for human review. The completed assessment has 1,290 valid judgments: 17 tasks receive ten Yes votes and 112 receive ten No votes, with no unresolved cases (Table 7).

Table 7: Symbolic accuracy for the seed 20260831. All positive tasks receive ten Yes votes; all remaining tasks receive ten No votes.
<table><tr><td>Domain</td><td>Positive</td><td>Tasks</td><td>SA (%)</td></tr><tr><td>Materials</td><td>1</td><td>25</td><td>4.00</td></tr><tr><td>Chemistry</td><td>5</td><td>36</td><td>13.89</td></tr><tr><td>Biology</td><td>3</td><td>24</td><td>12.50</td></tr><tr><td>Physics</td><td>8</td><td>44</td><td>18.18</td></tr><tr><td>Overall</td><td>17</td><td>129</td><td>13.18</td></tr></table>

## F MORE CASE STUDIES

The examples below report saved formulas, selected actions, acceptance decisions, and replayed errors. The residual encoder supplies continuous numerical tokens, and the Stage-B model outputs utility scores rather than textual reasoning.

## F.1 CRK22: A FOUR-STEP NUMERICAL REFINEMENT

The chemistry reference has the structure

$$
\dot { A } = - k A ^ { 2 } - k A \exp ( - k _ { s } t ) + \frac { k _ { z } A ^ { 2 } } { 1 + q A ^ { 4 } } ,\tag{18}
$$

where t is time and A is concentration.

Table 8: CRK22 action sequence. All four actions are additive and accepted by the training objective. Define $\mathrm { r q } ( z ) = ( 1 + \stackrel { \cdot } { z } ^ { 2 } ) ^ { - 1 }$ and $\log ( z ) = \log ( 1 + z ^ { 2 } )$
<table><tr><td>Attempt</td><td>Mode</td><td>Correction family</td><td>OOD NMSE</td></tr><tr><td>0</td><td></td><td>Fitted parent</td><td>2.0917</td></tr><tr><td>1</td><td>Additive</td><td> $\operatorname { r q } ( t ) \operatorname { t a n h } ( A )$ </td><td> $4 . 0 3 5 8 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>2</td><td>Additive</td><td> $\operatorname { r q } ( t ) \operatorname { l s q } ( A )$ </td><td> $2 . 9 3 9 8 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>3</td><td>Additive</td><td> $\operatorname { r q } ( t ) \sin ( A )$ </td><td> $1 . 7 1 4 3 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>4</td><td>Additive</td><td> $\sin ( t ) { \mathrm { ~ r q } } ( A )$ </td><td> $6 . 4 1 0 2 \times 1 0 ^ { - 6 }$ </td></tr></table>

ID NMSE changes from $1 . 6 7 5 7 \times 1 0 ^ { - 7 } { \mathrm { t o } } 4 . 7 0 7 2 \times 1 0 ^ { - 8 }$ . The OOD point fraction below each of the two main tolerances changes from zero to one. Each accepted correction changes the residual state used for the next ranking. The final formula is nevertheless not counted as symbolically equivalent by the ten-vote assessment.

## F.2 MATSCI9: ADDITIVE AND MULTIPLICATIVE CORRECTIONS

MatSci9 (Table 9) alternates between multiplicative and additive actions on strain ϵ and temperature T. A multiplicative action means $p  p ( 1 + h _ { a } )$ , not multiplication by the displayed operator product alone. Every listed factor again has fitted affine parameters, and all four attempts are accepted.

Table 9: MatSci9 refinement trajectory. Action names omit fitted affine parameters, amplitudes, and offsets.
<table><tr><td>Attempt</td><td>Mode</td><td>Correction family</td><td>OOD NMSE</td></tr><tr><td>0</td><td></td><td>Fitted parent</td><td> $5 . 4 7 8 \times 1 0 ^ { - 7 }$ </td></tr><tr><td>1</td><td>Multiplicative</td><td>cube(€) square(T)</td><td> $1 . 4 7 1 \times 1 0 ^ { - 8 }$ </td></tr><tr><td>2</td><td>Multiplicative</td><td> $\operatorname { t a n h } ( \epsilon ) \operatorname { l i n e a r } ( T )$ </td><td> $3 . 3 2 2 \times 1 0 ^ { - 9 }$ </td></tr><tr><td>3</td><td>Additive</td><td> $\mathrm { l s q } ( \epsilon ) \log ( T )$ </td><td> $2 . 8 7 8 \times 1 0 ^ { - 9 }$ </td></tr><tr><td>4</td><td>Multiplicative</td><td>cube(T)</td><td> $7 . 5 6 6 \times 1 0 ^ { - 9 }$ </td></tr></table>

ID NMSE decreases from $1 . 4 9 0 \times 1 0 ^ { - 9 }$ to $2 . 5 8 0 \times 1 0 ^ { - 1 2 }$ . Final OOD NMSE is lower than the parent’s, but higher than at the third prefix. The final OOD point fraction is one at both main tolerances. This nonmonotonic test error is consistent with training-only acceptance and is retained rather than replaced by the best test-scoring prefix.

## G UTILITY-RANKING EXPERIMENTS

Each synthetic example contains two current equations sharing the same observation cloud and the same eight candidate actions. Let s denote one state and s¯ its paired state. For the paired-state swap diagnostic, we use the scorer outputs produced for s¯ to select an action for s:

$$
a _ { s } ^ { \mathrm { s w a p } } = \mathop { \mathrm { a r g } \mathrm { m a x } } _ { a } \widehat { U } _ { \bar { s } , a } , \qquad U _ { s } ( a _ { s } ^ { \mathrm { s w a p } } ) \mathrm { i s ~ t h e n ~ e v a l u a t e d ~ u s i n g ~ t h e ~ u t i l i t y ~ l a b e l s ~ o f ~ } s .
$$

The same operation is performed in the opposite direction for s¯. Thus, the action pool and fitting protocol are unchanged; only the predicted scores are taken from the other current equation.

To directly test whether the learned scorer identifies promising corrections, we evaluate it on three held-out synthetic splits with a fixed pool of eight dimension-valid actions per state. Each split contains 1,024 paired problems, or 2,048 current-equation states. Every state is associated with a fixed pool of eight dimension-valid correction actions.

The refinement model is frozen during this evaluation. For every state and candidate action, the correction is fitted on the designated fitting rows using the same variable-projection procedure as in the refinement stage. The fitted equation is accepted only when it reduces the fitting loss by more than 0.1%. Its utility is then measured on the disjoint utility rows using the clipped logarithmic error reduction defined in Eq. 5. The reported score is the mean utility of the action selected by each selector.

The results are shown in Table 10. The learned scorer achieves mean utilities of 0.4627, 0.4099, and 0.4532 on new observation clouds, unseen product actions, and eight-variable inputs, respectively. The learned scorer outperforms residual-correlation ranking and uniform random selection on all three held-out splits. The paired-state swap is substantially worse, which is consistent with the scorer using information about the current equation rather than only the shared action pool. This result provides direct evidence that the learned utility scorer can identify more promising corrections than simple selection rules in controlled candidate pools.

## H COMPUTATIONAL ACCOUNTING

Search budget. Stage A permits at most 64 outer candidate submissions, including the initial seed equation. Invalid submissions also consume an outer slot. After the outer search terminates, executable candidates that reproduce their recorded training fit are compared using the training objective, and the candidate with the lowest training MSE is selected as $p _ { 0 }$ . Stage B first fits the coefficients of $p _ { 0 }$ under the relative-error objective. This calibration is separate from the subsequent budget of at most four new correction-fitting attempts.

Table 10: Held-out utility of the selected action in fixed eight-action candidate pools. Each column contains 2,048 states. Utilities are clipped base-10 log error reductions measured after coefficient fitting. Higher values are better.
<table><tr><td>Selector</td><td>New observation clouds</td><td>Unseen product actions</td><td>Eight-variable inputs</td></tr><tr><td>Learned utility scorer</td><td>0.46273</td><td>0.40994</td><td>0.45324</td></tr><tr><td>Residual-correlation ranking</td><td>0.39336</td><td>0.37877</td><td>0.39710</td></tr><tr><td>Uniform random selection</td><td>0.24101</td><td>0.22907</td><td>0.19557</td></tr><tr><td>Paired-state swap</td><td>0.27099</td><td>0.25677</td><td>0.22226</td></tr></table>

Measured cost. The Stage-A configuration generates four continuations per LLM request. A complete 64-slot task search therefore uses at most 64 LLM generation requests.

For the seed 20260831, in Stage B, the refinement logs contain 424 new correction attempts, of which 400 are accepted. The projected fitting objective is evaluated 461,860 times, and the recorded correction-fit durations sum to 433.98 seconds.