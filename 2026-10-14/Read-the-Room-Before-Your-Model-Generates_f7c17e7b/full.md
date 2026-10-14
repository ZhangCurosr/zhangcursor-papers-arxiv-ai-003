# Read the Room Before Your Model Generates

Muyao Kong<sup>∗</sup> <sup>†</sup> New York University muyao.kong@nyu.edu

Zhihao Lin<sup>∗</sup> Independent Researcher zhihaolinsde@gmail.com

## Abstract

Large language models (LLMs) generally rely on sampling temperature to regulate the amount of randomness in their outputs. Mainstream methods for selecting temperature typically involve fixed, heuristic global settings and context or token-level calibration, rather than a geometrically interpretable representation of the communicative scenario. In this work, we introduce a simple and interpretable temperature selection framework, the Scenario-Actor, which adapts decoding to the inferred situa tional characteristics. Scenario-Actor extracts scenario-relevant structures from Transformer hidden states and determines the temperature from scenario embedding, a structured geometric representation of the communicative setting. Empirical results show that on Llama-3.1- 8B-Instruct, Scenario-Actor achieves strong goal-completion performance in interactive social scenarios, significant gains on open-ended instruction following, and competitive performance on multi-turn conversation compared to multiple baselines.

## 1 Introduction

Large language models (LLMs) generate text by repeatedly sampling from a conditional next-token distribution. One of the simplest yet most consequential hyperparameters in language generation is the sampling temperature, which controls the sharpness of the next-token probability distribution and, in turn, the rigor and creativity of model output.

Existing systems typically set it globally or adapt it according to predictive uncertainty, context, or learned decoding objectives, which implicitly treat decoding primarily as a property of the model’s predictive state (Dhuliawala et al., 2024; Wang et al., 2020). But socially situated language generation exposes a different source of variation: the interaction itself. A prompt conveys not only semantics but also implies a communicative setting. For example, the conversation could be either an academic discussion or a friendly chitchat, whose desired responses are either rigorous or open-ended, formal or casual. The optimal temperature is then circumstantial, varying across scenarios.

![](images/35e1ea16969df08e6afe698de22d65883ff2261b96d9e73d0839c0bccd021a21.jpg)

![](images/b36ad9facf7b1e62c16f17610c4016ea7ae9bf764b23700a46addb77293ac7e0.jpg)  
Figure 1: AgentSense goal-outcome flip rate (top) and goal completion score (bottom) by changing temperature.

In our controlled experiment, we decode 245 AgentSense (Mou et al., 2025) social interactions with four fixed temperatures. We found that first, changing temperature substantially alters interaction quality, moving between mid-range temperatures flips 16 to 17% of individual goal outcomes; second, globally strongest temperature setting does not dominate across interactions.

Such results show a major limitation of viewing temperature as a globally fixed hyperparameter, which motivates us to propose a way to control scenario-aware interaction and language generation. In this work, we propose Scenario-Actor, a framework for scenario-adaptive temperature selection. Scenario-Actor focuses on social interactions and open-ended instruction following, where the prompt provides enough evidence to infer a communicative scenario. We introduce a structured representation of the communicative setting extracted from context - the scenario embedding<sup>1</sup>, whose span defines the scenario space, and use it to derive the sampling temperature. Specifically, our method maps a representation from the r-th layer of the Transformer to an m-dimensional scenario space, where each prompt is represented by a scenario embedding. An Actout function then maps the geometry of this vector to a prompt-level sampling.

Prior work in social psychology provides us with a theoretical support by introducing DIAMONDS (Rauthmann et al., 2014), a working model of situation characteristics showing that human behavior varies with situation characteristics. We construct scenario axes from nine pre-defined scenario labels. At the end, five scenario axes are constructed, corresponding to five of the eight DIAMONDS - Duty, Intellect, Adversity, Negativity, and Sociality (Rauthmann et al., 2014). Evaluations across fixed and random temperature policies, nucleus sampling (Holtzman et al., 2020), and EDT (Zhang et al., 2024) baselines indicate Scenario-Actor’s competitive performance on interactive social scenarios, open-ended instruction following, and multi-turn dialogue compared to most decoding strategies.

## 2 Background

An autoregressive language model generates a response one token at a time. In each next-token generation, a trained neural network first produces a set of scores, which we call logits, from the prompt $\mathcal { P }$ and the generated tokens over the vocabulary V that represents all possible next tokens. A Softmax function then transforms these logits to a distribution that gives how likely a token is to be the next. Let $y _ { t }$ and $y _ { < t }$ be the t-th token to be generated and already generated, and $z _ { t , v }$ be the t-th logits of the token v in the vocabulary. Concretely, we have

$$
\operatorname* { P r } ( y _ { t } = v \mid P , y _ { < t } ) = \frac { \exp ( z _ { t , v } ) } { \sum _ { u \in \mathcal { V } } \exp ( z _ { t , u } ) }
$$

Now, notice that if we divide each logit $z _ { t , v }$ with a positive scalar $T _ { \mathbf { \delta } }$ , the relative contribution of each numerator to the denominator will change with $T .$ When $T > 1$ , each $\exp ( z _ { t , v } )$ becomes less dominant over $\begin{array} { r } { \sum _ { u \in \mathcal { V } } \exp \mathopen { } \mathclose \bgroup \left( z _ { t , u } \aftergroup \egroup \right) } \end{array}$ , making the probabilities more evenly distributed. When $0 < T < 1$ each $\exp \bigl ( z _ { t , v } \bigr )$ becomes more dominant. This scalar therefore controls the sharpness of the nexttoken distribution, which is commonly referred to as the temperature. Formally, for $T > 0$

$$
\operatorname* { P r } ( y _ { t } = v \mid P , y _ { < t } ) = \frac { \exp ( \frac { z _ { t , v } } { T } ) } { \sum _ { u \in \mathcal { V } } \exp ( \frac { z _ { t , u } } { T } ) }
$$

With all these, the last thing we need to do is to select a token through a process called sampling. Recent work center around language model’s temperature and sampling strategies, which we discuss in the following paragraphs.

Fixed decoding hyperparameters. Early work and much of engineering practice predominantly relied on manually chosen global decoding strategies for task-specific models, such as beam search, top-k, and fixed sampling temperatures. Holtzman et al. (2020) showed that in open-ended text generation, likelihood-maximizing decoding methods often lead to bland, repetitive, and degenerate outputs, and proposed nucleus sampling (top-p) as a more effective alternative. Zhang et al. (2021) examined the trade-off between quality and diversity in different decoding strategies. Their findings indicate that all decoding methods perform similarly when diversity is a priority, but nucleus sampling (Holtzman et al., 2020) consistently outperforms all other evaluated algorithms when quality is viewed as more important. Wiher et al. (2022) revealed that the effectiveness of decoding strategies is highly task-dependent. Later in LLM era, Shi et al. (2024) made a more systematic horizontal comparison among decoding methods and revealed that decoding method performance strongly depends on task, model size, alignment, and quantization.

Token-level dynamic temperature. Fixed decoding hyperparameters are not always optimal at every decoding step. An early token-level architecture begins with Chang et al. (2023), who proposed to relax the fixed-temperature assumption and adjust temperature dynamically according to its relevance to the source through KL-divergence. Zhu et al. (2024) then introduced AdapT sampling for code generation. Zhang et al. (2024) further put forth EDT, which adjusts temperature directly from token-level entropy to better balance quality and diversity.

![](images/b9c8c30b2a5cbe7d94b3725d870c7d58de806cfb8642a6408446f2782c2319c6.jpg)  
Figure 2: The Scenarize module.

Context and example-level temperature. Another line of work concerns selecting sampling temperature from context. Wang et al. (2020) proposed contextual temperature, which learns a contextdependent temperature vector over the output vocabulary. Li et al. (2024) proposed DDS for opendomain dialogue system. DDS copes with two situations - chit-chat and factual QA - and maps context-dependent diversity score to a single decoding temperature. Dhuliawala et al. (2024) introduced adaptive decoding, which allows temperature to be predicted either once per example or dynamically at each decoding step.

## 3 The Scenario-Actor

Most modern LLMs are built using the celebrated Transformer architecture. In our framework, we assume that our prompt context is represented by a Transformer model and take the r-th layer hiddenstate matrix as input. The Scenario-Actor has two fundamental parts: Scenarize and Actout.<sup>2</sup> Scenarize takes the hidden-state matrix at the rth Transformer layer as its input and outputs the scenario embedding; Actout takes a scenario embedding and maps it to a prompt-level sampling temperature T.

![](images/1e34b38e201c8ad13e63489c0f1ca3849a6c0f50d272e9b55c93d1f4adc8d41f.jpg)  
Figure 3: The Scenario-Actor - framework overview.

In the remainder of this section, we discuss these two modules in depth.

## 3.1 Scenarize

Before introducing our first module, let’s establish some basic mathematical setup. For a prompt $\mathcal { P } ~ = ~ \{ p _ { 1 } , \ldots , p _ { l } \}$ with l tokens, let $H _ { \mathcal { P } } ^ { ( r ) } ~ =$ $[ h _ { p _ { 1 } } ^ { ( r ) } , \ldots , h _ { p _ { l } } ^ { ( r ) } ] ^ { T } \in \mathbb { R } ^ { l \times d }$ denote the hidden-state matrix at the r-th Transformer layer, and $s _ { \mathcal { P } } ~ \in$ ${ \mathcal { S } } \subseteq \mathbb { R } ^ { m }$ denote the learned m-dimensional scenario embedding of prompt $\mathcal { P } .$ The set $S : = { }$ $\left\{ s _ { \mathcal { P } } ~ \middle | ~ s _ { \mathcal { P } } = \mathrm { S c e n a r i z e } ( H _ { \mathcal { P } } ^ { ( r ) } ) \in \mathbb { R } ^ { m } \right\}$ be the set of all learned scenario embeddings. We define the scenario space as

$$
S : = \operatorname { s p a n } ( S ) \subseteq \mathbb { R } ^ { m } .
$$

Given that the language model interaction is textonly, the underlying prompt scenario is often underdetermined with shorter linguistic sequences. As more tokens are observed, the scenario is generally expected to be more identifiable since longer prompts typically provide additional disambiguating context. As such, we model Scenarize $( H _ { \mathcal { P } } ^ { ( r ) } )$ as a recurrent neural network (RNN) that updates the inferred scenario token by token. The final state at $p _ { l }$ therefore represents the most informed scenario embedding which we will consider. Overall, we compute scenario embedding as

$$
\begin{array} { r l } & { s _ { t } = \mathrm { S c e n a r i z e } ( h _ { p _ { t } } ^ { ( r ) } ) } \\ & { ~ = \phi ( U _ { \mathrm { s t m } } s _ { t - 1 } + W _ { \mathrm { s m m } } h _ { p _ { t } } ^ { ( r ) } + b ) } \end{array}\tag{1}
$$

Here, $\phi ( \cdot )$ is the activation function; $U _ { \mathrm { s t m } } \in$ R<sup>m×m</sup> is the scenario transition matrix (STM); $W _ { \mathrm { s m m } } \in \mathbb { R } ^ { m \times d }$ is the scenario mapping matrix (SMM); and $b \in \mathbb { R } ^ { m }$ is the bias term. Each scenario vector is computed as the learned affine combination of the previous token estimate and the current inference from context $h _ { p _ { t } } ^ { ( r ) } \in H _ { \mathcal { P } } ^ { ( r ) }$ , as depicted in Figure 2.

Both STM and SMM are expected to be trained on prompt triplets $( \mathcal { P } , \mathcal { P } ^ { + } , \mathcal { P } ^ { - } )$ , where $\mathcal { P }$ denotes the prompt with base scenario, ${ \mathcal { P } } ^ { + }$ the prompt with similar scenario, and ${ \mathcal { P } } ^ { \cdot }$ <sup>−</sup> the prompt with opposite scenario. Given such a triplet, we define the loss function as

$$
\mathcal { L } = \operatorname* { m a x } ( 0 , d ( s _ { \mathcal { P } } , s _ { \mathcal { P } ^ { + } } ) - d ( s _ { \mathcal { P } } , s _ { \mathcal { P } ^ { - } } ) + \beta )\tag{2}
$$

which compares the distance from the base prompt scenario vector to the similar and opposite scenario vector with a margin $\beta .$ In this work, we employ the squared Euclidean distance $d ( s p , s p + ) = \| s p - { \relax }$ $s _ { \mathcal { P } ^ { + } } \Vert ^ { 2 }$ and $d ( s _ { \mathcal { P } } , s _ { \mathcal { P } ^ { - } } ) = \| s _ { \mathcal { P } } - s _ { \mathcal { P } ^ { - } } \| ^ { 2 }$

## 3.2 Actout

Actout maps a scenario embedding to a temperature. Drawing on SemAxis (An et al., 2018), we define scenario axes from two non-zero opposite scenario vectors $s ^ { + }$ and $s ^ { - }$ . Each scenario axis $u _ { i }$ can be defined as $u _ { i } = s _ { i } ^ { + } - s _ { i } ^ { - } \in \mathbb { R } ^ { m }$ . We name the scenario vectors on all scenario axes as subscenario vectors, each representing an elementary component of a more complex scenario. For all scenario axes, two typical temperatures $\tau _ { i } ^ { + }$ and $\bar { \tau _ { i } } ^ { - }$ on both the positive and negative directions are applied. In general, $u _ { 1 } , \ldots , u _ { k }$ have one-to-one correspondence with $( \tau _ { 1 } ^ { + } , \tau _ { 1 } ^ { - } ) , \dots , ( \tau _ { k } ^ { + } , \tau _ { k } ^ { - } ) \in$ $\mathbb { R } _ { \geq 0 } \times \mathbb { R } _ { \geq 0 }$

Let $\rho _ { i } = \hat { s } _ { \mathcal P } ^ { \top } \hat { u } _ { i }$ be the signed cosine similarity between the normalized scenario embedding and the axis. For $\textstyle \sum _ { i = 1 } ^ { k } | \rho _ { i } | > 0$ , the prompt-level temperature T is then defined as

$$
T = \frac { \sum _ { i = 1 } ^ { k } \left| \rho _ { i } \right| \left( \frac { 1 + \rho _ { i } } { 2 } \tau _ { i } ^ { + } + \frac { 1 - \rho _ { i } } { 2 } \tau _ { i } ^ { - } \right) } { \sum _ { i = 1 } ^ { k } \left| \rho _ { i } \right| }\tag{3}
$$

Temperature is defined as a normalized weighted average over $k$ scenario axes. For each axis $\hat { u } _ { i } .$ , the cosine similarity $\rho _ { i }$ specifies where the scenario embedding lies along that axis. Endpoint temperatures $\tau _ { i } ^ { + }$ and $\bar { \tau _ { i } } ^ { - }$ <sup>−</sup> are linearly interpolated based on $\rho _ { i }$ , so that the estimate moves towards $\tau _ { i } ^ { + } \mathrm { i f } \rho _ { i }$ is positive and $\tau _ { i } ^ { - }$ if negative. The denominator normalizes the weighted sum, eventually yielding a temperature inferred from the prompt’s position in scenario space.

## 4 Experiment

We evaluate our proposed Scenario-Actor framework on benchmarks covering interactive social scenarios, open-ended instruction following and multi-turn question answering to test its performance on context-rich and socially-situated interactions. We also use GSM8K (Cobbe et al., 2021) as a task-only stress test to study the boundary of scenario-based decoding.

## 4.1 Experimental Setup

## 4.1.1 Backbone model

Despite the outstanding performance of mainstream LLMs such as ChatGPT and Gemini, we do not have access to the hidden states with API calls. Therefore, we use the open-weight Llama-3.1-8B-Instruct (Grattafiori et al., 2024) as our backbone model. The model is used for both hidden-state extraction and generation.

## 4.1.2 Data

We collect 37,357 conversational prompts from human-human and human-LLM interaction corpora, including WildChat (Zhao et al., 2024) and

DailyDialog (Li et al., 2017), together with targeted supplementary datasets for under-represented scenario types. Each prompt is annotated by GPT-4o (OpenAI, 2024) with one of nine scenario labels. A confidence-based filtering is applied to construct a high-confidence supervision set, resulting in 34,427 training prompts for triplet construction and 2,930 validation prompts.

## 4.1.3 Baselines

We compare Scenario-Actor against fixedtemperature decoding at $T \in \{ 0 . 5 , 0 . 7 , 1 . 0 , 1 . 5 \}$ greedy decoding, random temperature selection, nucleus sampling (Holtzman et al., 2020) with $\textit { p } = \ 0 . 9 5$ and $T ~ = ~ 1 . 0$ , and entropy-driven temperature decoding (EDT) (Zhang et al., 2024). EDT is included as a training-free and directly reproducible adaptive baseline. DDS and LPO (Li et al., 2024; Dhuliawala et al., 2024) both, in contrast, require training additional components. LPO trains its decoder module via preference optimization over sampled outputs, and DDS trains a diversity regressor designed for a two-way chit-chat vs. QA distinction, with, to our knowledge, no public implementation. Reproducing either method involves substantial heuristic design choices of our own, making the comparison hard to attribute in either direction.

## 4.1.4 Implementation details

Hidden states Llama-3.1-8B has 32 Transformer layers. For all experiments, we use the hidden states at the 16th Transformer layer. We found that scenario geometry collapses with deeper Transformer layer, which we discuss in Section 4.4.

Scenarize Scenarize maps the above hidden states into a 32-dimensional scenario embedding. The implementation follows Equation 1, where r = 16 is the Transformer layer, $\phi ( \cdot ) = \operatorname { t a n h } ( \cdot )$ is the activation function. Scenarize is trained according to Equation 2 with squared Euclidean triplet loss with margin $\beta = 0 . 3 .$ AdamW with learning rate 3.0e-4, weight decay 1.0e-4, gradient clipping at 1.0, and early stopping based on validation recall@1 is used as optimizer.

Actout We operationalize five DIAMONDS dimensions with nine scenario labels. We compute the centroid of each scenario label as a subscenario vector and use these vectors to define each scenario axes. GPT-4o (OpenAI, 2024) is used as an LLM annotator for the training set.

The nine scenario labels are rigorous, brainstorming, formal, playful, adversarial, institutional, casual\_conversation, emotional\_support\_pos, and emotional\_support\_neg.

<table><tr><td>Scenario Label</td><td> $\tau _ { i } ^ { + } / \tau _ { i } ^ { - }$ </td></tr><tr><td>rigorous</td><td>0.30</td></tr><tr><td>institutional</td><td>0.50</td></tr><tr><td>formal</td><td>0.55</td></tr><tr><td>emotional_support_neg</td><td>0.65</td></tr><tr><td>adversarial</td><td>0.75</td></tr><tr><td>emotional_support_pos</td><td>1.00</td></tr><tr><td>casual_conversation</td><td>1.10</td></tr><tr><td>brainstorming</td><td>1.40</td></tr><tr><td></td><td></td></tr><tr><td>playful</td><td>1.50</td></tr></table>

Table 2: Scenario labels and their endpoint temperatures.

Five scenario axes - institutional vs. casual\_conversation, rigorous vs. brainstorming, adversarial vs. non-adversarial; emotional\_support\_pos vs. emotional\_support\_neg, and formal vs. playful were constructed from combinations of nine subscenario vectors. Nonadversarial is the average over all subscenario vectors that are not adversarial. Detailed description of each scenario can be found in Appendix C. Such five scenario axes match five of the DIAMONDS: Duty, Intellect, Adversity, Negativity, and Sociality (Rauthmann et al., 2014).

Each of the five axes is assigned two endpoint temperatures $( \tau _ { i } ^ { + } , \tau _ { i } ^ { - } )$ , which are external hyperparameters of Actout, as shown in Table 2. Actout then maps a scenario embedding generated from Scenarize to a temperature within the range [0.30, 1.50] corresponding to Equation 3. In our experiment, we fixed $( \tau _ { i } ^ { + } , \tau _ { i } ^ { - } )$ for all benchmarks and comparisons.

## 4.2 Main Result

We evaluate Scenario-Actor across three benchmarks - AgentSense (Mou et al., 2025), AlpacaEval (Dubois et al., 2024), and MT-Bench (Zheng et al., 2023) - covering situational goal-completion, open-ended instruction following, and multi-turn interactions with respect to all baselines. Goal completion score, Length-Controlled Win Rate (LC-WR), and GPT-4 score were metrics for each of the three main evaluations. The performance is shown in Table 1. Scenario-Actor significantly outperforms most decoding methods on AgentSense and all baselines on AlpacaEval, while remaining competitive on MT-Bench in our experiment.

<table><tr><td>Baseline</td><td>AgentSense</td><td>AlpacaEval</td><td>MT-Bench</td></tr><tr><td>Metric</td><td>Goal completion</td><td>LC-WR (%)</td><td>GPT-4 score</td></tr><tr><td>Scenario-Actor</td><td>0.8015</td><td>30.03</td><td>7.91</td></tr><tr><td>Greedy</td><td>0.7582</td><td>22.87</td><td>7.79</td></tr><tr><td>Fixed  $T = 0 . 5$ </td><td>0.7540</td><td>23.63</td><td>7.79</td></tr><tr><td>Fixed  $T = 0 . 7$ </td><td>0.7801</td><td>25.68</td><td>7.83</td></tr><tr><td>Fixed  $T = 1 . 0$ </td><td>0.7692</td><td>23.42</td><td>7.61</td></tr><tr><td>Fixed  $T = 1 . 5$ </td><td>0.6989</td><td>8.60</td><td>6.76</td></tr><tr><td>EDT</td><td>0.7671</td><td>25.58</td><td>6.74</td></tr><tr><td>Nucleus  $\left( p = 0 . 9 5 \right)$ </td><td>0.7557</td><td>23.59</td><td>7.63</td></tr><tr><td> $\mathrm { R a n d o m } { \cdot } T$ </td><td>0.7699</td><td>20.44</td><td>7.71</td></tr></table>

Table 1: Benchmark results comparing Scenario-Actor with fixed-temperature and adaptive decoding baselines. Best results are highlighted.

Among available benchmarks, AgentSense directly evaluates the setting targeted by Scenario-Actor: interactive dialogue in which the model must adapt to complete different goals as the social context develops. Scenario-Actor achieves the highest goal completion score of 0.8015, outperforming greedy decoding, every fixed-temperature setting, EDT, nucleus sampling, and Random-T. Scenario-Actor attains significant improvements over all baselines on the range from 3.16 to 10.26 percentage points except for fixed $T ~ = ~ 0 . 7$ at 0.7801, for which Scenario-Actor’s 2.14 percentage points advantage is not statistically significant.

On AlpacaEval, Scenario-Actor achieves the strongest result among all tested methods with 30.03% Length-Controlled Win Rate (LC-WR), outperforming the best fixed-temperature baseline by 4.35 percentage points, EDT by 4.45 percentage points, and exceeding nucleus sampling, greedy decoding, Random-T, and class-conditioned T with statistical significance. On MT-Bench, our framework scores 7.91, which is significantly above fixed $T = 1 . 0 $ , fixed $T = 1 . 5$ , Random-T, nucleus sampling, EDT, and class-conditioned T, and is potentially competitive among greedy decoding, fixed $T = 0 . 5$ , and fixed $T = 0 . 7 .$ . Detailed statistical report can be found in Appendix B.

## 4.3 Analysis

Agreements To justify the use of LLM for scenario annotation and results judging, we compute both the LLM-human and human-human agreement on the models we used. A gold set on scenario annotation consisting of 306 prompts and a gold set on evaluation consisting of 50 AlpacaEval examples and 50 MT-Bench examples were manually annotated by us. The results show that the GPT-4o annotator agrees with each human annotator at $\kappa = 0 . 8 2$ and 0.85. The GPT-4-turbo (OpenAI, 2023) AlpacaEval judge reaches $\kappa \approx 0 . 7 2$ with each annotator; for MT-Bench scoring, it correlates with human raters at Pearson $r = 0 . 9 2$ and 0.86; for AgentSense, it achieves agreement rates of 80% and 72% with the two human judges, comparable to the human-human agreement of 76%, and with Cohen’s κ of 0.48 and 0.27, respectively. Detailed validation metrics are reported in Appendix E

Geometry To validate if Scenarize learns meaningful scenario geometry, we visualize the learned scenario space using both PCA and UMAP. PCA shows the global variance structure of 900 scenario embeddings, with 100 examples from each scenario label. The first two principal components explain 27.9% and 21.8% of the variance respectively. As shown in Figure 4, nine centroids obtained from nine scenario labels occupy different regions in scenario space. PCA view also shows the five proposed scenario axes used in our experiment - Duty, Intellect, Negativity, Sociality, and Adversity. The patterns therefore reflect that Scenarize learns a structured scenario space.

UMAP further confirms local structure of the learned space. The projection achieves a trustworthiness score of 0.96 using cosine distance. The embedding exhibits meaningful relationships among the nine scenario types. Semantically related scenarios also cluster near one another; for example, institutional prompts lie closer to rigorous prompts than to casual conversation.

![](images/859552570c63347d1320cffb4c394b7d6c23682221f2553b9f0731e03a7ae32a.jpg)

![](images/316c66fad67c93f75966a58b248c11ff7d37be8f4b9872add3956c4a48af70a9.jpg)  
Figure 4: PCA (left) and UMAP (right) of scenario embedding geometry.

GSM8K diagnosis To study the boundary of Scenario-Actor, we tested its performance on GSM8K against all other baselines. Scenario-Actor significantly underperforms greedy decoding, fixed $T = 0 . 5$ , fixed T = 0.7, EDT, and insignificantly lower than Random-T baseline by 0.53.

<table><tr><td>Baseline</td><td>GSM8K</td></tr><tr><td>Metric</td><td>Accuracy (%)</td></tr><tr><td>Scenario-Actor</td><td>65.28</td></tr><tr><td>Greedy</td><td>77.18</td></tr><tr><td>Fixed T = 0.5</td><td>69.52</td></tr><tr><td>Fixed  $T = 0 . 7$ </td><td>69.07</td></tr><tr><td>Fixed  $T = 1 . 0$ </td><td>62.40</td></tr><tr><td>Fixed  $T = 1 . 5$ </td><td>45.19</td></tr><tr><td>EDT</td><td>69.14</td></tr><tr><td>Nucleus  $\left( p = 0 . 9 5 \right)$ </td><td>61.33</td></tr><tr><td>Random-T</td><td>65.81</td></tr></table>

Table 3: GSM8K results comparing Scenario-Actor with fixed-temperature and adaptive decoding baselines. The best result is highlighted.

Our GSM8K temperature is distributed within the range of [0.8162, 0.8782]. In theory, GSM8K would exhibit better performance of the model when the sampling temperature is lower. In our experiment, Scenario-Actor selects a GSM8K temperature less than ideal. Nearest-centroid analysis shows that the distribution is concentrated in adversarial, institutional, and brainstorming scenarios, which together account for 77.9% of GSM8K prompts. Only 5.0% are mapped to the rigorous scenario.

<table><tr><td>Scenario</td><td>Count</td><td>Percentage</td></tr><tr><td>adversarial</td><td>350</td><td>26.5%</td></tr><tr><td>institutional</td><td>346</td><td>26.2%</td></tr><tr><td>brainstorming</td><td>332</td><td>25.2%</td></tr><tr><td>playful</td><td>79</td><td>6.0%</td></tr><tr><td>casual_conversation</td><td>67</td><td>5.1%</td></tr><tr><td>rigorous</td><td>66</td><td>5.0%</td></tr><tr><td>formal</td><td>40</td><td>3.0%</td></tr><tr><td>emotional_support_neg</td><td>23</td><td>1.7%</td></tr><tr><td>emotional_support_pos</td><td>16</td><td>1.2%</td></tr></table>

Table 4: GSM8K scenario count

All nine centroids look nearly identical to GSM8K prompts in cosine space:

<table><tr><td>Scenario Centroid</td><td>Cosine Similarity</td></tr><tr><td>adversarial</td><td>0.9765</td></tr><tr><td>institutional</td><td>0.9710</td></tr><tr><td>rigorous</td><td>0.9684</td></tr><tr><td>brainstorming</td><td>0.9510</td></tr><tr><td>playful casual_conversation</td><td>0.9499 0.9463</td></tr><tr><td></td><td></td></tr><tr><td>emotional_support_neg</td><td>0.9191</td></tr><tr><td>formal</td><td>0.9098</td></tr><tr><td>emotional_support_pos</td><td>0.9043</td></tr></table>

Table 5: Mean cosine similarity between GSM8K and nine centroids, with gap of 0.0247 between the nearest and the second-nearest centroid per prompt

The results reflect near-equidistance of GSM8K to all scenario centroids: GSM8K prompts occupy a high-similarity region of the learned space. The current geometry does not reliably map task-only mathematical prompts to a low-temperature rigorous regime, and Scenario-Actor fails to infer the “scenario” of GSM8K prompts. The result illustrates the scope and boundary of Scenario-Actor to be a decoding framework for prompts whose communicative scenario can be inferred from context.

Temperature We examine the temperatures selected by Scenario-Actor across all benchmarks. Scenario-Actor operates within the range of [0.67, 1.16] across turns generated by the evaluated Llama-3.1-8B-Instruct, [0.77, 1.01] on AlpacaEval, and [0.76, 1.02] on MT-Bench. AgentSense temperatures also change with dialogue context, with the mean temperature increasing by approximately 0.16 across turn positions.

Context length We compare outputs generated by Scenario-Actor under short context with only one prior turn versus long context with all prior turns on 100 DailyDialog multi-turn conversations. Long context Scenario-Actor achieves a win rate of 67.0% over short context with a 95% bootstrap confidence interval of [58.0%, 76.0%] in pairwise GPT-4-turbo evaluation. The finding supports our claim that scenario is generally expected to be more identifiable with more context, which motivates the RNN formulation of Scenarize.

Efficiency We measure the latency, memory usage, and throughput of Scenario-Actor over 80 MT-Bench prompts on a single NVIDIA A10G 24 GB GPU (ml.g5.2xlarge). Latency includes hiddenstate extraction, Scenarize, Actout, and response generation. As shown in Appendix A, Scenario-Actor has similar end-to-end latency and throughput to the tested decoding baselines. Latency is within 2.5% of greedy decoding, while the output and request throughput remain close to standard sampling methods. Peak allocated memory is also nearly unchanged.

## 4.4 Ablation

Geometry contributes more than endpoint temperature lookup. To test whether the gains come from the scenario geometry or from the manuallyspecified endpoint temperatures, we add a classconditioned temperature baseline which uses the same Scenarize module but removes Actout’s geo-

metric weighting. Class-conditioned baseline treats the nearest centroid as the prompt’s class label and assigns its sampling temperature from corresponding endpoint.
<table><tr><td colspan="2">Baseline</td></tr><tr><td>Benchmark Scenario-Actor Class-conditioned T</td><td></td></tr><tr><td>AgentSense</td><td>0.8015 0.7536</td></tr><tr><td>AlpacaEval</td><td>30.03 27.10</td></tr><tr><td>MT-Bench</td><td>7.91 7.64</td></tr></table>

Table 6: Comparison with the class-conditioned temperature baseline. Metrics are goal completion score, LC-WR (%), and GPT-4 score respectively. Best results are highlighted.

Class-conditioned temperature baseline achieves a goal completion score of 0.7536 on AgentSense, LC-WR of 27.10% on AlpacaEval and a score of 7.64 on MT-Bench, which is lower than Scenario-Actor by 4.79 and 2.93 percentage points, and 0.27 points respectively. The improvements are statistically significant on all three benchmarks: AgentSense $( [ + 1 . 8 5 , + 7 . 8 8 ] , p = 1 . 8 5 \times 1 0 ^ { - 3 } )$ AlpacaEval ([+0.55, +5.32], p = 1.6 × 10<sup>−2</sup>), and MT-Bench ([+0.02, +0.52], $p \ = \ 3 . 2 1 \times 1 0 ^ { - 2 } )$ Since both methods use the same endpoint values, these results suggest the effectiveness of Actout’s geometric interpolation compared to class-level temperature lookup.

Layer 16 provides the best tradeoff for scenario probing. We observe that layer 16 produces stronger scenario axis separation and lower representation saturation than deeper layers such as layer 30. This observation aligns with recent work suggesting that structured semantic relation signals in LLMs often peak in intermediate layers rather than final generation-oriented layers (Diera and Scherp, 2026). Full layer-sweep diagnostics are reported in Appendix D.

Larger triplet margins induce geometryperformance tradeoffs. We compare the default margin $\beta = 0 . 3$ against $\beta = 1 . 0$ while holding the architecture, training data, and inference protocol fixed. As in table 7, increasing margin produces stronger centroid separation and improves GSM8K accuracy, but reduces performance on multi-turn and open-ended evaluation tasks. These results suggest that larger margins reshape the scenario geometry but does not uniformly improve generation

quality.

<table><tr><td>Metric</td><td>β = 0.3  $\beta = { \bf 1 . 0 }$ </td></tr><tr><td>Geometry</td><td></td></tr><tr><td>Mean centroid cosine</td><td>0.9496 0.7103</td></tr><tr><td>Euclidean silhouette</td><td>0.1835 0.2668</td></tr><tr><td>Nearest-centroid agreement</td><td>0.6940 0.8877</td></tr><tr><td>Downstream</td><td></td></tr><tr><td>GSM8K Accuracy (%)</td><td>65.28 66.41</td></tr><tr><td>AlpacaEval LC-WR (%)</td><td>30.03 25.54</td></tr><tr><td>MT-Bench</td><td>7.91 7.66</td></tr><tr><td>GSM8K mean T</td><td>0.822 0.738</td></tr></table>

Table 7: Effect of the triplet-loss margin on scenario geometry and downstream performance. Full table can be found in Appendix G.

## 5 Conclusion

We propose Scenario-Actor, a prompt-level temperature selection framework that maps a hiddenstate matrix to a structured scenario embedding and derives a sampling temperature from its geometry along interpretable scenario axes. Comparisons of Scenario-Actor with multiple baselines show strong goal-completion performance on AgentSense, significant performance improvements on AlpacaEval, and competitive results on MT-Bench. In addition, we see Scenario-Actor as a step toward socially adaptive language models. As LLMs become embedded in human society through multimodal agents and embodied systems, their ability to recognize the situations they are in will become increasingly important. We hope this framework allows models to better understand social context, emotional nuance, and the expectations that shape human communication in the future.

## Limitations

LLM scenario is narrower than human scenario. Language models take only textual sequence as their input. In actual human-human interaction, scenarios are dependent on cues other than pure linguistic context. Therefore, current language models are still limited in their ability to fully integrate into all aspects of human interactions. The inference of scenario solely from linguistic context is itself insufficient.

Compromise on task-only prompts. Scenario-Actor is designed for settings with rich scenario cues, such as multi-turn conversations. But in our dataset, we see a great amount of them being straightforward and utilitarian. Many prompts ask the model to directly solve problems such as writing code, answering math or providing general information. Such a characteristic makes scenario inference almost impossible as task-oriented prompts often contain little evidence for a social setting. For these prompts, a standalone selector can route the input to a standard task-specific decoding policy, which is independent of our framework. This setup is also a compromise to current LLM usage as the language model itself is limited in its interaction modality. We expect this limitation to be alleviated in multimodal or embodied agent settings, where models can access richer interactional cues and become more deeply integrated into human social environments.

Model and taxonomy scope. The current study establishes the results and claims merely on Llama-3.1-8B-Instruct with the nine-label, five-axis scenario taxonomy. In theory, however, our framework is expected to be applicable to other model families and scenario taxonomies, which, due to limited budget and computational power, require more evaluations.

Source balance. A key limitation of the current training data is that scenario labels are not fully balanced from data source. In an ideal setting, each scenario category would be represented across multiple corpora so that Scenarize would be forced to learn merely the scenario structure rather than dataset style. In practice, however, no available dataset covers all of our scenario labels with balanced coverage. We therefore rely on multiple supplementary sources. Developing a more balanced scenario dataset remains an important direction for future work.

## Acknowledgments

We thank Professor Haifeng Xu at the University of Chicago and the anonymous reviewers for their valuable comments, as well as Professor Adam Meyers at New York University for his advice on publication. This work was conducted independently by the authors and received no institutional research funding by the time of submission.

## References

Jisun An, Haewoon Kwak, and Yong-Yeol Ahn. 2018. SemAxis: A lightweight framework to characterize domain-specific word semantics beyond sentiment. In Proceedings of the 56th Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 2450–2461, Melbourne, Australia. Association for Computational Linguistics.

Chung-Ching Chang, David Reitter, Renat Aksitov, and Yun-Hsuan Sung. 2023. KL-divergence guided temperature sampling. arXiv preprint arXiv:2306.01286.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. 2021. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168.

Shehzaad Dhuliawala, Ilia Kulikov, Ping Yu, Asli Celikyilmaz, Jason Weston, Sainbayar Sukhbaatar, and Jack Lanchantin. 2024. Adaptive decoding via latent preference optimization. arXiv preprint arXiv:2411.09661.

Andor Diera and Ansgar Scherp. 2026. Do language models encode semantic relations? Probing and sparse feature analysis. In Proceedings of the Fifteenth Language Resources and Evaluation Conference, pages 2115–2126, Palma de Mallorca, Spain. ELRA Language Resource Association.

Yann Dubois, Balázs Galambosi, Percy Liang, and Tatsunori B. Hashimoto. 2024. Length-controlled AlpacaEval: A simple way to debias automatic evaluators. In Conference on Language Modeling (COLM).

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, Amy Yang, Angela Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mi tra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, and 542 others. 2024. The Llama 3 herd of models. arXiv preprint arXiv:2407.21783.

Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, and Yejin Choi. 2020. The curious case of neural text degeneration. In International Conference on Learning Representations.

Neha Nayak Kennard, Tim O’Gorman, Rajarshi Das, Akshay Sharma, Chhandak Bagchi, Matthew Clinton, Pranay Kumar Yelugam, Hamed Zamani, and Andrew McCallum. 2022. DISAPERE: A dataset for discourse structure in peer review discussions. In Proceedings ofthe 2022 Conference ofthe North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 1234–1249, Seattle, United States. Association for Computational Linguistics.

Hyunwoo Kim, Jack Hessel, Liwei Jiang, Peter West, Ximing Lu, Youngjae Yu, Pei Zhou, Ronan Le Bras,

Malihe Alikhani, Gunhee Kim, Maarten Sap, and Yejin Choi. 2023. SODA: Million-scale dialogue distillation with social commonsense contextualization. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 12930–12949, Singapore. Association for Computational Linguistics.

Bryan Klimt and Yiming Yang. 2004. The Enron corpus: A new dataset for email classification research. In Machine Learning: ECML 2004, volume 3201, pages 217–226. Springer.

Andreas Köpf, Yannic Kilcher, Dimitri von Rütte, Sotiris Anagnostidis, Zhi Rui Tam, Keith Stevens, Abdullah Barhoum, Duc Nguyen, Oliver Stanley, Richárd Nagyfi, Shahul ES, Sameer Suri, David Glushkov, Arnav Dantuluri, Andrew Maguire, Christoph Schuhmann, Huu Nguyen, and Alexander Mattick. 2023. OpenAssistant conversations – democratizing large language model alignment. In Advances in Neural Information Processing Systems, volume 36, pages 47669–47681.

Yanran Li, Hui Su, Xiaoyu Shen, Wenjie Li, Ziqiang Cao, and Shuzi Niu. 2017. DailyDialog: A manually labelled multi-turn dialogue dataset. In Proceedings ofthe Eighth International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 986–995, Taipei, Taiwan. Asian Federation of Natural Language Processing.

Yiwei Li, Fei Mi, Yitong Li, Yasheng Wang, Bin Sun, Shaoxiong Feng, and Kan Li. 2024. Dynamic stochastic decoding strategy for open-domain dialogue generation. In Findings of the Association for Computational Linguistics: ACL 2024, pages 11585–11596, Bangkok, Thailand. Association for Computational Linguistics.

Siyang Liu, Chujie Zheng, Orianna Demasi, Sahand Sabour, Yu Li, Zhou Yu, Yong Jiang, and Minlie Huang. 2021. Towards emotional support dialog systems. In Proceedings of the 59th Annual Meeting ofthe Associationfor Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 3469–3483, Online. Association for Computational Linguistics.

Xinyi Mou, Jingcong Liang, Jiayu Lin, Xinnong Zhang, Xiawei Liu, Shiyue Yang, Rong Ye, Lei Chen, Haoyu Kuang, Xuanjing Huang, and Zhongyu Wei. 2025. AgentSense: Benchmarking social intelligence of language agents through interactive scenarios. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 4975–5001, Albuquerque, New Mexico. Association for Computational Linguistics.

OpenAI. 2023. GPT-4 technical report. arXiv preprint arXiv:2303.08774.

OpenAI. 2024. GPT-4o system card. arXiv preprint arXiv:2410.21276.

Hannah Rashkin, Eric Michael Smith, Margaret Li, and Y-Lan Boureau. 2019. Towards empathetic opendomain conversation models: A new benchmark and dataset. In Proceedings of the 57th Annual Meeting ofthe Associationfor Computational Linguistics, pages 5370–5381, Florence, Italy. Association for Computational Linguistics.

John F. Rauthmann, David Gallardo-Pujol, Esther M. Guillaume, Elysia Todd, Christopher S. Nave, Ryne A. Sherman, Matthias Ziegler, Ashley Bell Jones, and David C. Funder. 2014. The situational eight DIAMONDS: A taxonomy of major dimensions of situation characteristics. Journal of Personality and Social Psychology, 107(4):677–718.

Chufan Shi, Haoran Yang, Deng Cai, Zhisong Zhang, Yifan Wang, Yujiu Yang, and Wai Lam. 2024. A thorough examination of decoding methods in the era of LLMs. In Proceedings ofthe 2024 Conference on Empirical Methods in Natural Language Processing, pages 8601–8629, Miami, Florida, USA. Association for Computational Linguistics.

Chenhao Tan, Vlad Niculae, Cristian Danescu-Niculescu-Mizil, and Lillian Lee. 2016. Winning arguments: Interaction dynamics and persuasion strategies in good-faith online discussions. In Proceedings ofthe 25th International Conference on World Wide Web, pages 613–624, Montréal, Québec, Canada. International World Wide Web Conferences Steering Committee.

Pei-Hsin Wang, Sheng-Iou Hsieh, Shih-Chieh Chang, Yu-Ting Chen, Jia-Yu Pan, Wei Wei, and Da-Chang Juan. 2020. Contextual temperature for language modeling. arXiv preprint arXiv:2012.13575.

Gian Wiher, Clara Meister, and Ryan Cotterell. 2022. On decoding strategies for neural text generators. Transactions of the Association for Computational Linguistics, 10:997–1012.

Daoze Zhang, Zhijian Bao, Sihang Du, Zhiyi Zhao, Kuangling Zhang, Dezheng Bao, and Yang Yang. 2025. Re2: A consistency-ensured dataset for fullstage peer review and multi-turn rebuttal discussions. arXiv preprint arXiv:2505.07920.

Hugh Zhang, Daniel Duckworth, Daphne Ippolito, and Arvind Neelakantan. 2021. Trading off diversity and quality in natural language generation. In Proceedings ofthe Workshop on Human Evaluation ofNLP Systems (HumEval), pages 25–33, Online. Association for Computational Linguistics.

Shimao Zhang, Yu Bao, and Shujian Huang. 2024. EDT: Improving large language models’ generation by entropy-based dynamic temperature sampling. arXiv preprint arXiv:2403.14541.

Wenting Zhao, Xiang Ren, Jack Hessel, Claire Cardie, Yejin Choi, and Yuntian Deng. 2024. WildChat: 1M

ChatGPT interaction logs in the wild. In The Twelfth International Conference on Learning Representations.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. 2023. Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. In Advances in Neural Information Processing Systems, volume 36, pages 46595–46623.

Yuqi Zhu, Jia Li, Ge Li, YunFei Zhao, Jia Li, Zhi Jin, and Hong Mei. 2024. Hot or cold? Adaptive temperature sampling for code generation with large language models. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 38, pages 437–445.

## A Efficiency

<table><tr><td rowspan="2">Method</td><td rowspan="2">Latency (ms/episode)</td><td colspan="2">Peak memory (GiB)</td><td colspan="2">Throughput</td></tr><tr><td>Allocated</td><td>Reserved</td><td>Output (tokens/s)</td><td>Request (episodes/s)</td></tr><tr><td>Scenario-Actor</td><td>97,000</td><td>18.59</td><td>18.71</td><td>21.59</td><td>0.01031</td></tr><tr><td>Greedy</td><td>87,410</td><td>18.36</td><td>18.48</td><td>21.84</td><td>0.01144</td></tr><tr><td>Fixed  $T = 0 . 5$ </td><td>88,110</td><td>18.38</td><td>18.50</td><td>21.57</td><td>0.01135</td></tr><tr><td>Nucleus  $( p = 0 . 9 5 )$ </td><td>88,370</td><td>18.38</td><td>18.50</td><td>21.49</td><td>0.01132</td></tr><tr><td>Random-T</td><td>87,340</td><td>18.38</td><td>18.50</td><td>21.59</td><td>0.01145</td></tr><tr><td>EDT</td><td>87,650</td><td>18.38</td><td>18.55</td><td>21.64</td><td>0.01141</td></tr></table>

Table 8: End-to-end efficiency over 245 AgentSense templates in 15-turn closed-loop interactions.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Latency (ms/prompt)</td><td colspan="2">Peak memory (GiB)</td><td colspan="2">Throughput</td></tr><tr><td>Allocated</td><td>Reserved</td><td>Output (tokens/s)</td><td>Request (prompts/s)</td></tr><tr><td>Scenario-Actor</td><td>17,109</td><td>16.31</td><td>16.68</td><td>25.35</td><td>0.0584</td></tr><tr><td>Greedy</td><td>17,181</td><td>16.52</td><td>16.86</td><td>25.87</td><td>0.0582</td></tr><tr><td>Fixed  $T = 0 . 5$ </td><td>17,150</td><td>16.52</td><td>16.86</td><td>25.43</td><td>0.0583</td></tr><tr><td>Nucleus  $( p = 0 . 9 5 )$ </td><td>17,406</td><td>16.52</td><td>16.86</td><td>25.42</td><td>0.0575</td></tr><tr><td>Random-T</td><td>17,284</td><td>16.52</td><td>16.86</td><td>25.42</td><td>0.0579</td></tr><tr><td>EDT</td><td>17,336</td><td>16.52</td><td>16.86</td><td>25.59</td><td>0.0577</td></tr></table>

Table 9: End-to-end efficiency over 805 AlpacaEval prompts.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Latency (ms/prompt)</td><td colspan="2">Peak memory (GiB)</td><td colspan="2">Throughput</td></tr><tr><td>Allocated</td><td>Reserved</td><td>Output (tokens/s)</td><td>Request (prompts/s)</td></tr><tr><td>Scenario-Actor</td><td>16,917</td><td>15.22</td><td>15.54</td><td>25.26</td><td>0.0591</td></tr><tr><td>Greedy</td><td>16,513</td><td>15.19</td><td>15.26</td><td>25.75</td><td>0.0606</td></tr><tr><td>Fixed  $T = 0 . 5$ </td><td>17,128</td><td>15.19</td><td>15.54</td><td>25.24</td><td>0.0584</td></tr><tr><td>Nucleus (p = 0.95)</td><td>16,845</td><td>15.19</td><td>15.26</td><td>25.29</td><td>0.0594</td></tr><tr><td>Random-T</td><td>16,267</td><td>15.19</td><td>15.26</td><td>25.28</td><td>0.0615</td></tr><tr><td>EDT</td><td>16,957</td><td>15.24</td><td>21.99</td><td>25.41</td><td>0.0590</td></tr></table>

Table 10: End-to-end efficiency over 80 MT-Bench prompts.

B Statistical Significance Testing
<table><tr><td rowspan="2">Benchmark</td><td rowspan="2">Comparison</td><td colspan="3">Statistical significance</td></tr><tr><td>Difference</td><td>95% CI</td><td>p-value</td></tr><tr><td rowspan="9">AgentSense (∆ Goal completion)</td><td>Scenario-Actor vs Greedy</td><td>+4.33 pp.</td><td> $[ + 1 . 5 5 , + 7 . 1 5 ]$ </td><td> $2 . 4 4 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>Scenario-Actor vs Fixed T = 0.5</td><td>+4.75 pp.</td><td>[+1.82, +7.76]</td><td> $1 . 7 2 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>Scenario-Actor vs Fixed T = 0.7</td><td>+2.14 pp.</td><td>[−0.54, +4.89]</td><td> $1 . 2 2 \times 1 0 ^ { - 1 }$ </td></tr><tr><td>Scenario-Actor vs Fixed T = 1.0</td><td>+3.23 pp.</td><td>[+0.41,+6.11]</td><td> $2 . 6 3 \times 1 0 ^ { - 2 }$ </td></tr><tr><td>Scenario-Actor vs Fixed T = 1.5</td><td>+10.26 pp.</td><td>[+6.97, +13.53]</td><td> $8 . 7 3 \times 1 0 ^ { - 1 0 }$ </td></tr><tr><td>Scenario-Actor vs EDT</td><td>+3.44 pp.</td><td>[+0.50, +6.47]</td><td> $2 . 3 9 \times 1 0 ^ { - 2 }$ </td></tr><tr><td>Scenario-Actor vs Nucleus</td><td>+4.58 pp.</td><td>[+1.41, +7.70]</td><td> $4 . 3 1 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>Scenario-Actor vs Random-T</td><td>+3.16 pp.</td><td> $[ + 0 . 4 0 , + 5 . 9 7 ]$ </td><td> $2 . 6 2 \times 1 0 ^ { - 2 }$ </td></tr><tr><td>Scenario-Actor vs Class-conditioned T</td><td>+4.79 pp.</td><td>[+1.85, +7.88]</td><td> $1 . 8 5 \times 1 0 ^ { - 3 }$ </td></tr><tr><td rowspan="9">AlpacaEval (∆ LC-WR)</td><td>Scenario-Actor vs Greedy</td><td>+7.16</td><td>[+3.14, +11.18]</td><td> $5 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Scenario-Actor vs Fixed T = 0.5</td><td>+6.40</td><td>[+2.48, +10.32]</td><td> $1 . 4 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>Scenario-Actor vs Fixed T = 0.7</td><td>+4.35</td><td> $[ + 0 . 4 3 , + 8 . 2 7 ]$ </td><td> $2 . 9 6 \times 1 0 ^ { - 2 }$ </td></tr><tr><td>Scenario-Actor vs Fixed T = 1.0</td><td>+6.61</td><td> $[ + 2 . 8 9 , + 1 0 . 3 3 ]$ </td><td> $5 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Scenario-Actor vs Fixed T = 1.5</td><td>+21.43</td><td>[+16.63,+26.23]</td><td> $2 . 2 \times 1 0 ^ { - 1 8 }$ </td></tr><tr><td>Scenario-Actor vs EDT</td><td>+4.45</td><td> $[ + 0 . 4 3 , + 8 . 4 7 ]$ </td><td> $3 \times 1 0 ^ { - 2 }$ </td></tr><tr><td>Scenario-Actor vs Nucleus</td><td>+6.44</td><td>[+2.62, +10.26]</td><td> $1 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>Scenario-Actor vs Random-T</td><td>+9.59</td><td>[+5.28, +13.90]</td><td> $1 . 3 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Scenario-Actor vs Class-conditioned T</td><td>+2.93</td><td>[+0.55, +5.32]</td><td> $1 . 6 \times 1 0 ^ { - 2 }$ </td></tr><tr><td rowspan="8">MT-Bench (∆ Score)</td><td>Scenario-Actor vs Greedy</td><td>+0.12</td><td>[−0.11,+0.35]</td><td> $3 . 0 9 \times 1 0 ^ { - 1 }$ </td></tr><tr><td>Scenario-Actor vs Fixed T = 0.5</td><td>+0.12</td><td>[−0.12, +0.36]</td><td> $3 . 2 9 \times 1 0 ^ { - 1 }$ </td></tr><tr><td>Scenario-Actor vs Fixed T = 0.7</td><td>+0.08</td><td>[−0.13, +0.29]</td><td> $4 . 6 3 \times 1 0 ^ { - 1 }$ </td></tr><tr><td>Scenario-Actor vs Fixed T = 1.0</td><td>+0.30</td><td>[+0.05, +0.55]</td><td> $1 . 9 1 \times 1 0 ^ { - 2 }$ </td></tr><tr><td>Scenario-Actor vs Fixed T = 1.5</td><td>+1.15</td><td>[+0.79, +1.51]</td><td> $3 . 3 \times 1 0 ^ { - 1 0 }$ </td></tr><tr><td>Scenario-Actor vs EDT</td><td>+1.17</td><td>[+0.80, +1.54]</td><td> $9 . 0 3 \times 1 0 ^ { - 1 0 }$ </td></tr><tr><td>Scenario-Actor vs Nucleus</td><td>+0.28</td><td>[+0.05, +0.51]</td><td> $1 . 8 6 \times 1 0 ^ { - 2 }$ </td></tr><tr><td>Scenario-Actor vs Random-T</td><td>+0.20</td><td>[+0.01, +0.39]</td><td> $4 . 1 3 \times 1 0 ^ { - 2 }$ </td></tr><tr><td rowspan="8">GSM8K (∆ Accuracy)</td><td>Scenario-Actor vs Class-conditioned T</td><td>+0.27</td><td>[+0.02, +0.52]</td><td> $3 . 2 1 \times 1 0 ^ { - 2 }$ </td></tr><tr><td>Scenario-Actor vs Greedy</td><td>-11.90</td><td>[−14.71, -9.02]</td><td> $6 . 1 \times 1 0 ^ { - 1 6 }$ </td></tr><tr><td>Scenario-Actor vs Fixed T = 0.5</td><td>-4.25</td><td>[-7.13, -1.36]</td><td> $4 . 7 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>Scenario-Actor vs Fixed T = 0.7</td><td>-3.79</td><td>[-6.80, -0.80]</td><td> $1 . 2 \times 1 0 ^ { - 2 }$ </td></tr><tr><td>Scenario-Actor vs Fixed T = 1.0</td><td>+2.88</td><td>[-0.23, +5.99]</td><td> $7 . 5 \times 1 0 ^ { - 2 }$ </td></tr><tr><td>Scenario-Actor vs Fixed T = 1.5</td><td>+20.09</td><td>[+16.83, +23.35]</td><td> $2 . 5 \times 1 0 ^ { - 3 1 }$ </td></tr><tr><td>Scenario-Actor vs EDT</td><td>-3.87</td><td>[−6.90, -0.83]</td><td> $1 . 5 \times 1 0 ^ { - 2 }$ </td></tr><tr><td>Scenario-Actor vs Nucleus</td><td>+3.94</td><td>[+0.76, +7.20]</td><td> $1 . 7 \times 1 0 ^ { - 2 }$ </td></tr><tr><td>Scenario-Actor vs Random-T</td><td>-0.53</td><td>[-3.79, +2.73]</td><td> $7 . 8 \times 1 0 ^ { - 1 }$ </td></tr></table>

Table 11: Statistical significance testing results on AgentSense, AlpacaEval, MT-Bench, and GSM8K. Differences are computed as Scenario-Actor minus the corresponding baseline.

## C Scenario Descriptions

<table><tr><td>Scenario Label</td><td>Description</td></tr><tr><td>rigorous</td><td>Interactions centered on correctness, logical structure, technical precision, or verifiable reasoning.</td></tr><tr><td>institutional</td><td>Interactions organized around explicit roles, responsibilities, procedures, or task objectives. The respondent should help resolve the issue efficiently.</td></tr><tr><td>formal</td><td>Interactions where etiquette, tact, face-saving, and socially careful wording are central.</td></tr><tr><td>emotional_support_neg</td><td>Interactions in which the user expresses distress, worry, sadness, frustra- tion, vulnerability, or seeks comfort.</td></tr><tr><td>adversarial</td><td>Interactions involving disagreement, critique, challenge, defense, debate, or pressure to justify a position.</td></tr><tr><td>emotional_support_pos</td><td>Interactions in which the user shares positive affect, appreciation, celebra- tion, affection, or seeks a warm affirming response.</td></tr><tr><td>casual_conversation</td><td>Interactions that are low-stakes, socially light, and not primarily task-, role-, or problem-driven.</td></tr><tr><td>brainstorming</td><td>Interactions that ask for idea generation, alternatives, exploration, or open-</td></tr><tr><td>playful</td><td>ended ideation. Interactions that are primarily humorous, teasing, and entertaining.</td></tr></table>

Table 12: Scenario labels and their corresponding communicative descriptions.

D Layer Sweep and Geometric Diagnostics
<table><tr><td>Layer</td><td>Val Recall@1</td><td>Effective Rank</td><td>Tanh Sat.</td><td>PCA@95%</td><td>Max Axis Offdiag</td><td>Collapse</td></tr><tr><td>12</td><td>0.793</td><td>25.15</td><td>0.64%</td><td>21</td><td>0.63</td><td>acceptable</td></tr><tr><td>16</td><td>0.770</td><td>24.41</td><td>4.33%</td><td>20</td><td>0.60</td><td>acceptable</td></tr><tr><td>20</td><td>0.798</td><td>22.80</td><td>15.42%</td><td>16</td><td>0.80</td><td>acceptable</td></tr><tr><td>24</td><td>0.739</td><td>18.21</td><td>50.70%</td><td>11</td><td>0.74</td><td>moderate</td></tr><tr><td>29</td><td>0.576</td><td>10.95</td><td>88.17%</td><td>6</td><td>0.99</td><td>severe</td></tr><tr><td>30</td><td>0.518</td><td>7.08</td><td>92.24%</td><td>4</td><td>0.94</td><td>severe</td></tr></table>

Table 13: Layer sweep diagnostics for scenario geometry. Collapse severity is assessed jointly from tanh saturation, effective rank, and axis off-diagonal correlation. Deeper layers exhibit increasing saturation, rank collapse, and axis degeneracy.

## E Agreements

<table><tr><td>Metric</td><td>LLM-human</td><td>Human-human</td></tr><tr><td>Agreement (%)</td><td>83.66 / 86.27</td><td>81.37</td></tr><tr><td>Cohen&#x27;s κ</td><td>0.82 / 0.85</td><td>0.79</td></tr><tr><td>Macro Precision</td><td>0.84</td><td>0.83</td></tr><tr><td>Macro Recall</td><td>0.84</td><td>0.80</td></tr><tr><td>Macro F1</td><td>0.83</td><td>0.81</td></tr></table>

Table 14: LLM annotator validation, values are rounded to two decimal places

<table><tr><td>Metric</td><td>LLM-human Human-human</td></tr><tr><td>AE Agreement (%) 86.00 / 86.00</td><td>80.00</td></tr><tr><td>AE Cohen&#x27;s κ</td><td>0.72 / 0.72 0.60</td></tr><tr><td>MT Pearson r 0.92 / 0.86</td><td>0.87</td></tr><tr><td>MT Spearman ρ</td><td>0.84 / 0.70 0.79</td></tr><tr><td>MT MAE</td><td>0.44 / 0.90 0.78</td></tr><tr><td>AS Agreement (%)</td><td>80.00 / 72.00 76.00</td></tr><tr><td>AS Cohen&#x27;s κ</td><td>0.48 / 0.27 0.40</td></tr></table>

Table 15: LLM judge validation, values are rounded to two decimal places.

## F Source Composition

![](images/024debf8c2590a0ca02f0df8a588b63070d57a8fc8583153bed67e06d73b1acd.jpg)  
Figure 5: Source composition by scenario.

## G Triplet-Margin Ablation

<table><tr><td>Geometry metric</td><td> $\beta = \mathbf { 0 . 3 }$ </td><td> $\beta = { \bf 1 . 0 }$ </td><td>Δ</td></tr><tr><td>Mean pairwise centroid cosine</td><td>0.9496</td><td>0.7103</td><td>-0.2393</td></tr><tr><td>Centered centroid cosine</td><td>0.5258</td><td>0.5057</td><td>-0.0201</td></tr><tr><td>Centroid effective rank at 95%</td><td>5</td><td>6</td><td>+1</td></tr><tr><td>Euclidean silhouette</td><td>0.1835</td><td>0.2668</td><td>+0.0833</td></tr><tr><td>Nearest-centroid agreement</td><td>0.6940</td><td>0.8877</td><td>+0.1937</td></tr><tr><td>Tanh saturation (&gt; 0.95)</td><td>0.0938</td><td>0.0347</td><td>-0.0591</td></tr></table>

Table 16: Scenario geometry under two triplet-loss margins.
<table><tr><td>Benchmark</td><td>Metric</td><td> $\beta = \mathbf { 0 . 3 }$ </td><td> $\beta = { \bf 1 . 0 }$ </td><td>Δ</td></tr><tr><td>GSM8K</td><td>Accuracy (%)</td><td>65.28</td><td>66.41</td><td>+1.13</td></tr><tr><td>AlpacaEval</td><td>LC-WR (%)</td><td>30.03</td><td>25.54</td><td>-4.49</td></tr><tr><td>MT-Bench</td><td>GPT-4 score</td><td>7.91</td><td>7.66</td><td>-0.25</td></tr><tr><td>DailyDialog NextTurn</td><td>WR (%)</td><td>49.70</td><td>46.94</td><td>-2.76</td></tr><tr><td>DailyDialog ContextLen</td><td>WR (%)</td><td>67.00</td><td>62.00</td><td>-5.00</td></tr><tr><td>Diagnostic</td><td></td><td></td><td></td><td></td></tr><tr><td>GSM8K</td><td>Mean selected T</td><td>0.822</td><td>0.738</td><td>-0.084</td></tr></table>

Table 17: Downstream performance under two triplet-loss margins.

## H Computational Details

We keep the backbone model frozen and train only the Scenarize module, which maps the 4096-dimensional Llama hidden states into a 32- dimensional scenario embedding. All experiments run on a single NVIDIA A10G GPU with 24GB memory. The total computational budget, including hidden-state extraction, Scenarize training, layer sweeps, and evaluation runs, is approximately 125 A10G GPU-hours. This estimate excludes external API inference costs used for GPT-4o annotation and GPT-4-turbo evaluation.

## I Artifact licenses and terms of use.

We use Llama-3.1-8B-Instruct, WildChat, Daily-Dialog, ChangeMyView (Tan et al., 2016), SODA (Kim et al., 2023), EmpatheticDialogues (Rashkin et al., 2019), ESConv (Liu et al., 2021), OpenAssistant (Köpf et al., 2023), $\mathrm { R e ^ { 2 } }$ (Zhang et al., 2025), DISAPERE (Kennard et al., 2022), OpenReview review $\mathrm { t e x t } ^ { 3 }$ , Enron email corpus (Klimt and Yang, 2004), GPT-4o, and GPT-4-turbo, and previously proposed decoding baselines in accordance with their respective licenses or terms of use. All datasets are used only for research purposes, and no attempt is made to redistribute restricted model weights or proprietary model outputs beyond aggregate experimental results.