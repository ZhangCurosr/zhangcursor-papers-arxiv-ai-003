# Why machines will still not rule the world

Jobst Landgrebe and Barry Smith

October 9, 2026 jobst.landgrebe@usi.ch, phismith@bufalo.edu

## Abstract

In our book Why machines will never rule the world [13, 14] we argue that artificial general intelligence is mathematically impossible. This is because the human beings and the processes which exhibit intelligence are complex systems whose behaviour cannot be captured by the kinds of models that we can generate with or without computers. Proponents of contemporary machine intelligence respond with two lines of argument: a theoretical one, grounded in the universal approximation theorems for neural networks and the Church-Turing-Deutsch principle; and an empirical one, grounded in rapidly rising scores on standardized benchmarks. In this communication we examine and reject both responses. First, we show serious issues in the physicalist counter-argument based on the Church-Turing-Deutsch principle. Second, we review recent evidence to the efect that prominent benchmarks are compromised by training-data contamination, flawed test construction, and strategic optimization. Our central argument remains: That models required to perform cognitive behaviour in open-ended, thermodynamically complex and non-ergodic environments are not and will not become achievable.

Keywords: artificial general intelligence, complex systems, ergodicity, Church-Turing-Deutsch principle, benchmark contamination, Goodhart’s law.

## 1 Introduction

Few questions in contemporary philosophy of technology are as consequential, or as contested, as whether machines can come to match the general intelligence of human beings. Public discourse on the question has been shaped largely by empirical milestones: each new generation of large language models (LLMs) is introduced alongside a table of benchmark scores, and the steady upward movement of those scores is frequently interpreted as evidence that general intelligence is approaching. This leads to an extreme hyperboly according to which mankind is currently building the tools of its own destruction, which will occur when evil, hyperintelligent AI systems will ‘switch of’ mankind. In this cultural context we advance in our monograph [13, 14] the counter-claim that artificial general intelligence (AGI) is not merely distant or dificult, but that it is impossible, because the systems that would need to be modeled to produce it are of a kind that mathematics cannot adequately model.

This essay evaluates this thesis against its two sets of main objections. The first is a theoretical counter-argument holding that, because brains are physical systems and physical systems are computable, anything a brain does can in principle be done by a computer. This argument draws on the universal approximation theorems for neural networks [5, 10] and, in its strongest form, on the Church-Turing-Deutsch principle [8]. The second is an empirical counter-argument holding that benchmark performance demonstrates the progressive acquisition of general capabilities. We argue here that this theoretical argument against our impossibility claim establishes nothing and that recent methodological audits seriously undermine the empirical counter-argument.

## 2 Complex systems limit mathematical modeling

We build our argument on a distinction between what we call logic systems<sup>1</sup> on the one hand – by which we mean systems whose behavior can be captured by explicit equations – and complex systems, which means systems which exhibit certain thermodynamical properties, including evolutionary character, dependent interactions, force overlay, nonequilibrium dynamics, context dependence, and the absence of a stable phase space. Human intelligence, we contend, is the product of precisely such a system – the human organism embedded in a natural and social environment – and its most important capacities, including dialogue and the open-ended spontaneous adaptation to novel situations, are expressions of that complexity. Because the mathematics available to science can model complex systems only partially, approximately, and for limited purposes, we argue that no computer program can reproduce the behavior those systems generate. For this is a behaviour that requires the functioning of the entire system processes, as we can see from neurological or psychiatric diseases, where partial dysfunctions as experienced, for example in intoxication with psychoactive substances, are already detrimental to the brain’s functioning.

The argument is not a claim about computational speed or hardware. It is a claim about the availability of models. Computers execute algorithms, and algorithms implement mathematical models; if adequate models of a given system cannot be constructed, then no amount of computation will reproduce that system’s behavior. In an earlier paper, we applied a version of this reasoning to natural language, arguing that dialogues create contexts and ever-shifting intentions of dialogue participants that AI models cannot emulate [12]. The formal core of the argument concerns the statistical properties of the processes that complex systems generate, and in particular their lack of ergodic behaviour.

## 3 Ergodicity and non-ergodicity

The concept of ergodicity provides the clearest formal statement of why learning from past data may fail to support prediction of future behavior. The following definitions follow the standard measure-theoretic presentation formulated by Birkhof [1].

Definition 1. Measure-preserving dynamical system. A measure-preserving dynamical system is a quadruple $\{ \Omega , { \mathcal { F } } , \mu , T \}$ , where $\{ \Omega , { \mathcal { F } } , \mu \}$ is a probability space and $T : \Omega \to \Omega$ is a measurable transformation satisfying $\mu ( T ^ { - } ( A ) ) \ : = \ : \mu ( A ) , \forall A \in \mathcal { F } ,$ i.e. all A are elements of the σ-algebra $\mathcal { F }$

Definition 2. Ergodicity. A system is ergodic $i f$ every invariant set is trivial; that is, whenever $T ^ { - } ( A ) = A \to \mu ( A ) \in \{ 0 , 1 \}$ .

Intuitively, an ergodic system cannot be decomposed into separate regions that its trajectories never leave. Birkhof’s (1931) pointwise ergodic theorem states the operational consequence of this definition. For an ergodic system, an ergodic mapping $T$ and any integrable observable $f ,$ we have:

$$
\operatorname* { l i m } _ { n  \infty } { \frac { 1 } { n } } \sum _ { k = 0 } ^ { n - 1 } f ( T ^ { k } x ) = E [ f ] ,\tag{1}
$$

where E is the expectation of invariant sets of $T .$

For a stochastic process $X ( t )$ in continuous time, the corresponding statement is that the time average along a single realization converges to the ensemble average $E [ X ]$

$$
\operatorname* { l i m } _ { T \to \infty } \frac { 1 } { T } \int _ { 0 } ^ { T } X ( t ) d t = E [ X ] .\tag{2}
$$

Definition 3. Non-ergodicity. A process is non-ergodic with respect to an observable if equation 2 is not true – either because the time-average limit does not exist, or because it exists but difers from $E [ X ]$ on a set with a positive measure, so that what is observed along one trajectory depends on initial conditions and its past.

An everyday example is an idealised bottle of water at rest in perfect thermodynamic equilibrium. The particles in the bottle move in an ergodic manner, which means that each will at some point in time return to exactly the same position at which it was at a given moment of (idealised) observation. Or, in other words, the likelihood of every particle being anywhere in the bottle is identical for all molecules over time. If the bottle is heated, put under pressure, shaken or energy is introduced into it in any other manner, the particles behave in a non-ergodic fashion and the particles’ behaviour becomes nonergodic.

Peters [18] ofers a simple example from economics. Consider a repeated gamble in which wealth is multiplied by 1.5 or by 0.6 with equal probability. The ensemble average of the per-round growth factor is $\begin{array} { r } { E [ r ] = \frac { 1 } { 2 } { \cdot } 1 . 5 + \frac { 1 } { 2 } { \cdot } 0 . 6 = 1 . 0 5 } \end{array}$ , suggesting growth of 5% per round. The time-average growth factor experienced by any single individual, however, is the geometric mean: $\bar { g } = ( 1 . 5 \cdot 0 . 6 ) ^ { \frac { 1 } { 2 } } = \sqrt { 0 . 9 } \approx 0 . 9 4 9$ , implying a decline of about 5% per round. Almost every individual trajectory therefore decays toward zero even though the expectation across the ensemble grows. The two averages diverge because the process is multiplicative and therefore non-ergodic in wealth. That said, the fact that the ensemble average is ergodic, while the individual trajectory is not in this simple example does not mean that one can model non-ergodic phenomena as regular; the simple example does not generalise to real complex systems. Regularity here means the presence of recurrent patterns.

## 3.1 Machine learning

The relevance to machine learning follows from the structure of statistical learning theory. Supervised learning, the tool that generates all LLMs and other stochastic models by optimisation, minimizes the empirical risk

$$
{ \hat { R } } _ { n } ( f ) = { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } \ell ( f ( x _ { i } ) , y _ { i } ) ,\tag{3}
$$

where ℓ is a loss function, $f$ is a machine learning functional and R is risk. $\hat { R } _ { n } ( f )$ is used as an estimate of the true risk

$$
R ( f ) = E _ { x , y } \sim \mathcal { D } _ { \kappa } [ \ell ( f ( x ) , y ) ] ,\tag{4}
$$

where $\mathcal { D } _ { \kappa }$ is a training and test distribution.

In this setting the ability of the functional to generalise depends on the assumption that training, test and deployment data are drawn from the same fixed distribution $\mathcal { D } _ { \kappa }$ If the process generating the data is non-ergodic – so that the deployment distribution $\mathcal { D } _ { 1 }$ is not identical with the training distribution $\mathcal { D } _ { 0 }$ :

$$
\mathcal { D } _ { 1 } \cup \mathcal { D } _ { 0 } \ne \mathcal { D } _ { 0 } ,\tag{5}
$$

then even a small empirical risk (reflecting excellent training and test performance) provides no guarantee as to future performance. We argue [13] that this is the necessary condition of human intelligence and the environment in which it operates, since both are complex systems and non-ergodic. It is therefore impossible to sample from a distribution generated by human intelligence adequately, and equation (5) will always hold.

## 4 The Connectionist Counter-Thesis

## 4.1 Universal Function Approximation

The first theoretical response to our argument appeals to the expressive power of neural networks (NNs). Cybenko [5] and Hornik, Stinchcombe, and White [10] proved that feedforward networks with a single hidden layer are universal approximators. This result

is often formulated as follows, where it is important to note that a non-polynomial functional σ is required to approximate an arbitrary functional using a neural network:

Let $C ( X , \mathbb { R } ^ { m } )$ denote the set of continuous functions from a subset X of a Euclidean $\mathbb { R } ^ { n }$ space to a Euclidean space $\mathbb { R } ^ { m }$ . Let $\sigma \in C ( \mathbb { R } , \mathbb { R } )$

Then $\sigma$ is not polynomial $i f$ and only $i f$ for every $n \in \mathbb { N } , m \in \mathbb { N }$ , compact subspace $K \subseteq \mathbb { R } ^ { n } , f \in C ( K , \mathbb { R } ^ { m } ) , \varepsilon > 0$ , there exist $k \in \mathbb { N } , A \in \mathbb { R } ^ { k \times n } , b \in \mathbb { R } ^ { k } , C \in \mathbb { R } ^ { m \times k }$ such that

$$
\operatorname* { s u p } _ { x \in K } \| f ( x ) - g ( x ) \| < \varepsilon { \mathrm { ~ w h e r e ~ } } g ( x ) = C \cdot ( \sigma \circ ( A \cdot x + b ) )\tag{6}
$$

NN-connectionists infer that, if intelligent behavior is a function from inputs to outputs, then a suficiently large network can represent it. The theorem, however, carries three restrictions that matter for the present debate. (i) It is an existence result and says nothing about whether gradient-based training on finite data will ever find the needed approximating parameters. (ii) It applies to fixed functions on compact domains, whereas complex systems – such as inanimate and animate entities in nature – are characterized by open, shifting domains. And (iii) it presupposes further that the target function exists as a stable mapping, which is exactly what non-ergodicity calls into question. The universal approximation theorem therefore does indeed show that representational capacity is not the bottleneck; but it fails to show that the right function can be learned or even specified when it comes to modelling complex systems. It cannot be used to rebut the lack of a regular distribution resulting from non-ergodic processes.

## 4.2 The Church-Turing-Deutsch Principle

A deeper response grounds the possibility of AGI in physics rather than in network architecture. The classical Church-Turing thesis, formulated independently by Church [4] and Turing [24], concerns efective computability: any function computable by an idealized human following a mechanical procedure is computable by a Turing machine. The thesis is a claim about mathematical procedures that can be realised by idealised machines. Deutsch [8] proposed a physical strengthening – now commonly called the Church-Turing-Deutsch (CTD) principle – whereby every finitely realizable physical system can be perfectly simulated by a universal computing machine operating by finite means. Because Deutsch showed that classical Turing machines cannot satisfy this principle for quantum systems in the strict sense he intended, he introduced the universal quantum computer as the machine that can.

The physicalist argument against our thesis then proceeds in three steps.

1. The human brain is a finitely realizable physical system.

2. Its time evolution is governed by physical laws; in the quantum-mechanical description, a closed system with Hamiltonian H evolves under the unitary operator

$$
U ( t ) = e ^ { \frac { - i H t } { \hbar } } .\tag{7}
$$

3. By the CTD principle, a universal quantum computer can simulate this evolution to arbitrary precision. If the brain is a physical system of this kind, the conclusion follows that there is no in-principle barrier to its simulation.

This syllogism is however false and establishes nothing against our thesis. First, the CTD principle, though it is widely accepted among physicalists, is a conjecture about physics rather than a proven theorem. More specifically, we show in chapter 7 of [14] that the CTD-thesis is not supported by modern physics, which provides us with only very incomplete models of nature suitable, with small exceptions, only to artificial or highly idealised settings. Second, the argument establishes simulability given the Hamiltonian and the initial state. We do not claim that a complex system’s dynamics are uncomputable once known, but rather that the relevant models, equations, parameters, and initial conditions, cannot be obtained for complex systems. Of course we acknowledge that there are partial models for complex systems – for example LLMs – that are prefectly computable, such as the models for $\mathrm { \ m o s t l y ^ { 2 } }$ correct syntax. But the total behaviour of a complex system, such as the cognitive abilities of the human brain, cannot be modelled in this way. Parameterising equation (7) for a human brain would require specifying H and the initial quantum state of roughly $1 0 ^ { 2 6 }$ interacting particles, together with the environment with which the brain continuously exchanges energy and information. We do not have such models, let alone the ability to measure the initial conditions. Third, even an exact simulation of one brain would not constitute a predictive model of an open environment; it would reproduce one trajectory of a non-ergodic system, inheriting rather than resolving the forecasting problem described in equation (3).

The CTD principle therefore has nothing to say against our thesis. It claims to remove the suggestion that intelligence involves something non-physical. But we never make this suggestion. Rather, we defend a position we call ‘nomological monism’, which is a version of strict materialism [14, chapter 2]. All we say is that processes which cannot be mathematically modelled are not computable and that what we introspectively experience as the human mind cannot be modelled. The AGI hyperbole proponents claim nevertheless that such models can be acquired, and since they cannot support this using theoretical arguments, they turn from theory to what they see as empirical evidence.

## 5 So-called benchmark evidence

The empirical counter-thesis to our claim as to the limits of AI holds that, whatever the theoretical arguments, LLMs have demonstrably acquired general capabilities, as shown by their performance on benchmarks designed to test reasoning, expert knowledge, and open-ended problem solving. The force of this argument depends entirely on the validity of the benchmarks, and recent audits give substantial reason to doubt that validity.

## 5.1 Contamination and non-validity of benchmarks

The claim is regularly made that benchmark performance proves the cognitive abilities of models. An important benchmark to measure the quality of LLMs in software engineering, SWE-bench, is a model evaluation framework consisting of 2294 software engineering problems drawn from real GitHub issues. Its aim is to evaluate whether a model can resolve real issues drawn from open-source Python repositories on GitHub, where correctness is judged by the repositories’ own test suites [11]. Its human-validated subset, SWE-bench verified, became a standard metric in frontier model releases, with top systems reporting scores near 80%.

But in February 2026, OpenAI [17] announced that it would stop reporting SWE-bench verified results. Its evaluations team audited 138 problems that a strong model had failed to solve consistently and found that 59.4% contained material flaws in test design or problem descriptions – including tests that rejected functionally correct solutions. The audit also found that frontier models from several developers could reproduce some reference solutions without search (merely using the models’ configurations), a strong indication of training-data contamination arising from the public availability of the benchmark and its source repositories.

To limit contamination, SWE-bench pro, was constructed with held-out, publicly non available commercially sourced codebases. This yielded scores of roughly 23% for leading models at its release [7], compared with scores above 70% for comparable models on SWEbench verified. The comparison must be interpreted cautiously, because SWE-bench pro was deliberately designed to contain longer, harder tasks, and some of the gap reflects dificulty rather than contamination. Nevertheless, this finding shows that a score widely presented as evidence of autonomous software engineering capability was, to a material degree, a measure of training configuration and of the idiosyncrasies of a test harness. Thus the contrast between SWE-bench type model performance under contamination conditions, on the one hand, and contamination-resistant evaluation, where the model is unable to make usage of training-distribution derived configuration, on the other, is striking.

The practical consequences of this inability to cope with novel situations and with context diversity become evident when one looks at the domain in which LLM-generated output has been used for production purposes, namely software engineering. There the error rates are so high and the inability to take into account the architectural context in AI-generated code increases maintenance costs or leads all too often to outages once the model-generated code is deployed.<sup>3</sup>

## 5.1.1 Knowledge Benchmarks and Web-Scale Contamination

Similar concerns apply to knowledge-intensive benchmarks such as MMLU-Pro [25] and GPQA [19], which are often cited as evidence of expert level reasoning. Because LLMs are trained on web-scale corpora, benchmark items and their answers can enter training data through many indirect routes. Deng et al. [6] showed that some commercial models could guess deliberately masked incorrect options in multiple-choice test sets, a behavior dificult to explain without assuming exposure to the data. A systematic review of 55 studies [16] proposed a taxonomy of contamination ranging from exact duplication to task-level leakage and concluded that no existing detection method is consistently reliable across contamination types, model-access settings, and training stages. The implication is not that every high score is contaminated, but that contamination can rarely be ruled out, which weakens the inference from score to capability.

A related problem arises when models are evaluated by other models. LLM judges have been shown to favor longer and more verbose responses [26] and to be susceptible to authority cues and superficial features of presentation such as citations and formatting, independent of substantive correctness [2]. To the extent that evaluations reward the surface features of correct-looking answers, what is being measured is fluency in the conventions of expert discourse rather than the reasoning those conventions normally signal.

In other words, what we see when LLMs perform well at benchmarks is the result of a configuration with training data that contain many of the data which are used to score the models. In the sense of equations (3)-(5) this means that $\mathcal { D } _ { 0 } \cap \mathcal { D } _ { 1 } \neq \emptyset$ with the model performance depending on the size of the intersection between deployment $( \mathcal { D } _ { 0 } )$ and training (D<sub>1</sub>) set.

## 5.2 Human-preference leaderboards

Chatbot Arena is an LLM rating scheme allowing anyone to submit a prompt and subsequently rank two anonymous responses from diferent models. It was designed to address the static benchmark problem by collecting large numbers of pairwise human preference judgments on open-ended prompts [3]. Singh et al. [21], however, documented systematic distortions in its leaderboard. Among their findings was that some providers privately tested many model variants before public release and published only the best-performing one. They also found that

proprietary closed models are sampled at higher rates [. . . ] and have fewer models removed from the arena than open-weight and open-source alternatives. Both these policies lead to large data access asymmetries over time.

[. . . ] Together, these dynamics result in overfitting to Arena-specific dynamics rather than general model quality.

Taken together it seems that preference leaderboards measure the persuasiveness of outputs to typical users on typical queries rather than performance in sustained, consequential, multi-turn tasks.

## 5.3 Goodhart’s law

These failures share a common structure, captured by Goodhart’s law [9]. He observed that statistical regularities in monetary policy tended to collapse once they were used for control. Strathern [23] generalized the point into its familiar form, that a measure which becomes a target ceases to be a good measure. A benchmark is valid only so long as performance on it is correlated with the underlying capability it was designed to detect. Once the benchmark becomes the object of optimization – through training on public data that includes its items, through selective reporting, or through tuning toward the preferences of its raters – that correlation weakens, and the score increasingly measures the optimization process itself.

This analysis is directly related to the formal argument of the preceding sections. A static benchmark is a fixed sample from a fixed distribution. High performance on such a benchmark shows that a model has achieved low empirical risk with respect to that distribution (equation (3)). It cannot, by construction, show that the model will perform well when the distribution shifts (equation (5)). Benchmark contamination is in this sense an extreme case of the general problem: when the test distribution has leaked into the training distribution, the evaluation measures interpolation within known data rather than adaptation to the novel, non-ergodic conditions that we know are an essential characteristic of real environments.

## 6 Discussion

It would of course be a mistake to conclude that the benchmark evidence vindicates our impossibility thesis. Our claim is modal and universal: no machine will ever achieve artificial general intelligence (AGI). Benchmark failures are contingent and particular: specific measurement instruments have been compromised in specific ways. The fact that SWE-bench verified overstated capability does not show that capability is absent, only that this instrument cannot establish its presence. Moreover, models continue to show over time measurable, if more modest, performance on contamination-resistant evaluations. Continuously refreshed benchmarks that draw tasks from after a model’s training cutof ofer partial remedies to the contamination problem. The appropriate conclusion is therefore asymmetric: The audits substantially weaken the empirical case for general intelligence, but they do not by themselves supply an empirical case against its possibility.

Conversely, proponents of the connectionist position cannot rely on the CTD principle to make the case for AGI. In-principle simulability is compatible with what we assert as the in-practice unobtainability of relevant models. Can learning systems acquire models that remain reliable under sustained distributional shift? Can evaluation methods be designed that detect such reliability without themselves becoming targets of optimization? We remain confident that the answer to both of these questions is negative.

We argue that AGI is impossible because complex systems – which means all organisms and also all of inanimate nature – generate non-ergodic behavior, behavior that mathematical models, and therefore computers, cannot capture. We have provided a formal account of the ergodicity that lies at the heart of this argument and shown why non-ergodicity undermines the statistical assumptions on which machine learning’s generalization guarantees rest. We have reconstructed the strongest theoretical arguments, grounded in the universal approximation theorems and the Church-Turing-Deutsch principle, and shown that these results establish representational and in-principle computational adequacy without giving any evidence that the requisite models could ever be obtained. Contrarywise, both approaches make it clear why intelligence cannot be modelled, since both show that complex system behaviour is out of scope for mathematics.

Importantly, though it is impossible to model complex non-ergodic processes mathematically, the human brain can react to non-ergodic processes generating responses adequate to the situation without any preparation. That is what we call intelligence [14, secti. 3.2].

We also reviewed evidence that leading benchmarks have been compromised by contamination, flawed construction, and strategic optimization, in conformity with Goodhart’s law. The benchmarks do not show that machines have mastered complexity; they show, in large part, that developers have become adept at optimizing for benchmarks. The failure of a measurement instrument is of course not proof of the impossibility of what it was meant to measure. On the other hand, however, we postulate that evaluation regimes that are built to challenge algorithms with non-ergodic data that are held out at training time and continuously renewed and which are resistant to optimization will provide the proponents of AGI with evidence to the efect that machines will not become intelligent since they will not be able to cope with token sequences that are absent from their training material. This is already obvious from challenges to so-called ‘reasoning models’ presented in [20].

We do not think that the human mind is capable of creating mathematics that can holistically model complex systems, neither on its own with explicit modelling or with the help of optimisation algorithms developed by humans that yield huge implicit models such as LLMs or multi-modal models. Furthermore, the new edition of our book contains material regarding practical intelligence [22], a capability which operates below the level of consciousness which yet dominates human behaviour and grounds cognitive intelligence. This is intelligence of a form that is not even accessible to machine configuration, since no pertinent training data could ever be made available.

## References

[1] George D Birkhof. “Proof of the ergodic theorem”. In: Proceedings of the National Academy of Sciences 17.12 (1931), pp. 656–660.

[2] Guiming Hardy Chen et al. “Humans or LLMs as the judge? a study on judgement bias”. In: Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing. 2024, pp. 8301–8327.

[3] Wei-Lin Chiang et al. “Chatbot arena: An open platform for evaluating llms by human preference”. In: arxiv:org/abs/2403.04132 2.10 (2024).

[4] Alonzo Church. “A note on the Entscheidungsproblem”. In: Journal of Symbolic Logic 1 (1936), pp. 40–41.

[5] George Cybenko. “Approximation by superpositions of a sigmoidal function”. In: Mathematics of control, signals and systems 2.4 (1989), pp. 303–314.

[6] Chunyuan Deng et al. “Investigating data contamination in modern benchmarks for large language models”. In: Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers). 2024, pp. 8706–8719.

[7] Xiang Deng et al. “Swe-bench pro: Can ai agents solve long-horizon software engineering tasks?” In: arxiv:abs/2509.16941 (2025).

[8] David Deutsch. “Quantum theory, the Church–Turing principle and the universal quantum computer”. In: Proceedings of the Royal Society of London. A. Mathematical and Physical Sciences 400.1818 (1985), pp. 97–117.

[9] Charles AE Goodhart. “Problems of monetary management: the UK experience”. In: Monetary theory and practice: The UK experience. 1984, pp. 91–121.

[10] Kurt Hornik, Maxwell Stinchcombe, and Halbert White. “Multilayer feedforward networks are universal approximators”. In: Neural networks 2.5 (1989), pp. 359– 366.

[11] Carlos E Jimenez et al. “Swe-bench: Can language models resolve real-world github issues?” In: International Conference on Learning Representations. Vol. 2024. 2024, pp. 54107–54157.

[12] Jobst Landgrebe and Barry Smith. “Making AI meaningful again”. In: Synthese 198.3 (2021), pp. 2061–2081.

[13] Jobst Landgrebe and Barry Smith. Why machines will never rule the world. AI without fear. London: Routledge, 2022.

[14] Jobst Landgrebe and Barry Smith. Why machines will never rule the world. AI without fear. 2nd ed. London: Routledge, 2025.

[15] Elliot Murphy et al. “Fundamental Principles of Linguistic Structure Are Not Represented by ChatGPT”. In: Biolinguistics 19 (2025), pp. 1–55.

[16] Erfan Nourbakhsh et al. “Are LLM Benchmarks Already Contaminated? A Systematic Review of Contamination Detection Methods”. In: Proceedings of the Fifth Workshop on Generation, Evaluation and Metrics (GEM). 2026, pp. 518–539.

[17] OpenAI. Why SWE-bench Verified no longer measures frontier coding capabilities. Feb. 2026. url: https://openai.com/index/why-we-no-longer-evaluate-swebench-verified/.

[18] Ole Peters. “The ergodicity problem in economics”. In: Nature Physics 15.12 (2019), pp. 1216–1221.

[19] David Rein et al. “Gpqa: A graduate-level google-proof q&a benchmark”. In: arXiv:2311.12022 (2023).

[20] Parshin Shojaee et al. The Illusion of Thinking: Understanding the Strengths and Limitations of Reasoning Models via the Lens of Problem Complexity. 2025.

[21] Anjali Singh et al. “Protecting human cognition in the age of AI”. In: arXiv:2502.12447 (2025).

[22] Barry Smith. “LLMs and Practical Knowledge: What is Intelligence?” In: Electrifying the Future, 11th Budapest Visual Learning Conference. Ed. by Kristof Nyiri. Hungarian Academy of Science, 2024, pp. 19–26.

[23] Marilyn Strathern. “‘Improving ratings’: audit in the British University system”. In: European review 5.3 (1997), pp. 305–321.

[24] Alan Turing. “On Computable Numbers, with an Application to the Entscheidungsproblem”. In: Proceedings of the London Mathematical Society 42 (1) (1937), pp. 230–265.

[25] Ryan Xiao Wang and Sylvie Thiébaux. “Learning generalised policies for numeric planning”. In: Proceedings of the International Conference on Automated Planning and Scheduling. Vol. 34. 2024, pp. 633–642.

[26] Lianmin Zheng et al. “Judging llm-as-a-judge with mt-bench and chatbot arena”. In: Advances in neural information processing systems 36 (2023), pp. 46595–46623.