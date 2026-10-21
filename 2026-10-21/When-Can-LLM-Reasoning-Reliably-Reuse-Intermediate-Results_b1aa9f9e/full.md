# When Can LLM Reasoning Reliably Reuse Intermediate Results?

Deyuan Liu<sup>1</sup> Zecheng Wang<sup>2</sup> Xuanyi Liu<sup>3</sup> Jungang Li<sup>4</sup> Yihao Hu<sup>1,5</sup> Mingjie Tang<sup>6</sup> Bryan Dai<sup>6</sup> Jie Fu<sup>6\*</sup>

<sup>1</sup>Westlake University <sup>2</sup>Wechat AI <sup>3</sup>Peking University <sup>4</sup>HKUST <sup>5</sup>Ant Group <sup>6</sup>IQuest Research

## Abstract

Sampling complete solutions gives large language models more chances to solve a problem, but each attempt starts again. For tasks with several subproblems, all required answers must be correct in the same attempt. Retaining intermediate results can avoid repeating correct work, but errors in those results may affect later attempts. We study when this reuse is reliable and how much computation it requires. For local retries that stop when no candidate passes within the retry budget, we give conditions for correct completion and bound the average number of attempts. We also show how checking cost and the chance of passing a check affect total computation. Experiments across eight reasoning models examine composite mathematics and arithmetic. Across four DeepSeek models, average success on the second subproblem is 36.2 percent without first-subproblem context, 57.2 percent with its answer, and 58.3 percent with its reasoning and answer. These scores measure second-subproblem coverage and include continuations made when selection fails. Arithmetic experiments show that similar first-attempt accuracies can lead to very different coverage after repeated attempts. Code is available at https://github.com/SempraETY/Intermediate-State-Reuse.

![](images/f4c708322fbceb128299e4c1facbdf408a79c7109311523bbb9dada24cbc16c5.jpg)  
Q1. Find the quotient when x9 + 1 is divided by x − 1. A1. a± = Σk=0 xk = xδ + x7 + . . + x + 1 Q2. Compute 992 + 99 + 1 in your head. A2. a± = 9901

![](images/af46cff5a3a5c0c222a30801a85e4512156ffc6ff199d84e72ed8031aff672cf.jpg)

![](images/d945d033cc8a068fb49998dd54d84fa06f968fcf3466c8ebfcb99e7c00d7c853.jpg)  
Figure 1: Complete-solution sampling and intermediate-state reuse. Left: Independent sampling requires a successful complete run. The analyzed reuse loop preserves earlier accepted work, retries rejected candidates, and stops when a step’s budget is exhausted. Passing a check does not guarantee correctness. Right: An illustrative two-subproblem candidate-selection example, with SC denoting self-consistency, M the first-subproblem candidate budget, and B the continuation cap. Fractions count correct candidates among those shown; the example is not an experimental measurement or the single-state loop used in the proof.

## Contents

1 Introduction 3   
2 Preliminaries 4   
2.1 Tasks, States, and Sampling Metrics 4   
2.2 Complete-Solution Sampling . 4   
3 Reliability and Cost of State Reuse 5   
3.1 Local Retries and Reliable Completion . 5   
3.2 Expected Work under the Stopping Rule 6   
3.3 Checking Precision and Computation 6   
4 Experiments 7   
4.1 Experimental Setup . . 7   
4.2 Complete-Solution Sampling on Composite Tasks . 7   
4.3 Candidate Selection and Retained Context . 8   
4.4 Finite-Budget Arithmetic Retries 11   
4.5 Agent Score Prediction 11   
5 Related Work 12   
6 Discussion 13   
7 Conclusion 14   
Limitations 14   
AI Use Statement 14   
References 15   
A Modeling Assumptions and Their Scope 18   
B Proofs and Supporting Analysis 20   
C Implementation Details 36   
D Arithmetic Results by Base and Digit Count 40   
E Detailed Results of Repeated Arithmetic Attempts 40   
F Agent Score Prediction: Derivation and Details 49

## 1 Introduction

Many practical reasoning tasks require several distinct subproblems to be solved together. Solving each subproblem in isolation does not ensure that a model can solve the complete task (Zhang et al., 2025b). With model weights fixed, a system can spend more computation at inference time to improve its chance of finding a solution. A system can generate more complete answers, compare candidates, or check intermediate work (Brown et al., 2024; Wang et al., 2022; Lightman et al., 2024). How this computation is allocated matters alongside how much is available (Snell et al., 2025).

Under complete-solution sampling, all required subproblem answers must be correct in one run. One run may solve the first subproblem and another the second, while neither solves both. The sample pool contains correct answers to both subproblems, but no run has solved the complete task. As tasks get longer, more intermediate work can be lost when an attempt fails near the end. A restart retries the failed subproblem but repeats earlier work that may already be correct. Existing methods already use intermediate work: Least-to-Most prompting solves subproblems in sequence using earlier answers (Zhou et al., 2023), while Tree of Thoughts searches over partial reasoning states (Yao et al., 2023).

A subproblem may contain several reasoning steps; our analysis uses steps whose intermediate states can be checked and retained. Consider a three-step calculation that derives a formula, substitutes the given values, and computes a final quantity. Suppose the first two steps are correct but the final calculation fails. A complete restart generates all three steps again. If the formula and substitution can be checked and retained, a new attempt can repeat only the final calculation. The retained results become the retry’s input.

The difficulty is deciding what can safely be kept. A check may reject a correct candidate or accept an incorrect one; benchmarks document the difficulty of identifying errors in intermediate reasoning (Zheng et al., 2025a; Li et al., 2025). An error passed forward may then affect later attempts, as studies of self-conditioning illustrate (Sinha et al., 2025). Even a correct intermediate answer may leave out assumptions or other information needed to continue. Atom of Thoughts constructs self-contained reasoning states (Teng et al., 2025), while MEM1 learns to retain compact information across interaction steps (Zhou et al., 2025). Keeping a full reasoning trace instead supplies more context, but may also retain unchecked mistakes and increases the context to process. Checking itself consumes computation, so avoiding a restart does not by itself establish a saving. This report studies the conditions under which intermediate results can be reused reliably and the cost of doing so.

Recent search methods also avoid unnecessary branching: Chain-in-Tree continues routine steps sequentially and reuses generated children when branching is needed (Li, 2026). Process supervision evaluates intermediate reasoning (Lightman et al., 2024), and MAKER combines fine-grained decomposition with voting to execute long deterministic tasks (Meyerson et al., 2025). MAKER derives near-linear expected computation under its voting model, including the cost of discarded responses. We consider finite local retries, where the first candidate to pass a check is retained and may carry an error into later steps.

The analyzed procedure checks each new result and retries from the last retained state after a rejection. It stops when every candidate within the local budget is rejected. For fixed generation and checking, retries reduce exhaustion without improving the reliability of accepted states. We give an attainable reliability bound and matching worst-case bounds on expected attempts. The work bound includes paths after an incorrect acceptance, where correct-prefix acceptance guarantees no longer apply. Exhaustion limits the work spent at frequently rejected states. Sufficient precision and acceptance probability yield linear expected attempts; total cost also includes checking and state processing.

We also report sampling experiments across eight reasoning models. Arithmetic tasks let us ask how much repeated attempts add beyond first-attempt accuracy. Pairs of mathematics problems let us compare candidate selection and the context supplied to the second subproblem: no first-subproblem context, its answer alone, or its reasoning and answer. The paired subproblems are mathematically separate. Their continuation prompts differ in both content and instructions, so the comparison concerns entire prompt configurations. Budget sweeps examine how success and generated tokens change when more computation is spent on first-subproblem candidates or subsequent continuations.

Across four DeepSeek models, second-subproblem success is 36.2% without first-subproblem context, 57.2% with its answer, and 58.3% with its reasoning and answer. Paired uncertainty is unavailable for these prompt comparisons. These scores include fallback continuations and measure success on the second subproblem, rather than joint success of both answers. Arithmetic results show that similar first-attempt accuracies can lead to very different coverage after repeated attempts. A separate EdgeBench study compares two curves for predicting later agent scores from early observations and observations of the same agents on other tasks (Zhu et al., 2026).

## Report overview.

(a) Reliability of reuse. We give an attainable trajectory bound and the exact accepted-state error threshold for a finite cap to guarantee target reliability.

(b) Expected work and checking cost. We bound worst-case expected work, including erroneous paths, and connect reliable completion to acceptance probability and attempt cost.

(c) Empirical findings. We compare retries, answer selection, retained context, and budget allocation across eight reasoning models, and evaluate two curves for predicting later agent scores.

## 2 Preliminaries

We first define complete-task success, the state passed between steps, and the sampling metrics. These definitions distinguish finding a complete answer from continuing successfully after an intermediate result.

## 2.1 Tasks, States, and Sampling Metrics

A composite task $x _ { 1 } \oplus x _ { 2 }$ presents two subproblems in one prompt and is correct only when both returned subproblem answers are correct. Let $a ^ { \star } ( x )$ be the target answer to x, with $a ^ { \star } ( x _ { 1 } \oplus x _ { 2 } ) = [ a ^ { \star } ( x _ { 1 } ) , a ^ { \star } ( x _ { 2 } ) ]$ Let θ denote the model parameters and $p _ { \pmb { \theta } } ( \cdot \mid s )$ its distribution over generated text under a fixed decoding configuration and input context $s .$ For a problem $x ,$ fix a decomposition into $N \geq 1$ steps, where $N$ counts the steps in this chosen decomposition. Write $z _ { t }$ for the text generated at step $t \in \{ 1 , \ldots , N \}$ and $s _ { t } = [ x , z _ { 1 } , \dots , z _ { t } ]$ for the resulting state, with $s _ { 0 } = x$ . Each step is sampled from $p _ { \pmb { \theta } } ( \cdot \mid s _ { t - 1 } )$ . For reuse, $s _ { t }$ denotes the state actually passed forward: task information plus retained generated text or extracted intermediate answers. The prompt specifies how this state is presented to the model. Correctness concerns the information carried forward; a correct extracted answer need not imply that discarded reasoning was correct. We call a task stepwise decomposable if it can be split into steps whose intermediate states admit a task-specific accept-or-reject check. A state is accepted if it passes the check; acceptance need not imply correctness. A stage groups steps, such as those used to solve one subproblem. Let $y$ be the final answer extracted from a completed run and $L$ its total number of generated tokens. The single-run metrics are:

$$
A _ { \theta , 1 } ( x ) = P ( y = a ^ { \star } ( x ) ) , \qquad C _ { \theta , 1 } ( x ) = \mathbb { E } [ L ] .\tag{1}
$$

For an integer $K \geq 1$ , independent sampling draws K independent complete runs from the same distribution and counts success if any run returns the correct answer.

$$
A _ { \pmb \theta , K } ( \pmb x ) = 1 - ( 1 - A _ { \pmb \theta , 1 } ( \pmb x ) ) ^ { K } , \qquad C _ { \pmb \theta , K } ( \pmb x ) = K C _ { \pmb \theta , 1 } ( \pmb x ) .\tag{2}
$$

Let $\delta \in ( 0 , 1 )$ denote the target failure probability, so the desired reliability is $1 - \delta$

## 2.2 Complete-Solution Sampling

In a common pool of composite runs, joint success implies that each subproblem is covered, while separately correct subproblem answers may occur in different runs (Prop. 2). If standalone and composite subproblem marginals agree at $a _ { 1 } , a _ { 2 } \in ( 0 , 1 )$ , with independence within and across runs, then for $K \geq 2$

$$
\begin{array} { c } { 1 - ( 1 - a _ { 1 } a _ { 2 } ) ^ { K } < [ 1 - ( 1 - a _ { 1 } ) ^ { K } ] } \\ { \cdot [ 1 - ( 1 - a _ { 2 } ) ^ { K } ] . } \end{array}\tag{3}
$$

One run must solve both subproblems, whereas separate coverage can use different runs. This sampling constraint also applies to other stochastic solvers; it does not equate token costs across prompt formats. Under the additional assumptions of exponentially declining complete-run accuracy and linear generation cost (Zhang et al., 2025b), complete restarts require exponentially growing generation cost at fixed reliability. Appendix B.8 gives the product comparison, numerical example, and scaling calculation. We next examine the reliability and cost of retaining intermediate work.

## 3 Reliability and Cost of State Reuse

The reuse loop maintains one accepted state at a time. We analyze the probability that every retained state is correct, the number of executed attempts, and the resulting computational cost.

## 3.1 Local Retries and Reliable Completion

At step t, generate a candidate from the retained state $s _ { t - 1 } ^ { \check { \check { \mathbf { \alpha } } } }$ and check the state to be passed forward. Let $\nu _ { t }$ indicate acceptance and $\mathcal { G } _ { t }$ its correctness. Retain the first accepted state; if all R attempts are rejected, terminate the entire task. Conditional on the complete history at entry to each reached step, attempts are independent and identically distributed, including at incorrect inputs. Rejected candidates do not change the state or retry law; different steps may depend on earlier accepted results. Checking an extracted answer does not establish that a retained reasoning trace is also correct.

At a correct entry history s, let $\rho _ { t } ( s )$ be the rejection probability and $w _ { t } ( s )$ the probability of incorrect acceptance in one attempt. Assume $\rho _ { t } ( s ) \leq \rho \in [ 0 , 1 )$ and accepted-state error $\alpha _ { t } ( s ) \leq \alpha \in [ 0 , 1 )$ at every correct entry history. Conditioning on the first acceptance gives

$$
\alpha _ { t } ( R ; s ) = { \frac { w _ { t } ( s ) } { 1 - \rho _ { t } ( s ) } } = \alpha _ { t } ( s ) .\tag{4}
$$

Accepted-state precision is $1 - \alpha _ { t } ( s )$ ; it does not improve with $R .$ The local trajectory fails if the retry budget is exhausted or an incorrect candidate is accepted:

$$
\begin{array} { r l } { \beta _ { t } ( R ; s ) = \rho _ { t } ( s ) ^ { R } } & { { } } \\ { + \left( 1 - \rho _ { t } ( s ) ^ { R } \right) \alpha _ { t } ( s ) . } \end{array}\tag{5}
$$

More retries reduce exhaustion; improving accepted-state precision requires a different generation or checking rule. Let $H _ { t }$ mean that the first t steps complete with every retained state correct, with $H _ { 0 }$ certain. Assume $H _ { N }$ yields the target answer; later repair can also succeed outside $H _ { N }$

Theorem 1 (Trajectory reliability) . Under the preceding assumptions,

$$
\operatorname* { P r } ( H _ { N } ) \geq [ ( 1 - \rho ^ { R } ) ( 1 - \alpha ) ] ^ { N } .\tag{6}
$$

An admissible process attains equality for every $N , R .$ For $0 < \rho < 1$ , a finite common cap guarantees $1 - \delta$ over this class if and only if

$$
\alpha < 1 - ( 1 - \delta ) ^ { 1 / N } .\tag{7}
$$

Each correct prefix advances correctly with conditional probability at least $( 1 - \rho ^ { R } ) ( 1 - \alpha )$ ; the chain rule gives the product without independence between steps. A homogeneous process with irreversible errors after acceptance attains it. At equality in $\operatorname { E q . } \left( 7 \right)$ , the target is approached only with unbounded retries; for $\rho = 0$ , equality is allowed and one attempt suffices. This is a worst-case guarantee, not a necessary precision threshold for every system that can repair errors. Appendix B.10.1 gives the proof, exact minimum cap, and the simpler sufficient allocation $N \alpha < \delta$ used in Prop. 4.

## 3.2 Expected Work under the Stopping Rule

We measure work by W, the number of executed candidate attempts. The pathwise cap $W \le N R$ reserves the full budget at every step. Actual work can be smaller because a step ends as soon as a candidate passes. Let $\mathcal { P } _ { N , R }$ contain all processes following the preceding retry and stopping rules for at most N steps, with arbitrary history-dependent rejection probabilities and no correctness requirement. If acceptance were at least $h > 0$ at every reached state, the geometric waiting-time bound would give $\mathbb { E } [ W ] \leq N / h$ directly. After an incorrect acceptance, the process may reach states where the correct-prefix acceptance bound no longer holds. The next theorem includes work at these states.

Theorem 2 (Worst-case expected attempts) . For every pair of integers $N , R \geq 1$

$$
\operatorname* { s u p } _ { P \in \mathcal { P } _ { N , R } } \mathbb { E } _ { P } [ W ] = \Theta \left( N + \frac { N R } { \log ( N + 1 ) } \right) ,\tag{8}
$$

with positive implicit constants independent of both the process length N and the retry cap R.

Frequently rejected states are expensive to leave but likely to exhaust the cap, ending the run. The proof uses this tradeoff to bound their expected visits; a homogeneous process gives the matching lower bound (Appendix B.10.3). That lower bound concerns the unrestricted class, not every reliable process. The stopping pattern is related to classical start-up demonstration tests (Gera, 2013). Appendix B.10.6 gives the exact dynamic recursion and separates this uniform bound from fixed-law waiting-time calculations. Together with Thm. 1, this yields reliable completion with an ${ \cal O } ( \log N )$ cap and linear expected work. Here the target failure probability is fixed, acceptance at correct prefixes has a positive constant lower bound, and accepted-state error decreases with chain length as specified in Cor. 1.

Example: reserved budget and executed work. For 100 steps, a cap of 12 attempts reserves at most 1,200 attempts. Suppose every reached state has rejection probability $1 / 2$ , and every correct prefix has accepted-state error at most 0.0002. The reliability bound exceeds 0.956, while each reached step uses fewer than two attempts in expectation, giving fewer than 200 over the task. The probabilities in this example are hypothetical. The example distinguishes the reserved budget from expected work; Thm. 2 additionally permits arbitrary rejection probabilities after an incorrect acceptance.

## 3.3 Checking Precision and Computation

As a chain grows, the reliability bound requires a smaller error rate among accepted states. Reaching this precision may make checking more expensive or cause it to reject more candidates. Let $h _ { N }$ lower-bound acceptance probability at every correct prefix, and let $g _ { N }$ uniformly bound per-attempt cost, including generation, checking, and state processing at every reached state.

Corollary 1 (Precision, acceptance, and cost) . Fix $\delta \in ( 0 , 1 )$ and $c \in [ 0 , 1 )$ . If correct prefixes have acceptance probability at least $h _ { N } \in ( 0 , 1 ]$ and accepted-state error at most cδ/N, choose

$$
R _ { N } = \left\lceil \frac 1 { h _ { N } } \log \frac { N } { ( 1 - c ) \delta } \right\rceil .\tag{9}
$$

Then $\Pr ( H _ { N } ) \geq 1 - \delta$ and

$$
\mathbb { E } [ W ] = O ( N / h _ { N } ) , \qquad C _ { \mathrm { v s r } } = O ( N g _ { N } / h _ { N } ) .\tag{10}
$$

The attempt order $N / h _ { N }$ is attained within this reliable class for this cap.

The cap controls exhaustion; Thm. 2 bounds all executed attempts, including those after an incorrect acceptance (Appendix B.10.4). For constant $h _ { N }$ , expected attempts are linear, whereas the deterministic cost cap remains O(N log $\textit { N g } _ { \mathrm { N } } )$ . Since $W \geq N \mathbf { 1 } _ { H _ { N } }$ and $\mathbb { E } [ W \mid H _ { N } ] \le \mathbb { E } [ W ] / ( 1 - \delta )$ , this conclusion is compatible with reliable completion, rather than relying only on cheap failures.

At a common precision target, the relevant cost scale is $g _ { N } / h _ { N }$ . Under independent binary checks, requiring unanimity can make acceptance polynomially rare, yielding $\Theta ( N ^ { 1 + \gamma }$ log N) expected cost for some $\gamma > 0$ when every check is executed. With an additional separation condition, majority checking keeps acceptance bounded away from zero and gives O(N log N) expected cost (Hoeffding, 1963). Appendix B.10.5 specifies the noise model and derives both cases; these repeatedly check one candidate rather than compare different answers by self-consistency.

The work bound relies on the execution rule: feedback-dependent retries or continuation after exhaustion can each use NR attempts. The experiments below use batch selection, multiple continuations, and fallback with different success criteria. They do not directly test these guarantees.

## 4 Experiments

We compare coverage across budgets, selection rules, and continuation prompts. The tasks cover composite mathematics, complete-answer arithmetic retries, and agent-score prediction.

## 4.1 Experimental Setup

Models and tasks. We evaluate DeepSeek-R1-Distill (1.5B, 7B, 8B, 14B) (Guo et al., 2025), Qwen3- Thinking (4B, 30B-A3B) (Team, 2025), and GPT-OSS (20B, 120B) (OpenAI, 2025). LoRe-Compo contains 250 pairs of distinct-subject MATH500 problems, using each of the 500 subproblems once (Zhang et al., 2025b; Hendrycks et al., 2021). The second subproblem is a separate mathematical problem that does not require the first answer. Arithmetic uses 50 problems for each base (7 or 10), operation (addition or multiplication), and digit count $n \in \{ 1 , \ldots , 1 0 \}$ , with 32 complete-answer attempts per problem.

Procedures and success criteria. Composite Pass@K requires both correct answers in one of K complete runs. The two-stage protocol instead generates M answers to $x _ { 1 } .$ , retains at most B candidates, and continues on $x _ { 2 } .$ . Oracle selects reference-matching answers; self-consistency (SC) uses the modal answer when its vote share reaches 0.5. If selection returns none, fallback candidates are retained for continuation. Both procedures generate whole subproblem answers without checking individual reasoning steps.

The reported second-subproblem success rate $A _ { 2 } ^ { \mathrm { r e t } }$ counts any correct answer to x among the retained continuations, including fallbacks and regardless of $x _ { 1 }$ correctness. Joint retained-path success would require both answers to be correct along one shared path, with all 250 pairs still in the denominator. Thus $A _ { 2 } ^ { \mathrm { r e t } }$ upper-bounds joint coverage but does not determine it. It measures coverage, not final-answer selection accuracy. The theoretical loop returns one first-accepted state and stops at exhaustion; the experimental protocol uses multiple continuations and fallback. Appendix C.2 defines these events.

Sampling, matching, and cost. All mathematical generations have a 10,240-token cap. A composite run shares this cap across two answers, whereas staged solving grants one cap per subproblem. LoRe-Compo uses family-specific decoding settings and direct prompt strings without model-specific chat templates. Decoding differs between Pass@1 and larger budgets, some Pass@1 values come from prior work, and cross-budget nesting is unspecified for the composite results. Answer matching uses numeric tolerance or normalized strings, with separate extraction rules for single and paired answers (Appendix C.1). We report recorded generated-token counts and success percentage points per thousand tokens. Equal attempt counts allow different generation budgets. The token counts exclude some checking and state processing costs and are not matched for whole-task reliability.

## 4.2 Complete-Solution Sampling on Composite Tasks

Table 2 reports joint composite Pass@K. The eight-model mean is 8.2% at K = 1 and 33.3% at the largest recorded budget (K = 32 for R1-1.5B and K = 128 otherwise). Coverage improves across these

Table 1: Second-subproblem success after selection. $A _ { 2 } ^ { \mathrm { r e t } }$ counts a correct $x _ { 2 }$ answer on any retained path, including fallbacks, regardless of $x _ { 1 }$ correctness. Oracle and SC share $M = 8 , B = 4$ . This continuation metric differs from the complete-solution joint-success criterion in Table 2.
<table><tr><td>Model</td><td> $A _ { 2 } ^ { \mathrm { r e t } } \left( \% \right)$ </td></tr><tr><td>Oracle</td><td>SC</td></tr><tr><td>DeepSeek-R1-1.5B</td><td>55.6 55.2</td></tr><tr><td>DeepSeek-R1-7B</td><td>61.6 61.6</td></tr><tr><td>DeepSeek-R1-8B</td><td>51.2 52.8</td></tr><tr><td>DeepSeek-R1-14B</td><td>60.4 62.0</td></tr><tr><td>Qwen3-4B</td><td>53.6 54.4</td></tr><tr><td>Qwen3-30B-A3B</td><td>58.4 58.0</td></tr><tr><td>GPT-OSS-20B</td><td>47.2 48.4</td></tr><tr><td>GPT-OSS-120B</td><td>62.4 62.4</td></tr></table>

Table 2: Composite Pass@K under Independent Sampling. Success requires both correct answers in one of K complete runs. DeepSeek $K = 1$ values are from Zhang et al. (2025b); other cells are reported aggregates from this study. Superscripts give percentage-point changes from $K = 1 ;$ dashes indicate unreported cells. Decoding varies with K, and cross-budget sample nesting is unspecified.
<table><tr><td rowspan="2">Model</td><td colspan="4">Pass@ K Coverage (%)</td></tr><tr><td>1</td><td>8</td><td>32</td><td>128</td></tr><tr><td>DeepSeek-R1-1.5B</td><td>4.0</td><td> $1 7 . 2 ^ { \uparrow 1 3 . 2 }$ </td><td> $2 8 . 4 ^ { \uparrow 2 4 . 4 }$ </td><td></td></tr><tr><td>DeepSeek-R1-7B</td><td>2.4</td><td> $1 5 . 2 ^ { \uparrow 1 2 . 8 }$ </td><td> $2 6 . 0 \AA ^ { \uparrow 2 3 . 6 }$ </td><td> $3 2 . 4 ^ { \uparrow 3 0 . 0 }$ </td></tr><tr><td>DeepSeek-R1-8B</td><td>13.2</td><td> $3 0 . 4 ^ { \uparrow 1 7 . 2 }$ </td><td> $3 4 . 0 \AA ^ { \uparrow 2 0 . 8 }$ </td><td> $3 7 . 2 ^ { \uparrow 2 4 . 0 }$ </td></tr><tr><td>DeepSeek-R1-14B</td><td>2.0</td><td> $1 4 . 4 ^ { \uparrow \uparrow 1 2 . 4 }$ </td><td> $2 1 . 2 ^ { \uparrow \uparrow 1 9 . 2 }$ </td><td> $2 5 . 2 ^ { \uparrow 2 3 . 2 }$ </td></tr><tr><td>Qwen3-4B</td><td>4.8</td><td> $2 7 . 2 ^ { \uparrow 2 2 . 4 }$ </td><td></td><td> $3 8 . 0 \AA ^ { \uparrow 3 3 . 2 }$ </td></tr><tr><td>Qwen3-30B-A3B</td><td>2.8</td><td> $1 4 . 8 ^ { \uparrow 1 2 . 0 }$ </td><td> $2 7 . 2 ^ { \uparrow 2 4 . 4 }$ </td><td> $3 1 . 6 ^ { \uparrow 2 8 . 8 }$ </td></tr><tr><td>GPT-OSS-20B</td><td>16.8</td><td> $3 4 . 8 ^ { \uparrow 1 8 . 0 }$ </td><td> $3 3 . 2 ^ { \uparrow 1 6 . 4 }$ </td><td> $3 5 . 2 ^ { \uparrow 1 8 . 4 }$ </td></tr><tr><td>GPT-OSS-120B</td><td>19.6</td><td> $3 6 . 8 ^ { \uparrow 1 7 . 2 }$ </td><td> $3 8 . 0 ^ { \uparrow 1 8 . 4 }$ </td><td> $3 8 . 4 ^ { \uparrow 1 8 . 8 }$ </td></tr></table>

endpoint configurations, but the gains and model ordering vary. Qwen3-4B starts at 4.8%, below R1-8B at 13.2%, and reaches 38.0% compared with 37.2% at $K = 1 2 8$ . GPT-OSS-120B has the highest Pass@1 and Pass@128, at 19.6% and 38.4%. Model rankings depend on the sampling budget.  
GPT-OSS-20B changes from 34.8% at K = 8 to 33.2% at K = 32. Because nesting and decoding are not held fixed across all configurations, this decrease does not show that adding samples to an unchanged pool reduces coverage. The table reports how often a complete run answers both subproblems correctly. Measuring unused partial success requires the statistic in Appendix C.2.

## 4.3 Candidate Selection and Retained Context

Intermediate-answer selection. Table 1 compares Oracle and SC at M = 8, B = 4 using the same second-subproblem criterion. The four-DeepSeek SC mean is 57.9%; across all eight models, SC and Oracle differ by at most 1.6 percentage points. Here, Oracle and SC yield similar second-subproblem success despite using different selection rules. These scores include fallback continuations, so they do not show that the selectors retain equally accurate first answers. The subproblems are mathematically separate, so a wrong first answer need not prevent success on the second.

The reported SC-to-Pass@8 generated-token ratios are approximately 1.01–1.12 for R1-1.5B, R1-7B, R1-14B, and GPT-OSS-20B, and 1.4–1.5 for the other models. These procedures use different success criteria, so the ratios cannot establish savings at matched whole-task success rates.

Continuation-prompt configurations. At the same continuation budget, the four-DeepSeek means are 36.2% for no\_context, 57.2% for answer\_only, and 58.3% for full\_context (Table 3). The answer-only prompt scores 21.0 percentage points above no context; full context scores another 1.1 points higher. This small gap does not establish that the two prompts are equivalent. The answer-only mean matches the Oracle mean in Table 1; the SC mean reported above is a separate result. The available prompt-comparison summary does not identify the selector configuration.

Table 3: Context for the Second Subproblem. Four-DeepSeek means at the same continuation budget. ${ \hat { A } } _ { 2 } ^ { \mathrm { r e t } }$ counts second-subproblem success, including fallbacks. Change is the relative percentage increase over no\_context. Strategies supply no first-subproblem context, its answer only, or its reasoning and answer.
<table><tr><td>Strategy</td><td> $A _ { 2 } ^ { \mathrm { r e t } }$ </td><td>Change</td></tr><tr><td>no_context</td><td>36.2%</td><td>0%</td></tr><tr><td>answer_only</td><td>57.2%</td><td>+58%</td></tr><tr><td>full_context</td><td>58.3%</td><td>+61%</td></tr></table>

The prompts differ in instructions as well as content; these scores compare complete configurations and do not isolate

the contribution of the first answer. The answer-only protocol retains the same extracted answer assessed during selection, whereas full context also retains an unchecked reasoning trace truncated to fit the context budget.

How candidate and continuation budgets trade off. Table 4 compares four tested allocations. Its (8, 4) success column reproduces the Oracle column of Table 1; the aggregate table alone does not identify the selector used for every other allocation. The eight-model mean rises from 52.8% at 36.4K tokens for (4, 2) to 56.3% at 72.3K for (8, 4), a 3.5-point gain for approximately twice the generation. Moving to (16, 8) gives 57.4% at 144.5K, another 1.1 points for approximately twice the tokens again. The second budget expansion yields a smaller gain in success. We use (8, 4) as an intermediate reference; (4, 2) has higher success per thousand generated tokens.

The intermediate allocations separate changes in candidate count from changes in continuation count. At B = 2, increasing M from 4 to 8 raises mean success by 1.7 points, from 52.8% to 54.5%, for 27.0K additional tokens. At M = 8, increasing B from 2 to 4 raises mean success by 1.8 points, from 54.5% to 56.3%, for 8.9K additional tokens. Increasing the continuation budget therefore gives a similar mean gain with fewer additional generated tokens in this comparison.

The model-level responses also vary. At M = 8, raising B from 2 to 4 changes R1-8B from 44.8% to 51.2%, while GPT-OSS-120B stays at 62.4%. GPT-OSS-20B changes from 48.4% to 47.2%; without nested pools this cannot be attributed to keeping more candidates. Across (4, 2) to (8, 4), R1-8B gains 10.0 points, compared with 0.4 points each for R1-7B and GPT-OSS-120B. The gains therefore depend on both the model and the allocation of the sampling budget.

Resolution of the comparisons. With 250 pairs per model, one changed pair corresponds to 0.4 percentage points. The largest SC–Oracle difference is a net difference of four successful pairs. Paired uncertainty is unavailable for these comparisons and for the 1.1-point full-context increment, so we interpret the figures as descriptive comparisons of the evaluated configurations.

Table 4: Budget Allocation Sweep. M is the first-subproblem candidate budget and B the cap retained for continuation. Succ. is second-subproblem success (%), including fallbacks, regardless of first-answer correctness. Tokens reports recorded counts; Eff. divides Succ. by Tokens in thousands. Average-row efficiencies divide mean success by mean tokens; they are not means of model-level efficiencies. Superscripts give percentage-point differences from (4, 2). Orange marks the reference configuration (8, 4), not an efficiency optimum; blue marks configurations with higher success and a larger measured token cost.
<table><tr><td rowspan="2">Model</td><td colspan="3">M=4, B=2</td><td colspan="3">M=8, B=2</td><td colspan="3"> $\scriptstyle M = 8 , B = 4$ </td><td colspan="3">M=16, B=8</td></tr><tr><td>Succ.</td><td>Tokens</td><td>Eff.</td><td>Succ.</td><td>Tokens</td><td>Eff.</td><td>Succ.</td><td>Tokens</td><td>Eff.</td><td>Succ.</td><td>Tokens</td><td>Eff.</td></tr><tr><td>DeepSeek-R1-1.5B</td><td>53.6%</td><td>25.1K</td><td>2.13</td><td>54.0%</td><td>46.8K</td><td>1.15</td><td> $^ { 1 2 . 0 } 5 5 . 6 \%$ </td><td>50.0K</td><td>1.11</td><td>13.256.8%</td><td>100.6K</td><td>0.56</td></tr><tr><td>DeepSeek-R1-7B</td><td>61.2%</td><td>29.2K</td><td>2.09</td><td>61.2%</td><td>52.8K</td><td>1.16</td><td>1.461.6%</td><td>57.4K</td><td>1.07</td><td>61.2%</td><td>115.0K</td><td>0.53</td></tr><tr><td>DeepSeek-R1-8B</td><td>41.2%</td><td>59.6K</td><td>0.69</td><td>44.8%</td><td>100.7K</td><td>0.44</td><td>↑10.051.2%</td><td>116.7K</td><td>0.44</td><td>↑12.453.6%</td><td>233.0K</td><td>0.23</td></tr><tr><td>DeepSeek-R1-14B</td><td>59.6%</td><td>32.6K</td><td>1.83</td><td>60.0%</td><td>57.7K</td><td>1.04</td><td> $^ { \dag . 8 } 6 0 . 4 \%$ </td><td>63.7K</td><td>0.95</td><td>↑1.61.2%</td><td>128.1K</td><td>0.48</td></tr><tr><td>Qwen3-4B</td><td>47.2%</td><td>58.7K</td><td>0.80</td><td>48.0%</td><td>100.3K</td><td>0.48</td><td> $^ { \dag 6 . 4 } 5 3 . 6 \%$ </td><td>117.6K</td><td>0.46</td><td>53.2%</td><td>234.7K</td><td>0.23</td></tr><tr><td>Qwen3-30B-A3B</td><td>54.4%</td><td>55.4K</td><td></td><td>0.9857.2%</td><td>95.2K</td><td>0.60</td><td>14.058.4%</td><td>110.4K</td><td>0.53</td><td>16.060.4%</td><td>219.1K</td><td>0.28</td></tr><tr><td>GPT-OSS-20B</td><td>42.8%</td><td>15.9K</td><td>2.69</td><td>48.4%</td><td>28.2K</td><td>1.72</td><td> $^ { \dag 4 . 4 } 4 7 . 2 \%$ </td><td>33.2K</td><td>1.42</td><td>17.650.4%</td><td>68.3K</td><td>0.74</td></tr><tr><td>GPT-OSS-120B</td><td>62.0%</td><td>14.4K</td><td>4.31</td><td>62.4%</td><td>25.4K</td><td>2.46</td><td> $^ { \dag . 4 } 6 2 . 4 \%$ </td><td>29.0K</td><td>2.15</td><td>62.4%</td><td>57.3K</td><td>1.09</td></tr><tr><td>DeepSeek Avg</td><td>53.9%</td><td>36.6K</td><td>1.47</td><td>55.0%</td><td>64.5K</td><td>0.85</td><td>13.357.2%</td><td>71.9K</td><td>0.80</td><td>14.358.2%</td><td>144.2K</td><td>0.40</td></tr><tr><td>All Avg</td><td>52.8%</td><td>36.4K</td><td>1.45</td><td>54.5%</td><td>63.4K</td><td>0.86</td><td>13.56.3%</td><td>72.3K</td><td>0.78</td><td>↑4.657.4%</td><td>144.5K</td><td>0.40</td></tr></table>

![](images/b9f7a2041114e59164fde5337dda752699f05946d0a2b2fc5df4f27ac556ae4d.jpg)

![](images/b692952156fe3b66ff160e8892897017edc5355793a7eda47c6680893fe85f5f.jpg)

![](images/95756d525a492df7d871bb774fbca08acc7698bb48ea6b87325f084874555a9c.jpg)

![](images/6d5d1bb83b452a6fa3801cd9ab877f7d28e54bc9f36f4edb39d45709dca1f02c.jpg)  
Figure 2: Unsolved Fraction after 32 Attempts. Each cell reports $\hat { E } ( 3 2 )$ , the fraction of 50 problems unsolved after 32 attempts; n is operand digit count. Panels show Base 10 addition/multiplication and Base 7 addition/multiplication. All Base 10 addition problems are solved; Base 7 multiplication has the largest mean unsolved fraction. For R1-7B, the latter rises from 0.06 at $n = 3$ to 0.74 at $n = 4$ . GPT-OSS-120B shows nonmonotonic behavior at low digit counts under the recorded sampling protocol.

## 4.4 Finite-Budget Arithmetic Retries

The arithmetic experiments ask how many initially missed problems can be solved with more attempts. Here the candidate is an entire arithmetic answer, not a checked digit-level state. We write ${ \hat { E } } ( R )$ for the fraction of problems with no correct answer among their first R attempts; empirical Pass@R is $1 - { \hat { E } } ( R )$ . The checkpoints use prefixes of the same pool of 32 complete-answer attempts.

Across 320 model–base–operation–digit conditions, 235 have no unsolved problems after 32 attempts and 85 have a nonzero unsolved fraction (Figure 2). All 80 Base 10 addition conditions have zero unsolved fraction; Base 7 multiplication has a mean of 0.40, with 33 of 80 conditions above 0.5. The dependence on digit count varies across models, including nonmonotonic behavior at low digit counts for GPT-OSS-120B. Large retry gains appear around Pass@1 values of 0.05–0.35 (Figure 3), where repeated attempts can cover many initially missed problems.

First-attempt accuracy alone does not determine the gain from retries. On Base 7 addition at n = 3, R1-1.5B rises from Pass@1 of 0.08 to Pass@32 of 0.98, leaving $\hat { E } ( 3 2 ) = 0 . 0 2$ On Base 7 multiplication at $n \geq 7$ , the same model has Pass@1 of zero and $\hat { E } ( 3 2 ) = 1$ . R1-14B at $n = 6$ on Base

![](images/e61b1e2a7a3184b5170223067fe1105f2c3f8102f9ec3e44fd0a50d319240d2d.jpg)  
Figure 3: Gain from Additional Attempts. Each point is a combination of model, base, operation, and digit count. The gain is Pass@32 minus Pass@1; the dashed line is its maximum $1 - \operatorname { P a s s } \ @ 1$ . Large observed gains occur around Pass@1 values of 0.05–0.35 in these evaluated arithmetic conditions.

7 multiplication starts at Pass@1 of 0.14 but solves 84% of the evaluated problems with the same total sampling budget of 32 complete-answer attempts (Figure 8).

A low Pass@1 can therefore accompany either substantial retry gains or no observed success within the available budget. The observations concern finite pools of complete answers: $\hat { E } ( 3 2 ) > 0$ does not identify a permanent failure floor, and digit count is not the number of checked intermediate states. Appendix D and Appendix E give the curves, mixture fits, and token counts.

## 4.5 Agent Score Prediction

We also examine whether a repeated-attempt curve can predict later agent scores from early observations. The retry curve approaches a fitted score ceiling with diminishing gains; a sigmoid in log time provides an alternative model of bounded score growth (Appendix F.1).

EdgeBench (Zhu et al., 2026) supplies complete observations at 2, 4, 6, 8, 10, and 12 hours for 252 task–model pairs across five models and 51 tasks. For each target pair, we fit its 2-hour and 4-hour scores using category-level priors from other tasks. Priors exclude all model trajectories of the target task.

Median relative errors for the retry and sigmoid curves are 3.8% and 3.9% at 6 hours, increasing to 10.7% and 10.9% at 12 hours. Relative-error summaries use 248 pairs with positive observed scores at each prediction time, while all 252 pairs receive predictions. Table 6 reports all four horizons and the fractions below 10% and 20% relative error. Neither curve has the lower median error at every horizon; the retry curve has at least as many predictions below 10% error and more below 20% at each horizon. The two curves have similar prediction errors, although their priors constrain different parameters. The forecast does not identify internal checks or state reuse, or test savings from early stopping.

![](images/829b31582da9510168a90c0772dc6164ad9771ff370edcdb46a7047277066e7b.jpg)  
Figure 4: Agent Score Prediction at 12 Hours. Each point is a task–model pair. Predictions use its scores at 2 and 4 hours and parameter estimates from other tasks in the same category, excluding the target task. The retry curve models diminishing gains from repeated attempts; the comparison curve is sigmoid in log time (Appendix F.1). All 252 pairs are plotted; median relative errors are 10.7% and 10.9%, respectively, over 248 pairs with positive observed scores. Dashed and dotted lines mark perfect prediction and 20% bands.

## 5 Related Work

Reasoning compositionality and long-horizon reliability. Prior work documents difficulties composing intermediate results (Dziri et al., 2023), falling success with task duration (Kwa et al., 2025), and failures at high problem complexity (Shojaee et al., 2025). Zhang et al. (2025b) formulate compute and accuracy laws, measure compositionality gaps, and introduce SFT-Compo as a training intervention. We use their LoRe-Compo setting to study test-time sampling and state reuse.

Decomposition, search, and local error correction. Hierarchical reinforcement learning composes subgoals through options, MAXQ, and skill chaining (Sutton et al., 1999; Dietterich, 2000; Konidaris and Barto, 2009). Least-to-Most prompting uses earlier answers to solve later subproblems (Zhou et al., 2023); Tree of Thoughts searches and backtracks over partial states (Yao et al., 2023), and Graph of Thoughts combines and refines intermediate results (Besta et al., 2024). Chain-in-Tree chains routine steps and reuses generated children when branching becomes necessary (Li, 2026). Recursive and agentic inference systems similarly pass subcall outputs to later computations (Zhang et al., 2025a; Alzu’bi et al., 2026; Yu et al., 2025; Nielsen et al., 2025). MAKER (Meyerson et al., 2025) combines extreme decomposition, error detection, and first-to-k-ahead voting, with asymptotic foundations in Meyerson and Qiu (2025). Under its voting model, MAKER derives logarithmic voting budgets and Θ(N log N) total expected cost for fixed task reliability, assuming local success probability above one half. Our finite-cap analysis bounds work across correct and incorrect states under first acceptance and global stopping; its cost guarantee requires precision and acceptance bounds only at correct prefixes.

Stopped testing and restart analysis. Start-up demonstration tests study total-success and consecutivefailure stopping criteria, including dependent trials (Gera, 2013; Balakrishnan et al., 2014). Our result bounds expected work uniformly over the specified retry class and relates it to correctness at retained prefixes. Classical restart scheduling optimizes repeated complete runs (Luby et al., 1993); the present cap and stopping rule are fixed. Appendix B.10.6 gives the comparison and exact work recursion.

Verification and test-time sampling. Independent sampling improves candidate coverage (Brown et al., 2024), while adaptive test-time computation allocates effort according to task difficulty and verification quality (Snell et al., 2025; Setlur et al., 2025; Misaki et al., 2025). Self-consistency uses answer agreement as a proxy for correctness (Wang et al., 2022); process supervision and reward models assess intermediate reasoning (Uesato et al., 2022; Lightman et al., 2024; Zheng et al., 2025b). ProcessBench and Socratic-PRMBench document limits of identifying and evaluating reasoning errors (Zheng et al., 2025a; Li et al., 2025). Formal verification and autoformalization provide stronger, domain-specific checks (Leino, 2010; Szegedy, 2020). Our empirical Oracle is reference-answer matching, and SC is thresholded modal-answer selection; neither verifies a full reasoning trace. Labiad et al. (2026) train a search policy to guide a frozen answer model and show that gains over repeated sampling depend on decoding and exploration choices. Their comparisons match the answer-generation allocation; concept generation adds a small separate inference cost. Our arithmetic experiments measure repeated-answer coverage under fixed decoding.

Retained context and execution cost. Atom of Thoughts decomposes and contracts problems into self-contained states designed to preserve answer equivalence (Teng et al., 2025). MEM1 learns a compact recurrent state and discards redundant interaction history (Zhou et al., 2025). These methods address what to retain. Our analysis specifies the correctness and continuation conditions that the retained state must satisfy. Using counterfactual error histories, Sinha et al. (2025) show that earlier mistakes can impair later reasoning, with thinking mitigating this effect in their experiments. Context length and information placement can also affect model behavior (Liu et al., 2024). Adaptive sequential Monte Carlo avoids recomputing shared prefixes by reordering KV caches and particle tensors during resampling (Wang et al., 2026). This reduces execution cost without establishing the correctness of retained reasoning. Appendix A.1 gives a separate qualitative recurrence. The reliability proof instead controls errors at correct prefixes and bounds the probability of retaining an incorrect intermediate state.

## 6 Discussion

Acceptance and correctness. At a fixed retained state, more attempts increase the chance that a candidate passes the check. With the same candidate distribution and check, they leave the error rate among accepted candidates unchanged. Reliable continuation therefore requires both a sufficiently accurate check and a state that contains the information needed by later steps. Checking an answer and retaining its full reasoning trace impose different correctness requirements on what is passed forward.

Executed work and total cost. The local cap reserves enough attempts for difficult steps, while acceptance often allows the procedure to advance earlier. With constant acceptance probability and sufficient precision, the stopped loop uses linearly many attempts in expectation. The cost of generating candidates, checking them, and processing retained states must then be counted at every reached state. A stricter check can improve precision while increasing the cost of each attempt or reducing the chance of acceptance. Both effects enter the comparison between checking cost and avoided generation.

Choosing a sampling configuration. The arithmetic results show that the benefit of more attempts varies across tasks, even when first-attempt accuracy is low. On the paired mathematics problems, the answer-only prompt scores closer to full context than to no context, although the comparison changes instructions as well as retained information. Larger candidate and continuation budgets also give uneven gains across models. These observations help describe the configurations evaluated here. Establishing computational savings from reuse still requires a comparison at the same whole-task success rate, with generation, checking, and retained-state processing all included.

## 7 Conclusion

Reliable reuse depends on which candidates a check accepts and how later steps use the retained state. Under the stated retry and stopping rules, we give conditions for correct completion and bounds on executed attempts. Additional retries reduce exhaustion, while the candidate distribution and check determine accepted-state precision. Total cost depends on per-attempt cost and acceptance probability.

The experiments document second-subproblem coverage across continuation prompts and variable gains from repeated arithmetic answers. They describe practical sampling choices but do not establish the long-chain guarantees. Assessing reuse on a new task requires measuring retained-state error, continuation reliability, and total cost at matched whole-task success.

## Limitations

Execution and checking assumptions. The work theorem assumes conditionally independent, identically distributed retries at each retained state and termination when the local budget is exhausted. Feedback that changes the next attempt and fallback after exhaustion require separate analysis. The binary-check examples further assume that check outcomes are independent conditional on candidate correctness. The experiments do not directly test these execution assumptions.

Checks and retained information. Agreement can select common correct answers, but shared errors can also receive many votes (Wang et al., 2022). Similar SC and Oracle scores on the second subproblem do not tell us how often either selector keeps a wrong first answer. The correctness condition applies to everything passed forward: checking an extracted answer does not verify a reasoning trace retained alongside it. Process-verifier benchmarks document errors in learned checks; formal verification uses task-specific correctness criteria (Zheng et al., 2025a; Li et al., 2025; Leino, 2010; Szegedy, 2020).

Decomposition and empirical scope. The theorem takes the decomposition and correctness criterion as inputs. A smaller step can be easier to check but requires more checks; a larger step may retain information whose correctness is harder to establish. The LoRe-Compo subproblems are mathematically separate, and arithmetic retries whole answers, so neither experiment estimates uniform reliability at the prefixes of a long dependent chain. Open-ended tasks may need different representations and task-specific checks before the sufficient conditions become applicable. The available composite aggregates do not identify acceptedstate error or fallback contributions; prompt differences also prevent isolating an information-reuse effect.

Cost and auxiliary models. The total-cost bound includes generation, checking, and state processing at every reached state. Growing contexts or more accurate checks can increase that cost and must remain explicit in the budget expression. When complete solutions are already reliable, the extra checking may outweigh the work saved by reuse. The appendix’s qualitative error-propagation and mixture models describe possible behaviors; finite retry pools cannot identify a permanent capability boundary. Agentscore prediction also does not identify internal verification or establish a compute-allocation benefit.

## AI Use Statement

We used generative AI tools to assist manuscript writing, revision, and the development and checking of mathematical derivations. The language models used to generate candidate solutions and continuations, together with the answer-selection procedures, are part of the experimental protocols described in Section 4. We take responsibility for the final manuscript and its scientific claims.

## References

Salaheddin Alzu’bi, Baran Nama, Arda Kaz, Anushri Eswaran, Weiyuan Chen, Sarvesh Khetan, Rishab Bala, Tu Vu, and Sewoong Oh. Roma: Recursive open meta-agent framework for long-horizon multi-agent systems. arXiv preprint arXiv:2602.01848, 2026.

N. Balakrishnan, M. V. Koutras, and F. S. Milienos. Start-up demonstration tests: models, methods and applications, with some unifications. Applied Stochastic Models in Business and Industry, 30(4): 373–413, 2014. doi: 10.1002/asmb.2040.

Richard E Barlow and Frank Proschan. Mathematical theory of reliability. SIAM, 1996.

Maciej Besta, Nils Blach, Ales Kubicek, Robert Gerstenberger, Michal Podstawski, Lukas Gianinazzi, Joanna Gajda, Tomasz Lehmann, Hubert Niewiadomski, Piotr Nyczyk, et al. Graph of thoughts: Solving elaborate problems with large language models. In AAAI Conference on Artificial Intelligence (AAAI), volume 38, pages 17682–17690, 2024.

Bradley Brown, Jordan Juravsky, Ryan Ehrlich, Ronald Clark, Quoc V Le, Christopher Ré, and Azalia Mirhoseini. Large language monkeys: Scaling inference compute with repeated sampling. arXiv preprint arXiv:2407.21787, 2024.

Thomas G Dietterich. Hierarchical reinforcement learning with the maxq value function decomposition. Journal ofartificial intelligence research, 13:227–303, 2000.

Nouha Dziri, Ximing Lu, Melanie Sclar, Xiang Lorraine Li, Liwei Jiang, Bill Yuchen Lin, Sean Welleck, Peter West, Chandra Bhagavatula, Ronan Le Bras, et al. Faith and fate: Limits of transformers on compositionality. Advances in neural information processing systems, 36:70293–70332, 2023.

Ran El-Yaniv and Yair Wiener. On the foundations of noise-free selective classification. Journal of Machine Learning Research, 11(53):1605–1641, 2010. URL https://www.jmlr.org/papers/ v11/el-yaniv10a.html.

Amos E. Gera. A start-up demonstration procedure involving dependent tests. Statistics & Probability Letters, 83(10):2191–2196, 2013. doi: 10.1016/j.spl.2013.05.036.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. In Advances in Neural Information Processing Systems (NeurIPS), 2021.

Wassily Hoeffding. Probability inequalities for sums of bounded random variables. Journal of the American Statistical Association, 58(301):13–30, 1963. doi: 10.1080/01621459.1963.10500830.

George Konidaris and Andrew Barto. Skill discovery in continuous reinforcement learning domains using skill chaining. Advances in neural information processing systems, 22, 2009.

Thomas Kwa, Ben West, Joel Becker, Amy Deng, Katharyn Garcia, Max Hasin, Sami Jawhar, Megan Kinniment, Nate Rush, Sydney Von Arx, Ryan Bloom, Thomas Broadley, Haoxing Du, Brian Goodrich, Nikola Jurkovic, Luke Harold Miles, Seraphina Nix, Tao Lin, Neev Parikh, David Rein, Lucas Jun Koba Sato, Hjalmar Wijk, Daniel M. Ziegler, Elizabeth Barnes, and Lawrence Chan. Measuring ai ability to complete long tasks. arXiv preprint arXiv:2503.14499, 2025.

Ismail Labiad, Matthieu Kowalski, Marc Schoenauer, Rémi Munos, and Julia Kempe. Beyond repeated sampling: Learning search policies for llm reasoning. arXiv preprint arXiv:2609.26704, 2026.

K. Rustan M. Leino. Dafny: An automatic program verifier for functional correctness. In International Conference on Logicfor Programming Artificial Intelligence and Reasoning, pages 348–370. Springer, 2010.

Xiang Li, Haiyang Yu, Xinghua Zhang, Ziyang Huang, Shizhu He, Kang Liu, Jun Zhao, Fei Huang, and Yongbin Li. Socratic-prmbench: Benchmarking process reward models with systematic reasoning patterns. arXiv preprint arXiv:2505.23474, 2025.

Xinzhe Li. Chain-in-tree: Back to sequential reasoning in llm tree search. In Findings of the Association for Computational Linguistics: ACL 2026, pages 4365–4392, 2026.

Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In The Twelfth International Conference on Learning Representations, 2024.

Nelson F Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. Lost in the middle: How language models use long contexts. Transactions of the association for computational linguistics, 12:157–173, 2024.

Michael Luby, Alistair Sinclair, and David Zuckerman. Optimal speedup of Las Vegas algorithms. Information Processing Letters, 47(4):173–180, 1993. doi: 10.1016/0020-0190(93)90029-9.

Elliot Meyerson and Xin Qiu. Position: Scaling llm agents requires asymptotic analysis with llm primitives. In Forty-second International Conference on Machine Learning Position Paper Track, 2025.

Elliot Meyerson, Giuseppe Paolo, Roberto Dailey, Hormoz Shahrzad, Olivier Francon, Conor F Hayes, Xin Qiu, Babak Hodjat, and Risto Miikkulainen. Solving a million-step llm task with zero errors. arXiv preprint arXiv:2511.09030, 2025.

Kou Misaki, Yuichi Inoue, Yuki Imajuku, So Kuroki, Taishi Nakamura, and Takuya Akiba. Wider or deeper? scaling llm inference-time compute with adaptive branching tree search. CoRR, 2025.

Stefan Nielsen, Edoardo Cetin, Peter Schwendeman, Qi Sun, Jinglue Xu, and Yujin Tang. Learning to orchestrate agents in natural language with the conductor. arXiv preprint arXiv:2512.04388, 2025.

OpenAI. gpt-oss-120b & gpt-oss-20b model card, 2025. URL https://arxiv.org/abs/2508. 10925.

Amrith Setlur, Matthew Y. R. Yang, Charlie Snell, Jeremy Greer, Ian Wu, Virginia Smith, Max Simchowitz, and Aviral Kumar. e3: Learning to explore enables extrapolation of test-time compute for llms, 2025. URL https://arxiv.org/abs/2506.09026.

Parshin Shojaee, Iman Mirzadeh, Keivan Alizadeh, Maxwell Horton, Samy Bengio, and Mehrdad Farajtabar. The illusion of thinking: Understanding the strengths and limitations of reasoning models via the lens of problem complexity. arXiv preprint arXiv:2506.06941, 2025.

Akshit Sinha, Arvindh Arun, Shashwat Goel, Steffen Staab, and Jonas Geiping. The illusion of diminishing returns: Measuring long horizon execution in llms. arXiv preprint arXiv:2509.09677, 2025.

Charlie Victor Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. Scaling llm test-time compute optimally can be more effective than scaling parameters for reasoning. In International Conference on Learning Representations (ICLR), 2025.

Richard S Sutton, Doina Precup, and Satinder Singh. Between mdps and semi-mdps: A framework for temporal abstraction in reinforcement learning. Artificial intelligence, 112(1-2):181–211, 1999.

Christian Szegedy. A promising path towards autoformalization and general artificial intelligence. In International Conference on Intelligent Computer Mathematics, pages 3–20. Springer, 2020.

Qwen Team. Qwen3 technical report, 2025. URL https://arxiv.org/abs/2505.09388.

Fengwei Teng, Quan Shi, Zhaoyang Yu, Jiayi Zhang, Yuyu Luo, Chenglin Wu, and Zhijiang Guo. Atom of thoughts for Markov LLM test-time scaling. In Advances in Neural Information Processing Systems (NeurIPS), 2025.

Jonathan Uesato, Nate Kushman, Ramana Kumar, Michele Catasta, Johan Legrand, Jelena Luketina, Andrew Lampinen, Aja Brownsmith, Zoya Bylinskii, Victoria Ellison, et al. Solving math word problems with process-based and outcome-based feedback. CoRR, 2022.

Ke Wang, Zehao Yu, Luwei Wang, and Yongchao Huang. Cache coherent resampling for efficient test time scaling in llm reasoning via adaptive sequential monte carlo. In Forty-third International Conference on Machine Learning, 2026.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. arXiv preprint arXiv:2203.11171, 2022.

Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik Narasimhan. Tree of thoughts: Deliberate problem solving with large language models. Advances in Neural Information Processing Systems (NeurIPS), 36:11809–11822, 2023.

Zhaoyang Yu, Jiayi Zhang, Huixue Su, Yufan Zhao, Yifan Wu, Mingyi Deng, Jinyu Xiang, Yizhang Lin, Lingxiao Tang, Yuyu Luo, et al. Recode: Unify plan and action for universal granularity control. arXiv preprint arXiv:2510.23564, 2025.

Alex L Zhang, Tim Kraska, and Omar Khattab. Recursive language models. arXiv preprint arXiv:2512.24601, 2025a.

Junyu Zhang, Yifan Sun, Tianang Leng, Jingyan Shen, Liu Ziyin, Paul Pu Liang, and Huan Zhang. When reasoning meets its laws. arXiv preprint arXiv:2512.17901, 2025b.

Muru Zhang, Ofir Press, William Merrill, Alisa Liu, and Noah A Smith. How language model hallucinations can snowball. arXiv preprint arXiv:2305.13534, 2023.

Chujie Zheng, Zhenru Zhang, Beichen Zhang, Runji Lin, Keming Lu, Bowen Yu, Dayiheng Liu, Jingren Zhou, and Junyang Lin. Processbench: Identifying process errors in mathematical reasoning. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1009–1024, 2025a.

Congming Zheng, Jiachen Zhu, Zhuoying Ou, Yuxiang Chen, Kangning Zhang, Rong Shan, Zeyu Zheng, Mengyue Yang, Jianghao Lin, Yong Yu, et al. A survey of process reward models: From outcome signals to process supervisions for large language models. arXiv preprint arXiv:2510.08049, 2025b.

Denny Zhou, Nathanael Schärli, Le Hou, Jason Wei, Nathan Scales, Xuezhi Wang, Dale Schuurmans, Olivier Bousquet, Quoc Le, and Ed Chi. Least-to-most prompting enables complex reasoning in large language models. In International Conference on Learning Representations (ICLR), 2023.

Zijian Zhou, Ao Qu, Zhaoxuan Wu, Sunghwan Kim, Alok Prakash, Daniela Rus, Jinhua Zhao, Bryan Kian Hsiang Low, and Paul Pu Liang. Mem1: Learning to synergize memory and reasoning for efficient long-horizon agents. arXiv preprint arXiv:2506.15841, 2025.

Deyao Zhu, Xin Zhou, Shengling Qin, Xuekai Zhu, Hangliang Ding, Shu Zhong, et al. Edgebench: Unveiling scaling laws of learning from real-world environments, 2026. URL https://arxiv. org/abs/2607.05155.

## A Modeling Assumptions and Their Scope

The analysis uses explicit assumptions about sampling, checks, and task decomposition. This appendix states their scope and relates them to the experimental protocols. The Limitations section explains their implications for checking, continuation, and empirical interpretation.

Independence assumption (see Limitations). Prop. 3 requires matching subproblem success rates between composite and standalone prompts, within-run independence of subproblem correctness, and independent identically distributed composite runs. The reuse result in Prop. 5 requires the same reliability bounds for every correct accepted state passed to later stages, with the total bound on accepted-state errors below the target failure probability. Our experiments sample subproblem pairs from distinct MATH subjects as a practical way to reduce concept overlap between subproblems. Subject separation does not establish these assumptions: shared formatting, latent correlations, and position-dependent generation can affect both dependence and marginal success rates.

Error propagation assumptions (see Limitations). The error model in Appendix A.1 assumes that earlier failures add linearly to later failure probabilities, remain in context, and have a common baseline probability $q _ { t } ^ { \mathrm { c l e a n } } = \bar { q }$ when preceding steps are correct. Real autoregressive models exhibit nonlinear error interactions, heterogeneous step difficulty, and partial attention decay across long contexts (Liu et al., 2024), none of which the closed-form solution $\bar { q } ( 1 + c _ { \mathrm { c t x } } ) ^ { t - 1 }$ captures. Appendix A.1 gives the closed form; its parameters are not fitted to the experimental data.

Stepwise decomposition requirement (see Limitations). Prop. 5 requires each step to have a positive probability of producing a correct accepted state from every correct input state. A suitable decomposition must keep the individual steps within the model’s capabilities. Open-ended generation, planning without intermediate checks, and tasks without a step-level correctness criterion may not admit such a decomposition, so the product lower bound need not apply.

Task length must reflect reasoning steps. The scaling variable N counts steps in the chosen decomposition, rather than prompt length or operand digits. Memorization or familiarity can let a model answer without following that decomposition, changing the relation between input size and executed work. Comparing Base 7 and Base 10 varies the numeral representation; without observations of internal steps, it cannot determine the process used to obtain an answer.

Reliability of checks (see Limitations). On LoRe-Compo, self-consistency (SC) and Oracle selection yield similar second-subproblem success rates. This comparison does not estimate the accuracy of either check, and the arithmetic experiments do not compare these selection rules. The error rate among accepted SC answers can be measured by comparing them with reference answers, excluding fallback candidates that did not pass the check. Summaries that include fallback continuations describe all retained paths and cannot directly estimate the error rate among accepted states.

Table 5: Core notation used in the paper. Proof, fit, and metric symbols are defined where they appear.
<table><tr><td>Symbol</td><td>Meaning</td></tr><tr><td colspan="2">Base reasoning quantities</td></tr><tr><td> $\theta , p _ { \theta }$ </td><td>model parameters and induced generation distribution</td></tr><tr><td> $x , x _ { 1 } \oplus x _ { 2 }$ </td><td>task and composite task</td></tr><tr><td> $z _ { 1 : N } , z _ { t }$ </td><td>generated text sequence and text generated at one step</td></tr><tr><td> $s _ { t } , y , a ^ { \star } ( x )$   $N , n$ </td><td>intermediate state, extracted final answer, and target answer</td></tr><tr><td> $A _ { \pmb \theta , 1 } ( x ) , C _ { \pmb \theta , 1 } ( x )$ </td><td>number of steps in the chosen decomposition and digit count</td></tr><tr><td> $\lambda _ { \theta }$ </td><td>single-run answer accuracy and expected generated token count</td></tr><tr><td>Independent sampling quantities</td><td>accuracy decay rate per step in the conditional scaling model</td></tr><tr><td colspan="2">number of sampled runs</td></tr><tr><td> $K$   $A _ { \pmb \theta , K } ( \pmb x ) , C _ { \pmb \theta , K } ( \pmb x )$ </td><td>Pass@ K coverage and expected generated token count  $( K { = } 1 { : }$  one run)</td></tr><tr><td>σ</td><td>relative compositionality ratio</td></tr><tr><td colspan="2">Error propagation and repeated attempts</td></tr><tr><td> $\varepsilon , \bar { q }$   $q _ { t } , q _ { t } ^ { \mathrm { c l e a n } } , q _ { t } ^ { \mathrm { c l e a n } } ( R )$ </td><td>per-step failure probability and common failure probability from correct preceding states</td></tr><tr><td> $q _ { t } ^ { \mathrm { f i o o r } } , c _ { \mathrm { c t x } }$ </td><td>step-failure probability; from a correct prefix; no correct candidate in R attempts</td></tr><tr><td></td><td>asymptotic error floor and coefficient for the effect of earlier errors</td></tr><tr><td colspan="2">State reuse and retry quantities</td></tr><tr><td> $z _ { t } ^ { ( r ) } , s _ { t } ^ { \checkmark }$ </td><td>candidate text and accepted intermediate state</td></tr><tr><td> $\mathcal { G } , \mathcal { G } _ { t } ; \mathcal { V } , \mathcal { V } _ { t }$ </td><td>correctness indicators; accept-or-reject checks</td></tr><tr><td> $\rho _ { t } ( s ) , \alpha _ { t } ( R ; s )$ </td><td>rejection probability and accepted-state error, given input state s</td></tr><tr><td> $\beta _ { t } ( R ; s )$ </td><td>probability of budget exhaustion or an incorrect returned state</td></tr><tr><td> $\varepsilon _ { \mathrm { l o c } , t }$ </td><td>single-attempt failure probability on solvable instances in the mixture model</td></tr><tr><td> $R$ </td><td>local attempt cap; complete-answer attempt count in arithmetic</td></tr><tr><td> $H _ { t }$ </td><td>first t steps complete with every retained state correct</td></tr><tr><td> $W , T _ { t }$ </td><td>executed candidate attempts in the run and at step t</td></tr><tr><td> $h _ { N } , g _ { N }$ </td><td>acceptance-probability lower bound and per-attempt cost upper bound</td></tr><tr><td> $\mathcal { P } _ { N , R }$ </td><td>processes obeying the local retry cap and global stopping rule</td></tr><tr><td> $c _ { \mathrm { g e n } } , c _ { \mathrm { v e r } }$ </td><td>per-attempt generation and checking cost bounds in a common unit</td></tr><tr><td> $A _ { \mathrm { v s r } } , C _ { \mathrm { v s r } } , { \overline { { C } } } _ { \mathrm { v s r } } , C _ { \mathrm { i n d e p } }$ </td><td>reuse accuracy, expected reuse cost, reuse cost bound, and independent sampling cost</td></tr><tr><td colspan="2">Experimental metrics</td></tr><tr><td> $M , B$ </td><td>first-subproblem candidate budget and cap on candidates retained for continuation</td></tr><tr><td> $\hat { E } ( R )$ </td><td>empirical fraction of problems unsolved after R attempts</td></tr><tr><td> $\hat { E } ( 3 2 )$ </td><td>unsolved fraction after 32 complete-answer attempts</td></tr></table>

## A.1 Error Propagation Model

Errors in autoregressive generation may remain in the input context and affect later steps (Zhang et al., 2023; Liu et al., 2024). Let $q _ { t }$ be the failure probability at step t under the generated prefix, and $q _ { t } ^ { \mathrm { c l e a n } }$ its value when all preceding steps are correct. For a coefficient $c _ { \mathrm { c t x } } \geq 0$ describing the effect of earlier errors, consider the qualitative approximation

$$
q _ { t } \approx q _ { t } ^ { \mathrm { c l e a n } } + c _ { \mathrm { c t x } } \sum _ { j < t } q _ { j } .\tag{11}
$$

If $q _ { t } ^ { \mathrm { c l e a n } } = \bar { q }$ at every step, differencing the recurrence gives

$$
q _ { t } \approx \bar { q } ( 1 + c _ { \mathrm { c t x } } ) ^ { t - 1 } .\tag{12}
$$

When $\bar { q } > 0$ and $c _ { \mathrm { c t x } } > 0$ , this expression increases with the step index. If the baseline difficulty varies across steps, $c _ { \mathrm { c t x } } > 0$ alone does not imply increasing failure probability. The parameters $c _ { \mathrm { c t x } }$ and q¯ are not fitted here. The recurrence is a qualitative approximation with a linear, persistent context effect, valid only while predicted probabilities remain below one. The recurrence offers an intuition for context-carried errors, while the product comparison and reuse guarantee use their own explicit probability assumptions. Arithmetic retries do not observe step failures within a run and cannot directly test this recurrence.

## A.2 Correct Candidates before Checking

For a fixed step instance and correct input state $s ,$ let $p _ { t } ( s )$ be the probability that a generated candidate is correct before checking. With R independent identically distributed attempts from this fixed context, the probability of no correct candidate is $( 1 - p _ { t } ( s ) ) ^ { R }$ , which tends to zero whenever $p _ { t } ( s ) > 0$ . Across a distribution of step instances and correct input states, define the probability of generating no correct candidate and its asymptotic error floor by

$$
q _ { t } ^ { \mathrm { c l e a n } } ( R ) : = \mathbb { E } _ { s } \bigl [ ( 1 - p _ { t } ( s ) ) ^ { R } \bigr ] , \qquad q _ { t } ^ { \mathrm { f o o r } } : = \operatorname* { l i m } _ { R \to \infty } q _ { t } ^ { \mathrm { c l e a n } } ( R ) = P _ { s } ( p _ { t } ( s ) = 0 ) .\tag{13}
$$

The limit follows by bounded convergence; a nonzero floor can arise from positive mass on locally unsolvable instances, rather than from independent retries at a fixed positive success probability. A two-component approximation assigns mass $q _ { t } ^ { \mathrm { f l o o r } }$ to such instances and uses a common failure probability $\varepsilon _ { \mathrm { l o c } , t } \in [ 0 , 1 )$ for the remaining instances:

$$
q _ { t } ^ { \mathrm { c l e a n } } ( R ) \approx q _ { t } ^ { \mathrm { f o o r } } + ( 1 - q _ { t } ^ { \mathrm { f l o o r } } ) \varepsilon _ { \mathrm { l o c } , t } ^ { R } .\tag{14}
$$

This expression is exact for the stated two-point mixture; independence alone does not imply this mixture or a common failure probability on solvable instances. Its zero-floor special case is

$$
q _ { t } ^ { \mathrm { f l o o r } } = 0 \quad \Longrightarrow \quad q _ { t } ^ { \mathrm { c l e a n } } ( R ) \approx \varepsilon _ { \mathrm { l o c } , t } ^ { R } .\tag{15}
$$

These probabilities concern generated candidates before checking. They do not include the effect of accepting an incorrect candidate or rejecting a correct one. The arithmetic experiments in Section 4.4 use complete-answer attempts. The unsolved fraction at $R = 3 2$ neither identifies an asymptotic error floor nor estimates the theorem’s per-step failure probability.

## B Proofs and Supporting Analysis

This appendix gives the formal results and their proofs. It covers complete-solution sampling, retry budgets, stage composition, and stopped state reuse. Each result specifies its assumptions.

## B.1 Setup and Notation

We use the same model notation throughout the paper: $\pmb \theta$ denotes the parameters of the reasoning model, and $p _ { \pmb { \theta } }$ denotes its induced generation distribution. For a problem $x ,$ fix a decomposition into $N \geq 1$ steps and write a sampled reasoning chain as $z _ { 1 : N } = ( z _ { 1 } , \dots , z _ { N } )$ , where $z _ { t }$ is the text generated at step t. Let $s _ { t } = [ x , z _ { 1 : t } ]$ denote the full generated history after step t, with $s _ { 0 } = x$ . In reuse, $s _ { t }$ instead denotes the task information and intermediate results actually passed forward, as in Section 2; the prompt presents this state to the model. Each step is sampled autoregressively from $p _ { \pmb { \theta } } ( \cdot \ | \ s _ { t - 1 } )$ . The final answer y is extracted from the completed run. The reuse result assumes that correct completion of every stage yields a correct final answer: if every accepted state is correct and all stages terminate, extraction returns the composite target (for two stages, $[ a ^ { \star } ( x _ { 1 } ) , a ^ { \star } ( x _ { 2 } ) ]$ ]; likewise for multiple stages below). Correctness refers to the retained state, not to reasoning discarded during answer extraction. For local step analysis, $\mathcal { G } _ { t } ( s _ { t - 1 } , z _ { t } ) \in \{ 0 , 1 \}$ indicates whether the resulting state is correct, $q _ { t } : = P ( \mathcal { G } _ { t } = 0 )$ is its failure probability, and $q _ { t } ^ { \mathrm { c l e a n } }$ is the corresponding probability when all preceding states are correct. The check $\mathcal { V } _ { t } ( s _ { t - 1 } , z _ { t } ) \in \{ 0 , 1 \}$ separately records rejection or acceptance and may disagree with $\mathcal { G } _ { t }$ . We write $N ( x )$ for the length of the chosen decomposition of x, abbreviated as N when the problem is clear. The single-run quantities are $C _ { \pmb { \theta } , 1 } ( \boldsymbol { x } ) = \mathbb { E } [ L ]$ and $A _ { \pmb \theta , 1 } ( \boldsymbol x ) = P ( \boldsymbol y = \boldsymbol a ^ { \star } ( \boldsymbol x ) )$ , where $L$ is the total generated token count. For two subproblems $x _ { 1 } , x _ { 2 }$ , their composition $x _ { 1 } \oplus x _ { 2 }$ is formed by concatenating them with a connector prompt. We define its composite target as $a ^ { \star } ( x _ { 1 } \oplus x _ { 2 } ) : = [ a ^ { \star } ( x _ { 1 } ) , a ^ { \star } ( x _ { 2 } ) ]$ . For these chosen decompositions, step-count additivity means that the step counts satisfy

$$
N ( x _ { 1 } \oplus x _ { 2 } ) = N ( x _ { 1 } ) + N ( x _ { 2 } ) .\tag{16}
$$

Operationally, our experiments use distinct MATH subjects as a practical proxy for reduced concept overlap across subproblems. Step-count additivity does not imply the marginal-matching or probabilistic independence assumptions of Prop. 3. For reuse, a step must admit independent repeated attempts from a fixed input state and an accept-or-reject check . The ideal predicate $\mathcal { G }$ defines correctness, whereas is the check applied to candidate states; acceptance alone does not imply $\mathcal { G } = 1$ . In the reuse proofs, $0 < \rho < 1$ uniformly bounds $\rho _ { t } ( s )$ , and α uniformly bounds the error rate $\alpha _ { t } ( R ; s )$ among accepted states over all relevant steps, correct accepted input states s, and attempt budgets R. With R total attempts, equivalently $R { - } 1$ retries, all candidates are rejected with probability at most $\rho ^ { R }$ . Returning an incorrect accepted state is a separate source of failure. For independent sampling, the accuracy and generation cost of $K$ complete runs satisfy

$$
A _ { \theta , K } ( x ) = 1 - ( 1 - A _ { \theta , 1 } ( x ) ) ^ { K }\tag{17}
$$

$$
C _ { \pmb { \theta } , K } ( \pmb { x } ) = K \cdot C _ { \pmb { \theta } , 1 } ( \pmb { x } )\tag{18}
$$

Here $A _ { \pmb { \theta } , K }$ is Pass@K coverage and $C _ { \pmb { \theta } , K }$ is expected generation cost; $K = 1$ denotes a single complete run. Across task lengths N, the conditional scaling assumptions are

$$
C _ { \pmb { \theta } , 1 } ( x ) = \Theta ( N ) \quad ( \mathrm { C o m p u t e L a w } )\tag{19}
$$

$$
A _ { \pmb { \theta } , 1 } ( \pmb { x } ) = \mathrm { e x p } ( - \lambda _ { \pmb { \theta } } \pmb { N } ) , \quad \lambda _ { \pmb { \theta } } > 0 \quad \mathrm { ( A c c u r a c y L a w ) }\tag{20}
$$

Notation conventions. All probabilities denoted by $P ( \cdot )$ are taken under the relevant sampling process induced by p<sub>θ</sub>; p<sub>θ</sub> itself denotes the model distribution. For an integer $K , [ K ]$ denotes $\{ 1 , \ldots , K \}$ . The symbol $\lll$ denotes an informal “much smaller than” condition in limiting approximations. Failure budgets satisfy $\delta , \delta _ { i } , \delta _ { m } \in ( 0 , 1 )$ , with $\delta = \delta _ { 1 } + \delta _ { 2 }$ in the two-stage result and $\begin{array} { r } { \sum _ { m } \delta _ { m } = \delta } \end{array}$ in the multi-stage result, where the sum ranges over all stages of the task.

## B.2 Ideal Laws Under Per-Step Errors

We begin with the success probability and cost implied by independent per-step errors. This calculation concerns a step decomposition; the arithmetic experiments measure complete-answer coverage.

Proposition 1 (Exponential Decay of Success Probability) . Consider a chosen decomposition into N steps, each with independentfailure probability $\varepsilon \in ( 0 , 1 )$ , and assume the final answer is correct ifand only ifall N steps are correct. Then:

$$
A _ { \theta , 1 } ( x ) = ( 1 - \varepsilon ) ^ { N } \leq e ^ { - \varepsilon N } ,\tag{21}
$$

where the equality $( 1 - \varepsilon ) ^ { N }$ is exact and the upper bound $e ^ { - \varepsilon N }$ follows from log $( 1 - \varepsilon ) \leq - \varepsilon$ . This gives $E q .$ (71) with the per-step accuracy decay rate $\lambda _ { \pmb { \theta } } = - \log ( 1 - \varepsilon )$

Proof. Step 1: Success requires all steps. Let $F _ { i }$ indicate failure at step i; each step succeeds with probability $1 - \varepsilon .$ , so independence yields the following product:

$$
A _ { \pmb \theta , 1 } ( \pmb x ) = \prod _ { i = 1 } ^ { N } P ( F _ { i } = 0 ) = ( 1 - \varepsilon ) ^ { N } .\tag{22}
$$

Step 2: Logarithmic inequality. Since log is concave and $\log ( 1 ) = 0$ , the tangent line at $x = 1$ gives log $( x ) \leq x - 1$ for all $x > 0$ . Substituting $x = 1 - \varepsilon \colon \log ( 1 - \varepsilon ) \leq ( 1 - \varepsilon ) - 1 = - \varepsilon$ for all $\varepsilon \in ( 0 , 1 )$ Taking the natural logarithm of Step 1:

$$
\log A _ { \pmb \theta , 1 } ( x ) = N \log ( 1 - \varepsilon ) \leq - \varepsilon N .\tag{23}
$$

Step 3: Exponential bound and approximation. Exponentiating both sides:

$$
A _ { \pmb \theta , 1 } ( x ) \leq e ^ { - \varepsilon N } .\tag{24}
$$

For fixed N as $\varepsilon \to 0$ , the Taylor expansion lo $\displaystyle { \mathfrak { g } } ( 1 - \varepsilon ) = - \varepsilon + O ( \varepsilon ^ { 2 } )$ yields $A _ { \pmb { \theta } , 1 } ( x ) / e ^ { - \varepsilon N }  1$ □

## B.3 The Compositionality Breakdown under Independent Sampling

We first give a general event-inclusion bound for a common pool of composite runs, then prove the conditional product comparison used in the main text.

Proposition 2 (Event-Inclusion Bound) . For any $K \geq 1$ , sample a common pool of K composite runs for $x _ { 1 } \oplus x _ { 2 }$ . Let $E _ { i } ^ { K }$ denote coverage of a correct answer to $x _ { i }$ , and $E _ { 1 2 } ^ { K }$ coverage ofboth correct answers in one run. These events satisfy

$$
P ( E _ { 1 2 } ^ { K } ) \leq P ( E _ { 1 } ^ { K } \cap E _ { 2 } ^ { K } ) \leq \operatorname* { m i n } \{ P ( E _ { 1 } ^ { K } ) , P ( E _ { 2 } ^ { K } ) \} ,\tag{25}
$$

without independence. All probabilities refer to the common pool of composite runs.

Proposition 3 (Product Form) . Let $a : = A _ { \pmb \theta , 1 } ( x _ { 1 } ) \in ( 0 , 1 )$ and $b : = A _ { \pmb \theta , 1 } ( x _ { 2 } ) \in ( 0 , 1 )$ . Assume each subproblem has the same correctness probability in a composite run as in its standalone run, the two correctness indicators are independent within each composite run, and the composite runs are i.i.d. across K samples. Then $A _ { \pmb \theta , 1 } ( x _ { 1 } \oplus x _ { 2 } ) =$ ab and:

$$
A _ { \theta , K } ( x _ { 1 } \oplus x _ { 2 } ) \ < \ A _ { \theta , K } ( x _ { 1 } ) \cdot A _ { \theta , K } ( x _ { 2 } ) , \qquad \forall K \geq 2 .\tag{26}
$$

Remark 1 (Experimental proxy) . The main-text product comparison relies on the assumptions of Prop. 3; the same-pool event-inclusion bound of Prop. 2 holds without independence. Our experiments sample subproblem pairs from distinct MATH subjects to encourage reduced concept overlap across subproblems. The no\_context ablation in Table 3 gives a descriptive comparison: second-subproblem success without $x _ { 1 }$ information is 36.2%. This differsfrom joint success on both problems, and the available standalone $x _ { 2 }$ reference coversfewer models than the second-subproblem evaluation. As discussed in Appendix A, subject separation establishes neither independence within a run nor matching success rates across prompt formats.

ProofofProp. 2. For a pool of K composite runs with sample space $\Omega ,$ define

$$
E _ { 1 2 } ^ { K } : = \{ \exists k : \operatorname { r u n } k { \mathrm { ~ r e t u r n s ~ a ~ c o r r e c t ~ a n s w e r ~ t o ~ b o t h ~ } } x _ { 1 } { \mathrm { ~ a n d ~ } } x _ { 2 } \} ,
$$

$$
E _ { 1 } ^ { K } : = \{ \exists k : \mathrm { r u n } k \mathrm { r e t u r n s ~ a ~ c o r r e c t ~ a n s w e r ~ t o ~ } x _ { 1 } \} ,\tag{27}
$$

and analogously $E _ { 2 } ^ { K }$ for $x _ { 2 }$ . The inclusion $E _ { 1 2 } ^ { K } \subseteq E _ { 1 } ^ { K } \cap E _ { 2 } ^ { K }$ holds: if some run $k$ is correct on both subproblems, then that same run witnesses $E _ { 1 } ^ { \bar { K } }$ and witnesses $E _ { 2 } ^ { K }$ . Monotonicity of probability gives ${ \cal P } ( \hat { E } _ { 1 2 } ^ { K } ) \leq { \cal P } ( E _ { 1 } ^ { K } \cap E _ { 2 } ^ { K } ) \leq \operatorname* { m i n } \{ { \cal P } ( E _ { 1 } ^ { K } ) , \dot { \cal P } ( \dot { E } _ { 2 } ^ { K } ) \}$ , which is Eq. (25). This bound holds for every K without assuming independence of subproblem-answer correctness. □

ProofofProp. 3. Assume the matching composite/standalone marginals, within-run independence, and across-run i.i.d. conditions, and let $a : = A _ { \pmb { \theta } , 1 } ( x _ { 1 } )$ and $b : = A _ { \pmb { \theta } , 1 } ( x _ { 2 } )$ , so that $A _ { \pmb \theta , 1 } ( x _ { 1 } \oplus x _ { 2 } ) = a b$ . For K independently sampled runs, define indicator random variables: $X _ { k } \in \{ 0 , 1 \}$ records that sampled run k succeeds on $x _ { 1 }$ with $P ( X _ { k } = 1 ) = a ; Y _ { k } \in \{ 0 , 1 \}$ records that sampled run k succeeds on $x _ { 2 }$ with $P ( Y _ { k } = 1 ) = b ;$ and $Z _ { k } : = X _ { k } Y _ { k }$ records that sampled run k succeeds on both with $P ( Z _ { k } = 1 ) = a b$ The two sequences are independent, with i.i.d. entries within each sequence.

Step 1: Event formulation. Define success events:

$$
\begin{array} { r } { E _ { 1 } ^ { K } : = \{ \exists k : X _ { k } = 1 \} , \quad E _ { 2 } ^ { K } : = \{ \exists k : Y _ { k } = 1 \} , \quad E _ { 1 2 } ^ { K } : = \{ \exists k : Z _ { k } = 1 \} . } \end{array}\tag{28}
$$

Independence across runs gives the following probabilities for the three success events:

$$
P ( E _ { 1 } ^ { K } ) = 1 - ( 1 - a ) ^ { K } , \quad P ( E _ { 2 } ^ { K } ) = 1 - ( 1 - b ) ^ { K } , \quad P ( E _ { 1 2 } ^ { K } ) = 1 - ( 1 - a b ) ^ { K } .\tag{29}
$$

Step 2: Set inclusion. Key observation: $E _ { 1 2 } ^ { K } \subseteq E _ { 1 } ^ { K } \cap E _ { 2 } ^ { K }$ . To see this, suppose $\zeta \in E _ { 1 2 } ^ { K }$ , so there exists k with $Z _ { k } ( \zeta ) = X _ { k } ( \zeta ) Y _ { k } ( \zeta ) = 1$ . Then $X _ { k } ( \zeta ) = 1$ , hence $\zeta \in E _ { 1 } ^ { K }$ ; and $Y _ { k } ( \zeta ) = 1$ , hence $\zeta \in E _ { 2 } ^ { K }$ Therefore $\zeta \in E _ { 1 } ^ { K } \cap E _ { 2 } ^ { K }$ , completing the inclusion. Monotonicity of probability gives:

$$
P ( E _ { 1 2 } ^ { K } ) \leq P ( E _ { 1 } ^ { K } \cap E _ { 2 } ^ { K } ) .\tag{30}
$$

Step 3: Independence gives exact product. Since $\{ X _ { k } \} _ { k = 1 } ^ { K }$ and $\{ Y _ { k } \} _ { k = 1 } ^ { K }$ are mutually independent by assumption, any event measurable w.r.t. $\{ X _ { k } \}$ alone is independent of any event measurable w.r.t. $\left\{ Y _ { k } \right\}$ alone. $E _ { 1 } ^ { K } = \{ \exists k : X _ { k } = 1 \}$ depends only on $\{ X _ { k } \}$ , and $\bar { E } _ { 2 } ^ { K } = \{ \exists k : Y _ { k } = 1 \}$ depends only on $\left\{ Y _ { k } \right\}$ Therefore $E _ { 1 } ^ { K }$ and $E _ { 2 } ^ { K }$ are independent:

$$
P ( E _ { 1 } ^ { K } \cap E _ { 2 } ^ { K } ) = P ( E _ { 1 } ^ { K } ) P ( E _ { 2 } ^ { K } ) = [ 1 - ( 1 - a ) ^ { K } ] [ 1 - ( 1 - b ) ^ { K } ] .\tag{31}
$$

Combining this product identity with the event inclusion from Step 2 gives the comparison

$$
1 - ( 1 - a b ) ^ { K } \leq [ 1 - ( 1 - a ) ^ { K } ] [ 1 - ( 1 - b ) ^ { K } ] .\tag{32}
$$

This establishes the non-strict product comparison in Prop. 3; strictness is proved next.

Step 4: Strictness. For $K \geq 2$ and $a , b \in ( 0 , 1 )$ , the inequality is strict. Consider the event:

$$
\begin{array} { r } { E _ { 1 , 2 } : = \{ X _ { 1 } = 1 , Y _ { 1 } = 0 , X _ { 2 } = 0 , Y _ { 2 } = 1 , \forall k \geq 3 : X _ { k } Y _ { k } = 0 \} . } \end{array}\tag{33}
$$

The following three properties show that this event has positive mass outside joint success.

(i) $E _ { 1 , 2 } \subseteq E _ { 1 } ^ { K } \cap E _ { 2 } ^ { K } . \ X _ { 1 } = 1$ implies $E _ { 1 } ^ { K } ; Y _ { 2 } = 1$ implies $E _ { 2 } ^ { K }$

(ii) $E _ { 1 , 2 } \subseteq ( E _ { 1 2 } ^ { K } ) ^ { c } .$ . Under $E _ { 1 , 2 } \colon$ sampled run 1 has $Z _ { 1 } = X _ { 1 } Y _ { 1 } = 1 \cdot 0 = 0 \mathrm { { ; } }$ ; sampled run 2 has $Z _ { 2 } = X _ { 2 } Y _ { 2 } = 0 \cdot 1 = 0 ;$ and $Z _ { k } = 0$ for all $k \geq 3$ by construction. Hence no sampled run succeeds on both subproblems, so $E _ { 1 2 } ^ { K }$ does not occur under this construction.

(iii) $P ( E _ { 1 , 2 } ) > 0 .$ . By mutual independence of all $\{ X _ { k } \}$ and $\left\{ Y _ { k } \right\}$

$$
P ( E _ { 1 , 2 } ) = a ( 1 - b ) ( 1 - a ) b \prod _ { k = 3 } ^ { K } ( 1 - a b ) = a b ( 1 - a ) ( 1 - b ) ( 1 - a b ) ^ { K - 2 } > 0 ,\tag{34}
$$

The strict positivity follows because every factor is positive for $a , b \in ( 0 , 1 )$

Properties (i) and (ii) give $E _ { 1 , 2 } \ \subseteq \ ( E _ { 1 } ^ { K } \cap E _ { 2 } ^ { K } ) \setminus E _ { 1 2 } ^ { K } ,$ so $P ( ( E _ { 1 } ^ { K } \cap E _ { 2 } ^ { K } ) \setminus E _ { 1 2 } ^ { K } ) \geq P ( E _ { 1 , 2 } ) > 0 .$ establishing the required strict comparison $P ( E _ { 1 2 } ^ { K } ) < P ( E _ { 1 } ^ { K } \cap E _ { 2 } ^ { K } )$ for every $K \geq 2 .$ □

Low accuracy behavior in the difficult problem range. Under the assumptions of Prop. 3, consider the ratio between composite accuracy and the product of standalone accuracies when $K a , K b \ll 1$ The approximation $1 - ( 1 - x ) ^ { K } \approx K x$ for $K x \ll 1$ follows from $( 1 - x ) ^ { K } = 1 - K x + O ( K ^ { 2 } x ^ { 2 } )$ Applying this to each term:

$$
\begin{array} { c } { { A _ { \theta , K } ( x _ { 1 } ) = 1 - ( 1 - a ) ^ { K } \approx K a , \quad A _ { \theta , K } ( x _ { 2 } ) = 1 - ( 1 - b ) ^ { K } \approx K b , } } \\ { { A _ { \theta , K } ( x _ { 1 } \oplus x _ { 2 } ) = 1 - ( 1 - a b ) ^ { K } \approx K ( a b ) , } } \end{array}
$$

where the last line uses $K ( a b ) \leq K a \cdot b \leq K a \ll 1$ , ensuring the approximation is valid. Define $\sigma : = A _ { \pmb \theta , K } ( x _ { 1 } \oplus x _ { 2 } ) / [ A _ { \pmb \theta , K } ( x _ { 1 } ) A _ { \pmb \theta , K } ( x _ { 2 } ) ]$ . Then $\sigma = 1$ means accuracy compositionality, and $\sigma <$ means composite accuracy falls below the product of standalone accuracies. The resulting first-order ratio quantifies the relative gap before either accuracy saturates:

$$
\sigma = { \frac { A _ { \theta , K } ( x _ { 1 } \oplus x _ { 2 } ) } { A _ { \theta , K } ( x _ { 1 } ) \cdot A _ { \theta , K } ( x _ { 2 } ) } } \approx { \frac { K ( a b ) } { ( K a ) ( K b ) } } = { \frac { K a b } { K ^ { 2 } a b } } = { \frac { 1 } { K } } .\tag{35}
$$

This $1 / K$ degradation scales inversely with the number of samples in the low-accuracy regime, showing that the relative compositionality breakdown of independent sampling worsens before saturation. This first-order approximation requires $K a , K b \ll 1$ . Under the assumptions of Prop. 3, the product inequality (26) holds for all $K \geq 2$ and $a , b \in ( 0 , 1 )$ ; the common-pool inclusion in Eq. (25) remains valid for dependent subproblem-answer correctness.

## B.4 State Reuse: Mathematical Foundations

We now derive a product lower bound on success and a sufficient computation budget for state reuse.

Definition 1 (Retry Loop) . Start from $s _ { 0 } ^ { \checkmark } = x .$ . At step t, let $s = s _ { t - 1 } ^ { \checkmark }$ <sub>1</sub> be the fixed accepted input state and $\rho _ { t } ( s ) : = P ( \mathcal { V } _ { t } = 0 \mid s )$ its single-attempt rejection probability. The loop uses an integer cap $R \geq 1$ and repeats the following generation and checking steps:

1. Sample a candidate span $z _ { t } ^ { ( r ) } \sim p _ { \pmb { \theta } } ( \cdot \mid s )$ from the fixed input state.

2. Form the candidate intermediate state $\tilde { s } _ { t } ^ { ( r ) }$ , using either the full state $[ s , z _ { t } ^ { ( r ) } ]$ or the prescribed extracted intermediate answer.

3. Check $\mathcal { V } _ { t } \big ( \tilde { s } _ { t } ^ { ( r ) } \big ) .$ ; if accepted, return $s _ { t } ^ { \checkmark } = \tilde { s } _ { t } ^ { ( r ) }$ and proceed.

4. If rejected, retry from the same s until R total attempts have been used; if all are rejected, stop with failure.

Let $A _ { t } ^ { ( \bar { R } ) }$ be the event that the loop returns an accepted state, and let $\mathcal { G } _ { t }$ indicate whether that state is correct. Define accepted-state error, the error probability conditional on acceptance, as

$$
\alpha _ { t } ( R ; s ) : = P \Big ( \mathcal G _ { t } = 0 \mid A _ { t } ^ { ( R ) } , s \Big ) .\tag{36}
$$

Assume generation and check outcomes are independent and identically distributed across attempts conditional on s. Failure comprises budget exhaustion or an incorrect acceptance:

$$
\begin{array} { r l } & { \beta _ { t } ( R ; s ) : = \rho _ { t } ( s ) ^ { R } + \big ( 1 - \rho _ { t } ( s ) ^ { R } \big ) \alpha _ { t } ( R ; s ) } \\ & { \qquad \leq \rho _ { t } ( s ) ^ { R } + \alpha _ { t } ( R ; s ) . } \end{array}\tag{37}
$$

The two terms are budget exhaustion and an incorrect state returned as accepted; the formal loop does not return a fallback after all candidates are rejected.

Remark 2 (Relation to Correct Candidates before Checking) . The proofs use thefailure bound in Eq. (37); they do not assume that an observed small unsolvedfraction is an exactly zero asymptotic error floor. Their uniform bounds $\rho < 1$ and α $< \delta / N < 1$ imply a positive probability of a correct accepted candidate at every relevant correct prefix: at one attempt this probability is at least $( 1 - \rho ) ( 1 - \alpha ) > 0 .$ Thus the probability of generating no correct candidate from a fixed correct input state tends to zero under independent attempts. The assumptions already require each step to be solvable. The distribution considered in Appendix A.2 may also include instances that do not satisfy these assumptions.

Proposition 4 (Logarithmic Attempt Requirement) . Under the assumptions of Def. 1, suppose $\rho _ { t } ( s ) \leq \rho$ for a fixed $0 < \rho < 1$ and $\alpha _ { t } ( r ; s ) \le \alpha$ for every relevant step, correct accepted input state s, and budget r. To keep the probability of any step exhausting its budget or accepting an incorrect state at most δ, it suffices that $N \alpha < \delta$ and

$$
R \ge \left\lceil \frac { \log \left( N / ( \delta - N \alpha ) \right) } { \log ( 1 / \rho ) } \right\rceil .\tag{38}
$$

For fixed $0 < \rho < 1$ and $\alpha \leq c \delta / N$ withfixed $0 \leq c < 1$ , the sufficient budget above is $\Theta ( \log ( N / \delta ) )$ ;   
in the separate case $\rho = 0 ,$ one attempt suffices for the rejection term.

Proof. Step 1: Union bound. Let $H _ { i - 1 }$ be the event that the first i 1 steps complete with correct accepted states, and let $E _ { i } : = H _ { i - 1 } \cap$ step i fails after its R-attempt loop be the first-failure event at

step i. These disjoint first-failure events exhaust trajectory failure, so the union bound gives

$$
P ( \mathrm { g l o b a l ~ f a i l } ) = P \left( \bigcup _ { i = 1 } ^ { N } E _ { i } \right) \ \leq \ \sum _ { i = 1 } ^ { N } P ( E _ { i } ) .\tag{39}
$$

Conditioned on $H _ { i - 1 }$ , step i starts from a correct accepted prefix. By Eq. (37), $P ( E _ { i } ) \leq P ( { \mathrm { s t e p } } i { \mathrm { f a i l s } }$ $H _ { i - 1 } ) \leq \rho ^ { R } + \alpha$ , giving:

$$
P ( { \mathrm { g l o b a l f a i l } } ) \ \leq \ N { \big ( } \rho ^ { R } + \alpha { \big ) } .\tag{40}
$$

Step 2: Per-Step Target. For $N \alpha < \delta ,$ the per-step requirement is

$$
N ( \rho ^ { R } + \alpha ) \leq \delta \quad \Longleftrightarrow \quad \rho ^ { R } \leq \frac { \delta - N \alpha } { N } .\tag{41}
$$

Step 3: Solve for R. Taking logarithms (note log $\rho < 0 .$ , so the inequality flips):

$$
R \log \rho \leq \log \left( { \frac { \delta - N \alpha } { N } } \right)\tag{42}
$$

$$
R \geq \frac { \log ( N / ( \delta - N \alpha ) ) } { \log ( 1 / \rho ) } .\tag{43}
$$

Because the attempt budget is an integer, taking the ceiling gives a sufficient choice:

$$
R \ge \left\lceil \frac { \log \left( N / ( \delta - N \alpha ) \right) } { \log ( 1 / \rho ) } \right\rceil .\tag{44}
$$

If, for example, $N \alpha \leq c \delta$ for a fixed $c < 1$ , then $R = \Theta ( \log ( N / \delta ) )$ . More attempts alone cannot establish this guarantee when the bound on accepted-state errors already reaches δ. □

## B.5 Accepted-State Error under a Fixed Check

Fix a correct input state s and a step t. Suppose the same candidate-generation law and the same singlecandidate checking rule are used independently on every attempt, irrespective of the attempt cap R. Let $\rho = P ( \mathcal { V } _ { t } = 0 \ | \ s ) < 1$ and $w = P ( \mathcal { G } _ { t } = 0 , \mathcal { V } _ { t } = 1 \mid s )$ . Returning an incorrect state on attempt r requires rejection on the first $r - 1$ attempts and an incorrect acceptance on attempt r. Summing over the possible first-acceptance attempts gives the joint and acceptance probabilities

P(incorrect returned state,

$$
A _ { t } ^ { ( R ) } \mid s ) = \sum _ { r = 1 } ^ { R } \rho ^ { r - 1 } w = \frac { w ( 1 - \rho ^ { R } ) } { 1 - \rho } ,\tag{45}
$$

$$
P ( A _ { t } ^ { ( R ) } \mid s ) = 1 - \rho ^ { R } .\tag{46}
$$

Dividing by the acceptance probability yields the conditional error and the total local failure:

$$
\alpha _ { t } ( R ; s ) = \frac { w } { 1 - \rho } , \qquad \beta _ { t } ( R ; s ) = \rho ^ { R } + ( 1 - \rho ^ { R } ) \frac { w } { 1 - \rho } .\tag{47}
$$

The conditional error among accepted states is independent of R in this fixed-check setting, and the local failure probability approaches that error as R grows. This calculation includes false rejections through $\rho ; w / ( 1 - \rho )$ is the error conditional on acceptance, not the probability of acceptance conditional on an incorrect candidate. Changing the voting pool or acceptance threshold changes the checking rule and can therefore change the error among accepted states.

A batch selector could be treated as one local attempt only after its returned-state law, rejection event, and computation cost are specified. Independent batches would then need the corresponding conditional bounds at every correct prefix. The empirical protocol’s multiple retained states and fallback continuations are not obtained merely by substituting M for R in the single-state theorem.

## B.6 Product Lower Bound Under State Reuse

The next proposition composes stage reliability guarantees and adds their computation budgets.

Proposition 5 (Product Lower Bound for State Reuse) . For subproblems $x _ { 1 } , x _ { 2 }$ with $N _ { 1 } , N _ { 2 }$ steps, assume the loop of Def. 1. At stage i, let $\rho _ { i , t } ( s ) \leq \rho _ { i }$ with $0 < \rho _ { i } < 1$ and $\alpha _ { i , t } ( r ; s ) \le \alpha _ { i }$ for every step, correct accepted input state s, and budget r. For stage 2, these bounds must hold for every correct state passed from stage 1. Let $\delta _ { 1 } , \delta _ { 2 } > 0$ be target failure probabilities with $\delta _ { 1 } + \delta _ { 2 } = \delta .$ Assume each stage reaches a well-definedfinal state after its prescribed $N _ { i }$ accepted steps. Let $S _ { i }$ denote completion ofstage i with every accepted state correct. Completing both stages correctly must yield the target under answer extraction:

$$
S _ { 1 } \cap S _ { 2 } \subseteq \{ y = a ^ { \star } ( x _ { 1 } \oplus x _ { 2 } ) \} .
$$

When $N _ { i } \alpha _ { i } < \delta _ { i }$ , allocate the following sufficient attempt budget to each step in stage i:

$$
R _ { i } \ge \left\lceil \frac { \log \left( N _ { i } / ( \delta _ { i } - N _ { i } \alpha _ { i } ) \right) } { \log \left( 1 / \rho _ { i } \right) } \right\rceil ,\tag{48}
$$

By Prop. 4, stage i thenfails to complete with all accepted states correct with probability at most $\delta _ { i } .$ Under these assumptions, state reuse satisfies:

$$
A _ { \mathrm { v s r } } ( x _ { 1 } \oplus x _ { 2 } ) \ \geq \ ( 1 - \delta _ { 1 } ) ( 1 - \delta _ { 2 } ) \ = \ 1 - \delta + \delta _ { 1 } \delta _ { 2 } \ \geq \ 1 - \delta .\tag{49}
$$

This exact lower bound uses target stage guarantees rather than standalone accuracies.

Computation budget. Set $\delta _ { i } = \delta N _ { i } / ( N _ { 1 } + N _ { 2 } )$ and use the common error bound $\alpha _ { i } = \alpha$ . Both subproblems then have the same per-step target failure probability:

$$
\frac { \delta _ { 1 } } { N _ { 1 } } = \frac { \delta _ { 2 } } { N _ { 2 } } = \frac { \delta } { N _ { 1 } + N _ { 2 } } .\tag{50}
$$

If $N \alpha < \delta$ for $N = N _ { 1 } + N _ { 2 }$ and $\rho _ { i } \le \rho < 1$ , choose the common attempt count

$$
R ^ { * } = \left\lceil \frac { \log ( N / ( \delta - N \alpha ) ) } { \log ( 1 / \rho ) } \right\rceil\tag{51}
$$

This choice satisfies both stage constraints. Let $c _ { \mathrm { g e n } }$ and $c _ { \mathrm { v e r } }$ uniformly bound generation and checking costs per attempt in a common unit, for all reached states. The per-step computation budget is $\bar { c } _ { \mathrm { s t e p } } ( R ^ { * } ) =$ $R ^ { * } ( c _ { \mathrm { g e n } } + c _ { \mathrm { v e r } } )$ . It is an upper bound on actual cost and sums across stages:

$$
\begin{array} { r l r } & { } & { \overline { { C } } _ { \mathrm { v s r } } ( x _ { 1 } ) + \overline { { C } } _ { \mathrm { v s r } } ( x _ { 2 } ) = N _ { 1 } \bar { c } _ { \mathrm { s t e p } } ( R ^ { * } ) + N _ { 2 } \bar { c } _ { \mathrm { s t e p } } ( R ^ { * } ) } \\ & { } & { = ( N _ { 1 } + N _ { 2 } ) \bar { c } _ { \mathrm { s t e p } } ( R ^ { * } ) = \overline { { C } } _ { \mathrm { v s r } } ( x _ { 1 } \oplus x _ { 2 } ) . } \end{array}\tag{52}
$$

Proof of Prop. 5. Step 1: Reliability of $x _ { 1 }$ . For $R _ { 1 }$ from Eq. (48), every correct accepted prefix s has local failure bounded by the allocated per-step target for the first stage:

$$
\beta _ { 1 , t } ( R _ { 1 } ; s ) \ \leq \ \rho _ { 1 } ^ { R _ { 1 } } + \alpha _ { 1 } \ \leq \ \frac { \delta _ { 1 } } { N _ { 1 } } .\tag{53}
$$

A union bound over the $N _ { 1 }$ possible first-failure locations gives the first-stage guarantee:

$$
P ( S _ { 1 } ^ { c } ) \ \leq \ N _ { 1 } ( \rho _ { 1 } ^ { R _ { 1 } } + \alpha _ { 1 } ) \ \leq \ \delta _ { 1 } , \qquad \mathrm { h e n c e } \quad P ( S _ { 1 } ) \ \geq \ 1 - \delta _ { 1 } .\tag{54}
$$

Step 2: Reliability of $x _ { 2 }$ from correct accepted $x _ { 1 }$ states. With $R _ { 2 }$ satisfying Eq. (48), each attempt is drawn from the model given its fixed accepted input state. The bounds $\rho _ { 2 }$ and $\alpha _ { 2 }$ must hold for every correct accepted state passed from x ; acceptance alone does not establish this requirement. Conditioning on an arbitrary such state $s _ { N _ { 1 } } ^ { \checkmark }$ yields the same second-stage bound:

$$
P \Big ( S _ { 2 } ^ { c } \Big | \ : s _ { N _ { 1 } } ^ { \vee } , \ : S _ { 1 } \Big ) \ : \le \ : N _ { 2 } ( \rho _ { 2 } ^ { R _ { 2 } } + \alpha _ { 2 } ) \ : \le \ : \delta _ { 2 } .\tag{55}
$$

Marginalizing over correct accepted states (i.e., conditioning on $S _ { 1 } )$ :

$$
P ( S _ { 2 } ^ { c } \mid S _ { 1 } ) = \mathbb { E } \Big [ P \Big ( S _ { 2 } ^ { c } \mid s _ { N _ { 1 } } ^ { \check { \check { \cal N } } } , S _ { 1 } \Big ) \Bigm | S _ { 1 } \Big ] \ \le \ \delta _ { 2 } ,\tag{56}
$$

Thus the conditional second-stage reliability satisfies the allocated target, $P ( S _ { 2 } \mid S _ { 1 } ) \ge 1 - \delta _ { 2 }$

Step 3: Composite accuracy via chain rule. By assumption, $S _ { 1 } \cap S _ { 2 }$ implies that the prescribed extraction returns a correct composite answer. The probability chain rule gives:

$$
\begin{array} { r } { A _ { \mathrm { v s r } } ( x _ { 1 } \oplus x _ { 2 } ) \ \ge \ P ( S _ { 1 } \cap S _ { 2 } ) = P ( S _ { 1 } ) \cdot P ( S _ { 2 } \mid S _ { 1 } ) \ \ge \ ( 1 - \delta _ { 1 } ) ( 1 - \delta _ { 2 } ) . } \end{array}\tag{57}
$$

Step 4: Target level product bound. Expanding gives $( 1 - \delta _ { 1 } ) ( 1 - \delta _ { 2 } ) = 1 - \delta + \delta _ { 1 } \delta _ { 2 } \geq 1 - \delta$ . No small-$\delta _ { i }$ approximation is needed for this lower bound because the cross term is nonnegative. Equation (57) is an exact lower bound; equality with actual standalone accuracies is not implied. □

Comparison with independent sampling. The two propositions compare different quantities. Under Prop. 3, independent sampling yields composite accuracy below the product of standalone accuracies, with a ratio of approximately $1 / K$ when $K a , K b \ll 1$ . Under Prop. 5, reuse from accepted states yields a product lower bound in stage reliability guarantees; it concerns target guarantees rather than an equality of actual standalone accuracies.

## B.7 Multi-Stage Generalization

We extend Prop. 5 by induction to arbitrary multi-stage workflows with J subproblems $\boldsymbol { x } = \boldsymbol { x } _ { 1 } \oplus \cdot \cdot \cdot \oplus \boldsymbol { x } _ { J }$ each requiring $N _ { m }$ steps, for a total of $\begin{array} { r } { N = \sum _ { m = 1 } ^ { J } N _ { m } } \end{array}$

Step 1: Proportional budget allocation. Allocate failure budgets proportionally to step counts:

$$
\delta _ { m } : = \delta \frac { N _ { m } } { N } , \qquad \sum _ { m = 1 } ^ { J } \delta _ { m } = \delta .\tag{58}
$$

This yields a uniform per-step target across all subproblems:

$$
\frac { \delta _ { m } } { N _ { m } } = \frac { \delta } { N } \quad \mathrm { f o r \ a l l } \ m \in [ J ] .\tag{59}
$$

Step 2: Uniform attempt count. Assume $\rho _ { t } ( s ) \leq \rho$ with $0 < \rho < 1$ and $\alpha _ { t } ( r ; s ) \le \alpha$ for all stages, relevant steps, correct accepted input states $s ,$ and attempt budgets r, with $N \alpha < \delta .$ . Prop. 4 gives a common sufficient retry budget by imposing the constraint $\rho ^ { R } + \alpha \leq \delta / N$ at each step:

$$
R \ \geq \ \left\lceil { \frac { \log \left( N / ( \delta - N \alpha ) \right) } { \log ( 1 / \rho ) } } \right\rceil \ = \ \Theta \left( \log { \frac { N } { \delta } } \right)\tag{60}
$$

when $N \alpha \leq c \delta$ for a fixed $0 \leq c < 1$ . Since all stages receive the same error allowance per step, this common attempt budget meets the reliability requirement for every stage length $N _ { m }$

Step 3: Summing the computation budget. Let $R ^ { * }$ be the smallest integer budget in Eq. (60). Each attempt costs at most $c _ { \mathrm { g e n } } + c _ { \mathrm { v e r } }$ , so the per-step computation budget is

$$
\bar { c } _ { \mathrm { s t e p } } ( R ^ { * } ) = R ^ { * } ( c _ { \mathrm { g e n } } + c _ { \mathrm { v e r } } ) .\tag{61}
$$

Since $R ^ { * }$ is the same for all subproblems, the computation budget sums across stages. Actual cost is no larger because each loop stops at first acceptance:

$$
\begin{array} { c l l } { \overline { { C } } _ { \mathrm { v s r } } ( x ) = \displaystyle \sum _ { m = 1 } ^ { J } \overline { { C } } _ { \mathrm { v s r } } ( x _ { m } ) = \sum _ { m = 1 } ^ { J } N _ { m } \cdot \bar { c } _ { \mathrm { s t e p } } ( R ^ { * } ) } \\ { = \left( \displaystyle \sum _ { m = 1 } ^ { J } N _ { m } \right) \bar { c } _ { \mathrm { s t e p } } ( R ^ { * } ) = N \cdot \bar { c } _ { \mathrm { s t e p } } ( R ^ { * } ) , } \end{array}\tag{62}
$$

The additive budget upper-bounds the expected total cost of reuse.

Step 4: Success probability across stages. Assume each stage reaches a well-defined final state after its prescribed $N _ { m }$ accepted steps. Let $S _ { m }$ denote stage-m completion with every accepted state correct. Assume that completing all stages correctly yields a correct final answer under the prescribed extraction. The probability chain rule gives:

$$
P \left( \bigcap _ { m = 1 } ^ { J } S _ { m } \right) = P ( S _ { 1 } ) \prod _ { m = 2 } ^ { J } P \left( S _ { m } \left| \bigcap _ { j = 1 } ^ { m - 1 } S _ { j } \right. \right) .\tag{63}
$$

As in Step 2 of the proof of Prop. 5, completion of all earlier stages with correct states gives stage m a correct accepted input state. The uniform bounds give per-step failure probability at most $\rho ^ { R } + \alpha$ Applying Prop. 4 with target failure probability $\delta _ { m }$ and $N _ { m }$ steps yields:

$$
P \left( S _ { m } \left| \bigcap _ { j = 1 } ^ { m - 1 } S _ { j } \right. \right) \ \geq \ 1 - \delta _ { m } , \quad { \mathrm { f o r ~ a l l ~ } } m .\tag{64}
$$

Substituting these conditional stage guarantees into Eq. (63) gives

$$
A _ { \mathrm { v s r } } ( x ) ~ \ge ~ P \biggl ( \bigcap _ { m = 1 } ^ { J } S _ { m } \biggr ) ~ \ge ~ \prod _ { m = 1 } ^ { J } ( 1 - \delta _ { m } ) ,\tag{65}
$$

Step 5: Product bounds. The product of stage guarantees obeys the following exact inequality:

$$
\prod _ { m = 1 } ^ { J } ( 1 - \delta _ { m } ) \geq 1 - \sum _ { m } \delta _ { m } .
$$

For $J = 1$ there is equality. Suppose the inequality holds for $J - 1$ factors; the induction step gives

$$
\begin{array} { r l r } {  { \prod _ { m = 1 } ^ { J } ( 1 - \delta _ { m } ) = \prod _ { m = 1 } ^ { J - 1 } ( 1 - \delta _ { m } ) \cdot ( 1 - \delta _ { J } ) \geq \Big ( 1 - \sum _ { m = 1 } ^ { J - 1 } \delta _ { m } \Big ) ( 1 - \delta _ { J } ) } } \\ & { } & { = 1 - \sum _ { m = 1 } ^ { J } \delta _ { m } + \delta _ { J } \sum _ { m = 1 } ^ { J - 1 } \delta _ { m } \geq 1 - \sum _ { m = 1 } ^ { J } \delta _ { m } , } \end{array}
$$

where the last step drops the nonnegative cross term. Therefore:

$$
\prod _ { m = 1 } ^ { J } ( 1 - \delta _ { m } ) \ \geq \ 1 - \sum _ { m = 1 } ^ { J } \delta _ { m } = 1 - \delta .\tag{66}
$$

The exact product inequality therefore gives the desired guarantee $A _ { \mathrm { v s r } } ( x ) \geq 1 - \delta .$

Exponential lower bound: We use $1 - x \geq e ^ { - x / ( 1 }$ <sup>−x)</sup> for $x \in [ 0 , 1 )$ , which follows from log ${ \mathrm { \Omega } } _ { \beta } ( 1 - x ) \geq$ $- x / ( 1 - x )$ . To verify it, let $f ( x ) = \log ( 1 - x ) + x / ( 1 - x )$ ; then $f ( 0 ) = 0$ and $f ^ { \prime } ( x ) = x / ( 1 - x ) ^ { 2 } \geq 0$ For $\delta _ { m } = \delta N _ { m } / N \ll 1$ , we have $1 / ( 1 - \delta _ { m } ) = 1 + \delta _ { m } + O ( \delta _ { m } ^ { 2 } ) \approx 1$ , so $\delta _ { m } / ( 1 - \delta _ { m } )$ ≈ $\delta _ { m } + \delta _ { m } ^ { 2 }$ Substitution gives the following exponential comparison for the product of stage guarantees:

$$
\prod _ { m = 1 } ^ { J } ( 1 - \delta _ { m } ) \geq \exp \left( - \sum _ { m = 1 } ^ { J } \frac { \delta _ { m } } { 1 - \delta _ { m } } \right) \approx \exp \left( - \delta - \sum _ { m = 1 } ^ { J } \delta _ { m } ^ { 2 } \right) .\tag{67}
$$

For $\delta _ { m } = { \cal { O } } ( \delta / J )$ , the second-order term is $\begin{array} { r } { \sum _ { m } \delta _ { m } ^ { 2 } = O ( \delta ^ { 2 } / J ) } \end{array}$ . As the largest stage budget vanishes, we obtain the exponential bound below; the linear approximation also requires $\delta  0 \backslash$

$$
A _ { \mathrm { v s r } } ( x ) ~ \geq ~ \exp ( - \delta - o ( \delta ) ) ~ \approx ~ 1 - \delta .\tag{68}
$$

Together with Step 3, this establishes $A _ { \mathrm { v s r } } ( x ) \geq \prod _ { m } ( 1 - \delta _ { m } ) \geq 1 - \delta$ and an additive computation budget under proportional failure allocation, provided $N \alpha < \delta$

## B.8 Complete-Sampling Background

## B.8.1 Accuracy Compositionality under Independent Sampling

At budget K, accuracy compositionality means that composite accuracy equals $A _ { \pmb \theta , K } ( x _ { 1 } ) A _ { \pmb \theta , K } ( x _ { 2 } )$ the probability of solving both subproblems in separate, independent K-attempt experiments. In this established term, accuracy refers to Pass@K coverage. For the following comparison, write $a _ { i } =$ $A _ { \pmb \theta , 1 } ( x _ { i } ) \in ( 0 , 1 )$ for $i \in \{ 1 , 2 \}$ . Assume that each subproblem has the same probability of being answered correctly inside a composite run as in its standalone run, and that the two subproblem correctness indicators are independent within each composite run. Together with independence across the K runs, these assumptions give:

$$
\begin{array} { r l } & { A _ { \theta , 1 } ( x _ { 1 } \oplus x _ { 2 } ) = a _ { 1 } a _ { 2 } , } \\ & { A _ { \theta , K } ( x _ { 1 } \oplus x _ { 2 } ) = 1 - ( 1 - a _ { 1 } a _ { 2 } ) ^ { K } < A _ { \theta , K } ( x _ { 1 } ) A _ { \theta , K } ( x _ { 2 } ) , \quad K \geq 2 . } \end{array}\tag{69}
$$

For $K = 1$ , the corresponding product relation holds with equality. The strict gap for $K \geq 2$ arises because each composite success requires one run to solve both subproblems, whereas separately covered subproblems may be solved by different runs. For example, when both subproblems have single-run accuracy 0.5 and $K = 2$ , complete-solution success is 0.4375, compared with 0.5625 for separate sampling. This requirement that one run solve both subproblems is the single-run requirement under independent sampling. The event inclusion behind it holds without the product assumptions: a pool containing a jointly correct run also covers each subproblem separately, while the converse need not hold (Prop. 2). The comparison fixes attempts per subproblem; it does not equate measured token costs across prompt formats. This bottleneck is a property of sampling and aggregation, and would also occur for modular stochastic solvers with independent errors.

## B.8.2 Sampling Cost as Task Length Grows

To compare computational cost as task length grows at a fixed target success rate, we consider decompositions indexed by N and adopt the idealized scaling assumptions of Zhang et al. (2025b). For a fixed model and decoding configuration, assume:

$$
C _ { \theta , 1 } ( x ) = \Theta ( N ) , \qquad A _ { \theta , 1 } ( x ) = e ^ { - \lambda _ { \theta } N } , \qquad \lambda _ { \theta } > 0 .\tag{70}
$$

$\lambda _ { \theta }$ is the per-step accuracy decay rate, and linear cost means expected generated tokens scale linearly with N up to constant factors. For example, the exponential accuracy relation holds when steps are correct

independently with probability $1 - \varepsilon .$ , where $0 < \varepsilon < 1$ , and the final answer is correct if and only if all N steps are correct (Barlow and Proschan, 1996). In this case:

$$
A _ { \pmb \theta , 1 } ( \pmb x ) = ( 1 - \varepsilon ) ^ { N } = e ^ { - \lambda _ { \pmb \theta } N } , \qquad \lambda _ { \pmb \theta } = - \log ( 1 - \varepsilon ) .\tag{71}
$$

For single-run accuracy $p = A _ { \pmb \theta , 1 } ( x ) \in ( 0 , 1 )$ , target reliability $1 - \delta$ requires at least

$$
K _ { \delta } ( N ) = \left\lceil \frac { \log \delta } { \log ( 1 - p ) } \right\rceil .\tag{72}
$$

At fixed $\delta ,$ the exponential accuracy law gives the asymptotic total generation cost

$$
C _ { \mathrm { i n d e p } } ( N , \delta ) = K _ { \delta } ( N ) C _ { \pmb { \theta } , 1 } ( x ) = \Theta \Bigl ( N e ^ { \lambda _ { \theta } N } \log ( 1 / \delta ) \Bigr ) .\tag{73}
$$

This restart cost motivates the following analysis of reliability and cost under state reuse.

## B.9 Sufficient Budgets and Stage Composition

The following bounds give sufficient retry budgets and stage-composition guarantees. They follow from the local failure decomposition and remain valid under the execution assumptions of Section 3. The expected-work theorem sharpens the bound based on the reserved budget while retaining these guarantees. Summing the first-failure probabilities gives the trajectory bound:

$$
P ( { \mathrm { t r a j e c t o r y ~ f a i l u r e } } ) \leq N \big ( \rho ^ { R } + \alpha \big ) .\tag{74}
$$

For $0 < \rho < 1$ and $N \alpha < \delta .$ , a sufficient common cap is:

$$
R ^ { * } = \left\lceil \frac { \log ( N / ( \delta - N \alpha ) ) } { \log ( 1 / \rho ) } \right\rceil .\tag{75}
$$

For stages of $N _ { 1 } , N _ { 2 }$ steps, allocate $\delta _ { i } = \delta N _ { i } / ( N _ { 1 } + N _ { 2 } )$ . Let $S _ { i }$ denote correct completion of stage i. If every correct stage-1 output supports the stage-2 guarantee and $S _ { 1 } \cap S _ { 2 }$ yields the target answer, then:

$$
A _ { \mathrm { v s r } } ( x _ { 1 } \oplus x _ { 2 } ) \geq P ( S _ { 1 } \cap S _ { 2 } ) = P ( S _ { 1 } ) P ( S _ { 2 } \mid S _ { 1 } ) \geq ( 1 - \delta _ { 1 } ) ( 1 - \delta _ { 2 } ) \geq 1 - \delta .\tag{76}
$$

Let $c _ { \mathrm { g e n } } , c _ { \mathrm { v e r } }$ bound per-attempt generation and checking cost at all reached states, with other processing included in these units. The expected total cost is bounded by the deterministic budget:

$$
C _ { \mathrm { v s r } } ( N , \delta ) \leq \overline { { C } } _ { \mathrm { v s r } } ( N , \delta ) : = N R ^ { * } ( c _ { \mathrm { g e n } } + c _ { \mathrm { v e r } } ) .\tag{77}
$$

For fixed $\rho \in ( 0 , 1 )$ $N \alpha \leq$ cδ with fixed $c < 1$ , and bounded per-attempt costs, the sufficient scaling remains:

$$
R ^ { * } = \Theta \bigl ( \log ( N / \delta ) \bigr ) , \qquad C _ { \mathrm { v s r } } = O \bigl ( N \log ( N / \delta ) \bigr ) .\tag{78}
$$

At a fixed correct input state s, let $T _ { t }$ count local attempts. The truncated geometric identity gives:

$$
\mathbb { E } [ T _ { t } \mid s ] = \sum _ { r = 1 } ^ { R } \rho _ { t } ( s ) ^ { r - 1 } = { \frac { 1 - \rho _ { t } ( s ) ^ { R } } { 1 - \rho _ { t } ( s ) } } \leq { \frac { 1 } { 1 - \rho } } .\tag{79}
$$

The product factors are conditional stage guarantees, rather than standalone accuracies under different prompts. If per-attempt cost grows, its dependence on task length must remain in the cost bound.

## B.10 Proofs for Reliability and Expected Work

## B.10.1 Exact Reliability and Minimal Common Cap

Let $\mathcal { F } _ { t }$ contain the complete history at entry to step t, including the event that the step is reached. At a fixed correct entry history, put $q = \rho _ { t } ( s )$ and $w = \mathrm { P r } ( \mathcal { G } _ { t } = 0 , \mathcal { V } _ { t } = 1 \mid \mathcal { F } _ { t } )$ . The probability of first accepting an incorrect state on attempt r is $q ^ { r - 1 } w$ . Summing over $r \leq R$ and dividing by $1 - q ^ { R }$ gives $\alpha _ { t } ( R ; s ) = w / ( 1 - q )$ whenever $q < 1$ . This establishes Eq. (4) for every finite cap under the fixed retry law. The probability of correct advancement is exactly $( 1 - q ^ { R } ) ( 1 - \alpha _ { t } ( s ) )$ .

On $H _ { t - 1 }$ , all entry histories are correct, so their conditional average has correct-advancement probability at least $( 1 - \rho ^ { R } ) ( 1 - \alpha )$ . Multiplying these conditional lower bounds proves Thm. 1. No independence across entry states is needed. The homogeneous construction in that theorem realizes equality and, by making errors irreversible, equates final-answer and trajectory success. Consequently the lower bound is also the infimum of final-answer success over the specified model class.

Write $d = ( 1 - \delta ) ^ { 1 / N }$ . For $0 < \rho < 1$ , a common cap certifies the target over the entire class exactly when

$$
( 1 - \rho ^ { R } ) ( 1 - \alpha ) \geq d .\tag{80}
$$

For finite $R , 1 - \rho ^ { R } < 1$ , so this is possible if and only if $\alpha < 1 - d .$ . In that case its minimal integer solution is

$$
R _ { \operatorname* { m i n } } = \operatorname* { m a x } \left\{ 1 , \left\lceil \frac { \log \left( 1 - d / ( 1 - \alpha ) \right) } { \log { \rho } } \right\rceil \right\} .\tag{81}
$$

Both logarithms are negative; division by log ρ reverses the inequality used to solve for R. If $\alpha = 1 - d ,$ the target is attained only in the limit of unbounded retries. If $\alpha > 1 - d ,$ even that limit cannot give a uniform guarantee. When $\rho = 0$ , the exact condition is $( 1 - \alpha ) ^ { N } \geq 1 - \delta$ , and $R = 1$ suffices. $\mathrm { A t } \alpha = 1$ no positive reliability guarantee follows.

A convenient sufficient choice comes from

$$
\begin{array} { r } { 1 - \left[ ( 1 - \rho ^ { R } ) ( 1 - \alpha ) \right] ^ { N } \leq N \big [ \rho ^ { R } + ( 1 - \rho ^ { R } ) \alpha \big ] \leq N ( \rho ^ { R } + \alpha ) . } \end{array}\tag{82}
$$

The first inequality is $1 - ( 1 - u ) ^ { N } \leq N u$ for $u \in [ 0 , 1 ]$ . Solving $\rho ^ { R } \leq ( \delta - N \alpha ) / N$ gives Eq. (75) whenever $N \alpha < \delta . { \mathrm { A s } } N \to \infty$ at fixed δ, $1 - ( 1 - \delta ) ^ { 1 / N } = - \log ( 1 - \delta ) / N + O ( N ^ { - 2 } )$ . Replacing the numerator by $\delta$ additionally requires a small-δ approximation.

## B.10.2 Heterogeneous Steps and Multiple Stages

Suppose step t has predetermined cap $R _ { t }$ and bounds $\rho _ { t } < 1$ and $\alpha _ { t } < 1$ valid at every correct entry history. Conditioning on $H _ { t - 1 }$ as above gives

$$
\operatorname* { P r } ( H _ { N } ) \geq \prod _ { t = 1 } ^ { N } ( 1 - \rho _ { t } ^ { R _ { t } } ) ( 1 - \alpha _ { t } ) .\tag{83}
$$

A bound averaged over standalone inputs need not hold for states produced by earlier stages.

Partition the steps into stages of sizes $N _ { 1 } , \ldots , N _ { m }$ . If stage $j$ completes correctly with conditional probability at least $1 - \delta _ { j }$ from every correct entry state, then the probability of correct completion of all stages is at least

$$
\prod _ { j = 1 } ^ { m } ( 1 - \delta _ { j } ) \geq 1 - \sum _ { j = 1 } ^ { m } \delta _ { j } .\tag{84}
$$

For the common bounds of the main text, allocating $\delta _ { j } = \delta N _ { j } / N$ and using $\operatorname { E q . }$ . (75) supplies such guarantees. These factors need not equal standalone success probabilities. The second stage can meet its target with or without a correct preceding state, despite different success rates.

## B.10.3 Expected Attempts: Full Conditional Proof

Let $I _ { t }$ indicate that step t is reached, and define its conditional single-attempt rejection probability $\rho _ { t }$ given $\mathcal { F } _ { t }$ when $I _ { t } = 1$ . This abbreviation records rejection conditional on the entry history, at both correct and incorrect states. Assign $\rho _ { t } = 0$ on unreachable histories, which do not contribute to the sums below. Let $T _ { t } = 0$ when $I _ { t } = 0$ and otherwise count the attempts at this step. Conditional i.i.d. retries imply

$$
\mathbb { E } [ T _ { t } \mid { \mathcal { F } } _ { t } ] = I _ { t } \sum _ { r = 0 } ^ { R - 1 } \rho _ { t } ^ { r } .\tag{85}
$$

This identity includes $\rho _ { t } = 1$ , for which the sum is R. For any $\theta \in ( 0 , 1 )$ , the low-rejection contribution satisfies

$$
\sum _ { t = 1 } ^ { N } \mathbb { E } [ T _ { t } \mathbf { 1 } \{ \rho _ { t } \leq \theta \} ] \leq \frac { 1 } { 1 - \theta } \sum _ { t = 1 } ^ { N } \operatorname* { P r } ( I _ { t } = 1 , \rho _ { t } \leq \theta ) \leq \frac { N } { 1 - \theta } .\tag{86}
$$

Let $F _ { t }$ denote exhaustion at step t and $\begin{array} { r } { Z = \sum _ { t } I _ { t } \mathbf { 1 } \{ \rho _ { t } > \theta \} } \end{array}$ . Exhaustion has conditional probability $\rho _ { t } ^ { R }$ at a reached step, hence

$$
\theta ^ { R } \mathbb { E } [ Z ] \leq \sum _ { t = 1 } ^ { N } \mathbb { E } [ I _ { t } \mathbf { 1 } \{ \rho _ { t } > \theta \} \rho _ { t } ^ { R } ] = \sum _ { t = 1 } ^ { N } \operatorname* { P r } ( F _ { t } , \rho _ { t } > \theta ) \leq 1 .\tag{87}
$$

The last inequality uses mutual exclusivity of exhaustion events under global stopping. Since a highrejection visit costs at most $R ,$ its total expected contribution is at most $R \dot { \theta } ^ { - R }$ . Adding the two contributions gives

$$
\mathbb { E } [ W ] \leq \frac { N } { 1 - \theta } + \frac { R } { \theta ^ { R } } .\tag{88}
$$

This estimate requires no bound on $\rho _ { t } .$ , including at states reached after an incorrect acceptance.

For the uniform upper bound, put $\ell _ { N } = \log ( N + 1 )$ and $\theta = e ^ { - \ell _ { N } / ( 2 R ) }$ . The inequality $e ^ { u } \geq 1 +$ u yields $( 1 - e ^ { - u } ) ^ { - 1 } \leq 1 + 1 / u .$ , so

$$
\mathbb { E } [ W ] \le N + \frac { 2 N R } { \ell _ { N } } + R \sqrt { N + 1 } \le N + \frac { 3 N R } { \ell _ { N } } .\tag{89}
$$

To verify the last step for every $N \geq 1$ , set $x = \sqrt { N + 1 } \ge 1$ . The function $x - x ^ { - 1 } - 2$ log x is zero at one and has derivative $( x - 1 ) ^ { 2 } / x ^ { 2 } \ge 0$ . Thus log $( N + 1 ) \sqrt { N + 1 } \leq N$ . Independently of this expectation bound, the deterministic cap gives $\mathbb { E } [ W ] \le N R$

For the lower bound, fix $q = ( N + 1 ) ^ { - 1 / R }$ at every reached state and always advance on acceptance. Let $v = 1 - q ^ { R } = N / ( N + 1 )$ . The probability of reaching step t is $v ^ { t - 1 }$ , and its conditional expected attempts equal $v / ( 1 - q )$ . Therefore

$$
\mathbb { E } [ W ] = \frac { v } { 1 - q } \sum _ { t = 1 } ^ { N } v ^ { t - 1 } = \frac { N ( 1 - v ^ { N } ) } { 1 - e ^ { - \ell _ { N } / R } } .\tag{90}
$$

The binomial inequality $( 1 + 1 / N ) ^ { N } \geq 2$ gives $1 - v ^ { N } \ge 1 / 2$ . Also $1 - e ^ { - u } \leq \operatorname* { m i n } \{ 1 , u \}$ , so

$$
\mathbb { E } [ W ] \ge \frac { N } { 2 } \operatorname* { m a x } \left\{ 1 , \frac { R } { \ell _ { N } } \right\} \ge \frac { 1 } { 4 } \left( N + \frac { N R } { \ell _ { N } } \right) .\tag{91}
$$

Combining the upper and lower estimates yields the finite bounds

$$
{ \frac { 1 } { 4 } } \left( N + { \frac { N R } { \log ( N + 1 ) } } \right) \leq \operatorname* { s u p } _ { P \in { \mathcal { P } } _ { N , R } } \mathbb { E } _ { P } [ W ] \leq N + { \frac { 3 N R } { \log ( N + 1 ) } } .\tag{92}
$$

This proves the matching order for the entire class $\mathcal { P } _ { N , R }$ . It is a worst-case expectation over the class, not a lower bound on every individual process.

## B.10.4 Combining Reliability and Work

For fixed $\rho \in \left( 0 , 1 \right)$ and $\alpha ~ \leq ~ c \delta / N$ with $c < 1$ , the cap in Eq. (75) is at most $1 + \log ( N / ( 1 -$ $c ) \delta ) ) / \log ( 1 / \rho )$ At fixed $\delta ,$ this is ${ \cal O } ( \log N )$ for $N \ \geq \ 2$ Theorems 1 and 2 therefore imply $\Pr ( H _ { N } ) \geq 1 - \delta$ and $\mathbb { E } [ W ] = O ( N )$ . Since $W \ge N \mathbf { 1 } _ { H _ { N } }$ , its expectation is also at least $( 1 - \delta ) N$ A uniform per-attempt cost bound $g _ { N }$ on all reached states gives expected total cost $O ( N g _ { N } )$ and a deterministic total-cost cap for the entire run of O(N log ${ \cal { N } } g _ { N } )$ .

If $\delta _ { N } \geq N ^ { - d }$ for a fixed $d > 0 ,$ , the cap remains ${ \cal O } ( \log N )$ and the same expectation upper bound follows. For smaller targets, substitute the actual R into Thm. 2. For example, $\delta _ { N } = e ^ { - \kappa N }$ can require $R = \Theta ( N )$ under the sufficient allocation, yielding an $O ( N ^ { 2 } /$ log N) work bound from the unrestricted process class. This is an upper bound for reliable processes; the matching witness for the unrestricted class is not asserted to meet that exponentially small failure target.

Proof of the acceptance-sensitive bound. For the cap in Eq. (9), every correct prefix exhausts with probability at most $( 1 - c ) \delta / N$ . Together with $\alpha \leq c \delta / N$ , the first-failure bound gives $\Pr ( H _ { N } ) \geq 1 - \delta$ For fixed $c , \delta , \log ( N / ( ( 1 - c ) \delta ) ) / \log ( N + 1 )$ is bounded uniformly for $N \geq 1$ . The ceiling and $h _ {  { { N } } } \leq 1$ therefore give $R _ { N } / \log ( N + 1 ) = O ( 1 / h _ { N } )$ . Applying Thm. 2 proves $\mathbb { E } [ W ] = O ( N / h _ { N } )$ . For the lower construction, set $q = 1 - h _ { N }$ and accept only correct states. Writing $v = 1 - q ^ { R _ { N } }$ , each step’s expected attempts equal $v / h _ { N }$ and the success probability is $\begin{array} { r } { v ^ { N } \geq 1 - \delta . } \end{array}$ . Thus $v \geq 1 - \delta$ and the probability of reaching any step is at least $1 - \delta .$ , giving $\mathbb { E } [ W ] \geq N ( 1 - \delta ) ^ { 2 } / h _ { N }$ . This also covers $h _ { N } = 1$ , where exactly $N$ attempts are executed. If every attempt has cost $g _ { N }$ , the cost upper bound has a matching construction in those units as well.

## B.10.5 Proofs for the Binary-Checking Examples

Binary-checking model. At every correct prefix, a new candidate is correct with fixed probability $p \in ( 0 , 1 )$ . Given its correctness, repeated checks are independent: a correct candidate passes with probability a, an incorrect one with probability $b ,$ where $0 < b < a < 1$ . The parameters are fixed across correct prefixes and task lengths. Each candidate and its checks form one conditionally independent attempt at a fixed entry history. Different random seeds do not by themselves establish this noise model; shared candidate difficulty or systematic errors can violate it. For unanimity among k checks, the acceptance probability and accepted-state error are

$$
h _ { k } = p a ^ { k } + ( 1 - p ) b ^ { k } , \qquad \alpha _ { k } = \frac { ( 1 - p ) b ^ { k } } { p a ^ { k } + ( 1 - p ) b ^ { k } } .\tag{93}
$$

All k checks are executed when full-check costs are counted below.

Unanimity: precision and acceptance probability. The probability that all k checks accept a correct candidate is $a ^ { k }$ , and it is $b ^ { k }$ for an incorrect candidate. Bayes’ rule gives Eq. (93). For a target $\epsilon \in ( 0 , 1 )$ the precision condition is

$$
\alpha _ { k } \leq \epsilon \quad \Longleftrightarrow \quad \left( { \frac { b } { a } } \right) ^ { k } \leq { \frac { \epsilon p } { ( 1 - \epsilon ) ( 1 - p ) } } .\tag{94}
$$

For the smallest integer $k \geq 1$ satisfying this with $\epsilon = c \delta / N$ , where $p , a , b , c , \delta$ are fixed and $0 < c < 1$ we have $k = \log N / \log ( a / b ) + O ( 1 )$ . Hence $a ^ { k } = \Theta ( N ^ { - \gamma } )$ for $\gamma = - \log a / \log ( a / b ) > 0$ . Moreover, $h _ { k } = p a ^ { k } / ( 1 - \alpha _ { k } ) = \Theta ( N ^ { - \gamma } )$ . To make $( 1 - h _ { k } ) ^ { R }$ of order $1 / N$ , one needs $R = \Theta ( \log N / [ - \log ( 1 -$ $h _ { k } ) ] ) = \Theta ( N ^ { \gamma } \log N )$ . Together with the accepted-state error allocation, a sufficiently large constant in this cap guarantees $\Pr ( H _ { N } ) \geq 1 - \delta$

Unanimity: expected-cost lower bound. Use the homogeneous law at every correct prefix and make any incorrect acceptance irreversible. Write $h = h _ { k } , \alpha = \alpha _ { k }$ , and $u = ( 1 - ( 1 - h ) ^ { R } ) ( 1 - \alpha )$

Then $\mathrm { P r } ( H _ { N } ) = u ^ { N }$ . If $u ^ { N } \geq 1 - \delta$ , each correct-prefix reach probability $\operatorname* { P r } ( H _ { t - 1 } ) = u ^ { t - 1 }$ is at least $1 - \delta$ . Also $1 - ( 1 - h ) ^ { R } \geq u \geq 1 - \delta$ . The expected attempts from each such prefix are $[ 1 - ( 1 - h ) ^ { R } ] / h \geq ( 1 - \delta ) / h$ . Thus

$$
\mathbb { E } [ W ] \geq \sum _ { t = 1 } ^ { N } \operatorname* { P r } ( H _ { t - 1 } ) \frac { 1 - ( 1 - h ) ^ { R } } { h } \geq \frac { N ( 1 - \delta ) ^ { 2 } } { h } = \Omega ( N ^ { 1 + \gamma } ) .\tag{95}
$$

Counting k unit-cost checks on every attempt gives $\Omega ( N ^ { 1 + \gamma }$ log N) checking cost. If checking stops at the first failed check, the same candidate lower bound applies, but multiplying by k is no longer justified. The lower bound counts work on correct prefixes only, so arbitrary behavior after an error cannot invalidate it. For the matching upper bound, choose $R _ { N }$ from Eq. (9) with $h _ { N } = h _ { k }$ . All k unit-cost checks are executed on each candidate and generation has uniformly bounded cost, so $g _ { N } = \Theta ( k )$ on all reached states. Cor. 1 gives $O ( N k / h _ { k } ) = O ( N ^ { 1 + \gamma } \log N )$ , matching the lower bound. The upper bound permits arbitrary rejection at incorrect states, subject to the same conditional i.i.d. local retry and global stopping rules.

Majority: precision with nonvanishing acceptance probability. For odd $k ,$ let $u _ { k }$ be the probability of majority acceptance given a correct candidate and $v _ { k }$ the same probability given an incorrect candidate. Under $a > 1 / 2 > b _ { \mathrm { \scriptscriptstyle 2 } }$ , Hoeffding’s inequality yields $1 - u _ { k } \le e ^ { - \hat { 2 k } ( a - 1 / 2 ) ^ { 2 } }$ and $v _ { k } \leq e ^ { - 2 k ( 1 / 2 - b ) ^ { 2 } }$ . For all sufficiently large $k , u _ { k } \ge 1 / 2$ , so

$$
h _ { k } = p u _ { k } + ( 1 - p ) v _ { k } \geq p / 2 , \qquad \alpha _ { k } = \frac { ( 1 - p ) v _ { k } } { h _ { k } } \leq \frac { 2 ( 1 - p ) } { p } e ^ { - 2 k ( 1 / 2 - b ) ^ { 2 } } .\tag{96}
$$

Choose an odd k of order log $( N / \delta )$ with a sufficiently large constant to get $\alpha _ { k } \leq c \delta / N$ . The correct-prefix rejection bound $\rho = 1 - p / 2 < 1$ is independent of N. A cap $R = O ( \log ( N / \delta ) )$ now certifies reliability. For fixed $\delta ,$ conditional i.i.d. candidate attempts at every reached state and uniform per-check cost bounds give $\mathbb { E } [ W ] = O ( N )$ and $\mathbb { E } [ \mathrm { c o s t } ] = O ( N \log N )$ . The pathwise cap is $N R k = O ( N \log ^ { 2 } N )$ in the same units. The precision claim uses the noise law only at correct prefixes; the expected-work bound also requires the fixed retry law at incorrect states.

Dependence and boundary cases. If repeated checks share a latent blind spot, their false-acceptance probability need not decrease exponentially with k. Independent random seeds do not remove dependence caused by a shared candidate or systematic misunderstanding. The conditional Bernoulli law above fixes a and b within each correctness class and excludes such additional latent variation. The unanimity exponent assumes $a < 1$ and $b > 0 ;$ perfect true acceptance or zero false acceptance needs a separate calculation. Majority’s fixed threshold at one half additionally requires $a > 1 / 2 > b$

## B.10.6 Relation to Classical Stopped Testing

The stopping pattern. Each attempt is a binary trial: acceptance completes a step; rejection leaves it unfinished. The run ends at the Nth acceptance or the first run of $R$ consecutive rejections. With constant rejection probability, this is a special case of total-success/consecutive-failure stopping in startup demonstration testing. Gera (2013) studies such criteria under a previous-sum-dependent model; Balakrishnan et al. (2014) review i.i.d., Markov-dependent, exchangeable, and multistate formulations.

We connect worst-case expected work over $\mathcal { P } _ { N , R }$ to correctness at correct prefixes.

A prescribed homogeneous law. For rejection probability $q ~ \in ~ ( 0 , 1 )$ at every step, let $a _ { R } ( q ) =$ $\textstyle \sum _ { j = 0 } ^ { R - 1 } q ^ { j } = ( 1 - q ^ { R } ) / ( 1 - q )$ . A step takes $a _ { R } ( q )$ attempts in expectation and advances with probability $1 - q ^ { R }$ , so

$$
\mathbb { E } _ { q } [ W ] = a _ { R } ( q ) \sum _ { t = 0 } ^ { N - 1 } ( 1 - q ^ { R } ) ^ { t } = \frac { 1 - q ^ { R } } { 1 - q } \frac { 1 - ( 1 - q ^ { R } ) ^ { N } } { q ^ { R } } .\tag{97}
$$

At $q = 0$ the work is $N ;$ at $q = 1$ it is $R .$ . This fixed-law calculation supplies the lower construction in Thm. $2 ;$ it does not bound every process in $\mathcal { P } _ { N , R }$

Exact recursion for the process class. Let $V _ { n }$ be the supremum of expected remaining attempts with n steps left and the common cap R. Then $V _ { 0 } = 0$ and

$$
V _ { n } = \operatorname * { m a x } _ { q \in [ 0 , 1 ] } \left\{ a _ { R } ( q ) + ( 1 - q ^ { R } ) V _ { n - 1 } \right\} .\tag{98}
$$

Condition on the complete entry history. Current-step expected work is $a _ { R } ( q )$ ; acceptance leaves at most $V _ { n - 1 }$ expected attempts, while exhaustion ends the run. Continuity on [0, 1] guarantees a maximum. Choose a maximizing $q$ and, after acceptance, repeat for $n - 1$ steps. Induction gives equality and an attaining process whose rejection probability depends only on the number of remaining steps. History dependence alone is not the distinction: the unrestricted supremum reduces to this scalar recursion. Thm. 2 gives its order uniformly in N, R without solving the successive maximizations. This characterizes the process class; a model need not realize its maximizing law.

What is needed for reliable reuse. A uniform acceptance lower bound h on all reached states immediately gives $\mathbb { E } [ W ] \leq N / h$ by summing geometric waiting times. In Cor. 1, the acceptance lower bound holds only at correct prefixes. An incorrect acceptance may lead to states with arbitrary rejection proba bilities. Global stopping bounds their work without extending the correct-prefix acceptance assumption to those states. If rejection changes the retry law, rejecting the first $R - 1$ candidates and accepting the last at every step gives $W = N R$ . If exhaustion permits fallback and continuation, always rejecting likewise gives $W = N R$ . These constructions show why the execution assumptions matter. The lower witness for Thm. 2 completes all steps with probability $( { N } / ( { N + 1 } ) ) ^ { N }$ , approaching $e ^ { - 1 } \colon$ ; Cor. 1 supplies a separate matching construction within its reliable class.

Restart schedules and checking costs. Luby et al. (1993) optimize schedules for repeated complete runs of Las Vegas algorithms. Our cap is fixed, and exhaustion ends the task; the result does not optimize restart schedules or decomposition. Selective classification studies the risk–coverage tradeoff (El-Yaniv and Wiener, 2010). Here its coverage corresponds to acceptance probability, distinct from answer coverage in our experiments. MAKER already accounts for voting cost and the probability that a generated response survives red-flagging (Meyerson et al., 2025). Precision and acceptance probability are thus established concerns. Here they enter a first-acceptance process with a finite cap through $g _ { N } / h _ { N }$ and the accepted-state error requirement in Cor. 1. Standard Bayes and concentration calculations (Hoeffding, 1963) show why low error need not imply low cost.

Composite pair prompt.   
Answer the following questions in order:   
Q1. {question 1}   
Q2. {question 2}

## C Implementation Details

This section records the implementation details used for the LoRe-Compo experiments in Section 4.

The protocol first generates candidate answers to x , selects intermediate answers, and uses them when generating answers to x<sub>2</sub>. It also retains fallback candidates that did not pass the check. This manuscript package documents the protocol but does not include the implementation.

The finite-budget arithmetic experiments in Section 4.4 use a separate complete-answer retry setup, described after the LoRe-Compo details; they do not perform digit-level state reuse.

## C.1 LoRe-Compo Implementation

Dataset format and connector prompt. The LoRe-Compo data file contains 500 unique MATH500 subproblems and 250 composite pairs. Readers can obtain the benchmark online from the LoRe authors’ repository at https://github.com/ASTRAL-Group/LoRe, under LoRe-Bench/LoRe-Compo/ compo\_bench\_math500.jsonl; this paper does not redistribute the records. Each subproblem record stores a MATH subject label, problem text, solution, and ground-truth answer. A pair links two subproblems, their answers, and this prompt:

The Pass@K baseline uses this pair prompt directly for independent sampling over complete runs. The subject labels are used only as a proxy for reduced concept overlap across subproblems; they do not imply probabilistic independence of model outputs.

Independent sampling prompts. For each standalone or composite problem, independent sampling uses the template Question: {problem} followed by Answer:. For composite pairs, {problem} is the pair prompt above. We sample K 1, 2, 4, 8, 16, 32, 64, 128 complete runs with a maximum generation length of 10,240 tokens. The reported runs use vLLM with seed 42, repetition penalty 1.05, and family-specific temperatures 0.6/0.7/0.8 for DeepSeek/Qwen3/GPT-OSS. For K=1, the executed parameters use top-p = 1.0 and top-k = 1; the do\_sample field is not consumed by the vLLM call, so Pass@1 is not greedy decoding in this code path. For K>1, the executed parameters use top-p = 0.7 and top-k = 50. The LoRe-Compo scripts pass these prompt strings directly to vLLM; model-specific chat templates are not applied in this implementation.

Two-stage protocol prompts. Stage 1 samples M answers to x<sub>1</sub> with its standalone prompt.

The main two-stage comparison generates M = 8 candidates and retains up to B = 4 intermediate answers for continuation. These answers pass the selection rule when it succeeds; otherwise they are fallback candidates. The budget-allocation sweep also evaluates $( M , B ) \in \{ ( 4 , 2 ) , ( 8 , 2 ) , ( 8 , 4 ) , ( 1 6 , 8 ) \}$ Stages 1 and 2 use the same family-specific temperatures, top-p = 0.7, top-k = 50, seed 42, repetition penalty 1.05, and 10,240-token cap. The main stage-2 prompt supplies the retained answer through the following answer\_only template:

Two-stage protocol: stage-2 prompt (answer\_only).   
Question 1 has been solved. The answer was: {q1 answer}   
======================   
Now solve Question 2:

{question 2}   
Important: Question 2 is a different problem.   
Think step by step and provide the answer for Question 2.

The prompt ablation additionally evaluates no\_context, which passes only x<sub>2</sub>, and full\_context, which passes the generated x reasoning text and retained x answer; retained fallback answers are not accepted. For full\_context, the implementation truncates the retained $x _ { 1 }$ context to the most recent portion when it exceeds the configured context budget. no\_context uses the same single-subproblem prompt for $x _ { 2 }$ , while full\_context adds the retained $x _ { 1 }$ reasoning and answer before Question 2.

Answer extraction and matching. Before extraction, generated text is split at </think> when a reasoning tag is present, and answer extraction is applied first to the post-thinking answer text. For math-style answers, extraction first uses the final \boxed{...} expression if present, then explicit answer patterns such as Answer:, The answer is, and Final answer:, and finally falls back to the last numeric expression. For composite answers, extraction first tries to split the generated text into Q1 and Q2 sections and extract one answer from each section. If this split fails, it falls back to the first two numeric expressions when available. Correctness uses numeric comparison with tolerance $1 0 ^ { - 6 }$ when both prediction and target are numeric; otherwise strings are stripped, lowercased, spaces are removed, and exact string equality is used. No computer algebra system (CAS) checks algebraic equivalence.

Oracle and self-consistency selection. Oracle selection compares each stage-1 candidate’s extracted $x _ { 1 }$ answer with the reference answer. If reference-matching candidates exist, the first B matches are retained and each receives a continuation.

If none matches, the fallback keeps the first B candidates so that continuation tokens and secondsubproblem success are still measured. All such paths use fallback routing, which the formal loop excludes by terminating without returning a state whenever selection fails.

Self-consistency selection counts the extracted $x _ { 1 }$ answers among the M candidates. If the most common answer has vote share at least 0.5, it keeps up to B candidates with that answer.

If no answer reaches this threshold, the empirical protocol keeps the first B candidates as a fallback for measurement; the formal loop would instead stop without returning a state. Ties are not broken by a separate randomized procedure: the implementation uses the first most common answer returned by the counter over generated candidates.

Comparing accepted SC answers against reference answers measures the error rate among accepted states in the evaluated sample. It does not estimate the uniform theoretical bound α across all steps, correct input states, and budgets. An error estimate that includes fallbacks instead describes all retained candidates, including those rejected by the selection rule.

Retained-path correctness fields. The implementation defines path-level fields for first-answer correctness, second-answer correctness, and their conjunction.

The implementation field final\_correct denotes second-subproblem success: whether any retained continuation solves $x _ { 2 } ,$ , regardless of the retained $x _ { 1 }$ answer’s correctness. Joint success requires both answers to be correct on a shared continuation path.

Joint success must be computed from the path-level field requiring both answers to be correct. Estimating the error rate among accepted states additionally requires excluding fallback paths.

The distinction matters when fallback candidates are retained or the check accepts an incorrect answer. With Oracle selection and at least one reference match, the reused intermediate answers are correct according to the answer matcher; this does not verify the full reasoning traces.

Case Example: One LoRe-Compo Pair. Consider the pair with MATH500 indices (477, 21). The first subproblem asks for the number of possible tens digits B such that the two-digit number $" B 6 "$ is a square, with answer 2. The second asks for the greatest integer less than $( { \sqrt { 7 } } + { \sqrt { 5 } } ) ^ { 6 }$ , with answer 13535. The independent-sampling baseline presents both subproblems in one prompt and succeeds only when a single sampled run returns the correct pair [2, 13535].

The two-stage protocol samples M solutions for the first subproblem and selects extracted answers using Oracle matching or SC voting. If an answer passes the selection rule, the model receives that accepted intermediate answer when solving the second subproblem; otherwise it continues from fallback candidates to measure second-subproblem coverage.

The example contrasts complete-solution sampling with separate continuation prompts. The second subproblem does not mathematically depend on the first answer.

Case Example: Answer Matching and Selection. The pair with MATH500 indices (277, 222) contains a first subproblem asking for $\sqrt { 5 3 }$ in simplest radical form, with target answer $\backslash \mathsf { s q r t } \{ \mathsf { 5 3 } \}$ For nonnumeric answers, the checker strips whitespace, lowercases strings, removes spaces, and then applies exact string matching. Thus \sqrt{53} is accepted, while a semantically equivalent expression may not be accepted unless extracted into the same normalized string. For generated composite answers, if $\boldsymbol { \mathrm { Q 1 } } / \boldsymbol { \mathrm { Q 2 } }$ section extraction fails, the extractor falls back to the first two numeric expressions in the post-</think> answer text.

For Oracle selection, if no stage-1 candidate matches the target, the empirical fallback keeps the first B candidates so that continuation budget is still measured; these paths are neither accepted states nor part of the formal product theorem. Consider these $M { = } 8$ candidate answers:

$$
2 0 0 , \ 2 0 0 , \ 2 0 0 , \ 2 0 0 , \ 2 1 0 , \ 3 0 0 , \ 2 0 0 0 , \ \mathrm { P } \mathrm { A R } S \mathrm { E } \_ \mathrm { R R O R } .
$$

The answer 200 has vote share $4 / 8 = 0 . 5$ , meeting the self-consistency threshold, so up to $B = 4$ candidates with that answer are kept. If the top vote share were $3 / 8 ,$ the protocol would retain the first B candidates as a fallback. Ties use the first most common answer encountered by the counter; no randomized tie-breaking rule is applied.

## C.2 Metric Definitions and Limits of the Reported Aggregates

Joint versus second-subproblem coverage. Let $i = 1 , \ldots , D$ index the $D = 2 5 0$ composite pairs, and let $\mathcal { R } _ { i }$ be the retained paths that receive continuations. Write $G _ { i b } ^ { ( 1 ) }$ and $G _ { i b } ^ { ( 2 ) }$ for the correctness indicators of the first and second answers on path $b \in \mathcal { R } _ { i }$ . The distinction between second-subproblem coverage and joint coverage is expressed by the statistics

$$
\widehat { A } _ { 2 } ^ { \mathrm { r e t } } = \frac { 1 } { D } \sum _ { i = 1 } ^ { D } { \bf 1 } \{ \exists b \in \mathcal { R } _ { i } : G _ { i b } ^ { ( 2 ) } = 1 \} ,\tag{99}
$$

$$
\widehat { A } _ { \mathrm { j o i n t } } ^ { \mathrm { r e t } } = \frac { 1 } { D } \sum _ { i = 1 } ^ { D } \mathbf { 1 } \{ \exists b \in \mathcal { R } _ { i } : G _ { i b } ^ { ( 1 ) } = G _ { i b } ^ { ( 2 ) } = 1 \} .\tag{100}
$$

Only the first statistic is reported for the two-stage experiments. The reported aggregates cannot replace missing path-level records. The second cannot be reconstructed by multiplying marginal accuracies or from ${ \widehat { A } } _ { 2 } ^ { \mathrm { { r e t } } }$ alone. Restricting $\mathcal { R } _ { i }$ to paths that passed selection would define accepted-path coverage; its denominator must still contain all $D$ pairs, with an empty path set contributing zero. These definitions supply no new measurements and do not make the multiple-path protocol equivalent to the formal loop.

Acceptance and fallback. Error among accepted intermediate answers requires comparing those answers to the reference, excluding every fallback that did not pass selection. A calculation must specify whether it weights individual retained answers or one selected state per pair; multiple retained copies of a modal answer need not define the same distribution as the theorem’s returned state. An empirical average under either choice would not establish a uniform bound over all correct prefixes. The reported continuation aggregates do not determine this error rate, the frequency of fallback, or the joint contribution of accepted and fallback paths. Oracle reference matching checks an extracted answer; in the full-context ablation, the retained reasoning text is not separately checked.

Partial successes in a common candidate pool. For K complete composite runs, let $G _ { i 1 }$ <sub>r</sub> and $G _ { i 2 r }$ denote subproblem-answer correctness in run r. Define

$$
J _ { i } ( K ) = \mathbf { 1 } \{ \exists r \leq K : G _ { i 1 r } = G _ { i 2 r } = 1 \} ,\tag{101}
$$

$$
U _ { i } ( K ) = \mathbf 1 \{ \exists r \leq K : G _ { i 1 r } = 1 \} \mathbf 1 \{ \exists r \leq K : G _ { i 2 r } = 1 \} .\tag{102}
$$

Then $J _ { i } ( K ) \le U _ { i } ( K )$ for every realized pool, without an independence assumption. The average $\begin{array} { r } { D ^ { - 1 } \sum _ { i } [ U _ { i } ( K ) - J _ { i } ( K ) ] } \end{array}$ would measure pools containing separately correct subproblem answers but no jointly correct run. It is an oracle coverage diagnostic, not a deployable answer-selection rule. The reported Pass@K values average $J _ { i } ( K )$ ; they cannot identify the difference $U _ { i } ( K ) - J _ { i } ( K )$

Comparability and uncertainty. The decoding differences and unspecified cross-budget nesting in Appendix C.1 prevent interpreting all composite rows as one controlled budget curve. Single-subproblem and paired-answer extraction also follow different fallback rules: the paired extractor can return the first two numeric expressions when section parsing fails, while semantic equivalence is not checked by a CAS. The available aggregate results do not quantify extraction errors or their differential effect between protocols. Scores depend on the specified prompts, answer matching, and extraction.

There are 250 pairs per model, giving 0.4-percentage-point resolution for each model’s pair-level accuracy. SC–Oracle differences of at most 1.6 points correspond to a net difference of at most four successful pairs; the number of pairs on which the selectors disagree cannot be inferred from these marginal scores. Paired uncertainty cannot be recovered from the rounded marginal scores alone; the shared test pairs also make models’ aggregate outcomes dependent. We report the observed differences without asserting statistical equivalence, significance, or robustness across decoding seeds.

## C.3 Controlled Arithmetic Implementation

The Base 7 and Base 10 experiments use the same complete-answer sampling structure. Digit count indexes problem size; no digit-position state is checked or reused. For each base, task, and digit count, the scripts pre-generate 50 problems using seed $4 2 \cdot 1 0 0 0 0 + 1 0 0 \cdot j + n$ , where $j = 0$ for addition, $j = 1$ for multiplication, and n is the digit count. The evaluated digit counts are $n \in \{ 1 , \ldots , 1 0 \}$ and the tasks are addition and multiplication. Each problem receives $R _ { \mathrm { m a x } } = 3 2$ independent attempts in one vLLM call, with checkpoints at $R \in \{ 2 , 4 , 8 , 1 6 , 3 2 \}$

Thus each model is evaluated on 2 bases  2 tasks  10 digit counts  50 problems, with 32 attempts per problem; each combination of model, base, task, and digit count contains 1,600 completions.

Arithmetic prompts. The Base 7 template specifies its carry rule and answer format:

Base 7 addition prompt.

Task: Perform addition in Base-7 (septenary system).   
Calculate {a\_b7} + {b\_b7} in Base-7.   
Rules: In Base-7, digits are 0-6. When a digit-sum reaches 7,   
carry 1 to the next position.

Think step by step and give your final answer   
in Base-7 inside \boxed{}.

Base 7 multiplication uses the analogous prompt with multiplication-specific carry instructions. Base 10 uses the corresponding decimal arithmetic prompts, asking for the sum or product of two decimal operands and a final boxed numerical answer.

Arithmetic sampling and extraction. For DeepSeek-family models, the arithmetic scripts use temperature 0.6 and $\mathrm { t o p } { - } p = 0 . 9 5$ . For Qwen3 models, they use temperature 0.6, top-p = 0.95, $ { \mathrm { t o p } } \cdot k = 2 0$ and min- $- p = 0 . 0 \nonumber$ . For GPT-OSS models, they use temperature 1.0 and ${ \mathrm { t o p } } { - } p = 1 . 0$ . All arithmetic runs use seed 42, maximum generation length 10,240 tokens, and $\mathrm { n } = 3 2$ completions per prompt. Answer extraction first uses the final \boxed{...} expression if present. Base 7 falls back to the last digit string with commas and spaces removed; Base 10 removes commas and takes the last signed integer.

Unsolved fraction. For each problem, the retry matrix stores a binary correctness value for each of the 32 attempts. The checkpoint statistic $\hat { E } ( R )$ averages failure indicators over the first R attempts:

$$
\hat { E } ( R ) = \frac { 1 } { | \mathcal { D } | } \sum _ { x \in \mathcal { D } } \mathbf { 1 } \{ \mathrm { a t t e m p t s } \ 1 , \dots , R \mathrm { a l l ~ f a i l ~ o n } \ x \} .
$$

The unsolved fraction after 32 attempts is $\hat { E } ( 3 2 )$ , and empirical Pass@R is $1 - { \hat { E } } ( R )$ . Each value of R uses the first R of the same 32 attempts. In this arithmetic probe, Pass@1 is the first sampled attempt among the 32 returned completions, not a greedy decoding run.

Case Example: Base 7 Addition. For a one-digit Base 7 addition problem, the script samples decimal operands in [1, 6] and converts them to Base 7. If the sampled operands are decimal 5 and 6, the prompt asks for $5 + 6$ in Base 7. The target is decimal 11, written as 14 in Base 7. A final boxed answer of 14 is correct; a decimal answer 11 is incorrect because the output must be in Base 7.

Case Example: Counting Unsolved Problems. Suppose a condition has 50 problems and, for a given problem, the 32-attempt correctness row is

$$
0 , 0 , 0 , 1 , 0 , \ldots
$$

This problem contributes to $\hat { E } ( 1 ) , \hat { E } ( 2 )$ , and $\hat { E } ( 3 )$ because the first R attempts all fail for those checkpoints. It does not contribute to $\hat { E } ( 4 )$ or later checkpoints because an attempt has succeeded by $R = 4$ . Problems left unsolved within this budget are not necessarily impossible to solve.

## D Arithmetic Results by Base and Digit Count

Figure 5 replots the first-attempt accuracies and mean token counts shown in Figures 8 and 15. Figure 6 gives the corresponding unsolved fractions for each model and operation. Digit count indexes input size, not measured reasoning steps; the numerical ranges differ across bases.

## E Detailed Results of Repeated Arithmetic Attempts

The arithmetic protocol in Appendix C.3 yields 320 combinations of model, base, operation, and digit count, each with 50 problems and 32 complete-answer attempts per problem. The following figures show the per-condition curves, single-attempt scores, and token measurements.

![](images/e00ce44797456d0acdfc1a166635f1be7b1784a419e79933a589d87f130c2607.jpg)

Figure 5: Accuracy and Generated Tokens by Digit Count. Top: first-attempt accuracy from Figure 8. Bottom: mean generated tokens per attempt from Figure 15. Both rows plot the reported heatmap cells under the protocol in Appendix C.3; colors identify models.  
![](images/7938f31b2f113003c9b3f1075e891a24286c8ce6fcc5beefc930b5672fd10078.jpg)  
Figure 6: Unsolved Fraction in Base 7 and Base 10. Each curve reports E<sup>ˆ</sup>(32) by digit count for one model and operation, comparing the two bases under the stated sampling protocol.

## E.1 Unsolved-Fraction Curves

Figure 7 shows ${ \hat { E } } ( R )$ for all 320 conditions. At R = 32, 235 conditions have no unsolved problems and 85 have at least one. All 80 Base 10 addition conditions reach zero unsolved problems. Base 7 multiplication has the largest mean unsolved fraction, 0.40, with 33 of 80 conditions above 0.5. The curves vary across models and digit counts, including GPT-OSS-120B’s nonmonotonic low-digit Base 7 multiplication results. Appendix E.4 defines the two-component fits shown as dashed curves.

## E.2 First-Attempt Accuracy

Figure 8 complements the unsolved-fraction heatmap by showing Pass@1. Low Pass@1 can coexist with high Pass@32: R1-1.5B on Base 7 addition at $n = 3$ has Pass@1= 0.08 and Pass@32= 0.98, with $\hat { E } ( 3 2 ) = 0 . 0 2 .$ . Across the 320 conditions, 12 have a gain from additional attempts, Pass@32 minus Pass@1, greater than 0.80. For R1-1.5B on Base 7 multiplication at $n \geq 7$ , Pass@1 remains zero and $\hat { E } ( 3 2 ) = 1$ . These cases distinguish first-attempt accuracy from finite-pool answer coverage, without identifying how an intermediate state would affect later steps.

## E.3 Unsolved Fraction at Different Attempt Budgets

Figure 9 reports ${ \hat { E } } ( R )$ at $R \in \{ 2 , 4 , 8 , 1 6 , 3 2 \}$ . Across 320 conditions, the mean absolute gap between $\hat { E } ( 1 6 )$ and $\hat { E } ( 3 2 )$ is 0.031, with a maximum of 0.32. These differences describe changes within the observed budget; they do not establish convergence beyond it. Because every checkpoint uses a prefix of the same pool, solved problems remain covered at later checkpoints.

## E.4 Two-Component Mixture Fits

We fit the empirical curves by nonlinear least squares using

$$
\begin{array} { r } { \hat { E } ( R ) \approx \pi _ { s } \varepsilon _ { \mathrm { g o o d } } ^ { R } + ( 1 - \pi _ { s } ) \varepsilon _ { \mathrm { b a d } } ^ { R } , } \end{array}\tag{103}
$$

Here $\pi _ { s }$ is the weight of the faster-decaying component and $\varepsilon _ { \mathrm { g o o d } }$ its per-attempt failure rate. The remaining weight $1 - \pi _ { s }$ belongs to a slowly decaying component with $\varepsilon _ { \mathrm { b a d } }$ near one. Figures 10 and 11 display all 128 conditions with $n \in \{ 3 , 5 , 7 , 9 \}$ . The weight $1 - \pi _ { s }$ of the slowly decaying component tracks E<sup>ˆ</sup>(32) in the reported comparisons with $\hat { E } ( 3 2 ) > 0 . 1$ , but can exceed it when few problems remain unsolved. The fit summary also reports $\varepsilon _ { \mathrm { g o o d } } < 0 . 1$ for R1-14B in conditions where R1-1.5B has $\varepsilon _ { \mathrm { g o o d } } > 0 . 5$ . The weight $1 - \pi _ { s }$ describes the fitted component over the observed window; if both fitted failure rates are below one, the fitted curve tends to zero.

## E.5 Gain from Additional Attempts by Digit Count

Figure 12 plots Pass@R minus Pass@1 against digit count at $R \in \{ 4 , 8 , 1 6 , 3 2 \}$ , separately for addition and multiplication. The gain is at most $1 - \mathrm { P a s s } \ @ 1$ ; its distance from this maximum equals ${ \hat { E } } ( R )$ . For R1-7B on Base 7 multiplication at $n = 4 ,$ , Pass@1= 0 and Pass $\textcircled { a } 3 2 = 0 . 2 6$ , leaving $\hat { E } ( 3 2 ) = 0 . 7 4$ . The vertical axis spans the full gain range from zero to one.

## E.6 Eight-Digit Multiplication Trajectories

Figure 13 selects eight-digit multiplication in each base and plots the full $\hat { E } ( R )$ trajectory for every model. Several trajectories flatten around 8–16 attempts, while others continue to decrease through the observed window. Figure 9 reports the corresponding checkpoints across all conditions.

## E.7 Pass@1 and Pass@32 by Digit Count

Figure 14 pairs Pass@1 and Pass@32 at each digit count for all 320 conditions. Panels identify models, colors distinguish bases, and the two marker types identify the attempt budgets. Within-pair gaps measure retry gain; gaps above Pass@32 measure the unsolved fraction.

## E.8 Generated Token Counts

Figure 15 reports average generated tokens per attempt. The displayed range is 115–10,240 tokens, with the upper value set by the generation cap. Base 7 multiplication averages 7,391 tokens, compared with 2,702 for Base 10 addition, a ratio of about 2.7 . This contrast varies both the base and the operation. GPT-OSS-120B uses 115 tokens for one-digit Base 10 addition and 116–153 for one- and two-digit Base 10 multiplication; these short decimal outputs do not explain the different coverage pattern observed for Base 7 arithmetic under the same attempt budget.

![](images/c8d06109ef9d7d3f698c25bfd97660a23e5ef4477c6934f3ba7332ee63e30d48.jpg)  
Figure 7: Empirical Unsolved-Fraction Curves. Rows are models; columns are Base 10 addition, Base 10 multiplication, Base 7 addition, and Base 7 multiplication. Colors encode $n = 1 , \ldots , 1 0$ Solid curves show E<sup>ˆ</sup>(R) and dashed gray curves show fitted two-component models. The panels retain both zero and nonzero unsolved fractions within 32 attempts, including GPT-OSS-120B’s nonmonotonic Base 7 multiplication results.

![](images/f80a315252f80b4f627b5b9b2a9c94d507bc05b60d5a56a5483d5c85b79ae900.jpg)  
Figure 8: First-Attempt Accuracy. For Base 7 multiplication, R1-14B (n = 6) has Pass@1=0.14 and $\hat { E } ( 3 2 ) = 0 . 1 6$ (84% solved within 32 attempts); R1-1.5B (n = 7) has Pass@1=0 and E<sup>ˆ</sup>(32) = 1.

![](images/fc124dc57d1b79925c53390369a6d353f3c7074457d1cc68809c8ff5a7a8136a.jpg)  
Figure 9: Unsolved Fraction at Different Attempt Budgets. Rows correspond to $R \in \{ 2 , 4 , 8 , 1 6 , 3 2 \}$ and columns to the four base–operation combinations. Curves show ${ \hat { E } } ( R )$ against digit count; colors identify models. The reported mean absolute gap between $R = 1 6$ and $R = 3 2$ is 0.031 across 320 conditions, with a maximum of 0.32; every checkpoint uses the same nested attempt pool.

![](images/03596e7bc332acad68257caf06f1e0162e4dd6c34d41c22919c09970c6d63630.jpg)  
Figure 10: Two-Component Fits for Base 7. Solid curves show empirical ${ \hat { E } } ( R )$ ; dashed curves show nonlinear least-squares fits of $\pi _ { s } \varepsilon _ { \mathrm { g o o d } } ^ { R } + ( 1 - \pi _ { s } ) \varepsilon _ { \mathrm { b a d } } ^ { R }$ at n  3, 5, 7, 9 . The slowly decaying component’s weight $1 - \pi _ { s }$ is compared with the unsolved fraction in Appendix E.4.

![](images/0936629a3a91b3d60afa48f306576d7042ddb92bd3cfb8aad63d6e20562a3133.jpg)  
Figure 11: Two-Component Fits for Base 10. The layout, checkpoints, and curve styles follow Figure 10, with Base 10 addition and multiplication shown in the two panels.

![](images/519efaf102934e8e481208cffc7089d65cab325ea23f7e780d380026c871826f.jpg)  
Figure 12: Gain from Additional Attempts by Digit Count. Top: addition; bottom: multiplication. Panels identify models, the horizontal axis is digit count, colors encode $R \in \{ 4 , 8 , 1 6 , 3 2 \}$ , and line styles distinguish bases. The plotted quantity is Pass@R minus Pass@1; its gap to $1 - \mathrm { P a s s } \ @ 1$ equals ${ \hat { E } } ( R )$

![](images/2f605b962e6baa49e83a8d48b2db0ae0a4941b23f81fd275ae2665c4161730b8.jpg)  
Figure 13: Unsolved Fraction for Eight-Digit Multiplication. Each model panel shows ${ \hat { E } } ( R )$ for Base 7 (solid green) and Base 10 (dashed blue) at $n = 8$ . Markers identify $R \in \{ 2 , 4 , 8 , 1 6 , 3 2 \}$ . The curves retain the full observed range from 1 to 32 attempts for both numeral bases.

![](images/23953c26a274e0995ea6d3920a0bfe98a5ee889a2666ec39327f3dc2085c130c.jpg)  
Figure 14: Pass@1 and Pass@32 by Digit Count. Top: addition; bottom: multiplication. Panels identify models, colors distinguish Base 7 and Base 10, and markers distinguish Pass@1 from Pass@32. The vertical separation at each digit count is the gain from additional attempts.

![](images/dd40de44ca7348ebe91bbefcca5db9f43f05f0b20150151a1b5925cc41e17788.jpg)

![](images/ec4e1a89f08a8cb39ef5e017fa464f426f7835db14387427171c14429720c16a.jpg)

![](images/eb246f1bc8ae2b70f0f5d364093ce68380aed1743eed291dde6d2ceb77fddb62.jpg)

![](images/f66ca265d72cd87dd124f402606207de73d43b02e9dc29ac7c770ca264565f9d.jpg)  
Figure 15: Average Generated Tokens per Attempt. Cells show mean output length for each combination of model, base, operation, and digit count under a 10,240-token generation cap. The displayed range is 115–10,240 tokens; averages are 7,391 for Base 7 multiplication and 2,702 for Base 10 addition.

## F Agent Score Prediction: Derivation and Details

The forecasting comparison in Section 4.5 uses a curve motivated by a special case of the repeated-attempt model: every instance is solvable and checks accept exactly the correct candidates.

We evaluate agent score prediction using each task–model pair’s scores at 2 and 4 hours, together with priors fitted to other tasks in the same category, excluding the target task. At 12 hours, the retry curve and the sigmoid curve in log time obtain median relative errors of 10.7% and 10.9%, respectively, over 248 pairs with positive observed scores.

## F.1 From Repeated Attempts to a Score Curve

We extend the model for generating correct candidates in Appendix A.2 to continuous time. Equation (14) approximates the probability of no correct candidate after R attempts:

$$
q _ { t } ^ { \mathrm { c l e a n } } ( R ) \approx q _ { t } ^ { \mathrm { f l o o r } } + ( 1 - q _ { t } ^ { \mathrm { f l o o r } } ) \cdot \varepsilon _ { \mathrm { l o c } , t } ^ { R } .\tag{14, restated}
$$

In the special case $q _ { t } ^ { \mathrm { f l o o r } } = 0$ and $\varepsilon _ { \mathrm { l o c } , t } = \varepsilon _ { \mathrm { \ell } }$ , every instance has the same per-attempt failure probability. A perfect check gives $\alpha = 0$ and $\rho = \varepsilon$ , with success probability

$$
1 - q _ { t } ^ { \mathrm { c l e a n } } ( R ) \ = \ 1 - \varepsilon ^ { R } .\tag{104}
$$

The general loop also includes incorrect acceptance in its failure bound (Eq. (37)).

Mapping to agent tasks. For the prediction model, t is elapsed time in hours. We replace R with $t / T _ { \cdot }$ , where the positive time scale T is fitted together with the remaining parameters. The fitted score $S ( t ) \in [ 0 , S _ { \operatorname* { m a x } } ]$ is a continuous-time analogue of success probability, scaled by the task-specific ceiling $S _ { \mathrm { m a x } }$ to represent cumulative task credit.

Scaling the resulting success curve by the task-specific ceiling $S _ { \mathrm { m a x } }$ gives:

$$
S ( t ) = S _ { \operatorname* { m a x } } \cdot \big ( 1 - \varepsilon ^ { t / T } \big ) .\tag{105}
$$

Equation (105), the retry curve, substitutes $R = t / T$ and scales the result by the task ceiling.

Comparison curve. We also evaluate a sigmoid curve in log time:

$$
S ( t ) = \frac { S _ { \mathrm { m a x } } } { 1 + ( t _ { \mathrm { m i d } } / t ) ^ { \beta } } ,\tag{106}
$$

Here $t _ { \mathrm { m i d } } > 0$ is the time when $S = S _ { \mathrm { m a x } } / 2$ , and $\beta > 0$ controls curve steepness; this $\beta$ is unrelated to the local failure probability $\beta _ { t } ( R ; s )$ . The expression is a sigmoid function of log t, labeled “Sigmoid (log time)” throughout the figures and tables for this forecasting comparison.

Both models are monotone and bounded and approach their fitted ceilings as t increases.

## F.2 Experimental Setup

Data. We use publicly reported scores from the EdgeBench long-horizon agent benchmark (Zhu et al., 2026), which evaluates 5 frontier models across 51 tasks spanning 6 categories (Formal Mathematics, Games, Systems Engineering, Optimization, Scientific Computing, and Knowledge) at time budgets $t \in \{ 2 , 4 , 6 , 8 , 1 0 , 1 2 \}$ hours. All six scores are available for 252 task–model pairs.

Prediction protocol. For each target (task, model) pair, we use $S ( \mathrm { 2 h } )$ and $S ( \mathrm { 4 h } )$ as its observed scores, fit model parameters using the priors below, and predict $S ( t )$ for $t \in \{ 6 , 8 , 1 0 , 1 2 \} \mathrm { l }$ h. Both curves use category-level priors, meaning parameter estimates obtained from other tasks before fitting the target pair. The retry curve uses $\left( \varepsilon _ { \mathrm { c a t } } , c _ { \mathrm { c a t } } \right)$ , the median fitted failure parameter and score-ceiling ratio; the sigmoid curve in log time uses $\beta _ { \mathrm { c a t } }$ , the median slope. For each target task, priors are medians of fits to all six observations from other tasks in the same category. Every model trajectory belonging to the target task is excluded when computing these priors.

Parameter estimation. For the retry curve (Eq. 105), we jointly optimize $( \varepsilon , T , S _ { \mathrm { m a x } } )$ using maximum a posteriori (MAP) estimation: squared error on the two observed points plus a penalty for deviation from $\left( \varepsilon _ { \mathrm { c a t } } , c _ { \mathrm { c a t } } \right)$ . Because $\varepsilon ^ { t / T }$ depends on ε and $T$ only through $\log ( \varepsilon ) / T$ , we do not interpret the two parameters separately. For the sigmoid curve in log time (Eq. 106), we optimize $( S _ { \mathrm { m a x } } , t _ { \mathrm { m i d } } , \beta )$ with regularization toward $\beta _ { \mathrm { c a t } }$ . For both curves, fitting uses regularization coefficient 2.0 and Nelder–Mead optimization with at most 2,000 iterations and $1 0 ^ { - 4 }$ parameter and objective tolerances. The retry curve uses $0 . 0 1 <$ $\varepsilon < 0 . 9 9 9 , 0 . 0 5 < T \leq 1 0$ , and $0 < S _ { \mathrm { m a x } } \le 1 0 5 $ ; the sigmoid curve in log time uses $0 . 0 1 < t _ { \mathrm { m i d } } \leq 5 0 0$ $0 . 0 5 < \beta \leq 6$ , and $0 < S _ { \mathrm { m a x } } \le 1 0 5$ . MAP optimization produces numerical predictions for all 252 pairs, including the 53 cases where an analytic closed-form solution for the sigmoid curve in log time does not exist (i.e., when the observed growth ratio $S ( 4 \mathrm { { h } } ) / S ( 2 \mathrm { { h } } )$ is inconsistent with the prior slope $\beta _ { \mathrm { c a t } } )$ .

Fitting inputs and regularization. For fitting only, let $u = \mathrm { m a x } \{ S ( 2 ) , 0 \}$ and let $v = S ( 4 )$ when $S ( 4 ) > u , \mathrm { o r } v = 1 . 0 0 1 u + 0 . 0 1$ otherwise. Both procedures use these same adjusted early observations; evaluation uses the original later scores. The ceiling ratio for each source-task fit is $S _ { \mathrm { m a x } } / \operatorname* { m a x } \{ S ( 4 ) , 0 . 1 \}$ using that source task’s observed four-hour score. The prior $c _ { \mathrm { c a t } }$ is the median of these ratios after excluding the target task. Let $\sigma _ { \varepsilon } , \sigma _ { c } , \sigma _ { \beta }$ be the population standard deviations of the corresponding source-fit parameters, with

$$
s _ { \varepsilon } = \operatorname* { m a x } \{ \sigma _ { \varepsilon } , 0 . 0 5 \} , \quad \quad s _ { c } = \operatorname* { m a x } \{ \sigma _ { c } , 0 . 1 \} , \quad \quad s _ { \beta } = \operatorname* { m a x } \{ \sigma _ { \beta } , 0 . 1 5 \} .
$$

Writing $f _ { \mathrm { r e t r y } }$ and $f _ { \mathrm { s i g } }$ for the two score curves, the implemented objectives are

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { r e t r y } } = [ f _ { \mathrm { r e t r y } } ( 2 ) - u ] ^ { 2 } + [ f _ { \mathrm { r e t r y } } ( 4 ) - v ] ^ { 2 } } \\ & { \qquad + 2 \left[ \left( \frac { \varepsilon - \varepsilon _ { \mathrm { c a t } } } { s _ { \varepsilon } } \right) ^ { 2 } + \left( \frac { S _ { \mathrm { m a x } } / \operatorname* { m a x } \{ v , 0 . 1 \} - c _ { \mathrm { c a t } } } { s _ { c } } \right) ^ { 2 } \right] , } \\ & { \qquad \mathcal { L } _ { \mathrm { s i g } } = [ f _ { \mathrm { s i g } } ( 2 ) - u ] ^ { 2 } + [ f _ { \mathrm { s i g } } ( 4 ) - v ] ^ { 2 } + 2 \left( \frac { \beta - \beta _ { \mathrm { c a t } } } { s _ { \beta } } \right) ^ { 2 } . } \end{array}
$$

The sigmoid penalizes slope deviations. Both optimizers start with ceiling min $\{ v c _ { \mathrm { c a t } } , 1 0 0 \}$ . A coarse parameter grid supplies the starting point; the optimizer’s output replaces it only when the objective improves. Prior construction uses unregularized six-observation fits with up to 3,000 iterations, $S _ { \mathrm { m a x } } \leq$ 110, and, for the retry curve, $T \leq 8$ . The bounds above apply to the target-pair prediction fits.

## F.3 Results

Table 6 reports median relative errors and the fractions of predictions below the 10% and 20% error thresholds at all four prediction times. Median errors are similar, with neither curve having a smaller error at every prediction time. The retry curve has at least as many predictions below 10% relative error, and more below 20%, at every prediction time.

Table 6: Agent Score Prediction from 2- and 4-Hour Observations. Both curves use priors from other tasks as specified in Appendix F.2. Median relative errors and fractions below 10% and 20% use 248 pairs with positive observed scores at each prediction time. All 252 pairs receive predictions; four pairs with zero observed scores are excluded from relative errors at each prediction horizon.
<table><tr><td rowspan="2">Time</td><td colspan="3">Retry Curve (Eq. 105)</td><td colspan="3">Sigmoid (log time) (Eq. 106)</td></tr><tr><td>Median</td><td> ${ < } 1 0 \%$ </td><td>&lt;20%</td><td>Median</td><td>&lt;10%</td><td>&lt;20%</td></tr><tr><td>6 hours</td><td>3.8%</td><td>71%</td><td>87%</td><td>3.9%</td><td>67%</td><td>81%</td></tr><tr><td>8 hours</td><td>6.5%</td><td>61%</td><td>78%</td><td>5.8%</td><td>58%</td><td>73%</td></tr><tr><td>10 hours</td><td>8.9%</td><td>53%</td><td>69%</td><td>8.8%</td><td>52%</td><td>66%</td></tr><tr><td>12 hours</td><td>10.7%</td><td>48%</td><td>65%</td><td>10.9%</td><td>48%</td><td>62%</td></tr></table>

Figures 4 and 16 show the 12-hour predictions and increasing median error at later times. Both curves have median error below 11% at all four prediction times.

![](images/c5d23e27b018d2a059966991fc58ee25afb768f95a27d7c3f07e4d37de2d634b.jpg)  
Figure 16: Median Relative Error by Prediction Time. Both curves use the protocol in Appendix F.2. Relative errors use 248 pairs with positive observed scores at each prediction time; the dashed line marks 10%.

## F.4 Interpretation

The two curves have similar median errors when fitted to early observations with priors from other tasks in the same category. These predictions use early scores and information from other tasks; they do not test the checks assumed by the reuse theorem. Although the source tasks are shared, the penalties constrain different parameters: failure and ceiling ratio for the retry curve, and slope for the sigmoid. Both procedures also use the ceiling-ratio prior for initialization. The differences compare the fitted procedures; they do not isolate curve shape under identical parameter information. Predictive fit does not identify agents’ internal checking or state-reuse behavior. The experiment predicts later scores; it does not evaluate early stopping or compute allocation.