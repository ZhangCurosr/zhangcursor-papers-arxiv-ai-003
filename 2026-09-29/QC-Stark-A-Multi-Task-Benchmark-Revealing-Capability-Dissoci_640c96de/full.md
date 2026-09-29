# QC-Stark: A Multi-Task Benchmark Revealing Capability Dissociations in LLMs Evaluated on Quantum Computing Tasks

Pranav Gupta<sup>1[0000−0002−1412−0885]</sup>

Cisco, 2901 3rd Ave, Suite 600, Seattle, WA 98121, USA

pranavpg@cisco.com

Abstract. We introduce QC-Stark, a benchmark for evaluating large language models (LLMs) on 11 quantum computing (QC) tasks, spanning circuit construction, debugging, compilation, error correction, and simulation. Across 2,750 evaluations (10 models × 11 tasks × 5 dificulty levels × 5 seeds), we find that overall rankings mask substantial pertask variation. The Spearman correlation ρ between overall and per-task rankings is statistically insignificant for 4 out of the 11 tasks included in this benchmark. A 2-parameter Item Response Theory (IRT) model validates measurement quality, and prompt sensitivity analysis confirms ranking robustness across prompt conditions. All tasks are auto-verifiable via execution, thus not requiring any manual evaluation. We make the code and data publicly available on Huggingface.

Keywords: Large Language Models (LLMs) · Quantum Computing · Item Response Theory.

## 1 Introduction

While large language models (LLMs) are being increasingly applied to quantum computing (QC) tasks [1], existing evaluations still focus on specific aspects, such as algorithm implementation [2], challenge-style programming [3], or conceptual question-answering [4]. There is a lack of benchmarks that span the complete spectrum of a quantum computing practitioner’s day-to-day operations, ranging from circuit construction through compilation, debugging, error correction, and simulation.

This gap is important because LLM capabilities cannot be expected to generalize across tasks, given the diversity in their architecture, algorithms and training/evaluation data. A model that excels at oracle synthesis might score close to zero on hardware routing. In some cases, a model ranked first overall may score 0% on a challenging task. An ideal benchmark should ideally also include features such as run-time procedural generation, in order to avoid dataset contamination or leakage.

QC-Stark addresses these concerns by providing (1) 11 tasks covering the full QC workflow, (2) 5 dificulty levels per task, (3) seed-based procedural generation to resist against data contamination, and (4) automatic verification via code execution.

## 2 Related Work

QC-Stark complements existing quantum computing benchmarks. Quantum-Audit [4] evaluates conceptual knowledge via multiple-choice questions (best performance: 84%), Qiskit QuantumKatas [2] tests 350 educational exercises (best performance: 83%), QCoder [3] evaluates on contest problems with simulator feedback (best performance: 78%), whereas QCircuitBench [5] provides large-scale algorithm design data. QC-Stark spans the full operational spectrum with systematic dificulty scaling in terms of 5 separate levels.

Outside quantum computing, scientific coding benchmarks like SciCode [6] (best performance: 4.6%) and CMT-Benchmark [7] (best performance: 30%) find frontier models far from research-level capability, thus implying the need for continuously developing new benchmarks which are fundamentally distinct from existing ones.

## 3 Benchmark Design

Tasks. QC-Stark consists of 11 tasks organized by stages in a QC practitioner’s workflow: Circuit Construction (State Preparation, Trotter Decomposition, Oracle Synthesis), Code Understanding (Debugging, Noise Discrimination, Reverse Engineering), Verification (Equivalence Checking), Compilation (Hardware Routing), Simulation (Noise Fidelity Estimation, Variational Quantum Eigensolver (VQE)), and Error Correction (Syndrome Decoding). Appendix D provides full descriptions for each task.

Dificulty scaling. Each task has 5 levels: L1 (Textbook), L2 (Homework), L3 (Exam), L4 (Research), L5 (Open), controlled via problem size, constraint complexity, and domain-specific parameters. Appendix B provides example prompt templates.

Evaluation. Each instance is deterministically generated from a (task, level, seed) tuple. Models receive a system prompt with Qiskit 2.x [10] API guidance and are asked to output a solve() function, which is run by a verifier to check for correctness. The output is matched against ground truth, in terms of aspects such as state fidelity, functional equivalence and correct identification. By principle, we do not require any human judgment.

Models. We evaluated 10 models: o4-mini, Claude Sonnet 5, GPT-5.4, Gemma-4 31B, Claude Opus 4.1, GPT-4.1-mini, Gemini Flash Lite (Gemini FL), Gemini 3.5 Flash, Mistral Large 3, and LLaMA-3.3 70B. These models were chosen based on our availability constraints and a goal of encompassing various model parameter sizes in our study. Note that we used Claude Code (Opus 4.6 and higher) for designing the benchmark, hence there could be some bias towards Claude models. In the future, this can be remediated by the use of a council of LLMs collectively designing the benchmark or by excluding those model families from the evaluation. We chose to keep Anthropic/Claude models for the sake of completion.

Table 1. Mean accuracy per model and task (all models at maximum supported token budgets). Bold: best per task. Gray: worst. Last column: overall mean score ± standard deviation across seeds. Task codes: T1=State Preparation, T2=Trotter Decomposition, T3=Oracle Synthesis, T4=Debugging, T5=Noise Discrimination, T6=Reverse Engineering, T7=Equivalence, T8=Routing, T9=Noise Fidelity, T10=VQE, T11=Quantum Error Correction
<table><tr><td>Model</td><td>T1</td><td>T2</td><td>T3</td><td>T4</td><td>T5</td><td>T6</td><td>T7</td><td>T8</td><td>T9</td><td>T10</td><td>T11</td><td>Overall</td></tr><tr><td>Sonnet 5</td><td>.68</td><td>.24</td><td>1.00</td><td>.60</td><td>.64</td><td>.64</td><td>1.00</td><td>.20</td><td>1.00</td><td>.60</td><td>.68</td><td> $\mathbf { . 6 6 2 } \pm \mathbf { . 0 4 7 }$ </td></tr><tr><td>Gemini 3.5F</td><td>.92</td><td>.00</td><td>.96</td><td>.48</td><td>.60</td><td>.76</td><td>.84</td><td>.20</td><td>1.00</td><td>.80</td><td>.60</td><td> $. 6 5 1 { \pm } . 0 5 9$ </td></tr><tr><td>o4-mini</td><td>.52</td><td>.64</td><td>.84</td><td>.00</td><td>.56</td><td>.68</td><td>.88</td><td>.32</td><td>1.00</td><td>.40</td><td>.68</td><td> $. 5 9 3 { \pm } . 0 6 0$ </td></tr><tr><td>GPT-5.4</td><td>.40</td><td>.36</td><td>.44</td><td>.20</td><td>.60</td><td>.68</td><td>.84</td><td>.04</td><td>.80</td><td>.88</td><td>.64</td><td>.535±.083</td></tr><tr><td>Gemma-4</td><td>.40</td><td>.52</td><td>.52</td><td>.00</td><td>.52</td><td>.48</td><td>.76</td><td>.28</td><td>.72</td><td>.56</td><td>.64</td><td> $. 4 9 1 { \pm } . 0 8 8$ </td></tr><tr><td>GPT-4.1 mini</td><td>.40</td><td>.20</td><td>.24</td><td>.00</td><td>.16</td><td>.48</td><td>.80</td><td>.20</td><td>.84</td><td>.40</td><td>.60</td><td> $. 3 9 3 { \pm } . 0 3 4$ </td></tr><tr><td>Opus 4.1</td><td>.40</td><td>.24</td><td>.48</td><td>.00</td><td>.00</td><td>.32</td><td>.88</td><td>.16</td><td>.76</td><td>.56</td><td>.40</td><td> $. 3 8 2 \pm . 0 4 1$ </td></tr><tr><td>Gemini FL</td><td>.44</td><td>.04</td><td>.36</td><td>.00</td><td>.32</td><td>.40</td><td>1.00</td><td>.44</td><td>.48</td><td>.48</td><td>.08</td><td> $. 3 6 7 { \scriptstyle \pm . 0 6 7 }$ </td></tr><tr><td>Mistral L3</td><td>.40</td><td>.00</td><td>.16</td><td>.00</td><td>.00</td><td>.56</td><td>.80</td><td>.12</td><td>.04</td><td>.40</td><td>.40</td><td> $. 2 6 2 { \pm } . 0 1 5$ </td></tr><tr><td>LLaMA-70B</td><td>.40</td><td>.00</td><td>.00</td><td>.04</td><td>.00</td><td>.04</td><td>.00</td><td>.12</td><td>.00</td><td>.36</td><td>.36</td><td> $. 1 2 0 { \pm } . 0 0 9$ </td></tr><tr><td>Task mean</td><td>.50</td><td>.22</td><td>.50</td><td>.13</td><td>.34</td><td>.50</td><td>.78</td><td>.21</td><td>.66</td><td>.54</td><td>.51</td><td>.445</td></tr></table>

## 4 Results

## 4.1 Overall Performance and Rank Inversions

Table 1 presents the scores from 2,750 evaluations (10 models × 11 tasks × 5 seeds × 5 dificulty levels). The best model (Claude Sonnet 5) achieves 0.662 overall, but its performance drops 59% as we increase the dificulty level from L1 to L5. Task dificulty varies 5.9× (Equivalence: 0.78 vs. Debugging: 0.13).

The overall ranking is not predictive of per-task performance on 4 out of 11 tasks, where Spearman ρ between the overall and individual task rankings is statistically non-significant. These 4 tasks were Routing $( \rho = + 0 . 2 7 , p = 0 . 4 5 )$ ， Trotterization $( \rho = + 0 . 4 9 , p = 0 . 1 5 )$ , Debugging $( \rho = + 0 . 5 4 , p = 0 . 1 1 )$ , and Equivalence $( \rho = + 0 . 4 5 , p = 0 . 1 9 )$ . These inversions demonstrate that a single aggregate score is insuficient in guiding model selection for quantum computing tasks, and there is still progress to made for creating LLMs that outperform their predecessors on all kinds of benchmarks.

## 4.2 The “Debugging” Conundrum

Despite all models being run at their maximum supported token budgets, 6 of 10 models score 0.00 on debugging. Claude Sonnet 5 leads at 60% (15/25), followed by Gemini 3.5 Flash at 48% (12/25 correct), GPT-5.4 at 20% (5/25 correct), and LLaMA-70B at 4% (1/25 correct). Dedicated reasoning models (o4-mini) score 0% even at token budgets of 65,536 tokens.

## 4.3 Routing: A Universal Bottleneck

Hardware routing (T8) has a mean accuracy of 0.21 across all models, with Gemini Flash Lite leading at a mean accuracy of 0.44 (driven by strong L1/L2 performance but weaker performance in more dificult problems in L3 and beyond). The Spearman correlation between overall and routing rankings is nonsignificant $( \rho = + 0 . 2 7 , p = 0 . 4 5 )$ : the 8th-ranked model leads, while 4th-ranked GPT-5.4 scores merely 0.04 (1 is the maximum score in all these evaluations). Many evaluations produce circuits not functionally equivalent to input. This shows that models find it dificult to maintain circuit semantics while inserting SWAP gates for hardware connectivity.

## 4.4 Measurement Validation

IRT analysis. Our Item Response Theory (IRT) analysis follows ATLAS [8]. Equally weighted averages over all tasks to determine the final rankings can thus be misleading. A 2-parameter logistic (2PL) IRT model [12] fitted via Joint Maximum Likelihood Estimation (JMLE) yields marginal reliability = 0.985, with ability estimates spanning 3.6 standard units (Table 2). Of 275 items (11 tasks × 5 seeds × 5 dificulty levels), 216 (78.5%) achieve discrimination $a \mathrm { ~ > ~ }$ 1.0, which provides evidence that our benchmark problems are appropriate in terms of dificulty. As expected, dificulty progresses monotonically from L1 $( b =$ −0.88) to L5 $\left( b = + 0 . 8 9 \right)$ (see Appendix C for a per-level breakdown). The IRT ability ranking is nearly identical to the raw accuracy ordering $( \rho ~ = ~ 0 . 9 7 6$ $p < 0 . 0 0 1 $ . Claude Sonnet 5 leads in terms of both IRT ability (θ = 1.418) and mean accuracy. The equation for the IRT model is given below.

$$
\mathrm { P r } ( Y _ { m j s } = 1 ) ~ = ~ p _ { m j } ~ = ~ \frac { 1 } { 1 + e ^ { - a _ { j } ( \theta _ { m } - b _ { j } ) } }\tag{1}
$$

$$
\log a _ { j } \sim \mathcal { N } ( 0 , \sigma ^ { 2 } ) , \qquad \sigma = 0 . 5 , \qquad \lambda = 1 / \sigma ^ { 2 } = 4\tag{2}
$$

$$
\ell ( \theta , a , b ) = \sum _ { m = 1 } ^ { 1 0 } \sum _ { j = 1 } ^ { 5 5 } \Big [ k _ { m j } \log p _ { m j } + ( n _ { m j } - k _ { m j } ) \log ( 1 - p _ { m j } ) \Big ]\tag{3}
$$

$$
( \hat { \theta } , \hat { a } , \hat { b } ) \ = \ \arg \operatorname* { m a x } \ \Big \{ \ell ( \theta , a , b ) \ - \ \frac { \lambda } { 2 } \sum _ { j = 1 } ^ { 5 5 } ( \log a _ { j } ) ^ { 2 } \Big \}\tag{4}
$$

Here $m = 1 , \ldots , 1 0$ indexes models, $j = 1 , \ldots , 5 5$ indexes items (one per task– level cell), and $s = 1 , \ldots , 5$ indexes seeds. $Y _ { m j s } \in \{ 0 , 1 \}$ is whether model m solved seed s of item j; $p _ { m j }$ is the probability of a correct response, shared by all five seeds of a cell. $\theta _ { m }$ is the ability of model m, standardised so that θ has mean 0 and unit standard deviation; $b _ { j }$ is the dificulty of item $j$ on that same scale, and $a _ { j } > 0$ its discrimination, the slope of the response curve in θ. $\begin{array} { r } { k _ { m j } = \sum _ { s } Y _ { m j s } } \end{array}$ is the number of seeds solved and $n _ { m j } = 5$ the number attempted. ℓ is the binomial log-likelihood, σ the prior standard deviation of log $a _ { j }$ , and $\lambda = 1 / \sigma ^ { 2 }$ the equivalent penalty weight; hats denote estimates.

Prompt sensitivity. When we compared the canonical structured prompt against a second, minimal prompt across all 11 tasks and all 10 models (2,745 paired evaluations per condition), we found that model rankings are largely preserved $( \rho = 0 . 9 1 5$ , Kendall $\tau = 0 . 7 7 8 )$ ). The top-4 ranked models stay in the top 4 across conditions, whereas the bottom-4 do not (GPT-4.1-mini falls from 6th to 8th under the minimal prompt, displacing Opus 4.1). Further details can be found in Appendix A.

Error taxonomy and limitations. All 2,750 evaluations were successful with 0 API errors after retries. Each model was run at its maximum supported token budget: 65,536 for o4-mini, GPT-5.4, Sonnet 5, Gemini 3.5 Flash, and Mistral L3, 32,768 for GPT-4.1 mini, 32,000 for Opus 4.1, 8,192 for LLaMA-70B, and 4,096 for Gemma-4 and Gemini FL. 11 records remained truncated at hard API caps that cannot be increased. 46 evaluations timed out during verification (we used a limit of 300 seconds), which were concentrated in VQE and Trotterization tasks that produce computationally expensive circuits. The benchmark requires Qiskit 2.x Python output, running the risk of conflating quantum domain knowledge with framework-specific API proficiency. A framework-agnostic format such as OpenQASM could potentially isolate reasoning but it changes task dificulty $( \mathrm { e . g . }$ , state decomposition into elementary gates is substantially harder without high-level primitives). While the structured prompt’s Qiskit 2.x guidance functions as lightweight context injection, retrieval augmented generation (RAG) with oficial documentation could further isolate reasoning ability from API memorization.

## 5 Conclusion

QC-Stark demonstrates that QC competence is not unidimensional: dedicated reasoning models o4-mini and o3-mini score 0% on debugging even at 64K-token budgets (o3 scores 13%). Surprisingly, Gemini Flash Lite (8th overall) leads routing at 44%. The best accuracies per task display a huge range, from 13.2% in Debugging to 78.0% in Equivalence. Performance drops 59% as we increase the dificulty level from L1 to L5.

Prompt sensitivity analysis confirms ranking robustness $( \rho = 0 . 9 1 5 )$ . IRT analysis validates the benchmark as a reliable instrument $( r ~ = ~ 0 . 9 8 5 )$ with monotonic dificulty progression, establishing that the observed dissociations reflect genuine structure in capability rather than just noise. Future work includes RLVR-based fine-tuning [14], framework-agnostic representations, and retrievalaugmented generation (RAG).

## Acknowledgements

Claude [15] provided assistance for portions of benchmark infrastructure . All scientific claims and experimental design are solely the author’s responsibility. The data for this benchmark is available at https://huggingface.co/datasets/ pranavgupta/qc-stark.

## References

1. Cao, S., Zhang, Z., Alghadeer, M., Fasciati, S.D., Piscitelli, M., Bakr, M., Leek, P., Aspuru-Guzik, A.: Automating quantum computing laboratory experiments with an agent-based AI framework. Patterns 6(10), 101372 (2025)

2. Cruz-Benito, J., Faro, I.: Qiskit QuantumKatas: Adapting Microsoft’s quantum computing exercises for LLM evaluation. arXiv:2605.27210 (2026)

3. Mikuriya, T., et al.: QCoder Benchmark: Bridging language generation and quantum hardware through simulator-based feedback. INLG (2025)

4. Afane, M., et al.: Quantum-Audit: Evaluating the reasoning limits of LLMs on quantum computing. arXiv:2602.10092 (2026)

5. Yang, R., et al.: QCircuitBench: A large-scale dataset for benchmarking quantum algorithm design. arXiv:2410.07961 (2024)

6. Tian, M., et al.: SciCode: A research coding benchmark curated by scientists. arXiv:2407.13168 (2024)

7. Pan, H., et al.: CMT-Benchmark: A benchmark for condensed matter theory built by expert researchers. In: ICLR 2026. arXiv:2510.05228 (2025)

8. Liang, P., et al.: Holistic evaluation of language models. TMLR (2023). arXiv:2211.09110

9. Vishwakarma, S., et al.: Qiskit HumanEval: An evaluation benchmark for quantum code generative models. arXiv:2406.14712 (2024)

10. Treinish, M., Lishman, J., Bello, L., Gambetta, J., et al.: Qiskit/qiskit: Qiskit 2.4.1. Zenodo (2026). https://doi.org/10.5281/zenodo.19742751

11. Huang, D., Wang, Z.: Task complexity matters: An empirical study of reasoning in LLMs for sentiment analysis. In: Wong, R.C.W., et al. (eds.) Advances in Knowledge Discovery and Data Mining (PAKDD 2026). Lecture Notes in Computer Science, vol. 16600. Springer, Singapore (2026). https://doi.org/10.1007/ 978-981-92-1468-6\_18

12. Baker, F.B., Kim, S.-H.: Item Response Theory: Parameter Estimation Techniques. Marcel Dekker, New York (2004)

13. Schaefer, R., Miranda, B., Koyejo, S.: Are emergent abilities of large language models a mirage? In: NeurIPS 2023. arXiv:2304.15004 (2023)

14. Lambert, N., Morrison, J., Pyatkin, V., et al.: Tulu 3: Pushing frontiers in open language model post-training. arXiv:2411.15124 (2024)

15. Anthropic: Claude. https://www.anthropic.com/claude (2025)

## A Prompt Sensitivity Details

Spearman ρ = 0.915 (Kendall τ = 0.778) between model rankings under the two conditions; overall accuracy rises 0.403 → 0.446 with the structured prompt. The top-4 grouping did not change across prompt types, but the bottom-4 ranks changed.

## B Prompt Templates

## B.1 System Prompts

## Structured prompt:

You are a quantum computing expert writing Python code.

Environment: Python 3.13, Qiskit 2.x, numpy, scipy.

\- QuantumCircuit.qasm() removed -> use qiskit.qasm2.dumps(circuit) / loads(qasm\_str)

\- execute() removed -> use Statevector or StatevectorSimulator

Output requirements:

\- Respond with ONLY executable Python code. No markdown fences, no explanations.

\- Define a function called solve() that returns the answer.

Note: The canonical string uses a Unicode right-arrow (U+2192) where -> is shown above, and does not wrap the two long lines.

Minimal prompt:

You are a quantum computing expert. Respond with ONLY executable Python code using Qiskit (version 2.x). No explanations, no markdown fences, just the raw code. The code must define a function called solve() that returns the answer. Do NOT use deprecated APIs like .qasm() -- use qiskit.qasm2.dumps() if needed.

## B.2 Task Prompt Examples

T1 — State Preparation (L1, seed=1):

Write a Qiskit function solve() that returns a QuantumCircuit on 2 qubits which prepares the following quantum state from |00>:

|00>: 0.707106781186548 + 0.0i

P. Gupta et al.

|01>: 0.0 + 0.0i   
|10>: 0.0 + 0.0i   
|11>: 0.707106781186548 + 0.0i

Requirements:

\- You may use QuantumCircuit.initialize(statevector, qubits) or any standard gates.

\- Ensure the statevector is normalized before passing to initialize().

\- The circuit should achieve fidelity > 0.999 with the target state.

\- Return ONLY the QuantumCircuit object from solve().

## T4 — Debugging (L2, seed=3):

The following 3-qubit quantum circuit has exactly ONE bug (a wrong gate, swapped qubits, or missing gate).

Buggy circuit (OpenQASM 2.0):   
OPENQASM 2.0; include "qelib1.inc";   
qreg q[3]; h q[0]; cx q[0],q[1]; cx q[0],q[2]; ...

The INTENDED unitary transformation maps basis states as follows:   
|000> -> (0.7071+0.0000i)|000> + (0.7071+0.0000i)|111>   
|001> -> ...

Identify the bug and write a Qiskit function solve() that returns the CORRECTED QuantumCircuit.

## C Dificulty-Level Breakdown

Performance degrades monotonically with dificulty (59% drop from L1 to L5). The steepest L1→L5 drops occur on State Preparation (1.00 → 0.14), VQE (0.96 → 0.12), and Oracle Synthesis (0.88 → 0.24). Debugging falls least in absolute terms (0.16 → 0.02). Noise Fidelity is the only task that holds roughly flat across dificulty levels (0.70 → 0.66).

## D Task Descriptions

Table 2. 2PL IRT fit. Items are the 55 (task, level) cells, each administered to every model under 5 seeds; abilities are standardised to mean 0 and unit SD. 120 parameters (10 abilities, 55 dificulties, 55 discriminations) over 2750 responses. Discrimination carries a log-normal(0, 0.5) prior, without which the 55 slopes are not identifiable from 10 models. In (b), n is the number of identified items entering each mean: State Preparation L1 and L2 were passed by every model under every seed, leaving their dificulty unidentified, so they are excluded.  
(a) Model ability
<table><tr><td rowspan=1 colspan=1>Rank</td><td rowspan=1 colspan=1>Model</td><td rowspan=1 colspan=1>θ</td><td rowspan=1 colspan=1>SE</td><td rowspan=1 colspan=1>95% CI</td></tr><tr><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Gemini 3.5</td><td rowspan=1 colspan=1>1.122</td><td rowspan=1 colspan=1>0.088</td><td rowspan=1 colspan=1>[0.949, 1.295]</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>Sonnet 5</td><td rowspan=1 colspan=1>1.0900</td><td rowspan=1 colspan=1>.088</td><td rowspan=1 colspan=1>[0.919, 1.262]</td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>04-mini</td><td rowspan=1 colspan=1>0.669</td><td rowspan=1 colspan=1>0.085</td><td rowspan=1 colspan=1>[0.502, 0.837]</td></tr><tr><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>GPT-5.4</td><td rowspan=1 colspan=1>0.432</td><td rowspan=1 colspan=1>0.087</td><td rowspan=1 colspan=1>[0.262, 0.603]</td></tr><tr><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>Gemma-4</td><td rowspan=1 colspan=1>0.251</td><td rowspan=1 colspan=1>0.089</td><td rowspan=1 colspan=1>[0.077, 0.425]</td></tr><tr><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>Opus 4.1</td><td rowspan=1 colspan=1>-0.021</td><td rowspan=1 colspan=1>0.093</td><td rowspan=1 colspan=1>[−0.203, 0.161]</td></tr><tr><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>GPT-4.1m</td><td rowspan=1 colspan=1>-0.087</td><td rowspan=1 colspan=1>0.094</td><td rowspan=1 colspan=1>[-0.272, 0.098]</td></tr><tr><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>Gemini FL</td><td rowspan=1 colspan=1>-0.141</td><td rowspan=1 colspan=1>0.096</td><td rowspan=1 colspan=1>[−0.328, 0.046]</td></tr><tr><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>Mistral L3</td><td rowspan=1 colspan=1>-0.843</td><td rowspan=1 colspan=1>0.124</td><td rowspan=1 colspan=1>[-1.086, -0.600]</td></tr><tr><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>LLaMA-70B</td><td rowspan=1 colspan=1>-2.473</td><td rowspan=1 colspan=1>0.223</td><td rowspan=1 colspan=1>[−2.909, -2.037]</td></tr></table>

(b) Task dificulty and discrimination
<table><tr><td rowspan=1 colspan=1>Task</td><td rowspan=1 colspan=1>Mean difficulty b</td><td rowspan=1 colspan=1>Mean discrimination ān</td><td rowspan=1 colspan=1>n</td></tr><tr><td rowspan=1 colspan=1>Routing</td><td rowspan=1 colspan=1>1.897</td><td rowspan=1 colspan=1>0.862</td><td rowspan=1 colspan=1>5</td></tr><tr><td rowspan=1 colspan=1>Debugging</td><td rowspan=1 colspan=1>1.711</td><td rowspan=1 colspan=1>2.267</td><td rowspan=1 colspan=1>5</td></tr><tr><td rowspan=1 colspan=1>Trotter Decomposition</td><td rowspan=1 colspan=1>1.581</td><td rowspan=1 colspan=1>0.987</td><td rowspan=1 colspan=1>5</td></tr><tr><td rowspan=1 colspan=1>State Preparation</td><td rowspan=1 colspan=1>1.116</td><td rowspan=1 colspan=1>2.841</td><td rowspan=1 colspan=1>3</td></tr><tr><td rowspan=1 colspan=1>Noise Discrimination</td><td rowspan=1 colspan=1>0.662</td><td rowspan=1 colspan=1>1.560</td><td rowspan=1 colspan=1>5</td></tr><tr><td rowspan=1 colspan=1>Reverse Engineering</td><td rowspan=1 colspan=1>0.107</td><td rowspan=1 colspan=1>1.349</td><td rowspan=1 colspan=1>5</td></tr><tr><td rowspan=1 colspan=1>Oracle</td><td rowspan=1 colspan=1>0.015</td><td rowspan=1 colspan=1>2.824</td><td rowspan=1 colspan=1>5</td></tr><tr><td rowspan=1 colspan=1>QEC Decoding</td><td rowspan=1 colspan=1>-0.069</td><td rowspan=1 colspan=1>0.794</td><td rowspan=1 colspan=1>5</td></tr><tr><td rowspan=1 colspan=1>Noise Fidelity</td><td rowspan=1 colspan=1>-0.325</td><td rowspan=1 colspan=1>2.526</td><td rowspan=1 colspan=1>5</td></tr><tr><td rowspan=1 colspan=1>VQE</td><td rowspan=1 colspan=1>-0.917</td><td rowspan=1 colspan=1>1.243</td><td rowspan=1 colspan=1>5</td></tr><tr><td rowspan=1 colspan=1>Equivalence</td><td rowspan=1 colspan=1>-1.272</td><td rowspan=1 colspan=1>1.597</td><td rowspan=1 colspan=1>5</td></tr></table>

Table 3. Prompt sensitivity: mean binarized-accuracy diference (structured − minimal) by model and task, over all 11 tasks and all 10 models (2,745 paired evaluations per condition). The structured condition is the canonical system prompt used in the paper. 5 of the Gemma-4-31B State-Preparation L5 calls had to be excluded from this analysis: they could not be obtained in the minimal condition even after 6 retry rounds (persistent upstream 502/503 API errors).
<table><tr><td>Model</td><td>Δ |Task</td><td></td><td>Δ</td></tr><tr><td>o4-mini</td><td>+0.113</td><td>T11 (QEC Decoding)</td><td>+0.244</td></tr><tr><td>Gemma-4</td><td>+0.107</td><td>T9 (Noise Fidelity)</td><td>+0.072</td></tr><tr><td>GPT-4.1 mini +0.095</td><td></td><td>T7 (Equivalence)</td><td>+0.052</td></tr><tr><td>Mistral L3</td><td>+0.062</td><td>[T6 (Reverse Engineering)</td><td>+0.048</td></tr><tr><td>LLaMA-70B</td><td>+0.058</td><td>T8 (Routing)</td><td>+0.044</td></tr><tr><td>GPT-5.4</td><td>+0.029</td><td>T10 (VQE)</td><td>+0.036</td></tr><tr><td>Gemini FL</td><td>+0.007</td><td>T2 (Trotter Decomposition) +0.028</td><td></td></tr><tr><td>Sonnet 5</td><td>+0.000</td><td>T3 (Oracle Synthesis)</td><td>+0.004</td></tr><tr><td>Opus 4.1</td><td>-0.011</td><td>T4 (Debugging)</td><td>-0.016</td></tr><tr><td>Gemini 3.5F</td><td>-0.025</td><td>T5 (Noise Discrimination)</td><td>-0.016</td></tr><tr><td></td><td></td><td>T1 (State Preparation)</td><td>-0.020</td></tr></table>

Table 4. Mean accuracy by dificulty level across all models and tasks.
<table><tr><td>L1</td><td>L2</td><td>L3</td><td>L4</td><td>L5</td></tr><tr><td>Mean score 0.655 0.553 0.416 0.336 0.267</td><td></td><td></td><td></td><td></td></tr></table>

Table 5. QC-Stark task descriptions with scoring criteria and dificulty scaling parameters.
<table><tr><td>Task</td><td>Description</td><td colspan="2">Scoring</td><td>Difficulty Scaling</td></tr><tr><td>T1: State Preparation</td><td>Synthesize gate producing target state from |0) n</td><td>sequence State fidelity &gt; 0.999 quantum</td><td></td><td>L1-2: initialize() allowed; L3+: ele- mentary gates only; number of qubits: 2-6</td></tr><tr><td>T2: Trotter Decomposition</td><td>Decompose Trotter product formula</td><td>-iHt e</td><td>into Operator fidelity to exact Terms: 2-8; order: 1-4; time steps: 1- evolution 20</td><td></td></tr><tr><td>T3: Oracle Synthesis</td><td>Construct unitary from Boolean specification</td><td>oracle Functional target</td><td>equivalence</td><td>to Input bits: 2-5; function complexity: linear to non-linear</td></tr><tr><td>T4: Debugging</td><td>jected fault in quantum cir- cuit</td><td></td><td>Locate and fix single in- Binary: correct circuit or not Qubits: 3-7; fault types: gate substi-</td><td>tution, qubit swap, gate deletion</td></tr><tr><td>T5: Noise Discrimination</td><td>Distinguish noise-corrupted Correct identification circuits from structurally- buggy ones</td><td></td><td></td><td>Channel types: depolarizing, ampli- tude damping; similarity level</td></tr><tr><td>T6: Reverse Engineering</td><td>Identify algorithm from ob- Correct algorithm name fuscated circuit</td><td></td><td></td><td>Algorithms: Bell, GHZ, QFT, Grover, VQE; obfuscation: transpiled to basis gates</td></tr><tr><td>T7: Equivalence</td><td>Determine if two circuits im- Binary correctness plement same unitary</td><td></td><td></td><td>Number of qubits: 2-5; circuit pairs: identical, permuted, distinct</td></tr><tr><td>T8: Routing</td><td>connectivity</td><td>Insert SWAPs for hardware Functional topology</td><td>equivalence</td><td>on Number of qubits: 5-20; topologies: linear, T, heavy-hex</td></tr><tr><td>T9: Noise Fidelity</td><td>Construct matching error profile</td><td>noise</td><td>channel Channel fidelity to target</td><td>Error types: depolarizing, amplitude damping, combined; number of qubits: 1-3</td></tr><tr><td>T10: VQE</td><td>tum eigensolver for ground-</td><td>Implement variational quan- Energy accuracy</td><td></td><td>Qubit count, ansatz depth, Hamilto- nian complexity</td></tr><tr><td>T11: QEC Decoding</td><td>state energy estimation</td><td>Decode stabilizer syndrome Correct error identification</td><td>Codes: repetition,</td><td>Steane, surface;</td></tr></table>