# Unanimously Wrong: Certified Abstention from How Medical LLM Consensus Forms

Xiaoyang Wang Drexel University xw388@drexel.edu

Tianrui Wang University of Michigan, Ann Arbor tianruiw@umich.edu

Christopher C. Yang Drexel University chris.yang@drexel.edu

## Abstract

In clinical practice, agreement among independent experts is treated as evidence of reliability, and multi-round consensus has become a core mechanism of agentic medical question-answering systems. When such a system must decide whether to trust its own answer, the prevailing signal is again agreement, now among the sampled answers. But agreement is a fragile proxy for correctness. A system can be unanimously wrong, returning the same incorrect answer on every sample, and on these questions agreement-based signals carry no information. The cause is that these signals read only the final state of the consensus and discard how it was reached. Agreement that was reached by resolving disagreement with evidence looks identical, at the end, to agreement that was present from the first sample because every sample shares one misconception. ProbeGuard is a certified abstention framework that bases the abstention decision on how the consensus formed. All of its components are computed from the system’s execution log. Process features trace agreement trajectories, minority persistence, and retrieval saturation. For unanimous votes, rationale semantic entropy checks whether the reasons behind the vote cohere, and an active probe retrieves counter-evidence and measures whether the consensus survives. A stratified Learn-then-Test calibration then converts these scores into a distribution-free bound on selective risk. This lets the system separate unanimous votes that are backed by knowledge from those that are not, and answer only where its error rate is provably controlled. We evaluate ProbeGuard on three medical QA benchmarks and a hard-frontier reference, with a published multi-round agentic RAG substrate, against six abstention baselines. On MedQA, 13.4% of unanimous votes are wrong, and no agreement-based signal can flag them. Process signals raise the discrimination of correct from incorrect consensus from chance to 0.696 AUROC. The certified rule answers six in ten unanimous-layer questions at an observed selective risk of 9.0%, and nine in ten once in-domain calibration data accumulate. Code: https://github.com/ wangxiaoyang0412/probeguard.

## 1 Introduction

Language models now encode substantial clinical knowledge [Singhal et al., 2023] and are moving into clinical question answering, where their answers increasingly inform triage, guideline lookup, and decision support. In these settings the failure that matters most is a confidently wrong answer. In a randomized trial, physicians exposed to erroneous LLM suggestions lost 14 percentage points of diagnostic accuracy, despite prior AI-literacy training and voluntary consultation [Qazi et al., 2026]; an answer delivered without any signal of doubt propagates directly into downstream decisions. The model itself cannot be trusted to raise that signal: frontier models almost never abstain when they should, and reasoning-oriented training makes abstention worse [Kirichenko et al., 2025]. A deployed system therefore needs reliable abstention, so that the questions it cannot answer are routed to a clinician [Machcha et al., 2026, Li et al., 2024b].

![](images/e8bddc058d56af7ff3b0dd74ffae2c908e1026c7c95908654a585958f2124f29.jpg)  
Figure 1: Two unanimous consensuses are indistinguishable to snapshot-based abstention (a), yet one is knowledge-backed and one is inherited from a shared misconception. ProbeGuard (b) checks whether the rationales behind a unanimous vote support one another and stress-tests the consensus with retrieved counter-evidence, answering only where selective risk is certifiably bounded.

The dominant abstention signals estimate confidence from a single round of samples: the model’s verbalized certainty [Kadavath et al., 2022, Tian et al., 2023, Xiong et al., 2024b], or the agreement among independently sampled answers [Wang et al., 2023, Manakul et al., 2023, Kuhn et al., 2023]. Agreement scores have even been calibrated into conformal abstention rules directly [Yadkori et al., 2024]. Moving these signals to a modern multi-round consensus system [Wu et al., 2026] exposes their limits. Agreement can be inherited from a shared training-data error, in which case a model gives the same wrong answer eight times. Multi-round systems stop deliberating as soon as agreement is reached, so the terminal vote is 1 for every converged question. And on the questions where the system is unanimously wrong, no vote-based signal can raise an alarm. On our substrate, 59% of MedQA questions reach first-round unanimity, and 13.4% of those unanimous votes are wrong; on expert-level questions the rate rises to 79.0% (Figure 1).

These signals discard theformation process. Consider a panel of physicians who end up agreeing. All of them may have known the answer from the start, or all of them may share one misconception. A forced vote can also produce agreement, and so can a discussion in which they disagreed until one of them found the evidence that settled the question. The final tally is identical in each case, and the differences lie only in the record of the discussion. Multi-round consensus systems write this record to their execution logs, including which candidates disagreed, for how many rounds, which documents were retrieved, and whether the vote changed afterward. Yet prior work discards the trajectory once the answer is produced [Wu et al., 2026, Manczak et al., 2025]. When the vote is unanimous from the start, the trajectory is trivial and the check moves one level down, to the rationales. The reasons behind the vote can be checked for coherence, which extends semantic entropy [Farquhar et al., 2024] from answers to rationales, and the consensus can be tested directly by supplying opposing evidence.

Present work. We introduce ProbeGuard, which turns consensus formation into a certified abstention decision. Process features computed from the execution log trace how the vote evolved; for unanimous votes, rationale semantic entropy and an active counter-evidence probe read the evidence behind the vote; and Learn-then-Test calibration [Angelopoulos et al., 2022, 2024] within process strata yields a selective-risk bound that holds without distributional assumptions, where existing medical abstention methods provide heuristic scores [Dang et al., 2025, Shankar et al., 2025]. Every multi-round system already writes this log, and because the scores are training-free, the entire calibration set can be spent on threshold selection, and the thresholds it returns carry a guarantee a deployment can act on.

We make four contributions. (1) To our knowledge, we give the first abstention decision derived from how multi-round consensus forms, with a distribution-free selective-risk guarantee, in place of snapshot-based confidence assessment. (2) We instrument the execution log with trajectory features at zero additional inference cost, extend semantic entropy from answers to the concluding claims of rationales, and introduce an active counter-evidence probe whose keep rate measures whether a unanimous vote survives opposing evidence. (3) On questions where all N=8 candidates agree immediately, every vote-based signal is constant, yet 100 of 746 such votes on MedQA are wrong. We measure this failure mode across four benchmarks and three backbones. (4) The process score separates correct from incorrect consensus at 0.696 AUROC on MedQA, where terminal agreement scores 0.512, and at 0.704 on MMLU-Pro medicine. Wrong consensus flips under counter-evidence at 14.0–26.5%, against 3.3–5.5% for correct consensus. Stratified Learn-then-Test calibration answers 0.553 of the unanimous layer at 9.0% observed risk (α=0.15, 50 splits).

Table 1: Comparison of abstention approaches for medical QA. External resources, scoring signals, and risk guarantees are reported for each method.
<table><tr><td>Method</td><td>External resource</td><td>Abstention score</td><td>Risk guarantee</td></tr><tr><td>KnowGuard</td><td>medical knowledge graph</td><td>evidence sufficiency</td><td>x</td></tr><tr><td>Energy-based scoring</td><td>separately trained scorer</td><td>energy</td><td>x</td></tr><tr><td>Conformal abstention</td><td>none</td><td>answer agreement</td><td>global</td></tr><tr><td>ProbeGuard (ours)</td><td>none</td><td>consensus formation</td><td>per-stratum</td></tr></table>

## 2 Related Work

Abstention in medical question answering. Abstention is increasingly treated as a required capability of clinical language systems. KnowGuard grounds abstention in evidence-sufficiency scores over an external medical knowledge graph [Dang et al., 2025]; MediQ converts insufficient information into follow-up questions in interactive settings [Li et al., 2024b]; MedAbstain contributes a unified evaluation protocol with explicit abstention options and adversarial variants [Machcha et al., 2026]; and, closest to our setting, an energy-based model scores out-of-scope queries for a medical RAG assistant [Shankar et al., 2025]. These systems either require resources beyond the deployed pipeline or produce heuristic scores whose threshold comes with no error guarantee. ProbeGuard needs only the system’s own execution log and a probe built from its own retrieval stack, and returns a threshold with a certified selective-risk bound (Table 1).

Consistency- and confidence-based uncertainty. For black-box LLMs, the dominant signal families elicit confidence verbally [Kadavath et al., 2022, Tian et al., 2023, Xiong et al., 2024b] or measure agreement across independently sampled responses: self-consistency [Wang et al., 2023], sample-divergence checks [Manakul et al., 2023], and semantic entropy, which clusters samples by bidirectional entailment [Kuhn et al., 2023, Farquhar et al., 2024]; recent surveys place consistencybased estimators among the most widely used black-box approaches [Shorinwa et al., 2025, Liu et al., 2025]. All of these compute their score from a single round of samples. As discussed in §1, agreement can be a shared error, and answer-level agreement can hide disagreement at the reasoning level even in debate systems [Wang and Yang, 2026]. Our features use the same samples and additionally the rounds between the first sample and the final vote.

Multi-round deliberation and its evaluation. Agentic RAG pipelines improve question answering by coordinating specialized agents [Nguyen et al., 2025], and in medical QA by iterating conflictguided retrieval until consensus [Wu et al., 2026], while recent evaluations show that reliability degrades under multi-turn interaction [Manczak et al., 2025] and that reasoning-oriented training can make abstention worse [Kirichenko et al., 2025]. This literature uses deliberation to improve accuracy and leaves the intermediate states unused once the answer is produced. To our knowledge, no prior work uses the deliberation trace as an abstention signal.

Conformal guarantees for language models. Conformal risk control and Learn-then-Test provide distribution-free selective guarantees [Angelopoulos et al., 2024, 2022], instantiated for LLMs in sampling-based generation [Quach et al., 2024], factuality filtering [Mohri and Hashimoto, 2024], retrieval-augmented QA [Li et al., 2024a], answer-set prediction under outliers [Wang et al., 2025], selective trust of generated reports [Gui et al., 2024], and consistency-scored abstention [Yadkori et al., 2024]; in clinical QA, behavioural features of single reasoning traces have been calibrated with conformal sets [Testoni and Calixto, 2025]. In every case the calibrated score is a single-round quantity and the calibration is global over the population. We apply the same procedure to scores computed from the consensus process and calibrate within each process stratum, including the unanimous stratum on which the scores used in prior work are constant.

## 3 Method

Problem Formulation. Let x be a question posed to a multi-round consensus system. We seek an abstention decision

$$
A : x \ \mapsto \ \{ \mathrm { A N S W E R } , \mathrm { A B S T A I N } \} ,\tag{1}
$$

such that, among the questions the system chooses to answer, the error rate is certifiably bounded by a user-specified level α. We build on a published multi-round agentic RAG system for medical QA [Wu et al., 2026], used unmodified at a pinned commit. At each round t the system samples $N { = } 8$ candidate answers together with free-text rationales. If the candidates disagree, it generates conflict-guided queries, retrieves evidence from a medical corpus, injects the evidence into the next round’s context, and re-answers, stopping at unanimity or after $\check { T } { = } 8$ rounds. We add only a logging layer and modify no decision logic. For every round it records the candidate answers with their full rationales, the answer distribution, and all retrieval events (queries and document identifiers). We write $s _ { t } ( a ) \in \{ 0 , 1 / N , \ldots , 1 \}$ for the fraction of candidates voting option a at round $t , m _ { t } = \arg \operatorname* { m a x } _ { a } s _ { t } ( a )$ for the modal answer, $T _ { \mathrm { s t o p } } \leq T$ for the stopping round, and $D _ { t }$ for the set of documents retrieved after round t. ProbeGuard derives its abstention decision from this log through trajectory features of the consensus process (§3.1), evidence signals that remain informative when the vote is unanimous (§3.2), and a stratified calibration that certifies a decision rule from these scores (§3.3).

## 3.1 Consensus-Process Features

The trajectory recorded in the log carries information that is absent from the terminal vote. Two single-round statistics bracket the trajectory. Terminal agreement $s _ { T _ { \mathrm { s t o p } } } ( m _ { T _ { \mathrm { s t o p } } } )$ equals 1 for every converged question because of the early-stopping rule. First-round agreement

$$
a _ { 1 } ~ = ~ s _ { 1 } ( m _ { 1 } )\tag{2}
$$

is the strongest statistic available to single-round sampling methods. The trajectory between them is summarized by the trajectory-averaged agreement, which rewards early, stable consensus over late or forced convergence, and by the majority-flip count, the number of rounds at which the leading answer is overturned. Two further families capture who disagrees and what the retriever is doing. Minority persistence is the share of rounds in which the alternative answer that persists longest retains a supporter; a single alternative that survives every round signals genuine ambiguity, whereas rotating, short-lived minorities are sampling noise. The document-repeat rate is the fraction of newly retrieved documents already seen in earlier rounds; when it rises, the evidence space is saturating and further rounds are unlikely to resolve the disagreement. The full registry comprises 34 features across five families (first-round and trajectory agreement, convergence statistics, minority dynamics, rationale-level signals, and retrieval-side behavior), each with a definition fixed in advance of evaluation; formal definitions are in Appendix B. A combined process score is fit by $\ell _ { 2 }$ -regularized logistic regression over these features with question-level cross-validation, median imputation and missingness indicators computed within training folds (Appendix L).

## 3.2 Evidence Signals for Unanimous Votes

When the first-round vote is unanimous $( a _ { 1 } = 1 )$ , every agreement-based feature in §3.1 is constant: the trajectory has length one and no retrieval is triggered. These are also the questions on which a shared training-data error produces a confident mistake. If the answer rests on knowledge, the eight rationales cite the same mechanism. If it rests on a shared error, no common mechanism exists to cite, and each sample constructs its own justification, so the rationales diverge.

Rationale semantic entropy (passive). We formalize the question of whether eight identical answers share one justification as mutual entailment among the concluding claims of the rationales, treating the consensus as knowledge-backed when its rationales entail one another. This reduces our problem to one with a well-studied estimator, semantic entropy [Kuhn et al., 2023, Farquhar et al., 2024], which clusters sampled answers by bidirectional NLI entailment and computes the entropy of the cluster distribution. On a unanimous vote its answer-level value is identically zero, so we apply the estimator to the rationales instead, conditioned on answer agreement. From each of the N rationales we extract the concluding claim by a deterministic rule frozen before any rationale data were inspected (rule and unit tests in Appendix C). We score concluding claims instead of full rationales because full rationale exceed the NLI context length and share enough surface vocabulary that topical similarity masks logical disagreement. Each ordered claim pair is then scored by an NLI model, yielding entailment probabilities $e _ { i j }$ . Hard entailment clustering saturates at ln $N$ on the most suspicious questions, so we define the score on the continuous probabilities directly,

$$
\mathrm { R S E } \ : = \ : \mathrm { m e a n } _ { i < j } \ : \mathrm { m i n } ( e _ { i j } , e _ { j i } ) ,\tag{3}
$$

where the minimum requires each claim pair to support each other in both directions, and the mean runs over all $\binom { N } { 2 }$ unordered pairs, so that the score rises with the mutual support among the rationales.

Counter-evidence probe (active). The probe reuses the substrate’s retrieval loop with the direction reversed. On contested questions the substrate retrieves evidence to resolve a disagreement; the probe retrieves evidence against an existing agreement and checks whether the agreement survives. If the vote rests on knowledge, the model can state why the challenge does not apply and the vote should survive the strongest available counter-evidence; if it rests on a shared misconception, no mechanism is available to rebut the challenge and we expect the vote to be revised. Concretely, for each unanimous question we (i) prompt the model to generate retrieval queries that would support alternative answers or contradict the consensus, (ii) retrieve challenge documents with the system’s own retrieval stack under its standard parameters, and (iii) let all N candidates re-answer with the challenge documents in context, under an explicit devil’s-advocate instruction that licenses revision without demanding it (prompts in Appendix A; Algorithm 1 in Appendix D). The probe statistic is the keep rate

$$
\begin{array} { r } { \kappa = \frac { 1 } { N } \left| \left\{ i : \mathrm { c a n d i d a t e } \ i \mathrm { r e t a i n s ~ t h e ~ c o n s e n s u s ~ a n s w e r } \right\} \right| , } \end{array}\tag{4}
$$

where the count runs over the $N$ re-answers. The unanimous-layer score combines the passive and active signals as $z ( \kappa ) + z ( \mathrm { R S E } )$ , where $z ( \cdot )$ denotes per-dataset standardization. The probe fires only on the unanimous subset and costs one query-generation call, one retrieval round, and N re-answer calls per probed question; per-question latency and token accounting are reported in Appendix J.

## 3.3 Stratified Certification

Deployment needs a threshold whose selective risk is bounded with stated confidence, which a point estimate on held-out data does not provide. The unanimous and contested subpopulations differ in base error rate, in which signals are defined, and in score scale, so a single global calibration is dominated by this heterogeneity: thresholds that are valid on average can be badly miscalibrated within each subpopulation. Selective risk is also not monotone in the threshold, since admitting more questions can lower the error rate among answered questions, so naive quantile calibration over-rejects or over-admits depending on where the errors sit. We therefore certify per stratum, selecting each threshold with a multiple-testing procedure over a grid of candidates.

The target guarantee is

$$
\operatorname* { P r } \left[ \operatorname { r i s k } ( { \hat { \lambda } } ) \leq \alpha \right] \ \geq \ 1 - \delta ,\tag{5}
$$

where risk(λ) is the error rate among the questions a stratum answers at threshold $\lambda ,$ those scoring strictly above $\lambda ,$ , and the probability is over the calibration draw. The stratification $g$ was fixed before any calibration data were seen: $S _ { 1 }$ (unanimous first round), scored by $z ( \kappa ) + \bar { z } ( \mathrm { R S E } ) ; S _ { 2 }$ (contested but converged), scored by the trajectory-averaged agreement; and $S _ { 3 }$ (never converged), which abstains as a whole. Within each certified stratum we select the threshold by Learn-then-Test [Angelopoulos et al., 2022]: each candidate coverage level c on a grid places a threshold $\lambda _ { c }$ at the topc quantile of the calibration scores, and its selective risk is tested against the null $H _ { 0 } : \mathrm { r i s k } ( \lambda _ { c } ) > \bar { \alpha }$ with the exact binomial tail

$$
p _ { c } \ = \ \mathrm { P r } \big [ \mathrm { B i n } ( n _ { c } , \alpha ) \big \le E _ { c } \big ] ,\tag{6}
$$

where $n _ { c }$ is the number of calibration questions scoring strictly above $\lambda _ { c }$ and $E _ { c }$ the number of errors among them; the grid is tested at Bonferroni-corrected level $\delta / | c |$ (and across certified strata), and $\hat { \lambda }$ is the threshold of the largest coverage whose null is rejected (Algorithm 2, Appendix D).

Assumption 1. Within each stratum, calibration and deployment questions are drawn i.i.d.from the same distribution.

Theorem 1. Under Assumption 1, the threshold $\hat { \lambda }$ returned by Algorithm 2 satisfies the guarantee of Eq. (5) for every certified stratum. (Proof in Appendix E.)

![](images/df426de442f32e95e1f520ffc7c8c562650b933d910940edc3000392cd52a461.jpg)

![](images/b0b90167444837c74ece8c145f38b0588b23e7ec4f88f029f9db0349eed0ca12.jpg)

![](images/b931822575a60ed750ea97b59cb556a19a2919aef7649a2332336b8915b5acd9.jpg)  
Figure 2: (Left) Correct consensus starts near-unanimous, while wrong consensus starts from disagreement and converges to it slowly (MedQA, mean agreement per round, shaded standard errors). (Middle) Error rate rises with the round at which consensus concludes (bars: questions concluding per round; line: error rate at that round; dashed: overall). (Right) On the unanimous layer every vote-based signal is constant, and only the formation signals discriminate.

The guarantee does not depend on the quality of the score; a weak score lowers certified coverage but not the validity of the bound. Because the scores entering certification (RSE, κ, a¯) are training-free, the full calibration split is available for threshold selection, and we report certified coverage and realized risk as means over repeated calibration/test splits rather than a single draw.

## 4 Experiments and Results

## 4.1 Experimental Settings

Datasets. We evaluate on MedQA [Jin et al., 2020], the medicine subset of MMLU-Pro [Wang et al., 2024], and MedMCQA [Pal et al., 2022], in the form distributed with the substrate. MedXpertQA [Zuo et al., 2025] is a hard-frontier reference on which substrate accuracy approaches the guessing floor. Specialty labels on MedQA and MedMCQA support Appendix I.

Substrate and backbones. All experiments instrument the entropy-guided variant of the substrate at a pinned commit, with its retrieval stack unchanged (BM25 with MedCPT cross-encoder reranking [Jin et al., 2023] over MedCorp [Xiong et al., 2024a]). Final answers are scored by the substrate’s own rule (strict majority, ties as errors), and a logging layer records every round verbatim. We evaluate Qwen3-8B [Yang et al., 2025] as the primary backbone, Llama-3.1-8B-Instruct [Grattafiori et al., 2024], and the medical reasoning model HuatuoGPT-o1-8B [Chen et al., 2024], all under identical decoding configuration and seed (Appendix L).

Baselines. We compare against the abstention signals available to a deployed consensus system: terminal agreement, first-round agreement [Wang et al., 2023], mean sequence log-probability, verbalized confidence [Tian et al., 2023, Xiong et al., 2024b], and P(True) [Kadavath et al., 2022]. On multiple choice, semantic entropy [Kuhn et al., 2023, Farquhar et al., 2024] reduces to answerdistribution entropy, which we report as its representative. As the certification control, first-round agreement is calibrated with the same stratified machinery, following the single-round consistencyplus-conformal recipe [Yadkori et al., 2024].

Metrics and certification. Discrimination is reported as AUROC, selective prediction as risk– coverage curves, and probe behavior as flip and keep rates (Eq. (4)); flip rates carry Wilson intervals and certification numbers carry standard errors over repeated splits. The combined process score is fit by logistic regression under question-level cross-validation with fold-internal preprocessing, and rationale claim pairs are scored by DeBERTa-large fine-tuned on MNLI [He et al., 2021] (Appendix L). Certification follows the pre-registered stratified protocol of §3.3, calibrated independently per domain on an even calibration/test split, sweeping α ∈ {0.05, 0.10, 0.15, 0.20}.

## 4.2 Discrimination on the Full Question Stream

Figure 2 shows that wrong and correct consensus form differently. Correct consensus starts nearunanimous; wrong consensus starts from disagreement, and the error rate rises from 13.4% at instant consensus to 58.2% by round four. Table 2 reports how well each signal separates correct from incorrect final answers. Terminal agreement, the deployed default, is at chance on every benchmark. This follows from the early-stopping rule (§3.1). The process score, computed from the same runs, raises discrimination to 0.696 on MedQA, 0.704 on MMLU-Pro medicine, and 0.659 on MedMCQA, and yields the lowest selective risk on MedQA and MedMCQA (risk-coverage curves in Appendix F). The remaining single-round signals fall in a narrow band, which the process score exceeds on every benchmark. This aggregate comparison averages over the unanimous layer, which §4.3 examines separately.

Table 2: Discrimination (AUROC, higher is better) and selective risk (AURC, lower is better) of abstention signals on the Qwen3-8B substrate. Best value per column in bold. Verbalized confidence and P(True) were elicited on the two primary benchmarks.
<table><tr><td></td><td colspan="3">AUROC ↑</td><td colspan="3">AURC↓</td></tr><tr><td>Signal</td><td>MedQA</td><td>MMLU-Pro</td><td>MedMCQA</td><td>MedQA</td><td>MMLU-Pro</td><td>MedMCQA</td></tr><tr><td>Terminal agreement</td><td>0.512</td><td>0.504</td><td>0.507</td><td>0.221</td><td>0.300</td><td>0.330</td></tr><tr><td>First-round agreement [Wang et al., 2023]</td><td>0.677</td><td>0.698</td><td>0.640</td><td>0.155</td><td>0.200</td><td>0.250</td></tr><tr><td>Sequence log-probability</td><td>0.400</td><td>0.474</td><td>0.534</td><td>0.320</td><td>0.376</td><td>0.347</td></tr><tr><td>Answer entropy [Kuhn et al., 2023]</td><td>0.681</td><td>0.703</td><td>0.641</td><td>0.154</td><td>0.199</td><td>0.250</td></tr><tr><td>Verbalized confidence [Tian et al., 2023]</td><td>0.585</td><td>0.651</td><td></td><td>0.179</td><td>0.207</td><td></td></tr><tr><td>P(True) [Kadavath et al., 2022]</td><td>0.602</td><td>0.644</td><td>一</td><td>0.160</td><td>0.205</td><td></td></tr><tr><td>ProbeGuard (process score)</td><td>0.696</td><td>0.704</td><td>0.659</td><td>0.145</td><td>0.205</td><td>0.234</td></tr></table>

Table 3: Process score versus terminal agreement across backbones. ProbeGuard separates correct from incorrect consensus at 0.696–0.825 AUROC where terminal agreement is chance, and has the lower selective risk in every cell.
<table><tr><td></td><td colspan="4">MedQA</td><td colspan="4">MMLU-Pro med</td></tr><tr><td></td><td colspan="2">AUROC↑</td><td colspan="2">AURC↓</td><td colspan="2">AUROC↑</td><td colspan="2">AURC↓</td></tr><tr><td>Backbone</td><td>ProbeGuard</td><td>Term.</td><td>ProbeGuard</td><td>Term.</td><td>ProbeGuard</td><td>Term.</td><td>ProbeGuard</td><td>Term.</td></tr><tr><td>Qwen3-8B</td><td>0.696</td><td>0.512</td><td>0.145</td><td>0.221</td><td>0.704</td><td>0.504</td><td>0.205</td><td>0.300</td></tr><tr><td>HuatuoGPT-o1-8B</td><td>0.775</td><td>0.519</td><td>0.096</td><td>0.193</td><td>0.701</td><td>0.526</td><td>0.225</td><td>0.353</td></tr><tr><td>Llama-3.1-8B</td><td>0.825</td><td>0.548</td><td>0.142</td><td>0.289</td><td>0.782</td><td>0.551</td><td>0.246</td><td>0.406</td></tr></table>

The other backbones behave alike (Table 3): the largest gap is on Llama-3.1-8B, where the process score reaches 0.825 AUROC against 0.548 for terminal agreement. HuatuoGPT-o1-8B preserves the ordering, and the process score has the lower AURC in all six cells, less than half of terminal agreement’s on Llama/MedQA.

## 4.3 The Unanimous Layer

Questions on which all eight candidates agree in the first round are the ones where a wrong answer is hardest to catch. This layer covers 59%, 59%, and 55% of MedQA, MMLU-Pro medicine, and MedMCQA, where 13.4%, 17.7%, and 22.3% of unanimous answers are wrong. The error rate grows with difficulty; on expert-frontier MedXpertQA, where 29% of questions still reach first-round unanimity, 79.0% of those unanimous votes are wrong.

Rationale semantic entropy, computed from the existing rationales, separates correct from incorrect unanimity at 0.587 AUROC on MedQA and 0.613 on MMLU-Pro medicine, on a subpopulation where the answers themselves are identical.

The counter-evidence probe separates the two more sharply. Figure 3 shows each unanimous consensus re-asked with retrieved counter-evidence in context: wrong consensus flips at 14.0–26.5% across the three benchmarks, while correct consensus stays at 3.3–5.5%. The per-question keep rate scores this asymmetry (0.642, 0.638, and 0.729 AUROC on the three benchmarks), and combining it with rationale entropy lifts the unanimous layer to 0.685 and 0.688 on the two primary benchmarks. Sequence log-probability, at chance or inverted on the full stream (Table 2), recovers to 0.680 inside the layer, while the self-report baselines stay behind (P(True) 0.602, verbalized confidence 0.599). Adding it to the keep-and-entropy pair lifts the layer to 0.730 AUROC, so the likelihood and process signals are complementary.

Control arms on MedQA attribute the flips to the retrieved content (Appendix G). The devil’s-advocate instruction alone flips none of the 746 consensus votes, irrelevant documents flip 1.0% of wrong consensus, and the counter-evidence itself under a neutral instruction retains nearly the full separation (12.0% versus 3.6%).

![](images/71e039bda0e95f89186e3d96a9e7f83254499a5a888cf695ae6eedb5e3fff56e.jpg)  
Figure 3: Flip rates of wrong and correct consensus under counter-evidence. A consensus counts as flipped when the majority vote of the N=8 re-answers no longer lands on the original answer. Flip rates per backbone and benchmark with Wilson 95% confidence intervals. The asymmetry holds in every cell, and the per-question keep rate discriminates on all seven (AUROC 0.638–0.729).

The asymmetry recurs on the other backbones. On HuatuoGPT-o1-8B, whose MedQA unanimous layer errs at only 6.2%, wrong consensus still flips at 15.4% against 3.7% for correct. Llama-3.1-8B revises far more readily and flips even correct consensus at up to 28.3%, but wrong consensus flips at 54.2%. Across all seven cells of Figure 3 the keep-rate AUROC stays within 0.638–0.729.

## 4.4 Certification Results

We now calibrate these scores to the guarantee of Eq. (5). Without stratification, naive quantile calibration on the full question stream misses its target (realized test risk 15.7% at α=0.10 on MedQA), and Learn-then-Test applied globally is valid but returns no certified threshold at any α we sweep. Both fail for the heterogeneity described in §3.3: no single threshold on the pooled stream is both valid and non-trivial.

With per-stratum certification, on the MedQA unanimous layer the combined keep-and-entropy score certifies 0.553±0.038 of questions at α=0.15 over 50 splits, with a realized risk on answered questions of 9.0%, well below the bound. Every vote-based signal of Table 2 is constant on this stratum, so none of them can certify non-zero coverage here. Multiplied by the layer’s share of the benchmark, this certified region amounts to about 32% of all MedQA questions. Enlarging the calibration side with in-domain unanimous questions (n=875) lifts certified coverage at $\alpha { = } 0 . 1 5$ to 0.892±0.004 and certifies α=0.10 for the first time (0.219±0.023, realized risk 6.5%), with the test half untouched. On the other backbones, without augmentation, HuatuoGPT-o1-8B certifies 0.410 of its unanimous layer at α=0.10 and 0.997 at $\alpha { = } 0 . 1 5$ , and Llama-3.1-8B 0.838 at α=0.15 (Figure 4, Appendix E).

Calibration is per domain: pooling the calibration sets of MedQA and MMLU-Pro medicine yields an empty certified region at any level, because the two unanimous layers differ in base error rate (§4.3). Per-domain calibration is also what Assumption 1 requires.

## 5 Analysis and Discussion

## 5.1 Ablation Studies

Only the rationale family is non-redundant: dropping it lowers MedQA AUROC from 0.696 to 0.682. The trajectory families are redundant with one another; each alone supports an AUROC close to the full score, and each tracks difficulty proxies (retrieval volume, stopping round) at $| \rho | \geq 0 . 9 3$ Rationale entropy is the exception on both counts $( | \rho | \le 0 . 3 9 )$ : it is the only family whose removal lowers AUROC and the only one that does not track difficulty (full drop-one matrices in Appendix H).

Inside the unanimous layer, the two evidence signals are complementary: their combination outperforms either alone (§4.3). Replacing hard entailment clustering with the continuous score of Eq. (3) raises the unanimous-layer AUROC from 0.578 to 0.587 (MedQA) and 0.593 to 0.613 (MMLU-Pro medicine).

First-round agreement isolates the effect of aggregating votes without any process information; inside the unanimous layer, which carries 100 wrong answers on MedQA alone, it is constant, and the separation reported in §4.3 comes entirely from the process signals.

## 5.2 Cost and Sensitivity

The probe fires only on unanimous first-round votes, at a median of 114 s and 4381 completion tokens on MedQA, against 72 s for generation itself (full accounting in Appendix J); all passive features come from logs the system already writes. Excluding the probe-pilot questions from evaluation shifts the combined AUROC by 0.007, and gradient-boosted trees in place of logistic regression change no conclusion (Appendix H).

## 5.3 Discussion and Limitations

A hospital deploying a consensus QA system would calibrate ProbeGuard on its own case mix and refer every question outside the certified region to a clinician. Each referred question carries its challenge documents and the record of dissent, so the clinician sees the evidence that led the system to withhold an answer. The signals are defined for any system that deliberates in rounds and require nothing beyond its execution log. Tighter certificates require more calibration data, and in-domain unanimous questions accumulate as the system runs (§4.4). The guarantee applies in any stratum whose base error rate is at or below the target α; a stratum above it receives no certified region.

The evaluation is limited to multiple-choice benchmarks, so whether the flip asymmetry and the rationale signals carry over to free-text clinical questions is untested. The three backbones are 8B-scale models on a single published substrate; the signal definitions transfer to other deliberating systems, but we have not measured them elsewhere. The certificates hold under Assumption 1, so a deployment must calibrate on its own case mix and recalibrate when that mix drifts. And we have not evaluated whether clinicians find the returned dissent records useful; a prospective evaluation with clinicians is the next step.

## 6 Conclusion

We presented ProbeGuard, a certified abstention framework for multi-round consensus medical QA. Its abstention decision is computed from the record of how the agreement was reached: process features from the execution log, rationale entropy and a counter-evidence probe for unanimous votes, and a stratified Learn-then-Test calibration of the resulting scores at a user-specified selective-risk level. Across three backbones and three benchmarks, terminal agreement is at chance, the process signals separate correct from incorrect consensus on the full stream and inside the unanimous layer, and stratified calibration certifies answering on part of that layer at the target risk.

## References

Anastasios N. Angelopoulos, Stephen Bates, Emmanuel J. Candès, Michael I. Jordan, and Lihua Lei. Learn then test: Calibrating predictive algorithms to achieve risk control, 2022. URL https://arxiv.org/abs/2110.01052.

Anastasios N. Angelopoulos, Stephen Bates, Adam Fisch, Lihua Lei, and Tal Schuster. Conformal risk control, 2024. URL https://arxiv.org/abs/2208.02814.

Junying Chen, Zhenyang Cai, Ke Ji, Xidong Wang, Wanlong Liu, Rongsheng Wang, Jianye Hou, and Benyou Wang. Huatuogpt-o1, towards medical complex reasoning with llms, 2024. URL https://arxiv.org/abs/2412.18925.

Xilin Dang, Kexin Chen, Xiaorui Su, Ayush Noori, Iñaki Arango, Lucas Vittor, Xinyi Long, Yuyang Du, Marinka Zitnik, and Pheng Ann Heng. Knowguard: Knowledge-driven abstention for multiround clinical reasoning, 2025. URL https://arxiv.org/abs/2509.24816.

Sebastian Farquhar, Jannik Kossen, Lorenz Kuhn, and Yarin Gal. Detecting hallucinations in large language models using semantic entropy. Nature, 630(8017):625–630, 2024. ISSN 1476-4687. doi: 10.1038/s41586-024-07421-0. URL http://dx.doi.org/10.1038/s41586-024-07421-0.

Aaron Grattafiori et al. The llama 3 herd of models, 2024. URL https://arxiv.org/abs/2407. 21783.

Yu Gui, Ying Jin, and Zhimei Ren. Conformal alignment: Knowing when to trust foundation models with guarantees, 2024. URL https://arxiv.org/abs/2405.10301.

Pengcheng He, Xiaodong Liu, Jianfeng Gao, and Weizhu Chen. Deberta: Decoding-enhanced bert with disentangled attention, 2021. URL https://arxiv.org/abs/2006.03654.

Di Jin, Eileen Pan, Nassim Oufattole, Wei-Hung Weng, Hanyi Fang, and Peter Szolovits. What disease does this patient have? a large-scale open domain question answering dataset from medical exams, 2020. URL https://arxiv.org/abs/2009.13081.

Qiao Jin, Won Kim, Qingyu Chen, Donald C. Comeau, Lana Yeganova, W. John Wilbur, and Zhiyong Lu. Medcpt: Contrastive pre-trained transformers with large-scale pubmed search logs for zero-shot biomedical information retrieval, 2023. URL https://arxiv.org/abs/2307.00589.

Saurav Kadavath et al. Language models (mostly) know what they know, 2022. URL https: //arxiv.org/abs/2207.05221.

Polina Kirichenko, Mark Ibrahim, Kamalika Chaudhuri, and Samuel J. Bell. Abstentionbench: Reasoning llms fail on unanswerable questions, 2025. URL https://arxiv.org/abs/2506. 09038.

Lorenz Kuhn, Yarin Gal, and Sebastian Farquhar. Semantic uncertainty: Linguistic invariances for uncertainty estimation in natural language generation, 2023. URL https://arxiv.org/abs/ 2302.09664.

Shuo Li, Sangdon Park, Insup Lee, and Osbert Bastani. Traq: Trustworthy retrieval augmented question answering via conformal prediction, 2024a. URL https://arxiv.org/abs/2307. 04642.

Shuyue Stella Li, Vidhisha Balachandran, Shangbin Feng, Jonathan S. Ilgen, Emma Pierson, Pang Wei Koh, and Yulia Tsvetkov. Mediq: Question-asking llms and a benchmark for reliable interactive clinical reasoning, 2024b. URL https://arxiv.org/abs/2406.00922.

Xiaoou Liu, Tiejin Chen, Longchao Da, Chacha Chen, Zhen Lin, and Hua Wei. Uncertainty quantification and confidence calibration in large language models: A survey. In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2, KDD ’25, page 6107–6117. ACM, 2025. doi: 10.1145/3711896.3736569. URL http://dx.doi.org/10. 1145/3711896.3736569.

Sravanthi Machcha, Sushrita Yerra, Sahil Gupta, Aishwarya Sahoo, Sharmin Sultana, Hong Yu, and Zonghai Yao. Knowing when to abstain: Medical llms under clinical uncertainty, 2026. URL https://arxiv.org/abs/2601.12471.

Potsawee Manakul, Adian Liusie, and Mark J. F. Gales. Selfcheckgpt: Zero-resource black-box hallucination detection for generative large language models, 2023. URL https://arxiv.org/ abs/2303.08896.

Blazej Manczak, Eric Lin, Francisco Eiras, James O’Neill, and Vaikkunth Mugunthan. Shallow robustness, deep vulnerabilities: Multi-turn evaluation of medical llms, 2025. URL https: //arxiv.org/abs/2510.12255.

Christopher Mohri and Tatsunori Hashimoto. Language models with conformal factuality guarantees, 2024. URL https://arxiv.org/abs/2402.10978.

Thang Nguyen, Peter Chin, and Yu-Wing Tai. Ma-rag: Multi-agent retrieval-augmented generation via collaborative chain-of-thought reasoning, 2025. URL https://arxiv.org/abs/2505.20096.

Ankit Pal, Logesh Kumar Umapathi, and Malaikannan Sankarasubbu. Medmcqa : A large-scale multi-subject multi-choice dataset for medical domain question answering, 2022. URL https: //arxiv.org/abs/2203.14371.

Ihsan Ayyub Qazi, Ayesha Ali, Asad Ullah Khawaja, Muhammad Junaid Akhtar, Ali Zafar Sheikh, and Muhammad Hamad Alizai. Automation bias in large language model–assisted diagnostic reasoning among physicians trained in ai literacy — a randomized clinical trial. NEJM AI, 3(5), 2026. ISSN 2836-9386. doi: 10.1056/aioa2501001. URL http://dx.doi.org/10.1056/AIoa2501001.

Victor Quach, Adam Fisch, Tal Schuster, Adam Yala, Jae Ho Sohn, Tommi S. Jaakkola, and Regina Barzilay. Conformal language modeling, 2024. URL https://arxiv.org/abs/2306.10193.

Ravi Shankar, Sheng Wong, Lin Li, Magdalena Bachmann, Alex Silverthorne, Beth Albert, and Gabriel Davis Jones. Energy landscapes enable reliable abstention in retrieval-augmented large language models for healthcare, 2025. URL https://arxiv.org/abs/2509.04482.

Ola Shorinwa, Zhiting Mei, Justin Lidard, Allen Z. Ren, and Anirudha Majumdar. A survey on uncertainty quantification of large language models: Taxonomy, open research challenges, and future directions. ACM Computing Surveys, 58(3):1–38, 2025. ISSN 1557-7341. doi: 10.1145/3744238. URL http://dx.doi.org/10.1145/3744238.

Karan Singhal, Shekoofeh Azizi, Tao Tu, S. Sara Mahdavi, Jason Wei, Hyung Won Chung, Nathan Scales, Ajay Tanwani, Heather Cole-Lewis, Stephen Pfohl, Perry Payne, Martin Seneviratne, Paul Gamble, Chris Kelly, Abubakr Babiker, Nathanael Schärli, Aakanksha Chowdhery, Philip Mansfield, Dina Demner-Fushman, Blaise Agüera y Arcas, Dale Webster, Greg S. Corrado, Yossi Matias, Katherine Chou, Juraj Gottweis, Nenad Tomasev, Yun Liu, Alvin Rajkomar, Joelle Barral, Christopher Semturs, Alan Karthikesalingam, and Vivek Natarajan. Large language models encode clinical knowledge. Nature, 620(7972):172–180, 2023. ISSN 1476-4687. doi: 10.1038/s41586-023-06291-2. URL http://dx.doi.org/10.1038/s41586-023-06291-2.

Alberto Testoni and Iacer Calixto. Mind the gap: Benchmarking llm uncertainty and calibration with specialty-aware clinical qa and reasoning-based behavioural features, 2025. URL https: //arxiv.org/abs/2506.10769.

Katherine Tian, Eric Mitchell, Allan Zhou, Archit Sharma, Rafael Rafailov, Huaxiu Yao, Chelsea Finn, and Christopher D. Manning. Just ask for calibration: Strategies for eliciting calibrated confidence scores from language models fine-tuned with human feedback, 2023. URL https: //arxiv.org/abs/2305.14975.

Xiaoyang Wang and Christopher C. Yang. The consistency illusion: How multi-agent debate hides reasoning misalignment, 2026. URL https://arxiv.org/abs/2606.08457.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models, 2023. URL https://arxiv.org/abs/2203.11171.

Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, Tianle Li, Max Ku, Kai Wang, Alex Zhuang, Rongqi Fan, Xiang Yue, and Wenhu Chen. Mmlu-pro: A more robust and challenging multi-task language understanding benchmark, 2024. URL https://arxiv.org/abs/2406.01574.

Zhiyuan Wang, Qingni Wang, Yue Zhang, Tianlong Chen, Xiaofeng Zhu, Xiaoshuang Shi, and Kaidi Xu. Sconu: Selective conformal uncertainty in large language models, 2025. URL https: //arxiv.org/abs/2504.14154.

Wenhao Wu, Zhentao Tang, Yafu Li, Shixiong Kai, Mingxuan Yuan, Zhenhong Sun, Chunlin Chen, and Zhi Wang. From conflict to consensus: Boosting medical reasoning via multi-round agentic rag, 2026. URL https://arxiv.org/abs/2603.03292.

Guangzhi Xiong, Qiao Jin, Zhiyong Lu, and Aidong Zhang. Benchmarking retrieval-augmented generation for medicine, 2024a. URL https://arxiv.org/abs/2402.13178.

Miao Xiong, Zhiyuan Hu, Xinyang Lu, Yifei Li, Jie Fu, Junxian He, and Bryan Hooi. Can llms express their uncertainty? an empirical evaluation of confidence elicitation in llms, 2024b. URL https://arxiv.org/abs/2306.13063.

Yasin Abbasi Yadkori, Ilja Kuzborskij, David Stutz, András György, Adam Fisch, Arnaud Doucet, Iuliya Beloshapka, Wei-Hung Weng, Yao-Yuan Yang, Csaba Szepesvári, Ali Taylan Cemgil, and Nenad Tomasev. Mitigating llm hallucinations via conformal abstention, 2024. URL https: //arxiv.org/abs/2405.01563.

Yuxin Zuo, Shang Qu, Yifei Li, Zhangren Chen, Xuekai Zhu, Ermo Hua, Kaiyan Zhang, Ning Ding, and Bowen Zhou. Medxpertqa: Benchmarking expert-level medical reasoning and understanding, 2025. URL https://arxiv.org/abs/2501.18362.

An Yang et al. Qwen3 technical report, 2025. URL https://arxiv.org/abs/2505.09388.

```handlebars
Candidate answering (system, all rounds)
You are a medical assistant, please answer my medical questions. Give your final
choice (capital option) closed in tag ‘<answer>X</answer>‘ after your analysis.
Your response should be as detailed as possible, but please do not use any
subheadings.
Candidate answering (user, round 1)
Below is a multiple-choice question.
### Question
{{question}}
### Options
{{options}}
Please analyze the question and give your answer. Give your analysis and final
choice.
```

## A Prompts

This appendix reproduces every prompt in the pipeline verbatim. The first group belongs to the substrate and is unchanged from the pinned commit; we list it for self-containment, and because two of its design choices matter for interpreting our results. The second group is ours: the two probe templates and the two self-report elicitation prompts. Template variables appear in double braces.

Substrate prompts. Candidate answering uses a plain system instruction with a tagged-answer format; the same system prompt is reused by later rounds and by our probe re-answering, so no probe effect can be attributed to a change of role or format.

Two features of the substrate’s later-round prompt bear directly on §4.2. First, candidates are shown the previous answers sorted by token-entropy confidence, together with an explicit caveat that the ranking may be wrong. Second, the prompt closes by instructing the model to answer towards lower entropy: convergence is not only an emergent behavior of the panel but an instructed objective. A system whose members are told to converge will converge on wrong answers as readily as on right ones, which is one more reason the terminal agreement it produces carries no information about correctness.

```handlebars
Evidence-conditioned re-answering (user, rounds ≥ 2)
Below is a multiple-choice question.
### Question
{{question}}
### Options
{{options}}
### Documents
{{documents}}
I will provide several assistant’s previous answers, which may be incorrect.
The previous answers are sorted by their confidence (low entropy means high
confidence), but the entropy metric cannot accurately reflect whether the answer
is correct or not.
Please analyze the previous answers and then re-answer.
{{answers}}
The above are the question along with the assistant’s previous answers, and
some relevant documents. Please analyze and then re-answer. Please give your
response towards **lower entropy**. Give your analysis and final answer.
```

When the panel disagrees, the substrate generates retrieval queries from the disagreement itself. The query prompt asks for the dispute points among the answers, which is precisely the structure our counter-query prompt inverts.

```markdown
Conflict-guided query generation (system)
### Role & Goal:
You are a medical expert. I will provide multiple different answers to the
same question, which may be incorrect. Your task is to identify contradictions,
ambiguities, and core dispute points among the answers, and extract key concept
queries for further retrieval and verification.
### Processing Steps:
1. Analyze different answers and summarize the differences between them.
2. Extract keywords from these differences for retrieval.
3. Generate 1-4 precise queries for further search using **BM25** retriever.
### Output format:
ONLY give the queries:
[Query 1] xxx
[Query 2] xxx
(more queries)
```

Probe prompts. The counter-query prompt keeps the substrate’s query-generation structure (role header, numbered steps implied by the format, BM25-targeted output) and changes only the direction of search: instead of resolving a disagreement, it retrieves evidence against an agreement. Keeping the format aligned means the probe exercises the same retrieval machinery under the same interface, so flip behavior reflects the consensus, not a novel pipeline.

Counter-evidence query generation (system; thinking mode, temperature 0)   
### Role & Goal:   
You are a skeptical medical expert. All assistants unanimously chose the   
same answer to a question. Your task is to actively search for evidence that   
could CHALLENGE or CONTRADICT that consensus: evidence supporting plausible   
alternative options, known exceptions, contraindications, or updated guidelines.   
### Output format:   
ONLY give the queries:   
[Query 1] xxx   
[Query 2] xxx   
(more queries, 2-4 total, for a \*\*BM25\*\* retriever)

Counter-evidence query generation (user)   
Below is a multiple-choice question.   
### Question   
{{question}}   
### Options   
{{options}}   
All assistants unanimously answered: {{consensus}}.   
Generate 2-4 precise BM25 queries to retrieve evidence that could challenge this   
consensus (e.g., evidence favoring other options or against {{consensus}}). Give   
your queries in the given format.

The re-answering prompt needs the most care. A naive challenge instruction produces sycophantic flipping: models abandon any answer once an authority questions it, which would erase the very asymmetry the probe measures. The wording below therefore allows but does not require a change of answer; the key sentence is do not change your answer unless the evidence genuinely warrants it. Under this instruction correct consensus holds at 3.3–5.5% while wrong consensus flips at 14.0–26.5% (Figure 3), and the control arms of Appendix G confirm the division of labor directly: the instruction alone flips nothing, and the counter-evidence carries the asymmetry.

```handlebars
Devil’s-advocate re-answering (user; system prompt identical to candidate answering)
Below is a multiple-choice question.
### Question
{{question}}
### Options
{{options}}
```

```handlebars
### Challenge Documents
A skeptical reviewer retrieved the following documents that may challenge the
previous consensus:
{{documents}}
Previously, all assistants unanimously answered {{consensus}}. A devil’s
advocate now asks: in light of the challenge documents above, critically
re-examine whether {{consensus}} is really correct, or whether another option
is better supported. Weigh the challenge evidence seriously, but do not change
your answer unless the evidence genuinely warrants it. Give your analysis and
final choice.
```

Self-report elicitation prompts. Both baselines are elicited post hoc on the system’s final answer, one call per question at temperature 0. Verbalized confidence follows the 0–100 elicitation of Tian et al. [2023], Xiong et al. [2024b]; P(True) follows Kadavath et al. [2022], with the score read from the first generated token’s top-20 log-probability mass over True/False variants rather than from the sampled text.

```handlebars
Verbalized confidence (system + user)
You are a careful medical assistant. Rate how confident you are that a
given answer to a multiple-choice question is correct. Reply with a single
integer from 0 to 100 inside <conf></conf> tags, then one short sentence of
justification.
### Question
{{question}}
### Options
{{options}}
### Proposed answer: {{answer}}
How confident are you (0-100) that this answer is correct? Reply with
<conf>N</conf> first.
```

```handlebars
P(True) (system + user)
You are a careful medical assistant. Judge whether a given answer to a
multiple-choice question is correct. Reply with exactly one word: True or
False.
### Question
{{question}}
### Options
{{options}}
### Proposed answer: {{answer}}
Is the proposed answer correct? Reply with exactly one word: True or False.
```

## B Feature Definitions

The process score is fit over five feature families, each built from a small set of base quantities and their standard temporal aggregations (value at the first and last round, mean, slope, and maximum over rounds). Table 4 summarizes what each family measures; the representative statistics introduced in §3.1 are the trajectory-averaged agreement

$$
\begin{array} { r } { \bar { a } ~ = ~ \frac { 1 } { T _ { \mathrm { s t o p } } } \sum _ { t = 1 } ^ { T _ { \mathrm { s t o p } } } { s _ { t } ( m _ { t } ) } , } \end{array}\tag{7}
$$

the majority-flip count

$$
n _ { \mathrm { H i p } } ~ = ~ \Big | \{ t < T _ { \mathrm { s t o p } } : m _ { t + 1 } \neq m _ { t } \} \Big | ,\tag{8}
$$

minority persistence

$$
\pi = \operatorname* { m a x } _ { a \neq m ^ { * } } \frac { 1 } { T _ { \mathrm { s t o p } } } \big | \{ t : s _ { t } ( a ) \geq 1 / N \} \big | ,\tag{9}
$$

where $m ^ { * }$ is the final answer and the count is the number of rounds in which option a retains at least one supporter, and the document-repeat rate

$$
\begin{array} { r } { \rho _ { t } \ = \ \big \vert \ D _ { t } \cap \bigcup _ { t ^ { \prime } < t } D _ { t ^ { \prime } } \big \vert / | D _ { t } | , } \end{array}\tag{10}
$$

with first-round agreement $a _ { 1 }$ as in Eq. (2) and the rationale representative as in Eq. (3). The answer-distribution family is built on the per-round answer entropy

$$
\begin{array} { r } { H _ { t } ~ = ~ - \sum _ { a } s _ { t } ( a ) \ln s _ { t } ( a ) , } \end{array}\tag{11}
$$

aggregated as its first- and last-round values, its slope, and the fraction of rounds in which it does not increase, together with the Jensen–Shannon divergence between the answer distributions of adjacent rounds.

Table 4: Feature families of the process score.
<table><tr><td>Family</td><td>Measures</td><td>Representative</td></tr><tr><td>Agreement dynamics</td><td>how fast and how stably the vote converges</td><td>Eqs. (2), (7), (8)</td></tr><tr><td>Answer-distribution dynamics</td><td>entropy and drift of the candidate answer distribution</td><td>Eq. (11)</td></tr><tr><td>Minority dynamics</td><td>whether dissent is persistent or transient</td><td>Eq. (9)</td></tr><tr><td>Rationale signals</td><td>mutual entailment among the reasons behind the votes</td><td>Eq. (3)</td></tr><tr><td>Retrieval dynamics</td><td>whether deliberation still finds new evidence</td><td>Eq. (10)</td></tr></table>

Every definition, the probe protocol, and the stratification were frozen in a versioned registry before the corresponding data were collected; the machine-readable registry with exact formulas for all features ships with the code.

## C Claim Extraction Rule

Rationale semantic entropy scores mutual entailment among the concluding claims of the panel’s rationales, not among the rationales themselves. This appendix documents why that restriction is necessary and gives the exact extraction rule.

Scoring full rationales fails in two ways. Rationales run to hundreds of tokens, well past the effective input of the entailment model, so truncation decides what gets compared. More fundamentally, two rationales about the same clinical question share most of their surface vocabulary whether or not they agree: both walk the same options, cite the same anatomy, and eliminate the same distractors. Entailment scores over full rationales therefore saturate on topical similarity, and the logical disagreement, which lives almost entirely in the final verdict sentences, is drowned out. Restricting the comparison to concluding claims is what makes the NLI calls well-posed.

Prior work that decomposes generations into checkable units typically prompts a strong LLM to do the splitting [Mohri and Hashimoto, 2024]. We use a deterministic rule instead, for three reasons. A fixed rule can be frozen before any rationale data are inspected, which an LLM extractor cannot guarantee; it adds zero inference cost to a feature set whose selling point is being free; and it avoids a circularity, since asking a model to summarize its own reasoning gives it a second chance to smooth over exactly the incoherence the feature is trying to measure.

The rule itself has four steps and was fixed before any rationale data were inspected. The answer tag is stripped and the remaining text is split into sentences. Within the last five sentences, every sentence that contains a conclusion cue is kept, up to three of them and in their original order; the cue list has nine entries: answer, option, choice, correct, therefore, thus, hence, most likely, and best. When no sentence carries a cue, the final sentence serves as the claim. The implementation and its unit tests ship with the code.

Worked example (MedQA, dimeric-immunoglobulin question, candidate 2)

Rationale (final portion of 1,367 characters): “. . . Option C, ‘Inhibiting bacterial adherance and colonization of mucous membranes,’ is the most accurate description of the primary function of dimeric IgA. <answer>C</answer>”

Extracted claim: “Option C, ‘Inhibiting bacterial adherance and colonization of mucous membranes,’ is the most accurate description of the primary function of dimeric IgA.”

The rule is deliberately permissive: when several tail sentences carry cues, all of them survive, including option-elimination lines. This costs precision on individual extractions but keeps the rule simple, auditable, and free of tuned thresholds, and the resulting score separates knowledge-backed from inherited unanimity at 0.587 and 0.613 AUROC on the two primary benchmarks (§4.3).

## D Algorithms

Algorithm 1 gives the probe of §3.2; each line is one call into machinery the substrate already runs, with the prompts of Appendix A. Algorithm 2 gives the calibration of ${ \dot { \ S } } 3 . 3 ;$ following Quach et al. [2024], the statistical content stays in Eqs. (5) and (6), and the box only fixes the order of operations.

Algorithm 1 Counter-evidence probe   
Require: question x with unanimous first-round consensus $m ^ { * } ( a _ { 1 } = 1 ) ;$ panel size N; retriever R   
of the substrate; query budget $k \leq 4$   
Ensure: keep rate $\kappa \doteq \{ \bar { 0 } , { 1 } / { N } , \ldots , 1 \}$   
1: $q _ { 1 } , \dotsc , q _ { j }  ($ COUNTERQUERIES $( x , m ^ { * } ) , j \leq k$ ▷ temperature 0, thinking mode   
2: $D _ { \mathrm { c h } }  \mathcal { R } ( q _ { 1 } , . . . , q _ { j } )$ ▷ same BM25 + rerank stack, $\leq 8$ documents   
3: for $i = 1$ to N do   
4: $\tilde { a } _ { i } \gets \mathsf { R E A }$ NSWER $( x , D _ { \mathrm { c h } } , m ^ { * } )$ ▷ devil’s-advocate instruction; revision licensed, not   
demanded   
5: end for   
6: $\kappa \gets \left| \{ i : \tilde { a } _ { i } = m ^ { * } \} \right| / N$   
7: return κ ▷ low κ: the consensus collapses under challenge

Algorithm 2 Stratified Learn-then-Test calibration   
Require: calibration set $\mathcal { D } _ { \mathrm { c a l } } ;$ stratification g with certified strata $\{ S _ { 1 } , S _ { 2 } \}$ ; per-stratum scores $s _ { S } ;$   
risk level α; error budget $\delta ;$ coverage grid C   
Ensure: thresholds $\hat { \lambda } _ { S }$ , or ABSTAIN for strata certifying nothing   
1: for each certified stratum S do   
2: ${ \mathcal { D } } _ { S } \gets \{ ( x , y ) \in { \mathcal { D } } _ { \mathrm { c a l } } : g ( x ) = S \}$   
3: for $c \in { \bar { \mathcal { C } } }$ in descending order do   
4: $\lambda _ { c } \gets$ score at the top-c quantile of $\mathcal { D } _ { S }$   
5: $n _ { c } \gets | \{ x \in \mathcal { D } _ { S } : s _ { S } ( x ) \} > \lambda _ { c }  \|$ ; E<sub>c</sub> ← errors among them   
6: $p _ { c } \gets \mathrm { P r } \left[ \mathrm { B i n } ( n _ { c } , \alpha ) \leq E _ { c } \right]$ ▷ Eq. (6)   
7: i $: p _ { c } \leq \tilde { \delta \big / } \left( | \mathcal { C } | \cdot | \{ S _ { 1 } , S _ { 2 } \} | \right)$ then   
8: $\hat { \lambda } _ { S } \gets \lambda _ { c } ;$ break ▷ largest certified coverage for S   
9: end if   
10: end for   
11: if no rejection then $\hat { \lambda } _ { S } \gets \mathbf { A B S T A I N }$ ▷ stratum uncertified at level α   
12: end for   
13: return $\{ \hat { \lambda } _ { S } \}$ ▷ answer x $\in S$ iff $s _ { S } ( x ) > \hat { \lambda } _ { S } ; S _ { 3 }$ abstains as a whole

## E Certification Protocol and Proof

Protocol. The coverage grid runs from 0.05 to 1.00 in steps of 0.05, giving twenty candidate thresholds per stratum. The family-wise error budget is $\delta = 0 . 0 5$ , Bonferroni-corrected across the twenty grid points and the two certified strata, and we sweep $\alpha \in \{ 0 . 0 5 , 0 . 1 0 , 0 . 1 5 , 0 . 2 0 \}$ , each level constituting its own guarantee. Calibration and test halves are split evenly at random, and every certification number in the paper is the mean over 50 such splits with fixed seeds: a single split yields a valid certificate, but its reported coverage inherits the variance of one draw, and averaging over repeated splits reports the procedure rather than one particular partition. The per-split violation fraction is the quantity the $1 - \delta$ guarantee bounds, and it is 0 of 50 for every Learn-then-Test arm reported: no split’s realized test risk exceeds its target α. Calibration is per domain throughout; §4.4 reports what happens when domains are pooled.

Why a grid rather than fixed-sequence testing. Selective risk is not monotone in the threshold, so the fixed-sequence variant of Learn-then-Test, which walks thresholds from the most selective downward and stops at the first failure, is brittle here: one wrong answer among the few highestscoring calibration questions terminates the walk at zero certified coverage even when broader coverage levels are certifiable. Testing the whole grid under a Bonferroni correction is robust to the location of individual errors and costs only a constant factor in the effective test level.

Proof of Theorem 1. Fix a certified stratum with calibration pairs $( s _ { i } , y _ { i } ) _ { i \leq n }$ of score and correctness, and fix a grid point c. The threshold $\lambda _ { c }$ is an order statistic of the calibration scores, so the argument conditions on it. Let $A = \left\{ i : s _ { i } > \lambda _ { c } \right\}$ be the calibration questions above the threshold, so that $n _ { c } = | A |$ . The event that $\lambda _ { c }$ equals a value t and that the set above it equals A constrains the pairs in A only through $s _ { i } > t ,$ and places all its remaining conditions on the pairs outside A. Under Assumption 1 the pairs are independent, so conditional on $( \lambda _ { c } , A )$ the pairs in A are i.i.d. draws from the distribution of a question given $s > \lambda _ { c }$ . This is the population that the rule answers at threshold $\lambda _ { c } .$ , and its error rate is $\mathrm { r i s k } ( \lambda _ { c } )$ . Conditional on $( \lambda _ { c } , A )$ , the error count therefore satisfies $E _ { c } \sim \mathrm { B i n } ( n _ { c } , \mathrm { r i s k } ( \lambda _ { c } ) )$ . Whenever risk $\mathbf { \partial } \cdot ( \lambda _ { c } ) > \alpha$ , this count stochastically dominates $\mathrm { B i n } ( n _ { c } , \alpha )$ and the tail probability of Eq. (6) obeys Pr[ $p _ { c } \leq u \mid \lambda _ { c } , A \rfloor \leq \iota$ for every $u \in [ 0 , 1 ]$ Taking the expectation over $( \lambda _ { c } , A )$ gives $\mathrm { P r } [ \mathrm { r i s k } ( \lambda _ { c } ) > \alpha , p _ { c } \leq u ] \leq u _  \mathrm  $ , which makes $p _ { c }$ a valid p-value for $H _ { 0 } ^ { ( c ) } : \mathrm { r i s k } ( \lambda _ { c } ) > \alpha$ at the data-dependent threshold. Algorithm 2 rejects only when $p _ { c } \le \delta / ( | \boldsymbol { c } | \cdot 2 )$ so by the union bound over all grid points and both certified strata, the probability that any true null is rejected is at most δ. On the complementary event, of probability at least $1 - \delta ,$ every rejected null is false; in particular the selected $\hat { \lambda } ,$ which corresponds to a rejected null, satisfies $\mathrm { r i s k } ( \hat { \lambda } ) \leq \alpha$ The union-bound step is the Learn-then-Test argument of Angelopoulos et al. [2022] with the exact binomial tail. The conditioning step extends it to thresholds placed at calibration quantiles, and it is why the test counts the questions strictly above the threshold: the question that defines the threshold is not a draw from the answered population. □

Scope of the guarantee. The guarantee is marginal over the calibration draw and holds per stratum and per domain; it is distribution-free in the sense that no assumption is placed on the score. Trainingfree scores are what make the full calibration split available for threshold selection: no data are spent fitting a scorer.

![](images/f59ff3430d0c7941cb3e19a8fb1ad83a39ad4198dfad5286f5815ee354076ab5.jpg)

![](images/db9a298ce43252c2840dc96da92a8211ffd7f4fd0695d08f3d2412880ca95cee.jpg)  
Figure 4: Realized risk and certified coverage of the stratified rule. (Left) Realized selective risk against the target level: every Learn-then-Test arm stays below the $\mathrm { r i s k } = \alpha$ diagonal at every level it certifies, while naive quantile calibration crosses it. (Right) Certified coverage per backbone on the MedQA unanimous layer; printed values are the $\alpha { = } 0$ .15 coverages, and stars mark the two operating points of §4.4. Lines are means over 50 calibration/test splits; bands are ±1 standard deviation.

## F Full Per-Dataset Results

Table 5 extends the main results to every signal, benchmark, and backbone, reporting both discrimination and selective risk. Unanimous-layer metrics, computed on a different subpopulation, are in §4.3 and Figure 3 rather than here.

The full grid confirms the structure of §4.2 without exception. Terminal agreement is at chance in all seven cells, on the strongest backbone as on the weakest. Sequence log-probability is the only signal that gets worse than chance, and its selective risk degrades sharply on Llama-3.1-8B, where long free-text rationales make raw likelihood an actively misleading confidence proxy. Answerdistribution entropy tracks first-round agreement closely everywhere, as both are functionals of the same round-one vote distribution. The process score has the best discrimination in six of the seven cells and the best selective risk in five, and its margin over the strongest snapshot signal is widest where the backbone is weakest.

![](images/3000ff46775588b8580228a74669d814f1499fe77e1470a611a994a8353ba1b7.jpg)

Figure 5 plots the risk-coverage curves behind the AURC column on the Qwen3-8B substrate. The process score traces the lowest risk over nearly the whole coverage range on all three benchmarks, and the ordering of the other signals is stable: terminal agreement is flat, sequence log-probability is inverted at low coverage, and the two round-one functionals move together.

Table 5: Discrimination (AUROC) and selective risk (AURC) of all signals across benchmarks and backbones. Dashes mark cells outside the evaluation design: the cross-backbone replication covers the two primary benchmarks.
<table><tr><td rowspan="2">Signal</td><td colspan="3">AUROC ↑</td><td colspan="3">AURC↓</td></tr><tr><td>MedQA</td><td>MMLU-Pro</td><td>MedMCQA</td><td>MedQA</td><td>MMLU-Pro</td><td>MedMCQA</td></tr><tr><td colspan="7">Qwen3-8B</td></tr><tr><td>Terminal agreement</td><td>0.512</td><td>0.504</td><td>0.507</td><td>0.221</td><td>0.300</td><td>0.330</td></tr><tr><td>First-round agreement [Wang et al., 2023]</td><td>0.677</td><td>0.698</td><td>0.640</td><td>0.155</td><td>0.200</td><td>0.250</td></tr><tr><td>Sequence log-probability</td><td>0.400</td><td>0.474</td><td>0.534</td><td>0.320</td><td>0.376</td><td>0.347</td></tr><tr><td>Answer entropy [Kuhn et al., 2023]</td><td>0.681</td><td>0.703</td><td>0.641</td><td>0.154</td><td>0.199</td><td>0.250</td></tr><tr><td>Verbalized confidence [Tian et al., 2023]</td><td>0.585</td><td>0.651</td><td></td><td>0.179</td><td>0.207</td><td></td></tr><tr><td>P(True) [Kadavath et al., 2022]</td><td>0.602</td><td>0.644</td><td></td><td>0.160</td><td>0.205</td><td></td></tr><tr><td>ProbeGuard (process score)</td><td>0.696</td><td>0.704</td><td>0.659</td><td>0.145</td><td>0.205</td><td>0.234</td></tr><tr><td colspan="7">HuatuoGPT-o1-8B</td></tr><tr><td>Terminal agreement</td><td>0.519</td><td>0.526</td><td></td><td>0.193</td><td>0.353</td><td></td></tr><tr><td>First-round agreement [Wang et al., 2023]</td><td>0.766</td><td>0.701</td><td></td><td>0.093</td><td>0.232</td><td></td></tr><tr><td>Sequence log-probability</td><td>0.350</td><td>0.422</td><td></td><td>0.265</td><td>0.399</td><td></td></tr><tr><td>Answer entropy [Kuhn et al., 2023]</td><td>0.768</td><td>0.707</td><td></td><td>0.093</td><td>0.230</td><td></td></tr><tr><td>ProbeGuard (process score)</td><td>0.775</td><td>0.701</td><td></td><td>0.096</td><td>0.225</td><td></td></tr><tr><td colspan="7">Llama-3.1-8B</td></tr><tr><td>Terminal agreement</td><td>0.548</td><td>0.551</td><td></td><td>0.289</td><td>0.406</td><td></td></tr><tr><td>First-round agreement [Wang et al., 2023]</td><td>0.781</td><td>0.736</td><td></td><td>0.159</td><td>0.254</td><td></td></tr><tr><td>Sequence log-probability</td><td>0.205</td><td>0.217</td><td></td><td>0.547</td><td>0.640</td><td></td></tr><tr><td>Answer entropy [Kuhn et al., 2023]</td><td>0.790</td><td>0.744</td><td></td><td>0.156</td><td>0.252</td><td></td></tr><tr><td>ProbeGuard (process score)</td><td>0.825</td><td>0.782</td><td></td><td>0.142</td><td>0.246</td><td></td></tr></table>

Figure 5: Risk-coverage curves of the always-available signals on the Qwen3-8B substrate; lower is better, and the AURC values of Table 2 are the areas under these curves. The process score uses the same five-fold out-of-fold protocol as the tables; coverage below 5% is omitted.

## G Probe Control Arms

Table 6 isolates what drives the flip asymmetry of §4.3. Each control arm re-asks the same 746 unanimous MedQA questions with the same N=8 re-answers, varying one factor at a time: the instruction-only arm keeps the devil’s-advocate framing and removes all documents; the irrelevantdocuments arm keeps the framing and substitutes documents retrieved for other questions’ counterqueries; the neutral-instruction arm keeps this question’s counter-evidence documents and removes the adversarial framing.

The instruction alone flips no majority at all, and documents without evidential content flip 1.0% of wrong consensus. The counter-evidence under a neutral instruction retains nearly the full asymmetry (12.0% versus 3.6%), and the paired difference in asymmetry between the probe and each content-free arm excludes zero (bootstrap 95% intervals [3.0, 16.8] and [1.7, 16.8] percentage points). The flip responds to the content of the counter-evidence, and the probe inherits its discrimination from that content.

Table 6: Flip rates of the counter-evidence probe and its control arms on the MedQA unanimous layer (Qwen3-8B, N=8 re-answers per question).
<table><tr><td>Arm</td><td>Instruction</td><td>Documents</td><td>flip(wrong)</td><td>flip(right)</td></tr><tr><td>Counter-evidence probe</td><td>devil&#x27;s advocate</td><td>counter-evidence</td><td>14.0%</td><td>4.2%</td></tr><tr><td>Instruction only</td><td>devil&#x27;s advocate</td><td>none</td><td>0.0%</td><td>0.0%</td></tr><tr><td>Irrelevant documents</td><td>devil&#x27;s advocate</td><td>unrelated</td><td>1.0%</td><td>0.2%</td></tr><tr><td>Neutral instruction</td><td>neutral</td><td>counter-evidence</td><td>12.0%</td><td>3.6%</td></tr></table>

## H Ablation Matrices

Table 7 extends the drop-one ablation of §5.1 to all three benchmarks, removing one feature family at a time from the full process score, and adds the classifier swap.

Table 7: Drop-one family ablation and classifier swap (AUROC, Qwen3-8B substrate). Each row removes the feature family marked ✗ from the full process score (top row).
<table><tr><td colspan="5">Feature family</td><td colspan="3">AUROC</td></tr><tr><td>Agreement</td><td>Distribution</td><td>Minority</td><td>Rationale</td><td>Retrieval</td><td>MedQA</td><td>MMLU-Pro</td><td>MedMCQA</td></tr><tr><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>0.696</td><td>0.704</td><td>0.659</td></tr><tr><td>x</td><td>√</td><td>√</td><td>√</td><td>√</td><td>0.703</td><td>0.705</td><td>0.660</td></tr><tr><td>√</td><td>x</td><td>√</td><td>√</td><td>√</td><td>0.703</td><td>0.714</td><td>0.657</td></tr><tr><td>√</td><td>√</td><td>x</td><td>√</td><td>√</td><td>0.697</td><td>0.705</td><td>0.658</td></tr><tr><td>√</td><td>√</td><td>√</td><td>x</td><td>√</td><td>0.682</td><td>0.674</td><td>0.645</td></tr><tr><td>√</td><td>√</td><td>√</td><td>√</td><td>x</td><td>0.700</td><td>0.697</td><td>0.658</td></tr><tr><td colspan="4">GBT in place of logistic regression</td><td></td><td>0.644</td><td>0.643</td><td>0.619</td></tr></table>

The pattern of §5.1 holds on every benchmark, not only MedQA: removing the rationale family costs discrimination in all three columns, while removing any other family leaves the score unchanged or marginally improved. The trajectory families are informative individually but interchangeable collectively, each a different projection of the same consensus dynamics; the rationales are the one signal source the rest cannot substitute for. The classifier swap points the same direction on all benchmarks as well, so the choice of a default logistic regression is not doing hidden work. The hard-versus-soft aggregation comparison for the rationale score is in §5.1, and the pilot-question sensitivity check in §5.2.

## I Specialty-Level Certification

Specialty labels are available for 899 MedQA test questions, of which 529 fall in the unanimous layer. Table 8 breaks the layer-level certificate of §4.4 down by specialty, using the same score, level, and repeated-split protocol; specialties with fewer than 25 unanimous questions are omitted.

Table 8: Per-specialty breakdown of the certified rule on the MedQA unanimous layer (α=0.15, 50 splits). The certificate itself is calibrated at the layer level; this table reports how its coverage and realized risk distribute across specialties.
<table><tr><td>Specialty</td><td>n</td><td>Unanimous-layer error (%)</td><td>Certified coverage</td><td>Realized risk (%)</td></tr><tr><td>Cardiology</td><td>42</td><td>14.3</td><td>0.70</td><td>9.0</td></tr><tr><td>Hematology</td><td>29</td><td>17.2</td><td>0.69</td><td>13.3</td></tr><tr><td>Obstetrics and gynecology</td><td>51</td><td>11.8</td><td>0.66</td><td>14.5</td></tr><tr><td>Gastroenterology</td><td>57</td><td>15.8</td><td>0.60</td><td>11.1</td></tr><tr><td>Pediatrics</td><td>35</td><td>11.4</td><td>0.60</td><td>8.9</td></tr><tr><td>Psychiatry</td><td>33</td><td>12.1</td><td>0.59</td><td>7.3</td></tr><tr><td>Nephrology</td><td>30</td><td>13.3</td><td>0.59</td><td>8.4</td></tr><tr><td>Emergency medicine</td><td>42</td><td>26.2</td><td>0.57</td><td>19.1</td></tr><tr><td>Infectious diseases</td><td>36</td><td>13.9</td><td>0.57</td><td>10.0</td></tr><tr><td>Endocrinology</td><td>51</td><td>11.8</td><td>0.52</td><td>4.0</td></tr><tr><td>Neurology</td><td>37</td><td>10.8</td><td>0.49</td><td>7.7</td></tr></table>

Emergency medicine is the outlier on both axes: its unanimous consensus is wrong about twice as often as the layer average, and it is the one specialty whose realized risk under the layer-level threshold exceeds the target. This is consistent with clinical experience, as acute multi-system presentations are where a shared quick prior is most likely to mislead all eight candidates at once. Coverage does not simply track the error rate: endocrinology and neurology have among the lowest unanimous-layer error rates yet receive the most conservative coverage, while obstetrics and cardiology combine similar error rates with far wider certified regions, meaning the process signals are differently informative across specialties, not merely calibrated to difficulty. The spread matters for deployment: a layer-level certificate is honest in aggregate, and a site that needs per-specialty control, routing emergencymedicine questions more conservatively than cardiology ones, can re-run the same calibration within specialty strata once enough per-specialty calibration data accumulate.

## J Cost Accounting

The passive layers are computed from logs the system already writes, at no inference cost. The probe fires only on unanimous first-round votes (59% of MedQA, 59% of MMLU-Pro medicine, 55% of MedMCQA) and costs one counter-query call, one retrieval round, and N concurrent re-answers: a median of 114 s and 4381 completion tokens per probed MedQA question, against the 72 s the substrate already spends answering, so probing roughly doubles the cost of a unanimous question and adds nothing to a contested one. Rationale scoring is a single batched NLI pass, minutes per dataset on one GPU. All experiments ran on one node with six A40 $\mathrm { G P U s ; }$ no API models were used.

## K Qualitative Cases

The two cases of Figure 1, with their probe trajectories as logged. Both end in a unanimous panel;   
what differs is which answer survives.

Case 1: unanimously wrong, repaired (κ = 0). A patient hospitalized for a week after a stroke develops fever, purulent cough, and a new consolidation; all eight candidates answer Streptococcus pneumoniae, each rationale invoking the same community-acquired prior and none engaging the hospitalization timeline. The reasoning is not careless: one rationale methodically rules out tuberculosis and Haemophilus influenzae before concluding “Given the clinical presentation, the most likely causal microorganism is Streptococcus pneumoniae”; each step is sound, and the error is in what the chain never considers. The probe’s counter-query generator re-reads the stem and targets the discarded cue, generating “Hospital-acquired pneumonia in stroke patients: common pathogens” and “Role ofStaphylococcus aureus in hospital-acquired pneumonia with consolidation” among its four queries. With the eight retrieved challenge documents in context, all eight re-answers abandon the consensus and converge, again unanimously, on the gold answer Staphylococcus aureus. The probe did not surface unusual evidence; it directed attention back to context the candidates had ignored.

Case 2: unanimously correct, held $( \kappa = 1 )$ . On the dimeric-immunoglobulin question, all eight candidates answer that secretory IgA inhibits bacterial adherence to mucous membranes, and the eight rationales cite the same mechanism. The probe generates genuine counter-queries, one asking directly for “Evidence against IgA’s role in inhibiting bacterial adhesion”, and returns eight challenge documents. All eight re-answers weigh the challenge and retain the consensus.

These two trajectories are the endpoints of the keep rate; intermediate values, a consensus that partially fragments under challenge, populate the separation of Figure 3. The complete probe records, including all generated queries, retrieved document identifiers, and per-candidate re-answer votes, are in the released logs.

## L Implementation Details

The substrate runs unmodified at a pinned commit, and every component downstream of it is deterministic given the seed. Entailment is scored by DeBERTa-large fine-tuned on MNLI [He et al., 2021]; reranking uses the MedCPT cross-encoder [Jin et al., 2023]; both are used off the shelf without fine-tuning. Table 9 lists the hyperparameters of each stage. None of them were tuned on evaluation data: the generation and retrieval settings are the substrate’s own defaults, the classifier is a scikit-learn default configuration, and the certification constants of Appendix E were fixed before calibration data were collected. A single seed governs generation, data splits, cross-validation, and bootstrap resampling; exact software versions and the per-dataset settings manifest ship with the code.

Table 9: Hyperparameters by pipeline stage.
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Consensus generation</td><td></td></tr><tr><td>Panel size N</td><td>8</td></tr><tr><td>Maximum rounds T</td><td>8</td></tr><tr><td>Candidate temperature</td><td>0.7</td></tr><tr><td>Candidate max new tokens</td><td>2048</td></tr><tr><td>Query temperature (thinking mode)</td><td>0</td></tr><tr><td>Retrieval</td><td></td></tr><tr><td>BM25 (k1, b)</td><td>(0.9, 0.4)</td></tr><tr><td>Candidates per query</td><td>32</td></tr><tr><td>Kept after reranking</td><td>2</td></tr><tr><td>Documents per round</td><td>≤ 8</td></tr><tr><td>Rationale scoring and process score</td><td></td></tr><tr><td>NLI input limit (tokens)</td><td>256</td></tr><tr><td>Classifier</td><td>l2 logistic regression</td></tr><tr><td>Cross-validation folds</td><td>5</td></tr><tr><td>Random seed (all stages)</td><td>42</td></tr></table>