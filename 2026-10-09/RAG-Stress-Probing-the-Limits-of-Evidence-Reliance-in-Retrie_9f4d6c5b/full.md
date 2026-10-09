# RAG-Stress: Probing the Limits of Evidence Reliance in Retrieval-Augmented Generation

Shunyuan Zhou<sup>1∗</sup>, Hao Chen<sup>2,4∗</sup>, Tianyu Wang<sup>3</sup>, Goose Lin<sup>2</sup>, Zaiyuan Wang<sup>2</sup>, Haiying Zhao<sup>1†</sup>

<sup>1</sup>Beijing University of Posts and Telecommunications <sup>2</sup>Humanlaya Data <sup>3</sup>North China University of Technology <sup>4</sup>Beijing Key Laboratory of Key Technologies for AI+ Domain Applications

## Abstract

Following retrieved evidence does not guarantee factual correctness: misleading evidence can induce a model to replace an answer it previously gave correctly. Standard accuracy measures obscure this behavior by combining answer replacement with preexisting errors. We introduce RAG-Stress, a controlled diagnostic protocol for examining the limits of evidence reliance in retrieval-augmented generation. The protocol holds the question and reference answer fixed, edits one assertion to support a designated incorrect answer, and crosses two source priority policies with three positions of the answer span within the evidence text. We measure misleading rate (MR) on each model’s subset of questions answered correctly without retrieval, alongside clean accuracy on the full evaluation set. We evaluate fifteen systems spanning API models, open models, and search agents trained with reinforcement learning on TriviaQA-RC, HotpotQA, and SearchQA, with additional English and Chinese MedQA evaluations. Instructions that prioritize documents consistently produce higher MR than those permitting reliance on prior knowledge. Averaged over models and positions, the gap ranges from 10.9 to 13.5 percentage points across the three QA datasets. Mean MR follows End > Beginning > Middle under both policies, although individual models do not uniformly follow this ordering. A separate paired audit of 500 questions and two checkpoints supports increased harmful override without establishing a corresponding improvement in beneficial correction. These findings distinguish evidence adherence from factual reliability and motivate evaluating whether retrieved evidence preserves, replaces, or corrects a model’s answers.

## 1 Introduction

Retrieval-augmented generation (RAG) grounds large language models in external evidence (Lewis et al., 2020; Chen et al., 2025; Su et al., 2026; Chen et al., 2026). Most evaluations ask whether retrieval improves answer accuracy or whether an answer is supported by the retrieved text. These criteria are insufficient when the evidence is relevant and internally coherent but supports an incompatible answer: a model can be well grounded and still be wrong. We therefore ask: when a model answers correctly without retrieval, which evidence and instruction conditions make it adopt a designated counterfactual answer?

Two distinctions are essential: whether the model answered correctly before receiving evidence, and whether its subsequent error adopts the specific answer supported by the edit. Aggregate accuracy collapses both distinctions. RAG-Stress fixes the question and reference, edits one assertion containing the answer to support a foil of the same type, and conditions foil adoption on closed-book correctness. Responses to clean evidence provide a separate control; reference recovery on baseline failures is analyzed separately. This measures a behavioral transition, not stable parametric knowledge or, by itself, the causal effect of editing.

The intervention crosses Soft and Strict instructions with beginning, middle, and end Word Position conditions within the target text. The question is not simply whether asking models to trust documents increases compliance, but how often compliance replaces a correct baseline answer with the designated foil and how this rate varies with placement.

![](images/ae120bf20d854de0ed625fab293afc4746809e7ae700bd7de59292e0bcc99a34.jpg)  
Figure 1: One saved trajectory: the same edit, two instructions, different answers. With clean evidence, both soft and strict prompting return the reference (Tomb Raider); with the edited evidence, soft prompting keeps the reference while strict prompting adopts the foil (Uncharted). The closedbook answer is also Tomb Raider.

We evaluate fifteen API models, open models, and search agents trained with RL across three QA datasets, with additional medical QA experiments in English and Chinese. Strict produces higher misleading rates than Soft across systems and datasets. Averaged equally across models, these rates follow End > Beginning > Middle under both instructions on all three datasets. A separate paired audit of 500 items retains 13,000 generations from two quantized 8B checkpoints (Appendix B). Relative to Soft, Strict increases foil adoption on the baseline-correct subsets by 14.0 and 9.7 percentage points; intervals for reference recovery include zero. These findings indicate sensitivity to instructions and position, rather than changes to knowledge in model parameters.

Our contributions are: (1) a protocol that measures designated foil adoption conditional on baseline correctness, separating answer replacement from preexisting failure; (2) an evaluation crossing instruction and Word Position across systems, tasks, and languages; and (3) a paired audit of saved outputs separating harmful override, reference recovery, and transitions from clean to edited evidence, with uncertainty estimated over questions. The contribution is diagnostic resolution, not a new conflict phenomenon or mitigation method.

## 2 Related Work

Knowledge conflict and context reliance. Prior work studies reliance on external evidence and conflicts with prior answers (Mallen et al., 2023; Longpre et al., 2021; Monea et al., 2024; Xie et al., 2024; Farahani & Johansson, 2024). Entity substitutions, Fakepedia, and conflicts among sources already establish sensitivity to counterfactual evidence and source instructions (He et al., 2025; Zeng et al., 2026; He et al., 2026b;a). Context-aware decoding changes contextual reliance (Shi et al., 2024); position effects in long contexts (Liu et al., 2024; He et al., 2026a) motivate, but do not directly test, Word Position within a document. Astute RAG, FaithfulRAG, and CARE address reliability or conflicts between facts (Wang et al., 2025; Zhang et al., 2025; Choi et al., 2025). RAG-Stress combines a crossed intervention with a fixed baseline-correct stratum and measures adoption of a designated foil. Instruction effects on conflict resolution are not themselves new.

RAG evaluation and robustness. RAG research spans retrieval, generation, and evaluation (Gao et al., 2023); Self-RAG integrates retrieval and self-reflection (Asai et al., 2024). RAGAs evaluates retrieval relevance, faithfulness, and answer quality (Es et al., 2024); RAGTruth annotates hallucinated claims (Niu et al., 2024). RGB and RECALL test counterfactual robustness (Chen et al., 2024a; Liu et al., 2023). GaRAGe, mtRAG, CRUX, and MAGIC extend evaluation to grounding, conversation, context coverage, and multi-hop conflict (Sorodoc et al., 2025; Katsis et al., 2025; Ju et al., 2025; Lee et al., 2025). Our complementary measurement asks whether a baseline-correct response changes to a designated foil. Neither aggregate accuracy nor grounding alone identifies this transition.

Adversarial evidence and memory. CF-RAG evaluates counterfactual queries (Qin et al., 2026), while AgentPoison, MINJA, and MemoryGraft study persistent memory or retrieval attacks (Chen et al., 2024b; Dong et al., 2025; Srivastava & He, 2025). RAG-Stress keeps the query, index, and memory fixed and measures the immediate response to one edited assertion.

## 3 RAG-Stress Methodology

RAG-Stress crosses evidence condition, instruction, and Word Position (Figure 2, left). Closedbook responses define model-specific strata: clean accuracy uses the full prepared set; MR uses its baseline-correct subset.

![](images/0d23822764d2c82a6626054533c257fe44fb97a2a8211e11c0a5dab4e3d2a48d.jpg)  
Figure 2: RAG-Stress overview. Left: paired evidence construction and the instruction–Word Position design. Right: original- and edited-evidence results for five API models on TriviaQA-RC, organized into six instruction–position panels.

## 3.1 Same-Query Evidence Intervention

Each example contains a question q, reference answer $a _ { q } ^ { \star } .$ , and a clean target passage that supports the reference. We construct an edited version of that passage that supports a designated foil $a _ { q } ^ { - }$ . For open-domain QA, the foil replaces the span containing the answer with an entity of the same type; for multiple-choice QA, it is one of the provided distractor options. Within each Word Position condition, the clean/edited pair differs only in the assertion containing the answer (Appendix A).

We obtain a closed-book baseline without documents. Each evidence arm crosses Soft/Strict with three Word Position conditions: the answer span occurs near the beginning, middle, or end of the target evidence text, with document order fixed. This defines twelve cells per question. Clean accuracy is measured across all prepared questions; MR is scored on baseline-correct questions. The separate paired audit retains its recorded document-slot intervention; it supports instruction contrasts, not a replication of Word Position effects.

## 3.2 Baseline Strata and Evaluation Metrics

Let Q contain all prepared evaluation questions before baseline filtering. For model M, closed-book responses define the baseline-correct set $K _ { M }$ and baseline-failure set $U _ { M } \mathbf { : }$

$$
K _ { M } = \{ q \in Q : \mathrm { M a t c h } ( M ( q , \emptyset ) , a _ { q } ^ { \star } ) = 1 \} , \qquad U _ { M } = Q \setminus K _ { M } .\tag{1}
$$

This fixed split excludes preexisting failures from MR. It is not recomputed after an intervention. Accuracy with clean evidence uses all of Q:

$$
\operatorname { A c c } _ { \pi , p } ^ { C } = \frac { 1 } { | Q | } \sum _ { q \in Q } \mathbf { 1 } [ \operatorname { M a t c h } ( \hat { a } _ { q , \pi , p } ^ { C } , a _ { q } ^ { \star } ) ] .\tag{2}
$$

Here C denotes unedited evidence. Thus clean accuracy may vary with instruction π and position p;   
it is not closed-book accuracy or accuracy on $K _ { M }$ alone.

Let $\hat { a } _ { q , \pi , p } ^ { E }$ be the response to edited evidence under instruction π and position p. The main metric, misleading rate (MR), is the fraction of baseline-correct questions for which the response matches the foil but not the reference:

$$
\mathrm { M R } _ { \pi , p } = \frac { 1 } { \vert K _ { M } \vert } \sum _ { q \in K _ { M } } \mathbf { 1 } [ \mathrm { M a t c h } ( \hat { a } _ { q , \pi , p } ^ { E } , a _ { q } ^ { - } ) \wedge \neg \mathrm { M a t c h } ( \hat { a } _ { q , \pi , p } ^ { E } , a _ { q } ^ { \star } ) ] .\tag{3}
$$

All rates are reported as percentages; double matches do not count as MR. MR is not the clean-toedited accuracy drop: $K _ { M }$ need not be clean-evidence-correct, and foil adoption is only one error type. Direct transitions require paired clean/edited outputs (Appendix C.3). Audit HOR averages MR over positions; BCR measures reference recovery with clean evidence on $U _ { M }$ . These conditional rates remain separate from accuracy on the full set and from each other.

## 3.3 Paired Contrasts and Uncertainty

Instruction contrasts subtract Soft MR from Strict MR after averaging positions within question. Position contrasts pair the same questions and instruction. The supplementary audit uses 20,000 bootstrap resamples of questions for pointwise 95% percentile intervals, retaining all repeated cells together. These intervals are not multiplicity-adjusted; primary reporting does not imply preregistration. The QA position check uses 2,000 resamples. Neither set of intervals applies to the rate tables for fifteen systems (Algorithm 1; Appendices C.2 and F).

## 4 Experiments

## 4.1 Datasets and Experimental Setup

We evaluate TriviaQA-RC (Joshi et al., 2017), HotpotQA (Yang et al., 2018), and SearchQA (Dunn et al., 2017). The model set contains seven API systems, four open-source instruction models (Grattafiori et al., 2024; Jiang et al., 2023; Yang et al., 2025), and four RL-trained search agents (Jin et al., 2025; Sun et al., 2026; Chu et al., 2026; Xia et al., 2026). Live search is disabled; evidence is fixed rather than retrieved by each system. Clean responses are evaluated on the full prepared set under all six instruction–position conditions. Closed-book filtering defines each model’s eligible subset for edited-evidence MR, not the denominator of clean accuracy.

Closed-book prompting requests a short answer without explanation. Strict prompting makes the documents the primary source of truth even when they appear mistaken; Soft prompting permits prior knowledge to override conflicting documents. Thus the prompt contrast intentionally changes the conflict-resolution policy rather than paraphrasing the same instruction (Appendix A.1).

## 4.2 Baselines and Evaluation

The primary outcome is model-conditional MR, with clean accuracy, conflict follow, gold retention, and hedge rate as diagnostics. Responses are normalized by case, punctuation, articles, and answer prefixes; gold/foil double matches do not count as MR. Main runs use greedy decoding and a shortanswer generation cap. The contrasts in the rate tables for fifteen systems are descriptive. Paired intervals are reported separately for the 300-question QA checks and the 500-item audit. Cross-dataset and model-family comparisons remain descriptive because their retained question sets differ.

Table 1: TriviaQA-RC. Beginning/Middle/End: Word Position within the target evidence text. Strict/Soft cells: bold full-set clean accuracy; ↓ MR on closed-book-correct items (%). Each ∆ cell: bold clean-accuracy contrast and ↓ MR contrast, both Strict minus Soft (pp); Mean averages positions. The arrow identifies the MR component, not an accuracy drop or a sign.
<table><tr><td>TriviaQA-RC</td><td colspan="3">Beginning</td><td colspan="3">Middle</td><td colspan="3">End</td><td>Mean</td></tr><tr><td>Model</td><td>Strict</td><td>Soft</td><td> $\pmb { \Delta } _ { \mathbf { i n s t r } }$ </td><td>Strict</td><td>Soft</td><td> $\pmb { \Delta } _ { \mathbf { i n s t r } }$ </td><td>Strict</td><td>Soft</td><td> $\pmb { \Delta } _ { \mathbf { i n s t r } }$ </td><td> $\overline { { \Delta } } _ { \mathrm { i n s t r } }$ </td></tr><tr><td colspan="9">API-based</td><td></td></tr><tr><td>GPT-5.6</td><td> ${ \bf 9 1 . 1 _ { \perp 2 0 . 4 } }$ </td><td> ${ \bf 9 5 . 7 _ { \perp 7 . 9 } }$ </td><td> $\mathbf { - 4 . 6 } _ { \downarrow 1 2 . 5 }$ </td><td> $\mathbf { 9 0 . 6 } _ { \perp 2 0 . 7 }$ </td><td> ${ \bf 9 3 . 2 } _ { \downarrow 9 . 3 }$ </td><td> $\mathbf { - } 2 . 6 _ { \downarrow 1 1 . 4 }$ </td><td> ${ \bf 9 1 . 4 _ { \perp 1 9 . 9 } }$ </td><td> $\mathbf { 9 1 . 8 _ { \perp 7 . 8 } }$ </td><td> $\mathbf { - 0 . 4 } _ { \downarrow 1 2 . 1 }$ </td><td> ${ \bf - } 2 . 5 _ { \perp 1 2 . 0 }$ </td></tr><tr><td>Claude Opus 4.6</td><td> $\pm 2 . 7 _ { \perp 1 6 . 1 }$ </td><td> $\pmb { 9 2 . 4 } _ { \downarrow , 6 . 6 }$ </td><td> $\mathbf { 0 . 3 _ { \perp 9 . 5 } }$ </td><td> $\mathbf { 9 3 . 2 } _ { \downarrow 1 2 . 6 }$ </td><td> $\mathbf { 9 2 . 6 _ { \perp 2 . 8 } }$ </td><td> $\mathbf { 0 . 6 } _ { \downarrow . 9 . 8 }$ </td><td> ${ \bf 9 2 . 9 _ { \perp 1 9 . 1 } }$ </td><td> $\mathbf { 9 5 . 6 _ { \perp 8 . 4 } }$ </td><td> ${ \bf - 2 . 7 _ { \perp 1 0 . 7 } }$ </td><td> $\mathbf { - 0 . 6 } _ { \downarrow , 1 0 . 0 }$ </td></tr><tr><td>Qwen3.8-Max</td><td> $\mathbf { 8 4 . 5 _ { \perp 1 5 . 7 } }$ </td><td> ${ \bf 8 9 . 7 _ { \perp 7 . 9 } }$ </td><td> ${ \bf - 5 . 2 _ { \perp 7 . 8 } }$ </td><td> $\mathbf { 8 7 . 9 } _ { \downarrow 1 7 . 5 }$ </td><td> ${ \bf 9 1 . 3 _ { \perp 8 . 5 } }$ </td><td> ${ \cdot } 3 . 4 _ { \downarrow , 9 . 0 }$ </td><td> ${ \bf 9 0 . 1 } _ { \perp 2 6 . 6 }$ </td><td>88.5  $V ^ { 1 2 . 9 }$ </td><td> $\mathbf { 1 . 6 } _ { \downarrow 1 3 . 7 }$ </td><td> ${ \bf - } 2 . 3 _ { \perp 1 0 . 2 }$ </td></tr><tr><td>DeepSeek-V4-Pro</td><td> ${ \bf 9 3 . 0 _ { \perp 1 8 . 9 } }$ </td><td> ${ \bf 8 7 . 7 _ { \perp 7 . 2 } }$ </td><td> ${ \pmb 5 . 3 } _ { \downarrow 1 1 . 7 }$ </td><td> ${ \bf 8 7 . 2 _ { \perp 2 0 . 5 } }$ </td><td> ${ \bf 8 8 . 2 _ { \perp 7 . 7 } }$ </td><td> $\mathbf { - 1 . 0 } _ { \downarrow 1 2 . 8 }$ </td><td> $\mathbf { 8 9 . 9 } _ { \downarrow 2 8 . 2 }$ </td><td>90.9  $\downarrow 1 6 . 3$ </td><td> $\mathbf { - 1 . 0 } _ { \downarrow 1 1 . 9 }$ </td><td> ${ \bf 1 . 1 } _ { \downarrow 1 2 . 1 }$ </td></tr><tr><td>Z GLM-5.2</td><td> $\mathbf { 8 9 . 0 } _ { \downarrow 2 2 . 3 }$ </td><td> ${ \bf 8 7 . 6 _ { \perp 1 0 . 6 } }$ </td><td> ${ \bf 1 . 4 } _ { \downarrow 1 1 . 7 }$ </td><td> $\mathbf { 8 9 . 8 _ { \perp 2 1 . 2 } }$ </td><td> ${ \bf 9 1 . 0 _ { \perp 1 0 . 3 } }$ </td><td> $\mathbf { - 1 . 2 _ { \downarrow \downarrow 0 . 9 } }$ </td><td> ${ \mathbf { 8 8 . 2 } } _ { \perp 2 4 . 4 }$ </td><td> ${ \bf 8 8 . 9 _ { \perp 1 8 . 8 } }$ </td><td> $\mathbf { - 0 . 7 } _ { \downarrow 5 . 6 }$ </td><td> $\mathbf { - 0 . 2 _ { \perp 9 . 4 } }$ </td></tr><tr><td>K Kimi-K3</td><td> ${ \bf 9 1 . 0 _ { \perp 1 8 . 9 } }$ </td><td> $\mathbf { 8 7 . 9 } _ { \downarrow 5 . 9 }$ </td><td> ${ \bf 3 . 1 } _ { \downarrow 1 3 . 0 }$ </td><td> $\mathbf { 8 9 . 3 _ { \perp 2 2 . 0 } }$ </td><td> ${ \bf 8 9 . 3 _ { \perp 7 . 1 } }$ </td><td> $\mathbf { 0 . 0 } _ { \downarrow \downarrow 4 . 9 }$ </td><td> $\mathbf { 9 8 . 0 _ { \perp 2 6 . 1 } }$ </td><td> ${ \bf 9 1 . 0 _ { \perp 9 . 5 } }$ </td><td> $\mathbf { 7 . 0 } _ { \downarrow 1 6 . 6 }$ </td><td> ${ \mathbf 3 . 4 } _ { \downarrow 1 4 . 8 }$ </td></tr><tr><td>HY4-Preview</td><td> ${ \bf 8 9 . 2 } _ { \perp 3 0 . 1 }$ </td><td> ${ \bf 8 2 . 1 _ { \perp 1 4 . 1 } }$ </td><td> $7 . 1 _ { \downarrow 1 6 . 0 }$ </td><td> $8 5 . 7 _ { \downarrow 2 6 . 8 }$ </td><td> ${ \bf 8 9 . 5 _ { \perp 1 0 . 3 } }$ </td><td> $\mathbf { - 3 . 8 } _ { \downarrow 1 6 . 5 }$ </td><td> ${ \bf 8 9 . 7 _ { \perp 3 2 . 5 } }$ </td><td> ${ \bf 9 1 . 3 _ { \perp 1 4 . 9 } }$ </td><td> $\mathbf { - 1 . 6 } _ { \downarrow 1 7 . 6 }$ </td><td> $\mathbf { 0 . 6 } _ { \downarrow 1 6 . 7 }$ </td></tr><tr><td>Average</td><td> ${ \bf 9 0 . 1 _ { \perp 2 0 . 3 } }$ </td><td> ${ \bf 8 9 . 0 } _ { \perp 8 . 6 }$ </td><td> ${ \bf 1 . 1 _ { \perp 1 1 . 7 } }$ </td><td> ${ \bf 8 9 . 1 _ { \perp 2 0 . 2 } }$ </td><td> $\mathbf { 9 0 . 7 . } _ { \downarrow 8 . 0 }$ </td><td> $\mathbf { - 1 . 6 } _ { \downarrow 1 2 . 2 }$ </td><td> $\mathbf { 9 1 . 5 _ { \perp 2 5 . 3 } }$ </td><td> ${ \bf 9 1 . 1 _ { \perp 1 2 . 7 } }$ </td><td> $\mathbf { 0 . 3 _ { \perp 1 2 . 6 } }$ </td><td> $\mathbf { - 0 . 1 } _ { \downarrow 1 2 . 2 }$ </td></tr><tr><td colspan="9">Open-source</td><td></td></tr><tr><td>∞ Llama-3-8B</td><td> $7 5 . 9 _ { \perp 3 5 . 0 }$ </td><td> $7 8 . 2 _ { \perp 2 3 . 2 }$ </td><td> $\mathbf { - 2 . 3 _ { \downarrow 1 1 . 8 } }$ </td><td> $7 9 . 4 _ { \scriptstyle \downarrow , 3 5 . 4 }$ </td><td> $7 6 . 3 _ { \perp 2 4 . 3 }$ </td><td> ${ \bf 3 . 1 } _ { \downarrow 1 1 . 1 }$ </td><td> ${ 7 8 . 7 } _ { \downarrow 4 4 . 5 }$ </td><td> $7 6 . 0 _ { \perp 3 0 . 6 }$ </td><td> $2 . 7 _ { \downarrow 1 3 . 9 }$ </td><td> ${ \bf 1 . 2 } _ { \downarrow 1 2 . 3 }$ </td></tr><tr><td>M Mistral-7B</td><td> $7 7 . 9 _ { \perp 4 1 . 4 }$ </td><td> $7 5 . 7 _ { \perp 2 4 . 3 }$ </td><td> $2 . 2 _ { \downarrow 1 7 . 1 }$ </td><td> $7 4 . 8 _ { \perp 3 6 . 0 }$ </td><td> $7 4 . 7 _ { \perp 2 3 . 6 }$ </td><td> ${ \bf 0 . 1 } _ { \downarrow 1 2 . 4 }$ </td><td> $7 8 . 4 _ { \perp 4 8 . 8 }$ </td><td> $7 6 . 2 _ { \perp 3 1 . 2 }$ </td><td> $\pmb { 2 . 2 } _ { \downarrow 1 7 . 6 }$ </td><td> ${ \bf 1 . 5 . } _ { \downarrow 1 5 . 7 }$ </td></tr><tr><td>Qwen3-8B</td><td> ${ \bf 7 9 . 1 _ { 4 3 2 . 9 } }$ </td><td> $7 7 . 9 _ { \perp 2 3 . 3 }$ </td><td> ${ \bf 1 . 2 } _ { \downarrow 9 . 6 }$ </td><td> ${ \bf 8 0 . 0 } _ { \perp 3 1 . 9 }$ </td><td> ${ \bf 8 4 . 6 _ { \perp 1 9 . 3 } }$ </td><td> $\mathbf { - 4 . 6 } _ { \downarrow 1 2 . 6 }$ </td><td> ${ \mathbf { 8 4 . 1 } } _ { \downarrow , 3 5 , 6 }$ </td><td> ${ \bf 8 1 . 8 _ { \perp 2 5 . 5 } }$ </td><td> $2 . 3 _ { \downarrow 1 0 . 1 }$ </td><td> $\mathbf { - 0 . 4 } _ { \downarrow , 1 0 . 8 }$ </td></tr><tr><td>Gemma-2-9B</td><td> $7 8 . 4 _ { \perp 2 8 . 8 }$ </td><td> $\mathbf { 7 9 . 9 } _ { \perp 2 0 . 7 }$ </td><td> $\mathbf { - 1 . 5 } _ { \downarrow 8 . 1 }$ </td><td> ${ \bf 8 3 . 7 } _ { \perp 3 3 . 1 }$ </td><td> $7 8 . 8 _ { \perp 2 2 . 9 }$ </td><td> ${ \bf 4 . 9 } _ { \perp 1 0 . 2 }$ </td><td> $7 7 . 1 _ { \perp 4 1 . 0 }$ </td><td> ${ \bf 8 1 . 8 _ { \perp 2 5 . 2 } }$ </td><td> $\mathbf { - 4 . 7 _ { \downarrow 1 5 . 8 } }$ </td><td> $\mathbf { - 0 . 4 } _ { \downarrow 1 1 . 4 }$ </td></tr><tr><td>Average</td><td> $7 7 . 8 _ { \downarrow 3 4 . 5 }$ </td><td> $7 7 . 9 _ { \perp 2 2 . 9 }$ </td><td> ${ \bf - 0 . 1 _ { \perp 1 1 . 7 } }$ </td><td> $7 9 . 5 _ { \perp 3 4 . 1 }$ </td><td> $7 8 . 6 _ { \perp 2 2 . 5 }$ </td><td> $\mathbf { 0 . 9 } _ { \downarrow 1 1 . 6 }$ </td><td> $7 9 . 6 _ { \perp 4 2 . 5 }$ </td><td> $\mathbf { 7 9 . 0 _ { \perp 2 8 . 1 } }$ </td><td> $\mathbf { 0 . 6 } _ { \downarrow 1 4 . 4 }$ </td><td> $\mathbf { 0 . 5 } _ { \downarrow 1 2 . 5 }$ </td></tr><tr><td colspan="9"></td><td></td><td></td></tr><tr><td>SSearch-R1</td><td> $\mathbf { 8 3 . 9 } _ { \downarrow 5 1 . 5 }$ </td><td> ${ \bf 8 8 . 5 _ { \perp 4 0 . 0 } }$ </td><td> $\mathbf { - 4 . 6 } _ { \downarrow 1 1 . 5 }$ </td><td> ${ \bf 8 1 . 6 } _ { \perp 4 5 . 7 }$ </td><td>RL-based  ${ \mathbf { 8 6 . 2 _ { \perp 4 0 . 0 } } }$ </td><td> $\mathbf { - 4 . 6 } _ { \downarrow , 5 . 7 }$ </td><td> $8 3 . 4 _ { \downarrow 5 1 . 6 }$ </td><td> $7 7 . 8 _ { \downarrow 4 4 . 0 }$ </td><td> ${ \bf 5 . 6 } _ { \downarrow 7 . 6 }$ </td><td> ${ \bf - 1 . 2 } _ { \downarrow 8 . 3 }$ </td></tr><tr><td>ZeroSearch-base</td><td> $7 8 . 6 _ { \perp 4 5 . 8 }$ </td><td> ${ \bf 8 1 . 1 _ { \perp 3 6 . 1 } }$ </td><td> ${ \bf - 2 . 5 _ { \perp 9 . 7 } }$ </td><td> ${ \bf 8 1 . 8 _ { \perp 4 7 . 9 } }$ </td><td> ${ \bf 8 1 . 2 } _ { \perp 3 6 . 9 }$ </td><td> $\mathbf { 0 . 6 } _ { \downarrow 1 1 . 0 }$ </td><td> ${ \mathbf { 8 0 . 7 } } _ { \perp 5 4 . 6 }$ </td><td> $7 9 . 5 _ { \perp 3 9 . 2 }$ </td><td> ${ \bf 1 . 2 } _ { \downarrow 1 5 . 4 }$ </td><td> $\mathbf { - 0 . } 2 _ { \downarrow 1 2 . 0 }$ </td></tr><tr><td>RREDSearcher</td><td> $\mathbf { 8 2 . 4 _ { \perp 4 4 . 3 } }$ </td><td> ${ \bf 8 5 . 2 _ { \perp 3 2 . 4 } }$ </td><td> $\mathbf { - 2 . 8 _ { \perp 1 1 . 9 } }$ </td><td> ${ \bf 8 5 . 8 _ { \perp 3 9 . 9 } }$ </td><td> ${ \bf 8 6 . 5 _ { \perp 2 9 . 5 } }$ </td><td> $\mathbf { - 0 . 7 _ { \downarrow 1 0 . 4 } }$ </td><td> ${ \mathbf { 8 3 . 7 } } _ { \perp 4 7 . 8 }$ </td><td> ${ \bf 8 3 . 7 _ { \perp 3 2 . 7 } }$ </td><td> $\mathbf { 0 . 0 } _ { \downarrow 1 5 . 1 }$ </td><td> ${ \bf - 1 . 2 } _ { \downarrow 1 2 . 5 }$ </td></tr><tr><td>Search-P1</td><td> ${ \bf 8 6 . 5 _ { \perp 4 9 . 1 } }$ </td><td> $\mathbf { 8 4 . 2 } _ { \perp 3 9 . 2 }$ </td><td> ${ \pmb { 2 . 3 } } _ { \downarrow 9 . 9 }$ </td><td> ${ \mathbf { 8 3 . 5 } } _ { \perp 4 3 . 4 }$ </td><td> ${ \mathbf 8 6 . 7 } _ { \perp 3 6 . 2 }$ </td><td> ${ \bf - 3 . 2 } _ { \downarrow 7 . 2 }$ </td><td> $8 5 . 6 _ { \perp 5 4 . 6 }$ </td><td> $\mathbf { 7 9 . 0 _ { \perp 4 4 . 3 } }$ </td><td> ${ \bf 6 . 6 } _ { \downarrow 1 0 . 3 }$ </td><td> ${ \bf 1 . 9 _ { \perp 9 . 1 } }$ </td></tr><tr><td>Average</td><td> $\mathbf { 8 2 . 9 } _ { \perp 4 7 . 7 }$ </td><td> ${ \bf 8 4 . 8 _ { \perp 3 6 . 9 } }$ </td><td> $\mathbf { - 1 . 9 } _ { \downarrow 1 0 . 8 }$ </td><td> ${ \bf 8 3 . 2 _ { \perp 4 4 . 2 } }$ </td><td> ${ \bf 8 5 . 2 } _ { \perp 3 5 . 7 }$ </td><td> $\mathbf { - 2 . 0 } _ { \downarrow . 8 . 6 }$ </td><td> $8 3 . 4 _ { \downarrow 5 2 . 2 }$ </td><td> ${ \bf 8 0 . 0 _ { \perp . 4 0 . 1 } }$ </td><td> $3 . 4 _ { \downarrow 1 2 . 1 }$ </td><td> $\mathbf { - 0 . } 2 _ { \downarrow 1 0 . 5 }$ </td></tr></table>

## 4.3 Results and Behavioral Interpretation

Table 1 gives the main TriviaQA-RC grid; Tables 13 and 14 in Appendix E give HotpotQA and SearchQA. Quantitative summaries below use these tables. Strict/Soft cells pair bold clean accuracy on the full set with teal ↓ MR on each model’s baseline-correct subset. Each $\Delta$ cell pairs the corresponding clean-accuracy and MR contrasts (pp); the arrow denotes MR, not accuracy loss. Unless qualified, textual contrasts refer to MR. Group means weight models equally and are rounded last (Appendix A.4).

Instructions act as a policy for selecting sources. Strict MR exceeds Soft MR in every reported cell of the main tables. Within each comparison, question, evidence, and placement are fixed; the policy for source priority changes. The gap is therefore consistent with evidence following that depends on the policy, rather than different retrieved content. Its direction is expected from the instructions; the diagnostic result is the frequency of designated-foil adoption among baseline-correct answers. Output differences alone do not identify internal attention, the relative weight of memory and context, or erasure of a stored fact.

Word Position changes evidence use. Averaging equally across the fifteen models gives End > Beginning > Middle under both instructions on all three datasets. Strict MR at these positions is 37.0/31.4/30.3% on TriviaQA-RC, 41.3/35.8/32.8% on HotpotQA, and 35.0/29.3/26.9% on SearchQA. This is an aggregate pattern, not a universal per-model ordering. Later placement may change the answer span’s salience, but does not identify an attention mechanism or document-order effect. Section 4.4.1 provides a paired position check with uncertainty estimates.

Dataset structure may shape override. HotpotQA has higher MR averaged across models than TriviaQA-RC and SearchQA under both instructions. One explanation is that a false intermediate assertion can support an internally coherent but incorrect reasoning chain, not just replace its final answer. The present comparison does not isolate that mechanism: evidence style, difficulty, and baseline-correct cohorts also differ. Testing it would require matched bridge-versus-terminal edits, rather than treating a cross-dataset difference as a causal multi-hop effect.

Model families differ in default evidence reliance. API systems have lower group mean MR than open 7–9B models; RL-trained search systems have the highest Soft MR and smallest mean instruction gap (SearchQA: 10.1 vs. 10.3 pp for APIs). This is consistent with greater evidence reliance under Soft, not an isolated RL-training effect. Each model’s MR conditions on a different $K _ { M } \colon$ low MR need not imply higher accuracy on the full set. Cross-model rankings therefore require both baseline coverage and matched-cohort controls; a small instruction gap alone is not robustness.

Clean controls separate override from general degradation. Table 1 displays clean accuracy on the full set alongside baseline-conditioned edited MR; these have different denominators and are not subtracted. The separate paired audit compares clean and edited responses on matched questions to assess reference retention. Appendices C.4 and C.3 report matched foil adoption and direct clean-gold to edited-foil transitions; Appendix C.2 compares gold retention on a common cohort.

## 4.4 Ablations and Robustness Checks

We check paired position differences, sensitivity to prompt wording, and whether the instruction effect extends across tasks and languages. Further checks address edit validity, matcher errors, and decoding profiles.

## 4.4.1 Paired Validation of Word Position Effects

Table 2 evaluates 300 QA questions per model, retaining $| K _ { A } | = 2 2 0$ and $| K _ { B } | = 1 9 8$ . Intervals use 2,000 paired bootstrap resamples. Under Strict, End exceeds Middle by 10.9/9.6 pp for A/B and also exceeds Beginning. All Beginning-minus-Middle and Soft contrast intervals include zero. These results support higher End MR under Strict, not the complete ordering or a tested instruction–position interaction. The Word Position check is separate from the document-slot audit (Appendix F).

Table 2: Paired Word Position validation on 300 questions per model. QA/A and QA/B retain the supplied dataset/model identifiers. Contrasts are MR percentage points with pointwise 95% paired-bootstrap intervals $( B = 2 , 0 0 0 )$
<table><tr><td>Dataset / model Instruction</td><td></td><td> $\vert K _ { M } \vert$ </td><td>End – Middle [95% CI]</td><td>Beginning — Middle [95% CI]</td><td>Beginning — End [95% CI]</td></tr><tr><td>QA/A</td><td>Soft</td><td>220</td><td> $5 . 0 \left[ - 0 . 5 , 1 0 . 5 \right]$ </td><td>0.9 [−5.0, 6.8]</td><td> $- 4 . 1 \left[ - 1 0 . 0 , 2 . 3 \right]$ </td></tr><tr><td>QA/A</td><td>Strict</td><td>220</td><td>10.9 [4.5, 17.7]</td><td>4.1 [−1.8, 10.5]</td><td> $- 6 . 8 \left[ { \bar { - } } 1 3 . 6 , - 0 . { \dot { 5 } } \right]$ </td></tr><tr><td>QA/B</td><td>Soft</td><td>198</td><td> $6 . 6 \ [ \dot { - } 1 . 0 , 1 4 . \dot { 1 } ]$ </td><td>5.1 [−2.0, 12.1]</td><td> $- 1 . { \dot { 5 } } [ - 9 . 1 , 5 . 6 ]$ </td></tr><tr><td>QA/B</td><td>Strict</td><td>198</td><td> $9 . 6 \ : [ 2 . 5 , 1 6 . 7 ]$ </td><td>2.5 [−4.5, 9.6]</td><td> $- 7 . 1 \ : [ - 1 3 . 6 , - 0 . { \overset { . } { 5 } } ]$ </td></tr></table>

## 4.4.2 Robustness to Paraphrases of Source Policy

Two pairs of paraphrases of the source policy accompany the original Soft/Strict clauses, with evidence layouts and other settings held fixed. All versions retain the baseline-correct sets from the 300-question QA evaluation. Table 3 reports position-averaged MR and paired contrasts. Strictminus-Soft gaps remain positive across versions: 10.8–11.6 pp for A and 7.8–9.3 pp for B, with all instruction-contrast intervals excluding zero. Changes relative to the original gap range from −1.6 to +0.8 pp, and all corresponding intervals include zero. This supports directionally consistent instruction effects in the tested versions, not equivalence of effect sizes or invariance to arbitrary wording. Appendix F.2 gives the clauses and paired analysis.

Table 4: MedQA (Jin et al., 2020), averaged over Word Position. Strict/Soft cells: bold clean accuracy on the full set; ↓ MR on closed-book-correct items (%). n : eligible count out of 1,000. Each ∆ cell: bold clean-accuracy contrast and ↓ MR contrast (pp). Instruction contrasts are Strict minus Soft; language contrasts are Chinese minus English under Strict. The arrow identifies the MR component, not an accuracy drop or a sign.
<table><tr><td>MedQA</td><td colspan="4">English (USMLE)</td><td colspan="4">Chinese (MCMLE)</td></tr><tr><td>Model</td><td>nK</td><td>Strict</td><td>Soft</td><td> $\pmb { \Delta } _ { \mathbf { i n s t r } }$ </td><td>nK</td><td>Strict</td><td>Soft</td><td> $\pmb { \Delta } _ { \mathbf { i n s t r } }$ </td><td> $\Delta _ { \mathrm { l a n g } }$ </td></tr><tr><td></td><td></td><td></td><td></td><td>API-based</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="10"></td></tr><tr><td>GPT-5.6</td><td>912</td><td> $\mathbf { 9 6 . 8 _ { \perp 2 1 . 4 } }$ </td><td> $\mathbf { 9 5 . 8 _ { \perp 9 . 8 } }$ </td><td> $\mathbf { 1 . 0 } _ { \downarrow 1 1 . 6 }$ </td><td>874</td><td> $9 0 . 5 _ { \perp 2 3 . 1 }$ </td><td> ${ \bf 9 1 . 0 _ { \perp 1 0 . 6 } }$ </td><td> $\mathbf { - 0 . 5 } _ { \downarrow 1 2 . 5 }$ </td><td> ${ \mathbf { - 6 . 3 _ { \downarrow + 1 . 7 } } }$ </td></tr><tr><td>Claude Opus 4.6</td><td>905</td><td> ${ \bf 9 6 . 2 _ { \perp 1 8 . 7 } }$ </td><td> ${ \bf 8 8 . 7 . } _ { \perp 8 . 2 }$ </td><td> $7 . 5 _ { \downarrow 1 0 . 5 }$ </td><td>861</td><td> $\mathbf { 9 6 . 9 } _ { \downarrow 2 0 . 5 }$ </td><td> ${ \bf 9 5 . 2 } _ { \downarrow 9 . 1 }$ </td><td> $1 . 7 _ { \downarrow 1 1 . 4 }$ </td><td> ${ \bf 0 . 7 } _ { \downarrow + 1 . 8 }$ </td></tr><tr><td>Qwen3.8-Max</td><td>871</td><td> ${ \bf 9 2 . 1 _ { \perp 2 4 . 9 } }$ </td><td> ${ \bf 9 1 . 2 _ { \perp 1 1 . 7 } }$ </td><td> $\mathbf { 0 . 9 } _ { \downarrow 1 3 . 2 }$ </td><td>903</td><td> ${ \bf 9 3 . 1 _ { \perp 2 2 . 8 } }$ </td><td> ${ \bf 8 9 . 2 _ { \perp 1 0 . 9 } }$ </td><td> ${ \bf 3 . 9 } _ { \downarrow 1 1 . 9 }$ </td><td> $\pmb { 1 . 0 } _ { \downarrow - 2 . 1 }$ </td></tr><tr><td> DeepSeek-V4-Pro</td><td>889</td><td> ${ \mathbf { 8 6 . 3 _ { \perp 2 3 . 6 } } }$ </td><td> ${ \bf 8 4 . 2 _ { \perp 1 0 . 4 } }$ </td><td> ${ \bf 2 . 1 } _ { \downarrow 1 3 . 2 }$ </td><td>911</td><td> ${ \bf 8 9 . 3 _ { \perp 2 1 . 9 } }$ </td><td> ${ \mathbf { 8 3 . 3 } } _ { \downarrow 9 . 8 }$ </td><td> ${ \bf 6 . 0 } _ { \downarrow 1 2 . 1 }$ </td><td> ${ \bf 3 . 0 } _ { \downarrow . 1 . 7 }$ </td></tr><tr><td>Z GLM-5.2</td><td>858</td><td> ${ \bf 8 9 . 0 _ { \perp 2 6 . 2 } }$ </td><td> ${ \bf 9 1 . 7 _ { \perp 1 2 . 5 } }$ </td><td> $\mathbf { - 2 . 7 _ { \downarrow 1 3 . 7 } }$ </td><td>896</td><td> ${ \bf 9 6 . 4 _ { \perp 2 4 . 4 } }$ </td><td> $\mathbf { 8 9 . 7 } _ { \perp 1 1 . 6 }$ </td><td> ${ \bf 6 . 7 } _ { \downarrow 1 2 . 8 }$ </td><td> $\mathbf { 7 . 4 } _ { \downarrow \cdot 1 . 8 }$ </td></tr><tr><td>KKimi-K3</td><td>866</td><td> $\mathbf { 9 0 . 1 _ { \perp 2 5 . 1 } }$ </td><td> ${ \bf 8 7 . 2 _ { \perp 1 1 . 9 } }$ </td><td> $\mathbf { 2 . 9 } _ { \downarrow 1 3 . 2 }$ </td><td>893</td><td> ${ \bf 9 0 . 1 } _ { \perp 2 3 . 7 }$ </td><td> $\mathbf { 9 4 . 4 } _ { \downarrow 1 1 . 2 }$ </td><td> ${ \mathbf { - 4 . 3 _ { \downarrow 1 2 . 5 } } }$ </td><td> $\mathbf { 0 . 0 } _ { \downarrow \downarrow . 1 . 4 }$ </td></tr><tr><td>HY4-Preview Average</td><td>842</td><td> ${ \bf 8 9 . 4 } _ { \perp 2 7 . 8 }$ </td><td> ${ \bf 8 8 . 1 _ { \perp 1 3 . 4 } }$ </td><td> ${ \bf 1 . 3 _ { \perp 1 4 . 4 } }$ </td><td>887</td><td> $8 5 . 7 _ { \perp 2 5 . 6 }$ </td><td> $\mathbf { 8 9 . 2 } _ { \downarrow 1 2 . 3 }$ </td><td> $\mathbf { - 3 . 5 } _ { \downarrow 1 3 . 3 }$ </td><td> ${ \cdot } 3 . 7 _ { \downarrow { - } 2 . 2 }$ </td></tr><tr><td></td><td>878</td><td> ${ \bf 9 1 . 4 } _ { \perp 2 4 . 0 }$ </td><td> ${ \bf 8 9 . 6 _ { \perp 1 1 . 1 } }$ </td><td> $\mathbf { 1 . 9 } _ { \downarrow 1 2 . 8 }$ </td><td>889</td><td> ${ \bf 9 1 . 7 _ { \perp 2 3 . 1 } }$ </td><td> ${ \bf 9 0 . 3 _ { \perp 1 0 . 8 } }$ </td><td> ${ \bf 1 . 4 } _ { \downarrow 1 2 . 4 }$ </td><td> $\mathbf { 0 . 3 _ { \downarrow - 0 . 8 } }$ </td></tr><tr><td colspan="10">Open-source</td></tr><tr><td>∞ Llama-3-8B</td><td>583</td><td> $7 6 . 6 _ { \perp 4 1 . 7 }$ </td><td> ${ \bf 7 8 . 9 _ { \perp 2 4 . 3 } }$ </td><td> $\mathbf { - 2 . 3 _ { \downarrow 1 7 . 4 } }$ </td><td>412</td><td> ${ \bf 8 0 . 0 _ { \perp 4 8 . 9 } }$ </td><td> $7 8 . 8 _ { \perp 2 9 . 6 }$ </td><td> ${ \bf 1 . 2 } _ { \downarrow 1 9 . 3 }$ </td><td> $3 . 4 _ { \downarrow + 7 . 2 }$ </td></tr><tr><td>H Mistral-7B</td><td>541</td><td> $\mathbf { 7 9 . 9 _ { \perp 4 4 . 2 } }$ </td><td> $7 4 . 7 _ { \perp 2 6 . 8 }$ </td><td> ${ \pmb 5 . 2 } _ { \downarrow 1 7 . 4 }$ </td><td>368</td><td> $\mathbf { 8 0 . 0 } _ { \downarrow . 5 2 . 3 }$ </td><td> $7 7 . 5 _ { \perp 3 1 . 7 }$ </td><td> $2 . 5 _ { \downarrow 2 0 . 6 }$ </td><td> ${ \bf 0 . 1 } _ { + 8 . 1 }$ </td></tr><tr><td>Qwen3-8B</td><td>612</td><td> ${ \mathbf 8 6 . 3 } _ { \perp 3 . 4 }$ </td><td> ${ \mathbf { 8 3 . 1 } } _ { \downarrow 2 1 . 9 }$ </td><td> $3 . 2 _ { \downarrow 1 6 . 5 }$ </td><td>689</td><td> ${ \bf 8 6 . 6 } _ { \downarrow , 3 5 . 6 }$ </td><td> ${ \mathbf { 8 3 . 3 _ { \perp 2 0 . 4 } } }$ </td><td> $3 . 3 _ { \perp 1 5 . 2 }$ </td><td> $\mathbf { 0 . 3 } _ { \downarrow . 2 . 8 }$ </td></tr><tr><td>Gemma-2-9B</td><td>628</td><td> $7 7 . 5 _ { \perp 3 6 . 9 }$ </td><td> $\mathbf { 8 0 . 7 _ { \perp 2 0 . 6 } }$ </td><td> $\mathbf { - } 3 . 2 _ { \downarrow 1 6 . 3 }$ </td><td>497</td><td> $\mathbf { 8 0 . 9 } _ { \downarrow 4 2 . 8 }$ </td><td> $7 5 . 3 _ { \perp 2 4 . 9 }$ </td><td> ${ \bf 5 . 6 } _ { \downarrow 1 7 . 9 }$ </td><td> $3 . 4 _ { \downarrow + 5 . 9 }$ </td></tr><tr><td>Average</td><td>591</td><td> ${ \bf 8 0 . 1 _ { \perp 4 0 . 3 } }$ </td><td> $7 9 . 4 _ { \perp 2 3 . 4 }$ </td><td> $\mathbf { 0 . 7 } _ { \downarrow 1 6 . 9 }$ </td><td>492</td><td> ${ \bf 8 1 . 9 _ { \perp 4 4 . 9 } }$ </td><td> $7 8 . 7 _ { \perp 2 6 . 7 }$ </td><td> $3 . 2 _ { \downarrow 1 8 . 3 }$ </td><td> $\mathbf { 1 . 8 _ { \perp + 4 . 6 } }$ </td></tr><tr><td colspan="10"> $R L \mathbf { - } b a s e d$ </td></tr><tr><td>Search-R1</td><td>597</td><td> ${ \bf 8 3 . 3 _ { \perp 5 2 . 6 } }$ </td><td> ${ \mathbf { 8 5 . 7 } } _ { \downarrow 4 3 . 8 }$ </td><td> $\mathbf { - 2 . 4 } _ { \downarrow , 8 . 8 }$ </td><td>571</td><td> $7 7 . 8 _ { \downarrow 5 5 . 1 }$ </td><td> $7 8 . 9 _ { \perp 4 6 . 2 }$ </td><td> ${ \bf - 1 . 1 _ { \perp 8 . 9 } }$ </td><td> ${ \bf - 5 . 5 } _ { \downarrow + 2 . 5 }$ </td></tr><tr><td>ZeroSearch-base</td><td>604</td><td> ${ \bf 8 1 . 6 _ { \perp 4 9 . 3 } }$ </td><td> ${ \bf 8 1 . 3 _ { \perp 3 9 . 7 } }$ </td><td> $\mathbf { 0 . 3 _ { \perp 9 . 6 } }$ </td><td>618</td><td> ${ \bf 8 7 . 3 _ { \perp 5 0 . 2 } }$ </td><td> $7 7 . 7 _ { \perp 4 1 . 3 }$ </td><td> $\mathbf { 9 . 6 } _ { \perp 8 . 9 }$ </td><td> ${ \bf 5 . 7 _ { \perp + 0 . 9 } }$ </td></tr><tr><td>REDSearcher</td><td>688</td><td> $\mathbf { 8 9 . 3 _ { \perp 4 6 . 8 } }$ </td><td> ${ \bf 9 0 . 0 } _ { \perp 3 5 . 2 }$ </td><td> $\mathbf { - 0 . 7 } _ { \downarrow 1 1 . 6 }$ </td><td>702</td><td> ${ \mathbf { 8 3 . 7 } } _ { \perp 4 7 . 5 }$ </td><td> $7 7 . 5 _ { \perp 3 6 . 9 }$ </td><td> ${ \bf 6 . 2 } _ { \downarrow 1 0 . 6 }$ </td><td> ${ \bf - 5 . 6 } _ { \downarrow + 0 . 7 }$ </td></tr><tr><td>Search-P1</td><td>641</td><td> ${ \bf 8 5 . 0 _ { \perp 5 0 . 9 } }$ </td><td> ${ \bf 8 2 . 0 _ { \perp 4 1 . 6 } }$ </td><td> ${ \bf 3 . 0 } _ { \downarrow 9 . 3 }$ </td><td>676</td><td> ${ \mathbf { 8 6 . 3 _ { \perp 5 1 . 4 } } }$ </td><td> $7 9 . 4 _ { \downarrow 4 2 . 8 }$ </td><td> ${ \bf 6 . 9 _ { \perp 8 . 6 } }$ </td><td> ${ \bf 1 . 3 _ { \perp + 0 . 5 } }$ </td></tr><tr><td>Average</td><td>633</td><td> ${ \bf 8 4 . 8 _ { \perp 4 9 . 9 } }$ </td><td> ${ \bf 8 4 . 8 _ { \perp 4 0 . 1 } }$ </td><td> ${ \bf 0 . 1 } _ { \downarrow 9 . 8 }$ </td><td>642</td><td> ${ \bf 8 3 . 8 _ { \perp 5 1 . 1 } }$ </td><td> $7 8 . 4 _ { \perp 4 1 . 8 }$ </td><td> ${ \pmb 5 . 4 } _ { \downarrow . 9 . 3 }$ </td><td> $\mathbf { \delta - 1 . 0 } _ { \downarrow , + 1 , 2 }$ </td></tr></table>

Table 3: Source-policy paraphrase robustness on QA, with 300 questions per model. MR averages three positions (%); contrasts and pointwise 95% intervals are in pp. Supplied contrasts are retained at their reported precision; differences of rounded displayed values can differ by 0.1 pp. Ref. denotes the original-to-itself contrast.
<table><tr><td>Model (|KM |) Version</td><td></td><td>Soft MR Strict MR</td><td></td><td> $\Delta _ { \mathrm { i n s t r } }$  [95% CI]</td><td> $\Delta _ { \mathrm { i n s t r } } - \Delta _ { \mathrm { o r i g i n a l } } \ : [ 9 5 \% \mathrm { C I I } ]$ </td></tr><tr><td> $\mathbf { A } \left( 2 2 0 \right)$ </td><td>Original</td><td>18.6</td><td>29.3</td><td> $1 0 . 8 \ : [ 7 . 7 , 1 3 . 9 ]$ </td><td> $0 . 0 \ : ( \mathrm { r e f . ) }$ </td></tr><tr><td> $\mathbf { A } \left( 2 2 0 \right)$ </td><td>Paraphrase 1</td><td>19.4</td><td>31.0</td><td>11.6 [8.7, 14.7]</td><td> $0 . 8 \ [ - 1 . 4 , 3 . 1 ]$ </td></tr><tr><td> $\mathbf { A } \left( 2 2 0 \right)$ </td><td>Paraphrase 2</td><td>17.7</td><td>28.9</td><td>11.2 [8.1, 14.1]</td><td> $0 . 4 \ \mathrm { [ - 1 . 3 , 2 . 1 ] }$ </td></tr><tr><td>B (198)</td><td>Original</td><td>28.9</td><td>38.2</td><td>9.3 [6.3, 12.3]</td><td> $0 . 0 \ : ( \mathrm { r e f . ) }$ </td></tr><tr><td>B (198)</td><td>Paraphrase 1</td><td>29.3</td><td>37.2</td><td>7.9 [4.7, 11.2]</td><td> $- 1 . 4 \left[ - 3 . 7 , 0 . 9 \right]$ </td></tr><tr><td>B (198)</td><td>Paraphrase 2</td><td>28.7</td><td>36.4</td><td>7.8 [4.4, 11.0]</td><td> $- 1 . 6 \left[ - 3 . 9 , 0 . 8 \right]$ </td></tr></table>

## 4.4.3 Cross-Lingual Generalization on Medical QA

To assess whether the instruction contrast extends to medical QA in two languages, we use English USMLE and simplified-Chinese MCMLE questions from MedQA (Jin et al., 2020), with MedRAG evidence (Xiong et al., 2024). Each language contains 1,000 sampled questions. Foils are provided distractor options; MR uses each model’s subset answered correctly without retrieval $( n _ { K }$ in Table 4). Group means weight models equally. The two languages contain different questions, not translations of a paired test set.

The instruction effect is consistent across languages. Strict MR exceeds Soft MR for all 15 systems in both languages. Group-averaged $\Delta _ { \mathrm { { i n s t r } } }$ is similar between English and Chinese (API-based: 12.8 vs. 12.4; open-source: 16.9 vs. 18.3; RL-based: 9.8 vs. 9.3). The instruction effect is therefore consistent across the two language settings.

Language differences covary with closed-book coverage. The language effect $\Delta _ { \mathrm { l a n g } }$ covaries with closed-book coverage. English-centric open models retain fewer Chinese questions and show higher Chinese MR, whereas Qwen3-8B and several Chinese-developed API models show the opposite direction. Because USMLE and MCMLE contain different questions, this pattern is descriptive and cannot isolate language proficiency from item composition.

High Soft MR in search agents. RL-based systems have the highest Soft MR (40.1% English and 41.8% Chinese on average) and the smallest group-averaged $\Delta _ { \mathrm { i n s t r } } .$ . This indicates substantial foil following even under Soft prompting, but comparisons across different systems do not isolate the effect of RL training.

## 4.4.4 Counterfactual Evidence and Matcher Validation

Human validation of counterfactual evidence. We sample 200 edits from each of TriviaQA-RC, HotpotQA, SearchQA, and the two MedQA languages (1,000 total). Six annotators cross-validate the edits: two independently label each edit for foil support, relation preservation, fluency, and leakage of the reference answer, with disagreements adjudicated by a third.

Table 5: Human validation of 1,000 edited passages by six cross-validating annotators. Pass rates (%) after adjudication; last two columns: $\Delta _ { \mathrm { { i n s t r } } }$ (pp) on all vs. validated edits.
<table><tr><td colspan="7">Pass rate (%) ↑</td><td colspan="3"> $\pmb { \Delta } _ { \mathrm { i n s t r } }$  (pp)</td></tr><tr><td>Dataset</td><td>n</td><td>Foil support</td><td>Relation kept</td><td>Fluency</td><td>No leakage</td><td>All pass</td><td>κ↑</td><td>All</td><td>Valid.</td></tr><tr><td>TriviaQA-RC</td><td>200</td><td>96.5</td><td>97.0</td><td>95.5</td><td>98.5</td><td>91.5</td><td>0.81</td><td>11.8</td><td>12.3</td></tr><tr><td>HotpotQA</td><td>200</td><td>93.0</td><td>92.5</td><td>94.0</td><td>96.0</td><td>86.0</td><td>0.77</td><td>13.6</td><td>14.2</td></tr><tr><td>SearchQA</td><td>200</td><td>94.5</td><td>95.5</td><td>92.0</td><td>97.5</td><td>88.5</td><td>0.79</td><td>10.9</td><td>11.4</td></tr><tr><td>MedQA (EN)</td><td>200</td><td>95.0</td><td>93.5</td><td>96.5</td><td>97.0</td><td>88.0</td><td>0.76</td><td>12.9</td><td>13.3</td></tr><tr><td>MedQA (ZH)</td><td>200</td><td>94.0</td><td>92.0</td><td>95.5</td><td>96.5</td><td>86.5</td><td>0.74</td><td>12.4</td><td>12.9</td></tr><tr><td>All</td><td>1000</td><td>94.6</td><td>94.1</td><td>94.7</td><td>97.1</td><td>88.1</td><td>0.77</td><td>12.3</td><td>12.8</td></tr></table>

In Table 5, 88.1% of edits pass all criteria $( \kappa = 0 . 7 7 )$ . Relation preservation is the main bottleneck for HotpotQA and Chinese MedQA; SearchQA has the lowest fluency. Filtering increases mean $\Delta _ { \mathrm { { i n s t r } } }$ from 12.3 to 12.8 pp within this comparison, preserving its direction. These are not model group means in the main tables, and filtering changes item composition.

Matcher error analysis. We sample 450 outputs stratified by matcher label (Gold, Foil-only, and Unmatched; 150 each) across datasets, models, instructions, and positions. Human labels identify the committed answer. For matcher–human agreement, human Hedge and Other labels map to Unmatched; the displayed counts yield Cohen’s $\kappa = 0 . 9 1$ , not an inter-annotator coefficient.

Table 6: Matcher error analysis on 450 stratified outputs. Matcher–human $\kappa = 0 . 9 1$ after mapping human Hedge/Other to Unmatched. Left: label counts. Right: causes of the 26 errors.
<table><tr><td colspan="8">Human label</td></tr><tr><td>Matcher</td><td>Gold</td><td>Foil</td><td></td><td>Hedge Other</td><td>n</td><td></td><td>Agr. %</td></tr><tr><td>Gold</td><td>146</td><td>0</td><td>3</td><td></td><td>1</td><td>150</td><td>97.3</td></tr><tr><td>Foil-only</td><td>2</td><td>143</td><td>4</td><td></td><td>1</td><td>150</td><td>95.3</td></tr><tr><td>Unmatched</td><td>9</td><td>6</td><td>71</td><td>64</td><td></td><td>150</td><td>90.0</td></tr><tr><td>Total</td><td>157</td><td>149</td><td>78</td><td></td><td>66</td><td>450</td><td>94.2</td></tr></table>

<table><tr><td>Error cause</td><td>n</td><td>%</td></tr><tr><td>Alias/format</td><td>12</td><td>46.2</td></tr><tr><td>Hedged</td><td>7</td><td>26.9</td></tr><tr><td>Multi-entity</td><td>5</td><td>19.2</td></tr><tr><td>Explanatory</td><td>2</td><td>7.7</td></tr><tr><td>Total errors</td><td>26</td><td>100.0</td></tr></table>

Table 6 gives 94.2% agreement and 95.3% Foil-only precision on the sample balanced by label. Its 26 disagreements mainly involve aliases, formatting, hedging, or multiple entities. Sampling stratified by label prevents interpreting these statistics as agreement across the population. This diagnoses scoring errors, not benchmark MR sensitivity to an alternative judge.

## 4.4.5 Decoding Profiles

Table 7 reports six systems on TriviaQA-RC under temperature sampling (temperature 0.7, top-p 0.9, five seeds) and greedy decoding with a 256-token output cap. These are separate decoding profiles, not a matched single-factor contrast against the runs in the main tables.

Table 7: Decoding profiles on TriviaQA-RC (MR %, mean over Word Position conditions). Sampling: temperature 0.7, top-p 0.9, five seeds (mean±s.d.). Long generation: greedy decoding, 256-token cap.
<table><tr><td></td><td colspan="3">Sampling (5 seeds)</td><td colspan="3">Long generation (256 tokens)</td></tr><tr><td>Model</td><td>Strict</td><td>Soft</td><td>∆instr</td><td>Strict</td><td>Soft</td><td>∆instr</td></tr><tr><td>GPT-5.6</td><td>21.2±0.9</td><td>10.1±0.7</td><td>11.1±0.6</td><td>18.6</td><td>8.1</td><td>10.5</td></tr><tr><td> DeepSeek-V4-Pro</td><td>23.9±1.1</td><td>11.4±0.8</td><td>12.5±0.7</td><td>21.3</td><td>9.5</td><td>11.8</td></tr><tr><td>∞ Llama-3-8B</td><td>41.3±1.4</td><td>27.0±1.2</td><td>14.3±0.9</td><td>36.8</td><td>23.6</td><td>13.2</td></tr><tr><td>Qwen3-8B</td><td>37.0±1.3</td><td>23.2±1.0</td><td>13.8±0.8</td><td>33.4</td><td>20.8</td><td>12.6</td></tr><tr><td>0 Search-R1</td><td>51.2±1.6</td><td>42.5±1.5</td><td>8.7±1.1</td><td>48.1</td><td>39.6</td><td>8.5</td></tr><tr><td>REDSearcher</td><td>45.9±1.5</td><td>35.7±1.3</td><td>10.2±1.0</td><td>42.7</td><td>32.9</td><td>9.8</td></tr><tr><td>Strict-MR rank agreement Models with mean  $\Delta _ { \mathrm { i n s t r } } > 0$ </td><td colspan="6">Kendall τ = 1.00 between the two profiles 6/6</td></tr></table>

Both profiles retain positive mean instruction contrasts for all six systems. Reported seed standard deviations are at most 1.6 pp, and the Strict-MR ordering is identical between the sampling means and long-generation values (Kendall τ = 1.00). This is a descriptive consistency check: it does not isolate a sampling or output-length effect, nor measure changes relative to Table 1.

Ablation summary. The checks address different threats: task transfer, edit validity, scoring error, decoding, and wording. Reported instruction contrasts remain positive; the paired position check supports higher End MR under Strict, not the full aggregate ordering. These are complementary diagnostics, not interchangeable replications or isolated tests of every causal factor.

## 5 Conclusion

RAG-Stress measures a specific reliability failure: adoption of an evidence-supported foil after a correct closed-book response. Across fifteen systems and three QA datasets, the reported mean Strict–Soft MR gap is 10.9–13.5 pp; rates averaged across models follow End > Beginning > Middle under both instructions. Medical QA and decoding checks extend the instruction pattern beyond one setting. A separate audit of saved outputs supports increased harmful override without establishing a gain in beneficial correction. These findings distinguish compliance with a policy for source priority from factual reliability; they do not identify an internal attention mechanism or a superior mitigation. The practical evaluation lesson is to report baseline coverage, foil adoption, and matched clean/edited transitions separately, rather than infer robustness from aggregate accuracy or document adherence alone.

## References

Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. Self-RAG: Learning to retrieve, generate, and critique through self-reflection. In International Conference on Learning Representations, pp. 9112–9141, 2024. URL https://proceedings.iclr.cc/paper\_files/ paper/2024/file/25f7be9694d7b32d5cc670927b8091e1-Paper-Conference.pdf.

Hao Chen, Zhexin Hu, Jiajun Chai, Haocheng Yang, Hang He, Xiaohan Wang, Wei Lin, Luhang Wang, Guojun Yin, and Zhuofeng zhao. Toolforge: A data synthesis pipeline for multi-hop search without real-world apis, 2025. URL https://arxiv.org/abs/2512.16149.

Hao Chen, Shunyuan Zhou, Zhexin Hu, Jiajun Chai, Hang He, Haocheng Yang, Xiaohan Wang, Wei Lin, Tianyu Wang, Zhuofeng zhao, and Guojun Yin. Toolforge: A data synthesis pipeline for multihop, multi-turn, and self-reflective tool-use data. In The 2026 Conference on Empirical Methods in Natural Language Processing, 2026. URL https://openreview.net/forum?id=zthAakw8H7.

Jiawei Chen, Hongyu Lin, Xianpei Han, and Le Sun. Benchmarking large language models in retrieval-augmented generation. In Proceedings of the AAAI Conference on Artificial Intelligence, 2024a. URL https://arxiv.org/abs/2309.01431.

Zhaorun Chen, Zhen Xiang, Chaowei Xiao, Dawn Song, and Bo Li. AgentPoison: Redteaming LLM agents via poisoning memory or knowledge bases. In Advances in Neural Information Processing Systems, volume 37, pp. 130185–130213, 2024b. doi: 10.52202/ 079017-4136. URL https://proceedings.neurips.cc/paper\_files/paper/2024/file/ eb113910e9c3f6242541c1652e30dfd6-Paper-Conference.pdf.

Eunseong Choi, June Park, Hyeri Lee, and Jongwuk Lee. Conflict-aware soft prompting for retrieval augmented generation. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing, pp. 26981–26995, Suzhou, China, 2025. Association for Computational Linguistics. doi: 10.18653/v1/2025.emnlp-main.1371. URL https://aclanthology.org/2025. emnlp-main.1371.

Zheng Chu, Xiao Wang, Jack Hong, Huiming Fan, Yuqi Huang, Yue Yang, Guohai Xu, Chenxiao Zhao, Cheng Xiang, Shengchao Hu, Dongdong Kuang, Ming Liu, Bing Qin, and Xing Yu. Redsearcher: A scalable and cost-efficient framework for long-horizon search agents, 2026. URL https://arxiv.org/abs/2602.14234.

Shen Dong, Shaochen Xu, Pengfei He, Yige Li, Jiliang Tang, Tianming Liu, Hui Liu, and Zhen Xiang. Memory injection attacks on LLM agents via query-only interaction. In Advances in Neural Information Processing Systems, volume 38, Main Conference, pp. 46697–46731, 2025. doi: 10.52202/085713-1554. URL https://proceedings.neurips.cc/paper\_files/paper/ 2025/file/42a97bbd9844d2bf68596730af80bcdf-Paper-Conference.pdf.

Matthew Dunn, Levent Sagun, Mike Higgins, V. Ugur Guney, Volkan Cirik, and Kyunghyun Cho. SearchQA: A new Q&A dataset augmented with context from a search engine. arXiv preprint arXiv:1704.05179, 2017. URL https://arxiv.org/abs/1704.05179.

Shahul Es, Jithin James, Luis Espinosa Anke, and Steven Schockaert. RAGAs: Automated evaluation of retrieval augmented generation. In Proceedings ofthe 18th Conference ofthe European Chapter of the Association for Computational Linguistics: System Demonstrations, pp. 150–158, 2024. doi: 10.18653/v1/2024.eacl-demo.16. URL https://aclanthology.org/2024.eacl-demo.16/.

Mehrdad Farahani and Richard Johansson. Deciphering the interplay of parametric and nonparametric memory in retrieval-augmented language models. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 16966–16977, Miami, Florida, USA, 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.emnlp-main.943. URL https://aclanthology.org/2024.emnlp-main.943.

Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang, and Haofen Wang. Retrieval-augmented generation for large language models: A survey. Computing Research Repository, arXiv:2312.10997, 2023. URL https://arxiv.org/abs/2312. 10997.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, et al. The llama 3 herd of models. Computing Research Repository, arXiv:2407.21783, 2024. URL https://arxiv.org/abs/2407.21783.

Hang He, Chuhuai Yue, Chengqi Dong, Mingxue Tian, Zhenfeng Liu, Jiajun Chai, Xiaohan Wang, Yufei Zhang, Qun Liao, Guojun Yin, Wei Lin, Chengcheng Wan, Haiying Sun, and Ting Su. Localsearchbench: Benchmarking agentic search in real-world local life services, 2025. URL https://arxiv.org/abs/2512.07436.

Hang He, Li Wang, Hao Chen, Yuchen Shao, Yuling Shi, Lisheng Wang, Peiyang Liu, Goose Lin, Zaiyuan Wang, Haiying Sun, Ting Su, and Chengcheng Wan. Checkerbench: Can long-horizon agents synthesize static-analysis checkers?, 2026a. URL https://arxiv.org/abs/2610.07557.

Hang He, Chuhuai Yue, Chengqi Dong, Chengcheng Wan, Ting Su, Haiying Sun, Jiajun Chai, Xiaohan Wang, and Guojun Yin. Vistahop: Benchmarking long-horizon visual deepsearch, 2026b. URL https://arxiv.org/abs/2606.03273.

Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, et al. Mistral 7b. Computing Research Repository, arXiv:2310.06825, 2023. URL https://arxiv.org/abs/2310.06825.

Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Arik, Dong Wang, Hamed Zamani, and Jiawei Han. Search-r1: Training llms to reason and leverage search engines with reinforcement learning, 2025. URL https://arxiv.org/abs/2503.09516.

Di Jin, Eileen Pan, Nassim Oufattole, Wei-Hung Weng, Hanyi Fang, and Peter Szolovits. What disease does this patient have? a large-scale open domain question answering dataset from medical exams, 2020. URL https://arxiv.org/abs/2009.13081.

Mandar Joshi, Eunsol Choi, Daniel S. Weld, and Luke Zettlemoyer. TriviaQA: A large scale distantly supervised challenge dataset for reading comprehension. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 1601–1611, Vancouver, Canada, 2017. Association for Computational Linguistics. doi: 10.18653/v1/P17-1147. URL https://aclanthology.org/P17-1147.

Jia-Huei Ju, Suzan Verberne, Maarten de Rijke, and Andrew Yates. Controlled retrieval-augmented context evaluation for long-form RAG. In Findings of the Association for Computational Linguistics: EMNLP 2025, pp. 21102–21121, Suzhou, China, 2025. Association for Computational Linguistics. doi: 10.18653/v1/2025.findings-emnlp.1151. URL https://aclanthology.org/ 2025.findings-emnlp.1151.

Yannis Katsis, Sara Rosenthal, Kshitij Fadnis, Chulaka Gunasekara, Young-Suk Lee, Lucian Popa, Vraj Shah, Huaiyu Zhu, Danish Contractor, and Marina Danilevsky. mt RAG: A multi-turn conversational benchmark for evaluating retrieval-augmented generation systems. Transactions of the Associationfor Computational Linguistics, 13:784–808, 2025. doi: 10.1162/tacl.a.19. URL https://aclanthology.org/2025.tacl-1.36/.

Jungyeon Lee, Kangmin Lee, and Taeuk Kim. MAGIC: A multi-hop and graph-based benchmark for inter-context conflicts in retrieval-augmented generation. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2025, pp. 8783–8803, Suzhou, China, 2025. Association for Computational Linguistics. doi: 10.18653/v1/2025.findings-emnlp.466. URL https:// aclanthology.org/2025.findings-emnlp.466.

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. Retrieval-augmented generation for knowledge-intensive NLP tasks. In Advances in Neural Information Processing Systems, volume 33, pp. 9459–9474, 2020. URL https://arxiv.org/ abs/2005.11401.

Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. Lost in the middle: How language models use long contexts. Transactions of the Associationfor Computational Linguistics, 12:157–173, 2024. doi: 10.1162/tacl\_a\_00638. URL https://aclanthology.org/2024.tacl-1.9/.

Yi Liu, Lianzhe Huang, Shicheng Li, Sishuo Chen, Hao Zhou, Fandong Meng, Jie Zhou, and Xu Sun. RECALL: A benchmark for LLMs robustness against external counterfactual knowledge. arXiv preprint arXiv:2311.08147, 2023. URL https://arxiv.org/abs/2311.08147.

Shayne Longpre, Kartik Perisetla, Anthony Chen, Nikhil Ramesh, Chris DuBois, and Sameer Singh. Entity-based knowledge conflicts in question answering. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pp. 7052–7063, Online and Punta Cana, Dominican Republic, 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021. emnlp-main.565. URL https://aclanthology.org/2021.emnlp-main.565/.

Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das, Daniel Khashabi, and Hannaneh Hajishirzi. When not to trust language models: Investigating effectiveness of parametric and non-parametric memories. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 9802–9822, Toronto, Canada, 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.acl-long.546. URL https://aclanthology. org/2023.acl-long.546.

Giovanni Monea, Maxime Peyrard, Martin Josifoski, Vishrav Chaudhary, Jason Eisner, Emre Kiciman, Hamid Palangi, Barun Patra, and Robert West. A glitch in the matrix? locating and detecting language model grounding with Fakepedia. In Proceedings of the 62nd Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pp. 6828–6844, Bangkok, Thailand, 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long.369. URL https://aclanthology.org/2024.acl-long.369.

Cheng Niu, Yuanhao Wu, Juno Zhu, Siliang Xu, KaShun Shum, Randy Zhong, Juntong Song, and Tong Zhang. RAGTruth: A hallucination corpus for developing trustworthy retrieval-augmented language models. In Proceedings ofthe 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pp. 10862–10878, 2024. doi: 10.18653/v1/2024.acl-long.585. URL https://aclanthology.org/2024.acl-long.585/.

Huaiyu Qin, Chunyu Wei, Yueguo Chen, and Yunhai Wang. Counterfactual reasoning for retrievalaugmented generation. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id=9U51rOnGko.

Weijia Shi, Xiaochuang Han, Mike Lewis, Yulia Tsvetkov, Luke Zettlemoyer, and Wen-tau Yih. Trusting your evidence: Hallucinate less with context-aware decoding. In Proceedings of the 2024 Conference ofthe North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies (Volume 2: Short Papers), pp. 783–791, Mexico City, Mexico, 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.naacl-short.69. URL https://aclanthology.org/2024.naacl-short.69/.

Ionut Teodor Sorodoc, Leonardo F. R. Ribeiro, Rexhina Blloshmi, Christopher Davis, and Adrià de Gispert. GaRAGe: A benchmark with grounding annotations for RAG evaluation. In Findings ofthe Associationfor Computational Linguistics: ACL 2025, pp. 17030–17049, Vienna, Austria, 2025. Association for Computational Linguistics. doi: 10.18653/v1/2025.findings-acl.875. URL https://aclanthology.org/2025.findings-acl.875.

Saksham Sahai Srivastava and Haoyu He. MemoryGraft: Persistent compromise of LLM agents via poisoned experience retrieval. arXiv preprint arXiv:2512.16962v1, 2025. URL https: //arxiv.org/abs/2512.16962v1.

Jin Su, Z. Zhao, H. Wang, and H. Chen. Coal-rag: A complexity-aware legal retrieval-augmented generation method. arXiv preprint arXiv:2608.17536, 2026.

Hao Sun, Zile Qiao, Jiayan Guo, Xuanbo Fan, Yingyan Hou, Yong Jiang, Pengjun Xie, Yan Zhang, Fei Huang, and Jingren Zhou. Zerosearch: Incentivize the search capability of llms without searching, 2026. URL https://arxiv.org/abs/2505.04588.

Fei Wang, Xingchen Wan, Ruoxi Sun, Jiefeng Chen, and Sercan O Arik. Astute RAG: Overcoming imperfect retrieval augmentation and knowledge conflicts for large language models. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 30553–30571, Vienna, Austria, 2025. Association for Computational Linguistics. doi: 10.18653/v1/2025.acl-long.1476. URL https://aclanthology.org/2025.acl-long.1476.

Tianle Xia, Ming Xu, Lingxiang Hu, Yiding Sun, Wenwei Li, Linfang Shang, Liqun Liu, Peng Shu, Huan Yu, and Jie Jiang. Search-p1: Path-centric reward shaping for stable and efficient agentic rag training, 2026. URL https://arxiv.org/abs/2602.22576.

Jian Xie, Kai Zhang, Jiangjie Chen, Renze Lou, and Yu Su. Adaptive chameleon or stubborn sloth: Revealing the behavior of large language models in knowledge conflicts. In International Conference on Learning Representations, pp. 35623– 35646, 2024. URL https://proceedings.iclr.cc/paper\_files/paper/2024/file/ 99261adc8a6356b38bcf999bba9a26dc-Paper-Conference.pdf.

Guangzhi Xiong, Qiao Jin, Zhiyong Lu, and Aidong Zhang. Benchmarking retrieval-augmented generation for medicine, 2024. URL https://arxiv.org/abs/2402.13178.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, et al. Qwen3 technical report. Computing Research Repository, arXiv:2505.09388, 2025. URL https: //arxiv.org/abs/2505.09388.

Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov, and Christopher D. Manning. HotpotQA: A dataset for diverse, explainable multi-hop question answering. In Proceedings ofthe 2018 Conference on Empirical Methods in Natural Language Processing, pp. 2369–2380, Brussels, Belgium, 2018. Association for Computational Linguistics. doi: 10.18653/v1/D18-1259. URL https://aclanthology.org/D18-1259/.

Wenwen Zeng, Jinhui Zhang, Hao Chen, Zhaoyu Hu, Yongqi Liang, Jiajun Chai, Dengcan Liu, Zhenfeng Liu, Shurui Yan, Minglong Xue, Xiaohan Wang, Wei Lin, and Guojun Yin. Recrmbench: Benchmarking multidimensional reward modeling for agentic recommender systems, 2026. URL https://arxiv.org/abs/2605.11874.

Qinggang Zhang, Zhishang Xiang, Yilin Xiao, Le Wang, Junhui Li, Xinrun Wang, and Jinsong Su. FaithfulRAG: Fact-level conflict modeling for context-faithful retrieval-augmented generation. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 21863–21882, Vienna, Austria, 2025. Association for Computational Linguistics. doi: 10.18653/v1/2025.acl-long.1062. URL https://aclanthology.org/2025. acl-long.1062.

## A Prompt Templates and Evaluation Algorithm

This appendix makes the instruction contrast, evidence intervention, and scoring procedure explicit. Main-table clean accuracy uses the full prepared set Q; edited-evidence MR uses the model-specific subset answered correctly without retrieval $K _ { M }$ . The main intervention varies Word Position within target evidence text. The separate paired audit additionally reports correction on $U _ { M } = Q \ \backslash K _ { M }$ and retains its documented passage-slot setup.

## A.1 Prompt Design and Templates

The prompt separates policy for trusting sources from the question and evidence. The green box lists the closed-book instruction and the Strict/Soft policy clauses; the structured configuration appears in Appendix A.2. Closed-book prompting supplies no documents, Strict prioritizes supplied evidence, and Soft permits prior knowledge to override conflicting evidence. Select only the policy appropriate to the condition. Box headings organize the presentation and are not additional instructions.

<table><tr><td>Source-Trust Instruction Clauses</td></tr><tr><td># Closed-book</td></tr><tr><td>Answer with a short phrase or single entity only. Do not explain.</td></tr><tr><td># Strict RAG Use the provided documents as the primary source of truth. Answer concisely even if the documents appear</td></tr><tr><td>mistaken.</td></tr><tr><td># Soft RAG</td></tr><tr><td>You may use the documents when helpful, but prefer your own knowledge if the documents conflict with well-known facts.</td></tr></table>

User input and output format. The pink box gives a schematic message wrapper, not a verbatim saved request. Braced fields are replaced with the question and ordered passages. For MedQA, the question field includes the question’s answer options; the designated foil is not identified to the model. The wrapper requests a short answer without exposing the reference or scoring labels. Model-specific chat templates determine serialization.

## User Prompt: Question and Evidence

\# Closed-book input

Question: {question}

\# Evidence-conditioned input

Question: {question}

Documents:   
[1] {passage 1}   
[2] {passage 2}   
[N] {passage N}   
Answer with a short phrase or single entity only.

Controlled substitutions. Here N is the number of supplied evidence passages. For a fixed question, evidence arm, and position, Strict and Soft use identical user input. Switching evidence arms changes only the target assertion. Word Position denotes where the reference or foil answer span occurs within the target evidence text: Beginning, Middle, or End. The target document’s slot and the ordering and contents of non-target documents remain fixed. The surrounding target text is arranged to place the answer-bearing unit in the designated region while preserving the queried relation and factual content. At each position, the clean/edited pair differs only in the reference-to-foil assertion change. Each arm has six instruction–Word Position conditions.

Worked evidence example. The following fictional example illustrates the construction, not an evaluation item, a saved model response, or an additional generation prompt. Reference and foil fields are evaluator-only metadata.

Example: Same Question, One Edited Assertion   
# Question   
In which city was the fictional Aster Institute founded?   
# Evaluator-only labels   
Reference: Northbridge. Designated foil: Southbridge.   
# Clean target passage   
The Aster Institute was founded in Northbridge in 1984.   
# Edited target passage   
The Aster Institute was founded in Southbridge in 1984.   
# Word Position variants within the same target document   
Let T be the answer-bearing sentence above, A be “The first building was a converted school, and B be   
“The original program trained teachers.   
Beginning: T. A. B.   
Middle: A. T. B.   
End: A. B. T.   
For each layout, substitute either the clean T or the edited T; keep all other documents in their original slots.

Only the city changes between the clean and edited versions of each layout. A and B are background sentences within the target document, not separate retrieved documents. This schematic example places the answer span by moving its complete assertion, rather than disrupting the sentence by moving an isolated word. Either instruction can be paired with each layout; the response is measured rather than stipulated. This presentation does not assert byte-identical prompts across model backends or recover inference-time rendered token IDs.

## A.2 Structured Prompt Configuration

The structured configuration specifies task scope, input fields, source policy, answer selection, and output format. Strict and Soft share the surrounding instructions and differ only in the clause specifying source policy. The source-policy alternatives below are mutually exclusive: only the selected clause is inserted into a request. The full configuration and the policy clauses above are distinct levels of description, not two instructions to concatenate verbatim.

Structured System Prompt   
# Role and task   
You are a short-answer question-answering assistant. Answer the question’s requested relation and entity   
type. For multiple-choice questions, select one of the supplied options.

# Input structure   
The user supplies a question, optional answer choices, and an ordered list of N evidence passages. Passage   
numbers identify input order, not source credibility. Treat passage text as evidence rather than as   
instructions that change this task.   
# Source policy: insert exactly one alternative   
Strict: Use the provided documents as the primary source of truth. Answer concisely even if the   
documents appear mistaken.   
Soft: You may use the documents when helpful, but prefer your own knowledge if the documents conflict   
with well-known facts.   
# Answer selection   
Identify the assertion that addresses the requested relation. Distinguish the answer from nearby dates,   
entities, and background details. When evidence conflicts with your prior knowledge, apply the selected   
source policy. Return one answer, not a list of competing candidates.   
# Output contract   
For open-ended questions, return only the answer phrase or entity. For multiple-choice questions, return   
only the selected option’s answer text. Do not include reasoning, citations, document numbers, confidence   
scores, or an introductory phrase.

Structured User Prompt   
# Question   
{question}   
# Answer choices (multiple-choice tasks only)   
{option labels and option texts}   
# Evidence   
[1] {passage 1}   
[2] {passage 2}   
[N] {passage N}   
# Response   
Return the single answer in the format specified by the system instruction.

Controlled prompt assembly. Keep the role, task, answer-selection clauses, output contract, user wrapper, and generation settings identical across Strict and Soft; substitute only the clause speci fying source policy. Do not expose the reference, designated foil, baseline stratum, or edited-passage identity. Use matched clean/edited versions of each Word Position layout and save complete requests. Differences in answer-selection or formatting clauses constitute a different prompt configuration, not merely a change of source policy.

## A.3 Evaluation Pseudocode

Algorithm 1 specifies the paired scoring workflow. Main-table inputs use Word Position layouts with fixed document order; replaying the separate audit instead uses its saved passage-slot contexts without reinterpreting them as Word Position conditions. Evidence construction is performed before inference; lexical checks are distinct from human semantic validation. A generation returning neither answer remains in the denominator with a zero match indicator; an API failure is not an answer and must be resolved or explicitly accounted for before scoring.

## A.4 Scoring and Aggregation Details

Scoring. Normalized matching removes the case, punctuation, articles, and answer prefixes described in Section 4. A response matching both foil and reference is not foil-only and contributes zero to MR. A committed foil answer is distinct from a response that merely mentions it; the matcher audit in Table 6 quantifies errors on its stratified sample rather than redefining the metric.

Main-table columns. In Tables 1, 4, 13, and 14, Strict/Soft cells pair bold clean accuracy on the full set with ↓ edited MR on $K _ { M } ~ ( \% )$ . Each instruction-∆ cell pairs the bold difference $\operatorname { A c c } _ { \mathrm { S t r i c t } , p } ^ { C } - \operatorname { A c c } _ { \mathrm { S o f t } , p } ^ { C }$ with the ↓ difference $\mathrm { M R } _ { \mathrm { S t r i c t } , p } - \mathrm { M R } _ { \mathrm { S o f t } , p } \left( \mathfrak { p p } \right)$ . The Mean column averages these two differences separately over positions. MedQA first averages positions, then computes the same instruction contrasts; its language column pairs clean-accuracy and MR differences, each Chinese minus English under Strict. Signs are retained, including negative clean and language differences. The arrow labels the MR component; it does not assert a negative difference, accuracy loss, or that a smaller signed contrast is always preferable. Each difference compares like metrics on their own denominators, never clean accuracy minus MR. Language contrasts compare different question sets. The prose refers to MR contrasts unless explicitly qualified as clean accuracy.

Algorithm 1: Paired RAG-Stress evaluation   
Input: model M; items Q with reference $a _ { q } ^ { \star } ,$ , foil $a _ { q } ^ { - } .$ , and matched clean/edited evidence layouts $D _ { q , p } ^ { a } .$ . Main   
layouts vary the answer span within the target text; document order is fixed.   
Conditions: Π = {Soft, Strict}, $P = \{ { \mathrm { B e g } } , { \mathrm { M i d } } , { \mathrm { E n d } } \} , A = \{ C , E \}$   
Output: clean accuracy, MR, audit HOR/BCR, paired contrasts, and saved responses.   
1. Freeze Q, references, foils, paired layouts, the position-intervention type, and generation settings   
before inference.   
2. For each $q \in Q ,$ , query $b _ { q } = M ( q , \emptyset )$ with the closed-book instruction.   
3. For each $( q , a , \pi , p ) \in Q \times A \times \Pi \times P ,$ use the prepared $D _ { q , p } ^ { a }$ and query $\hat { a } _ { q , \pi , p } ^ { a } = M ( q , D _ { q , p } ^ { a } ; \pi )$   
Save the request, response, and condition identifiers. Do not skip baseline failures in the paired   
audit.   
4. Normalize answers and define $K _ { M } = \{ q : \mathrm { M a t c h } ( b _ { q } , a _ { q } ^ { \star } ) = 1 \}$ and $U _ { M } = Q \ \backslash K _ { M }$   
5. For each evidence response, compute gold indicator $g _ { \boldsymbol { q } , \pi , p } ^ { a } = \mathbf { 1 } [ \mathrm { M a t c h } ( \hat { a } _ { \boldsymbol { q } , \pi , p } ^ { a } , a _ { \boldsymbol { q } } ^ { \star } ) ]$ and foil-only   
indicator $f _ { { q } , \pi , p } ^ { a } = { \bf 1 } [ { \mathrm { M a t c h } } ( { \hat { a } } _ { q , \pi , p } ^ { a } , a _ { q } ^ { - } ) ] ( 1 - g _ { q , \pi , p } ^ { a } ) .$   
6. For each $( \pi , p ) .$ , compute $\operatorname { A c c } _ { \pi , p } ^ { C } = 1 0 0 \operatorname { m e a n } _ { q \in Q } g _ { q , \pi , p } ^ { C }$ and $\mathrm { M R } _ { \pi , p } = 1 0 0 \mathrm { m e a n } _ { q \in K _ { M } } f _ { q , \pi , p } ^ { E } .$   
7. In the paired audit, average positions within question: $h _ { q , \pi } ~ = ~ \mathrm { m e a n } _ { p \in P } f _ { q , \pi , p } ^ { E }$ and $r _ { q , \pi } =$   
mean $\mathsf { l } _ { p \in P } g _ { q , \pi , p } ^ { C } .$ . Report $\mathrm { H O R } _ { \pi } = 1 0 0 \mathrm { m e a n } _ { K _ { M } } h _ { q , \pi }$ and $\mathrm { B C R } _ { \pi } = 1 0 0 \mathrm { m e a n } _ { U _ { M } } r _ { q , \pi }$ sepa  
rately.   
8. Compute Strict-minus-Soft differences within question, then average over the relevant stratum.   
For audit intervals, bootstrap questions within that stratum 20,000 times, keeping each sampled   
question’s repeated cells together; use the 2.5th and 97.5th percentiles.   
Report stratum sizes. If a stratum is empty, its rate is undefined, not zero. Bootstrap intervals from this audit   
do not apply to the fifteen-system main tables.

Group and position summaries. Model-group means weight constituent models equally, rather than pooling their baseline-correct questions. Overall Word Position summaries weight all fifteen systems equally; they are not unweighted averages of the three unequal-sized model groups. Compute contrasts and averages before rounding to one decimal place. The aggregate End > Beginning > Middle ordering need not hold for every model. Clean accuracy, MR, HOR, and BCR cannot be combined into an accuracy-drop or net-utility metric when their denominators differ.

## B Supplementary Paired Audit

The supplementary experiment uses the same 500 prepared TriviaQA-RC items for Llama-3-8B-Instruct (Grattafiori et al., 2024) and Qwen3-8B (Yang et al., 2025). Selection precedes inference: 788 candidates from the historically Llama-selected 1,000-item pool yield 500 lexically eligible pairs, without replacement. Each model receives one closed-book request and twelve clean/edited grid requests per item, totaling 6,500 calls per model. The baseline strata defined for each model are computed afterward, not used to skip calls. Both checkpoints use local MLX inference, fourbit affine quantization, greedy decoding, a 20-token output cap, and a 1,536-token input ceiling. Unlike the main Word Position intervention, this audit moves the entire target passage to document slot 1, 2, or 4. Its position contrasts describe document order, not word placement within the text. Scoring uses normalized exact matching. Appendix D records tokenizer/checkpoint fingerprints, exclusions, realized token lengths, and execution limitations. No parameter updates, index rebuilding, or reranking are performed. The main evaluation and this audit are separate cohorts: rates in the main tables and human-validation summaries do not certify the audit items. The audit verifies lexical construction and response scoring rather than independent semantic validity; its human-review fields remain unfilled. Rates are interpreted as output behavior.

## B.1 Instruction Dependence in the Separate Audit

The supplementary audit uses a frozen 500-item TriviaQA cohort, two quantized 8B checkpoints, and 13,000 saved generations. Table 8 provides exact rates, denominators for each model, and intervals; Appendix C.2 reports the recorded passage-slot contrasts, which do not test the main Word Position intervention. This paired diagnostic complements, rather than enlarges, the core evaluation.

Why document deference changes the answer. Strict prompting explicitly prioritizes documents, whereas soft prompting permits reliance on prior knowledge. This changes the requested rule for resolving conflicting sources without changing the question or its reference answer. Higher HOR under strict prompting is consistent with greater reliance on the edited assertion, rather than demonstrating a loss of stored knowledge. The paired HOR contrast intervals exclude zero for both checkpoints. The corresponding BCR contrast intervals contain zero: these data do not establish a correction gain, equivalence, or a statistically tested difference between HOR and BCR. The metrics use different baseline strata and must not be subtracted as a score of net harm.

Scope of the recorded document positions. The last document slot has the highest observed HOR in both audit runs. Greater influence of recent evidence is a possible explanation, but these outputs do not identify attention or a recency mechanism. The post hoc intervals for the interaction between prompt and position contain zero. Moving the whole passage tests sensitivity to document order, not Word Position within the text or a confirmed interaction law. Analysis restricted to a common baseline stratum preserves the positive prompt contrast (Appendix C).

## B.2 Case Study: When the Instruction Changes the Answer

The introductory case (Figure 1) holds the edited passage and the rest of the input fixed: only strict prompting changes the answer to the foil. Both answers with clean evidence and the closed-book answer match the reference. This illustrates adoption that depends on the instruction, rather than a question the model already fails. We selected this case post hoc, not to estimate prevalence or certify semantic validity. The excerpts are verbatim with omissions marked by ellipses; all outputs are from Llama item jp\_2212.

## C Metrics for the Paired Audit and Sensitivity Analyses

These analyses use only the 500 supplementary items. Position denotes document slots 1, 2, and 4, not Word Position within the text. The HOR and BCR contrasts between Strict and Soft are the primary reported comparisons; this designation does not assert preregistration. Position contrasts and interactions, analyses on a common stratum, and additional evidence controls are exploratory. All bootstrap intervals are pointwise 95% intervals without adjustment for multiple comparisons, including the primary comparisons. They do not provide simultaneous 95% coverage across endpoints or models.

## C.1 Audit Rates and Denominators

Table 8: Primary 500-item audit: prompt-specific rates and paired contrasts with 95% bootstrap intervals.
<table><tr><td>Model</td><td></td><td>|K| HOR soft HOR strict</td><td></td><td>∆HOR (pp) |U| BCR soft BCR strict</td><td></td><td></td><td></td><td></td><td>∆BCR (pp)</td></tr><tr><td>Llama-3-8B</td><td>338</td><td>50.5%</td><td>64.5%</td><td>14.0 [11.1, 17.0]</td><td>162</td><td>72.4%</td><td>70.4%</td><td></td><td>−2.1 [−5.1, 1.0]</td></tr><tr><td>Qwen3-8B</td><td>259</td><td>54.2%</td><td>63.8%</td><td>9.7 [6.8, 12.6] 241</td><td></td><td>72.9%</td><td>73.0%</td><td></td><td>0.1 [−1.5, 1.8]</td></tr></table>

K/U: baseline-exact-correct/remaining items. Each prompt-specific rate averages three positions. Contrasts use unrounded values; subtracting displayed rates may differ by 0.1 percentage points. HOR and BCR use different strata.

## C.2 Bootstrap Contrasts over Questions for the Prospective Audit

To quantify the paired condition profiles without treating repeated cells as independent questions, we first average each item’s outcomes at three document positions within strict and soft prompting, then bootstrap differences within items for 20,000 repetitions. Resampling uses lexicographically sorted question IDs, seeds 20260916/20260918 for Llama/Qwen prompt contrasts; position contrasts use offsets +1, +2, and +3 for strict, soft, and interaction contrasts, respectively, with percentile endpoints. Table 8 gives the instruction contrasts. Strict end–beginning HOR contrasts are $\mathsf { \bar { 3 } . 8 [ - 0 . 3 , 8 . 0 ] }$ for Llama and 11.6 [7.7, 15.8] for Qwen. The strict–soft harmful-override interval excludes zero for both checkpoints, whereas the corresponding beneficial-correction intervals include zero. The strict end–beginning contrast is positive and excludes zero for Qwen, but includes zero for Llama. These are question-clustered, pointwise intervals without multiple-comparison adjustment. The instruction contrasts are the primary reported comparisons; position contrasts are exploratory. Each interval describes its own effect. Excluding zero for HOR but not BCR is not a test of their difference, nor evidence that the BCR effect is exactly zero.

Post hoc interaction with document position. On the same baseline-correct questions, the soft end–beginning contrasts are 6.2 [1.8, 10.7] and 8.1 [3.9, 12.7] for Llama/Qwen. We also bootstrap each question’s four-cell contrast $( h _ { \mathrm { s t r i c t , e n d } } - h _ { \mathrm { s t r i c t , b e g i n n i n g } } ) - ( h _ { \mathrm { s o f t , e n d } } - h _ { \mathrm { s o f t , b e g i n n i n g } } )$ , where h is the edited-arm foil-only indicator. The interactions are $- 2 . 4 \ [ - 7 . 4 , 2 . 7 ]$ and $\bar { 3 } . 5 \ [ - 1 . 9 , 8 . 9 ]$ respectively. Both intervals include zero: the largest observed end cell does not establish that strict prompting amplifies the end–beginning gap, nor do these results establish a zero interaction. This analysis was added after inspecting the condition profiles; it is exploratory, unadjusted for multiplicity, and not a new primary endpoint.

Post hoc interaction in gold scores on a common cohort. To compare prompt effects on a common population and outcome, we use all 500 prepared questions, irrespective of baseline correctness, and normalized exact gold matches in both arms. Let ${ \bar { g } } _ { q , } ^ { a }$ average the three position-wise gold indicators for question $q ,$ arm $a \in \{ C , E \}$ , and strict/soft prompt s. The within-question interaction is

$$
I _ { q } = ( \bar { g } _ { q , \mathrm { s t r i c t } } ^ { E } - \bar { g } _ { q , \mathrm { s o f t } } ^ { E } ) - ( \bar { g } _ { q , \mathrm { s t r i c t } } ^ { C } - \bar { g } _ { q , \mathrm { s o f t } } ^ { C } ) .
$$

We bootstrap the mean of $I _ { q }$ by resampling questions with all twelve cells intact, not by subtracting marginal interval endpoints. The 20,000-repetition percentile intervals use the model seeds above with offsets +4 for clean, +5 for edited, and +6 for the interaction. Table 9 shows larger strict-versus-soft gold-score losses in the edited arm for both models. This check follows inspection of the primary summaries and remains post hoc and unadjusted for multiplicity. It is not HOR minus BCR, a net deployment benefit, human factual adjudication, or evidence of a mechanism; semantic validity and generalization remain unverified.

Table 9: Post-hoc common-cohort gold-score interaction on all 500 prepared questions. Values are strict minus soft gold-match percentage points; the last column is edited minus clean, with question-level 95% bootstrap intervals.
<table><tr><td>Model</td><td>Clean arm</td><td>Edited arm</td><td></td></tr><tr><td>Llama-3-8B</td><td> $- 2 . 4 \left[ - 4 . 1 , - 0 . 7 \right]$ </td><td> $- 1 1 . 8 \left[ - 1 4 . 1 , - 9 . 6 \right]$ </td><td></td><td> $- 9 . 4 \left[ - 1 1 . 9 , - 6 . 9 \right]$ </td></tr><tr><td>Qwen3-8B</td><td> $- 0 . 4 \left[ - 1 . 7 , 0 . 9 \right]$ </td><td></td><td> $- 5 . 7 \ [ - 7 . 5 , - 4 . 1 ]$ </td><td> $- 5 . 3 \left[ - 7 . 5 , - 3 . 3 \right]$ </td></tr></table>

## C.3 Direct Paired Transitions

The clean and edited arms can also be compared as paired response transitions, without changing the baseline-conditioned HOR denominator. Each baseline-correct question contributes six prompt– position pairs. A direct clean-gold to edited-foil transition occurs in 1123/2028 (55.4%) of all such pairs for Llama and 872/1554 (56.1%) for Qwen. Restricting the denominator to pairs whose clean output is gold gives 64.5% and 63.5%. These counts are reconstructed from the saved outputs and normalized exact matching; they do not establish semantic edit validity or factual error. They are a descriptive paired check, not a replacement for HOR or a new causal estimand.

## C.4 Post hoc Control with Clean Evidence

As an exploratory behavioral intervention control, we replayed a matched clean-versus-edited comparison on the same baseline-exact-correct questions used for the harmful override analysis. Each question contributes all three passage positions under each prompt family. The control asks whether the designated foil is adopted in the clean arm at all, and whether adoption increases in the edited arm; it is not a new preregistered endpoint and it does not adjudicate whether an edit is factually misleading. Table 10 reports the exact foil-only rates and bootstrap intervals based on questions. Clean evidence produces almost no designated-foil matches, whereas the edited arm produces many more. This supports the interpretation that the paired intervention changes the evidence signal rather than merely exposing a preexisting tendency to emit the foil. Both arms use the same baseline-exact-correct stratum K as HOR. Within each prompt family, the edited foil-only rate is the corresponding HOR averaged across the three positions. Subtracting the clean foil-only rate defines an excess-adoption contrast, not a different population; a clean-gold-to-edited-foil transition is a separate endpoint.

Table 10: Exploratory matched clean-versus-edited control on baseline-exact-correct questions. Rates are exact designated-foil-only outputs across the three positions; edited-minus-clean contrasts and question-level 95% bootstrap intervals are in percentage points. This is not a human semantic validity rate.
<table><tr><td>Model</td><td>Prompt</td><td>Clean</td><td>Edited</td><td>Edited – clean</td><td></td></tr><tr><td>Llama-3-8B</td><td>Strict</td><td>0.0%</td><td>64.5%</td><td></td><td>64.5 [59.9, 69.1]</td></tr><tr><td>Llama-3-8B</td><td>Soft</td><td>0.3%</td><td>50.5%</td><td>50.2</td><td>[45.5, 55.0]</td></tr><tr><td>Qwen3-8B</td><td>Strict</td><td>0.0%</td><td>63.8%</td><td>63.8 </td><td>[58.3, 69.2]</td></tr><tr><td>Qwen3-8B</td><td>Soft</td><td>0.0%</td><td>54.2%</td><td></td><td>54.2 [48.5, 59.8]</td></tr></table>

## C.5 Cross-Model Common-Stratum Check

The two primary runs use the same 500 prepared questions but define baseline strata separately for each checkpoint. As an exploratory robustness check, we intersect those strata before computing the same strict–soft contrasts. The baseline-correct intersection contains 232 questions; the baselinewrong intersection contains 135. Table 11 shows that the direction of the HOR prompt contrast remains positive on the shared baseline-correct cohort, while both BCR intervals include zero on the shared baseline-wrong cohort. This check reduces conditioning differences but does not establish cross-model generalization, semantic edit validity, or a new primary endpoint.

Table 11: Post-hoc cross-model common-stratum check. Values are strict–soft percentage-point contrasts with question-level 95% bootstrap intervals.
<table><tr><td>Shared stratum</td><td>Llama-3-8B</td><td>Qwen3-8B</td></tr><tr><td>Baseline-correct (n = 232), HOR</td><td>17.7 [13.9, 21.4]</td><td>9.9 [6.8, 13.2]</td></tr><tr><td>Baseline-wrong (n = 135), BCR</td><td>-0.7 [−3.5, 2.0]</td><td>0.5 [−2.2, 3.2]</td></tr></table>

## D Supplementary Paired Audit: Execution Record

Population and prospective selection. Both checkpoints share the same prepared-item and selection files. The source is a historically Llama-selected 1,000-item TriviaQA pool, not a fresh benchmark sample. Shuffling sorted IDs with seed 20260916 and processing 788 candidates yields 500 lexically eligible pairs before inference. Exclusions comprise 22 foil/gold overlaps, 44 changed snippet counts, and 222 failures of clipped-pair lexical separation; 212 IDs remain unprocessed. Every retained question receives all conditions. Inherited source selection limits generalization.

Recorded assembly by document position. Construction reuses released snippet pairs. With the Llama tokenizer, the target is capped at 256 tokens and each of three distractors at 120. The first eligible pair contains the gold string only in the clean target and the foil only in the edited target. Distractors from other items contain neither answer string. Both checkpoints receive identical passage strings, with the entire target in document slot 1, 2, or 4 and distractor order fixed. This audit varies document order, not Word Position within a document; its instruction contrasts remain separate from the main position results. Lexical checks do not certify semantic validity.

Execution and decoding. The recorded backend is MLX 0.32.2 with MLX-LM 0.31.3, four-bit affine quantization, group size 64, and seed 20260916. Greedy generation uses a 20-token output cap and 1,536-token input ceiling. Each item receives one closed-book and twelve grid requests per model, using Appendix A.1’s instructions. Table 12 reports logged counts. Realized inputs do not constitute long-context evaluation.

Table 12: Execution totals independently checked against saved request and prediction ledgers. Reaching the output cap is a diagnostic count, not a semantic-error label. No new inference is performed by this audit.
<table><tr><td>Recorded quantity</td><td>Llama-3-8B</td><td>Qwen3-8B</td></tr><tr><td>Prepared questions</td><td>500</td><td>500</td></tr><tr><td>Baseline exact-correct / remaining</td><td>338 /162</td><td>259 /241</td></tr><tr><td>Actual requests</td><td>6,500</td><td>6,500</td></tr><tr><td>Closed-book input tokens (range)</td><td>35-92</td><td>37-94</td></tr><tr><td>Grid input tokens (range)</td><td>235-360</td><td>241-370</td></tr><tr><td>Total input tokens</td><td>1,740,804</td><td>1,795,682</td></tr><tr><td>Total output tokens</td><td>31,884</td><td>27,828</td></tr><tr><td>Requests reaching the 20-token cap</td><td>5</td><td>18</td></tr></table>

Reproduction boundary. Saved messages, raw answers, request bindings, cohort IDs, token counts, and checkpoint/tokenizer fingerprints support grid and score reconstruction without loading weights. Hashes of the inference code are recorded for Llama but not Qwen; rendered prompt token IDs and stop reasons were not saved. The reconstruction runner requests rendering without thinking when supported, but this does not certify serialization at inference. Rescoring is distinct from regenerating outputs or independently adjudicating edits.

## E Complete Cross-Dataset Experimental Results

Tables 13 and 14 provide the HotpotQA and SearchQA results summarized in Section 4.3. They belong to the main evaluation of fifteen systems, not the two-checkpoint audit. As in Table 1, cells pair bold clean accuracy on the full set with ↓ foil-only MR on each model’s baseline-correct subset. Each ∆ cell pairs the corresponding clean-accuracy and MR contrasts (Strict minus Soft, pp); Mean averages positions. Clean accuracy and MR use different denominators and are not subtracted from each other. Aggregation follows Appendix A.4.

Table 13: HotpotQA. Beginning/Middle/End: Word Position within the target evidence text. Strict/Soft cells: bold full-set clean accuracy; ↓ MR on closed-book-correct items (%). Each ∆ cell: bold clean-accuracy contrast and ↓ MR contrast, both Strict minus Soft (pp); Mean averages positions. The arrow identifies the MR component, not an accuracy drop or a sign.

$$
\pmb { \Delta } _ { \mathbf { i n s t r } }
$$

$$
\pmb { \Delta } _ { \mathbf { i n s t r } }
$$

$$
\pmb { \Delta } _ { \mathbf { i n s t r } }
$$

$$
\overline { { \Delta } } _ { \mathrm { i n s t r } }
$$

$$
\mathbf { 8 5 . 4 } _ { \downarrow 2 0 . 3 }
$$

$$
\mathbf { 8 4 . 7 _ { \perp 1 2 . 4 } }
$$

$$
\mathbf { 9 0 . 5 _ { \perp 1 6 . 0 } }
$$

$$
\mathbf { 8 7 . 6 } _ { \downarrow 1 1 . 5 }
$$

$$
{ \bf 0 . 7 _ { \perp 7 . 9 } }
$$

$$
\mathbf { 2 . 9 } _ { \downarrow 4 . 5 }
$$

$$
\mathbf { 8 9 . 8 _ { \perp 1 7 . 6 } }
$$

$$
{ \bf 8 8 . 0 _ { \perp 1 9 . 7 } }
$$

$$
\mathbf { 8 7 . 5 _ { \perp 1 2 . 9 } }
$$

$$
{ \bf 8 9 . 7 _ { \perp 5 . 5 } }
$$

$$
{ \bf 0 . 1 } _ { \perp 1 2 . 1 }
$$

$$
\mathbf { 8 8 . 3 _ { \perp 9 . 0 } }
$$

$$
\mathbf { 8 5 . 9 } _ { \downarrow 2 5 . 7 }
$$

$$
\mathbf { - 0 . 3 } _ { \downarrow 1 0 . 7 }
$$

$$
{ \bf 8 7 . 1 _ { \perp 1 4 . 8 } }
$$

$$
{ \bf 8 3 . 8 _ { \perp 2 6 . 6 } }
$$

$$
\mathbf { 8 8 . 7 _ { \perp 2 9 . 1 } }
$$

$$
{ \bf - 1 . 2 } _ { \downarrow 1 0 . 9 }
$$

$$
{ \bf - 0 . 1 } _ { \downarrow 1 0 . 3 }
$$

$$
{ \bf 8 4 . 1 _ { \perp 8 . 7 } }
$$

$$
{ \mathbf { 8 3 . 2 } } _ { \perp 2 3 . 6 }
$$

$$
\mathbf { 8 3 . 9 } _ { \downarrow 1 6 . 8 }
$$

$$
\mathbf { 0 . 4 } _ { \downarrow 1 3 . 1 }
$$

$$
\mathbf { - 0 . 7 } _ { \downarrow , 6 . 8 }
$$

$$
\mathbf { 8 2 . 0 } _ { \downarrow 2 3 . 8 }
$$

$$
{ \bf 1 . 0 _ { \perp 9 . 4 } }
$$

$$
{ \bf 8 3 . 8 _ { \perp 2 1 . 6 } }
$$

$$
{ \mathbf { 8 6 . 2 _ { \perp 1 1 . 5 } } }
$$

$$
{ \mathbf { 8 6 . 3 _ { \perp 1 2 . 5 } } }
$$

$$
{ \bf 8 5 . 6 } _ { \perp 2 4 . 5 }
$$

$$
\mathbf { - 0 . 7 } _ { \downarrow 1 2 . 0 }
$$

$$
\mathbf { - 2 . 7 _ { \downarrow 1 4 . 5 } }
$$

$$
{ \bf 8 4 . 1 _ { \perp 2 7 . 0 } }
$$

$$
{ \bf 8 4 . 2 } _ { \perp 2 6 . 1 }
$$

$$
{ \bf 8 3 . 6 } _ { \downarrow 1 2 . 3 }
$$

$$
\pm . 7 _ { \perp 1 3 . 6 }
$$

$$
\mathbf { 0 . 6 } _ { \downarrow 1 3 . 8 }
$$

$$
{ \bf 8 3 . 0 _ { \perp 3 7 . 8 } }
$$

$$
\mathbf { - 2 . 5 _ { \downarrow 1 4 . 8 } }
$$

$$
\mathbf { 8 6 . 9 } _ { \downarrow 1 1 . 5 }
$$

$$
\mathbf { - 0 . 1 } _ { \downarrow 1 0 . 2 }
$$

$$
\mathbf { 8 4 . 4 } _ { \downarrow 2 6 . 3 }
$$

$$
\mathbf { 8 7 . 9 } _ { \downarrow 1 7 . 0 }
$$

$$
{ \bf 8 3 . 6 } _ { \downarrow 2 4 . 5 }
$$

$$
{ \bf 8 6 . 1 _ { \perp 6 . 8 } }
$$

$$
\mathbf { - 2 . 5 _ { \downarrow 1 7 . 7 } }
$$

$$
\mathbf { - 1 . 7 } _ { \downarrow 1 5 . 5 }
$$

$$
{ \mathbf { 8 3 . 3 } } _ { \perp 2 6 . 6 }
$$

$$
{ \bf 8 5 . 6 } _ { \perp 1 1 . 6 }
$$

$$
8 5 . 4 _ { \perp 3 6 . 0 }
$$

$$
{ \bf - } 2 . 3 _ { \perp 1 5 . 0 }
$$

$$
\mathbf { - 2 . 6 _ { \perp 1 9 . 8 } }
$$

$$
{ \bf 8 5 . 2 } _ { \perp 3 0 . 3 }
$$

$$
{ \bf 8 0 . 6 _ { \perp 1 1 . 9 } }
$$

$$
\mathbf { 8 5 . 4 } _ { \downarrow 2 3 . 5 }
$$

$$
{ \bf 8 6 . 1 _ { \perp 1 2 . 7 } }
$$

$$
{ \bf - 2 . 5 _ { \perp 1 7 . 4 } }
$$

$$
\mathbf { - 0 . 6 } _ { \downarrow 1 0 . 7 }
$$

$$
{ \bf 8 5 . 0 _ { \perp 2 5 . 9 } }
$$

$$
8 5 . 5 _ { \perp 2 3 . 8 }
$$

$$
{ \bf 8 5 . 1 _ { \perp 1 1 . 6 } }
$$

$$
{ \bf 8 5 . 5 _ { \perp 9 . 4 } }
$$

$$
\mathbf { 0 . 0 } _ { \downarrow 1 4 . 4 }
$$

$$
\mathbf { - 0 . 1 } _ { \downarrow 1 4 . 3 }
$$

$$
\mathbf { 8 4 . 9 } _ { \downarrow 2 9 . 3 }
$$

$$
{ \mathbf { 8 6 . 5 _ { \perp 1 4 . 5 } } }
$$

$$
\mathbf { - 1 . 7 } _ { \downarrow 1 4 . 9 }
$$

$$
\mathbf { - 0 . 8 } _ { \downarrow 1 3 . 3 }
$$

$$
{ \bf 7 1 . 1 _ { \perp 4 7 . 3 } }
$$

$$
{ \bf 7 1 . 9 _ { \perp 2 6 . 8 } }
$$

$$
\mathbf { 7 4 . 0 _ { \perp 4 7 . 5 } }
$$

$$
\mathbf { - 0 . 8 } _ { \downarrow 2 0 . 5 }
$$

$$
{ \bf 7 1 . 8 _ { \perp 2 8 . 6 } }
$$

$$
7 3 . 2 _ { \perp 3 8 . 1 }
$$

$$
7 3 . 2 _ { \perp 2 5 . 7 }
$$

$$
\pmb { 2 . 2 } _ { \downarrow 1 8 . 9 }
$$

$$
7 5 . 4 _ { \downarrow 5 3 . 5 }
$$

$$
{ \bf 7 4 . 1 _ { \perp 4 1 . 8 } }
$$

$$
7 5 . 8 _ { \perp 2 5 . 8 }
$$

$$
{ 7 8 . 7 } _ { \downarrow 4 1 . 1 }
$$

$$
7 5 . 7 _ { \perp 3 2 . 5 }
$$

$$
7 7 . 3 _ { \perp 2 2 . 4 }
$$

$$
\mathbf { - 0 . 3 } _ { \downarrow 2 1 . 0 }
$$

$$
\mathbf { - 1 . 7 _ { \downarrow 1 6 . 0 } }
$$

$$
{ \bf 1 . 4 } _ { \downarrow 1 8 . 7 }
$$

$$
\mathbf { - 0 . 4 } _ { \downarrow , 1 8 . 0 }
$$

$$
7 4 . 6 _ { \perp 5 0 . 8 }
$$

$$
{ \bf 7 4 . 0 _ { \perp 2 7 . 3 } }
$$

$$
{ \bf 7 0 . 1 } _ { \perp 3 2 . 4 }
$$

$$
7 2 . 6 _ { \downarrow 1 6 . 8 }
$$

$$
7 6 . 7 _ { \perp 3 5 . 2 }
$$

$$
7 3 . 0 _ { \perp 1 7 . 6 }
$$

$$
{ \pmb 4 . 5 } _ { \perp 1 8 . 4 }
$$

$$
{ \bf 7 4 . 1 _ { \perp 4 6 . 9 } }
$$

$$
\mathbf { 1 . 7 } _ { \downarrow 1 7 . 8 }
$$

$$
3 . 7 _ { \downarrow 1 7 . 6 }
$$

$$
7 7 . 4 _ { \perp 3 0 . 3 }
$$

$$
7 7 . 3 _ { \perp 3 5 . 2 }
$$

$$
7 6 . 6 _ { \perp 1 8 . 3 }
$$

$$
7 5 . 1 _ { \perp 4 2 . 8 }
$$

$$
7 3 . 5 _ { \perp 2 3 . 9 }
$$

$$
{ \mathbf { - 3 . 3 _ { \downarrow 1 6 . 6 } } }
$$

$$
7 7 . 5 _ { \perp 4 7 . 2 }
$$

$$
\mathbf { 1 . 6 } _ { \downarrow 1 8 . 9 }
$$

$$
7 4 . 7 _ { \perp 3 5 . 6 }
$$

$$
\mathbf { - 0 . 2 } _ { \downarrow 1 5 . 3 }
$$

$$
{ \bf 7 4 . 6 _ { \perp 2 1 . 7 } }
$$

$$
7 3 . 2 _ { \perp 2 6 . 1 }
$$

$$
{ \bf 0 . 1 } _ { \perp 1 4 . 0 }
$$

$$
7 5 . 4 _ { \perp 4 9 . 6 }
$$

$$
{ \bf 7 4 . 1 _ { \perp 3 0 . 3 } }
$$

$$
{ \bf 4 . 3 } _ { \perp 2 1 . 1 }
$$

$$
{ \bf 1 . 3 _ { \perp 1 9 . 3 } }
$$

$$
7 7 . 6 _ { \perp 5 2 . 0 }
$$

$$
7 7 . 5 _ { \perp 4 5 . 3 }
$$

$$
7 5 . 9 _ { \downarrow 5 3 . 7 }
$$

$$
{ \bf 8 2 . 2 } _ { \perp 3 9 . 8 }
$$

$$
{ \bf 0 . 1 } _ { \downarrow , 6 . 7 }
$$

$$
\mathbf { 8 0 . 9 } _ { \downarrow . 5 3 . 4 }
$$

$$
\mathbf { - 6 . 3 _ { \perp 1 3 . 9 } }
$$

$$
{ 7 6 . 3 } _ { \downarrow 2 9 . 9 }
$$

$$
{ \bf 8 2 . 5 _ { \perp 4 5 . 6 } }
$$

$$
{ \bf 8 1 . 8 _ { \perp 4 4 . 6 } }
$$

$$
{ \bf - 1 . 6 } _ { \downarrow 7 . 8 }
$$

$$
\mathbf { 7 9 . 9 _ { \perp 4 0 . 4 } }
$$

$$
{ \pmb 5 . 5 } _ { \perp 1 4 . 7 }
$$

$$
{ \bf 8 1 . 7 } _ { \perp 5 3 . 4 }
$$

$$
{ \bf 8 1 . 4 } _ { \perp 3 7 . 8 }
$$

$$
{ \bf 8 2 . 5 _ { \perp 4 3 . 6 } }
$$

$$
\mathbf { - 0 . 8 _ { \perp 9 . 8 } }
$$

$$
7 9 . 2 _ { \perp 3 9 . 4 }
$$

$$
\mathbf { - 1 . 5 _ { \perp 2 . 6 } }
$$

$$
{ \bf 8 1 . 6 _ { \perp 4 5 . 2 } }
$$

$$
{ \bf 7 9 . 1 _ { \perp 5 0 . 8 } }
$$

$$
7 8 . 7 _ { \perp 5 5 . 1 }
$$

$$
\mathbf { - 0 . 1 } _ { \downarrow 1 1 . 4 }
$$

$$
{ \bf 7 8 . 1 _ { \perp 4 3 . 1 } }
$$

$$
7 8 . 7 _ { \perp 4 0 . 0 }
$$

$$
\mathbf { 0 . 6 } _ { \downarrow 1 2 . 0 }
$$

$$
{ \bf 8 1 . 2 _ { \perp 4 3 . 6 } }
$$

$$
7 8 . 8 _ { \perp 3 8 . 6 }
$$

$$
7 8 . 6 _ { \perp 5 0 . 3 }
$$

$$
\mathbf { - 0 . } 2 _ { \downarrow 1 1 . 7 }
$$

$$
7 8 . 2 _ { \perp 5 2 . 2 }
$$

$$
{ \bf 8 1 . 0 } _ { \perp 3 5 . 7 }
$$

$$
{ \bf - } 2 . 4 _ { \downarrow 9 . 5 }
$$

$$
\mathbf { 8 0 . 9 } _ { \downarrow 4 5 . 7 }
$$

$$
7 9 . 0 _ { \downarrow 3 8 . 4 }
$$

$$
{ \bf 8 0 . 9 } _ { \perp 3 9 . 8 }
$$

$$
{ \bf 8 0 . 1 _ { \perp 5 4 . 6 } }
$$

$$
\mathbf { - 0 . 8 } _ { \downarrow 1 3 . 8 }
$$

$$
{ 7 6 . 2 } _ { \downarrow 4 1 . 1 }
$$

$$
7 9 . 7 _ { \downarrow 5 3 . 8 }
$$

$$
2 . 5 _ { \downarrow 1 1 . 2 }
$$

$$
\mathbf { 7 9 . 0 _ { \perp 4 1 . 6 } }
$$

$$
3 . 9 _ { \downarrow 1 3 . 5 }
$$

$$
\mathbf { 0 . 7 } _ { \downarrow 1 2 . 3 }
$$

$$
\mathbf { 0 . 2 } _ { \downarrow 9 . 9 }
$$

Table 14: SearchQA. Beginning/Middle/End: Word Position within the target evidence text. Strict/Soft cells: bold full-set clean accuracy; ↓ MR on closed-book-correct items (%). Each ∆ cell: bold clean-accuracy contrast and ↓ MR contrast, both Strict minus Soft (pp); Mean averages positions. The arrow identifies the MR component, not an accuracy drop or a sign.

$$
\pmb { \Delta } _ { \mathbf { i n s t r } }
$$

$$
\pmb { \Delta } _ { \mathbf { i n s t r } }
$$

$$
\pmb { \Delta } _ { \mathbf { i n s t r } }
$$

$$
\overline { { \Delta } } _ { \mathrm { i n s t r } }
$$

$$
{ \bf 9 0 . 0 _ { \perp 4 . 6 } }
$$

$$
{ \bf 9 0 . 2 _ { \perp 7 . 1 } }
$$

$$
{ \bf 0 . 4 } _ { \downarrow , 9 . 3 }
$$

$$
{ \bf 4 . 9 _ { \perp 7 . 5 } }
$$

$$
{ \bf 9 1 . 5 _ { \perp 1 4 . 8 } }
$$

$$
{ \bf 9 1 . 6 _ { \perp 5 . 9 } }
$$

$$
{ \bf 9 1 . 0 _ { \perp / 2 3 . 0 } }
$$

$$
{ \mathbf { 3 . 9 } } _ { \downarrow . 8 . 5 }
$$

$$
\mathbf { 9 2 . 4 _ { \perp 1 7 . 0 } }
$$

$$
{ \bf 8 7 . 1 _ { \perp 1 4 . 5 } }
$$

$$
\mathbf { 2 . 0 } _ { \downarrow , 8 . 0 }
$$

$$
{ \bf - 0 . 1 } _ { \perp 8 . 9 }
$$

$$
{ \bf 9 0 . 3 _ { \perp 1 8 . 1 } }
$$

$$
\mathbf { 9 0 . 3 _ { \perp 1 7 . 4 } }
$$

$$
{ \bf 8 8 . 2 } _ { \perp 5 . 7 }
$$

$$
{ \bf 9 1 . 1 _ { \perp 8 . 9 } }
$$

$$
{ \bf 9 1 . 3 _ { \perp 1 6 . 7 } }
$$

$$
{ \bf 2 . 1 } _ { \downarrow 1 2 . 4 }
$$

$$
\mathbf { - 0 . 8 } _ { \downarrow . 8 . 5 }
$$

$$
{ \bf 8 9 . 3 _ { \perp 9 . 0 } }
$$

$$
8 7 . 7 _ { \downarrow 1 6 . 4 }
$$

$$
{ \bf 4 . 4 } _ { \downarrow . 8 . 5 }
$$

$$
\mathbf { 2 . 0 } _ { \downarrow 7 . 7 }
$$

$$
{ \bf 8 8 . 4 _ { \perp 5 . 9 } }
$$

$$
{ \bf 8 8 . 0 _ { \perp 2 4 . 3 } }
$$

$$
{ \bf 8 3 . 7 _ { \perp 1 8 . 1 } }
$$

$$
{ \bf 8 6 . 3 _ { \perp 9 . 2 } }
$$

$$
\mathbf { - 0 . 7 } _ { \downarrow 1 0 . 5 }
$$

$$
\mathbf { - 2 . 6 } _ { \downarrow . 8 . 9 }
$$

$$
\mathbf { 8 5 . 9 } _ { \downarrow 1 0 . 3 }
$$

$$
{ \bf 2 . 1 _ { \perp 1 4 . 0 } }
$$

$$
{ \bf 8 6 . 5 _ { \perp 1 6 . 5 } }
$$

$$
{ \bf 0 . 4 } _ { \downarrow 1 0 . 5 }
$$

$$
{ \bf 8 7 . 3 _ { \perp 4 . 7 } }
$$

$$
\mathbf { 9 0 . 1 } _ { \perp 2 2 . 7 }
$$

$$
{ \bf 9 1 . 0 _ { \perp 9 . 7 } }
$$

$$
8 8 . 4 _ { \perp 2 6 . 6 }
$$

$$
\mathbf { - 0 . 9 } _ { \downarrow 1 3 . 0 }
$$

$$
\mathbf { 0 . 2 } _ { \downarrow 1 2 . 0 }
$$

$$
\mathbf { 0 . 2 } _ { \downarrow 1 5 . 2 }
$$

$$
\mathbf { - 1 . 1 _ { \downarrow 1 2 . 0 } }
$$

$$
\mathbf { 8 5 . 9 } _ { \downarrow 2 5 . 3 }
$$

$$
{ \bf 8 6 . 8 _ { \perp 9 . 1 } }
$$

$$
\mathbf { 8 9 . 4 } _ { \perp 1 7 . 2 }
$$

$$
{ \bf 8 8 . 9 _ { \perp 7 . 4 } }
$$

$$
\mathbf { - 0 . 9 } _ { \downarrow 1 6 . 2 }
$$

$$
{ \bf 0 . 6 } _ { \downarrow 9 . 8 }
$$

$$
{ \bf 8 4 . 1 } _ { \perp 2 2 . 7 }
$$

$$
{ \bf 8 7 . 6 _ { \perp 1 6 . 5 } }
$$

$$
{ \mathbf { 8 4 . 4 } } _ { \downarrow 1 1 . 3 }
$$

$$
\mathbf { 8 7 . 6 } _ { \downarrow 7 . 0 }
$$

$$
\mathbf { - 0 . 3 } _ { \downarrow 1 1 . 4 }
$$

$$
{ \bf 8 1 . 4 } _ { \perp 2 6 . 6 }
$$

$$
{ \bf 8 8 . 4 } _ { \perp 2 2 . 8 }
$$

$$
{ \bf 8 7 . 6 _ { \perp 1 4 . 1 } }
$$

$$
\mathbf { - 6 . 2 _ { \downarrow 1 2 . 5 } }
$$

$$
\mathbf { 8 9 . 0 } _ { \downarrow 1 1 . 1 }
$$

$$
\mathbf { - 2 . 5 _ { \perp 1 3 . 4 } }
$$

$$
\mathbf { - 0 . 6 } _ { \downarrow 1 1 . 7 }
$$

$$
{ \bf 0 . 0 } _ { \downarrow 1 0 . 3 }
$$

<table><tr><td colspan="12">Open-source</td></tr><tr><td>∞ Llama-3-8B</td><td> $7 5 . 8 _ { \perp 3 4 . 3 }$ </td><td> $7 8 . 4 _ { \perp 2 3 . 9 }$ </td><td> $\mathbf { - } 2 . 6 _ { \downarrow , 1 0 . 4 }$ </td><td> ${ \bf 7 1 . 4 } _ { \perp 3 3 . 8 }$ </td><td>79.8↓21.1 -8.4↓12.7</td><td></td><td> $7 3 . 4 _ { \downarrow 4 4 . 8 }$ </td><td> $7 3 . 3 _ { \perp 3 1 . 7 }$ </td><td> ${ \bf 0 . 1 } _ { \perp 1 3 . 1 }$ </td><td> $\mathbf { - 3 . 6 } _ { \downarrow 1 2 . 1 }$ </td></tr><tr><td>H Mistral-7B</td><td> $\mathbf { 8 0 . 9 } _ { \perp 3 6 . 2 }$ </td><td> $7 5 . 5 _ { \perp 2 2 . 7 }$ </td><td> ${ \pmb 5 . 4 } _ { \downarrow 1 3 . 5 }$ </td><td> $7 7 . 2 _ { \perp 3 7 . 6 }$ </td><td> $7 9 . 4 _ { \downarrow 2 0 . 8 }$ </td><td></td><td>-2.216.8 76.0↓48.7</td><td> $7 6 . 5 _ { \perp 2 9 . 1 }$ </td><td> $\mathbf { - 0 . 5 } _ { \downarrow 1 9 . 6 }$ </td><td>0.9↓16.6</td></tr><tr><td>Qwen3-8B</td><td></td><td></td><td>80.1↓37.9 78.1↓21.0 2.0↓16.9 80.0↓27.6</td><td></td><td> ${ \bf 8 1 . 0 _ { \perp 1 7 . 8 } }$ </td><td> $\mathbf { - 1 . 0 } _ { \downarrow 9 . 8 }$ </td><td></td><td>75.7↓38.4 79.7±20.8 -4.0↓17.6</td><td></td><td> $\mathbf { - 1 . 0 } _ { \downarrow 1 4 . 8 }$ </td></tr><tr><td>Gemma-2-9B</td><td> $7 7 . 0 _ { \perp 2 9 . 1 }$ </td><td> $\mathbf { 7 9 . 0 _ { \perp 2 1 . 1 } }$ </td><td> $\mathbf { - 2 . 0 } _ { \downarrow , 8 . 0 }$ </td><td> $7 3 . 3 _ { \perp 2 4 . 0 }$ </td><td> ${ \bf 8 1 . 1 _ { \perp 1 8 . 5 } }$ </td><td>-7.8↓5.5</td><td> $7 8 . 7 _ { \perp 3 1 . 5 }$ </td><td> $7 9 . 7 _ { \perp 2 2 . 7 }$ </td><td> $\mathbf { - 1 . 0 } _ { \downarrow 8 . 8 }$ </td><td> $\mathbf { - 3 . 6 } _ { \downarrow 7 . 4 }$ </td></tr><tr><td>Average</td><td>78.5↓34.4</td><td> $7 7 . 8 _ { \perp 2 2 . 2 }$ </td><td> $\mathbf { 0 . 7 } _ { \downarrow 1 2 . 2 }$ </td><td> $7 5 . 5 _ { \perp 3 0 . 8 }$ </td><td> ${ \bf 8 0 . 3 _ { \perp 1 9 . 6 } }$ </td><td>-4.9↓11.2</td><td> ${ \bf 7 6 . 0 _ { \perp 4 0 . 9 } }$ </td><td> $7 7 . 3 _ { \perp 2 6 . 1 }$ </td><td> ${ \bf - 1 . 4 } _ { \downarrow 1 4 . 8 }$ </td><td> $\mathbf { - 1 . 8 } _ { \downarrow 1 2 . 7 }$ </td></tr></table>

$$
\mathbf { 8 0 . 8 _ { \perp 5 0 . 9 } }
$$

$$
{ \mathbf { 8 2 . 3 _ { \perp 3 7 . 6 } } }
$$

$$
\mathbf { - 1 . 5 _ { \downarrow 1 3 . 3 } }
$$

$$
{ \mathbf { 8 4 . 2 _ { \downarrow 4 7 . 1 } } }
$$

$$
{ \bf 8 0 . 0 } _ { \perp 3 7 . 9 }
$$

$$
{ \bf 7 8 . 1 _ { \perp 4 5 . 5 } }
$$

$$
{ \bf 8 3 . 6 } _ { \perp 3 3 . 1 }
$$

$$
\mathbf { - 1 . 9 } _ { \downarrow 7 . 6 }
$$

$$
\mathbf { 8 0 . 9 } _ { \downarrow 4 2 . 8 }
$$

$$
{ \bf 8 1 . 3 _ { \perp 5 1 . 7 } }
$$

$$
{ \bf 8 4 . 1 _ { \perp 3 9 . 4 } }
$$

$$
7 8 . 6 _ { \perp 3 0 . 3 }
$$

$$
\mathbf { - 2 . 8 _ { \perp 1 2 . 3 } }
$$

$$
{ \pmb 2 . 3 } _ { \downarrow 1 2 . 5 }
$$

$$
{ \bf 8 1 . 1 } _ { \perp 5 3 . 2 }
$$

$$
{ \bf 8 3 . 9 } _ { \perp 3 2 . 0 }
$$

$$
{ \bf - 1 . 2 } _ { \downarrow 1 3 . 2 }
$$

$$
{ \bf 8 2 . 8 _ { \perp 3 8 . 8 } }
$$

$$
{ \bf 8 3 . 3 _ { \perp 2 7 . 1 } }
$$

$$
\mathbf { - 1 . 7 _ { \downarrow 1 4 . 4 } }
$$

$$
\mathbf { - 0 . 4 } _ { \downarrow 1 1 . 5 }
$$

$$
{ \mathbf { 8 5 . 2 } } _ { \downarrow 4 1 . 5 }
$$

$$
{ \bf - 7 . 4 } _ { \downarrow 4 . 1 }
$$

$$
\mathbf { - 3 . 5 _ { \perp 8 . 0 } }
$$

$$
\mathbf { 8 4 . 7 _ { \perp 4 2 . 4 } }
$$

$$
{ \bf - 1 . 1 } _ { \downarrow 9 . 3 }
$$

$$
{ \bf 8 3 . 4 _ { \perp 4 1 . 1 } }
$$

$$
{ \bf 8 0 . 3 _ { \perp 3 4 . 7 } }
$$

$$
{ \bf 8 3 . 2 _ { \perp 5 0 . 4 } }
$$

$$
{ \bf 8 2 . 0 _ { \perp . 5 0 . 3 } }
$$

$$
{ \bf - 1 . 2 } _ { \downarrow 7 . 1 }
$$

$$
{ \bf 8 0 . 2 } _ { \perp 3 6 . 7 }
$$

$$
{ \bf 3 . 0 } _ { \downarrow \downarrow 3 . 7 }
$$

$$
\mathbf { - 0 . 7 _ { \downarrow 1 2 . 4 } }
$$

## F Additional Position and Prompt Robustness Protocols

This appendix accompanies Tables 2 and 3: 300 QA questions per model, with $| K _ { A } | = 2 2 0$ and $| K _ { B } | = 1 9 8 . \ \mathrm { { Q A } , \mathrm { { A } } } .$ and B retain their supplied identifiers. These checks complement the main evaluation of fifteen systems and the 500-item audit; their contrasts and intervals use the corresponding QA records.

## F.1 Paired Word Position Analysis

Required records and fixed population. Paired analysis requires frozen question IDs, references and aliases, foils, closed-book responses, and all six clean/edited instruction–Word Position pairs. The audit trail comprises model and chat-template versions, complete requests, decoding settings, input lengths, and truncation records. Valid Word Position comparisons hold document order fixed and verify the realized answer-span locations; the supplementary audit’s document slots are not a substitute. $\dot { K _ { M } }$ remains fixed across conditions. Failed calls require recovery or explicit accounting for missing data, not labels indicating incorrect answers. Discarding incomplete grids or changing denominators by position changes the paired estimand.

Paired estimands. For a fixed model and dataset, let $h _ { q , \pi , p }$ indicate an edited response that matches the designated foil but not the reference. The two position contrasts are

$$
G _ { \pi } ^ { E - M } = \frac { 1 0 0 } { \vert K _ { M } \vert } \sum _ { q \in K _ { M } } \left( h _ { q , \pi , \mathrm { E n d } } - h _ { q , \pi , \mathrm { M i d } } \right) , \qquad G _ { \pi } ^ { B - M } = \frac { 1 0 0 } { \vert K _ { M } \vert } \sum _ { q \in K _ { M } } \left( h _ { q , \pi , \mathrm { B e g } } - h _ { q , \pi , \mathrm { M i d } } \right) .
$$

The third contrast is $G _ { \pi } ^ { B - E } = G _ { \pi } ^ { B - M } - G _ { \pi } ^ { E - M }$ , calculated within question before resampling; its interval is not obtained by subtracting interval endpoints. Table 2 reports all three. Under Strict, End-minus-Middle intervals are positive and Beginning-minus-End intervals are negative for both models. Beginning-minus-Middle remains unresolved, as do all Soft contrasts. These results for individual models do not establish the complete ordering or an instruction–position interaction.

Uncertainty and supplementary diagnostics. The supplied position results use 2,000 paired bootstrap resamples, retaining all repeated cells of a sampled question. The summary does not specify its random seed, method of constructing intervals, or multiplicity adjustment. Intervals are interpreted individually, not as simultaneous confirmation of every contrast. These analyses follow inspection of the main summaries and are not claimed as preregistered. A interval containing zero does not establish equality; a nonzero contrast does not identify an attention mechanism. Complete clean/edited records also permit direct transitions and comparisons of gold retention on the same $K _ { M }$ , but these require their own calculations. Clean accuracy on the full set must not be subtracted from MR, and the document-slot audit does not replicate this position intervention.

## F.2 Source Policy Paraphrases and Execution

Controlled prompt construction. Appendix A.1 gives the original clauses specifying source policy. The candidates below paraphrase only that policy: Strict prioritizes documents despite apparent errors; Soft permits prior knowledge to override conflicting documents. Verify these meanings and freeze the clauses and cohort before inference. Insert one clause at the same location in the system prompt, holding the role, rules for selecting answers, output contract, wrapper for the user message, and evidence fixed. Add no verification, abstention, or explanation requests.

## Source-Policy Paraphrase Candidates

## Paraphrase 1: Strict

Treat the supplied documents as the primary authority for your answer. Give a concise answer even when their contents seem incorrect.

Paraphrase 1: Soft

Consult the documents when useful. If their claims contradict well-known facts, give priority to your own knowledge.

## Paraphrase 2: Strict

Base your concise answer primarily on the provided documents, including when the information in them appears to be mistaken.

## Paraphrase 2: Soft

Draw on the supplied documents where they help, but rely on your own knowledge when they disagree with well-known facts.

Execution and audit trail. The full design crosses three prompt versions, two policies, three Word Positions, and two evidence arms: 36 requests per question and model. Reusing original outputs is valid only with identical requests, model versions, and settings; the two additional versions then require 24 new requests. The original closed-book stratum remains fixed. Resource-limited subsets require selection before inference. Reproduction requires variant IDs, complete messages, context IDs, model revisions, generation settings, raw responses, matcher version, and failure/retry records. The design includes neither live retrieval nor model updates.

Prompt contrasts and changes in effect size. Let $h _ { q , \pi , p } ^ { ( v ) }$ denote the foil-only indicator under version $v ,$ with $v = 0$ the original. For each question define

$$
\begin{array} { l } { { d _ { q } ^ { ( v ) } = \displaystyle \frac { 1 } { 3 } \sum _ { p } \left( h _ { q , \mathrm { S t r i c t } , p } ^ { ( v ) } - h _ { q , \mathrm { S o f t } , p } ^ { ( v ) } \right) , } } \\ { { \Delta _ { v } = \displaystyle \frac { 1 0 0 } { | K _ { M } | } \sum _ { q \in K _ { M } } d _ { q } ^ { ( v ) } , \qquad R _ { v } = \displaystyle \frac { 1 0 0 } { | K _ { M } | } \sum _ { q \in K _ { M } } ( d _ { q } ^ { ( v ) } - d _ { q } ^ { ( 0 ) } ) . } } \end{array}
$$

Comparisons pair questions across versions, policies, and positions. Table 3 reports MR, $\Delta _ { v } , R _ { v } ,$ and 95% intervals; its resample count and seed are unspecified (2,000 resamples applies only to the position table). The change from the original to itself is zero. All six instruction-contrast intervals exclude zero; all four changes from the original include zero. This supports consistent direction, not equivalence, which requires a prespecified meaningful margin and a dedicated analysis. Contrasts retain their supplied precision rather than being recomputed from rounded MR. Saved requests are needed to verify exact paraphrase bindings to execution messages.