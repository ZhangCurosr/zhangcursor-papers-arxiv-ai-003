# Structure Tax: How Structured Output affects LLMs Performance

Vineet Kumar Kanishka\* Bhuvanesh Mandora

PayPal Artificial Intelligence

PayPal, Bengaluru India

vkumar32@paypal.com, kanishka21dhar@gmail.com, bmandora@paypal.com

## Abstract

Deploying large language models in production often requires constraining outputs to structured formats such as JSON or XML, and prior work treats the resulting accuracy loss as an inherent ‘structure tax’. We re-examine this claim by evaluating a battery of models, datasets and schemas, measuring task accuracy, confidence calibration, and hidden-state geometry. The tax turns out to depend on schema design rather than on structure per se: reasoning-first field ordering matches or exceeds free-form accuracy, while answer-first ordering causes steep drops, particularly in smaller models. Format sensitivity scales inversely with a task’s own structural constraints, and schemas that preserve reasoning order also improve calibration with CKA showing greater separability between correct and incorrect representations in middle transformer layers. Our findings indicate that properly designed structured formats can match or exceed free-form performance, reframing the critical question from ‘whether to structure’ to ‘how to structure’ for optimal reasoning preservation.

## 1 Introduction

Large language models are now deployed across diverse domains, including mathematical reasoning, knowledge retrieval (Rathore et al., 2026), and code generation (Kumar et al., 2025). In practice, their utility depends not only on producing correct answers, but also on how outputs are structured and how reliably models communicate uncertainty. As a result, structured output formats such as XML tags, JSON schemas etc. have become common prompting strategies for improving reliability and downstream usability. However, their impact on model behavior remains underexplored. Do structured formats merely alter surface form, or do they change how models internally process and represent problems? And if structure affects reasoning, does it also influence confidence calibration?

LLMs are known to be sensitive to a wide range of prompt features separator tokens, casing, template choice, few-shot ordering and this brittleness has been documented across model scales and instruction-tuning regimes (Sclar et al., 2024; He et al., 2024). Within this broader landscape, output format constraints are distinctive on two counts. First, they are the part of a prompt that practitioners actually ship to production: free-form responses are rarely viable in deployed systems, so the output schema is not an optional choice but a structural commitment with downstream consequences. Second, unlike surface-token brittleness whose magnitude is empirical and whose direction is not predictable a priori the effect of output structure on reasoning admits a causal hypothesis grounded in the architecture itself. We therefore focus on field ordering as a mechanistically tractable sub-axis of format sensitivity, and ask whether the apparent cost of structured outputs reflects a property of structure per se or of a specific design choice within it.

In this work, we investigate Structure Tax: the systematic effect of output format constraints on task accuracy, calibration, and internal representations. Across five models and four benchmarks, we show that the apparent cost is not inherent to structure but contingent on a single design choice – field ordering with second-order consequences for calibration and hidden-state geometry.

A key design dimension we examine isfield ordering: whether the reasoning chain (R) is placed before or after the final answer (A) within a structured schema. This distinction matters because autoregressive generation is strictly causal; each token conditions on all preceding tokens, but not on those yet to be produced. A reasoning-first schema (R,A) forces the model to produce its reasoning trace before committing to an answer, allowing the answer token to be informed by a fully elaborated chain of thought. An answer-first schema (A,R) inverts this dependency: the model commits to a specific answer token before any reasoning is generated, after which the reasoning field can only rationalize a decision already made. Figure 1 illustrates both orderings alongside a free-form baseline, making the structural difference concrete.

If field ordering matters because of autoregressive causality, then (R,A) schemas should preserve or improve reasoning quality while (A,R) schemas should degrade it, with the magnitude of degradation scaling with how much a task depends on multi-step inference. We confirm this prediction empirically. (R,A) schemas match or exceed freeform accuracy across the large majority of configurations, while (A,R) causes severe degradation on multi-step tasks, with the magnitude scaling inversely with a task’s inherent structural organisation: low-structure tasks (knowledge QA, logical reasoning) benefit most; high-structure tasks (textto-SQL) are largely format-invariant. This inversestructure gradient holds at three levels of analysis — task accuracy, confidence calibration, and hidden-state geometry via CKA and linear probes and structured formatting simultaneously reduces expected calibration error, establishing field ordering as both an accuracy and a calibration lever.

## 2 Related Work

Tam et al. (2024) report that JSON and XML output constraints can degrade reasoning accuracy, motivating the notion of a behavioural “structure tax”; Long et al. (2025) broaden this to fifteen formats across four categories and propose mitigation through format-aware prompting and fine-tuning. A parallel body of work documents input-side sensitivity: Sclar et al. (2024) show that semantically irrelevant surface features like separators, casing, spacing can swing accuracy with brittleness persisting across model scales and instruction tuning, and He et al. (2024) report a 40-point spread on GPT-3.5 from template variation alone. These studies treat each format as atomic asking which format to use and lack a causal mechanism for the observed effects. We isolate a within-format axis, the relative position of reasoning and answer fields, where the direction of the effect is predictable from autoregressive generation: a token emitted at position t can only condition on positions strictly less than t, so an answer token committed before its rationale

cannot be revised by it.

The benefits of placing reasoning before the answer are established for chain-of-thought prompting (Wei et al., 2022). Most closely related, Fu et al. (2026) compare answer-direct and reasoningthen-answer prompts on multiple-choice tasks and find that reasoning-first increases verbalised confidence regardless of correctness. Our work differs in three ways: we span a structural-density continuum rather than MCQ alone, we separate the formatfamily axis (JSON vs XML) from the field-ordering axis, and we trace the mechanism into hidden-state geometry. In a complementary fine-tuning setting, Chen et al. (2025) argue that post-thinking distillation (answer before rationale) is more robust to rationale errors for small models whether our prompted-generation finding persists under such training is an open question. Our results also sharpen the connection to chain-of-thought faithfulness (Lanham et al., 2023): plausible-sounding reasoning can still be post-hoc if the answer token precedes it in the generation order.

Calibration is commonly assessed using Expected Calibration Error (Minderer et al., 2021) and reliability diagrams (Guo et al., 2017). Hidden states are known to encode correctness signals beyond surface outputs (Servedio et al., 2025; Orgad et al., 2025; Zhang et al., 2025), motivating combined behavioural-representational analysis. We compare layer representations using Centred Kernel Alignment (Kornblith et al., 2019), mindful of documented interpretive caveats (Davari et al., 2023; Cui et al., 2022), and verify CKA directions against linear probes and accuracy. To our knowledge, no prior work jointly examines how output format constraints affect task performance, calibration, and correctness-related representational geometry; our study addresses this gap.

## 3 Experimental Design

We investigate three complementary aspects of format constraints task accuracy, calibration, and internal representations detailed in §3.2.

## 3.1 Dataset

We select four datasets that sample a continuum of intrinsic task structure the degree to which the task itself imposes compositional or sequential constraints on a valid answer, independent of any output format we impose. This continuum framing is central to our analysis: if format scaffolding interacts with intrinsic structure, the magnitude of any format effect should scale with where a task sits on the continuum. Spider (Yu et al., 2018) anchors the high-structure end, a large-scale text-to-SQL dataset where SQL grammar imposes rigid compositional constraints on valid outputs. GSM8K (Cobbe et al., 2021) occupies a middle position: grade-school arithmetic word problems require multi-step reasoning that provides sequential scaffolding but no syntactic enforcement. MMLU (Hendrycks et al., 2021), a 57-subject multitask benchmark, and LogicBench (Parmar et al., 2024), designed to evaluate logical reasoning across reasoning types and complexity levels, sit at the low-structure end: the multiple-choice format is externally imposed and the reasoning path is otherwise unconstrained. JSON and XML are the two formats we evaluate; together they cover the principal axis a practitioner faces key-ordered serialisation (JSON) versus tag-bounded markup (XML) and span the formats most commonly shipped in production pipelines.

![](images/801c17ac9dd954bc5f328c417f8a7d49511ea632cb94c517b6762314a263f729.jpg)  
Figure 1: Three response formats elicited from GPT-4o on the same MMLU Anatomy question. (Left) Free-form: natural-language reasoning arrives at the correct answer D. (Middle) Answer-first JSON: the model commits to "answer": "A" before generating a post-hoc rationale that cites only one of three valid targets, yielding an incorrect answer. (Right) Reasoning-first XML: rationale unrolled inside <reasoning> precedes the commitment, recovering the correct <answer>D</answer>.

## 3.2 Experiment protocol

Our experiment is three-pronged. We first measure task accuracy under structured output formats (JSON and XML, each in answer-first and reasoning-first orderings) against a free-form baseline. We then assess calibration by analyzing both verbalized confidence scores and token-level probability signals to determine whether format constraints shift the alignment between model confidence and empirical accuracy. Finally, we probe layer-wise hidden states using Centered Kernel Alignment (CKA) to characterize how format constraints alter the internal geometry of correct versus incorrect representations across model layers.

Why include the answer-first (A,R) ordering? We include (A,R) for three reasons. First, it mirrors a common production pattern where developers place the answer before the rationale because it reads more naturally to human reviewers (e.g., “rate this text and provide a rationale”). Second, while the direction of the effect is intuitive, its magnitude and model dependence reqquires empirical characterization. Third, (A,R) isolates the effect of field ordering from the broader format-family distinction (JSON vs. XML), allowing us to attribute performance differences specifically to when the answer is committed rather than to surface syntax.

## 3.3 Models

We evaluate five LLMs spanning both proprietary and open-weight families. The frontier proprietary models: GPT-4o (OpenAI, 2024), Gemini-Flash-2.0 (Google, 2025), and Claude-Sonnet-4.5 (Anthropic, 2025) – span the three major frontier providers, reducing single-provider bias in our conclusions. The open-weight models – Llama-3.1- 8B (Grattafiori et al., 2024) and Gemma3-4B (Kamath et al., 2025) span two scales (4B and 8B parameters) and two architecture families, which lets us check whether the representational patterns we observe (in Section §6) generalise across architectures rather than depending on a single model design. The open-weight subset also enables reproducible, large-scale ablations under practical compute constraints.

## 3.4 Evaluation Metrics

Let $\hat { y } _ { i }$ denote the model prediction and $p _ { i }$ the confidence for input $x _ { i }$ with ground truth $y _ { i }$

• Accuracy measures exact-match fraction; for Spider (Yu et al., 2018) we use Execution Accuracy (EX), which checks whether executing the predicted SQL yields the same result as the reference query.

• ECE quantifies calibration by comparing confidence to empirical accuracy across M bins:

$$
\mathrm { E C E } = \sum _ { m = 1 } ^ { M } \frac { | B _ { m } | } { N } \left| \operatorname { a c c } ( B _ { m } ) - \operatorname { c o n f } ( B _ { m } ) \right|
$$

• Confidence–Correctness Correlation $( \rho )$ is the Spearman rank correlation between $p _ { i }$ and $\mathbf { 1 } [ \hat { y } _ { i } = y _ { i } ]$ , capturing whether higher confidence predicts correctness.

• Brier Score and Entropy, used in Appendix C.2 and C.3, are defined there.

## 4 Structure Tax

We ask whether imposing a structured output schema incurs a universal accuracy penalty a structure tax or whether its effect depends on how the schema is designed.

Setup. Each model receives a task-specific system prompt specifying the required output schema (JSON or XML; R,A or A,R); all other prompt content is held constant across conditions. Responses are collected at temperature = 0 via each provider’s API. Evaluation uses exact-match accuracy for GSM8K, MMLU, and LogicBench, and execution accuracy (EX) for Spider. Prompt templates are reproduced in Appendix A.1.

Results (Table 1). Reasoning-first formats dominate. XML (R,A) or JSON (R,A) wins the bestformat cell in 13 of 20 model-dataset pairs and never collapses on tasks where A,R wins. The A,R penalty is sharpest on GSM8K XML (A,R) falls to 10.98% for both Llama-3.1-8B and Gemma3- 4B, against 83-87% under (R,A); confirming that committing the answer token before the reasoning chain breaks multi-step arithmetic entirely. On Spider (the high-structure task), freeform remains competitive for three of five models, consistent with SQL generation providing its own intrinsic structure.

Analysis. The severity of the ordering effect scales with task reasoning depth: the R,A advantage is largest on GSM8K, moderate on MMLU and LogicBench, and negligible on Spider. Within (R,A) XML outperforms JSON (margin up to 12.2 pp on LogicBench/Claude-Sonnet-4.5), suggesting that explicit tag boundaries scaffold the reasoning-answer transition more reliably than JSON key ordering alone. Smaller models are more fragile: Llama-3.1-8B and Gemma3-4B show the largest collapses under A,R and higher run-to-run variance, whereas frontier models are sensitive to ordering but do not catastrophically fail. Crucially, R,A formats simultaneously improve accuracy and reduce variance.

Robustness to surface prompt variation. A natural concern is whether the (R,A) advantage is an artefact of the specific prompts used. We test this by perturbing instructional phrasing, schema casing, and whitespace across three variants per condition on GSM8K with Llama-3.1-8B and GPT-4o (full setup in Appendix F). The (R,A) > (A,R) gap is preserved in every (model, format-family, variant) cell: the smallest gap across all 24 combinations is +37 points (GPT-4o JSON), while within-cell spread under perturbation is at most 11.5 points. Field ordering’s effect on accuracy is qualitatively invariant to surface prompt changes, distinguishing it from the format-template brittleness documented by (Sclar et al., 2024) and (He et al., 2024).

## KEY TAKEAWAY

Field ordering, not structure itself, governs the structure tax: reasoning-first schemas (R,A) consistently match or nearly match the best-performing formats, while answerfirst schemas (A,R) may cause severe and in some settings catastrophic degradation on reasoning-heavy tasks.

<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Model</td><td rowspan="2">Free Form</td><td colspan="2">Answer-first (A,R)</td><td colspan="2">Reasoning-first (R,A)</td></tr><tr><td>JSON</td><td>XML</td><td>JSON</td><td>XML</td></tr><tr><td rowspan="5">GSM8K</td><td>GPT-40</td><td> $8 8 . 7 8 \pm 1 . 0 2 $ </td><td> $5 5 . 3 4 \pm 1 . 5 8$ </td><td> $5 6 . 8 6 \pm 2 . 7 1$ </td><td> $9 5 . 6 0 \pm 1 . 0 1$ </td><td>95.60 ±0.17</td></tr><tr><td>Gemini-Flash-2.0</td><td> $7 9 . 3 8 \pm 0 . 2 8$ </td><td> $6 3 . 0 8 \pm 0 . 5 3$ </td><td> $5 4 . 9 0 \pm 3 . 6 4$ </td><td> $\mathbf { 9 5 . 5 3 \bot 0 . 5 9 }$ </td><td> $9 5 . 4 5 \pm 0 . 6 0$ </td></tr><tr><td>Claude-Sonnet-4.5</td><td> $8 6 . 5 0 \pm 1 . 1 3$ </td><td> $9 6 . 1 3 \pm 0 . 8 8$ </td><td> $6 2 . 3 5 \pm 2 . 2 9$ </td><td> $8 7 . 7 9 \pm 0 . 9 8$ </td><td> $\mathbf { 9 7 . 6 5 \pm } 0 . 3 9$ </td></tr><tr><td>Llama-3.1-8B</td><td> $7 9 . 9 8 \pm 1 . 1 0$ </td><td> $3 1 . 0 1 \pm 1 . 5 8$ </td><td> $1 0 . 9 8 \pm 2 . 6 6$ </td><td> $8 0 . 2 1 \pm 0 . 4 5$ </td><td> $\mathbf { 8 3 . 5 3 \pm } 2 . 5 8$ </td></tr><tr><td>Gemma3-4B</td><td> $7 8 . 8 2 \pm 1 . 4 6$ </td><td> $2 7 . 4 5 \pm 4 . 7 0$ </td><td> $1 0 . 9 8 \pm 2 . 1 3$ </td><td> $8 3 . 5 3 \pm 1 . 9 6$ </td><td> $\mathbf { 8 7 . 0 6 } \pm 2 . 3 4$ </td></tr><tr><td rowspan="5">LogicBench</td><td>GPT-40</td><td> $7 9 . 0 0 \pm 0 . 9 6$ </td><td> $8 3 . 8 0 \pm 1 . 2 9$ </td><td> $8 3 . 0 0 \pm 1 . 2 9$ </td><td> $8 3 . 2 0 \pm 0 . 7 3$ </td><td> $\mathbf { 8 4 . 2 0 \pm 1 . 9 3 }$ </td></tr><tr><td>Gemini-Flash-2.0</td><td> $7 2 . 8 0 \pm 2 . 1 1$ </td><td> $7 6 . 8 0 \pm 1 . 6 4$ </td><td> $8 4 . 7 7 \pm 2 . 3 4$ </td><td> $8 2 . 4 0 \pm 1 . 7 5$ </td><td> $\mathbf { 8 5 . 4 0 \pm 1 . 1 6 }$ </td></tr><tr><td>Claude-Sonnet-4.5</td><td> $8 7 . 6 0 \pm 2 . 1 7$ </td><td> $5 6 . 6 0 \pm 2 . 7 0$ </td><td> $\mathbf { 8 8 . 4 0 \pm 0 . 5 5 }$ </td><td> $7 5 . 0 0 \pm 0 . 7 7$ </td><td> $8 7 . 2 0 \pm 1 . 3 5$ </td></tr><tr><td>Llama-3.1-8B</td><td> $5 7 . 2 0 \pm 0 . 7 9$ </td><td> ${ \bf 7 6 . 0 0 \pm 1 . 0 5 }$ </td><td> $7 0 . 3 4 \pm 1 . 3 6$ </td><td> $7 4 . 6 0 \pm 2 . 0 5$ </td><td> $7 3 . 0 0 \pm 2 . 5 0$ </td></tr><tr><td> $\mathrm { G e m m a } 3 – 4 \mathrm { B }$ </td><td> $6 1 . 2 0 \pm 0 . 7 1$ </td><td> ${ \bf 6 9 . 4 0 \pm } 1 . 3 3 $ </td><td> $6 7 . 4 0 \pm 1 . 8 7$ </td><td> $6 6 . 0 0 \pm 1 . 6 1$ </td><td> $6 6 . 2 0 \pm 2 . 2 2$ </td></tr><tr><td rowspan="5">MMLU</td><td>GPT-40</td><td> $7 9 . 9 5 \pm 0 . 8 4$ </td><td> $8 2 . 5 0 \pm 0 . 8 3$ </td><td> $8 1 . 5 8 \pm 1 . 3 6 $ </td><td> ${ \bf 8 7 . 4 6 \pm 0 . 7 0 }$ </td><td> $8 6 . 9 4 \pm 0 . 7 8$ </td></tr><tr><td> $\mathrm { G e m i n i - F l a s h { - } } 2 . 0$ </td><td> $5 6 . 2 4 \pm 0 . 7 8$ </td><td> $8 4 . 5 2 \pm 0 . 6 1$ </td><td> $8 3 . 9 6 \pm 0 . 9 4$ </td><td> $8 4 . 2 6 \pm 1 . 1 3$ </td><td> $\mathbf { 8 6 . 1 4 \ : \pm 0 . 7 6 }$ </td></tr><tr><td>Claude-Sonnet-4.5</td><td> $8 5 . 3 7 \pm 0 . 9 8$ </td><td> $9 0 . 8 4 \pm 0 . 6 5$ </td><td> $9 0 . 6 6 \pm 0 . 6 3$ </td><td> $9 0 . 9 2 \pm 0 . 3 2$ </td><td> $\mathbf { 9 } 2 . \mathbf { 0 3 } \pm 0 . 6 4$ </td></tr><tr><td>Llama-3.1-8B</td><td> $6 2 . 8 3 \pm 0 . 8 0$ </td><td> $6 4 . 1 4 \pm 1 . 7 1$ </td><td> $5 5 . 9 8 \pm 0 . 7 5$ </td><td> ${ \bf 7 1 . 2 6 \pm 0 . 7 0 }$ </td><td> $6 8 . 0 6 \pm 0 . 8 6$ </td></tr><tr><td>Gemma3-4B</td><td> $5 9 . 7 6 \pm 1 . 0 8$ </td><td> $6 0 . 2 9 \pm 0 . 8 3$ </td><td> $5 8 . 7 2 \pm 1 . 0 8 $ </td><td> ${ \bf 6 3 . 1 0 \pm 1 . 0 3 }$ </td><td> $6 2 . 1 2 \pm 1 . 5 4$ </td></tr><tr><td rowspan="5">Spider</td><td> $\mathrm { G P T } { \cdot } 4 0$ </td><td> $6 9 . 1 5 \pm 1 . 7 8$ </td><td> $6 9 . 0 5 \pm 1 . 3 2$ </td><td> $7 1 . 7 6 \pm 1 . 5 7$ </td><td> $7 1 . 4 7 \pm 0 . 9 3$ </td><td> $7 3 . 0 2 \pm 1 . 4 0$ </td></tr><tr><td>Gemini-Flash-2.0</td><td> $\mathbf { 7 9 . 4 0 \pm 0 . 8 4 }$ </td><td> $7 7 . 3 7 \pm 1 . 3 3$ </td><td> $7 8 . 2 0 \pm 1 . 0 9$ </td><td> $7 6 . 6 9 \pm 1 . 3 4$ </td><td> $7 7 . 1 8 \pm 1 . 3 3$ </td></tr><tr><td>Claude-Sonnet-4.5</td><td> $7 7 . 4 7 \pm 1 . 6 0$ </td><td> $7 7 . 4 7 \pm 1 . 2 3 $ </td><td> $7 8 . 0 5 \pm 1 . 2 0$ </td><td> $7 7 . 3 7 \pm 1 . 3 6$ </td><td> $7 7 . 6 6 \pm 1 . 5 2$ </td></tr><tr><td>Llama-3.1-8B</td><td> ${ \bf 6 4 . 9 9 \pm 1 . 8 5 }$ </td><td> $5 8 . 0 3 \pm 0 . 8 6$ </td><td> $6 4 . 3 1 \pm 1 . 1 5$ </td><td> $5 4 . 8 4 \pm 2 . 7 1$ </td><td> $6 2 . 1 9 \pm 1 . 6 0$ </td></tr><tr><td> $\mathrm { G e m m a } 3 – 4 \mathrm { B }$ </td><td> ${ \bf 7 0 . 2 1 \pm 1 . 3 5 }$ </td><td> $6 6 . 1 5 \pm 1 . 4 8$ </td><td> $6 8 . 3 8 \pm 0 . 8 9$ </td><td> $6 7 . 5 0 \pm 0 . 9 7$ </td><td> $6 3 . 3 5 \pm 0 . 5 7$ </td></tr></table>

Table 1: Impact of output structure constraints on accuracy across benchmark tasks. We compare Answer-first (answer before reasoning) and Reasoning-first (reasoning before answer) orderings under JSON and XML formats, against an unconstrained Free Form baseline. Values are mean accuracy over 5 independent runs (standard deviation in gray). Best format per model–dataset pair is bolded and highlighted.

## 5 Calibration and Confidence Estimation

Accuracy gains from structured formatting are only meaningful if models also communicate reliable uncertainty. We ask whether output format affects calibration i.e. alignment between stated confidence and empirical accuracy. Confidence is elicited as a verbal 1–10 integer score and calibration is measured via ECE; elicitation strategy and logitverbal alignment are explained in Appendix C. Because the accuracy experiments in Section 4 identify XML with reasoning-first ordering, (R,A), as the recommended structured format, we evaluate the calibration of this configuration against freeform generation. The results show that the recommended XML (R,A) format also reduces expected calibration error relative to free-form output.

results: format sensitivity is highest on tasks lacking inherent structure and negligible where intrinsic scaffolding already exists. LogicBench shows consistent gains under XML across all models; MMLU shows gains for four of five models, the exception being GPT-4o where freeform is marginally better calibrated. GSM8K is effectively formatinvariant — arithmetic provides its own sequential discipline that format cannot augment. Spider (the high-structure task) is the dataset where freeform is better calibrated for some model, again consistent with SQL imposing its own structural constraints. These results establish XML formatting not merely as an accuracy intervention but as a calibration mechanism: it regularises confidence expression alongside reasoning quality.

## 5.1 Output Format as a Calibration Mechanism

## 5.2 Calibration Profiles Under Structured Output

Figure 2 reports ∆ECE = $\mathrm { E C E } _ { \mathrm { f r e e f o r m } }$ − ECE<sub>XML</sub>; positive values indicate freeform is worse calibrated. The pattern closely mirrors the accuracy

Table 2 reports ECE, overconfidence rate (OvConf), and Spearman ρ between confidence and correctness under XML prompting. The task-level ordering is stable across models: GSM8K is best calibrated, Spider is worst, with LogicBench and MMLU in between. This ordering reflects how much intrinsic task structure guides confidence. Arithmetic answers are either right or wrong in a verifiable chain, while SQL execution correctness is opaque to the model at generation time Across all settings, the confidence-accuracy correlation ρ remains low, indicating that while XML improves aggregate calibration, per-example discrimination is limited. Appendix Figures 4 and 5 visualize the confidence–accuracy gap and per-bin calibration across all model–dataset conditions.

![](images/c6bcfd8e721e81641595516fc990b6662c6ea42007d67c0345050f4abf435778.jpg)  
Figure 2: ECE increase under freeform relative to XML $( \Delta \mathrm { E C E } = \mathrm { E C E } _ { \mathrm { f r e e f o r m } } - \mathrm { E C E } _ { \mathrm { X M L } } )$ . Positive values (red) indicate freeform is worse calibrated; negative (green) indicates the reverse. LogicBench shows the largest format sensitivity; GSM8K is format-invariant.

<table><tr><td>Dataset</td><td>Model</td><td>ECE↓</td><td>OvConf↓</td><td>Corr (ρ)↑</td></tr><tr><td rowspan="5">GSM8K</td><td>GPT-40</td><td>0.050±0.005</td><td>0.051±0.005</td><td>0.169±0.028</td></tr><tr><td>Gemini-Flash-2.0</td><td>0.042±0.002</td><td>0.043±0.002</td><td>0.134±0.084</td></tr><tr><td>Claude-Sonnet-4.5</td><td>0.020±0.004</td><td>0.022 ±0.004</td><td>0.264±0.063</td></tr><tr><td>Llama-3.1-8B</td><td>0.136±0.008</td><td>0.138±0.009</td><td>0.189±0.053</td></tr><tr><td>Gemma3-4B</td><td>0.109±0.028</td><td>0.111±0.029</td><td>0.350±0.146</td></tr><tr><td rowspan="5">LogicBench</td><td>GPT-40</td><td>0.097 ±0.018</td><td>0.197 ±0.018</td><td>0.253±0.078</td></tr><tr><td>Gemini-Flash-2.0</td><td>0.097±0.017</td><td>0.131 ±0.022</td><td>0.291 ±0.049</td></tr><tr><td>Claude-Sonnet-4.5</td><td>0.060 ±0.009</td><td>0.108 ±0.012</td><td>0.251 ±0.037</td></tr><tr><td>Llama-3.1-8B</td><td>0.103±0.016</td><td>0.244±0.020</td><td>0.222±0.064</td></tr><tr><td>Gemma3-4B</td><td>0.243±0.015</td><td>0.337±0.015</td><td>0.256±0.042</td></tr><tr><td rowspan="5">MMLU</td><td>GPT-40</td><td>0.073±0.005</td><td>0.132±0.006</td><td>0.289±0.023</td></tr><tr><td>Gemini-Flash-2.0</td><td>0.104±0.010</td><td>0.131 ±0.011</td><td>0.282 ±0.040</td></tr><tr><td>Claude-Sonnet-4.5</td><td>0.041 ±0.007</td><td>0.082 ±0.004</td><td>0.396±0.026</td></tr><tr><td>Llama-3.1-8B</td><td>0.208 ±0.014</td><td>0.321 ±0.012</td><td>0.173 ±0.021</td></tr><tr><td>Gemma3-4B</td><td>0.292±0.016</td><td>0.365±0.014</td><td>0.214±0.026</td></tr><tr><td rowspan="5">Spider</td><td>GPT-40</td><td>0.201 ±0.018</td><td>0.258±0.017</td><td>0.247±0.025</td></tr><tr><td>Gemini-Flash-2.0</td><td>0.210±0.011</td><td>0.211 ±0.010</td><td>0.115±0.032</td></tr><tr><td>Claude-Sonnet-4.5</td><td>0.185 ±0.016</td><td>0.235 ±0.015</td><td>0.299±0.038</td></tr><tr><td>Llama-3.1-8B</td><td>0.273 ±0.009</td><td>0.351±0.009</td><td>0.171±0.018</td></tr><tr><td>Gemma3-4B</td><td>0.283±0.006</td><td>0.326±0.008</td><td>0.331 ±0.050</td></tr></table>

Table 2: Calibration under XML-structured output. ECE and OvConf are lower-is-better; Corr (ρ) is higher-isbetter. Values are mean over 5 runs (std. in gray). Best model per metric–dataset pair is bolded.

Elicitation strategy and logit-verbal alignment. Absolute elicitation generally outperforms distributional prompting and is used as the default (Appendix C.2). Verbal scores correlate moderately with logits only in stronger models; smaller models show little alignment, with large KL divergence between verbal and token-level uncertainty signals (Appendix C.3). Thus, for smaller models, verbalized confidence is better viewed as a taskconditioned heuristic than a reflection of internal uncertainty.

## KEY TAKEAWAY

Output format acts as a calibration mechanism, not just an accuracy intervention: XML (R,A) reduces ECE alongside improving reasoning quality, with gains following the same inverse-structure gradient observed in Section 4.

## 6 Internal Representations and Mechanism

The accuracy and calibration results establish what field ordering does; this section asks why. We open the model’s intermediate computations with two complementary tools. CKA (Kornblith et al., 2019) acts as a representational separability thermometer: given hidden states for correct and incorrect predictions at each layer, lower cross-group CKA means the model has built more geometrically distinct internal representations for the two outcomes. Linear probes act as an information readout: a linear classifier trained on hidden states at layer ℓ tells us whether the answer identity is already decodable at that depth, before it is generated. A third lens reasoning length connects the surface output to the representational evidence. All three converge on the same account of how field ordering shapes computation.

## 6.1 Representational Separability: CKA

We apply CKA to Llama-3.1-8B & Gemma3- 4B under XML (R,A), XML (A,R), and freeform across all four datasets; extraction details are in Appendix D.1.

Figure 3 (left) shows that format sensitivity tracks the same inverse-structure gradient seen in accuracy and calibration. On MMLU the task with the least inherent structure freeform cross-group CKA rises steadily from early layers and stays elevated, whereas XML (R,A) keeps it below 0.15 through the middle layers: the model is building representations that separate right from wrong only when the output schema forces deliberate reasoning first. LogicBench shows a moderate version of the same pattern, converging in late layers as both conditions reach the same reasoning conclusion. GSM8K exhibits near-parity across formats arithmetic provides its own sequential scaffold while Spider reverses the gradient entirely, the only task where freeform yields lower cross-group CKA than XML, consistent with SQL imposing its own compositional structure that XML tags disrupt. Critically, the direction of the CKA effect correctly predicts the direction of the accuracy effect in all four tasks: Spider is the only task where XML reduces accuracy and the only task where the CKA ratio falls below parity. Gemma3-4B replicates this pattern at reduced magnitude (Appendix D.4), confirming findings are not model-specific.

![](images/558a40be4e8d3d13820040b2863088df903f3f54b8260b13b5fa0fe300b10b95.jpg)

![](images/0d119ecf1982eb77146ffa884135742ca745446c85389e0b5d04b7185c790cf6.jpg)

![](images/c49c80c70558a35c2a6538e04ac326e7146c2cb8491f5ca9c3a45c19b059e8ab.jpg)  
(a) Cross-group CKA per layer

![](images/fcd4759e5bffec74b3258732824550351fe955b93a8d0414c23feb59e2b6d3ab.jpg)

![](images/ebe032dc03d480f0c51fd418e181a806954d15a627ba6bfc4986d89dc0ff9134.jpg)  
(b) Linear probe balanced accuracy per layer  
Figure 3: Internal representations for Llama-3.1-8B on MMLU and LogicBench. Left: Cross-group CKA under XML (R,A) (solid), XML (A,R) (dash-dot), and Freeform (dashed) across four benchmarks. Higher CKA = weaker separation between correct and incorrect representations; R,A yields the lowest mid-layer CKA on reasoning tasks while Spider reverses the pattern. Right: Linear probe balanced accuracy decoding the predicted answer label from hidden states at each layer. Solid = (R,A); dashed = (A,R); dotted = chance; gap panel shows the (R,A)−(A,R) advantage with peak layer annotated. Gemma3-4B replicates both patterns at reduced magnitude (Appendix D.4).

An instructive dissociation appears on MMLU: despite the largest representational suppression under freeform prompting, Llama gains only modest accuracy compared to LogicBench. This highlights the limit of format scaffolding i.e., accuracy remains constrained by the model’s underlying knowledge. XML can enforce deliberate ordering, not supply missing facts. Within-group CKA heatmaps further show that XML induces clearer stage-wise block structure in correct representations (Appendix, Figures 6–7).

## 6.2 Answer Encoding: Linear Probes

While CKA measures whether representations are separable, probes ask a more targeted question: does the model already know the answer during the reasoning phase, before the answer token is generated? We train a linear classifier on pre-answer hidden states at each layer to decode the predicted label and compare accuracy under (R,A) vs (A,R) on MMLU and LogicBench (GSM8K and Spider excluded; see Appendix D.3).

The answer is unambiguous (Figure 3, right). Under (R,A), answer identity becomes linearly decodable from middle layers onward, peaking at balanced accuracy of 0.64 (MMLU, L19) and 0.70 (LogicBench, L18) well above chance and sustained across the final third of the network. Under (A,R), the same probe stagnates near chance through early and middle layers: the model has committed to an answer token but has not built a representation of which answer it chose during the reasoning phase the answer is present in the output stream. With the answer field first, the model must emit the answer letter without preceding explicit reasoning. The pre-answer-letter state may therefore reflect less task-relevant computation, increasing the risk that commitment precedes deliberate inference. Gemma3-4B replicates the directional pattern at reduced magnitude (Appendix D.4), again confirming this is a general property of field ordering, not an artefact of one model family.

## 6.3 Reasoning Length and the Two Failure Modes

The probe and CKA results describe what happens inside the model; reasoning length reveals the surface signature. We measure word count across 80 cells (5 models × 4 datasets × 4 format-orderings) as a proxy for reasoning effort; extraction details and full results appear in Appendix E.

The data reveal two qualitatively distinct failure modes of the (A,R) ordering. The first is rationalization: on MCQ tasks, committing the answer token causes the reasoning field to collapse JSON (A,R) produces just 33 words on MMLU against 121 under XML (R,A), a fourfold compression into a brief post-hoc justification rather than deliberate inference. The second is premature commitment: on GSM8K, reasoning length is flat across all four orderings (108–127 words), yet accuracy collapses from 0.92 under R,A to 0.55 under JSON (A,R) the model writes as much, but arithmetic that should determine the answer is instead forced to justify one already committed.

Length alone does not predict accuracy $( r \ =$ +0.04 overall); the ordering dimension carries the signal. JSON (A,R)’s correlation of $r = - 0 . 7 3$ $( p < 0 . 0 0 1 )$ the only significant result across all four format-orderings (Appendix Table 9) arises because the cells where JSON (A,R) does not compress (GSM8K, Spider) are precisely those where accuracy is worst: verbose post-commitment reasoning is worse than compressed post-commitment reasoning, because ordering determines whether effort precedes or follows the decision.

## KEY TAKEAWAY

CKA, linear probes, and reasoning-length analysis all support the same conclusion: reasoning-first ordering lets the model form clear answer-related representations before committing to a final response, preserving deliberate reasoning. In contrast, answerfirst ordering leads to either compressed posthoc rationalization in MCQ tasks or incorrect computation despite long reasoning traces in arithmetic tasks. XML (R,A) avoids both by allowing the answer to be generated only after the reasoning process is fully developed.

## 7 Discussion and Conclusion

Two failure modes: practical diagnostics. The two failure modes share a single underlying cause autoregressive generation forces the answer token to be sampled either before or after the reasoning trace but they call for different diagnostics in deployment. Rationalization on MCQ tasks is detectable from output length alone: a sharp drop in reasoning-field token count under (A,R) relative to (R,A) is a reliable signal that the model is producing post-hoc justification rather than deliberate inference. Premature commitment on arithmetic tasks is harder to detect, because reasoning length is preserved (108–127 words across all four orderings on GSM8K); here the diagnostic must check whether the reasoning is consistent with the stated answer, not merely whether reasoning is present. Practitioners auditing structured pipelines for reasoning failures should therefore monitor both the volume of the reasoning field and its semantic alignment with the committed answer.

The “structure tax” attributed to structured output constraints is largely an artefact of answer-first field ordering rather than structure per se. Autoregressive generation is strictly causal, so an answer token emitted before its rationale cannot be revised by it; reasoning-first (R,A) schemas avoid this and consistently match or exceed free-form accuracy across five models and four datasets, while answerfirst schemas produce either compressed post-hoc rationalization on MCQ tasks or premature commitment on arithmetic tasks. The format-sensitivity gradient is consistent across accuracy (Table 1), calibration (Figure 2), and hidden-state geometry (Figure 3), with gains largest on low-structure tasks and negligible on high-structure tasks.

Scope and applicability. The gains we report concentrate where the task lacks intrinsic structure and the model is mid-sized: low-structure benchmarks (MMLU, LogicBench) under smaller openweight models show the largest absolute (R,A) advantages and the most dramatic (A,R) collapses. On high-structure tasks like text-to-SQL, where compositional constraints are imposed by the output language itself, field ordering matters less and the choice between (R,A) and free-form is close to a wash. Frontier models are sensitive to ordering but rarely catastrophic under (A,R), suggesting larger models partially compensate for the causalordering constraint. For practitioners, reasoningfirst XML is therefore the safer default: it never meaningfully hurts on the tasks where structure matters, and it preserves accuracy and calibration on the tasks where it does—without sacrificing downstream parseability.

## Limitations

A recurring pattern across our analyses is that format sensitivity scales inversely with a task’s inherent structural organisation Spider at the highstructure end, MMLU and LogicBench at the lowstructure end, GSM8K between them. Our four datasets sample this continuum at only four coarse points, which is sufficient to establish the inversescaling trend but insufficient to formally characterise a sweet spot — the level of intrinsic structural density at which format scaffolding is maximally beneficial. Formally locating such a sweet spot would require systematic variation of structural density within a single task family (e.g., varying constraint density in SQL generation, or step-count and carry-chain length in arithmetic), holding all other factors constant. We flag this as a concrete and important direction for future work.

Representational analysis via CKA and linear probes is restricted to two open-weight models (Llama-3.1-8B and Gemma3-4B), leaving open whether the observed middle-layer patterns generalise across model scales and architectures. We examine only two field orderings (answer-first and reasoning-first); real-world schemas involve finergrained structure such as planning fields, intermediate steps, and verification stages, and how structural granularity beyond binary ordering affects reasoning remains an open question. All experiments rely on prompting rather than training-time adaptation, so it is unclear whether fine-tuning on structured formats would produce the same representational geometry. Finally, extending the analysis crosslingually may reveal whether format scaffolding compensates more strongly in settings where the model has weaker internalised task structure.

## References

Anthropic. 2025. Claude sonnet 4.5. Accessed: 2026- 03-16.

Xiaoshu Chen, Sihang Zhou, Ke Liang, and Xinwang Liu. 2025. Distilling reasoning ability from large language models with adaptive thinking. Preprint, arXiv:2404.09170.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias

Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. 2021. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168.

Tianyu Cui, Yogesh Kumar, Pekka Marttinen, and Samuel Kaski. 2022. Deconfounded representation similarity for neural networks. In Advances in Neural Information Processing Systems (NeurIPS).

MohammadReza Davari, Stefan Horoi, Amine Natik, Guillaume Lajoie, Guy Wolf, and Eugene Belilovsky. 2023. Reliability of CKA as a similarity measure in deep learning. In The Eleventh International Conference on Learning Representations.

Tairan Fu, Javier Conde, Gonzalo Martinez, Maria Grandury, and Pedro Reviriego. 2026. Multiple choice questions: Reasoning makes large language models (llms) more self-confident, specially when they are wrong. IEEE Intelligent Systems, page 1–10.

Google. 2025. Gemini flash 2.0 model documentation. Accessed: 2026-03-16.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, Amy Yang, Angela Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, and 542 others. 2024. The llama 3 herd of models. Preprint, arXiv:2407.21783.

Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. 2017. On calibration of modern neural networks. In Proceedings ofthe 34th International Conference on Machine Learning (ICML), volume 70 of Proceedings of Machine Learning Research, pages 1321–1330. PMLR.

Jia He, Mukund Rungta, David Koleczek, Arshdeep Sekhon, Franklin X Wang, and Sadid Hasan. 2024. Does prompt formatting have any impact on llm performance? Preprint, arXiv:2411.10541.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2021. Measuring massive multitask language understanding. In International Conference on Learning Representations.

Aishwarya Kamath, Johan Ferret, Shreya Pathak, Nino Vieillard, Ramona Merhej, Sarah Perrin, Tatiana Matejovicova, Alexandre Ramé, Morgane Rivière, Louis Rouillard, Thomas Mesnard, Geoffrey Cideron, Jean bastien Grill, Sabela Ramos, Edouard Yvinec, Michelle Casbon, Etienne Pot, Ivo Penchev, Gaël Liu, and 196 others. 2025. Gemma 3 technical report. Preprint, arXiv:2503.19786.

Simon Kornblith, Mohammad Norouzi, Honglak Lee, and Geoffrey Hinton. 2019. Similarity of neural network representations revisited. In Proceedings of the 36th International Conference on Machine Learning (ICML), volume 97 of Proceedings ofMachine Learning Research, pages 3519–3529. PMLR.

Vineet Kumar, Ronald Tony, Darshita Rathore, Vipasha Rana, Bhuvanesh Mandora, . Kanishka, Chetna Bansal, and Anindya Moitra. 2025. Genicious: Contextual few-shot prompting for insights discovery. In Proceedings ofthe 8th International Conference on Data Science and Management ofData (12th ACM IKDD CODS and 30th COMAD), CODS-COMAD ’24, page 405–409, New York, NY, USA. Association for Computing Machinery.

Tamera Lanham, Anna Chen, Ansh Radhakrishnan, Benoit Steiner, Carson Denison, Danny Hernandez, Dustin Li, Esin Durmus, Evan Hubinger, Jackson Kernion, Kamile Lukoši ˙ ut¯ e, Karina Nguyen, Newton˙ Cheng, Nicholas Joseph, Nicholas Schiefer, Oliver Rausch, Robin Larson, Sam McCandlish, Sandipan Kundu, and 11 others. 2023. Measuring faithfulness in chain-of-thought reasoning. Preprint, arXiv:2307.13702.

Do Xuan Long, Ngoc-Hai Nguyen, Tiviatis Sim, Hieu Dao, Shafiq Joty, Kenji Kawaguchi, Nancy F. Chen, and Min-Yen Kan. 2025. LLMs are biased towards output formats! systematically evaluating and mitigating output format bias of LLMs. In Proceedings ofthe 2025 Conference ofthe Nations ofthe Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 299–330, Albuquerque, New Mexico. Association for Computational Linguistics.

Matthias Minderer, Josip Djolonga, Rob Romijnders, Frances Ann Hubis, Xiaohua Zhai, Neil Houlsby, Dustin Tran, and Mario Lucic. 2021. Revisiting the calibration of modern neural networks. In Advances in Neural Information Processing Systems.

OpenAI. 2024. Hello gpt-4o. Accessed: 2026-03-16.

Hadas Orgad, Michael Toker, Zorik Gekhman, Roi Reichart, Idan Szpektor, Hadas Kotek, and Yonatan Belinkov. 2025. LLMs know more than they show: On the intrinsic representation of LLM hallucinations. In The Thirteenth International Conference on Learning Representations.

Mihir Parmar, Nisarg Patel, Neeraj Varshney, Mutsumi Nakamura, Man Luo, Santosh Mashetty, Arindam Mitra, and Chitta Baral. 2024. LogicBench: Towards systematic evaluation of logical reasoning ability of large language models. In Proceedings ofthe 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 13679– 13707, Bangkok, Thailand. Association for Computational Linguistics.

Darshita Rathore, Vineet Kumar, Vaibhav Singal, Ankur Vivek Singh, and Anindya Moitra. 2026. Conversational query engine for mixed-modality heterogeneous enterprise data sources. Preprint, arXiv:2606.28370.

Melanie Sclar, Yejin Choi, Yulia Tsvetkov, and Alane Suhr. 2024. Quantifying language models’ sensitivity to spurious features in prompt design or: How i

learned to start worrying about prompt formatting. In The Twelfth International Conference on Learning Representations.

Giovanni Servedio, Alessandro De Bellis, Dario Di Palma, Vito Walter Anelli, and Tommaso Di Noia. 2025. Are the hidden states hiding something? testing the limits of factuality-encoding capabilities in LLMs. In Proceedings of the 63rd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 6089–6104, Vienna, Austria. Association for Computational Linguistics.

Zhi Rui Tam, Cheng-Kuang Wu, Yi-Lin Tsai, Chieh-Yen Lin, Hung-yi Lee, and Yun-Nung Chen. 2024. Let me speak freely? a study on the impact of format restrictions on large language model performance. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing: Industry Track, pages 1218–1236, Miami, Florida, US. Association for Computational Linguistics.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, brian ichter, Fei Xia, Ed H. Chi, Quoc V Le, and Denny Zhou. 2022. Chain of thought prompting elicits reasoning in large language models. In Advances in Neural Information Processing Systems.

Tao Yu, Rui Zhang, Kai Yang, Michihiro Yasunaga, Dongxu Wang, Zifan Li, James Ma, Irene Li, Qingning Yao, Shanelle Roman, Zilin Zhang, and Dragomir Radev. 2018. Spider: A large-scale human-labeled dataset for complex and cross-domain semantic parsing and text-to-SQL task. In Proceedings ofthe 2018 Conference on Empirical Methods in Natural Language Processing, pages 3911–3921, Brussels, Belgium. Association for Computational Linguistics.

Anqi Zhang, Yulin Chen, Jane Pan, Chen Zhao, Aurojit Panda, Jinyang Li, and He He. 2025. Reasoning models know when they’re right: Probing hidden states for self-verification. In Second Conference on Language Modeling.

## Appendix Contents

A Complete Prompt Collection 11   
A.1 Structured Format Evaluation 11   
A.2 Calibration Analysis Prompts 13   
A.3 Schema violation statistics 13   
A.4 Logit Confidence Analysis Prompts 13   
B Dataset Summary Statistics 14   
C Calibration: Supplementary Details 14   
C.1 Implementation Details 14   
C.2 Confidence Elicitation: Absolute vs. Relative 14   
C.3 Verbal Confidence vs. Internal Probabilities 14   
C.4 Calibration Visualisations 15   
D CKA and Probe: Method Details 16   
D.1 CKA Extraction 16   
D.2 Within-Group CKA Heatmaps: Layout and Reading Guide 17   
D.3 Linear Probe Setup 23   
D.4 Gemma3-4B: Representational Results 23   
E Reasoning Length: Measurement and Results 24   
F Prompt-Brittleness Ablation 24

## A Appendix: Complete Prompt Collection

This appendix documents all prompts used in our three experimental conditions: Structured Format Evaluation (§A.1), Calibration Analysis (§A.2), and Logit Confidence Analysis (§A.4).

## A.1 Structured Format Evaluation GSM8K (Mathematical Reasoning).

[XML AR] You are a math expert. Solve step-by-step in XML format.   
<answer> final numerical answer </answer>   
<reasoning> step-by-step solution </reasoning>   
Problem: {problem}

[XML RA] You are a math expert. Solve step-by-step in XML format.   
<reasoning> step-by-step solution </reasoning>   
<answer> final numerical answer </answer>   
Problem: {problem}

```jsonl
[JSON AR] You are a math expert. Solve step-by-step in JSON format.
{ "answer": "final answer", "reasoning": "solution" }
Problem: {problem}
```

```jsonl
[JSON RA] You are a math expert. Solve step-by-step in JSON format.
{ "reasoning": "solution", "answer": "final answer" }
Problem: {problem}
```

[Freeform] You are a math expert. Solve step-by-step.   
Problem: {problem}

## MMLU (Multiple-Choice Knowledge).

[XML AR] You are a helpful AI assistant. Analyze and provide your response in XML format.   
<answer> option letter (A–D) </answer>   
<reasoning> analysis </reasoning>   
Problem: {problem} Options: {options}

[XML RA] You are a helpful AI assistant. Analyze and provide your response in XML format.   
<reasoning> analysis and thought process </reasoning>

```yaml
<answer> option letter (A–D) </answer>
Problem: {problem} Options: {options}
```

```jsonl
[JSON AR] You are a helpful AI assistant. Analyze and provide your response in JSON format.
{ "answer": "option letter (A–D)", "reasoning": "explanation" }
Problem: {problem} Options: {options}
```

[JSON RA] You are a helpful AI assistant. Analyze and provide your response in JSON format.   
{ "reasoning": "explanation", "answer": "option letter (A–D)" }   
Problem: {problem} Options: {options}

[Freeform] You are a helpful AI assistant. Solve this MCQ.   
Problem: {problem} Options: {options}

## LogicBench (Logical Reasoning).

[XML AR] You are a reasoning expert. Answer in XML format.   
<answer> option letter </answer>   
<reasoning> analysis </reasoning>   
Context: {context} Problem: {question} Options: {options}

[XML RA] You are a reasoning expert. Answer in XML format.   
<reasoning> analysis </reasoning>   
<answer> option letter </answer>   
Context: {context} Problem: {question} Options: {options}

```jsonl
[JSON AR] You are a reasoning expert. Answer in JSON format.
{ "answer": "option letter", "reasoning": "explanation" }
Context: {context} Problem: {question} Options: {options}
```

[JSON RA] You are a reasoning expert. Answer in JSON format.   
{ "reasoning": "explanation", "answer": "option letter" }   
Context: {context} Problem: {question} Options: {options}

[Freeform] You are a reasoning expert. Answer the following question.   
Context: {context} Problem: {question} Options: {options}

## Spider (Text-to-SQL).

[XML AR] You are a SQL expert. Generate query in XML format.   
<answer> SQL query </answer>   
<reasoning> rationale </reasoning>   
Schema: {schema} Question: {question}

[XML RA] You are a SQL expert. Generate query in XML format.   
<reasoning> rationale </reasoning>   
<answer> SQL query </answer>   
Schema: {schema} Question: {question}

[JSON AR] You are a SQL expert. Return query in JSON.   
{ "answer": "SQL query", "reasoning": "explanation" }   
Schema: {schema} Question: {question}

[JSON RA] You are a SQL expert. Return query in JSON.   
{ "reasoning": "explanation", "answer": "SQL query" }   
Schema: {schema} Question: {question}

[Freeform] You are a SQL expert. Generate a SQL query.   
Schema: {schema} Question: {question}

## A.2 Calibration Analysis Prompts

Calibration prompts extend each format with an explicit confidence request on a 1–10 scale. Below we show the XML and Freeform variants; JSON calibration follows the same JSON structure as §A.1 with an additional "confidence" field.

XML Calibration. All four datasets share the same XML structure extended by a <confidence> field:

```html
[XML + Confidence] (role and task instruction as above)
<reasoning> analysis </reasoning>
<answer> answer </answer>
<confidence> integer 1–10 (1 = not confident, 10 = very confident) </confidence>
Question fields: identical to Structured Format prompts above.
```

Freeform Calibration.

[Freeform + Confidence — MMLU]   
You are a helpful AI assistant. Answer the question directly. Before you finish, state your   
confidence level on a scale of 1–10 (1 = not confident, 10 = very confident).   
Problem: {problem} Options: {options}

[Freeform + Confidence — GSM8K]   
You are a math expert. Solve step-by-step. At the end, state your confidence level on a scale of   
1–10.   
Problem: {problem}

[Freeform + Confidence — LogicBench]   
You are a reasoning expert. Answer the following question. State your confidence on a scale of 1–10.   
Context: {context} Problem: {question} Options: {options}

[Freeform + Confidence — Spider]   
You are a SQL expert. Generate a SQL query. Before you finish, state your confidence on a scale of   
1–10.   
Schema: {schema} Question: {question}

## A.3 Schema-violation statistics

Structured-output schema violations were zero for all experiments. The only parsing failures occurred in free-form outputs due to the Gemma3-4B.

Table 3: Free-form empty-extraction rates caused by the Gemma3-4B parser bug.
<table><tr><td>Dataset</td><td>Empty Extraction Rate</td></tr><tr><td>MMLU</td><td>29.8%</td></tr><tr><td>LogicBench</td><td>45.2%</td></tr></table>

## A.4 Logit Confidence Analysis Prompts

Logit-based confidence extraction uses minimal prompts that elicit a single-token answer, enabling direct extraction of confidence from the token probability distribution over {A, B, C, D}.

[Logit — MMLU & LogicBench]   
Answer the following question with only the single letter of the correct option (A, B, C, or D). Do   
not include any explanation or punctuation — just the letter.   
Problem: {problem} Options: {options} Answer:

Confidence extraction. For logit prompts, we record the softmax probability assigned to each option token at the Answer: position. For XML/JSON prompts, we parse the <confidence> or "confidence" field and normalise to [0, 1]. For freeform prompts, confidence is extracted via regex (e.g., "confidence[:\s]+(\d+)").

Reproducibility. All experiments use temperature = 0.0 to ensure deterministic outputs. Prompts are identical across models; no model-specific tuning was applied.

## B Dataset summary statistics

<table><tr><td>Dataset</td><td>Split</td><td>Examples</td><td>Task Type</td><td>Answer Format</td></tr><tr><td>GSM8K</td><td>Test</td><td>1,319</td><td>Math reasoning</td><td>Numeric</td></tr><tr><td>LogicBench</td><td>Eval</td><td>500</td><td>Logical inference</td><td>MCQ</td></tr><tr><td>MMLU</td><td>Validation</td><td>1,531</td><td>Knowledge QA</td><td>MCQ</td></tr><tr><td>Spider</td><td>Validation</td><td>1,030</td><td>Text-to-SQL</td><td>SQL query</td></tr></table>

Table 4: Dataset statistics and properties. All splits are held-out evaluation sets; no training data is used. Spider is evaluated via execution accuracy; all other datasets use exact-match accuracy.

## C Calibration: Supplementary Details

## C.1 Implementation Details

Calibration is measured with Expected Calibration Error (ECE) using M = 10 equal-width bins over [0, 1], computed as $\begin{array} { r } { \mathrm { E C E } = \sum _ { m = 1 } ^ { 1 0 } \frac { | B _ { m } | } { N } | \mathrm { a c c } ( B _ { m } ) - \mathrm { c o n f } ( B _ { m } ) | } \end{array}$ , where the final bin is closed on both sides. Confidence is elicited as a verbal integer on a 1–10 scale (“1 = not confident, 10 = very confident”) and normalized by dividing by 10. Scores are extracted from free-text responses via regex matching on the pattern confidence $[ : \backslash { \mathsf { s } } ] { \mathsf { + } } ( \backslash { \mathsf { d } } + )$ . All API calls use temperature = 0. Brier Score measures mean squared error between predicted probabilities and binary correctness: $\textstyle { \frac { 1 } { N } } \sum _ { i } ( p _ { i } - y _ { i } ) ^ { 2 }$ . Entropy $H ( p ) =$ $\begin{array} { r } { \sum _ { y } p ( y | x _ { i } ) \log p ( y | x _ { i } ) } \end{array}$ measures predictive uncertainty; used under relative confidence elicitation.

## C.2 Confidence Elicitation: Absolute vs. Relative

We compare two elicitation strategies on MMLU and LogicBench (MCQ tasks only, as relative prompting requires a fixed option set). Absolute prompting requests a single scalar score. Relative prompting asks the model to distribute confidence across the answer options (A/B/C/D), from which we extract the chosen-option probability. Table 5 reports ECE, average confidence, entropy, and Brier score.

Absolute prompting achieves lower or equal ECE in 6 of 8 model–dataset pairs. Relative prompting produces lower average confidence (−0.035 to −0.16) and higher entropy (0.47–1.14), indicating models express greater distributional uncertainty, but this does not translate into better-calibrated scalar estimates for most models. The Brier score under relative prompting is consistently worse for weaker models (Llama-3.1-8B: 0.452–0.582), while Claude-Sonnet-4.5 achieves the lowest Brier score under relative prompting on both datasets, suggesting stronger models can exploit distributional elicitation more effectively. Gemma3-4B is an exception to the absolute-wins pattern: relative prompting reduces its ECE by $\Delta = + 0 . 1 3 8$ (LogicBench) and +0.083 (MMLU), the two largest per-model improvements in the comparison, consistent with its tendency toward overconfident scalar responses.

## C.3 Verbal Confidence vs. Internal Probabilities

We compare verbal 1–10 scores against logit-based confidence the softmax probability assigned to the selected answer token on MMLU and LogicBench for GPT-4o, Llama-3.1-8B, and Gemma3-4B (the models for which logits are accessible).

Results show a clear model-size gradient. GPT-4o verbal and logit confidence are moderately correlated $( r _ { s } = 0 . 4 4$ on MMLU, 0.67 on LogicBench) with low KL divergence $( \approx 0 . 5 )$ , suggesting verbal scores partially reflect internal probability estimates. Llama-3.1-8B breaks down on MMLU $( r _ { s } = - 0 . 0 5 $ $p = . 0 7 , \mathrm { K L } = 3 . 6 2 )$ : verbal confidence is effectively decoupled from token-level probabilities, yet logit ECE (0.156) is better than verbal ECE (0.191), making logits the more reliable signal for this model. Gemma3-4B is the most extreme case: near-zero Spearman on MMLU $( r _ { s } = 0 . 0 2 4$ , not significant, KL = 1.676), and logit ECE (0.402) is far worse than verbal ECE (0.212) logits are dominated by the token-level distribution rather than calibrated uncertainty, while verbal confidence, though imprecise, is at least weakly correlated with accuracy. On LogicBench, all models improve: GPT-4o reaches $r _ { s } = 0 . 6 7 $ and even Gemma3-4B achieves a moderate Spearman (0.148), suggesting task difficulty moderates verbal-logit alignment.

<table><tr><td></td><td></td><td colspan="3">ECE↓</td><td colspan="2">AvgConf</td><td rowspan="2">Entropy ↓</td><td rowspan="2">Brier↓</td></tr><tr><td>Dataset</td><td>Model</td><td>Abs.</td><td>Rel.</td><td> $\Delta ^ { \dagger }$ </td><td>Abs.</td><td>Rel.</td></tr><tr><td rowspan="5">LogicBench</td><td>GPT-40</td><td> $0 . 0 7 2 \pm . 0 1 0$ </td><td> $0 . 0 8 4 \pm . 0 1 7$ </td><td> $- 0 . 0 1 2$ </td><td> $0 . 8 8 3 \pm . 0 0 6$ </td><td> $0 . 7 4 1 \pm . 0 0 5$ </td><td> $1 . 1 4 1 \pm . 0 1 5$ </td><td> $0 . 2 9 7 \pm . 0 2 3$ </td></tr><tr><td>Gemini-Flash-2.0</td><td> $0 . 0 9 4 \pm . 0 1 2$ </td><td> $\mathbf { 0 . 0 5 } 2 \pm . 0 0 5$ </td><td> $+ 0 . 0 4 2$ </td><td> $0 . 9 4 1 \pm . 0 0 4$ </td><td> $0 . 7 9 8 \pm . 0 0 4$ </td><td> $0 . 9 3 0 \pm . 0 2 7$ </td><td> $0 . 2 4 4 \pm . 0 1 4$ </td></tr><tr><td>1 Claude-Sonnet-4.5</td><td> ${ \bf 0 . 0 4 5 \pm . 0 0 7 }$ </td><td> $0 . 0 6 6 \pm . 0 1 0$ </td><td> $- 0 . 0 2 1$ </td><td> $0 . 8 7 9 \pm . 0 0 7$ </td><td> $0 . 8 3 4 \pm . 0 0 3$ </td><td> $0 . 7 5 9 \pm . 0 1 4$ </td><td> ${ \bf 0 . 2 0 0 } \pm . 0 1 6$ </td></tr><tr><td>Llama-3.1-8B</td><td> $0 . 1 6 6 \pm . 0 1 5$ </td><td> $0 . 1 6 1 \pm . 0 1 5$ </td><td> $+ 0 . 0 0 5$ </td><td> $0 . 9 1 4 \pm . 0 0 6$ </td><td> $0 . 8 7 8 \pm . 0 0 7$ </td><td> ${ \bf 0 . 4 9 6 } \pm . 0 1 9$ </td><td> $0 . 4 5 2 \pm . 0 3 6$ </td></tr><tr><td>Gemma3-4B</td><td> $0 . 2 9 0 \pm . 0 1 2$ </td><td> $0 . 1 5 2 \pm . 0 1 7$ </td><td>+0.138</td><td> $0 . 9 3 8 \pm . 0 0 2$ </td><td> $0 . 7 7 5 \pm . 0 0 2$ </td><td> $1 . 0 4 6 \pm . 0 1 0$ </td><td> $0 . 5 3 0 \pm . 0 1 9$ </td></tr><tr><td rowspan="5">MMLU</td><td>GPT-40</td><td> $0 . 0 6 3 \pm . 0 0 7$ </td><td> $0 . 1 1 7 \pm . 0 0 7$ </td><td> $- 0 . 0 5 3$ </td><td> $0 . 9 2 4 \pm . 0 0 2$ </td><td> $0 . 7 5 0 \pm . 0 0 5$ </td><td> $1 . 1 4 2 \pm . 0 1 0$ </td><td> $0 . 2 4 4 \pm . 0 1 1$ </td></tr><tr><td>Gemini-Flash-2.0</td><td> $0 . 1 0 3 \pm . 0 0 8$ </td><td> $0 . 0 7 8 \pm . 0 1 1$ </td><td> $+ 0 . 0 2 5$ </td><td> $0 . 9 6 9 \pm . 0 0 2$ </td><td> $0 . 8 0 3 \pm . 0 0 4$ </td><td> $0 . 9 2 1 \pm . 0 1 0$ </td><td> $0 . 2 4 2 \pm . 0 0 7$ </td></tr><tr><td>Claude-Sonnet-4.5</td><td> ${ \bf 0 . 0 2 4 } \pm . 0 0 5$ </td><td> $\mathbf { 0 . 0 5 7 \pm . 0 0 5 }$ </td><td>-0.033</td><td> $0 . 9 3 3 \pm . 0 0 3$ </td><td> $0 . 8 6 4 \pm . 0 0 3$ </td><td> ${ \bf 0 . 6 7 1 } \pm . 0 0 2$ </td><td> $\mathbf { 0 . 1 4 7 \pm . 0 0 7 }$ </td></tr><tr><td>Llama-3.1-8B</td><td> $0 . 2 4 7 \pm . 0 0 8$ </td><td> $0 . 1 9 2 \pm . 0 1 0$ </td><td> $+ 0 . 0 5 5$ </td><td> $0 . 9 2 7 \pm . 0 0 1$ </td><td> $0 . 8 3 4 \pm . 0 0 4$ </td><td> $0 . 6 8 2 \pm . 0 0 4$ </td><td> $0 . 5 8 2 \pm . 0 1 8$ </td></tr><tr><td> $\mathrm { G e m m a } 3 – 4 \mathrm { B }$ </td><td> $0 . 2 8 1 \pm . 0 1 6$ </td><td> $0 . 1 9 8 \pm . 0 2 3$ </td><td> ${ \bf + 0 . 0 8 3 }$ </td><td> $0 . 9 4 8 \pm . 0 0 2$ </td><td> $0 . 7 4 4 \pm . 0 1 2$ </td><td> $1 . 0 9 3 \pm . 0 2 3$ </td><td> $0 . 5 9 1 \pm . 0 2 7$ </td></tr></table>

Table 5: Absolute vs. relative confidence elicitation on MCQ datasets. $^ { \dag } \Delta = \mathrm { E C E } _ { \mathrm { A b s } } - \mathrm { E C E } _ { \mathrm { R e l } } ;$ positive = relative prompting improves calibration. Entropy and Brier defined only under relative prompting. Highlighted $\Delta$ values mark the largest per-column improvements. Values are mean over 5 runs (std. in gray).
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Model</td><td colspan="3">Correlation &amp; Divergence</td><td colspan="2">ECE↓</td><td colspan="2">Brier Score ↓</td></tr><tr><td>Spearman ρ</td><td>Pearson r</td><td>Mean KL</td><td>Logit</td><td>Verbal</td><td>Logit</td><td>Verbal</td></tr><tr><td rowspan="3">MMLU</td><td>GPT-40</td><td>0.439</td><td>0.128</td><td>0.500</td><td>0.117</td><td>0.116</td><td>0.276</td><td>0.247</td></tr><tr><td>Llama-3.1-8B</td><td> $- 0 . 0 4 9 ^ { \dagger }$ </td><td> $- 0 . 0 4 0 ^ { \dagger }$ </td><td>3.617</td><td>0.156</td><td>0.191</td><td>0.496</td><td>0.582</td></tr><tr><td>Gemma3-4B</td><td> $0 . 0 2 4 ^ { \dagger }$ </td><td> $0 . 0 0 2 ^ { \dagger }$ </td><td>1.676</td><td>0.402</td><td>0.212</td><td>0.813</td><td>0.602</td></tr><tr><td rowspan="3">LogicBench</td><td>GPT-40</td><td>0.673</td><td>0.312</td><td>0.423</td><td>0.122</td><td>0.084</td><td>0.274</td><td>0.305</td></tr><tr><td>Llama-3.1-8B</td><td>0.282</td><td>0.230</td><td>2.304</td><td>0.141</td><td>0.162</td><td>0.379</td><td>0.453</td></tr><tr><td>Gemma3-4B</td><td>0.148</td><td>0.099</td><td>0.757</td><td>0.311</td><td>0.146</td><td>0.627</td><td>0.532</td></tr></table>

Table 6: Logit vs. verbal confidence comparison. Spearman $\rho$ and Pearson r measure scalar agreement on the chosen-option score; Mean KL measures full distributional divergence. ECE and Brier assess calibration against correctness. ${ \dag } _ { p } > 0 . 0 5$

## C.4 Calibration Visualisations

Figure 4 plots mean confidence against mean accuracy for each model–dataset combination under freeform (squares) and XML (circles). All models cluster in the high-confidence region (> 85%) regardless of format while accuracy spans a wide range, placing most points well above the diagonal that marks perfect calibration. The format effect appears as a systematic downward-and-rightward shift: XML moves points closer to the diagonal, reducing the confidence–accuracy gap across all conditions. The displacement is largest for weaker models on low-structure tasks, consistent with the ECE reductions in Table 2. GSM8K points cluster near the diagonal under both formats, confirming it as the best-calibrated task irrespective of schema.

Figure 5 shows per-bin reliability diagrams for all 20 model–dataset conditions. Under freeform, highconfidence bins are heavily populated while empirical accuracy lags, producing bars far below the diagonal the visual signature of overconfidence. XML shifts bars toward the diagonal across most configurations, with the largest recalibration on LogicBench and MMLU where freeform calibration is worst. GSM8K bars are near-diagonal under both formats, reinforcing the format-invariance of well-structured tasks. Together, the two figures confirm that the ECE reductions in Table 2 and Figure 2 reflect genuine per-bin recalibration rather than an artefact of aggregation.

![](images/e17fff7a62a8490a9399e77c3cd60dad8006f3ebb9e89dde88d977203ef8d2f3.jpg)  
Figure 4: Accuracy vs. confidence scatter for all 5 models across 4 datasets under freeform (squares) and XML (circles) prompting. All models express high confidence (> 85%) regardless of format, while accuracy varies substantially. The gap between confidence and the diagonal is consistently larger under freeform, most dramatically for Gemma3-4B on MMLU and LogicBench (<26% accuracy, ∼100% confidence) and Llama-3.1-8B on LogicBench (23% accuracy, 93% confidence). XML structured output reduces overconfidence across all models, bringing points closer to the calibration diagonal.

![](images/3a27bac6679c0a73f07f8de1dcb944918ce47c0626d277fe3cf2d8ad2be02600.jpg)  
Figure 5: Reliability diagrams comparing freeform and XML calibration across 20 model–dataset combinations (5 models × 4 datasets, each shown in freeform and XML format). XML bars align closer to the diagonal in most configurations, with the largest improvements on LogicBench where freeform produces severe overconfidence (ECE up to 0.703 for Llama-3.1-8B and 0.645 for Gemma3-4B). Gemma3-4B shows the highest overall freeform ECE, particularly on MMLU (0.721). GSM8K remains comparatively well-calibrated under both formats across all models.

## D CKA and Probe: Method Details

## D.1 CKA Extraction

For each question we run a single forward pass through the frozen model with no generation and extract the last-token hidden state at every layer ℓ, yielding representation matrices $\mathbf { X } _ { \mathrm { C } } ^ { \ell } , \mathbf { X } _ { \mathrm { I } } ^ { \ell } \in \mathbb { R } ^ { n \times d }$ for the correct and incorrect prediction groups respectively (n = 200 examples per dataset, sampled uniformly). Inputs

are formatted identically to the main benchmark runs i.e. XML or freeform, ensuring representational differences are attributable to output format, not input variation. We apply this to Llama-3.1-8B (d = 4096, 33 layers) and Gemma3-4B (d = 2560, 35 layers).

We compute linear CKA via the HSIC estimator:

$$
\operatorname { C K A } ( \mathbf { X } , \mathbf { Y } ) = { \frac { \operatorname { H S I C } ( \mathbf { X } \mathbf { X } ^ { \top } , \mathbf { Y } \mathbf { Y } ^ { \top } ) } { { \sqrt { \operatorname { H S I C } ( \mathbf { X } \mathbf { X } ^ { \top } , \mathbf { X } \mathbf { X } ^ { \top } ) \cdot \operatorname { H S I C } ( \mathbf { Y } \mathbf { Y } ^ { \top } , \mathbf { Y } \mathbf { Y } ^ { \top } ) } } } }
$$

where HSI $\begin{array} { r } { \Upsilon ( K , L ) = \frac { 1 } { ( n - 1 ) ^ { 2 } } \mathrm { t r } ( K _ { c } L _ { c } ) } \end{array}$ and subscript c denotes centering. This yields three quantities per layer: (i) a per-layer cross-group CKA scalar measuring geometric alignment between correct and incorrect representations; (ii) a within-correct inter-layer CKA matrix; and (iii) an analogous withinincorrect matrix. We partition layers into three processing stages Early, Middle, and Late at roughly equal thirds of network depth.

Within-group inter-layer CKA matrices are visualised in Figures 6–7 (Llama-3.1-8B) and Figures 8–9 (Gemma3-4B); see §D.2 for layout and reading guide.

## D.2 Within-Group CKA Heatmaps: Layout and Reading Guide

Matrix construction. For each (model, dataset, format) triple we partition the $n = 2 0 0$ probe examples into two outcome groups: correct $( C ,$ , examples the model answered correctly) and incorrect (I). For each group $g \in \{ C , I \}$ we collect the last-token hidden state at every layer $\ell \in \{ 0 , \ldots , L - 1 \}$ , yielding $\mathbf { X } _ { g } ^ { \ell } \in \mathbb { R } ^ { n _ { g } \times d }$ . The within-group inter-layer CKA matrix $\mathbf { M } _ { g } \in \mathbb { R } ^ { L \times L }$ then has entry

$$
{ \bf M } _ { g } [ i , j ] = \mathrm { C K A } \bigl ( { \bf X } _ { g } ^ { i } , { \bf X } _ { g } ^ { j } \bigr ) ,
$$

i.e. the linear CKA similarity between the population of layer-i and layer-j representations for examples in group g. This produces two $L \times L$ matrices per (format, dataset): one for the correct group and one for the incorrect group. We additionally show their difference, $\mathbf { D } = \mathbf { M } _ { C } - \mathbf { M } _ { I }$ , which isolates representational-geometry differences attributable to outcome.

Panel layout. Each figure is a $2 \times 3$ grid:

• Rows: XML (R,A) (top), Freeform (bottom). Both rows use identically-sampled inputs; only the output format differs.

• Columns: (1) within-correct $\mathbf { M } _ { C } ; ( 2 )$ within-incorrect ${ \mathbf { M } } _ { I } ; ( 3 )$ difference $\mathbf { D } = \mathbf { M } _ { C } - \mathbf { M } _ { I }$

• Colormaps: columns 1–2 use viridis on [0, 1] (CKA is in [0, 1] by construction); column 3 uses a diverging $\mathsf { R d B u \_ r }$ colormap symmetric about zero, with limits ±0.20 for Llama and $\pm 0 . 4 0$ for Gemma to accommodate the larger inter-layer variance of the smaller model.

• Axes: both axes index layer number, 0 to $L - 1 \mathrm { : }$ the matrices are symmetric about the main diagonal by construction.

## Reading guide.

• Block-diagonal structure in the within-group panels (cols 1–2) indicates the model organises its forward pass into discrete representational stages: a contiguous run of layers that are mutually similar (a bright block on the diagonal) followed by an abrupt transition to a new regime. Three blocks is typical, corresponding to early (input encoding), middle (task computation), and late (output preparation) stages.

• Off-diagonal bright patches indicate two non-adjacent layers maintain similar representations – a residual-stream signature. Conversely, dark off-diagonal patches mark regions where representations have substantially diverged.

• Difference panel (col 3): blue cells indicate the correct group has higher within-layer similarity than the incorrect group at that $( i , j )$ location; red cells indicate the reverse. Saturation tracks magnitude.

## What to attend to.

• Block-diagonal sharpness across rows. Compare the top (XML R,A) and bottom (Freeform) rows in cols 1–2. On low-structure tasks (MMLU, LogicBench), XML (R,A) tends to produce a more crisply delineated three-block pattern in the correct group than Freeform does, consistent with structured prompting inducing cleaner stage-wise processing.

• Asymmetry between correct and incorrect. Within a single row, compare cols 1 vs. 2: a sharper block-diagonal pattern in the correct panel than in the incorrect panel suggests that correct answers are produced by more discretely-staged computation, while incorrect answers reflect noisier inter-layer transitions.

• Difference panel as a summary. A predominantly-blue difference panel indicates the correct group consistently has higher within-stage coherence than the incorrect group. A predominantly-red or near-white difference panel indicates outcome does not separate the representational geometry.

• Where the difference concentrates. A blue patch in the upper-left of D (early layers) suggests divergence begins early in the network; a mid-matrix concentration localises the differentiation to the task-computation stage.

• Spider as the high-structure baseline. On Spider, both formats and both outcome groups exhibit similar, less-pronounced block structure and a near-white difference panel – the task’s inherent SQL scaffolding already constrains representations, leaving little room for output format to modulate the internal geometry.

![](images/abde1e836120e9d6a936c522adfc37a9bf092612d2a1fecc69221fd0c1909dce.jpg)  
(b) MMLU  
Figure 6: Llama-3.1-8B within-group CKA heatmaps for (a) GSM8K and (b) MMLU. Each panel shows XML (R,A) (top row) and Freeform (bottom row) across three views: correct answers only, incorrect answers only, and their difference (correct − incorrect). See §D.2 for construction and reading guide.

![](images/d019e6f61b0abe9adf515df3970c976a63ed6af1227e559a5c9c77f7269b6234.jpg)  
(b) Spider  
Figure 7: Llama-3.1-8B within-group CKA heatmaps for (a) LogicBench and (b) Spider. Same layout as Figure 6; see §D.2. Spider shows weak differentiation between outcome groups under both formats, consistent with the high-structure task pattern.

![](images/18c9ebf81508f5e32985e1eec2d45de97750207e5fd00f14e8cad34fdd0bba5c.jpg)  
(b) MMLU  
Figure 8: Gemma3-4B within-group CKA heatmaps for (a) GSM8K and (b) MMLU. Each panel shows XML (R,A) (top row) and Freeform (bottom row) across correct answers, incorrect answers, and their difference (correct − incorrect). See §D.2 for construction and reading guide.

![](images/b03b25f345884bd18d0433ee61181e6c9be297aea42ba606417abef362787701.jpg)  
(b) Spider  
Figure 9: Gemma3-4B within-group CKA heatmaps for (a) LogicBench and (b) Spider. Same layout as Figure 8; see §D.2. Spider shows weak differentiation between outcome groups under both formats, consistent with the high-structure task pattern.

## D.3 Linear Probe Setup

We train a balanced linear probe (logistic regression with $\ell _ { 2 }$ regularisation, $C = 1 . 0 ,$ 5-fold crossvalidation) on the pre-answer hidden states at each layer to predict the model’s own predicted label. Hidden states are extracted from the last token of the reasoning field, immediately before the answer field begins, under both (R,A) and (A,R) orderings.

In both orderings, the probe reads the hidden state at the token immediately preceding the answer letter: the > closing the <answer> opening tag. This is offset −1 relative to the answer-letter token.

<table><tr><td>Ordering</td><td>Probe Token</td><td>Offset</td><td>Preceding Context</td></tr><tr><td>(R,A)</td><td>&gt; in &lt;answer&gt;</td><td>-1</td><td>Full reasoning block</td></tr><tr><td>(A,R)</td><td>&gt; in &lt;answer&gt;</td><td>-1</td><td>No reasoning; answer is the first output field</td></tr></table>

Table 7: Probe location relative to the answer token under different field orderings.

We report balanced accuracy to account for class imbalance across answer options. Probes are run on MMLU and LogicBench for Llama-3.1-8B and Gemma3-4B; GSM8K is excluded as it requires a binary correct/incorrect probe rather than a label-identity probe, and Spider is excluded due to its open-ended SQL output space. Hidden states for both models are extracted via forward pass on NVIDIA T4 GPUs.

## D.4 Gemma3-4B: Representational Results

Figure 10 replicates the Llama-3.1-8B analysis of Section 6 for Gemma3-4B (d = 2560, 35 layers). The inverse-structure gradient in cross-group CKA is preserved: MMLU and LogicBench exhibit lower mid-layer CKA under XML (R,A) than freeform, while Spider again reverses the pattern qualitatively identical to Llama but at reduced absolute magnitude, consistent with Gemma’s smaller overall accuracy gap under (R,A) in Table 1. Linear probes tell the same story: the (R,A) advantage in decoding answer identity from intermediate representations peaks at +0.27 on MMLU (L17) and +0.26 on LogicBench (L13), versus +0.64 and +0.70 for Llama directionally aligned but attenuated, as expected for a model with fewer parameters and shallower representational depth. The convergence across two architecturally distinct open-weight models strengthens the interpretation that mid-layer answer encoding under (R,A) is a general consequence of field ordering rather than a property of any single architecture.

![](images/660e6719fa1f9e2c535a412186e329153dfb224f5204e99b282fab3ebdc0ebe1.jpg)

![](images/7c232d79b2e65e292eccb51dfd1fd3954e643ed2a25777cc5d4e46c4203c74cf.jpg)

![](images/60a053457eed12a89db6165666b9a3fadee0330a356e4005091981014302b08c.jpg)

![](images/da83e17b5de32b16cb64e725ecc0bfd949dfcee0b8f066f1462714f5b9beac84.jpg)

![](images/155d77a8322114364abd1d1b661241b5f0d86340b9321e67a20a26f34d63e968.jpg)

![](images/31f5136edacf7206a18eba04fd8b8223a8b9f96dcc57d5c649ad38d372851298.jpg)  
(a) Cross-group CKA per layer

![](images/bdef4a75b7148a5c56d583a9ab6fc1f241f3235ddb6ad007da7cc54a05d78896.jpg)

![](images/6ee3d4da64ea70c763b348673809f2926a3393f08cf9bc2293f74b627be1035f.jpg)  
(b) Linear probe balanced accuracy per layer

Figure 10: Internal representations for Gemma3-4B, same layout as Figure 3. Left: Cross-group CKA preserves the inverse-structure gradient of Llama-3.1-8B MMLU and LogicBench show lower mid-layer CKA under XML (R,A) than freeform; Spider reverses the pattern at reduced overall magnitude. Right: Linear probe balanced accuracy on MMLU and LogicBench. The (R,A) advantage is directionally consistent with Llama but smaller in magnitude (peak gap +0.27 on MMLU at L17; +0.26 on LogicBench at L13), in line with Gemma’s smaller behavioural gap under R,A in Table 1.

## E Reasoning Length: Measurement and Results

Reasoning length is measured by extracting the reasoning field from each model response and counting words via str.split(), which closely tracks subword-token count under standard tokenizers. For XML formats we extract the content of the <reasoning> tag; for JSON formats we parse the "reasoning" key, with a regex fallback for malformed outputs. For Spider under JSON (A,R), parsed\_reasoning is empty in the primary data file and is extracted directly from raw response CSVs. This yields 80 cells (5 models × 4 datasets × 4 format-orderings); each cell reports median word count and mean accuracy over all examples in that condition.

Tables 8 and 9 report the full per-cell medians and Pearson correlations discussed in §6.3.
<table><tr><td>Dataset</td><td>XML (R,A)</td><td>XML (A,R)</td><td>JSON (R,A)</td><td>JSON (A,R)</td></tr><tr><td>GSM8K</td><td>126</td><td>127</td><td>108</td><td>123</td></tr><tr><td>MMLU</td><td>121</td><td>99</td><td>121</td><td>33</td></tr><tr><td>LogicBench</td><td>114</td><td>92</td><td>113</td><td>96</td></tr><tr><td>Spider</td><td>54</td><td>58</td><td>62</td><td>64</td></tr></table>

Table 8: Median reasoning-field word count per format–ordering, aggregated over 5 models. Bold = shortest per dataset. JSON (A,R) collapses to 33 words on MMLU vs. 121 under XML (R,A); GSM8K is flat across all variants.

<table><tr><td colspan="3">By dataset</td><td colspan="3">By format-ordering</td></tr><tr><td>Dataset</td><td>r</td><td>n</td><td>Format</td><td>r</td><td>n</td></tr><tr><td>GSM8K</td><td>-0.11</td><td>20</td><td>XML (R,A)</td><td>+0.42</td><td>20</td></tr><tr><td>MMLU</td><td>+0.11</td><td>20</td><td>XML (A,R)</td><td>+0.28</td><td>20</td></tr><tr><td>LogicBench</td><td>+0.06</td><td>20</td><td>JSON (R,A)</td><td>+0.25</td><td>20</td></tr><tr><td>Spider</td><td>+0.16</td><td>20</td><td>JSON (A,R)</td><td>-0.73**</td><td>20</td></tr><tr><td>Overall</td><td>+0.04</td><td>80</td><td></td><td></td><td></td></tr></table>

Table 9: Pearson r between median reasoning length and accuracy across 80 cells (5 models × 4 datasets × 4 format-orderings). <sup>∗∗</sup>p<0.01. Conditioning on dataset yields no signal; conditioning on format-ordering reveals JSON (A,R) as the sole significant predictor and it is negative.

## F Prompt-Brittleness Ablation

Motivation. LLM behaviour is known to be sensitive to surface-level prompt variations. Small changes in instructional phrasing, schema tag casing, or whitespace can shift accuracy by several points without changing semantics as emperically demonstrated by (Sclar et al., 2024; He et al., 2024). A natural concern is whether the (R,A) vs. (A,R) accuracy gap reported in Section 4 is similarly an artefact of the specific system prompts used. We address this concern with a focused robustness study.

Setup. We hold the model, dataset, and field ordering fixed and perturb only surface features of the system prompt. For each of the 5 formats (Free-form, JSON (A,R), JSON (R,A), XML (A,R), XML (R,A)), we construct three perturbed variants:

• V1 Instructional phrasing. Paraphrase the role and task sentence (“You are a math expert. Solve the problem step-by-step . . . ” → “ Please solve the following mathematics problem carefully. Work through your reasoning . . . ”). Schema definition is byte-identical to baseline.

• V2 Surface casing. For XML, tag names become PascalCase (<reasoning> → <Reasoning>); for JSON, keys become PascalCase ("answer" → "Answer"); for Free-form, instruction-frame headers are uppercased. Instructional sentence is preserved.

• V3 Whitespace. For XML, blank lines around the schema are removed (compact form); for JSON, the schema definition is collapsed to a single line; for Free-form, additional blank lines are inserted around the separator. All wording and schema names are preserved.

We evaluate each (model, format, variant) cell on GSM8K dataset, across two models: Llama-3.1-8B (the model with the largest (A,R) collapse in the main results) and GPT-4o (a frontier API model).

<table><tr><td>Model</td><td>Format</td><td>Base</td><td>V1</td><td>V2</td><td>V3</td></tr><tr><td>Llama-3.1-8B</td><td>Free-form JSON(A,R) JSON (R,A) XML (A,R) XML (R,A)</td><td>77.5 32.0 81.5 10.5 84.0</td><td>81.0 35.0 86.0 11.5 87.5</td><td>78.5 34.5 74.5 11.0 82.0</td><td>80.0 36.0 77.0 11.5 82.0</td></tr><tr><td>GPT-40</td><td>Free-form JSON (A,R) JSON(R,A) XML (A,R) XML (R,A)</td><td>90.0 59.0 96.0 57.0 96.0</td><td>86.0 56.5 98.0 53.5 97.0</td><td>90.0 55.5 96.0 55.5 98.0</td><td>86.5 56.5 97.0 53.5 98.5</td></tr></table>

Table 10: GSM8K accuracy (%) dataset under three prompt-perturbation variants. Base = original prompt used in the main results. Within each (model, format-family) pair, the (R,A) row consistently exceeds the (A,R) row across all four conditions.

![](images/8c73ba140180e5abc2088e5e121dfa190052f1cabde6af560c5e75a47c4c2b78.jpg)  
Figure 11: GSM8K accuracy under prompt perturbations. Each group of bars is one (model, format) cell; withingroup order is baseline, V1, V2, V3. The $( \mathbf { R } , \mathbf { A } ) - ( \mathbf { A } , \mathbf { R } )$ gap (annotated ∆ over each XML / JSON family) is preserved across all perturbations: the sign of the field-ordering effect is invariant to surface prompt changes, even though absolute accuracy shifts by a few points.

Findings. (1) Sign of the gap is preserved. For every (model, format-family, variant) cell, accuracy under (R,A) exceeds accuracy under (A,R); no perturbation inverts the ordering. The (R,A) − (A,R) gap on Llama XML is +73.5 pts (range across variants: $+ 7 0 . 5 \mathrm { t o } + 7 6 . 0 ) ;$ ; on GPT-4o XML it is +39.0 pts (range +39.0 to +45.0); analogous magnitudes hold for JSON. The specific concern that field ordering’s effect could disappear under slight prompt changes is empirically rejected.

(2) Absolute accuracy shifts are small relative to the ordering effect. The largest within-cell spread across {Base, V1, V2, V3} is 11.5 pts (Llama JSON (R,A); Table 10); most cells move by under 5 pts. By contrast, the smallest $( \mathbf { R } , \mathbf { A } ) - ( \mathbf { A } , \mathbf { R } )$ gap across all (model, family, variant) combinations is +37 pts (GPT-4o JSON baseline).