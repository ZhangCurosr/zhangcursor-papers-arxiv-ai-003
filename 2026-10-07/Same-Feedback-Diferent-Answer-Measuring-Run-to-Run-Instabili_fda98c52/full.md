# Same Feedback, Diferent Answer: Measuring Run-to-Run Instability in Frontier-Model Customer Feedback Analysis

Viraj Bagal, Raviraja Ganta, Prabhath Chellingi

## Abstract

AI agents are increasingly being programmed to automate knowledge work over large collections of unstructured data. Such automation requires repeatability: when the underlying evidence is unchanged, the agent’s categories, priorities, and counts should not shift materially between runs, even if each individual answer appears plausible. We introduce a repeat run evaluation framework that aligns semantically equivalent categories and focuses on two operating metrics: theme churn, the normalized change in the returned category set, and volume disagreement, the change in counts for categories that persist. We evaluate three recurring customer-feedback tasks across eight frontier models, corpus sizes from 100 to 5,000 records, multiple prompts, and three execution designs: raw generation, taxonomy-free hierarchical decomposition, and a taxonomy-grounded agent (TGA) using persistent themes, subthemes, and record-level predictions. With Claude Opus 4.8 and the 1,000-record corpus fixed, TGA reduces theme churn by 86–88% relative to both raw generation and hierarchical decomposition, while matched-theme volumes have zero disagreement. The taxonomy-grounded agent is more stable than every raw model in the screen, remains more stable at each corpus size, and keeps this advantage when theme matching is made stricter or looser. Although evaluated on customer feedback, the framework targets repeated synthesis of unstructured corpora more broadly, including financial reports, legal documents, incident records, and scientific literature. Overall, these results show that taxonomy grounding produces more consistent and repeatable outputs for recurring knowledge work.

## 1 Introduction

A customer-operations analyst may ask, “What are the top customer pain points this month?” A fluent answer is useful once. An operating answer must also survive repetition: the same records, question, and system configuration should not yield a substantially diferent theme set or diferent counts on the next run. This distinction matters because theme identities become dashboard categories, rankings guide roadmaps, and volumes trigger prioritization. Broad language-model evaluations increasingly treat reliability as multidimensional rather than reducible to task accuracy alone (Liang et al. 2023). Yet repeatability for open-ended corpus analysis remains weakly specified.

Prior work demonstrates sensitivity to prompt wording and formatting (Sclar et al. 2024; Chatterjee et al. 2024; Errica et al. 2025), self-inconsistency under ambiguity (Bartsch et al. 2023; Sedova et al. 2024), and the use of repeated samples to estimate confidence or hallucination risk (Lyu et al. 2025; Manakul, Liusie, and Gales 2023; Farquhar et al. 2024). Customer-feedback analysis creates a diferent measurement problem. Each answer is an ordered set of naturallanguage themes, each record may support multiple themes, theme labels can be paraphrased, and counts are comparable only after themes are aligned. Exact-string agreement therefore overstates instability, while a single scalar similarity obscures whether the model changed the categories, their ranks, their counts, or merely the representative examples.

We study three execution designs. A raw frontier model receives every feedback summary in a frozen time window and reclusters the corpus in one call. A taxonomy-free hierarchical workflow splits the same records into fixed batches, analyzes them independently, and synthesizes their outputs. A production taxonomy-grounded agent (TGA) instead queries persistent themes, subthemes, and record-level taxonomy predictions before composing the answer. The grounded system does not see a hand-built answer key; category identity and membership are materialized before the question is asked. All systems return the same strict structured-output contract, enabling identical downstream analysis. Figure 1 summarizes the comparison.

Our contributions are:

• A stability-specific evaluation protocol for repeated openended, multi-theme corpus analysis. It aligns semantic theme names one-to-one and decomposes instability into theme churn, rank, volume, and representative-evidence distances, with birth/death retained as a directional diagnostic.

• A broad screen of eight raw frontier models with ten requested runs for each of three customer-feedback questions, followed by a scale study on nested 100-, 1,000-, and 5,000-record corpora.

• A controlled 1,000-record ablation that fixes Claude Opus 4.8, compares three prompt variants, and separates a raw call, a taxonomy-free ten-worker hierarchy, and persistent taxonomy grounding.

• Evidence that the most stable eligible raw model still changes much of its operating theme set between runs. With model and corpus controlled, grounded theme churn is 0.104 versus 0.765 for raw generation and 0.841 for the taxonomy-free hierarchy; grounded matched-theme volume disagreement is zero.

• Run-level uncertainty estimates, coverage-aware interpretation of conditional metrics, full invalid-output accounting, cost reporting, and a semantic-alignment threshold sweep.

This paper measures stability, not correctness. A system can repeat the same wrong answer. We make no claim that the more stable answers are more accurate, useful, or actionable; those properties require separate human-judgment and taskquality studies.

## 2 Related Work

## Generation Consistency and Prompt Sensitivity

Recent work shows that repeated LLM evaluations should not ignore nondeterminism: outputs can vary across repeated requests under default sampling, greedy decoding, and hosted settings expected to be deterministic (Ouyang et al. 2025; Song et al. 2025; Atıl et al. 2025). Self-consistency is often used as a technique for improving reasoning: sample multiple chains and select the modal answer (Wang et al. 2023). Other work treats agreement among samples as evidence about confidence. Lyu et al. (2025) compare consistency-based confidence measures across models and reasoning tasks; semantic uncertainty groups generations by meaning before measuring entropy (Kuhn, Gal, and Farquhar 2023; Farquhar et al. 2024); and black-box confidence-elicitation studies combine prompting, sampling, and aggregation (Xiong et al. 2024). SelfCheckGPT similarly detects unsupported content through inconsistency across sampled passages (Manakul, Liusie, and Gales 2023). These methods usually reduce repeated generations to a confidence or selection signal. We instead retain disagreement as the object of study and decompose where an operating answer changes.

Prompt-sensitivity research asks whether intentpreserving surface changes alter predictions. Spurious formatting can produce large performance swings (Sclar et al. 2024); POSIX measures likelihood changes under prompt perturbations (Chatterjee et al. 2024); and sensitivity/consistency metrics expose classification failures across prompt rephrasings (Errica et al. 2025). Ambiguity studies also show that a model can remain partly consistent while assigning meaningful probability to alternatives (Bartsch et al. 2023; Sedova et al. 2024). Agent benchmarks such as τ-bench evaluate repeated trials through reliability metrics such as pass<sup>k</sup> (Yao et al. 2024); those tasks have verifiable goal states, while recurring corpus synthesis often has no single gold operating answer. Our repeat-query condition is stricter than prompt perturbation: within an experiment cell, the prompt, corpus, provider configuration, and system path are fixed. The measured variation is run-to-run semantic variation in a structured set answer.

## Topic Discovery and Customer Feedback

Classical topic models represent documents as mixtures over latent topics (Blei, Ng, and Jordan 2003); neural topic methods such as BERTopic cluster dense document representations and derive interpretable labels (Grootendorst 2022). TopicGPT instead prompts a language model to generate and assign natural-language topics (Pham et al. 2024). Topic-model stability has long been recognized as a model-selection concern: independently fitted solutions may yield diferent topics, motivating explicit stability analysis (Greene, O’Callaghan, and Cunningham 2014). Our setting difers because the output is an ordered, question-conditioned set of themes rather than a complete corpus model, but it inherits the same identity problem: labels must be aligned before variation can be quantified.

Customer-feedback systems often impose structure to turn free text into operational categories. InsightNet, for example, combines a multi-level taxonomy with multi-label topic extraction for customer reviews (Mukku et al. 2023). Largescale multi-label classification research studies hierarchical and evolving label spaces (Chalkidis et al. 2020; Tabatabaei et al. 2025), while recent analysis finds that autoregressive language models exhibit distinctive multi-label generation behavior (Ma et al. 2025). These works primarily evaluate predictive quality. We ask whether a question-conditioned aggregation of those labels remains stable across repeated executions.

## Persistent Context and Semantic Matching

Retrieval-augmented generation conditions answers on retrieved external evidence (Lewis et al. 2020); graph-based retrieval and community summaries extend this idea to global corpus questions (Edge et al. 2024). Our grounded arm uses a related system principle—persist domain structure outside the generation—but the experiment is not a retrieval-quality benchmark. It tests whether persistent taxonomy predictions provide a repeatable operating frame relative to reclustering raw records on every call.

Comparing natural-language labels requires semantic rather than lexical equivalence. Sentence-level embeddings make paraphrase-aware similarity practical (Reimers and Gurevych 2019). We freeze one embedding model, threshold, and one-to-one assignment rule, then rerun analysis across a threshold range. This prevents an answer with one broad theme from matching several narrower themes in the other run.

## 3 Evaluation Framework

## Repeated Structured Answers

An experiment cell fixes a system s, corpus $D ,$ , and question q. Independent executions produce runs $R _ { 1 } , \ldots , R _ { n } .$ . A run is an ordered list of theme results,

$$
R _ { i } = [ ( z _ { k } , v _ { k } , e _ { k } , u _ { k } ) ] _ { k = 1 } ^ { K _ { i } } ,\tag{1}
$$

where $z _ { k }$ is a theme name, $v _ { k } \geq 0$ its reported volume, $e _ { k }$ a summary, and $u _ { k }$ a set of at most five supporting record identifiers. The schema forbids additional fields. Quotes are explicitly excluded.

A feedback record may belong to several themes. Consequently, volumes need not sum to |D|, and the supporting-ID sets are representative evidence rather than exhaustive membership lists. The five-ID bound keeps raw and grounded outputs feasible at large scale. It also makes evidence churn a sample-selection diagnostic, not a direct estimate of complete membership churn.

![](images/9f15f11f076b50496c4e757fa2ace17089160b41f8e5ec118c4aa68cf53cf3e2.jpg)  
Figure 1: Evaluation pipeline. The systems share timestamp windows, questions, output schema, alignment, and metrics The Opus-controlled experiment distinguishes one-call generation, fixed hierarchical decomposition without taxonomy, and persistent taxonomy grounding.

Generation, alignment, and analysis are separate stages. Generation receipts identify the corpus, question, candidate configuration, individual attempts, provider usage, cost, source commit, and raw structured output. An alignment artifact references its generation artifact and frozen embedding configuration. Analysis references both, so metrics can be added or removed without regenerating answers.

## Semantic Theme Alignment

For a pair $( R _ { i } , R _ { j } )$ , we embed every theme name z with a frozen text embedding model and compute cosine similarities. Edges below threshold $\gamma$ are removed. Among remaining edges, we first maximize match cardinality and then total similarity, producing a one-to-one assignment $A _ { i j } ;$ this is a lexicographic assignment problem related to maximum-weight bipartite matching (Kuhn 1955). We freeze text-embedding-3-large and $\gamma = 0 . 8 2$ for primary analysis.

Theme names, rather than summaries, define alignment because the operating category is the object whose persistence we test. Summaries and evidence may change even when a category remains the same. The threshold sweep in Section 6 tests whether the headline ordering depends on this choice.

## Pairwise Stability Metrics

Let $K _ { i } = | R _ { i } | , K _ { j } = | R _ { j } |$ , and $M = | A _ { i j } |$ |. Every reported metric is a distance, so lower is more stable.

Theme churn. We use the complement of the Sørensen– Dice overlap (Dice 1945):

$$
d _ { \mathrm { t h e m e } } ( i , j ) = 1 - \frac { 2 M } { K _ { i } + K _ { j } } .\tag{2}
$$

It is zero when all themes align and one when none align. Theme churn is the primary metric because it remains defined when rank and volume comparisons cannot be made.

Rank instability. For aligned themes, let $\tau _ { b }$ be Kendall’s rank correlation with tie correction (Kendall 1938, 1945). We map it to [0, 1]:

$$
d _ { \mathrm { r a n k } } ( i , j ) = \frac { 1 - \tau _ { b } } { 2 } .\tag{3}
$$

The metric is undefined if fewer than two themes align. Thus zero rank instability means only that the matched themes preserve relative order; it does not imply identical answers.

Theme birth/death. Directional unmatched-theme rates are averaged:

$$
d _ { \mathrm { b d } } ( i , j ) = \frac { 1 } { 2 } \left( \frac { K _ { i } - M } { K _ { i } } + \frac { K _ { j } - M } { K _ { j } } \right) .\tag{4}
$$

When $K _ { i } \ = \ K _ { j } ,$ , this equals theme churn. We therefore treat it as a directional diagnostic rather than independent evidence.

Volume disagreement. Over matched pairs $( a , b ) \in A _ { i j }$

$$
d _ { \mathrm { v o l } } ( i , j ) = \frac { 1 } { M } \sum _ { ( a , b ) \in A _ { i j } } \frac { \left| v _ { a } - v _ { b } \right| } { \operatorname* { m a x } ( v _ { a } , v _ { b } , 1 ) } .\tag{5}
$$

It is undefined when $M = 0$ . The normalization makes a one-record diference more consequential for small themes while bounding every term by one.

Evidence churn. Let $U _ { a }$ and $U _ { b }$ be the representative supporting-ID sets. We average their Jaccard distances:

$$
d _ { \mathrm { e v } } ( i , j ) = \frac { 1 } { M } \sum _ { ( a , b ) \in A _ { i j } } \left( 1 - \frac { \left| U _ { a } \cap U _ { b } \right| } { \left| U _ { a } \cup U _ { b } \right| } \right) .\tag{6}
$$

This metric asks whether the system cites the same examples for a matched theme. It is deliberately secondary because each output contains at most five IDs.

## Aggregation and Uncertainty

For n valid runs, we compute all  <sup>n</sup> unordered run pairs. The question-level score is the mean pairwise distance,

$$
\bar { d } _ { s , D , q } = \binom { n } { 2 } ^ { - 1 } \sum _ { i < j } d ( R _ { i } , R _ { j } ) ,\tag{7}
$$

with defined/undefined pair counts retained in the artifact. A system-tier macro score is the unweighted mean of the three question means. This prevents a question with more valid runs, and therefore more pairs, from dominating the result.

Pairwise observations are dependent because each run appears in several pairs. For the scale experiment, we estimate 95% intervals by deleting one complete run at a time within each question, recomputing the U-statistic, and applying the jackknife standard-error estimator (Efron 1981). We combine question-level variances for the macro mean using a Welch– Satterthwaite t interval. The controlled Opus ablation instead reports per-question run-level bootstrap percentile intervals; when resampling repeats an original run, its artificial selfcomparison is omitted. Both procedures resample complete runs rather than treating pair distances as independent.

## 4 Experimental Setup

## Corpus and Questions

The corpus contains anonymized, multilingual enterprise customer-support feedback. We expose no raw text, record identifiers, tenant identifiers, product names, or user metadata. Records are ordered by timestamp and record identifier to create immutable nested prefixes. Every presented window starts at 2026-04-01 15:16:43 UTC; exclusive endpoints are 2026-06-12 03:26:12 for 100 records, 2026-06-16 16:30:56 for 1,000, and 2026-06-29 15:15:55 for 5,000.

We freeze three operational questions:

1. summarize the top five reasons customers provide feedback;

2. identify the top three customer pain points and afected workflows; and

3. identify and prioritize the top feature requests, their underlying needs, and frequency.

The full prompts request a volume and concise summary for every theme and explicitly state that customer quotes are unnecessary. The question text does not change between repeated runs.

## Systems Compared

Raw frontier model. The raw arm receives the complete list of feedback summaries, timestamps, and record IDs in the window, plus the question and JSON schema. It has no production taxonomy predictions. Each execution therefore induces its own grouping of the records. Provider adapters use structured-output facilities when available and validate the same schema after generation.

Taxonomy-free hierarchical workflow. This condition uses fixed rather than adaptive decomposition. It splits the 1,000-record corpus into ten ordered batches of 100, invokes ten parallel structured workers, and passes their ordered outputs to one structured synthesis call. It has no tools, retrieval, planning, adaptive branching, taxonomy, or persistent record predictions. Thus every final answer uses 11 stochastic model calls while holding both the corpus and model fixed.

Taxonomy-grounded agent. The TGA is a production supervisor/worker architecture with general-purpose workers and workers that query the persistent taxonomy store. It queries persistent themes, subthemes, and record-level predictions constrained to the frozen timestamp window. The agent is prompted to return the shared answer schema and omit quotes; no deterministic post-processing selects themes or evidence. Experiments 1–2 use a Claude Sonnet 4.6 supervisor with provider-default reasoning. Across 105 requested TGA answers in Experiments 1–2, 98 valid outputs are evaluated; retained traces for these outputs contain no subagent events, so the supervisor performs the analysis directly. Experiment 3 explicitly pins the same production architecture to Claude Opus 4.8. Exact record-level retrieval parity is neither expected nor required because the TGA retrieves aggregate taxonomy objects; corpus parity is enforced by auditing the timestamp window in its analytical queries.

Experiments 1–2 compare complete deployed system designs, so model and architecture difer. Experiment 3 controls the language model and adds the taxonomy-free hierarchy to test whether decomposition alone accounts for the stability gap. All arms use provider-default reasoning settings. A sanitized supplemental model-ID table lists the provider-facing identifiers retained in artifacts or resolved from the production model catalog.

## Experiment 1: Frontier-Model Screen

At 100 records, we request ten independent runs for every model–question cell. The eight raw configurations span OpenAI, Bedrock Converse, Google GenAI, and Fireworks provider interfaces. Output limits are 32,768 tokens in this screen. Provider defaults are used for sampling because the compared frontier interfaces do not expose a common, semantically equivalent temperature control.

A raw model is eligible for selection only if every question yields at least nine valid structured outputs. We rank eligible models by macro theme churn, then rank instability, then birth/death. Invalid attempts remain in the artifact store and in cost totals; we do not rerun a cell until it happens to pass. The TGA is a comparator, not a candidate in raw-model selection.

## Experiment 2: Scale Comparison

We carry the most stable eligible raw configuration into the scale experiment and compare it with the TGA at 100 and 1,000 records, requesting ten fresh runs for each question. The 100-record raw cells are regenerated under the scale execution control (one call at a time) rather than reused from the concurrent model screen. This isolates scale from withinarm concurrency changes.

The 5,000-record tier is a preconfigured operational stress test with five requested runs per question. Each raw call contains roughly 744,000 input tokens. To remove tokenthroughput throttling as a confounder, calls run serially, provider retries are disabled, the output limit is 8,192 tokens, and each billable request is followed by a recorded 120–150 second cooldown. The experiment harness applies exponential backof to zero-usage retryable failures. This produced zero throughput throttles, but only three valid raw answers per question, so 5,000-record uncertainty is necessarily wider.

## Experiment 3: Opus-Controlled Prompt and Architecture Ablation

We freeze Claude Opus 4.8, provider-default reasoning, the 1,000-record corpus, the three questions, and the shared schema. No condition requests hidden reasoning text in the output. Prompt sensitivity is evaluated first, without inspecting a grounded result: raw and taxonomy-free systems each use three frozen system prompts—minimal (task and schema only), bottom-up (record-first extraction followed by semantic consolidation), and consistency-first (canonical labels, stable boundaries, non-overlap, and deterministic tiebreaking). This is 2 systems ×3 prompts ×3 questions ×10 runs, or 180 final outputs.

A prompt is eligible only if every one of its six system– question cells has at least 8/10 valid outputs. We select the lowest unweighted macro theme churn, breaking ties by birth/death, volume disagreement, and then rank instability when coverage is adequate. Exact selected raw and workflow cells are reused in the architecture comparison; they are not regenerated after selection. We then add 30 fresh TGA outputs (3 questions ×10 runs), yielding 210 planned full answers.

Raw and taxonomy-free model calls use deterministic aliases R0000–R0999, mapped back to immutable corpus IDs. Both use the same evidence normalizer: preserve firstseen order, discard out-of-scope aliases, remove duplicates, and retain at most five IDs and never more than the reported volume. This prevents identifier transport from becoming an uncontrolled diference. Before full execution, all configurations pass two-run smoke tests; the TGA smoke test is rerun after its production runtime changes and is pinned to runtime commit 1f2634478f7a.

## Validity, Cost, and Traceability

An answer is valid only if it parses into the strict schema, contains no more than five unique evidence IDs per theme, and uses IDs from the frozen corpus where IDs are returned. Metrics use valid outputs only; the paper reports validity alongside stability. Costs include every billable physical generation attempt, including malformed answers and retries. Raw cost uses captured provider token usage and the rate vector frozen in each generation artifact. TGA cost uses the production session estimate over its internal model calls; retained cost artifacts expose aggregate session-level usage, not a supervisor/worker/retrieval split. Embedding and local analysis costs are excluded.

Every stage is immutable and content addressed. Generation receipts include source commit, dependency lock hash, corpus hash, candidate parameters, timestamps, attempts, usage, output, and raw-provider model identifiers where the harness calls providers directly. TGA receipts add session IDs, trace IDs, usage events, and production model selectors. Alignment receipts add the embedding model, threshold, and generation reference. Analysis receipts add metric versions and alignment references. The Opus ablation uses generation source 55c4159ef6c0 and analysis source 96060b122abb; its prompt and architecture exports carry SHA-256 digests and resolve every result to attempts, prompts, model receipts, corpus receipts, and usage. This lineage permits post-hoc metric and threshold analysis without new generation.

![](images/0eb50a9997b55b5fb5cf5f4d7eaf081c0d651c965aa25ffe690e2a454d254cbc.jpg)  
Figure 2: Macro mean pairwise theme churn with 95% complete-run jackknife intervals. The 5,000-record raw interval is wide because only three valid runs per question survived.

## 5 Results

## Experiment 1: Frontier-Model Variation

Table 1 reports the screen. All raw models exhibit substantial theme churn. Claude Fable 5 is the most stable eligible raw model at 0.616, followed by GPT-5.6 Sol at 0.706. Three candidates fail the pre-specified nine-valid-per-question floor. Their metrics remain visible but do not participate in winner selection. The TGA returns 30 valid answers and has zero theme, rank, birth/death, and volume distance across 135 run pairs.

Low conditional rank scores do not contradict high theme churn. Fable’s 0.000 rank instability says that themes which align preserve their order; its 0.616 churn says many themes do not align at all. Theme churn is therefore the selection metric, while rank is a diagnostic conditional on theme survival.

## Experiment 2: Stability Across Corpus Scale

Figure 2 and Table 2 show the requested 100-, 1,000-, and 5,000-record sequence. Raw Fable theme churn increases from 0.552 to 0.623 and 0.726. A least-squares descriptive slope over these three points is +0.0325 churn per additional 1,000 records. The TGA’s corresponding churn values are 0.013, 0.148, and 0.033, substantially below the raw model at every presented tier.

The gap is operationally large. At 100 records the raw model changes more than half of its aligned theme-set mass between two runs on average; at 5,000 it changes nearly three quarters. In contrast, the TGA retains nearly all theme identity in the stress test. The TGA’s 1,000-record churn is concentrated in theme appearance/disappearance rather than count movement: volume disagreement remains exactly zero.

Table 1: Experiment 1 at 100 records. Valid outputs are shown for feedback reasons / pain points / feature requests. Lower stability metrics are better. Ineligible candidates fail the validity floor; the TGA is a comparator. Cost includes invalid and retried generation attempts.
<table><tr><td>Candidate</td><td>Valid</td><td>Churn</td><td>Rank</td><td>Birth/death</td><td>Volume</td><td>Cost (USD)</td><td>Eligible</td></tr><tr><td>Taxonomy-grounded agent</td><td>10/10/10</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.000</td><td>14.488</td><td>Comparator</td></tr><tr><td>Claude Fable 5</td><td>10/10/9</td><td>0.616</td><td>0.000</td><td>0.616</td><td>0.116</td><td>8.809</td><td>Yes</td></tr><tr><td>GLM 5.2</td><td>9/10/8</td><td>0.657</td><td>0.192</td><td>0.657</td><td>0.152</td><td>3.237</td><td>No</td></tr><tr><td>Claude Opus 4.8</td><td>9/10/6</td><td>0.669</td><td>0.000</td><td>0.669</td><td>0.108</td><td>7.218</td><td>No</td></tr><tr><td>GPT-5.6 Sol</td><td>10/10/10</td><td>0.706</td><td>0.073</td><td>0.706</td><td>0.082</td><td>4.083</td><td>Yes</td></tr><tr><td>Kimi K2.7 Code</td><td>9/10/9</td><td>0.715</td><td>0.056</td><td>0.715</td><td>0.249</td><td>2.276</td><td>Yes</td></tr><tr><td>Qwen 3.7 Plus</td><td>5/3/6</td><td>0.817</td><td>0.167</td><td>0.706</td><td>0.167</td><td>0.784</td><td>No</td></tr><tr><td>DeepSeek V4 Pro</td><td>9/10/10</td><td>0.854</td><td>0.074</td><td>0.854</td><td>0.136</td><td>2.009</td><td>Yes</td></tr><tr><td>Gemini 3.1 Pro Preview</td><td>10/10/10</td><td>0.877</td><td>0.067</td><td>0.877</td><td>0.128</td><td>5.149</td><td>Yes</td></tr></table>

Table 2: Stability across the presented scale sequence. Coverage in parentheses is the number of defined pairs for a conditional metric over all valid run pairs. The 5,000-record tier requested five rather than ten runs per cell.
<table><tr><td>Records</td><td>Arm</td><td>Valid</td><td>Pairs</td><td>Theme churn [95% CI]</td><td>Rank (coverage)</td><td>Birth/death</td><td>Volume (coverage)</td><td>Cost</td></tr><tr><td>100</td><td>Raw Fable</td><td>30/30</td><td>135</td><td>0.552 [0.487, 0.616]</td><td>0.023 (91/135)</td><td>0.551</td><td>0.083 (123/135)</td><td>$8.790</td></tr><tr><td>100</td><td>TGA</td><td>29/30</td><td>126</td><td>0.013 [0.000, 0.043]</td><td>0.026 (126/126)</td><td>0.013</td><td>0.000 (126/126)</td><td>$14.934</td></tr><tr><td>1,000</td><td>Raw Fable</td><td>30/30</td><td>135</td><td>0.623 [0.534, 0.712]</td><td>0.015 (72/135)</td><td>0.623</td><td>0.098 (111/135)</td><td>$51.249</td></tr><tr><td>1,000</td><td>TGA</td><td>27/30</td><td>109</td><td>0.148 [0.104, 0.193]</td><td>0.030 (109/109)</td><td>0.144</td><td>0.000 (109/109)</td><td>$17.605</td></tr><tr><td>5,000</td><td>Raw Fable</td><td>9/15</td><td>9</td><td>0.726 [0.197, 1.000]</td><td>0.000 (4/9)</td><td>0.726</td><td>0.075 (6/9)</td><td>$143.245</td></tr><tr><td>5,000</td><td>TGA</td><td>12/15</td><td>19</td><td>0.033 [0.008, 0.059]</td><td>0.000 (19/19)</td><td>0.029</td><td>0.000 (19/19)</td><td>$14.812</td></tr></table>

## Experiment 3: Opus-Controlled Ablation

The prompt-selection pass tests minimal, bottom-up, and consistency-first prompts before inspecting grounded results. Minimal and bottom-up are ineligible because at least one taxonomy-free cell falls below 8/10 valid outputs. Consistency-first is the only eligible prompt and also has the lowest macro theme churn, 0.803. Prompt wording changes the measured values, but none of the raw or taxonomy-free prompt configurations approaches the grounded result.

With Opus 4.8 fixed, the TGA records 0.104 macro theme churn, compared with 0.765 for the raw call and 0.841 for the taxonomy-free hierarchy (Table 3). These are reductions of 86.4% and 87.6% from the displayed values. Decomposition into ten workers and a synthesis call therefore does not explain the grounded system’s stability. Its matched-theme volumes are identical in all 117 valid run pairs. Rank is also defined for all 117 pairs, versus 47/135 for raw and 43/118 for taxonomy-free; the apparently low taxonomy-free rank score of 0.047 describes only the small intersection that survives its high churn.

The separation holds for each question. Run-level bootstrap intervals are [0.659, 0.826], [0.600, 0.842], and [0.759, 0.870] for raw Opus, while grounded intervals are [0.000, 0.000], [0.000, 0.281], and [0.078, 0.156]. A degenerate interval means no variation was observed in the ten runs, not that population variance is proven to be zero. Representativeevidence churn does not separate the systems: TGA is 0.461 and raw is 0.448. Because each answer exposes at most five examples, this metric measures sample-evidence turnover rather than complete taxonomy membership.

## Counts, Rank, and Evidence

Matched-theme volumes provide the clearest operatinganswer result. Raw volume disagreement is 0.083, 0.098, and 0.075 at the three tiers; TGA volume disagreement is 0.000 in every defined pair. Thus, once a grounded theme is present in both runs, its reported volume does not move. This directly tests the expectation that semantic judgment may vary while theme counts remain stable.

Rank requires more caution. At 1,000 records, raw rank instability is only 0.015, but it is defined for 72 of 135 pairs; at 5,000 it is zero for only 4 of 9 pairs. The low values describe the relative ordering of a small surviving intersection. The TGA supports rank calculation for every valid pair at all three tiers. Reporting rank without coverage would invert the practical interpretation.

Evidence churn remains nonzero for both systems: raw/TGA values are 0.318/0.207 at 100, 0.295/0.432 at 1,000, and 0.485/0.395 at 5,000. A stable category can rotate among five recent examples, especially as the corpus grows. Evidence churn therefore should not be treated as taxonomy churn.

## Structured-Output Reliability and Cost

Stability is conditional on obtaining a valid answer, so validity is a separate outcome. In the 100-record screen, validity ranges from 14/30 for Qwen 3.7 Plus to 30/30 for GPT-5.6 Sol, Gemini 3.1 Pro Preview, and the TGA. In the 100- and 1,000-record scale cells, raw Fable yields 60/60 valid outputs while the TGA yields 56/60. At 5,000 records, Fable yields 9/15 valid outputs despite zero throughput throttles; four failures return the answer field as a string and two return an empty object. The TGA yields 12/15 valid outputs.

Table 3: Opus-controlled experiment at 1,000 records. Lower metric values are more stable. Raw and taxonomy-free use the selected consistency-first prompt while TGA uses its production prompt; cost covers each 30-output segment. Rank coverage is defined pairs / all valid pairs. Per-question rows give theme churn with 95% run-level bootstrap percentile intervals.  
A. Architecture comparison after prompt selection
<table><tr><td>System</td><td>Valid</td><td>Theme</td><td>Birth/death</td><td>Volume</td><td>Rank</td><td>Evidence</td><td>Rank cov.</td><td>Cost</td></tr><tr><td>Raw Opus</td><td>10/10/10</td><td>0.765</td><td>0.764</td><td>0.244</td><td>0.105</td><td>0.448</td><td>47/135</td><td>$21.821</td></tr><tr><td>Taxonomy-free hierarchy</td><td>10/10/8</td><td>0.841</td><td>0.840</td><td>0.247</td><td>0.047</td><td>0.507</td><td>43/118</td><td>$37.299</td></tr><tr><td>Taxonomy-grounded agent</td><td>10/9/9</td><td>0.104</td><td>0.099</td><td>0.000</td><td>0.008</td><td>0.461</td><td>117/117</td><td>$16.536</td></tr></table>

B. Theme churn by question
<table><tr><td>System</td><td>Feedback reasons</td><td>Customer pain points</td><td>Feature requests</td></tr><tr><td>Raw Opus</td><td>0.756 [0.659, 0.826]</td><td>0.719 [0.600, 0.842]</td><td>0.820 [0.759, 0.870]</td></tr><tr><td>Taxonomy-free hierarchy</td><td>0.689 [0.595, 0.766]</td><td>0.904 [0.783, 0.984]</td><td>0.931 [0.906, 0.949]</td></tr><tr><td>Taxonomy-grounded agent</td><td>0.000 [0.000, 0.000]</td><td>0.185 [0.000, 0.281]</td><td>0.126 [0.078, 0.156]</td></tr></table>

Full-corpus raw cost grows from \$8.79 at 100 to \$51.25 at 1,000 and \$143.25 at 5,000. TGA cost is \$14.93, \$17.61, and \$14.81. These totals use system-specific accounting and exclude amortized taxonomy ingestion cost, so they are descriptive rather than a controlled price benchmark. They nevertheless show the operational consequence of repeatedly transmitting the entire corpus versus querying a preprocessed context layer.

The Opus-controlled program requests 210 full answers and yields 196 valid outputs: raw 90/90, taxonomyfree 78/90, and TGA 28/30. Taxonomy-free failures are structured-output violations, not throughput throttles. The two invalid TGA trials are an empty structured object and an exact-window query-audit failure. One additional TGA feature-request attempt returns HTTP 502 with zero usage; its retry succeeds and the failure is recorded as unbilled. Full-generation costs are \$65.77 for raw prompt sensitivity, \$114.78 for taxonomy-free prompt sensitivity, and \$16.54 for fresh TGA runs. Including \$90.72 of preliminary, hardening, and original smoke and \$8.98 of refreshed TGA smoke, the complete program contains 66 submitted generation jobs, 306 physical provider attempts, and costs \$296.79.

## 6 Robustness and Diagnostics

## Alignment-Threshold Sensitivity

The primary matcher uses cosine threshold 0.82. To test dependence on this choice, we reuse stored theme embeddings and rematerialize alignment and analysis at thresholds 0.76, 0.78, . . . , 0.90. This creates no new model generations or embedding calls. All primary-threshold analysis artifacts reproduce exactly, including means, defined-pair counts, and intervals.

At thresholds 0.76, 0.82, and 0.90, raw Fable churn is 0.357/0.552/0.812 at 100 records, 0.439/0.623/0.814 at 1,000, and 0.667/0.726/0.889 at 5,000. TGA remains lower: 0.000/0.013/0.040, 0.148/0.148/0.148, and 0.033/0.033/0.033. The system ordering never reverses. At 1,000 and 5,000 records, TGA churn is unchanged across thresholds because recurring taxonomy labels align nearverbatim and the remaining unmatched categories stay unmatched throughout the sweep.

## Uncertainty and Pair Dependence

At 100 records, the 95% churn intervals are [0.487, 0.616] for Fable and [0.000, 0.043] for TGA. At 1,000 they are [0.534, 0.712] and [0.104, 0.193]. These complete-run jackknife intervals are separated. The 5,000-record intervals are [0.197, 1.000] and [0.008, 0.059]; the raw interval is broad because deleting one of only three valid runs leaves little pairwise information. We present the point as a stress test, not as a precision-equivalent replacement for the ten-run tiers.

## 7 Limitations and Ethical Considerations

Stability is not quality: repeatability can preserve a systematic error, and this study does not compare themes against a gold taxonomy, operator judgments, or decision outcomes. Prior work similarly distinguishes agreement from correctness (Lyu et al. 2025; Kuhn, Gal, and Farquhar 2023). The corpus comes from one private deployment and a narrow period, so customer mix, language distribution, taxonomy maturity, channel, and bursty arrivals may afect clustering.

TGA stability is conditional on a fixed taxonomy and prediction artifact; rebuilding the taxonomy could move instability to ingestion time. Thus the result is that taxonomy grounding amortizes and stabilizes repeated question answering after ingestion, not that construction-time stability is solved. Other limits remain; private feedback, IDs, endpoints, and organization identifiers are not released.

## References

Atıl, B.; Aykent, S.; Chittams, A.; Fu, L.; Passonneau, R. J.; Radclife, E.; Rajan Rajagopal, G.; Sloan, A.; Tudrej, T.; Ture, F.; Wu, Z.; Xu, L.; and Baldwin, B. 2025. Non-Determinism of “Deterministic” LLM System Settings in Hosted Environments. In Proceedings of the 5th Workshop on Evaluation and Comparison ofNLP Systems, 135–148. Association for Computational Linguistics.

Bartsch, H.; Jorgensen, O.; Rosati, D.; Hoelscher-Obermaier, J.; and Pfau, J. 2023. Self-Consistency of Large Language

Models under Ambiguity. In Proceedings of the 6th BlackboxNLP Workshop: Analyzing and Interpreting Neural Networksfor NLP, 89–105. Association for Computational Linguistics.

Blei, D. M.; Ng, A. Y.; and Jordan, M. I. 2003. Latent Dirichlet Allocation. Journal of Machine Learning Research, 3: 993–1022.

Chalkidis, I.; Fergadiotis, M.; Kotitsas, S.; Malakasiotis, P.; Aletras, N.; and Androutsopoulos, I. 2020. An Empirical Study on Large-Scale Multi-Label Text Classification Including Few and Zero-Shot Labels. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, 7503–7515. Association for Computational Linguistics.

Chatterjee, A.; Renduchintala, H. S. V. N. S. K.; Bhatia, S.; and Chakraborty, T. 2024. POSIX: A Prompt Sensitivity Index for Large Language Models. In Findings of the Association for Computational Linguistics: EMNLP 2024, 14550–14565. Association for Computational Linguistics.

Dice, L. R. 1945. Measures of the Amount of Ecologic Association Between Species. Ecology, 26(3): 297–302.

Edge, D.; Trinh, H.; Cheng, N.; Bradley, J.; Chao, A.; Mody, A.; Truitt, S.; and Larson, J. 2024. From Local to Global: A Graph RAG Approach to Query-Focused Summarization. arXiv preprint arXiv:2404.16130.

Efron, B. 1981. Nonparametric Estimates of Standard Error: The Jackknife, the Bootstrap and Other Methods. Biometrika, 68(3): 589–599.

Errica, F.; Sanvito, D.; Siracusano, G.; and Bifulco, R. 2025. What Did I Do Wrong? Quantifying LLMs’ Sensitivity and Consistency to Prompt Engineering. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies, 1543–1558. Association for Computational Linguistics.

Farquhar, S.; Kossen, J.; Kuhn, L.; and Gal, Y. 2024. Detecting Hallucinations in Large Language Models Using Semantic Entropy. Nature, 630: 625–630.

Greene, D.; O’Callaghan, D.; and Cunningham, P. 2014. How Many Topics? Stability Analysis for Topic Models. In Machine Learning and Knowledge Discovery in Databases, 498–513. Springer.

Grootendorst, M. 2022. BERTopic: Neural Topic Modeling with a Class-Based TF-IDF Procedure. arXiv preprint arXiv:2203.05794.

Kendall, M. G. 1938. A New Measure of Rank Correlation. Biometrika, 30(1/2): 81–93.

Kendall, M. G. 1945. The Treatment of Ties in Ranking Problems. Biometrika, 33(3): 239–251.

Kuhn, H. W. 1955. The Hungarian Method for the Assignment Problem. Naval Research Logistics Quarterly, 2(1–2): 83–97.

Kuhn, L.; Gal, Y.; and Farquhar, S. 2023. Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation. In International Conference on Learning Representations.

Lewis, P.; Perez, E.; Piktus, A.; Petroni, F.; Karpukhin, V.;Goyal, N.; Küttler, H.; Lewis, M.; Yih, W.-t.; Rocktäschel, T.;

Riedel, S.; and Kiela, D. 2020. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. In Advances in Neural Information Processing Systems, volume 33, 9459– 9474.

Liang, P.; Bommasani, R.; Lee, T.; Tsipras, D.; Soylu, D.; Yasunaga, M.; Zhang, Y.; Narayanan, D.; Wu, Y.; Kumar, A.; et al. 2023. Holistic Evaluation of Language Models. Transactions on Machine Learning Research.

Lyu, Q.; Shridhar, K.; Malaviya, C.; Zhang, L.; Elazar, Y.; Tandon, N.; Apidianaki, M.; Sachan, M.; and Callison-Burch, C. 2025. Calibrating Large Language Models with Sample Consistency. Proceedings of the AAAI Conference on Artificial Intelligence, 39(18): 19260–19268.

Ma, M.; Chochlakis, G.; Pandiyan, N. M.; Thomason, J.; and Narayanan, S. 2025. Large Language Models Do Multi-Label Classification Diferently. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing, 2472–2495. Association for Computational Linguistics.

Manakul, P.; Liusie, A.; and Gales, M. J. F. 2023. SelfCheck-GPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, 9004–9017. Association for Computational Linguistics.

Mukku, S. S.; Soni, M.; Aggarwal, C.; Rana, J.; Yenigalla, P.; Patange, R.; and Mohan, S. 2023. InsightNet: Structured Insight Mining from Customer Feedback. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing: Industry Track, 552–566. Association for Computational Linguistics.

Ouyang, S.; Zhang, J. M.; Harman, M.; and Wang, M. 2025. An Empirical Study of the Non-Determinism of ChatGPT in Code Generation. ACM Transactions on Software Engineering and Methodology, 34(2): 1–28.

Pham, C. M.; Hoyle, A.; Sun, S.; Resnik, P.; and Iyyer, M. 2024. TopicGPT: A Prompt-Based Topic Modeling Framework. In Proceedings of the 2024 Conference of the North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, 2956–2984. Association for Computational Linguistics.

Reimers, N.; and Gurevych, I. 2019. Sentence-BERT: Sentence Embeddings Using Siamese BERT-Networks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing, 3982–3992. Association for Computational Linguistics.

Sclar, M.; Choi, Y.; Tsvetkov, Y.; and Suhr, A. 2024. Quantifying Language Models’ Sensitivity to Spurious Features in Prompt Design or: How I Learned to Start Worrying about Prompt Formatting. In International Conference on Learning Representations.

Sedova, A.; Litschko, R.; Frassinelli, D.; Roth, B.; and Plank, B. 2024. To Know or Not to Know? Analyzing Self-Consistency of Large Language Models under Ambiguity. In Findings of the Association for Computational Linguistics:

EMNLP 2024, 17203–17217. Association for Computational Linguistics.

Song, Y.; Wang, G.; Li, S.; and Lin, B. Y. 2025. The Good, The Bad, and The Greedy: Evaluation of LLMs Should Not Ignore Non-Determinism. In Proceedings of the 2025 Conference ofthe Nations oftheAmericas Chapter oftheAssociation for Computational Linguistics: Human Language Technologies, 4195–4206. Association for Computational Linguistics.

Tabatabaei, S. A.; Fancher, S.; Parsons, M.; and Askari, A. 2025. Can Large Language Models Serve as Efective Classifiers for Hierarchical Multi-Label Classification of Scientific Documents at Industrial Scale? In Proceedings of the 31st International Conference on Computational Linguistics: Industry Track, 163–174. Association for Computational Linguistics.

Wang, X.; Wei, J.; Schuurmans, D.; Le, Q. V.; Chi, E. H.; Narang, S.; Chowdhery, A.; and Zhou, D. 2023. Self-Consistency Improves Chain of Thought Reasoning in Language Models. In International Conference on Learning Representations.

Xiong, M.; Hu, Z.; Lu, X.; Li, Y.; Fu, J.; He, J.; and Hooi, B. 2024. Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs. In International Conference on Learning Representations.

Yao, S.; Shinn, N.; Razavi, P.; and Narasimhan, K. 2024. τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains. arXiv:2406.12045.