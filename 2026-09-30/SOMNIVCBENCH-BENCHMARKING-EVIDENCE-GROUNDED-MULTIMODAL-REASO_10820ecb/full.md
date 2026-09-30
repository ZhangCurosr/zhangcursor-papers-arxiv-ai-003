![](images/a8e2d15842d0fae45734d4069f87ce616a4f87c31c8bb16c5a3eea392afec4a1.jpg)

# SOMNIVCBENCH: BENCHMARKING EVIDENCE-GROUNDED MULTIMODAL REASONING TOWARDS AI VIRTUAL CELLS

Manyu Li Fudan University, Shanghai, China 24210240029@m.fudan.edu.cn

Xunkai Li Beijing Institute of Technology, Beijing, China cs.xunkai.li@gmail.com

Yongfu Xiong & Yi Liu Chongqing Ant Consumer Finance Co,. Ltd {xiongyongfu.xyf, larry.liuy}@myxiaojin.cn

Rong-Hua Li \*& Guoren Wang Beijing Institute of Technology, Beijing, China {1ironghuabit, wanggrbit}@162.com

## ABSTRACT

Artificial Intelligence Virtual Cells (AIVCs) are envisioned as scientific agents that simulate cellular responses, explain underlying mechanisms, and support hypothesis-driven discovery. Existing AIVC benchmarks, however, operate primarily at the simulation layer, motivating complementary evaluation of how models interpret experimental evidence and formulate biological hypotheses. We introduce OMNIVCBENCH, a figure-centric, source-traceable benchmark for the interpretation component of an AIVC. It contains 6,077 curated single- and multi-subfigure question-answer pairs derived from figures and experimental contexts in the scientific literature. Guided by Bloom's taxonomy, we instantiate interpretation-layer counterparts of the AIVC Predict-Explain-Discover agenda through three scientific reasoning tasks. These comprise L1 evidence-conditioned inference, L2 mechanistic explanation, and L3 evidence-grounded hypothesis proposal. We further introduce AIVC-Judge, a task-conditioned MLLM-as-ajudge framework with category-specific, reference-aware rubrics for evaluating open-ended responses. A complementary Model-Derived Hard-Negative Mining (MDHNM) strategy converts plausible errors observed during model inference into MCQ distractors for lower-cost evaluation. Within the evaluated heterogeneous model pool, MCQ accuracy correlates positively with AIVC-Judge scores, providing a complementary view of performance alongside open-response evaluation. Evaluating both proprietary and open-weight multimodal models shows that even the best proprietary model reaches only 3.28 out of 5.00 under AIVC-Judge. Supervised fine-tuning and retrieval-augmented generation on the auxiliary OMNIVCTRAIN corpus provide modest, configuration-dependent gains. Together, OMNIVCBENCH combines source-aligned cellular figures, interpretationrole organization around Predict-Explain-Discover, and paired open-response and controlled MCQ evaluation. It provides a source-traceable resource for assessing evidence-grounded reasoning in candidate AIVC interpretation components. Code and data demo are available at https: //anonymous. 4open. science/r/OmniVCBench.

![](images/7aad43a51da310770eecd406015f46801e4683c8553fb3676c14244eb13b4ad1.jpg)  
Figure 1: The interpretation component within the AIVC workflow. The simulation layer predicts perturbed cell states; experiments return observations; the interpretation layer compares and explains the evidence, informing the next experiment or model revision (feedback arrow). OM-NIVCBENCH evaluates only the interpretation component (highlighted), through three scientific task roles (L1–L3); the remaining stages are application context.

## 1 INTRODUCTION

Artificial intelligence virtual cells (AIVCs) aim to predict cellular behavior, explain biological mechanisms, and support hypothesis-driven discovery (Bunne et al., 2024; Noutahi et al., 2025). The Predict-Explain-Discover (P-E-D) agenda connects these functions into a workflow in which simulation is only the beginning: a predicted response motivates an experiment, and its value is realized when the results are distilled into a mechanistic explanation and a testable next hypothesis. Accurate state prediction alone cannot show whether a system can interpret experimental observations, recognize disagreement with new evidence, or propose what to test next. Current virtual-cell benchmarks primarily evaluate the simulation layer, measuring how accurately models predict cellular states and phenotypes under intervention (Roohani et al., 2025; Szałata et al., 2024; Wu et al., 2025b; Wei et al., 2026b; Mao et al., 2026; Li et al., 2026a); this complementary capability is not yet organized and tested systematically in AIVC evaluation. Figure 1 locates both capabilities within the same workflow, linking simulation to the scientific use of its predictions.

This division of labor raises a role-assignment question that current virtual-cell evaluations leave open: what role should multimodal LLMs play in an AI virtual cell, and how do they close the loop with the simulation layer? Simulation-layer models operate on omics profiles, yet the recorded products of cellular experiments—fluorescence microscopy fields, immunoblots, dose— response curves, and composite multi-panel figures——are visual, and using these results scientifically means reading them. We therefore distinguish the MLLM component, which maps figures and questions to evidence-grounded answers, from the MLLM agent, which adds planning, tool use, and iterative refinement around it; the agent's reliability depends on this component's interpretation quality. The component's domain is what we call the interpretation layer. Confronted with an experimental result, a researcher asks in sequence: what do the observations support, how can the phenomenon be explained, and what can be tested next; these become three task roles: L1 evidenceconditioned inference derives a result from displayed evidence—retrospective decoding of what is observed, confirming or contradicting prior expectations, distinct from the simulation layer's forward forecasting of unobserved states; L2 mechanistic explanation accounts for results through biological processes; and L3 evidence-grounded hypothesis proposal formulates a testable extension of the observations. OMNIVCBENCH operationalizes this component-level evaluation, with Section 4.5 exercising the component inside executable loops.

Scientific-figure benchmarks already assess important parts of this reasoning process (Roberts et al., 2024; Li et al., 2024b; Zhang et al., 2025b), including hypothesis generation and experiment proposal in biology-focused settings (Burgess et al., 2025; Laurent et al., 2026). We build on these precedents with OMNIVCBENCH, a cellular-evidence benchmark organized around the interpretation roles of an AIVC. Its 6,077 questions draw on paper-linked figures and experimental contexts from OmniScience (Tao et al., 2026), combining single- and multi-subfigure questions with a traceable source record per answer. Published experiments are the evidence source because their figures provide traceable observations from real experimental systems within a closed, reviewable record; they remain a proxy for the live workflow, covering curated scientific communication artifacts annotated, post-processed figures—rather than primary instrument readouts, raw-data quality, or failed experiments. We evaluate general-purpose multimodal LLMs as candidate interpretation components from the image and question alone, an interface complementing the omics-native inputs and outputs of single-cell foundation models (Cui et al., 2024; Theodoris et al., 2023; Hao et al., 2024).

AIVC-Judge scores open responses against source references with claim-level evidence traces; Model-Derived Hard-Negative Mining (MDHNM) turns observed inference errors into discriminative MCQ options for low-cost comparison; a unanimous three-annotator audit decides which candidates enter the benchmark; and human rescoring checks the judge on shared textual evidence. OMNIVCTRAIN supplies an adaptation resource, and executable pilots test the component inside the loop. These pieces yield four contributions:

• A source-traceable benchmark of 6,077 cellular-research questions spanning inference, explanation, and hypothesis proposal in the interpretation layer.

• Paired open-response and MCQ evaluation on identical items, with human-checked reference-aware scoring, model-derived distractors, and a unanimous three-annotator item audit.

• Evaluation of proprietary and open-weight MLLMs around four research questions: judgment reliability, cross-format agreement, adaptation gains, and where explanation and hypothesis proposal fall short.

• Executable evidence-acquisition pilots on real biological outputs, testing prediction revision inside the loop's evidence-acquisition and revision step (Section 4.5).

## 2 RELATED WORK

## 2.1 VIRTUAL-CELL EVALUATION

AIVC roadmaps connect prediction of cellular behavior with mechanistic explanation and scientific discovery (Bunne et al., 2024; Noutahi et al., 2025). Perturbation benchmarks evaluate the simulation layer: the Virtual Cell Challenge and OP3 assess intervention responses (Roohani et al., 2025; Szałata et al., 2024); PerturBench and scPerturBench standardize generalization tests (Wu et al., 2025b; Wei et al., 2026b); and VCBench targets in-the-wild responses (Mao et al., 2026). MVCBench adds transcriptomic and morphological outputs (Li et al., 2026a), and VCWorld adds knowledge-guided reasoning to simulation (Wei et al., 2026a). PerturbQA, CellVerse, and SC-Arena extend evaluation to language-based biological reasoning (Wu et al., 2025a; Zhang et al., 2025a; Zhao et al., 2026), though over textual serializations of omics profiles rather than visual evidence. OMNIVCBENCH complements these simulation-layer settings (Table 1) by evaluating natural-language interpretation of observed cellular figures.

## 2.2 SCIENTIFIC MULTIMODAL REASONING

SciFIBench and MMSci assess scientific-figure understanding across disciplines, and HiSciBench organizes tasks from reading to discovery (Roberts et al., 2024; Li et al., 2024b; Zhang et al., 2025b). The closest biology-focused precedents are MicroVQA and FigQA2 within LABBench2 (Table 1). MicroVQA explicitly tests mechanistic hypothesis generation and experiment proposals alongside expert image interpretation, with multi-image questions and Bloom-based cognitive analysis (Burgess et al., 2025). FigQA2 uses open responses and source-specific questions across Image, Paper, and Retrieval modes (Laurent et al., 2026). Building on this coverage, OMNIVCBENCH curates OmniScience's paper-linked figures, captions, and contexts (Tao et al., 2026) into a larger cellular-evidence dataset with consistent L1-L3 organization, open-response and MCQ protocols paired on identical items, and diagnostics of visual grounding, scoring agreement, and model errors. As Table 1 summarizes, OMNIVCBENCH complements the full AIVC pipeline at its downstream stage: where perturbation benchmarks validate what a virtual cell predicts, we evaluate how the system reads, explains, and builds on experimental evidence.

Table 1: Recent virtual-cell and scientific-figure benchmarks.
<table><tr><td>Benchmark</td><td>Scale</td><td>What it evaluates / contains</td><td>Layer</td></tr><tr><td>Virtual Cell Challenge (Roohani et al., 2025)</td><td>~300k cells</td><td>CRISPRi perturbation responses</td><td>Simulation</td></tr><tr><td>OP3 (Szałata et al., 2024)</td><td>144 compounds</td><td>Held-out perturbation expression</td><td>Simulation</td></tr><tr><td>PerturBench (Wu et al., 2025b)</td><td>6 datasets</td><td>Perturbation-response prediction</td><td>Simulation</td></tr><tr><td>scPerturBench (Wei et al., 2026b)</td><td>29 datasets</td><td>Unseen perturbations and contexts</td><td>Simulation</td></tr><tr><td>VCBench (Mao et al., 2026)</td><td>7 datasets</td><td>In-the-wild perturbation response</td><td>Simulation</td></tr><tr><td>MVCBench (Li et al., 2026a)</td><td>~1.1M profiles</td><td>Transcriptomic and morphology outputs</td><td>Simulation</td></tr><tr><td>MicroVQA (Burgess et al., 2025)</td><td>1,042 MCQs</td><td>Microscopy interpretation, hypothesis and experiment proposal</td><td>Interpretation</td></tr><tr><td>LABBench2 (Laurent et al., 2026)</td><td>~1,900 tasks</td><td>Figure, protocol, and literature QA</td><td>Interpretation</td></tr><tr><td>SciFIBench (Roberts et al., 2024)</td><td>~2,000 questions</td><td>Scientific-figure interpretation</td><td>Interpretation</td></tr><tr><td>OmniScience (Tao et al., 2026)</td><td>~1.5M records</td><td>Figure-caption-context corpus</td><td>Data layer</td></tr><tr><td>OMNIVCBENCH</td><td>6,077 items</td><td>Evidence-grounded inference, explanation, and hypothesis proposal</td><td>Interpretation</td></tr></table>

## 2.3 TASK ORGANIZATION AND RESPONSE EVALUATION

P-E-D describes scientific functions, while Bloom's taxonomy supplies a complementary account of cognitive demand (Noutahi et al., 2025; Bloom et al., 1956; Anderson & Krathwohl, 2001). Mechanistic explanations, in particular, connect observations to organized biological entities and activities (Machamer et al., 2000). Open-response evaluation uses LLM judges to assess such semantic content (Liu et al., 2023; Zheng et al., 2023), with documented preferences for response styles and their own generations (Panickssery et al., 2024; Ye et al., 2025). MCQ evaluation instead depends on plausible and discriminative distractors (Alhazmi et al., 2024). RefineBot uses solver reflection and iterative rewriting (Burgess et al., 2025), while AutoConverter combines distractor proposal, review, selection, and correctness refinement (Zhang et al., 2025c). MDHNM pools low-scoring open responses and rewrites them into candidate options. Our focus is the connection between these response formats: applying both to the same cellular evidence supports comparing answer selection with generated reasoning.

## 3 METHOD

## 3.1 TASK FORMULATION

An OmniScience source record is $\boldsymbol { u } = ( t , \mathcal { F } , C ) ;$ the paper title t, scientific images with captions $\mathcal { F } = \{ ( I _ { j } , c _ { j } ) \} _ { j = 1 } ^ { m } { } _ { \mathrm { : } }$ , and the surrounding paper context $C _ { i }$ , with no questions or answers provided. Our construction pipeline G maps each retained record to a set of benchmark items grounded in its images and experimental context,

$$
G ( u ) \longrightarrow \{ z _ { i } \} , \qquad z _ { i } = ( { \mathcal { T } } _ { i } , q _ { i } , a _ { i } , \ell _ { i } ) , \qquad { \mathcal { T } } _ { i } \subseteq \{ I _ { j } \} _ { j = 1 } ^ { m } ,\tag{1}
$$

where $\mathcal { T } _ { i }$ contains one or more images, $q _ { i }$ is a newly generated question, $a _ { i }$ is its reference answer, and $\ell _ { i } \in \{ 1 , 2 , 3 \}$ is the interpretation level. At inference time, the evaluated model receives only the image set and question; paper titles, captions, and surrounding contexts are retained as source evidence for data construction and reference-aware scoring. We evaluate the same image-question pair through open-response generation and controlled answer selection:

$$
\widehat { r } _ { i } = F _ { \theta } ( \mathbb { Z } _ { i } , q _ { i } ) , \qquad \widehat { k } _ { i } = F _ { \theta } ( \mathbb { Z } _ { i } , q _ { i } , \mathcal { O } _ { i } ) ,\tag{2}
$$

where $\widehat { r _ { i } }$ is an open response and $\widehat { k } _ { i }$ selects from a six-option MCQ set $\mathcal { O } _ { i }$ . The open protocol preserves explanatory detail and is scored by AIVC-Judge (Section 3.3); the MCQ protocol supports inexpensive, deterministic evaluation across models and task levels.

The task levels specify what a response must accomplish. An L1 answer derives a result from the stated conditions and displayed observations; an L2 answer explains the biological process connecting them. An L3 answer proposes an evidence-consistent hypothesis with testable consequences. Published experiments anchor these proposals in traceable observations, and the source interpretation supplies a reference for assessment. A defensible alternative can extend that interpretation while preserving its observational basis. Accordingly, L3 concerns evidence-grounded hypothesis proposal, with novelty and experimental confirmation outside its measured scope. P-E-D describes these functional roles, whereas Bloom supplies construction-time cognitive labels. Our bounded, source-referenced formulation of proposal maps to Bloom's Evaluate (B5)—judging candidate explanations against evidence—rather than unconstrained Create (B6), which remains outside the measured scope. Distinguishing the two axes separates inference from explanation even when both receive an Analyze label. Task definitions and composition statistics appear in Appendices A.2 and A.5, respectively.

![](images/0ed5fe8431aea85d256f6514e6f1436dfa9967a4dc397f50ca58d4a8ad8f0c63.jpg)  
Figure 2: Construction and paired evaluation pipeline. Filtered OmniScience records are expanded into QA pairs and curated; retained items form OMNIVCBENCH (open + MCQ tracks), while coarse QA forms OMNIVCTRAIN for SFT/RAG.

## 3.2 OMNIVCBENCH CONSTRUCTION

## 3.2.1 OPEN-ENDED QA GENERATION

Source records and domain filtering. OmniScience supplies paper-linked figures, titles, authorwritten captions, and surrounding contexts, which form the source evidence for construction (Tao et al., 2026). We retrieve records using terms covering cellular perturbations, single-cell and spatial omics, pathways, mechanisms, and disease-relevant cell biology (Figure 2). This filtering concentrates the corpus on experiments that support interpretation of cellular behavior and its biological basis. The benchmark sources are Nature Communications papers from 2011–2017, dominated by mouse and human studies, microscopy, and immunoblot assays (Appendix A.5).

Question and answer synthesis. For each retained record, one of two generator models is selected with equal probability per sample: Kimi-K2.6 (Moonshot AI, 2026) or Qwen-VL-MAX (Alibaba Cloud, 2025). The selected generator receives the images and source text and produces candidate question-answer pairs. This two-generator assignment diversifies question phrasing, requested operations, and answer style across the corpus, reducing the stylistic imprint that any single construction model would leave on the benchmark. A record can support several questions when its panels expose distinct scientific operations. The question specifies the required inference, explanation, or hypothesis proposal, while the reference answer remains linked to the source evidence. This separation lets answering models operate on figures and questions, while construction and scoring retain the underlying experimental context as source evidence.

Quality control. Candidate items undergo schema, evidence, answer-specificity, and visualgrounding checks. Flagged items are rewritten or removed before a manual audit in which three cell biology PhD annotators review each candidate independently—blind to one another's judgments and to the generator model's identity—and only unanimously approved items are retained. The audit retains 6,077 of 7,923 candidates (76.70%), with unanimous decisions on 93.15% of candidates and Fleiss’κ = 0.8560 [0.8442, 0.8674] (Appendix A.3), yielding 6,077 items from 2,625 source records and 1,080 papers. These checks connect source-level curation to the quality of the questions and options presented during evaluation (Appendix A.4).

## 3.2.2 MODEL-DERIVED HARD-NEGATIVE MINING FOR MCQ GENERATION

MDHNM constructs MCQ distractors from errors observed during open-response inference, linking options to failures on the same image and question rather than generator-proposed errors. Twelve rollouts per model from a heterogeneous model pool supply candidate responses. We retain those assigned an AIVC-Judge overall score of at most 2 (judged with the image included, as in production scoring), forming an empirical pool of low-scoring responses to the same underlying question:

$$
\begin{array} { r } { \mathcal { E } _ { i } = \left\{ r _ { i , m , t } \middle | r _ { i , m , t } = F _ { m } ( \mathbb { Z } _ { i } , q _ { i } ) , J _ { \ell _ { i } } ( q _ { i } , r _ { i , m , t } , E _ { i } , \mathbb { Z } _ { i } ) \leq 2 \right\} , } \end{array}\tag{3}
$$

where m indexes models and t indexes repeated rollouts. Errors shared across models are preferred as recurrent scientific failure patterns in the candidate pool.

The generation model then rewrites candidates from $\mathcal { E } _ { i }$ to match the reference answer's syntax, detail, and length while preserving the underlying error. A reviewer then selects five distractors $\mathcal { D } _ { i }$ using checks on scientific incorrectness, relevance, uniqueness, and answer length:

$$
\begin{array} { r l r } { | \mathscr { D } _ { i } | = 5 , } & { \mathrm { W r o n g } ( d \mid z _ { i } ) = 1 , } & { \mathrm { G r o u n d e d } ( d \mid \mathscr { Z } _ { i } , q _ { i } ) = 1 , } \\ { \mathrm { U n i q u e } ( \{ a _ { i } \} \cup \mathscr { D } _ { i } ) = 1 , } & { 0 . 9 5 \leq | d | / | a _ { i } | \leq 1 . 2 5 , } & { d \in \mathscr { D } _ { i } . } \end{array}\tag{4}
$$

These criteria target incorrectness, relevance, semantic non-overlap, and style consistency; | · | denotes length in characters. The checks address both a distractor's scientific error and surface cues such as wording and length. Finally, $\mathcal { O } _ { i } = \pi ( \{ a _ { i } \} \cup \mathcal { D } _ { i } )$ randomly orders the reference answer and five distractors into an $\mathrm { A - F }$ question, and the unanimous three-annotator audit applies to the resulting items as to all candidates. Appendix A.7 reports the screening criteria and comparisons between the complete MDHNM workflow and direct distractor generation.

## 3.2.3 OMNIVCTRAIN

The pipeline also produces OMNIVCTRAIN, a corpus of 548,450 coarse image-question-answer examples for adaptation. These examples retain the same evidence-to-language interface, while benchmark items receive additional manual validation. After upstream paper deduplication, we remove training examples whose paper title or DOI matches a benchmark source. Complementary text and image overlap checks are reported in Appendix A.12. We use this resource for supervised fine-tuning on the QA triples and for retrieval of related demonstrations at answer time, with source exclusions retained in both (Appendix B).

## 3.3 AIVC-JUDGE

Open responses require semantic evaluation that remains anchored to the source evidence. AIVC-Judge (Figure 3) performs rubric-conditioned pointwise scoring on $\left( q _ { i } , r _ { i } , E _ { i } , \mathcal { T } _ { i } \right)$ . Here, $E _ { i }$ contains the caption, surrounding context, and reference answer, and $\mathcal { T } _ { i }$ is the original figure image set shown to the answering model. The level-specific judge returns dimension scores, an overall score, and an evidence trace:

$$
J _ { \ell _ { i } } ( q _ { i } , r _ { i } , E _ { i } , { \mathcal { T } } _ { i } ) \longrightarrow ( { \mathbf { s } } _ { i } , o _ { i } , { \mathcal { C } } _ { i } ) ,\tag{5}
$$

where $\mathbf { s } _ { i }$ contains level-specific dimension scores, $o _ { i } \in [ 1 , 5 ]$ is the overall score, and $\mathcal { C } _ { i }$ records the claim-level evidence trace. The judge decomposes the response into atomic claims and checks each against the source evidence. It then assigns dimension scores and a holistic overall judgment using the level-specific rubric; the overall score is not programmatically clamped by the dimension caps. This sequence separates the factual support for individual claims from the quality of the response as a whole.

L1 assesses correctness and evidence support, while L2 examines faithfulness, causal completeness, mechanistic granularity, and evidence consistency. L3 scores judgment correctness, argumentation quality, and evidence weighing under a source-referenced critique rubric (Figure 3), assessing critical justification of the proposal rather than proposal formulation itself. Critical factual or directional errors cap the corresponding correctness or faithfulness dimension. The overall score summarizes performance under the selected rubric; dimension scores and evidence traces expose the basis for that assessment. Full checklists and scoring definitions appear in Appendix A.8.

![](images/b21a1147c57245b0c9be9e8a24d46b8c4ed9cabcfc09d0f944260ebbb147fa53.jpg)  
Figure 3: AIVC-Judge. Given the question, response, original figure image, and closed textual evidence, the judge verifies atomic claims against evidence, then scores level-specific dimensions and returns dimension scores, an overall score, and an evidence trace.

The production judge is DeepSeek-V4-Flash-Vision-Exp (Xu et al., 2026), a multimodal judge that scores each response from the original figure image and the closed textual evidence. It scores pointwise at temperature 0 with candidate identity omitted (Appendix A.8). Human validation scores the same responses under the production rubric on the stratified 300-item sample: mean human overall ratings correlate with the production judge at Spearman ρ = 0.877 across 884 paired responses; annotators scored without the original figure, so this agreement validates reference-grounded textual rubric application rather than the judge's visual reading (Appendix A.10). A frozen 24-item L3 pilot rescoring archived responses under an evidence-matched proposal-oriented rubric preserves candidate rankings while improving cross-annotator agreement (QWK 0.459→0.767), indicating that L3 totals are rubric-sensitive, not a pure measure of proposal ability (Appendix A.11).

## 4 EXPERIMENTS

## 4.1 SETUP

Models and protocols. We evaluate four proprietary and eleven open-weight MLLMs, with the latter spanning 2B-8B parameters. The MCQ track includes these fifteen base models and an SFT variant; eleven configurations also have open-response evaluation, with the SFT variant represented by the rank-16 checkpoint. MCQ accuracy uses the original six-option keys, and open-response scores average the judge's raw overall output. Model identifiers and clustered statistics appear in Appendices A.1 and A.13.

Adaptation settings. We adapt Qwen3-VL-8B through LoRA supervised fine-tuning in LLaMA-Factory (Hu et al., 2021; Zheng et al., 2024) and multimodal retrieval-augmented generation (Lewis et al., 2020). Both use OMNIVCTRAIN: SFT supplies training examples, and RAG retrieves related QA demonstrations through joint image-question embeddings. Training and retrieval configurations, including source-exclusion rules, are detailed in Appendix B.

Table 2: Paired results on OMNIVCBENCH: MCQ accuracy (%, left) and AIVC-Judge scores (1– 5, right). Within each block, models without Judge scores precede models sorted by Judge Avg (ascending). † denotes the rank-16, two-epoch SFT configuration on OMNIVCTRAIN. Each statistic summarizes performance over individual QA items in the corresponding track.
<table><tr><td rowspan="2">Model</td><td colspan="4">MCQ accuracy (%)↑</td><td colspan="4">AIVC-Judge (1–5)↑</td></tr><tr><td>L1</td><td>L2</td><td>L3</td><td>Avg</td><td>L1</td><td>L2</td><td>L3</td><td>Avg</td></tr><tr><td>Proprietary models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Claude-Sonnet-4-5 (Anthropic, 2025)</td><td></td><td></td><td></td><td>45.3 52.1 52.5 49.4</td><td></td><td>2.55 2.92 2.62 2.68</td><td></td><td></td></tr><tr><td>Grok-4.6 (xAI, 2026)</td><td></td><td></td><td></td><td>48.6 53.9 49.2 50.3</td><td></td><td>3.26 3.39 2.56 3.10</td><td></td><td></td></tr><tr><td>GPT-5.6-luna (OpenAI, 2025)</td><td></td><td></td><td></td><td>47.2 48.3 33.9 43.7</td><td></td><td>3.25 3.47 2.72 3.16</td><td></td><td></td></tr><tr><td>GPT-5.6-sol (OpenAI, 2025)</td><td></td><td></td><td></td><td>58.6 60.6 44.2 55.1</td><td></td><td>3.37 3.68 2.76 3.28</td><td></td><td></td></tr><tr><td>Open-weight models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>InternVL3.5-4B (Wang et al., 2025)</td><td></td><td></td><td></td><td>30.3 31.9 23.6 28.9</td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen2.5-VL-3B (Bai et al., 2025b)</td><td></td><td></td><td></td><td>25.2 24.4 16.6 22.5</td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen3-VL-2B (Bai et al., 2025a)</td><td></td><td></td><td></td><td>21.8 25.1 18.1 21.7</td><td></td><td></td><td></td><td></td></tr><tr><td>InternVL3.5-2B (Wang et al., 2025)</td><td></td><td></td><td></td><td>22.6 21.9 17.2 20.9</td><td></td><td></td><td></td><td></td></tr><tr><td>SmolVLM2-2.2B (Marafioti et al., 2025) 17.9 18.7 15.1 17.3</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LLaVA-Med-Mistral-7B (Li et al., 2023)</td><td></td><td></td><td></td><td>17.1 17.5 12.4 15.9</td><td></td><td></td><td>2.14 1.68 1.63 1.86</td><td></td></tr><tr><td>LLaVA-OneVision-7B (Li et al., 2024a)</td><td></td><td></td><td></td><td>25.7 30.5 20.4 25.5</td><td></td><td></td><td>2.32 1.83 1.70 2.00</td><td></td></tr><tr><td>Qwen2.5-VL-7B (Bai et al., 2025b)</td><td></td><td></td><td></td><td>28.3 30.6 20.1 26.6</td><td></td><td></td><td>2.27 2.10 1.88 2.11</td><td></td></tr><tr><td>Qwen3-VL-4B (Bai et al., 2025a)</td><td></td><td></td><td></td><td>31.3 33.1 26.5 30.5</td><td></td><td></td><td>2.37 2.29 1.96 2.23</td><td></td></tr><tr><td>InternVL3.5-8B (Wang et al., 2025)</td><td></td><td></td><td></td><td>30.4 34.3 26.0 30.3</td><td></td><td></td><td>2.68 2.24 1.87 2.32</td><td></td></tr><tr><td>Qwen3-VL-8B (Bai et al., 2025a)</td><td></td><td></td><td></td><td>34.2 37.4 29.6 33.8</td><td></td><td></td><td>2.48 2.46 1.99 2.33</td><td></td></tr><tr><td>Qwen3-VL-8B-SFT† (Bai et al., 2025a)</td><td></td><td></td><td></td><td>33.8 36.5 32.7 34.3</td><td></td><td></td><td>2.54 2.38 2.14 2.38</td><td></td></tr></table>

## 4.2 BENCHMARKING MLLMS WITH OMNIVCBENCH

RQ1: Can MLLMs judge reliably from experimental observations? Table 2 reports paired MCQ and open-response results. GPT-5.6-sol leads both tracks, reaching 55.1% MCQ accuracy and 3.28/5 under AIVC-Judge. Open-weight base models span 15.9%–33.8% MCQ accuracy and score below 2.40 on open responses. For the strongest model, the open-response score declines from 3.68 on L2 to 2.76 on L3 (the decline concentrates in unprompted critique dimensions; in the frozen 24-item pilot of Appendix A.11, the same responses score 4.75–4.83).

RQ2: How far do MCQs reflect open-response performance? Across the eleven shared models, MCQ and open-response scores correlate at Spearman ρ = 0.964 (Figure 4a). The pooled relationship includes strong separation between proprietary and open-weight systems; within the four proprietary models the correlation is much weaker (Pearson r = 0.208), with individual rank inversions (Appendix A.13). The two formats capture related but distinct aspects of performance on the same questions.

Human reference and visual evidence. On the stratified 300-item sample, human MCQ accuracy ranges from 55.3% to 74.7% across three annotators, bracketing GPT-5.6-sol's 57.3% with images (Appendix A.9). In a no-image comparison on the same sample, GPT-5.6-sol accuracy falls to 34.7% without the figure—still twice the 16.7% random baseline—demonstrating a substantial visual contribution alongside question text, option cues, and domain priors; two open-weight models show no drop at all (Appendix A.6).

## 4.3 ADAPTATION WITH OMNIVCTRAIN

RQ3: Which capabilities can added training data improve? Table 3 compares adaptation of Qwen3-VL-8B on the MCQ track. The rank-32 SFT configuration (100k subset) improves average accuracy by 1.97 points, its largest gain at L3 (29.6% to 34.8%); multimodal RAG improves it by 1.60 points, and the lower-rank configuration changes it by +0.46. Both gains are significant under source-paper-clustered Holm-corrected analysis (Appendix B); Figure 4b shows their distinct level distributions. The coarse QA corpus thus supports both adaptation routes, with gains depending on how it is used.

Joint image-question retrieval scores 35.41% versus 33.62% image-only and 33.40% text-only, so the gain occurs with the joint representation. Across base models, RAG improvements range from

![](images/85fb47ca29674b019ee95e8dc7a62bd02b18f197b57bd99ad20dd41a22cf31e0.jpg)

![](images/4eca04044bb8c3170a5558cccdc8f0012a73325ae82e0e7588d7dfbadee4bc0a.jpg)

(c) L3 dimension profile  
![](images/1fca81c85f79f97445aa27446d0cac0d68a115582d05d3063c91f47427925a36.jpg)  
Figure 4: Evaluation agreement and adaptation gains. (a) MCQ accuracy vs. AIVC-Judge score over eleven models. (b) Qwen3-VL-8B adaptation across L1–L3. (c) L3 dimension profile.

Table 3: Adaptation of Qwen3-VL-8B on the MCQ track. ∆ denotes the change from the base model; p is the p-value from the raw paper-block permutation test. Corrected comparisons appear in Appendix B.
<table><tr><td>Method</td><td>Resource</td><td>L1</td><td>L2</td><td>L3</td><td>Avg</td><td>∆</td><td>p</td></tr><tr><td>Base</td><td></td><td>34.2</td><td>37.4</td><td>29.6</td><td>33.82</td><td></td><td></td></tr><tr><td>SFT, full corpus</td><td>548k, 2 epochs</td><td>33.8</td><td>36.5</td><td>32.7</td><td>34.28</td><td>+0.46</td><td>0.42</td></tr><tr><td>SFT, rank 32</td><td>100k, 2 epochs</td><td>34.6</td><td>38.5</td><td>34.8</td><td>35.79</td><td>+1.97</td><td> $3 . 2 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Multimodal RAG</td><td>548k index, k = 3</td><td>35.6</td><td>39.3</td><td>31.2</td><td>35.41</td><td>+1.60</td><td> $5 . 6 \times 1 0 ^ { - 4 }$ </td></tr></table>

0.76 to 1.60 points (Appendix B.2); the strongest adapted Qwen configuration reaches 35.79%, leaving a substantial gap to the leading proprietary model.

## 4.4 ERROR ANALYSIS ON OMNIVCBENCH

RQ4: Where do explanation and hypothesis proposal fall short? The selected cases expose response-level differences behind the aggregate scores (Appendix F): reversed effect directions and incorrect associations between interventions, panels, and biological entities at L1/L2. Paired evaluation exposes these inconsistencies: in the IFNAR1 and CCP1 examples, a keyed MCQ selection coexists with an inconsistent open explanation; conversely, correct open relations can accompany an option that changes direction or effect. Reading both responses therefore reveals errors that an MCQ score alone can conceal.

L3 additionally requires separating the motivating observation from the intervention and outcome a hypothesis predicts; a proposal can be testable yet rest on a misread observation (the hairregeneration example). A model can also produce a defensible open hypothesis while selecting an option that reverses its own prediction or contradicts the pathway diagram (Cases H and I, Appendix F). The cases connect this distinction to the benchmark's central purpose: assessing the relation between evidence and a scientific answer.

## 4.5 EVIDENCE ACQUISITION IN THE AIVC REFINEMENT LOOP

The benchmark evaluates this stage on curated literature figures; to test the same component inside executable loops, we run four GPT-5.6-sol pilots with one shared structure: predict a hidden biological output, select one additional measurement, observe it, and revise for rescoring. The settings span four real evidence types: non-additive double-gene responses from GEARS on Norman CRISPRa data (Norman et al., 2019; Roohani et al., 2023), drug-combination responses from a locally trained CPA with combinations held out (Lotfollahi et al., 2023), mechanism classes from real fluorescence microscopy (Caie et al., 2010; Ljosa et al., 2012), and signaling interventions on Sachs data (Sachs et al., 2005). Initial accuracies range from 66.7% to 93.3%, so no pilot sits at ceiling. Acquiring the selected measurement yields nine corrections and zero regressions across the three numerical pilots (77.1%→89.6%, 93.3%→100.0%, and 66.7%→70.0%), while self-review improves no pilot and degrades two, and real-microscopy classification shows no gain. These are one-step replay loops without simulator retraining and with corrections confined to queried readouts, reported as revision pilots rather than a validated discovery loop (Appendix C).

## 5 CONCLUSION

AIVC benchmarks primarily assess cellular simulation, while the AIVC agenda also requires interpreting evidence and proposing hypotheses. We introduce OMNIVCBENCH, operationalizing this component through 6,077 source-traceable questions spanning inference, explanation, and hypothesis proposal. AIVC-Judge and model-derived distractors provide paired assessments of generated explanations and answer selection, with strong ranking agreement between tracks and distinct evidence-handling behaviors in case studies. The strongest model scores below three of five on L3 open responses, indicating weaker evidence-grounded argumentation. Human validation supports the shared-evidence scoring protocol, and OMNIVCTRAIN adaptation yields modest gains. Returning to our opening question: four one-step evidence-acquisition pilots on real biological outputs position the MLLM as the interpretation component that closes the loop between simulation and experimental evidence, a role current models fill only partially. With the released judge implementation, rubrics, and data demo, these resources support developing interpretation components that connect predictions with evidence-grounded explanations and testable hypotheses for the next loop iteration of AIVC development and beyond.

## AI USE STATEMENT

In this work, we used generative AI tools to generate synthetic data: candidate question-answer pairs, MCQ distractors, and auxiliary task-type annotations in OMNIVCBENCH and OMNIVC-TRAIN were produced by multimodal LLMs, and every retained item was approved by human annotators. We also used generative AI tools to interpret results and support qualitative data analysis, through the MLLM-as-a-judge pipeline that verifies candidate responses against source evidence. We have not used generative AI tools to develop theoretical models or conceptual frameworks, formulate mathematical claims or assist in proofs, design the research methodology, or implement methods, and the remaining required-disclosure tasks are not applicable to this work. We take responsibility for the final content of this work, including text, claims, and artifacts produced with the aid of generative AI.

## REFERENCES

Elaf Alhazmi, Quan Z Sheng, Wei Emma Zhang, Munazza Zaib, and Ahoud Alhazmi. Distractor generation in multiple-choice tasks: A survey of methods, datasets, and evaluation. In Proceedings of the 2024 conference on empirical methods in natural language processing, pp. 14437– 14458, 2024.

Alibaba Cloud. Qwen-VL-Max. https://www.alibabacloud.com/help/en/ model-studio/vision,2025.

Lorin W Anderson and David R Krathwohl. A taxonomy for learning, teaching, and assessing: A revision of Bloom's taxonomy of educational objectives: complete edition. Addison Wesley Longman, Inc., 2001.

Anthropic. Claude sonnet 4.5 system card. System card, September 2025. URL https : / /www . anthropic.com/claude-sonnet-4-5-system-card.

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025a.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025b.

Benjamin S Bloom, Max D Engelhart, Edward J Furst, Walker H Hill, David R Krathwohl, et al. Handbook i: cognitive domain. New York: David McKay, pp. 483–498, 1956.

Kristofer Bodvard, Ken Peeters, Friederike Roger, Natalie Romanov, Aeid Igbaria, Niek Welkenhuysen, Gaël Palais, Wolfgang Reiter, Michel B Toledano, Mikael Käll, et al. Light-sensing via hydrogen peroxide and a peroxiredoxin. Nature communications, 8(1):14791, 2017. doi: 10.1038/ncomms14791.

Charlotte Bunne, Yusuf Roohani, Yanay Rosen, Ankit Gupta, Xikun Zhang, Marcel Roed, Theo Alexandrov, Mohammed AlQuraishi, Patricia Brennan, Daniel B Burkhardt, et al. How to build the virtual cell with artificial intelligence: Priorities and opportunities. Cell, 187(25):7045–7063, 2024.

James Burgess, Jeffrey J Nirschl, Laura Bravo-Sánchez, Alejandro Lozano, Sanket Rajan Gupte, Jesus G Galaz-Montoya, Yuhui Zhang, Yuchang Su, Disha Bhowmik, Zachary Coman, et al. Microvqa: A multimodal reasoning benchmark for microscopy-based scientific research. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 19553–19564. IEEE, 2025.

Peter D Caie, Rebecca E Walls, Alexandra Ingleston-Orme, Sandeep Daya, Tom Houslay, Rob Eagle, Mark E Roberts, and Neil O Carragher. High-content phenotypic profiling of drug response signatures across distinct cancer cells. Molecular cancer therapeutics, 9(6):1913–1926, 2010.

Junying Chen, Ruyi Ouyang, Anningzhe Gao, Shunian Chen, Guiming Hardy Chen, Xidong Wang, Ruifei Zhang, Zhenyang Cai, Ke Ji, Guangjun Yu, Xiang Wan, and Benyou Wang. Huatuogptvision, towards injecting medical visual knowledge into multimodal llms at scale, 2024. URL https://arxiv.org/abs/2406.19280.

Daniela Chmiest, Nanaocha Sharma, Natacha Zanin, Christine Viaris de Lesegno, Massiullah Shafaq-Zadah, Vonick Sibut, Florent Dingli, Philippe Hupé, Stephan Wilmes, Jacob Piehler, et al. Spatiotemporal control of interferon-induced jak/stat signalling and gene transcription by the retromer complex. Nature communications, 7(1):13476, 2016. doi: 10.1038/ncomms13476.

Haotian Cui, Chloe Wang, Hassaan Maan, Kuan Pang, Fengning Luo, Nan Duan, and Bo Wang. scgpt: toward building a foundation model for single-cell multi-omics using generative ai. Nature methods, 21(8):1470–1480, 2024.

Marie-Laure Fogeron, Hannah Müller, Sophia Schade, Felix Dreher, Verena Lehmann, Anne Kühnel, Anne-Kathrin Scholz, Karl Kashofer, Alexandra Zerck, Beatrix Fauler, et al. Lgals3bp regulates centriole biogenesis and centrosome hypertrophy in cancer cells. Nature communications, 4(1):1531, 2013. doi: 10.1038/ncomms2517.

Google. Gemini 3.1 Flash, 2026. URL https://ai.google.dev/gemini-api/docs/ models.

Kyungsoo Ha, Chengxian Ma, Han Lin, Lichun Tang, Zhusheng Lian, Fang Zhao, Ju-Mei Li, Bei Zhen, Huadong Pei, Suxia Han, et al. The anaphase promoting complex impacts repair choice by protecting ubiquitin signalling at dna damage sites. Nature communications, 8(1):15751, 2017. doi: 10.1038/ncomms15751.

Minsheng Hao, Jing Gong, Xin Zeng, Chiming Liu, Yucheng Guo, Xingyi Cheng, Taifeng Wang, Jianzhu Ma, Xuegong Zhang, and Le Song. Large-scale foundation model on single-cell transcriptomics. Nature methods, 21(8):1481–1491, 2024.

Ashwini Hinge, Juying Xu, Jose Javier, Eucabeth Mose, Sachin Kumar, Reuben Kapur, Edward F Srour, Punam Malik, Bruce J Aronow, and Marie-Dominique Filippi. p190-b rhogap and intracellular cytokine signals balance hematopoietic stem and progenitor cell self-renewal and differentiation. Nature communications, 8(1):14382, 2017. doi: 10.1038/ncomms14382.

Wenyi Hong, Xiaotao Gu, Ziyang Pan, et al. GLM-5V-Turbo: Toward a native foundation model for multimodal agents. arXiv preprint arXiv:2604.26752, 2026. URL https : / /arxiv. org/ abs/2604.26752.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021.

Jon M Laurent, Albert Bou, Michael Pieler, Conor Igoe, Alex Andonian, Siddharth Narayanan, James Braza, Alexandros Sanchez Vassopoulos, Jacob L Steenwyk, Blake Lash, et al. Labbench2: An improved benchmark for ai systems performing biology research. arXiv preprint arXiv:2604.09554, 2026.

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented generation for knowledge-intensive nlp tasks. Advances in neural information processing systems, 33: 9459–9474, 2020.

Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024a.

Bo Li, Qing Wang, Shihang Wang, Bob Zhang, Yuzhong Peng, Pinxian Zeng, Chengliang Liu, Mengran Li, Ziyang Tang, Xiaojun Yao, et al. Mvcbench: A multimodal benchmark for druginduced virtual cell phenotypes. bioRxiv, 2026a.

Chunyuan Li, Cliff Wong, Sheng Zhang, Naoto Usuyama, Haotian Liu, Jianwei Yang, Tristan Naumann, Hoifung Poon, and Jianfeng Gao. Llava-med: Training a large language-and-vision assistant for biomedicine in one day. Advances in neural information processing systems, 36:28541– 28564, 2023.

Mingxin Li, Yanzhao Zhang, Dingkun Long, Keqin Chen, Sibo Song, Shuai Bai, Zhibo Yang, Pengjun Xie, An Yang, Dayiheng Liu, et al. Qwen3-vl-embedding and qwen3-vl-reranker: A unified framework for state-of-the-art multimodal retrieval and ranking. arXiv preprint arXiv:2601.04720, 2026b.

Zekun Li, Xianjun Yang, Kyuri Choi, Wanrong Zhu, Ryan Hsieh, HyeonJung Kim, Jin Hyuk Lim, Sungyoung Ji, Byungju Lee, Xifeng Yan, et al. Mmsci: A dataset for graduate-level multidiscipline multimodal scientific understanding. arXiv preprint arXiv:2407.04903, 2024b.

Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang, Ruochen Xu, and Chenguang Zhu. G-eval: Nlg evaluation using gpt-4 with better human alignment. In Proceedings of the 2023 conference on empirical methods in natural language processing, pp. 2511–2522, 2023.

Vebjorn Ljosa, Katherine L Sokolnicki, and Anne E Carpenter. Annotated high-throughput microscopy image sets for validation. Nature methods, 9(7):637, 2012.

Mohammad Lotfollahi, Anna Klimovskaia Susmelj, Carlo De Donno, Leon Hetzel, Yuge Ji, Ignacio L Ibarra, Sanjay R Srivatsan, Mohsen Naghipourfar, Riza M Daza, Beth Martin, et al. Predicting cellular responses to complex perturbations in high-throughput screens. Molecular systems biology, 19(6):MSB202211517, 2023.

Peter Machamer, Lindley Darden, and Carl F Craver. Thinking about mechanisms. Philosophy of science, 67(1):1–25, 2000.

Xinjie Mao, Songming Zhang, Qianhong Wen, Xiangyu Wen, Kedu Jin, Hao Wu, Shuizhou Chen, Yuqiang Li, Lei Bai, Qi Liu, et al. Benchmarking virtual cell models for in-the-wild perturbation response. arXiv preprint arXiv:2604.27646, 2026.

Andrés Marafioti, Orr Zohar, Miquel Farré, Merve Noyan, Elie Bakouch, Pedro Cuenca, Cyril Zakka, Loubna Ben Allal, Anton Lozhkov, Nouamane Tazi, et al. Smolvlm: Redefining small and efficient multimodal models. arXiv preprint arXiv:2504.05299, 2025.

MiniMax AI. MiniMax-M3, 2026. URL https://github.com/MiniMax-AI/ MiniMax-M3.

Norio Miyamura, Shoji Hata, Tohru Itoh, Minoru Tanaka, Miki Nishio, Michiko Itoh, Yoshihiro Ogawa, Shuji Terai, Isao Sakaida, Akira Suzuki, et al. Yap determines the cell fate of injured mouse hepatocytes in vivo. Nature communications, 8(1):16017, 2017. doi: 10.1038/ ncomms16017.

Moonshot AI. Kimi K2.6. https://www.kimi.ai/ai-models/kimi-k2-6,2026.

Jennifer K Ness, Kristin M Snyder, and Nikos Tapinos. Lck tyrosine kinase mediates β1-integrin signalling to regulate schwann cell migration and myelination. Nature communications, 4(1): 1912, 2013. doi: 10.1038/ncomms2928.

Thomas M Norman, Max A Horlbeck, Joseph M Replogle, Alex Y Ge, Albert Xu, Marco Jost, Luke A Gilbert, and Jonathan S Weissman. Exploring genetic interaction manifolds constructed from rich single-cell phenotypes. Science, 365(6455):786–793, 2019.

Emmanuel Noutahi, Jason Hartford, Prudencio Tossou, Shawn Whitfield, Alisandra K Denton, Cas Wognum, Kristina Ulicna, Michael Craig, Jonathan Hsu, Michael Cuccarese, et al. Virtual cells: Predict, explain, discover. arXiv preprint arXiv:2505.14613, 2025.

OpenAI. Gpt-5 system card. System card, August 2025. URL https : //cdn. openai. com/ gpt-5-system-card. pdf. Published 13 August 2025; family-level source for GPT-5 variants.

Arjun Panickssery, Samuel R Bowman, and Shi Feng. Llm evaluators recognize and favor their own generations. Advances in Neural Information Processing Systems, 37:68772–68802, 2024.

Jonathan Roberts, Kai Han, Neil Houlsby, and Samuel Albanie. Scifibench: Benchmarking large multimodal models for scientific figure interpretation. Advances in Neural Information Processing Systems, 37:18695–18728, 2024.

Yusuf Roohani, Kexin Huang, and Jure Leskovec. Predicting transcriptional outcomes of novel multigene perturbations with gears. Nature Biotechnology, 2023.

Yusuf H Roohani, Tony J Hua, Po-Yuan Tung, Lexi R Bounds, Feiqiao B Yu, Alexander Dobin, Noam Teyssier, Abhinav Adduri, Alden Woodrow, Brian S Plosky, et al. Virtual cell challenge: Toward a turing test for the virtual cell. Cell, 188(13):3370–3374, 2025.

Karen Sachs, Omar Perez, Dana Pe'er, Douglas A Lauffenburger, and Garry P Nolan. Causal protein-signaling networks derived from multiparameter single-cell data. Science, 308(5721): 523–529, 2005.

Artur Szałata, Andrew Benz, Robrecht Cannoodt, Mauricio Cortes, Jason Fong, Sunil Kuppasani, Richard Lieberman, Tianyu Liu, Javier A Mas-Rosario, Rico Meinl, et al. A benchmark for prediction of transcriptomic responses to chemical perturbations across cell types. Advances in Neural Information Processing Systems, 37:20566–20616, 2024.

Haoyi Tao, Chaozheng Huang, Nan Wang, Han Lyu, Linfeng Zhang, Guolin Ke, and Xi Fang. Omniscience: A large-scale multi-modal dataset for scientific image understanding. In Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2, pp. 9860–9870, 2026.

Christina V Theodoris, Ling Xiao, Anant Chopra, Mark D Chaffin, Zeina R Al Sayed, Matthew C Hill, Helene Mantineo, Elizabeth M Brydon, Zexian Zeng, X Shirley Liu, et al. Transfer learning enables predictions in network biology. Nature, 618(7965):616–624, 2023.

Koh-ei Toyoshima, Kyosuke Asakawa, Naoko Ishibashi, Hiroshi Toki, Miho Ogawa, Tomoko Hasegawa, Tarou Irié, Tetsuhiko Tachikawa, Akio Sato, Akira Takeda, et al. Fully functional hair follicle regeneration through the rearrangement of stem cells and their niches. Nature communications, 3(1):784, 2012. doi: 10.1038/ncomms1784.

Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, et al. Internvl3. 5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265, 2025.

Zhijian Wei, Runze Ma, Zichen Wang, Zhongmin Li, Shuotong Song, and Shuangjia Zheng. Vcworld: a biological world model for virtual cell simulation. In International Conference on Learning Representations, volume 2026, pp. 72795–72823, 2026a.

Zhiting Wei, Yiheng Wang, Yicheng Gao, Shuguang Wang, Ping Li, Duanmiao Si, Yuli Gao, Siqi Wu, Danlu Li, Kejing Dong, et al. Benchmarking algorithms for generalizable single-cell perturbation response prediction. Nature Methods, 23(2):451–464, 2026b.

Menghua Rachel Wu, Russell Littman, Jacob Levine, Lin Qiu, Tommaso Biancalani, David Richmond, and Jan-Christian Huetter. Contextualizing biological perturbation experiments through language. In International Conference on Learning Representations, volume 2025, pp. 73598– 73624, 2025a.

Yan Wu, Esther Wershof, Sebastian Schmon, Marcel Nassar, Błażej Osiński, Ridvan Eksi, Zichao Yan, Rory Stark, Kun Zhang, and Thore Graepel. Perturbench: Benchmarking machine learning models for cellular perturbation analysis. Advances in Neural Information Processing Systems, 38, 2025b.

xAI. Introducing grok 4.6. Official model announcement, August 2026. URL https : / /x. ai/ news/grok-4-6. August 2026.

Anyi Xu, Bangcai Lin, Bing Xue, Bingxuan Wang, Bingzheng Xu, Bochao Wu, Bowei Zhang, Chaofan Lin, Chen Dong, Chenchen Ling, et al. Deepseek-v4: Towards highly efficient milliontoken context intelligence. arXiv preprint arXiv:2606.19348, 2026.

Jiayi Ye, Yanbo Wang, Yue Huang, Dongping Chen, Qihui Zhang, Nuno Moniz, Tian Gao, Werner Geyer, Chao Huang, Pin-Yu Chen, et al. Justice or prejudice? quantifying biases in llm-as-ajudge. In International Conference on Learning Representations, volume 2025, pp. 102351– 102390, 2025.

Fan Zhang, Tianyu Liu, Zhihong Zhu, Hao Wu, Haixin Wang, Donghao Zhou, Yefeng Zheng, Kun Wang, Xian Wu, and Pheng-Ann Heng. Cellverse: Do large language models really understand cell biology? Advances in Neural Information Processing Systems, 38, 2025a.

Yaping Zhang, Qixuan Zhang, Xingquan Zhang, Zhiyuan Chen, Wenwen Zhuang, Yupu Liang, Lu Xiang, Yang Zhao, Jiajun Zhang, Yu Zhou, et al. Hiscibench: A hierarchical multi-disciplinary benchmark for scientific intelligence from reading to discovery. arXiv preprint arXiv:2512.22899, 2025b.

Yuhui Zhang, Yuchang Su, Yiming Liu, Xiaohan Wang, James Burgess, Elaine Sui, Chenyu Wang, Josiah Aklilu, Alejandro Lozano, Anjiang Wei, et al. Automated generation of challenging multiple-choice questions for vision language model evaluation. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 29580–29590. IEEE, 2025c.

Jiahao Zhao, Feng Jiang, Shaowei Qin, Zhonghui Zhang, Junhao Liu, Guibing Guo, Hamid Alinejad-Rokny, and Min Yang. Sc-arena: A natural language benchmark for single-cell reasoning with knowledge-augmented evaluation. arXiv preprint arXiv:2602.23199, 2026.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. Judging llm-as-a-judge with mt-bench and chatbot arena. Advances in neural information processing systems, 36:46595–46623, 2023.

Yaowei Zheng, Richong Zhang, Junhao Zhang, Yanhan Ye, and Zheyan Luo. Llamafactory: Unified efficient fine-tuning of 100+ language models. In Proceedings of the 62nd annual meeting of the association for computational linguistics (volume 3: system demonstrations), pp. 400–410, 2024.

Pingping Zhu, Yanying Wang, Ying Du, Lei He, Guanling Huang, Geng Zhang, Xinlong Yan, and Zusen Fan. C8orf4 negatively regulates self-renewal of liver cancer stem cells via suppression of notch2 signalling. Nature communications, 6(1):7122, 2015. doi: 10.1038/ncomms8122.

## Appendix Contents

A Construction and Evaluation Details . . . . . . . 16   
A.1 Model Versions and Access Routes . .............. ............... ... .... ..16   
A.2 Interpretation-Layer Task Definitions ................................... ..16   
A.3 Construction Funnel .... ... 17   
A.4 Source Filtering and QA Curation . .. ..17   
A.5 Dataset Composition and Difficulty Distribution . ..................................... ....18   
A.6 Item and Option Audits .... .................................... ... 20   
A.7 MDHNM Screening Criteria..... ..21   
A.8 AIVC-Judge Rubrics and Production Configuration . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ...22   
A.9 Human Validation Sample and MCQ Performance . . . ..23   
A.10 Human Scoring of Open-Ended Responses . . . . . . ...23   
A.11 Proposal-Oriented Rubric Sensitivity Pilot . . . . ... ..25   
A.12 Release-Level Train-Benchmark Overlap Audit . ..26   
A.13 Cluster-Robust Scores and Paper-Weighted Statistics . . . . . . . . . . .. . . . . . . .. . . . . . . .. . .. ...27   
A.14 Prompts and Run Settings .. ............................. ...28   
B OMNIVCTRAIN Adaptation Details . . . . . . . . . . . ...29   
B.1 Supervised Fine-Tuning .. ...29   
B.2 Retrieval-Augmented Generation.... ...29   
C Biological Evidence Acquisition and Prediction Revision . . . . . . . . . . . . . . . . . . . . . . . . . . ...30   
C.1 Task Structure and Evidence Settings ... ..30   
C.2 Results ... ...30   
C.3 Scope of the Pilots . . . ...31   
D AIVC-Judge Dimension-Wise Results . .. . . . . . . . .. . . . .. . . . . . . . . . . . . . . ...32   
E Limitations and Outlook . . . . ...32   
F Case Studies of Reasoning and Scoring. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ...34   
Single- versus multi-subfigure difficulty . . ..35   
Case A: LGALS3BP readout polarity (L1).............................................. ...36   
Case B: IFNAR1 counterfactual across formats (L1) ............... . . . . . . . . . . . . . . . . . . . . ...37   
Case C: C8orf4 effect direction across models (L1) . . . . . . . . . . ..38   
Case D: ROS–TGFBRI intervention–reversal chain (L2) . . . .. . . . . . . . . . . . . . . . . . . .. . .. . . . . .. ....39   
Case E: CCP1 sign from reciprocal perturbations (L2) . . . ..40   
Case F: BRCA1 spatial integration (L2) . . . .. ...41   
Case G: hair-regeneration hypothesis on a faulty premise (L3) . . . . . . . . . . . . . . . . . . . . . . . . . . . ...42   
Case H: YAP–Ect2/Fgd3 paired-track divergence (L3) . . . ..43   
Case I: integrin–Lck paired-track divergence (L3) . . . . . ...44   
G Ethics, License, and Data Availability . . . . ..45

## A CONSTRUCTION AND EVALUATION DETAILS

## A.1 MODEL VERSIONS AND ACCESS ROUTES

Table 4 lists the exact identifiers, versions, and access routes of the evaluated systems, the construction models, the production judge, the auxiliary annotation and distractor models, and the retrieval embedder.

Table 4: Model identifiers, versions, and access routes. API models are accessed through the listed provider; open-weight checkpoints are run locally from HuggingFace.
<table><tr><td>Model</td><td>Model ID / version</td><td>Source and role</td></tr><tr><td>GPT-5.6-sol</td><td>gpt-5.6-sol</td><td>OpenRouter API; main benchmark</td></tr><tr><td>GPT-5.6-luna</td><td>gpt-5.6-luna</td><td>OpenRouter API; main benchmark</td></tr><tr><td>Grok-4.6</td><td>grok-4.6</td><td>OpenRouter API; main benchmark</td></tr><tr><td>Claude-Sonnet-4-5</td><td>claude-sonnet-4-5</td><td>OpenRouter API; main benchmark</td></tr><tr><td>DeepSeek-V4-Flash-Vision-Exp</td><td>deepseek-v4-flash-vision-exp</td><td>Official DeepSeek API; production AIVC-Judge with image input, temperature 0</td></tr><tr><td>GPT-5.6-terra</td><td>gpt-5.6-terra</td><td>OpenRouter API; auxiliary task-type annotation and distractor generation</td></tr><tr><td>Gemini-3.7-flash-high</td><td>gemini-3.7-flash-high</td><td>OpenRouter API; auxiliary task-type annotation and distractor generation</td></tr><tr><td>Gemini-3.1-flash-lite</td><td>gemini-3.1-flash-lite</td><td>OpenRouter API; direct and mined distractor ablation generator</td></tr><tr><td>Kimi-K2.6</td><td>Kimi-K2.6</td><td>Official Moonshot AI API; QA generation, evidence transcription, and MDHNM distractor rewriting</td></tr><tr><td>Qwen-VL-MAX</td><td>qwen-vl-max</td><td>Alibaba Cloud Bailian API; QA generation</td></tr><tr><td>glm-5v-turbo</td><td>glm-5v-turbo</td><td>OpenRouter API; MDHNM error-pool generation</td></tr><tr><td>minimax-m3</td><td>minimax-m3</td><td>OpenRouter API; MDHNM error-pool generation</td></tr><tr><td>HuatuoGPT-Vision-7B</td><td>HuatuoGPT-Vision-7B</td><td>HuggingFace checkpoint; MDHNM error-pool generation</td></tr><tr><td>InternVL3.5-4B</td><td>InternVL3.5-4B</td><td>HuggingFace checkpoint; local inference</td></tr><tr><td>Qwen2.5-VL-3B</td><td>Qwen2.5-VL-3B</td><td>HuggingFace checkpoint; local inference</td></tr><tr><td>Qwen3-VL-2B</td><td>Qwen3-VL-2B</td><td>HuggingFace checkpoint; local inference</td></tr><tr><td>InternVL3.5-2B</td><td>InternVL3.5-2B</td><td>HuggingFace checkpoint; local inference</td></tr><tr><td>SmolVLM2-2.2B</td><td>SmolVLM2-2.2B</td><td>HuggingFace checkpoint; local inference</td></tr><tr><td>LLaVA-Med-Mistral-7B</td><td>LLaVA-Med-Mistral-7B</td><td>HuggingFace checkpoint; local inference</td></tr><tr><td>LLaVA-OneVision-7B</td><td>LLaVA-OneVision-7B</td><td>HuggingFace checkpoint; local inference</td></tr><tr><td>Qwen2.5-VL-7B</td><td>Qwen2.5-VL-7B</td><td>HuggingFace checkpoint; local inference</td></tr><tr><td>Qwen3-VL-4B</td><td>Qwen3-VL-4B</td><td>HuggingFace checkpoint; local inference</td></tr><tr><td>InternVL3.5-8B</td><td>InternVL3.5-8B</td><td>HuggingFace checkpoint; local inference</td></tr><tr><td>Qwen3-VL-8B</td><td>Qwen3-VL-8B</td><td>HuggingFace checkpoint; local inference</td></tr><tr><td>Qwen3-VL-8B-SFT</td><td>Qwen3-VL-8B + LoRA</td><td>Rank 16, two epochs; local inference; rank-32 comparison in Table 3</td></tr><tr><td>Qwen3-VL-Embedding-8B</td><td>Qwen3-VL-Embedding-8B</td><td>HuggingFace checkpoint; local inference; multimodal RAG index construction</td></tr></table>

## A.2 INTERPRETATION-LAYER TASK DEFINITIONS

Table 5 reports the operational definitions of the three interpretation levels; P-E-D denotes functional alignment, not label equivalence.

Table 5: Interpretation-layer task definitions. P-E-D denotes functional alignment, not label equivalence.
<table><tr><td>Level</td><td>Required operation</td><td>P-E-D alignment</td><td>Bloom levels</td><td>n</td></tr><tr><td>L1</td><td>Infer a value, direction, relation, or outcome from displayed evidence</td><td>Predict analogue</td><td>2,3,4</td><td>2,576</td></tr><tr><td>L2</td><td>Explain an observed result through a biologically grounded mechanism</td><td>Explain</td><td>4</td><td>1,763</td></tr><tr><td>L3</td><td>Propose a testable hypothesis grounded in the displayed evidence</td><td>Discover prerequisite</td><td>5</td><td>1,738</td></tr></table>

L3 task-type audit. To verify that released L3 items match the hypothesis-proposal definition, two independent annotator models (Gemini-3.7-flash-high and GPT-5.6-terra) each labeled all 1,738 L3 items by task type (Table 6). The two labelers agree closely (observed agreement 0.9937; Cohen's κ = 0.9068) and both assign over 96% of items to propose\_novel; design\_experiment, other, and evaluate\_given are rare. The audit identifies hypothesis proposal as the dominant requested operation within L3. It concerns task type, while Table 9 records the construction-time Bloom labels. Level labels are construction-time assignments, and boundary items exist: Case F (Appendix F.7) requests a spatial-relation comparison that sits close to the L1 definition although labeled L2.

Table 6: Task-type audit of all 1,738 L3 items by two independent annotator models (counts). Observed agreement 0.9937; Cohen's κ = 0.9068.
<table><tr><td>Task type</td><td>Gemini-3.7-flash-high</td><td>GPT-5.6-terra</td></tr><tr><td>propose_novel</td><td>1,679</td><td>1,676</td></tr><tr><td>design_experiment</td><td>31</td><td>35</td></tr><tr><td>other</td><td>21</td><td>20</td></tr><tr><td>evaluate_given</td><td>7</td><td>7</td></tr></table>

## A.3 CONSTRUCTION FUNNEL

Table 7 reports the retained counts at the principal construction stages. The final audit stage was performed by three cell biology PhD annotators who reviewed candidates independently, blind to one another's judgments and to the generator model's identity; labels were merged only after all three completed their review, and a candidate item entered the benchmark only when all three approved it. The table intentionally reports only the stage totals and does not treat records and QA items as interchangeable units.

Table 7: Retained counts in the OMNISCIENCE-OMNIVCBENCH construction funnel.
<table><tr><td>Stage</td><td>Retained count</td></tr><tr><td>OmniScience initial figure-caption pairs</td><td>1,525,179</td></tr><tr><td>Stage 0 keyword filtering</td><td>98,632</td></tr><tr><td>QA generation + collaborative filtering</td><td>7,923</td></tr><tr><td>Unanimous three-annotator audit</td><td>6,077</td></tr></table>

The audit decided 7,923 candidate items: 6,077 were accepted by all three annotators and retained, 1,303 were rejected by all three, and 543 carried disagreement and were not retained. The retention rate is 76.70%, and the three annotators fully agreed on 93.15% of candidates. Pairwise Cohen's κ is 0.8545 [0.8397, 0.8687] (annotators 1–2), 0.8506 [0.8357, 0.8650] (annotators 1–3), and 0.8630 [0.8487, 0.8769] (annotators 2–3); overall Fleiss'κ is 0.8560 [0.8442, 0.8674], with bracketed values denoting 95% confidence intervals.

## A.4 SOURCE FILTERING AND QA CURATION

All captions and surrounding contexts are the author-written text of the source papers, with no model rewriting in the OmniScience corpus (Tao et al., 2026); these captions are typically self-contained, describing both the visual content and the experimental context of their figures. The initial retrieval vocabulary covers six AIVC-relevant themes: virtual cells and cellular digital twins; perturbation responses; single-cell and spatial omics; multi-omics integration; mechanisms and pathways; and disease-relevant cell biology. Representative terms are shown in Table 8.

Table 8: Representative terms used to retrieve AIVC-relevant source records from OmniScience.
<table><tr><td>Theme</td><td>Representative terms</td></tr><tr><td>Virtual cells</td><td>virtual cell; AI virtual cell; digital twin cell; whole-cell model</td></tr><tr><td>Perturbations</td><td>perturbation; CRISPR screen; knockout; drug or dose response</td></tr><tr><td>Single-cell/spatial</td><td>single-cell RNA-seq; spatial transcriptomics; UMAP; RNA velocity</td></tr><tr><td>Multi-omics</td><td>multi-omics; ATAC-seq; proteomics; chromatin accessibility</td></tr><tr><td>Mechanisms</td><td>signalling pathway; gene-regulatory network; causal mechanism</td></tr><tr><td>Disease biology</td><td>cancer; drug resistance; therapeutic hypothesis; patient-derived model</td></tr></table>

Candidate QA pairs undergo schema, source-evidence, answer-specificity, and visual-grounding checks. Schema checks verify fields, panel references, and answer length. Source-evidence checks compare reference answers with captions and context. Answer checks assess whether the question specifies its requested inference or explanation. L3 references anchor hypothesis assessment to source observations, while hypothesis validity depends on evidence consistency and testable consequences. Visual checks screen for text-only shortcuts. Flagged candidates are rewritten or removed before the unanimous three-annotator audit (Appendix A.3). Option-level uniqueness is assessed separately for the six-option track (Appendix A.7).

Table 9 (left) reports the empirical relation between the final interpretation levels and P-E-D labels. The small off-diagonal counts show why we treat P-E-D as a functional alignment rather than an identity: cognitive depth and answer function provide distinct information.

Table 9: Interpretation level by P-E-D label (left) and by Bloom level (right; item counts) in the final benchmark. Parentheses give the within-level percentage of the predominant P-E-D label.
<table><tr><td colspan="5">P-E-D label</td><td colspan="5">Bloom level</td></tr><tr><td>Level</td><td>Predict</td><td>Explain</td><td>Discover</td><td>Total</td><td>B2</td><td>B3</td><td>B4</td><td>B5</td><td>Total</td></tr><tr><td>L1</td><td>2,362 (91.7%)</td><td>214</td><td>0</td><td>2,576</td><td>261</td><td>773</td><td>1,542</td><td>0</td><td>2,576</td></tr><tr><td>L2</td><td></td><td>0 1,763 (100%)</td><td>0</td><td>1,763</td><td>0</td><td>0</td><td>1,763</td><td>0</td><td>1,763</td></tr><tr><td>L3</td><td>9</td><td>9</td><td>1,720 (99.0%)</td><td>1,738</td><td>0</td><td>0</td><td>0</td><td>1,738</td><td>1,738</td></tr></table>

## A.5 DATASET COMPOSITION AND DIFFICULTY DISTRIBUTION

An item is one question-answer pair, a source record is an OmniScience figure record, and a paper is a source article. An image path identifies a materialized panel crop or full figure. A source record can yield multiple images and questions, so these units are counted separately. Tables 9 and 10 report the construction-label distribution. Retained items use Bloom labels B2-B5, with 83.0% assigned to Analyze or Evaluate. These labels describe the dataset's construction taxonomy. Within B4, the task-role labels distinguish 1,542 inference items from 1,763 mechanistic-explanation items. L3 items carry B5 labels and request evidence-grounded hypotheses; these construction labels are distinct from a validated measurement of cognitive difficulty.

Table 10: Distribution over Bloom's taxonomy levels (left) and P-E-D functions (right) in the final benchmark.
<table><tr><td colspan="4">Bloom&#x27;s taxonomy</td><td colspan="4">P-E-D function</td></tr><tr><td>Level</td><td>Category</td><td>n</td><td>Share</td><td>Function</td><td>Operational meaning</td><td></td><td>Share</td></tr><tr><td>2</td><td>Understand</td><td>261</td><td>4.3%</td><td>Predict</td><td>Evidence-conditioned inference</td><td>2,371</td><td>39.0%</td></tr><tr><td>3</td><td>Apply</td><td>773</td><td>12.7%</td><td>Explain</td><td>Mechanistic explanation</td><td>1,986</td><td>32.7%</td></tr><tr><td>4</td><td>Analyze</td><td>3,305</td><td>54.4%</td><td>Discover</td><td>Testable-hypothesis proposal grounded in displayed evidence</td><td>1,720</td><td>28.3%</td></tr><tr><td>5</td><td>Evaluate</td><td>1,738</td><td>28.6%</td><td></td><td></td><td></td><td></td></tr></table>

Corpus units and provenance. The released corpora derive from 1,525,179 OmniScience source records. Keyword filtering retains 98,632 records; 96,007 of them, spanning 33,802 papers, expand into 548,450 OMNIVCTRAIN examples (327,601 materialized image paths), while 2,625 records from 1,080 papers yield the 6,077 benchmark items (4,428 image paths). Source coverage is therefore 6.29% and 0.17% of the upstream corpus, respectively. Here multi-subfigure items integrate evidence across subfigures (panels) of a single composite figure: each item is materialized as one image (the whole figure or a panel crop), 50.9% of items are built over multi-panel figures whose evidence spans several panels, and items reference a mean of 1.78 panels. Table 11 reports unit-level statistics.

Table 11: Unit-level statistics of the released corpora.
<table><tr><td>Unit</td><td>OMNIVCTRAIN</td><td>OMNIVCBENCH</td></tr><tr><td>Released QA rows</td><td>548,450</td><td>6,077</td></tr><tr><td>Unique source records</td><td>96,007</td><td>2,625</td></tr><tr><td>Unique papers</td><td>33,802</td><td>1,080</td></tr><tr><td>Materialized image paths</td><td>327,601</td><td>4,428</td></tr><tr><td>QA per source record (mean / median)</td><td>5.71 / 6</td><td>2.32 /2</td></tr><tr><td>QA per paper (mean / median)</td><td>16.23 / 12</td><td>5.63 / 4</td></tr></table>

Subject, source, and length distribution. Subject labels are OmniScience's coarse source metadata rather than manual content annotations; the two released corpora concentrate in biology and medicine subjects (Table 12). All benchmark items derive from Nature Communications, whereas OMNIVCTRAIN draws mainly from Nature Communications (64.0%), bioRxiv (10.5%), Cell Reports (8.7%), Scientific Reports (5.7%), and PLOS ONE (4.8%). Table 13 reports question and answer lengths.

Table 12: Subject metadata distribution (QA-row weights; source-record weights differ by < 1 point in each cell).
<table><tr><td>Subject</td><td>OMNIVCTRAIN</td><td>OMNIVCBENCH</td></tr><tr><td>Biology</td><td>75.5%</td><td>73.3%</td></tr><tr><td>Medicine</td><td>21.7%</td><td>26.4%</td></tr><tr><td>Physics</td><td>0.2%</td><td>0.4%</td></tr><tr><td>Others (train only)</td><td>2.6%</td><td>0.0%</td></tr></table>

Table 13: Question and answer lengths (Unicode characters / whitespace-separated words).
<table><tr><td>Field</td><td>Median chars</td><td>Mean chars</td><td>Median words</td><td>Mean words</td></tr><tr><td>Train questions</td><td>147</td><td>154.2</td><td>24</td><td>25.3</td></tr><tr><td>Train answers</td><td>235</td><td>249.1</td><td>35</td><td>37.3</td></tr><tr><td>Bench questions</td><td>234</td><td>238.0</td><td>37</td><td>37.3</td></tr><tr><td>Bench answers</td><td>358</td><td>364.1</td><td>52</td><td>53.6</td></tr></table>

Source-paper publication years. Crossref DOI records resolve the publication years of all 1,080 source papers, with standardized titles matching the benchmark records. The papers span 2011–2017 (Table 14). MCQ accuracy varies across years without a monotone trend in the evaluated models (Table 15). The overlap audit in Appendix A.12 addresses the released training and benchmark corpora; the present evaluation uses this historical source distribution.

Table 14: Publication-year distribution of the 1,080 source papers and the 6,077 benchmark items (Crossref, by DOI). Papers are deduplicated by DOI; QA counts keep the per-paper item weight.
<table><tr><td>Year</td><td>Papers</td><td>QA items</td></tr><tr><td>2011</td><td>30 (2.8%)</td><td>182 (3.0%)</td></tr><tr><td>2012</td><td>121 (11.2%)</td><td>723 (11.9%)</td></tr><tr><td>2013</td><td>233 (21.6%)</td><td>912 (15.0%)</td></tr><tr><td>2014</td><td>25 (2.3%)</td><td>110 (1.8%)</td></tr><tr><td>2015</td><td>160 (14.8%)</td><td>737 (12.1%)</td></tr><tr><td>2016</td><td>152 (14.1%)</td><td>861 (14.2%)</td></tr><tr><td>2017</td><td>359 (33.2%)</td><td>2,552 (42.0%)</td></tr></table>

Table 15: Item-macro MCQ accuracy (%) by source-paper publication year. Year cells cover 110– 2,552 items each (Table 14); no model shows a monotone year trend.
<table><tr><td>Model</td><td>2011</td><td>2012</td><td>2013</td><td>2014</td><td>2015</td><td>2016</td><td>2017</td></tr><tr><td>GPT-5.6-sol</td><td>51.1</td><td>53.3</td><td>61.0</td><td>57.3</td><td>55.8</td><td>54.1</td><td>53.8</td></tr><tr><td>Grok-4.6</td><td>49.5</td><td>55.3</td><td>42.7</td><td>59.1</td><td>57.9</td><td>50.4</td><td>49.1</td></tr><tr><td>Claude-Sonnet-4-5</td><td>49.5</td><td>43.9</td><td>52.6</td><td>44.6</td><td>48.2</td><td>48.6</td><td>50.6</td></tr><tr><td>GPT-5.6-luna</td><td>47.3</td><td>38.0</td><td>48.9</td><td>38.2</td><td>41.7</td><td>42.0</td><td>44.7</td></tr><tr><td>Qwen3-VL-8B-SFT</td><td>29.1</td><td>32.6</td><td>36.6</td><td>38.2</td><td>30.3</td><td>35.8</td><td>34.8</td></tr><tr><td>Qwen3-VL-8B</td><td>29.1</td><td>32.2</td><td>35.8</td><td>35.5</td><td>30.8</td><td>33.7</td><td>34.8</td></tr><tr><td>Qwen3-VL-4B</td><td>26.4</td><td>27.4</td><td>31.7</td><td>34.6</td><td>31.5</td><td>31.6</td><td>30.4</td></tr><tr><td>InternVL3.5-8B</td><td>25.8</td><td>30.7</td><td>31.5</td><td>32.7</td><td>28.5</td><td>28.2</td><td>31.2</td></tr><tr><td>InternVL3.5-4B</td><td>23.1</td><td>25.6</td><td>29.9</td><td>33.6</td><td>30.0</td><td>28.7</td><td>29.4</td></tr><tr><td>Qwen2.5-VL-7B</td><td>25.3</td><td>25.6</td><td>27.5</td><td>28.2</td><td>24.4</td><td>27.1</td><td>27.1</td></tr><tr><td>LLaVA-OneVision-7B</td><td>22.5</td><td>23.4</td><td>27.7</td><td>25.5</td><td>25.4</td><td>24.9</td><td>25.9</td></tr><tr><td>Qwen2.5-VL-3B</td><td>20.3</td><td>19.2</td><td>25.8</td><td>17.3</td><td>23.1</td><td>23.1</td><td>22.3</td></tr><tr><td>InternVL3.5-2B</td><td>18.1</td><td>18.1</td><td>22.5</td><td>21.8</td><td>20.6</td><td>22.0</td><td>20.9</td></tr><tr><td>Qwen3-VL-2B</td><td>14.8</td><td>18.7</td><td>24.8</td><td>18.2</td><td>21.3</td><td>23.3</td><td>21.6</td></tr><tr><td>SmolVLM2-2.2B</td><td>17.0</td><td>18.8</td><td>17.0</td><td>17.3</td><td>15.5</td><td>18.2</td><td>17.2</td></tr><tr><td>LLaVA-Med-Mistral-7B</td><td>17.6</td><td>15.8</td><td>17.5</td><td>20.0</td><td>15.3</td><td>16.1</td><td>15.1</td></tr></table>

Species, assay, and cell-type coverage. All 1,080 source papers were annotated for species, experimental assays, and cell types (Table 16). The corpus is dominated by mouse (49.9% of papers) and human (38.2%) studies, with smaller shares of rat (5.1%), bacteria (3.4%), fruit fly (2.5%), yeast (2.3%), zebrafish (2.2%), plant (1.5%), and worm (1.0%). Microscopy (61.9%) and immunoblot (41.8%) are the most frequent assays, and the top cell types are standard cell lines (HeLa 7.4%, HEK293T 4.1%, HEK293 3.4%).

Table 16: Coverage of the 1,080 source papers (% of papers carrying each label). Labels are multivalued and not normalized (variants such as MEF/MEFs and MCF-7/MCF7 co-occur), so entries within a block need not sum to 100%.
<table><tr><td>Species</td><td>Assays</td><td></td><td></td><td>Cell types (top 10)</td></tr><tr><td>mouse</td><td>49.9</td><td>microscopy</td><td>61.9</td><td>HeLa 7.4</td></tr><tr><td>human</td><td>38.2</td><td>immunoblot</td><td>41.8</td><td>HEK293T 4.1</td></tr><tr><td>rat</td><td>5.1</td><td>genetic perturbation</td><td>37.4</td><td>HEK293 3.4</td></tr><tr><td>bacteria</td><td>3.4</td><td>qPCR</td><td>36.9</td><td>U2OS 2.3</td></tr><tr><td>fruit fly</td><td>2.5</td><td>binding interaction</td><td>19.4</td><td>fibroblasts 2.3</td></tr><tr><td>yeast</td><td>2.3</td><td>flow cytometry</td><td>19.4</td><td>MDA-MB-231 2.2</td></tr><tr><td>zebrafish</td><td>2.2</td><td>viability/proliferation</td><td>17.1</td><td>MEF 2.2</td></tr><tr><td>plant</td><td>1.5</td><td>chromatin assay</td><td>12.7</td><td>endothelial cells 2.2</td></tr><tr><td>worm</td><td>1.0</td><td>microarray</td><td>10.0</td><td>neurons 2.2</td></tr><tr><td>other</td><td>6.0</td><td>RNA sequencing</td><td>9.9</td><td>MCF-7 2.1</td></tr></table>

## A.6 ITEM AND OPTION AUDITS

The response-derived difficulty statistic has median 0.25 and mean 0.3166. Its point-biserial correlation is defined per item as the leave-one-item-out point-biserial correlation of correctness across models; it is available for 5,606/6,077 items (mean 0.3143, median 0.3877, 10th percentile —0.1839, 90th percentile 0.7269), with the remaining 471 items answered incorrectly by every model (zero variance, undefined correlation). An eleven-model sensitivity analysis yields mean 0.2922 (5,518 items). This is a descriptive discrimination statistic within the evaluated model pool, not a human answerability measure. Correct-option counts are A/B/C/D/E/F = 1,001/999/964/1,011/1,036/1,066 (15.86%–17.54%), and every distractor originates from an observed model error. These statistics are based on model responses and do not measure human answerability or inter-annotator agreement.

A stratified 300-item sample—127 L1, 87 L2, and 86 L3 questions from 251 source papers, balanced by single- and multi-subfigure structure—supports the input comparison below, the distractor ablations of Appendix A.7, and the case studies of Appendix F.

No-image shortcut test. Five models answer the stratified 300-item sample using only the question and six options (Table 17). GPT-5.6-sol drops from 57.3% with images to 34.7% without images, a decrease of 22.6 percentage points. InternVL3.5-2B and SmolVLM2-2.2B each score 13.7% without images, against a 16.7% random baseline. LLaVA-OneVision-7B changes from 23.0% to 24.0%, and Qwen3-VL-2B from 20.3% to 21.7%. The contribution of visual input is therefore model-dependent. GPT-5.6-sol benefits from images and also extracts useful information from the question and options, domain priors, and potentially memorized source content (Appendix E).

Table 17: No-image shortcut test on the stratified 300-item sample: MCQ accuracy (%) when models receive only the question and the six options, without the image or any transcribed evidence. The six-option random baseline is 16.7%. For reference, GPT-5.6-sol scores 57.3% with the image on the same items.
<table><tr><td>Model</td><td>L1</td><td>L2</td><td>L3</td><td>Avg</td></tr><tr><td>GPT-5.6-sol</td><td>37.8</td><td>39.1</td><td>25.6</td><td>34.7</td></tr><tr><td>LLaVA-OneVision-7B</td><td>23.6</td><td>25.3</td><td>23.3</td><td>24.0</td></tr><tr><td>Qwen3-VL-2B</td><td>22.0</td><td>23.0</td><td>19.8</td><td>21.7</td></tr><tr><td>InternVL3.5-2B</td><td>11.0</td><td>16.1</td><td>15.1</td><td>13.7</td></tr><tr><td>SmolVLM2-2.2B</td><td>11.8</td><td>16.1</td><td>14.0</td><td>13.7</td></tr></table>

## A.7 MDHNM SCREENING CRITERIA

For each open-ended item, twelve rollouts per pool model provide naturally occurring candidate errors. We use HuatuoGPT-Vision-7B (Chen et al., 2024), minimax-m3 (MiniMax AI, 2026) and glm-5v-turbo (Hong et al., 2026) to generate the error pool. Responses with AIVC-Judge overall scores above 2, judged with the original image included as in production scoring, are removed from the error pool. Kimi-K2.6 then rewrites at most five retained errors using minimal edits, preserving the wrong inference while aligning answer length, syntax, units, and level of detail. The final review applies the criteria in Table 18; near-duplicates are merged or removed before option ordering. These construction criteria are complemented by the unanimous three-annotator audit (Appendix A.3).

Table 18: Criteria for selecting MDHNM distractors.
<table><tr><td>Criterion</td><td>Requirement</td></tr><tr><td>Strict incorrectness</td><td>The screening target is an option contradicted by the evidence or incompatible with the requested answer; an alternative hypothesis requires a substantive scientific</td></tr><tr><td>Plausibility</td><td>distinction. The error should reflect a biologically or visually plausible model failure.</td></tr><tr><td>Relevance and grounding</td><td>The option must answer the question and refer only to entities or patterns present in the item.</td></tr><tr><td>Diversity Uniqueness</td><td>The five distractors should represent distinct error modes rather than paraphrases. The screening target is one correct option with semantically distinct distractors;</td></tr><tr><td></td><td>subsequent audits examine violations of this target.</td></tr><tr><td>Style consistency</td><td>Length, grammar, specificity, and units should not reveal the correct answer.</td></tr></table>

## A.7.1 DIRECT-GENERATION DISTRACTOR ABLATION

We compare MDHNM distractors with distractors written directly by strong generators on the stratified 300-item sample (Appendix A.6). Each baseline generator—Grok-4.6 (xAI, 2026), Gemini-3.1-flash-lite (Google, 2026), and Kimi-K2.6 (Moonshot AI, 2026)—receives only the question and the reference answer and writes five distractors directly, without access to the MDHNM error pool; the original reference answer is inserted once among the six options, with option letters matching the corresponding released MDHNM item. A fixed solver, GPT-5.6-sol, answers at temperature 0 from the original image, the question, and the six options. The primary metric is the fixed solver's MCQ accuracy: the easier a generator's distractors, the higher the solver scores. Confidence intervals for accuracy differences use 20,000 source-paper cluster bootstrap replicates, and paired significance uses an exact two-sided McNemar test.

Table 19: Direct-generation distractor ablation on the stratified 300-item sample: MCQ accuracy (%) of the fixed solver (GPT-5.6-sol, temperature 0) on items whose distractors were written directly by each baseline generator, versus the released MDHNM items on the same questions. Discordant pairs are (correct only on direct) / (correct only on MDHNM); p is the exact two-sided McNemar test, and bracketed intervals are source-paper cluster bootstrap 95% CIs (20,000 replicates).
<table><tr><td>Distractor generator</td><td>Direct</td><td>MDHNM</td><td>∆ (pp) [95% CI]</td><td>Discordant</td><td>McNemar p</td></tr><tr><td>Grok-4.6</td><td>87.7</td><td>57.3</td><td>+30.3 [+23.8, +36.7]</td><td>111/20</td><td> $< 1 0 ^ { - 6 }$ </td></tr><tr><td>Gemini-3.1-flash-lite</td><td>95.3</td><td>57.3</td><td>+38.0 [+32.2, +43.9]</td><td>120/6</td><td> $< 1 0 ^ { - 6 }$ </td></tr><tr><td>Kimi-K2.6</td><td>93.3</td><td>57.3</td><td>+36.0 [+30.1, +41.9]</td><td>116/8</td><td> $< 1 0 ^ { - 6 }$ </td></tr></table>

Same-evidence ablation. The direct-generation baselines above receive less evidence than the MDHNM pipeline. To compare direct generation with error conditioning under matched inputs, we built a same-evidence comparison in which two generator arms both receive the original image, question, reference answer, and caption/context under identical budgets (five distractors, temperature 0, 8,192 output tokens, at most two attempts), with the correct option's text and position frozen; the direct arm writes distractors from scratch while the mined arm is conditioned on mined inference errors. Two solvers answer the resulting items: GPT-5.6-sol and Grok-4.6, neither of which contributed to the released items’ construction. Table 21 reports accuracy across five distractor arms. Direct distractors remain far easier than the released MDHNM items for both solvers, and the paired

Table 20: Per-level breakdown of the direct-generation ablation (127 L1, 87 L2, 86 L3 items). The MDHNM column is the fixed solver's accuracy on the released items; it repeats within each level block because it does not depend on the baseline generator.
<table><tr><td>Level</td><td>Distractor generator</td><td>Direct</td><td>MDHNM</td><td> $\Delta \left( \mathsf { p p } \right)$ </td></tr><tr><td>L1</td><td>Grok-4.6</td><td>86.6</td><td>63.8</td><td>+22.8</td></tr><tr><td>L1</td><td>Gemini-3.1-flash-lite</td><td>91.3</td><td>63.8</td><td>+27.6</td></tr><tr><td>L1</td><td>Kimi-K2.6</td><td>92.9</td><td>63.8</td><td>+29.1</td></tr><tr><td>L2</td><td>Grok-4.6</td><td>86.2</td><td>65.5</td><td>+20.7</td></tr><tr><td>L2</td><td>Gemini-3.1-flash-lite</td><td>97.7</td><td>65.5</td><td>+32.2</td></tr><tr><td>L2</td><td>Kimi-K2.6</td><td>92.0</td><td>65.5</td><td>+26.4</td></tr><tr><td>L3</td><td>Grok-4.6</td><td>90.7</td><td></td><td> $3 9 . 5 \quad + 5 1 . 2 \quad$ </td></tr><tr><td>L3</td><td>Gemini-3.1-flash-lite</td><td>98.8</td><td></td><td> $3 9 . 5 \quad + 5 9 . 3 \quad$ </td></tr><tr><td>L3</td><td>Kimi-K2.6</td><td>95.3</td><td></td><td> $3 9 . 5 \quad + 5 5 . 8 \quad$ </td></tr></table>

direct-versus-mined comparison for the same Gemini-3.1-flash-lite generator is small and not significant after Holm correction: the direct arm exceeds the mined arm $\mathrm { \ b y + 3 . 7 p p }$ for GPT-5.6-sol $[ + 0 . 3 , + 7 . 1 ]$ and $+ 2 . 7 \mathrm { p p }$ for Gro $) \mathbf { k } { - } 4 . 6 ~ [ - 0 . 3 , + 5 . 7 ]$ (CI on the accuracy difference; significance by exact McNemar, $p = . 0 5 2$ and $p = . 1 1 5 )$ One arm carries a length artifact: the correct option is the longest in 57.3% of Gemini-3.7-flash-high direct items (0.037–0.150 elsewhere); a symmetric length/identity filter $( n = 2 9 9 )$ leaves all conclusions unchanged.

Table 21: Same-evidence distractor ablation on the stratified 300-item sample: MCQ accuracy (%) of two solvers on five distractor arms. Both generator arms receive identical evidence and budgets; mined arms are conditioned on mined inference errors, direct arms are not. original MDHNM denotes the released items. For the paired gemini-3.1-flash-lite arms, a positive direct-minus-mined difference means the direct distractors are easier.
<table><tr><td>Distractor arm</td><td>GPT-5.6-sol</td><td>Grok-4.6</td></tr><tr><td>gemini-3.1-flash-lite direct</td><td>93.7</td><td>94.0</td></tr><tr><td>gemini-3.1-flash-lite mined</td><td>90.0</td><td>91.3</td></tr><tr><td>gemini-3.7-flash-high direct</td><td>94.3</td><td>93.7</td></tr><tr><td>gpt-5.6-terra direct</td><td>92.0</td><td>91.7</td></tr><tr><td>original MDHNM</td><td>57.3</td><td>48.3</td></tr></table>

Under the question-and-reference setting, GPT-5.6-sol scores 30.3–38.0 percentage points higher on directly generated distractors than on released MDHNM items. The same-evidence study extends the comparison to two solvers. With Gemini-3.1-flash-lite as the generator, direct-minus-mined differences are 3.7 and 2.7 points. Neither contrast reaches significance under the reported Holm correction. Regenerated mined arms yield 90.0% and 91.3% solver accuracy, compared with 57.3% and 48.3% for the released items. The observed difficulty difference therefore concerns the complete released construction configuration. We report MDHNM as the benchmark's distractor-construction procedure, with difficulty and option validity assessed separately: error-pool mining supplies candidate failure seeds, while the iterative rewriting, screening criteria, and unanimous audit funnel drive the final discriminative difficulty.

## A.8 AIVC-JUDGE RUBRICS AND PRODUCTION CONFIGURATION

Table 22 lists the scoring dimensions selected for each task level. Each dimension uses a 1–5 integer scale with checklist-based guidance. For L1, a key numerical, directional, or entity error caps correctness at 2. For L2, a claim contradicted by the closed evidence caps faithfulness at 2. These rules penalize decisive scientific errors even in otherwise fluent responses.

The judge returns atomic claims, evidence verdicts, a scoring rationale, dimension scores, and a raw overall score. The raw overall score is a holistic judgment made after claim checking and dimension scoring. The parser rounds overall scores to integers and clips them to the 1–5 scale. Records with dimension ratings but no overall score contribute only to dimension-based analyses. Main tables average these item-level overall scores; dimension averages provide separate diagnostics. JSON validation checks the required fields and the 1–5 score range. Malformed responses are retried, and persistent failures are logged separately from valid scores

Table 22: Level-conditioned dimensions used by AIVC-Judge. The recorded L3 dimensions assess source-referenced critique and argumentation.
<table><tr><td>Level</td><td>Scoring dimensions</td></tr><tr><td>L1</td><td>correctness; evidence support</td></tr><tr><td>L2</td><td>faithfulness; causal completeness; mechanistic granularity; evidence consistency</td></tr><tr><td>L3</td><td>judgment correctness; argumentation quality; evidence weighing</td></tr></table>

Table 23: Production AIVC-Judge configuration used for the reported open-response scores.
<table><tr><td>Component</td><td>Reported configuration</td></tr><tr><td>Judge</td><td>DeepSeek-V4-Flash-Vision-Exp(deepseek-v4-flash-vision-exp);single pointwise judge; temperature 0</td></tr><tr><td>Response input</td><td>question and candidate response</td></tr><tr><td>Closed evidence</td><td>figure caption, surrounding paper context, and complete reference answer</td></tr><tr><td>Visual input</td><td>original figure image set, as shown to the answering model</td></tr><tr><td>Reference key points</td><td>disabled; the complete reference answer provides the scoring anchor</td></tr><tr><td>Anonymization</td><td>evaluated model identity omitted from the judge input</td></tr><tr><td>Output</td><td>claim verdicts, rationale, dimension scores, and raw overall score</td></tr></table>

## A.9 HUMAN VALIDATION SAMPLE AND MCQ PERFORMANCE

The validation sample contained 300 questions from 251 source papers, stratified by AIVC level and single-/multi-subfigure structure. It comprised 127 L1, 87 L2, and 86 L3 questions. Three human annotators, denoted Human 1-3, participated in both MCQ answering and open-response scoring. They are cell biology PhD annotators, forming a group independent of the three construction auditors (Appendix A.3). For MCQ answering, they received the original image, question, and six options, with the reference answer and answer key withheld. Responses followed the sample order with fixed option positions.

MCQ accuracy used all 300 assigned questions, counting missing answers as incorrect (Table 24). Human 1 provided 283 valid answers; Humans 2 and 3 each provided 300. Accuracy was 55.3%, 71.3%, and 74.7%, respectively, with the lowest level-wise accuracy on L3 for each annotator.

Table 24: Human MCQ accuracy (%) on the fixed 300-question sample. Level denominators are L1/L2/L3 = 127/87/86. Human 1's 17 missing answers count as incorrect. Brackets denote 95% source-paper cluster bootstrap confidence intervals.
<table><tr><td>Annotator</td><td>Correct/assigned</td><td>Overall [95% CI]</td><td>L1</td><td>L2</td><td>L3</td></tr><tr><td>Human 1</td><td>166/300</td><td>55.3 [50.0, 60.7]</td><td>55.9</td><td>57.5</td><td>52.3</td></tr><tr><td>Human 2</td><td>214/300</td><td>71.3 [65.8, 76.7]</td><td>76.4</td><td>81.6</td><td>53.5</td></tr><tr><td>Human 3</td><td>224/300</td><td>74.7 [69.1, 79.5]</td><td>75.6</td><td>87.4</td><td>60.5</td></tr></table>

Option agreement was estimated on the 283 questions with valid answers from all three annotators. Fleiss’κ was 0.589 [0.543, 0.635], and nominal Krippendorff's α was also 0.589 [0.543, 0.635]. Table 25 reports pairwise agreement, treating the six answer options as nominal categories. Missing answers were excluded from agreement estimation and were never assigned an artificial option.

## A.10 HUMAN SCORING OF OPEN-ENDED RESPONSES

The same three annotators (Human 1-3, Appendix A.9) scored existing responses from GPT-5.6- sol, Qwen3-VL-8B, and LLaVA-Med-Mistral-7B on the stratified 300-item sample (Appendix A.6). Each candidate supplied one nonempty response per question, yielding 900 candidate-question response cells. Human scorers received the question, anonymous candidate response, caption, surrounding context, and reference answer, but neither the original image nor other scorers' ratings.

Table 25: Human-human agreement on selected MCQ options. Each comparison uses the same 283 complete questions. Brackets denote 95% source-paper cluster bootstrap confidence intervals.
<table><tr><td>Annotator pair</td><td>Cohen&#x27;s κ [95% CI]</td><td>Exact agreement</td></tr><tr><td>Human 1–Human 2</td><td>0.456 [0.391, 0.520]</td><td>54.8%</td></tr><tr><td>Human 1–Human 3</td><td>0.457 [0.391, 0.523]</td><td>54.8%</td></tr><tr><td>Human 2–Human 3</td><td>0.855 [0.805, 0.897]</td><td>88.0%</td></tr></table>

The production judge scores the same responses with the original figure image as additional visual input; human-judge agreement therefore concerns the shared rubric and closed textual evidence rather than identical visual inputs.

Scores, coverage, and statistical units. The primary validation uses raw overall scores on the same 1–5 scale as the main results. Available raw ratings numbered 887, 900, and 900 for Humans 1–3, and 897 for the production judge. All four scorers had valid raw ratings on 884 response cells from 297 questions and 249 papers. Pairwise analyses use each scorer pair's available intersection; the panel analysis requires valid scores from all four scorers. No missing score was imputed for correlations or agreement coefficients. Repeated ratings were resolved by retaining the last valid record for each annotator-candidate-question combination.

The secondary analysis uses rubric-weighted dimension scores, with weights of 0.60/0.40 for the two L1 dimensions. Weights for L2 are 0.40/0.25/0.20/0.15, and those for L3 are 0.40/0.30/0.30, in the dimension order of Table 22. These weights are unequal, and weighted scores are retained without rounding. Valid dimension records numbered 890, 900, and 900 for Humans 1–3, and 898 for the judge. Their common intersection contained 888 cells from 297 questions and 249 papers.

All Human-validation confidence intervals use 2,000 source-paper cluster bootstrap replicates, with percentile endpoints at 2.5% and 97.5%. Each replicate resamples source papers and retains their associated questions, candidate responses, and paired ratings. Point estimates weight response cells equally; paper clustering accounts for their shared sources in interval estimation. Spearman correlations retain score ties, and pairwise raw-score QWK uses the five integer categories with quadratic weights $( ( a - b ) / 4 ) ^ { 2 }$ . The human panel mean remains unrounded and is summarized using correlation, bias, and mean absolute error (MAE).

Candidate ranking and overall agreement. All three annotators and the production judge ranked GPT-5.6-sol above Qwen3-VL-8B, followed by LLaVA-Med-Mistral-7B (Table 26). A fixed-300 sensitivity assigning missing scores the scale minimum of 1 preserved this ordering. Absolute score levels differed across annotators, as shown by their candidate means.

Table 26: Observed mean raw overall scores for the three response candidates.
<table><tr><td>Scorer</td><td>GPT-5.6-sol</td><td>Qwen3-VL-8B</td><td>LLaVA-Med-Mistral-7B</td></tr><tr><td>Human 1</td><td>3.163</td><td>2.186</td><td>1.727</td></tr><tr><td>Human 2</td><td>4.127</td><td>2.657</td><td>1.917</td></tr><tr><td>Human 3</td><td>3.933</td><td>2.447</td><td>1.797</td></tr><tr><td>Production judge</td><td>3.354</td><td>2.257</td><td>1.790</td></tr></table>

The panel mean correlated with the production judge at Spearman $\rho = 0 . 8 7 7 \left[ 0 . 8 5 5 , 0 . 8 9 7 \right]$ across 884 raw-score cells. Pearson correlation was r = 0.880 [0.857, 0.899], with MAE 0.464 and a human-minus-judge bias of +0.188. Within-candidate Spearman correlations ranged from 0.778 to 0.865 (Table 27). The rubric-weighted analysis gave $\rho = 0 . 8 9 3 \ [ 0 . 8 7 3 , 0 . 9 0 9 ]$ across 888 cells. Equal dimension weighting yielded a similar correlation of 0.892 [0.872, 0.909] on the same cells.

Human-human agreement and L3 diagnostics. Pairwise human raw-score correlations ranged from 0.820 to 0.953, with QWK from 0.738 to 0.944 (Table 28). Human–judge correlations ranged from 0.832 to 0.854, and QWK ranged from 0.779 to 0.851. These comparisons quantify both agreement among annotators and agreement with the production judge.

Table 27: Production-judge agreement with the mean of the three human ratings. The unit is a candidate-question response cell. Rows use raw overall scores except the final weighted-score sensitivity. Bias is human minus judge; brackets denote 95% source-paper cluster bootstrap confidence intervals.
<table><tr><td>Scope</td><td>Paired cells</td><td>Spearman [95% CI]</td><td>MAE</td><td>Bias</td></tr><tr><td>All, raw overall</td><td>884</td><td>0.877 [0.855, 0.897]</td><td>0.464</td><td>+0.188</td></tr><tr><td>L1</td><td>377</td><td>0.909 [0.887, 0.926]</td><td>0.389</td><td>+0.193</td></tr><tr><td>L2</td><td>257</td><td>0.875 [0.837, 0.905]</td><td>0.450</td><td>+0.014</td></tr><tr><td>L3</td><td>250</td><td>0.709 [0.637, 0.774]</td><td>0.592</td><td>+0.360</td></tr><tr><td>GPT-5.6-sol</td><td>291</td><td>0.840 [0.797, 0.874]</td><td>0.608</td><td>+0.379</td></tr><tr><td>Qwen3-VL-8B</td><td>296</td><td>0.865 [0.823, 0.896]</td><td>0.430</td><td>+0.169</td></tr><tr><td>LLaVA-Med-Mistral-7B</td><td>297</td><td>0.778 [0.721, 0.829]</td><td>0.357</td><td>+0.020</td></tr><tr><td>All, rubric-weighted</td><td>888</td><td>0.893 [0.873, 0.909]</td><td>0.487</td><td>+0.293</td></tr></table>

Table 28: Pairwise agreement on raw overall scores, using each pair's available response cells. QWK denotes quadratic-weighted Cohen's κ on integer 1–5 ratings. Brackets denote 95% source-paper cluster bootstrap confidence intervals.
<table><tr><td>Scorer pair</td><td>Paired cells</td><td>Spearman [95% CI]</td><td>QWK [95% CI]</td></tr><tr><td>Human 1–Human 2</td><td>887</td><td>0.820 [0.788, 0.849]</td><td>0.738 [0.694, 0.776]</td></tr><tr><td>Human 1–Human 3</td><td>887</td><td>0.835 [0.807, 0.859]</td><td>0.800 [0.766, 0.829]</td></tr><tr><td>Human 2–Human 3</td><td>900</td><td>0.953 [0.942, 0.962]</td><td>0.944 [0.930, 0.956]</td></tr><tr><td>Human 1-judge</td><td>884</td><td>0.838 [0.810, 0.863]</td><td>0.851 [0.821, 0.876]</td></tr><tr><td>Human 2–judge</td><td>897</td><td>0.832 [0.800, 0.860]</td><td>0.779 [0.740, 0.815]</td></tr><tr><td>Human 3-judge</td><td>897</td><td>0.854 [0.830, 0.876]</td><td>0.837 [0.806, 0.863]</td></tr></table>

L3 showed lower overall agreement than L1 or L2, with panel-judge $\rho = 0 . 7 0 9$ on 250 raw-score cells. The recorded L3 rubric includes competing explanations, reasons for accepting or rejecting alternatives, and evidential limitations in its checklists. Its dimension scores therefore provide source-referenced critique and argumentation diagnostics. The validation supports overall score agreement and candidate ordering under this rubric, while fine-grained L3 dimension agreement is lower.

## A.11 PROPOSAL-ORIENTED RUBRIC SENSITIVITY PILOT

The L3 rubric mismatch noted above motivates a direct test of how sensitive L3 scores are to the rubric itself. We froze 24 L3 items from the stratified 300-item sample's 86 L3 items, selected by a fixed hash order over item IDs (24 source papers; selection fixed before inspecting any scores), and archived all three candidates' open responses, yielding 72 fixed response cells. Two additional cell biology PhD annotators, denoted Human 4 and Human 5 and independent of Human 1–3, rescored every cell under three frozen conditions: (i) image and question only, under a proposal-oriented rubric; (ii) full source evidence (caption, context, and reference answer), under the same rubric; and (iii) full source evidence, under a reimplementation of the critique-oriented checklist of Table 22. The proposal rubric's core dimensions are observational fidelity, scientific plausibility, and falsifiable specificity, each an integer from 1 to 5; a discriminative-test dimension is scored only when the item requests it, and unrequested argumentation does not enter the core score. Scoring was blind to candidate identity, options, and answer keys, at temperature 0.

Agreement and ranking. On identical full-source inputs, cross-annotator agreement on overall scores is QWK 0.767 [0.646, 0.868] under the proposal rubric, versus QWK 0.466 [0.332, 0.585] under the critique checklist; on the 56 cells valid under both, the paired values are 0.767 versus 0.459, a difference of +0.308 [0.130, 0.468] under 2,000 source-paper cluster bootstrap resamples. The candidate ranking GPT-5.6-sol > Qwen3-VL-8B > LLaVA-Med-Mistral-7B is preserved on the 17-item intersection valid for every annotator and condition, and on 19 items under a parser-relaxed sensitivity analysis.

Score levels and input effect. Proposal-rubric totals exceed critique totals for the strongest candidate (+0.800 [0.600, 0.950] for Human 4 and +1.750 [1.417, 2.042] for Human 5 on paired cells), so historical L3 totals are rubric-sensitive rather than a pure measure of hypothesis-proposal ability. Under the proposal rubric with full evidence, Human 4 and Human 5 rate GPT-5.6-sol at 4.750 and 4.833 overall; among core dimensions, falsifiable specificity is the weakest for the open-weight candidates (2.611/3.292 for Qwen3-VL-8B and 1.611/1.625 for LLaVA-Med-Mistral-7B under Human 4/Human 5). Adding source evidence under the fixed proposal rubric changes totals by +0.074 [-0.060, 0.196] for Human 4 and -0.028 [-0.167, 0.125] for Human 5, so the source-evidence fields have only a small average effect once the rubric is fixed.

Stress controls. On four frozen base items, concise hypotheses and non-reference alternative hypotheses score 4.25–5.00 overall, pure rhetorical expansion adds at most +0.25, and injected observation errors reduce totals by 1.00 for Human 4 and 2.00 for Human 5. These four-item controls are diagnostic only, but they are consistent with the proposal rubric rewarding valid alternatives and penalizing observational errors without rewarding verbosity.

We read this pilot as measurement-sensitivity evidence: it supports stable candidate rankings and improved cross-annotator agreement under the proposal-oriented rubric, while establishing neither a validated hypothesis-ability scale nor full-benchmark rescoring, which remain future work.

## A.12 RELEASE-LEVEL TRAIN-BENCHMARK OVERLAP AUDIT

This appendix audits the released files themselves—the 548,450-row OMNIVCTRAIN parquet against the 6,077-item OMNIVCBENCH release—rather than inferring isolation from the construction scripts.

OMNIVCTRAIN covers 33,802 papers and the benchmark 1,080. Four cross-set checks— normalized DOI/paper-ID exact match, normalized title exact match, and title character-TFIDF cosine at thresholds .90 and .95—return zero candidates (Table 29). The largest nearest-title cosine observed is 0.8896, below the .90 threshold; that nearest pair (a DAF-16/FOXO record) was manually reviewed and involves a different DOI and a different study.

Table 29: Source-level cross-set overlap checks between the released OMNIVCTRAIN (33,802 papers) and OMNIVCBENCH (1,080 papers).
<table><tr><td>Check</td><td>Cross-set candidates</td><td>Confirmed duplicates</td></tr><tr><td>Normalized DOI / paper ID (exact)</td><td>0</td><td>0</td></tr><tr><td>Normalized title (exact)</td><td>0</td><td>0</td></tr><tr><td>Title char-TFIDF cosine ≥ .90</td><td>0</td><td>0</td></tr><tr><td>Title char-TFIDF cosine ≥ .95</td><td>0</td><td>0</td></tr></table>

Text overlap is computed after Unicode normalization, lowercasing, and word tokenization; candidate pairs are retrieved by bottom-12 signatures (word trigrams for questions, 5-grams for captions) and accepted at Jaccard ≥ .80 or containment $\geq . 9 0$ . Table 30 reports the counts: questions (548,390 unique train / 6,077 bench), captions (96,007 / 2,625), and answers (exact match only; 544,652 / 6,077); exact and near pairs are all zero. Signature retrieval is not a semantic-paraphrase detector, so Table 31 reports detection rates on synthetic perturbations.

Table 30: Text-level cross-set overlap between the released corpora. Near pairs require Jaccard ≥ .80 or containment ≥ .90 under bottom-12 signature retrieval.
<table><tr><td>Field</td><td>Train uniques</td><td>Bench uniques</td><td>Exact pairs</td><td>Near pairs</td></tr><tr><td>Question</td><td>548,390</td><td>6,077</td><td>0</td><td>0</td></tr><tr><td>Caption</td><td>96,007</td><td>2,625</td><td>0</td><td>0</td></tr><tr><td>Answer (exact only)</td><td>544,652</td><td>6,077</td><td>0</td><td></td></tr></table>

Image overlap combines SHA-256 with perceptual hashes (pHash and dHash on EXIF-corrected grayscale images), accepting candidates at Hamming distance $\leq 6 .$ We hash 327,601 train and 4,428 benchmark images (Table 32): zero exact pairs; 22,768 perceptual candidate pairs involving 606 benchmark images under a single hash; and 45 pairs involving 8 benchmark images passing both hashes at distance $\leq 6 ,$ all of which were manually reviewed without confirming any identical, cropped, or recomposed figure. Perceptual similarity alone cannot establish leakage, since microscopy images and charts share generic layouts; Table 33 reports detection rates on synthetic image perturbations.

Table 31: Synthetic-text sensitivity of the overlap detector $( n ~ = ~ 5 0 0$ perturbations per row): signature-retrieval recall and acceptance recall (Jaccard ≥ .80 or containment ≥ .90).
<table><tr><td>Perturbation</td><td>Retrieval</td><td>Acceptance</td></tr><tr><td>Question: append 10% words</td><td>100%</td><td>100%</td></tr><tr><td>Question: delete one word</td><td>100%</td><td>98.2%</td></tr><tr><td>Question: substitute one word</td><td>100%</td><td>87.8%</td></tr><tr><td>Caption: append 10% words</td><td>100%</td><td>100%</td></tr><tr><td>Caption: delete one word</td><td>100%</td><td>100%</td></tr><tr><td>Caption: substitute one word</td><td>100%</td><td>100%</td></tr></table>

Table 32: Image-level cross-set overlap checks (327,601 train and 4,428 benchmark images hashed).
<table><tr><td>Check</td><td></td><td>Pairs Bench images involved Confirmed duplicates</td><td></td></tr><tr><td>SHA-256 exact</td><td>0</td><td>0</td><td>0</td></tr><tr><td>Single-hash perceptual candidates</td><td>22,768</td><td>606</td><td></td></tr><tr><td>Dual-hash (pHash and dHash) Hamming ≤ 6</td><td>45</td><td>8</td><td>0</td></tr></table>

Table 33: Synthetic-image sensitivity of the perceptual-hash detector $( n = 3 0 0$ perturbations per row).
<table><tr><td>Perturbation</td><td>Detection rate</td></tr><tr><td>JPEG quality 75</td><td>100%</td></tr><tr><td>Resize to half</td><td>100%</td></tr><tr><td>Brightness ×1.2</td><td>97.7%</td></tr><tr><td>Crop 2%</td><td>95.0%</td></tr><tr><td>Add 2% border</td><td>97.3%</td></tr></table>

Across all four groups of checks, no duplicates were confirmed after manual review; we therefore describe the released corpora as having no confirmed overlap rather than claiming absolute leakfreeness.

## A.13 CLUSTER-ROBUST SCORES AND PAPER-WEIGHTED STATISTICS

This appendix recomputes the main scores and cross-track correlations on the full benchmark (6,077 items, 1,080 papers) from item-level files without rounding, covering the sixteen MCQ configurations and the eleven open-track models. Confidence intervals are source-paper cluster bootstrap; paper-weighted statistics first average within each source paper and then weight papers equally. Table 34 reports both weightings for every model; the item-level means agree with the rounded values of Table 2 (main text).

Using the original 0/1 predictions on all 6,077 items, exact overall MCQ accuracies are GPT-5.6-sol 55.09%, Grok-4.6 50.30%, Claude-Sonnet-4-5 49.37%, GPT-5.6-luna 43.74%, Qwen3-VL-8B-SFT 34.28%, and base Qwen3-VL-8B 33.82%. Across the eleven models shared by both tracks, modellevel Pearson correlation is 0.94 (permutation $p = 6 . 0 \times 1 0 ^ { - 5 } )$ and Spearman correlation is 0.96 $( p = 2 . 0 \times 1 0 ^ { - 5 } )$

Correlation robustness. Across the eleven shared models, unrounded correlations are Pearson $r ~ = ~ 0 . 9 3 8$ (95% model-bootstrap CI [0.843, 0.993]) and Spearman $\rho \ = \ 0 . 9 6 4 \ ( [ 0 . 7 6 7 , 1 . 0 0 0 ] )$ Open-weight models correlate strongly $( r = . 9 5 5 , n = 7 ) \colon$ the four proprietary systems have a weaker correlation $( r = . 2 0 8 )$ . The pooled statistic therefore includes between-group separation. Leave-one-family-out Pearson correlations remain 0.922–0.971 on strict common items with paper weighting (Table 36). Human validation checks the raw overall score directly on the stratified 300- item sample (Appendix A.10).

Table 34: Item-level and paper-weighted scores with source-paper cluster bootstrap 95% CIs, computed from unrounded item-level files (6,077 items, 1,080 papers). Top block: MCQ accuracy (%), sixteen configurations; bottom block: AIVC-Judge (1–5), eleven models.
<table><tr><td>Model</td><td>Item-level [95% CI]</td><td>Paper-weighted [95% CI]</td><td></td></tr><tr><td>MCQ accuracy (%)</td><td></td><td></td><td></td></tr><tr><td>GPT-5.6-sol</td><td>55.093 [53.768, 56.444]</td><td></td><td>56.033 [54.267, 57.794]</td></tr><tr><td>Grok-4.6</td><td>50.304 [48.862, 51.729]</td><td></td><td>51.492 [49.738, 53.286]</td></tr><tr><td>Claude-Sonnet-4-5</td><td>49.366 [47.937, 50.768]</td><td></td><td>50.123 [48.313, 51.889]</td></tr><tr><td>GPT-5.6-luna</td><td>43.739 [42.415, 45.135]</td><td></td><td>44.478 [42.678, 46.314]</td></tr><tr><td>Qwen3-VL-8B-SFT</td><td>34.277 [33.018, 35.566]</td><td></td><td>34.111 [32.444, 35.821]</td></tr><tr><td>Qwen3-VL-8B</td><td>33.816 [32.599, 35.047]</td><td></td><td>34.233 [32.558, 35.937]</td></tr><tr><td>Qwen3-VL-4B</td><td>30.476 [29.248, 31.725]</td><td></td><td>31.259 [29.595, 32.902]</td></tr><tr><td>InternVL3.5-8B</td><td>30.278 [29.001, 31.550]</td><td></td><td>30.557 [28.982, 32.177]</td></tr><tr><td>InternVL3.5-4B</td><td>28.879 [27.617, 30.157]</td><td></td><td>29.525 [27.870, 31.198]</td></tr><tr><td>Qwen2.5-VL-7B</td><td>26.625 [25.455, 27.830]</td><td></td><td>26.971 [25.440, 28.560]</td></tr><tr><td>LLaVA-OneVision-7B</td><td>25.539 [24.413, 26.683]</td><td></td><td>25.564 [24.078, 27.074]</td></tr><tr><td>Qwen2.5-VL-3B</td><td>22.495 [21.433, 23.576]</td><td></td><td>22.583 [21.164, 24.038]</td></tr><tr><td>Qwen3-VL-2B</td><td>21.688 [20.603, 22.761]</td><td></td><td>22.122 [20.694, 23.572]</td></tr><tr><td>InternVL3.5-2B</td><td>20.866 [19.794, 21.964]</td><td></td><td>21.308 [19.911, 22.781]</td></tr><tr><td>SmolVLM2-2.2B</td><td>17.311 [16.407, 18.244]</td><td></td><td>17.596 [16.361, 18.898]</td></tr><tr><td>LLaVA-Med-Mistral-7B</td><td>15.863 [14.961, 16.807]</td><td></td><td>16.538 [15.243, 17.912]</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td>AIVC-Judge (1–5) GPT-5.6-sol</td><td>3.285 [3.243, 3.325]</td><td>3.286 [3.237, 3.335]</td><td></td></tr><tr><td>GPT-5.6-luna</td><td>3.161 [3.121, 3.201]</td><td></td><td>3.140 [3.090, 3.190]</td></tr><tr><td>Grok-4.6</td><td>3.100 [3.059, 3.142]</td><td></td><td>3.111 [3.060, 3.163]</td></tr><tr><td>Claude-Sonnet-4-5</td><td>2.681 [2.638, 2.724]</td><td></td><td>2.702 [2.650, 2.757]</td></tr><tr><td>Qwen3-VL-8B-SFT</td><td>2.379 [2.341, 2.416]</td><td></td><td>2.387 [2.340, 2.435]</td></tr><tr><td>Qwen3-VL-8B</td><td></td><td></td><td>2.342 [2.294, 2.390]</td></tr><tr><td>InternVL3.5-8B</td><td>2.333 [2.297, 2.371]</td><td></td><td>2.316 [2.268, 2.365]</td></tr><tr><td></td><td>2.320 [2.283, 2.356]</td><td></td><td></td></tr><tr><td>Qwen3-VL-4B</td><td>2.226 [2.189, 2.262]</td><td></td><td>2.206 [2.162, 2.252]</td></tr><tr><td>Qwen2.5-VL-7B</td><td>2.108 [2.076, 2.141]</td><td></td><td>2.077 [2.037, 2.118]</td></tr><tr><td>LLaVA-OneVision-7B</td><td>2.003 [1.970, 2.036]</td><td></td><td>1.984 [1.943, 2.026]</td></tr><tr><td>LLaVA-Med-Mistral-7B</td><td>1.861 [1.830, 1.893]</td><td></td><td>1.861 [1.820, 1.903]</td></tr></table>

Table 35 reports the MCQ-open-response model-level correlation under three item sets and both weightings. The strict common set contains 5,654 items from 1,058 papers. The paper-cluster bootstrap quantifies sensitivity to paper sampling; it does not increase the number of independent model points, which remains eleven.

Table 35: MCQ-open-response model-level correlation (n = 11 models) by item set and weighting. Bracketed values are source-paper cluster bootstrap 95% CIs; p values are permutation-based.
<table><tr><td>Scope</td><td>Pearson</td><td>Spearman</td></tr><tr><td>Published full-track, item</td><td>0.938  $( p { = } 6 \times 1 0 ^ { - 5 } )$ </td><td>0.964  $( p { = } 2 { \times } 1 0 ^ { - 5 } )$ </td></tr><tr><td>Published full-track, paper</td><td>0.945</td><td>0.955</td></tr><tr><td>Model-specific matched, item</td><td>0.936</td><td>0.964</td></tr><tr><td>Model-specific matched, paper 0.943</td><td></td><td>0.955</td></tr><tr><td>Strict common, item</td><td></td><td>0.935 [0.919, 0.946] (p=8× 10−5) 0.964 [0.927, 0.973] (p=3× 10−5)</td></tr><tr><td>Strict common, paper</td><td>0.943 [0.923, 0.956]</td><td>0.955 [0.927, 0.973]</td></tr></table>

Table 36 reports leave-one-family-out correlations on the strict common set with paper weighting; the Pearson range is 0.922–0.971 and the Spearman range 0.917–0.967. Family definitions were frozen before this analysis, and removing single-model families is a leverage check rather than a subgroup claim.

## A.14 PROMPTS AND RUN SETTINGS

AIVC-Judge uses temperature 0 for pointwise scoring. Its inputs are the question, candidate response, original figure image, caption, surrounding context, and reference answer. Candidate identity is omitted (Table 23). Answering models receive the original image and question, plus six options for MCQ. Proprietary systems use the API routes listed in Table 4; open-weight checkpoints run locally. The construction-audit protocol and agreement statistics appear in Appendix A.3, and human answering and scoring protocols appear in Appendices A.9 and A.10. Prompt templates and evaluation code are provided through https : //anonymous. 4open. science/r/ OmniVCBench.

Table 36: Leave-one-family-out MCQ-open-response correlation (strict common items, paperweighted; n = remaining models).
<table><tr><td>Excluded family</td><td>n</td><td>Pearson</td><td>Spearman</td></tr><tr><td>OpenAI GPT</td><td>9</td><td>0.956</td><td>0.967</td></tr><tr><td>xAI Grok</td><td>10</td><td>0.932</td><td>0.964</td></tr><tr><td>Anthropic Claude</td><td>10</td><td>0.971</td><td>0.964</td></tr><tr><td>Qwen3-VL (incl. SFT)</td><td>8</td><td>0.942</td><td>0.929</td></tr><tr><td>Qwen2.5-VL</td><td>10</td><td>0.939</td><td>0.939</td></tr><tr><td>InternVL</td><td>10</td><td>0.943</td><td>0.952</td></tr><tr><td>LLaVA</td><td>9</td><td>0.922</td><td>0.917</td></tr></table>

## B OMNIVCTRAIN ADAPTATION DETAILS

This appendix details the adaptation experiments behind Section 4.3. Holm-corrected values retain the original adaptation comparison family, including configurations omitted from the main-table presentation. The rank-32 configuration gains 1.97 percentage points (95% paper-cluster bootstrap CI [0.93, 3.02]; corrected p = .006), and multimodal RAG gains 1.60 points (CI [0.71, 2.48]; corrected $p = . 0 1 0 )$ . The rank-16 SFT change is 0.46 points (CI [–0.62, 1.54]). The principal results on the MCQ track are reported in Table 3 (main text); open-response scores in Table 2 cover only the rank-16 SFT configuration, and open-track scoring of the strongest MCQ configurations remains future work. The subsections below describe the supervised fine-tuning and retrieval-augmented generation protocols and report the corresponding ablations.

## B.1 SUPERVISED FINE-TUNING

We format OMNIVCTRAIN as multimodal instruction triples and adapt Qwen3-VL-8B with LLaMA-Factory. Table 37 compares two LoRA settings: rank 16 on the full corpus (548,016 loaded examples) and rank 32 on a 100,000-example subset, both for two epochs. The two settings differ in rank, data size, and batch size jointly, so this is a configuration comparison rather than a rankonly ablation. An independent audit of the archived training logs and inference scripts recovers the shared configuration (learning rate $1 0 ^ { - 4 }$ , cosine schedule, warmup ratio 0.03, bf16, sequence cutoff 2,048, seed 42, frozen visual tower and projector, LoRA α twice the rank, target modules down/gate/k/o/q/up/v\_proj, effective batch sizes 64 and 56; PEFT 0.18.1, Transformers 5.8.0, Torch 2.8.0) and reproduces all three adaptation accuracies of Table 3 exactly from the archived predictions (34.2768%, 35.7907%, and 35.4122% before rounding). Rank 32 yields the larger observed improvement, 1.97 percentage points over the 33.82% base model. Source-paper-clustered comparisons and corrected significance are reported in Section 4.3.

Table 37: Principal SFT configurations for Qwen3-VL-8B. Raw paper-block permutation tests compare each setting with the 33.82% base model on all 6,077 items; Holm-corrected values over the original adaptation comparison family are quoted in the main text.
<table><tr><td>Setting</td><td>Data</td><td>Epochs</td><td>Rank</td><td>Avg</td><td>∆</td><td>p</td></tr><tr><td>Full corpus</td><td>548k</td><td>2</td><td>16</td><td>34.28</td><td>+0.46</td><td>0.42</td></tr><tr><td>Rank 32, 100k subset</td><td>100k</td><td>2</td><td>32</td><td>35.79</td><td>+1.97</td><td> $3 . 2 \times 1 0 ^ { - 4 }$ </td></tr></table>

## B.2 RETRIEVAL-AUGMENTED GENERATION

We construct a 548,450-entry index with Qwen3-VL-Embedding-8B (Li et al., 2026b). Each entry jointly encodes its image and question into a 4,096-dimensional, L2-normalized vector. At test time, cosine similarity retrieves k = 3 QA demonstrations after excluding entries from the same source article and entries with the same figure basename. Retrieved questions and answers are prepended as text demonstrations; the test image remains the visual input to the evaluated MLLM. Retrieved images are discarded, so this baseline evaluates text-demonstration prompting conditioned on multimodal retrieval rather than interleaved multimodal in-context learning, which remains an unexplored adaptation route.

Table 38 compares joint image-question retrieval with image-only and text-only controls at $k = 3 .$ Joint retrieval increases accuracy from 33.82% to 35.41%, while the unimodal controls score 33.62% and 33.40%, respectively. Table 39 reports joint-retrieval gains for three base models, ranging from 0.76 to 1.60 percentage points.

Table 38: Retrieval-signal ablation for Qwen3-VL-8B with $k = 3 .$ The base accuracy is 33.82%. p: raw paper-block permutation test vs. base.
<table><tr><td>Retrieval signal</td><td>L1</td><td>L2</td><td>L3</td><td> $\operatorname { A v g }$ </td><td> $\Delta$ </td><td>p</td></tr><tr><td>Joint image-question</td><td>35.6</td><td>39.3</td><td>31.2</td><td>35.41</td><td>+1.60</td><td> $5 . 6 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Image only</td><td>34.1</td><td>37.5</td><td>28.9</td><td>33.62</td><td>-0.20</td><td>.64</td></tr><tr><td>Text only</td><td>34.1</td><td>37.3</td><td>28.5</td><td>33.40</td><td>-0.41</td><td>.31</td></tr></table>

Table 39: Joint multimodal RAG across base models. Improvements are positive across all three models but vary in statistical strength; the InternVL3.5-8B change does not reach significance. p: raw paper-block permutation test vs. the corresponding base model.
<table><tr><td>Model</td><td>Base</td><td>+RAG</td><td>∆</td><td>p</td></tr><tr><td>Qwen3-VL-8B</td><td>33.82</td><td>35.41</td><td>+1.60</td><td> $5 . 6 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Qwen3-VL-2B</td><td>21.69</td><td>23.19</td><td>+1.50</td><td> $2 . 9 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>InternVL3.5-8B</td><td>30.28</td><td>31.04</td><td>+0.76</td><td>.10</td></tr></table>

## C BIOLOGICAL EVIDENCE ACQUISITION AND PREDICTION REVISION

This appendix documents the four executable pilots behind Section 4.5. Their purpose is to test whether the interpretation component measured by OMNIVCBENCH can acquire evidence and revise predictions inside the refinement loop of Figure 1, over real biological outputs rather than curated literature figures. All four pilots use GPT-5.6-sol via the route listed in Table 4, with fixed cases chosen before any model call; every model response is archived, and all scores are deterministic recomputations from the archived responses.

## C.1 TASK STRUCTURE AND EVIDENCE SETTINGS

The four pilots share one structure (Figure 5). The model first predicts a hidden biological output from initial evidence and explains its prediction. A self-review pass then re-examines the same evidence without new information. Independently of the self-review, the model selects one additional measurement to acquire from reserved cells or images; the reserved data are revealed, the model revises its prediction, and the revision is rescored against the hidden ground truth. Each case therefore contributes three scored answers—initial, self-reviewed, and post-evidence—and 36 cases yield the 108 archived model calls. Three pilots update a response-calibration layer on top of the simulator's outputs, and the revision propagates to unqueried readouts through that layer; the microscopy pilot updates class probabilities and the accompanying explanation. Simulator weights are never retrained within a run, and no new wet-lab measurement is performed. Table 40 summarizes the four evidence settings.

## C.2 RESULTS

Figure 6 and Table 41 report per-pilot outcomes. Initial accuracy is below 100% in all four pilots (66.7%–93.3%). Acquiring the chosen measurement corrects predictions in the three numerical pilots, with nine wrong-to-right and zero right-to-wrong transitions in total; self-review alone improves no pilot and degrades two. The microscopy pilot shows no classification gain when reading real fluorescence pixels (Figure 7). Two qualifications accompany these numbers. First, corrections are local: unqueried-readout accuracy does not improve after the calibration update, so a local fix must not be read as a general improvement of the remaining outputs. Second, the gene-pair pilot is class-skewed—38 of its 48 readouts have near-zero ground truth, so a constant near-zero baseline reaches 79.2%, above the model's initial 77.1%; macro-F1 and continuous errors are archived in the run records. Case-bootstrap intervals in the run records describe small-sample replay uncertainty not independent biological replicates.

Table 40: The four evidence-acquisition pilots. Each case admits exactly one additional measurement; readouts are the scored prediction fields. The response adapter is the calibration layer updated by the acquired measurement.
<table><tr><td>Pilot</td><td>Simulator / data</td><td>Prediction target</td><td>Evidence choice</td><td>Cases / readouts</td></tr><tr><td>Gene interactions</td><td>GEARS (Roohani et al., 2023); Norman K562 CRISPRa (Norman et al., 2019)</td><td>non-additive double-gene response AB − A − B per program</td><td>one cellular program in reserved AB cells</td><td>12 /48</td></tr><tr><td>Drug combinations</td><td>CPA trained locally (Lotfollahi et al., 2023); ComboSciPlex A549</td><td>program mean change under held- out combinations</td><td>one program in 48 reserved cells</td><td>10 / 30</td></tr><tr><td>Real microscopy</td><td>BBBC021 (Caie et al., 2010; Ljosa et al., 2012); MCF-7, 5 MoA classes</td><td>phenotype-supported class</td><td>mechanism F-actin channel or a second field of the same well</td><td>9/9</td></tr><tr><td>Signaling</td><td>discrete Bayesian network; Sachs discretized cells (Sachs et al., 2005)</td><td>high-state fraction change under each intervention</td><td>one non-target readout in 96 5 / 30 reserved cells</td><td></td></tr></table>

![](images/7339183a0fd936f9813b63da730d52fe8cd94a962205937b35bfd32660bec3a6.jpg)  
Figure 5: Shared pilot structure. Each pilot runs predict, explain and select, measure, then update and test. Three pilots revise predictions through a response adapter after active evidence acquisition; the microscopy pilot revises its interpretation. These are one-step replay loops: simulator weights and free-text biological mechanisms are not validated or updated.

Table 41: Per-pilot accuracy (%) before and after acquiring one chosen measurement. Corrections count wrong-to-right transitions against zero right-to-wrong transitions in every pilot.
<table><tr><td>Pilot</td><td>Readouts</td><td>Initial</td><td>Self-review</td><td>After evidence</td><td>Corrections</td></tr><tr><td>Gene interactions</td><td>48</td><td>77.1</td><td>77.1</td><td>89.6</td><td>+6/-0</td></tr><tr><td>Drug combinations</td><td>30</td><td>93.3</td><td>80.0</td><td>100.0</td><td>+2/-0</td></tr><tr><td>Real microscopy</td><td>9</td><td>66.7</td><td>66.7</td><td>66.7</td><td>0/-0</td></tr><tr><td>Signaling</td><td>30</td><td>66.7</td><td>60.0</td><td>70.0</td><td>+1/-0</td></tr></table>

## C.3 SCOPE OF THE PILOTS

These are one-step replay loops: simulator weights are frozen, no new wet-lab measurement is performed, and corrections update a response-calibration layer rather than the simulator. Case counts are small (36 cases, 108 archived model calls), drug-combination accuracy saturates after evidence, and the microscopy pilot has no simulator at all. The pilots therefore test evidence acquisition and prediction revision on real biological outputs. They do not validate a complete virtual-cell discovery loop, do not establish that OMNIVCBENCH scores transfer to loop performance, and do not isolate the contribution of multimodal inputs, since numerical tasks also expose numeric tables. Freetext mechanisms are retained as explanation traces without an independent gold standard, so no mechanism-discovery claim is made.

![](images/d69b23e554d208bf2dd769a7fcc4cf39be890c0b4f8c1d5a2b087624f6600701.jpg)  
Figure 6: Per-pilot outcomes. Correct-outcome rate under the initial prediction, self-review with the same evidence, and revision after one chosen measurement, for each of the four pilots. Case counts and readout counts differ across pilots; results are descriptive and support no pooled accuracy or causal-discovery claim.

## D AIVC-JUDGE DIMENSION-WISE RESULTS

Table 42 reports dimension scores alongside the raw overall metric used in the main results. Overall scores are holistic judgments, so their means need not equal averages of dimension means. Proprietary models lead on every dimension in this evaluated pool. The production L3 profile separates judgment correctness from source-referenced argumentation and evidence weighing. For GPT-5.6- sol, these scores are 4.04, 1.92, and 1.67, respectively. The human audit shows weaker agreement on the latter two dimensions than on overall scores (Appendix A.10). We therefore use them to describe behavior under the critique-oriented rubric rather than as a validated scale of proposal quality. Cases G-I (Appendix F) separately examine observational fidelity and the scientific defensibility of proposed hypotheses. For Qwen3-VL-8B, SFT changes the L3 dimensions from 2.91/1.25/1.20 to 3.01/1.44/1.33.

Table 42: AIVC-Judge dimension scores (1–5; higher is better) by interpretation level. L1: C = correctness, ES = evidence support. L2: F = faithfulness, CC = causal completeness, MG = mechanistic granularity, EC = evidence consistency. L3: J = judgment correctness, A = argumentation quality, EW = evidence weighing. † denotes SFT on OMNIVCTRAIN.
<table><tr><td rowspan="2">Model</td><td colspan="2">L1</td><td colspan="4">L2</td><td colspan="3">L3</td></tr><tr><td>C</td><td>ES</td><td>F</td><td>CC</td><td>MG</td><td>EC</td><td>J</td><td>A</td><td>EW</td></tr><tr><td>GPT-5.6-sol</td><td>3.35</td><td>3.35</td><td>3.88</td><td>3.72</td><td>3.63</td><td>3.85</td><td>4.04</td><td>1.92</td><td>1.67</td></tr><tr><td>GPT-5.6-luna</td><td></td><td></td><td></td><td></td><td></td><td>3.23 3.23 3.703.493.46 3.663.94 1.92 1.67</td><td></td><td></td><td></td></tr><tr><td>Grok-4.6</td><td>3.23</td><td>3.25</td><td>3.64 3.38</td><td></td><td></td><td>3.363.62</td><td></td><td>3.87 1.71 1.53</td><td></td></tr><tr><td>Claude-Sonnet-4-5</td><td>2.52 2.55</td><td></td><td>3.07</td><td>2.99</td><td>2.97</td><td>3.10</td><td>3.64</td><td></td><td>1.991.73</td></tr><tr><td>Qwen3-VL-8B-SFT†</td><td>2.50 2.55 2.71 2.25</td><td></td><td></td><td></td><td></td><td>52.34 2.50 3.011</td><td></td><td></td><td>1.44 1.33</td></tr><tr><td>Qwen3-VL-8B</td><td></td><td></td><td></td><td></td><td></td><td>2.45 2.502.892.302.32 2.68</td><td>2.91</td><td></td><td>1.25 1.20</td></tr><tr><td>InternVL3.5-8B</td><td>2.66 2.69 2.92 2.03</td><td></td><td></td><td></td><td>1.97</td><td>2.59</td><td></td><td></td><td>2.66 1.15 1.12</td></tr><tr><td>Qwen3-VL-4B</td><td></td><td></td><td></td><td></td><td></td><td>2.33 2.38 2.77 2.11 2.13 2.51</td><td></td><td></td><td>2.791.271.23</td></tr><tr><td>Qwen2.5-VL-7B</td><td>2.23 2.27</td><td></td><td></td><td>2.45 1.99</td><td>2.01</td><td>2.29</td><td>2.57</td><td></td><td>1.25 1.19</td></tr><tr><td>LLaVA-OneVision-7B</td><td></td><td></td><td></td><td></td><td></td><td>2.302.35 2.84 1.53 1.46 2.02</td><td></td><td></td><td>2.301.091.07</td></tr><tr><td>LLaVA-Med-Mistral-7B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>3 2.13 2.15 2.33 1.47 1.43 1.80 2.08 1.16 1.12</td></tr></table>

## E LIMITATIONS AND OUTLOOK

Scope of inference. OMNIVCBENCH evaluates interpretation of closed, literature-derived evidence. Its 1,080 source papers were published in Nature Communications during 2011-2017, and mouse and human studies, microscopy, and immunoblot assays dominate the corpus (Tables 12 and 16). This narrow and dated source distribution is the benchmark's largest limitation, and the resulting comparisons apply to this source and assay distribution. Several considerations nonetheless qualify its severity. The corpus was filtered by MLLM screening followed by manual review toward topics close to virtual-cell construction—single-cell and spatial omics, perturbation atlases, and CRISPR screens——so the evaluated evidence centers on the experimental techniques that AIVC interpretation must handle. Older publication years do not make the underlying conclusions less important: these are peer-reviewed studies whose figures remain valid scientific evidence, and the per-year analysis reports no monotone accuracy trend across 2011-2017 for any evaluated model (Table 15), providing no evidence that item difficulty drifts with source age within this range. Even where models may have encountered these papers during pretraining, performance remains far from saturated on both tracks (the strongest model reaches 55.1% MCQ accuracy and 3.28/5 under AIVC-Judge), so potential exposure alone does not resolve the tasks; whether memorization assists a subset of items nonetheless remains unquantified. Source-level filtering and the release-level audit found no confirmed overlap within the assembled resources (Appendix A.12); they do not assess pretraining exposure to the published literature. As future work, we plan to extend the source pool to diverse open-access journals from 2018 to the present, for which the construction pipeline and curation criteria transfer directly. Finally, L3 operationalizes bounded hypothesis proposal and assessment. It measures neither experimental execution nor the novelty of a discovery, and MCQ key agreement measures selection among supplied alternatives rather than free hypothesis generation.

![](images/32f395377e17a380c3b5c3bc99ff14499d0550d28db48ca1d971bb45bce040b9.jpg)  
Figure 7: Real microscopy evidence. Reference compounds of the microscopy pilot (BBBC021), shown as DNA, tubulin, and actin channels; test compounds are disjoint from these references. The task reads real fluorescence pixels, not synthetic bar plots.

Text-only shortcuts. The no-image control (Appendix A.6) shows that GPT-5.6-sol retains 34.7% MCQ accuracy without the figure: the question text, the option content, domain priors, and potential memorization of the source literature all contribute substantially to MCQ performance. This text-derived signal is a genuine limitation of the controlled track. The 22.6-point gap to the 57.3% with-image score nonetheless shows that visual evidence carries a large share of the achievable accuracy, and open-response evaluation further constrains purely text-driven answering, since generated explanations are checked against the source evidence by the judge.

Scoring and answer validity. Human scoring provides a check on the production judge under the same reference-conditioned protocol: the panel-mean raw overall score correlates with AIVC-Judge at Spearman $\rho = 0 . 8 7 7$ over 884 paired responses (Appendix A.10). Human scorers applied the same rubric without the original image, so this agreement validates the shared textual evidence and rubric application rather than the judge's image reading; same-input expert scoring remains an open validation step. Agreement is lower on L3 (ρ = 0.709, 250 responses), and the historical argumentation and evidence-weighing dimensions assess critique and justification. These dimensions are diagnostics under that rubric, not a validated scale of hypothesis-generation ability. L3 overall scores can therefore conflate proposal quality with unrequested critique, and they are not directly comparable with L1/L2 totals, which use different dimensions. A rubric that directly rewards proposal quality— observational fidelity, falsifiable specificity, and discriminative test design—has been piloted on a frozen 24-item L3 subset, where it preserves candidate rankings and improves cross-annotator agreement (Appendix A.11); full-benchmark rescoring under it remains future work. The scoring evidence includes caption, context, and reference information beyond the answering model's image and question. Moreover, AIVC-Judge contributes to both distractor filtering and open-response scoring, so cross-track correlation is an internal consistency measure.

System coverage. The evaluated systems are multimodal language models. The benchmark assesses their interpretation of cellular experiments, rather than the state-prediction accuracy of cell foundation models. The open-weight pool spans 2B-8B parameters, so comparisons with the proprietary systems mix model capacity, compute, and inference configuration and should not be read as open- versus closed-identity effects. Connecting cell models to language interfaces defines an extension of this evaluation setting.

Adaptation headroom. OMNIVCTRAIN provides 548k source-aligned examples, yet our adaptation study covers only LoRA fine-tuning and retrieval augmentation on a single 8B backbone, and the observed gains are configuration-dependent rather than a ceiling on the corpus. We currently lack the computational resources to explore stronger recipes—larger backbones, longer training schedules, preference optimization, or agentic training on evidence-grounded tasks. The corpus's potential for improving evidence-grounded interpretation therefore remains largely untapped, and we view it as a resource for the community as much as a baseline for this paper.

## F CASE STUDIES OF REASONING AND SCORING

We examine nine cases selected purposively after inspecting the stratified 300-item sample (Appendix A.6), with three cases per level from nine different source papers. Cases A-G illustrate response-level evidence and reasoning errors; Cases H and I examine L3 items where a defensible open hypothesis coexists with a flawed option selection. Each exhibit reproduces the source figure, item, reference answer, and six options, with the answer key highlighted in green. The examples provide qualitative diagnoses, not estimates of error prevalence. Any reported judge scores are archived values under the production rubric.

Preserving relations in evidence. Cases A and C concern comparison polarity and effect direction. Cases D and E require explanations to preserve intervention-reversal and reciprocal-perturbation relations. Case F requires the correct association of protein pairs, panels, and spatial overlap. These diagnoses describe inconsistencies in the responses without assigning a separate causal contribution to perception, knowledge, or reasoning.

Separating observations from predictions. Case B concerns a prediction that contradicts the intervention premise. Case G distinguishes a falsifiable proposal from an unsupported observational premise or necessity claim. In Cases H and I, the models' open answers make testable predictions that extend the displayed observations, while their selected options contain direction or mechanism errors; the two response formats therefore carry different diagnoses for the same item.

Paired-track interpretation. Cases B and E show that key agreement can coexist with an inconsistent open explanation; Cases C, D, and F show that an accurate relation in an open answer need not appear in the selected option. Cases H and I extend this divergence to L3 hypothesis items, where a well-formed proposal accompanies a flawed selection. This analysis motivates reading the two tracks together, with explicit attention to the options and scoring rule.

## F.1 SINGLE- VERSUS MULTI-SUBFIGURE DIFFICULTY

Multi-subfigure items account for 3,094/6,077 items (50.9%). All five models in Table 43 are at least as accurate on multi-subfigure items as on single-subfigure items, with differences of 0.4–7.8 percentage points. For GPT-5.6-sol, accuracy increases from 56.0% to 63.7% on L1 and from 58.0% to 62.6% on L2, and changes from 44.7% to 44.1% on L3. Thus, panel count alone does not define task difficulty; Case F illustrates the more specific requirement to preserve relations across panels.

Table 43: MCQ accuracy (%) on single- versus multi-subfigure items (five representative models of the eleven with open-response coverage).
<table><tr><td>Model</td><td>Single</td><td>Multi</td><td>∆(pp)</td></tr><tr><td>GPT-5.6-sol</td><td>54.5</td><td>55.7</td><td>+1.2</td></tr><tr><td>Claude-Sonnet-4-5</td><td>45.4</td><td>53.2</td><td>+7.8</td></tr><tr><td>Qwen3-VL-8B</td><td>33.6</td><td>34.0</td><td>+0.4</td></tr><tr><td>InternVL3.5-8B</td><td>29.6</td><td>30.9</td><td>+1.3</td></tr><tr><td>LLaVA-OneVision-7B</td><td>25.0</td><td>26.1</td><td>+1.1</td></tr></table>

![](images/fd91faa0f3f18e28c4281d3914eb85c5585d32a79cc3774ac0554decdfe42a8a.jpg)  
Source. LGALS3BP silencing and abnormal centriolar structures in interphase versus mitotic cells (panel b); L1 / Predict; Fogeron et al. (2013).  
A: Mitotic cells show a higher susceptibility to forming supernumerary centrioles upon LGALS3BP depletion. Approximately 40% of mitotic cells exhibit abnormal (>2) centriolar structures, compared to approximately 20% in interphase cells, indicating a more pronounced effect during mitosis.

## F.2 CASE A: LGALS3BP READOUT POLARITY (L1)

B: Mitotic U2OS cells show a higher susceptibility to forming supernumerary centrioles upon LGALS3BP depletion. Approximately 78% of mitotic cells exhibit abnormal (>2) centriolar structures, compared to approximately 58% in interphase cells, indicating a 20% higher incidence.

C: Mitotic cells show a higher susceptibility to forming supernumerary centrioles upon LGALS3BP depletion. Approximately 78% of mitotic cells exhibit abnormal (>2) centriolar structures, compared to approximately 52% in interphase cells, indicating a more pronounced effect during mitosis.

D: Interphase cells show a higher susceptibility to forming supernumerary centrioles upon LGALS3BP depletion. Approximately 78% of interphase cells exhibit abnormal (>2) centriolar structures, compared to approximately 52% in mitotic cells, indicating a more pronounced effect during interphase.

E: Mitotic cells show a higher susceptibility to forming supernumerary centrioles upon LGALS3BP depletion. Approximately 78% of mitotic cells exhibit abnormal (>2) centriolar structures, compared to approximately 58% in interphase cells, indicating a 20% higher incidence in mitotic cells.

F: Mitotic U2OS cells show a higher susceptibility to forming supernumerary centrioles upon LGALS3BP depletion. Approximately 40% of mitotic cells exhibit abnormal (>2) centriolar structures, compared to approximately 20% in interphase cells, indicating a more pronounced effect during mitosis.

Figure 8: Case A exhibit (L1 / Predict). Source figure, item, reference answer, and six MCQ options; the keyed option (C) is highlighted in green.

Panel b marks cells with >2 centriolar structures in red. After LGALS3BP siRNA, this fraction is approximately 52% in interphase and 78% in mitosis. GPT-5.6-sol preserves the comparison (“about 80 versus 50"; overall 5; keyed C). Claude-Sonnet-4-5 and Qwen3-VL-8B reverse it in their open responses (both overall 1). Both select E, whose error is numerical rather than directional, so the open-response diagnosis and the selected letter carry different information. The measured outcome is the number of centrin-positive structures, not the formation of mature, functional centrioles.

## F.3 CASE B: IFNAR1 COUNTERFACTUAL ACROSS FORMATS (L1)

Source. IFNAR1 surface loss after stimulation, with cycloheximide pretreatment supplied as the question premise (panels f/g); L1 / Predict; Chmiest et al. (2016).

![](images/ec86b54b02645dcceb0d9444bf89cfd93d76bc002797e41e7dd64e503198a47a.jpg)

Title: Spatiotemporal control of interferon-induced JAK/STAT signalling and gene transcription by the retromer complex

Question: Based on the immunoblot data in subfigure e and the quantification in subfigure f, if the experiment were repeated without cycloheximide pretreatment, how would the observed decrease in IFNAR1 levels at 1 h and 2 h likely compare to the results shown, and why?

Reference\_answer: Without cycloheximide (which blocks new protein synthesis), the observed decrease in IFNAR1 levels at 1 h and 2 h would likely be less pronounced or masked. This is because ongoing de novo synthesis of IFNAR1 would partially compensate for the lysosomal degradation occurring simultaneously, whereas cycloheximide isolates the degradation kinetics by preventing new receptor production.

A: Without cycloheximide (which blocks new protein synthesis), the observed decrease in IFNAR1 levels at 1 h and 2 h would be more pronounced than shown. This is because cycloheximide inhibits new protein synthesis, and its absence would allow for continued degradation of IFNAR1 without compensatory new protein production, contrasting with the results where new synthesis is blocked.

B: Without cycloheximide (which blocks new protein synthesis), the observed decrease in IFNAR1 levels at 1 h and 2 h would likely be less pronounced or masked. This is because ongoing de novo synthesis of IFNAR1 would partially compensate for the lysosomal degradation occurring simultaneously, whereas cycloheximide isolates the degradation kinetics by preventing new receptor production.

C: Without cycloheximide (which blocks new protein synthesis), the observed decrease in IFNAR1 levels at 1 h and 2 h would likely be more pronounced. This is because cycloheximide inhibits new protein synthesis; thus, its absence would allow for continued degradation of IFNAR1 without new synthesis to compensate, making the decrease greater than in the shown results.

D: Without cycloheximide (which blocks new protein synthesis), the observed decrease in IFNAR1 levels at 1 h and 2 h would likely be more pronounced compared to the results shown. This is because cycloheximide inhibits protein synthesis, and its absence would allow for continued protein degradation, leading to a greater reduction in IFNAR1 levels, whereas the shown results reflect blocked synthesis.

E: Without cycloheximide (which blocks new protein synthesis), the observed decrease in IFNAR1 levels at 1 h and 2 h would likely be more pronounced compared to the results shown in subfigure f. This is because cycloheximide inhibits protein synthesis, which could prevent the degradation of IFNAR1 that occurs after treatment; without this inhibition, the levels would decrease more rapidly, leading to a steeper decline in the quantified values.

F: Without cycloheximide (which blocks new protein synthesis), the observed decrease in IFNAR1 levels at 1 h and 2 h would likely be more pronounced compared to the results shown. This is because cycloheximide pretreatment can inhibit protein synthesis, which may have an effect on the observed changes; without the pretreatment, the decrease would be even more significant, as the protein synthesis process would not be inhibited to compensate for degradation.

Figure 9: Case B exhibit (L1 / Predict). Source figure, item, reference answer, and six MCQ options; the keyed option (B) is highlighted in green.

Under the question's premise, removing the synthesis inhibitor allows resynthesis to offset degradation, predicting a weaker net IFNAR1 decline at 1 and 2 h. Qwen3-VL-8B instead predicts stronger loss by invoking absent compensatory synthesis, contradicting that premise (overall 1). GPT-5.6- sol and Claude-Sonnet-4-5 preserve the expected direction (both 5). All three select the keyed B, illustrating that a correct choice can coexist with an inconsistent generated explanation. The prediction follows from the counterfactual premise and the inhibitor's role; the displayed panels report the original treatment condition.

## F.4 CASE C: C8ORF4 EFFECT DIRECTION ACROSS MODELS (L1)

Source. C8orf4 knockout in Huh7 (panel b) and knockdown in primary HCC (panel f) and sphereinitiating-cell frequency; L1 / Predict; Zhu et al. (2015).

![](images/b8b10d2bed020f4273468fb2decda69a74ca7c944c5ceb82d346dbea96663830.jpg)

Title: C8orf4 negatively regulates self-renewal of liver cancer stem cells via suppression of NOTCH2 signalling

Question: Based on the sphere formation results in Huh7 cells (panel b) and HCC primary cells (panel f), predict whether the effect of C8orf4 loss on self-renewal is specific to established cell lines or a general feature of liver cancer stem cells.

Reference\_answer: The effect is a general feature of liver cancer stem cells, not specific to established cell lines. This is evidenced by the fact that both C8orf4 knockout in Huh7 cells and C8orf4 knockdown in HCC primary cells result in a significant increase in sphere-initiating cells, demonstrating that the loss of C8orf4 consistently enhances self-renewal across different liver cancer models.

A: The effect of C8orf4 loss on self-renewal is a general feature of liver cancer stem cells, not specific to established cell lines. This is evidenced by the fact that both C8orf4 knockout in Huh7 cells and C8orf4 knockdown in HCC primary cells result in a significant decrease in sphere-initiating cells, demonstrating that the loss of C8orf4 consistently impairs self-renewal across different liver cancer models.

B: The effect is specific to established cell lines. In Huh7 cells (panel b), C8orf4 knockout significantly decreases sphere-initiating cells relative to WT, as WT is highest and KO bars are lower with an asterisk. In HCC primary cells (panel f), shCtrl is highest and shC8 bars are lower but without asterisk, indicating no significant change. Thus, loss of C8orf4 only impairs self-renewal in established lines, and the larger decrease in Huh7 reinforces this line-specificity.

C: The effect of C8orf4 loss on self-renewal is specific to established cell lines, not a general feature of liver cancer stem cells. This is evidenced by the fact that C8orf4 knockout in Huh7 cells results in a significant increase in sphere-initiating cells, whereas C8orf4 knockdown in HCC primary cells does not significantly affect sphere formation, demonstrating that the loss of C8orf4 enhances self-renewal only in established cell lines.

D: The effect of C8orf4 loss on self-renewal is a general feature of liver cancer stem cells, not specific to established cell lines. This is evidenced by the fact that both C8orf4 knockout in Huh7 cells and C8orf4 knockdown in HCC primary cells result in a significant reduction in sphere-initiating cells, demonstrating that the loss of C8orf4 consistently impairs self-renewal across different liver cancer models.

E: The effect is a general feature of liver cancer stem cells, not specific to established cell lines. This is evidenced by the fact that both C8orf4 knockout in Huh7 cells and C8orf4 knockdown in HCC primary cells result in a significant increase in sphere-initiating cells, demonstrating that the loss of C8orf4 consistently enhances self-renewal across different liver cancer models.

F: The effect of C8orf4 loss on self-renewal is a general feature of liver cancer stem cells, not specific to established cell lines. This is evidenced by the fact that both C8orf4 knockout in Huh7 cells (panel b) and C8orf4 knockdown in HCC primary cells (panel f) result in a significant decrease in sphere-initiating cells, demonstrating that the loss of C8orf4 consistently impairs self-renewal across different liver cancer models.

Figure 10: Case C exhibit (L1 / Predict). Source figure, item, reference answer, and six MCQ options; the keyed option (E) is highlighted in green.

C8orf4 knockout in Huh7 cells and knockdown in primary HCC both increase the sphere-initiatingcell fraction. Qwen3-VL-8B notes the shared effect but describes it as impairing sphere formation (overall 2). GPT-5.6-sol and Claude-Sonnet-4-5 preserve the increase (both 5). GPT-5.6-sol selects the keyed E, whereas Claude-Sonnet-4-5 and Qwen3-VL-8B select F; Claude's open answer and selected option therefore disagree in direction. In fixed-prompt image controls, GPT-5.6-sol changes from E with the original image to A with no image or a swapped image. This is an item-level observation of input sensitivity. The biological comparison concerns sphere formation in these two experimental systems.

## F.5 CASE D: ROS-TGFBRI INTERVENTION-REVERSAL CHAIN (L2)

Source. ROS, TGFBRI inhibition, and paired-daughter division symmetry (panels a/d; b/c auxiliary); L2 / Explain; Hinge et al. (2017).

![](images/46cdd9ac4b59d261c885f4d768d82bcacd56c831938b5fa0ebd6a67443dade4a.jpg)

![](images/de0075d21fd1bc2b0fd4bee90f9e646ccf8ec6600d8ed990cf5cfa9551bae9e0.jpg)

![](images/2837e67459ef97e256c134018d3f855933b787b669d7881b0ee0f18014aa99c9.jpg)

![](images/c75402580cccf11ad1a1a018b63eb23f43d9ab7e15f84fa82ea713ee4022f2dc.jpg)

![](images/118e77edf813b432f6ac86a1ac1e58c358d73e911f8ec496a4c72e89833cb3a8.jpg)  
Figure 11: Case D exhibit (L2 / Explain). Source figure, item, reference answer, and six MCQ options; the keyed option (F) is highlighted in green.

In panel d, approximately 80% of control daughter pairs divide symmetrically. $\mathrm { H _ { 2 } O _ { 2 } }$ reduces this fraction to approximately 25%, and TGFBRI inhibition restores the distribution towards control. GPT-5.6-sol and Claude-Sonnet-4-5 link ROS to TGF-β-dependent asymmetric division (both overall 5). Qwen3-VL-8B instead states that ROS promotes symmetric division, reversing the observed direction (overall 1). GPT-5.6-sol selects the keyed F. Claude-Sonnet-4-5's B preserves the ROS direction but states that the inhibitor enhances the shift, conflicting with the reversal. The evidence supports an $\mathrm { H _ { 2 } O _ { 2 } } \cdot$ -associated, TGFBRI-inhibitor-sensitive change in this assay; it does not identify a unique direct molecular target.

## F.6 CASE E: CCP1 SIGN FROM RECIPROCAL PERTURBATIONS (L2)

Source. CCP1 deletion and overexpression versus nuclear Msn2 (panels d/g); L2 / Explain; Bodvard et al. (2017).

![](images/ca6d217f154ca916b6ddac0958b31bffae575e5ee1b1c0733392d086deb244c1.jpg)  
Figure 12: Case E exhibit (L2 / Explain). Source figure, item, reference answer, and six MCQ options; the keyed option (A) is highlighted in green.

CCP1 deletion increases the nuclear-Msn2 fraction relative to the wild type in panel d, whereas overexpression decreases it relative to the wild type in panel g. These reciprocal perturbations support negative regulation. Qwen3-VL-8B instead states that CCP1 is required and that deletion impairs translocation (overall 1). GPT-5.6-sol and Claude-Sonnet-4-5 preserve the regulatory sign (overall 5 and 3), and all three select the keyed A. Claude's faithfulness and evidence-consistency scores are both 5, separating correct direction from the overall assessment of mechanistic detail. Comparisons use each panel's own wild-type baseline, and overexpression attenuates the signal.

## F.7 CASE F: BRCA1 SPATIAL INTEGRATION (L2)

Source. BRCA1–RIF1 versus BRCA1–53BP1 spatial overlap (panels a/c); L2 / Explain; Ha et al.   
(2017).

![](images/273b38607df4bfe66e5300d51a0014c1e26ae652e87f1ecb01db39e1dd7b6b4a.jpg)

Title: The anaphase promoting complex impacts repair choice by protecting ubiquitin signalling at DNA damage sites

Question: According to panels a and c, what is the difference in the spatial relationship between BRCA1 and RIF1 compared to BRCA1 and 53BP1 at DNA damage sites 1 hour after ionizing radiation?

Reference\_answer: BRCA1 and RIF1 form distinct, non-overlapping foci with minimal co-localization, evidenced by alternating peaks in the intensity profile in panel a. In contrast, BRCA1 and 53BP1 co-localize at some DNA damage sites, forming yellow foci in the merge image and showing overlapping peaks in the intensity profile in panel c.

A: At DNA damage sites 1 hour after ionizing radiation, BRCA1 and RIF1 are more closely associated with each other, forming overlapping foci with closer proximity in the intensity profile in panel a. In contrast, BRCA1 and 53BP1 show a more dispersed distribution, with distinct non-overlapping foci and alternating peaks in the intensity profile in panel c.

B: BRCA1 and RIF1 co-localize at DNA damage sites, as shown by yellow foci in panel a's merge image and overlapping peaks in its intensity profile over 18.9 μm. In contrast, BRCA1 and 53BP1 form distinct non-overlapping foci with alternating peaks in panel c's intensity profile over 22.2 μm and separate red/green foci in the merge.

C: BRCA1 and RIF1 form distinct, non-overlapping foci with minimal co-localization, evidenced by alternating peaks in the intensity profile in panel a. In contrast, BRCA1 and 53BP1 co-localize at some DNA damage sites, forming yellow foci in the merge image and showing overlapping peaks in the intensity profile in panel c.

D: At DNA damage sites 1 hour after ionizing radiation, BRCA1 and RIF1 are closer together, forming overlapping foci with coincident peaks in the intensity profile in panel a. In contrast, BRCA1 and 53BP1 are farther apart, with minimal overlapping foci and alternating peaks in the intensity profile in panel c.

E: At DNA damage sites 1 hour after ionizing radiation, BRCA1 and RIF1 are more closely associated with each other than with 53BP1, indicating a stronger interaction, as seen by overlapping peaks in the intensity profile in panel a. In contrast, BRCA1 and 53BP1 show weaker association, with distinct foci and alternating peaks in the intensity profile in panel c.

F: At DNA damage sites 1 hour after ionizing radiation, BRCA1 and RIF1 show a more colocalized spatial relationship, evidenced by overlapping peaks in the intensity profile in panel a. In contrast, BRCA1 and 53BP1 exhibit less overlap and more dispersed foci at some DNA damage sites, with alternating peaks in the intensity profile in panel c.

Figure 13: Case F exhibit (L2 / Explain). Source figure, item, reference answer, and six MCQ options; the keyed option (C) is highlighted in green.

Panel a shows largely separate or adjacent BRCA1–RIF1 signals; panel c shows partial BRCA1– 53BP1 overlap at some sites. Qwen3-VL-8B reverses these relations (overall 1; F). GPT-5.6-sol and Claude-Sonnet-4-5 state the observed comparison (both overall 5), although only GPT-5.6-sol selects the keyed C and Claude selects B. The task requires preserving the association between each protein pair and its spatial relation across panels. These image readouts describe local overlap, which is distinct from direct physical binding or a global colocalization percentage.

## F.8 CASE G: HAIR-REGENERATION HYPOTHESIS ON A FAULTY PREMISE (L3)

Source. Hair regeneration with SB versus PHM populations; pigmentation and shaft morphology (panels c/d); L3 / Discover; Toyoshima et al. (2012).

![](images/b1b5d1f9d6a34fec48858b85fa6089a2b6a6d2a113869a375da6a1ea3fb5706e.jpg)  
Figure 14: Case G exhibit (L3 / Discover). Source figure, item, reference answer, and six MCQ options; the keyed option (D) is highlighted in green.

The source context reports pigmentation recovery of 68.3% with SB and 40.1% with PHM, and normal morphology of 27.0% and 83.3%, respectively. GPT-5.6-sol proposes different contributions to pigmentation and shaft organization, with selective addition or ablation as a test (overall 4; keyed D). Claude-Sonnet-4-5 builds a cortex/medulla hypothesis on a melanin-localization contrast, although dark granules appear in central TEM fields under both conditions (overall 1; C). The identifiable error is the asserted observation. Qwen3-VL-8B captures the direction but describes PHM as “essential," a necessity claim stronger than the comparison supports (overall 3; C). The observations establish contrasting regeneration outcomes; the proposed division of cellular roles remains a hypothesis.

## F.9 CASE H: YAP-ECT2/FGD3 PAIRED-TRACK DIVERGENCE (L3)

Source. Ect2/Fgd3 induction under activated YAP with EtOH/HTVi injury (panels d/e; f is a schematic); L3 / Discover; Miyamura et al. (2017).

![](images/3f96b41e7259b2b7ba9887e6f84f6a06f48924316f47b4734495a3ba4a701a95.jpg)  
Figure 15: Case H exhibit (L3 / Discover). Source figure, item, reference answer, and six MCQ options; the keyed option (B) is highlighted in green. Option F's knockdown prediction contradicts its own mechanism claim, as discussed below.

Observed evidence and hypotheses. Panel e reports Ect2/Fgd3 expression changes under activated YAP with injury, and panel f summarizes the proposed mechanism. GPT-5.6-sol proposes knockdown with migration, elimination, and proliferation readouts; Claude-Sonnet-4-5 also proposes an intervention test. These answers supply falsifiable extensions of the expression evidence. Restoration of proliferation is a separate predicted readout, and expression association does not establish direct promoter binding or a loss-of-function effect.

Paired-track divergence. The key is B. GPT-5.6-sol selects F, Claude-Sonnet-4-5 selects B, and Qwen3-VL-8B selects C. Option F identifies Ect2/Fgd3 as downstream mediators of the damage response, yet predicts that their knockdown would enhance migration, apoptosis, and engulfment— contradicting both its own mechanism claim and the direction implied by the expression evidence. GPT-5.6-sol's open answer proposes exactly the knockdown test with the expected direction, so its selected option reverses the prediction that its own hypothesis implies. The case therefore separates the quality of the generated hypothesis from the correctness of the selected option.

## F.10 CASE I: INTEGRIN-LCK PAIRED-TRACK DIVERGENCE (L3)

Source. Lck inhibition/knockdown and downstream paxillin/CrkII phosphorylation, with the integrin→Lck source model (panels a/b/h); L3 / Discover; Ness et al. (2013).

![](images/484a12ac0361e25bce41a1384f8105b6e9cdd935b0cdcc4a1a706224cd0704c1.jpg)

Title: Lck tyrosine kinase mediates β1-integrin signalling to regulate Schwann cell migration and myelination

Question: Subfigure h illustrates that ITGB1 (β1-integrin) at the membrane connects to Lck to initiate the signaling cascade. Propose a falsifiable hypothesis regarding the effect of blocking β1-integrin with a neutralizing antibody on the downstream phosphorylation events measured in subfigures a and b.

Reference\_answer: Blocking β1-integrin with a neutralizing antibody would prevent the activation of membrane-associated Lck. Consequently, this would lead to a significant decrease in the downstream phosphorylation of paxillin and Crkll, mimicking the effects observed with direct Lck inhibition or knockdown, as the integrin is required to initiate the Lck-dependent signaling cascade.

A: Blocking β1-integrin with a neutralizing antibody would prevent the activation of membrane-associated Lck. Consequently, this would lead to a significant decrease in the downstream phosphorylation of paxillin and Crkll, as directly demonstrated in subfigures a and b, because the integrin is required to initiate the Lck-dependent signaling cascade.

B: Blocking β1-integrin with a neutralizing antibody would prevent the activation of membrane-associated Lck. Consequently, this would lead to a significant decrease in the downstream phosphorylation of paxillin and Crkll, while Lck phosphorylation remains unchanged, since integrin engagement transmits signals to Lck without altering its phosphorylation state, mimicking the effects observed with direct Lck inhibition or knockdown, as the integrin is required to initiate the Lck-dependent signaling cascade.

C: Blocking β1-integrin with a neutralizing antibody would decrease paxillin and Crkll phosphorylation only in the laminin condition of panel a, because the Lck inhibitor effect was observed on laminin, whereas the siRNA knockdown in panel b did not require laminin, so integrin blockade is condition-dependent and would not affect Schwann cells on other substrates.

D: Blocking β1-integrin with a neutralizing antibody would prevent the activation of membrane-associated Lck. Consequently, this would lead to a significant decrease in the downstream phosphorylation of paxillin and Crkll, mimicking the effects observed with direct Lck inhibition or knockdown, as the integrin is required to initiate the Lck-dependent signaling cascade.

E: Blocking β1-integrin with a neutralizing antibody would selectively reduce paxillin phosphorylation at 68 kDa but not Crkll at 40 kDa, because the P-value for paxillin (P<0.03) in panel a is less significant than that for Crkll (P<0.001), and the quantification in panel b shows a similar pattern (P<0.05 vs P<0.01), indicating that Crkll is less dependent on Lck activity downstream of integrin engagement.

F: Blocking β1-integrin with a neutralizing antibody would increase the phosphorylation of paxillin and Crkll, as a compensatory feedback loop would upregulate Lck activity, opposite to the decreased signals seen with Lck inhibitor (panel a) or siRNA (panel b), but the schematic shows integrin as an activator, so loss of integrin would paradoxically activate Lck.

Figure 16: Case I exhibit (L3 / Discover). Source figure, item, reference answer, and six MCQ options; the keyed option (D) is highlighted in green. Option B contradicts the integrin→Lck initiation shown in panel h, as discussed below.

Observed evidence and hypotheses. Panels a and b report Lck inhibition and knockdown, while panel h depicts integrin→Lck signalling. Predicting reduced paxillin/CrkII phosphorylation after β1-integrin antibody blockade is a testable extension of this pathway. GPT-5.6-sol adds an isotypecontrol comparison and stable total-protein levels as control expectations; Claude-Sonnet-4-5 proposes a conditional pathway test, and Qwen3-VL-8B predicts the same phosphorylation direction. These are intervention predictions, not observations of an antibody experiment in panels a and b.

Paired-track divergence. The key is D. GPT-5.6-sol selects B, while Claude-Sonnet-4-5 and Qwen3-VL-8B select D. Option B predicts reduced paxillin/CrkII phosphorylation but claims that Lck phosphorylation remains unchanged under integrin blockade, contradicting panel h, in which β1-integrin connects to Lck to initiate the cascade, and contradicting the kinase's phosphorylationdependent activation. GPT-5.6-sol's open answer supplies a well-formed intervention prediction with appropriate controls, yet its selected option carries this mechanism error. Option A separately conflates the proposed antibody experiment with the displayed panels by treating its effect as directly demonstrated.

## G ETHICS, LICENSE, AND DATA AVAILABILITY

Data provenance. All benchmark items are derived from figures and experimental contexts of peer-reviewed publications, assembled through the OmniScience corpus (Tao et al., 2026). No new biological experiments were conducted, and the benchmark contains no personal or otherwise identifiable information.

License. OMNIVCBENCH and OMNIVCTRAIN are released under the Creative Commons Attribution–NonCommercial–ShareAlike 4.0 International license (CC BY-NC-SA 4.0), the same license as OmniScience (Tao et al., 2026); we impose no additional or differing terms of use. The released packages host the materialized figure crops directly. Redistribution of the underlying source figures remains subject to the terms of the original publishers.

Intended use and misuse risks. The benchmark is intended solely for research evaluation of multimodal scientific reasoning. Reference answers, MDHNM distractors, and AIVC-Judge scores are evaluation annotations; they and the evaluated model outputs are not clinical or experimental decision-making evidence. Evaluated models were accessed through public APIs or their openweight releases under the respective model licenses.

Code and demo release. The anonymous repository at https://anonymous.4open. science/r/OmniVCBench includes the full AIVC-Judge implementation (level-specific rubrics, prompts, and the claim-decomposition scoring pipeline) together with a 300-item demo release pairing each open-response QA item with its MDHNM MCQ variant. The complete 6,077- item OMNIVCBENCH and 548,450-item OMNIVCTRAIN datasets will be released after the review process concludes; the demo uses the same stratified 300-item sample as Appendix A.6.