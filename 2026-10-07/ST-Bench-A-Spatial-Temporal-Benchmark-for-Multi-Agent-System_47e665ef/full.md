# ST-Bench: A Spatial-Temporal Benchmark for Multi-Agent System Generation on Scientific Research Tasks

Qi Cheng   
Rutgers University   
New Brunswick, NJ   
q.cheng@rutgers.edu   
Shengyu Chen   
University of Pittsburgh   
Pittsburgh, PA   
shc160@pitt.edu

Zhengzhang Chen NEC Labs America Princeton, NJ zchen@nec-labs.com

Haifeng Chen   
NEC Labs America   
Princeton, NJ   
haifeng@nec-labs.com   
Licheng Liu   
University of Minnesota   
Minneapolis, MN   
lichengl@umn.edu

Wei Cheng NEC Labs America Princeton, NJ weicheng@nec-labs.com

Xiaowei Jia   
Rutgers University   
New Brunswick, NJ   
xj159@cs.rutgers.edu   
Rongchao Dong   
Rutgers University   
New Brunswick, NJ   
rc.dong@rutgers.edu

Dan Lu Oak Ridge National Laboratory Oak Ridge, TN lud1@ornl.gov

Yiqun Xie   
University of Maryland   
College Park, MD   
xie@umd.edu   
Haoyu Wang   
NEC Labs America   
Princeton, NJ   
haoyu@nec-labs.com

## Abstract

The rapid progress of LLM-based multi-agent systems (MAS) has shown that they largely outperform single agents on coding, math, and QA tasks, where executable tests provide a binary success signal. Whether this advantage transfers to real scientific data analysis remains untested. We introduce ST-Bench, a benchmark designed to answer two questions: whether MAS outperform single agents on complex scientific data analysis tasks, and if so, by how much and at what additional cost. ST-Bench contains 100 data science tasks adapted from published Earth science studies across hydrology, agriculture, and wetland methane research, expanded into 2,067 queries grounded in additional published studies and validated by domain experts. Using ST-Bench, we evaluate five recent MAS generation methods under two training protocols, against single-agent baselines on the same GPT-5 backbone. Nine of the ten MAS configurations exceed the cheapest single-agent baseline, with the strongest reaching nearly three times its composite score. This gain is primarily attributable to coverage: trained workflows produce realistic numerical metrics on a larger fraction of queries, while the quality of those metrics, conditional on producing realistic output, is comparable to that of the single-agent baseline. The strongest configuration requires approximately four times the single-agent inference time, whereas a more economical workflow captures the majority of the benefit at less than twice the cost. MAS specialization confers measurable benefit on scientific data analysis, but the benefit is conditional rather than universal.

![](images/a7ba1d650ad8ea74b06d79289ef35c5fe4514d33e87b9a8915cadb46a55f8e5b.jpg)  
Figure 1: Construction of ST-Bench. We first define 16 task categories spanning the data-science pipeline (Stage 1), then instantiate each (category, domain) cell with one or more peer-reviewed source papers (Stage 2). Every resulting task is expanded to ∼20 paper-grounded queries via dataset scopes (Stage 3). The 2,067 queries are stratified into per-task train/val/test splits that feed both the MAS optimizer and our evaluation framework (Stage 4).

## 1 Introduction

To amplify the capabilities of LLMs, a substantial series of recent works have orchestrated LLMs into multi-agent systems (MAS), in which a set of specialized agents coordinate through designed workflows to solve tasks more reliably than single agents. Early MAS frameworks established the design space through hand-crafted role decompositions and communication protocols [Li et al., 2023, Wu et al., 2023, Hong et al., 2024, Wang et al., 2024, Madaan et al., 2023]. More recent work treats the workflow itself as an object to be optimized, which automatically search for strong workflows over a target task distribution [Zhang et al., 2025b, Hu et al., 2025b, Zhuge et al., 2024, Zhang et al., 2025c, Nie et al., 2025, Gao et al., 2025, Zhang et al., 2025a]. Across coding benchmarks [Chen et al., 2021, Austin et al., 2021], math benchmarks [Cobbe et al., 2021, Hendrycks et al., 2021], and question answering benchmarks [Yang et al., 2018, Dua et al., 2019], these methods report substantial improvements over single-agent baselines. Building on these results, it is natural to ask whether MAS approaches can also advance automated scientific research, where tasks are often more complex, data-centric, and open-ended.

In this work, we argue that no existing MAS benchmarks can be used to answer this research question. Most benchmarks used to validate MAS methods share a structural property that scientific research often lacks: a binary oracle. HumanEval and MBPP grade through executable unit tests, and GSM8K and MATH grade through a single numerical answer. Each candidate workflow can therefore be assigned a clean binary score, which is precisely the feedback signal that MAS inner-loop optimizers were designed to use. Real scientific data analysis tasks are much more diverse and often do not offer such signal. A clustering task can produce many valid clusterings, a forecasting task can be partially correct along multiple numerical axes, and a gap filling task can produce a usable answer that does not match any single reference exactly. In these settings, success is better measured through continuous or partial-credit metrics rather than binary correctness. It therefore remains an open empirical question whether workflow specialization, which has been shown to be effective under binary-oracle benchmarks, still provides a measurable advantage for diverse scientific tasks evaluated with more nuanced scientific metrics.

Although several benchmarks for scientific tasks already exist, they do not support such a test. Scientific agent benchmarks [Chen et al., 2025, Gu et al., 2024, Majumder et al., 2024] provide carefully curated scientific tasks, but were designed to evaluate a single agent on a single attempt per task. They therefore lack the within-task pool of samples that MAS generation methods require: an inner-loop optimizer cannot iterate on a single query, and a held-out single query cannot test whether an optimized workflow generalizes. Spatial-temporal scientific benchmarks [Takamoto et al., 2024, Newman et al., 2022] provide the real heterogeneous data that scientific tasks involve, but they were designed to compare fixed model classes or forecasting skills, not LLM-generated analysis workflows, and lack the prompt and metric that lets a generated MAS read raw files, choose preprocessing steps, run analysis code, and report named scientific metrics. Hence, the comparative question we wish to answer cannot be answered by any available benchmark. A comparison between available benchmarks can be found in Table 1 and extended related work is provided in Appendix A..

Table 1: Comparison of ST-Bench to agent benchmarks. MAS Opt. indicates whether the benchmark provides within-task query pools for MAS generation systems. Reward Signal distinguishes between a binary oracle and a continuous reward. Multi-axis Eval. indicates evaluation along orthogonal axes beyond accuracy.
<table><tr><td>Benchmark</td><td>Task Sources</td><td>Code Gen.</td><td>MAS Opt.</td><td>Reward Signal</td><td>Multi-axis Eval.</td></tr><tr><td>SWE-Bench [Jimenez et al., 2023]</td><td>GitHub</td><td>File-level</td><td>x</td><td>Binary</td><td>x</td></tr><tr><td>AgentBench [Liu et al., 2023]</td><td>Mixed</td><td>Mixed</td><td>x</td><td>Binary</td><td>x</td></tr><tr><td>TaskBench [Shen et al., 2024]</td><td>Synthetic</td><td>No code</td><td>X</td><td>Binary</td><td>x</td></tr><tr><td>MLAgentBench [Huang et al., 2023]</td><td>Kaggle</td><td>File-level</td><td>x</td><td>Continuous</td><td>x</td></tr><tr><td>BLADE [Gu et al., 2024]</td><td>31 Publications</td><td>Function</td><td>x</td><td>Binary</td><td>x</td></tr><tr><td>DiscoveryBench [Majumder et al., 2024]</td><td>27 Publications</td><td>Code Gen</td><td>x</td><td>Binary</td><td>x</td></tr><tr><td>ScienceAgentBench [Chen et al., 2025]</td><td>44 Publications</td><td>File-level</td><td>x</td><td>Mixed</td><td>Partial</td></tr><tr><td>ST-Bench (Ours)</td><td>Publications</td><td>File-level</td><td>√</td><td>Continuous</td><td>√</td></tr></table>

To this end, we present ST-Bench, a benchmark and evaluation framework designed to enable rigorous comparison both between MAS generation systems and against single-agent baselines on complex scientific research tasks. The construction of ST-Bench follows four design principles. (1) Tasks grounded in real published scientific work: We adapt 100 tasks from peer-reviewed Earth science publications spanning three science domains. Each task has processed data files, a natural language prompt, and a list of metrics providing what spatial-temporal data benchmarks lack. (2) Query pools per task: We expand each task into approximately twenty queries by attaching dataset scopes drawn from additional peer-reviewed publications. This within-task pool supplies the data samples that MAS generation methods require to optimize, while the test queries let them do evaluations, neither of which is possible on current agentic benchmarks. (3) Per-task reward signals: Multi-agent generation methods rely on optimizers that select between candidate workflows by using a score per candidate. Existing task specific benchmarks often provide this signal through unit tests, but existing scientific benchmarks do not. We therefore equip every task in ST-Bench with a continuous reward signal, which is a realistic filter that excludes defensive defaults paired with a reference-aware verifier returns a continuous scorer. To our knowledge, ST-Bench is the first scientific benchmark whose per-task signals are fine-grained enough to drive MAS workflow optimization on complex science problems rather than only grading a final attempt. (4) A multi-axis evaluation framework: ST-Bench reports performance along two orthogonal sets of axes. The first is task structure: the 100 tasks span 16 categories covering the standard data science pipeline, three difficulty tiers, and four analytical tag sets recording the capability dimension, pipeline stage, cognitive skill, and MASspecific challenge that each task most stresses. The second is system property where a per-task win rate against the single-agent baseline, cost ratios in both inference time and token consumption, an efficiency score expressing accuracy gain per unit of additional cost, a generalizability measure that compares workflows trained at per-task and per-category granularity, and additional axes for scalability across LLM backbones and internal workflow parallelism. Together, these tasks and metric axes let ST-Bench produce a comprehensive diagnostic of where each MAS succeeds, where it fails, and along which dimensions its specialization is worth its cost. The construction pipeline are shown in Figure 1.

Using ST-Bench, we evaluate five recent MAS generation methods against single-agent baselines. Comparison against the single agent addresses the inter-system half of our research goal. Comparison among the five methods, and between each method’s per-task and per-category variants, addresses the intra-system half. We also report cost along two dimensions, wall-clock inference time and token consumption where token accounting is available. Our contribution is therefore not a claim that current MAS methods solve scientific data analysis, but a benchmark and evaluation framework together with empirical findings that establish, both within MAS generation methods and against single-agent baselines, when these systems improve, by how much, and at what cost. We position ST-Bench as a measurement tool for the open question of when MAS is worth its added compute, and as a way to provide insights for future MAS designs that can compute over real heterogeneous data, score real scientific metrics, and manage inference cost effectively.

## 2 Dataset

In this section, we introduce ST-Bench, which aims to evaluate MAS generation systems and singleagent baselines on the real research tasks that involve large-scale spatio-temporal data. We therefore anchor every task in ST-Bench to a peer-reviewed Earth science publication. That publication defines the scientific question, identifies the data on which it is studied, and defines the metric used to evaluate performance. Each task is then turned into a MAS readable instance that includes a natural language prompt, a directory of preprocessed uni-format files, and a list of named scientific metrics. Hence, ST-Bench moves beyond synthetic tests of MAS coordination on toy problems and instead evaluates whether a generated MAS can solve scientific problems to the standards established by peer-reviewed research.

## 2.1 Problem Formulation

Given a natural language prompt and a directory of raw data files, a system under evaluation must read and process the data, decompose the task, run the analysis the task requires, and output related metric values. We resolve the task at the level of a single query rather than a free-form code submission, so that the same machinery used to grade a final attempt can also drive an MAS optimizer. Here we introduce the four components of each query, each chosen to support both single-agent inference and MAS workflow generation.

Task Prompt. A task prompt that over-specifies the algorithm short-circuits the workflow generation problem we aim to study, while one that under-specifies the goal leaves no measurable target. We therefore phrase each prompt as a concise scientific objective and avoid prescribing a particular algorithmic approach. Open-ended prompts force a generated MAS to actually decompose the task and select tools, rather than retrieve a memorized solution from a coding benchmark. They also leave the metric space large enough that the optimizer can find improvements rather than running into a single optimum on the first iteration. Each prompt also contains a block that specifies the subset of the dataset to be used.

Dataset Files. Raw scientific datasets ship in many file formats, and a benchmark that exposed every published format directly would force tested MAS systems to spend more effort on format parsing than on scientific reasoning. We therefore convert every dataset we used and their subsets to a uniform Parquet representation before staging it into the working directory, so that file-format handling is not the variable under test.

Defined Metrics. The MAS optimizer’s inner loop requires a continuous, partial-credit fitness signal per task; a binary oracle does not transfer to scientific data analysis. We therefore equip each task with a list of named numeric metrics, each paired with an inequality target such as “> 0.3” (when higher values are better) or “lower is better” (when no fixed threshold is specified). Inequality targets provide two advantages over gold answers (i.e., single ground-truth answers). First, they accommodate the open-endedness of valid scientific solutions: many different clusterings can clear a Silhouette > 0.3 bar, so the candidate space the MAS generation method optimizes over is genuinely large rather than a single retrieval. Second, the numeric value the system reports, not just whether it cleared the target, is a continuous quantity, which is exactly what an inner-loop fitness function needs to differentiate between candidate workflows that all happen to pass the target. The training-time verifier (§3.3) consumes this continuous quantity directly.

Reference. We attach to each task a citation to the publication it was adapted from. Note that this field is stored only for expert validation and for analysis of benchmark coverage, and it is not exposed to MAS generation systems and single agent baselines.

## 2.2 Dataset Construction

ST-Bench was built top-down rather than by accumulating tasks from a single source. We first fixed task categories from data science capabilities, then filled each category with papers from three real Earth science domains, and expanded each resulting task into a pool of paper-grounded queries. The end product is 100 tasks expanded into 2,067 queries. Figure 5 summarizes the resulting task and query distribution.

Table 2: A representative task in ST-Bench: C3.4 (Model Comparison, hydrology).
<table><tr><td>Field</td><td>Value</td></tr><tr><td>Task ID</td><td>C3.4</td></tr><tr><td>Difficulty</td><td>Easy</td></tr><tr><td>Category</td><td>Model Comparison</td></tr><tr><td>Domain</td><td>Hydrology (CAMELS)</td></tr><tr><td>Source</td><td>Kratzert et al. (2018), HESS [Kratzert et al., 2018]</td></tr><tr><td>Prompt</td><td>You are given daily meteorological forcing and observed streamflow for 10 CAMELS basins spanning diverse hydrological regimes. Train an LSTM, a random forest, and a linear regression model independently for each basin, evaluate them using NSE, KGE, and the flow duration curve, and analyze residual patterns where the three models disagree. Report the median NSE across basins and the cross-basin consistency of model rankings. The data files can be found in {Scope file path }</td></tr><tr><td>Metrics</td><td>Median NSE across basins; model ranking consistency across basins.</td></tr></table>

Task Categories. Before any source paper or dataset was selected, we defined 16 task categories spanning the standard data science pipeline: Clustering, Cross-Dataset Evaluation, Data Augmentation, Exploratory Data Analysis, Feature Analysis, Feature and Label Correlation Checking, Forecasting Modeling, Missing Value Imputation, Model Comparison, Model Distillation, Novel Evaluation Metrics, Performance Stratification, Prediction Modeling, Sampling Strategies, Transfer Learning, and Up/Downscaling. The categories cover data preparation, exploration, modeling, and evaluation, so every internal role a generated MAS might specialize in is exercised on multiple tasks. They are also concrete enough that two source papers in the same category produce comparable problems, which lets per-category training (§3.1) operate on a coherent training pool.

Science Domains. With the categories fixed, we chose three Earth science domains to populate them. We instantiate the 16 categories across three Earth-science domains: hydrology using CAMELS [Newman et al., 2022], agriculture using two crop yield datasets [Khaki et al., 2020, Paudel et al., 2024], and wetland methane using X-MethaneWet/FLUXNET-CH4 [Sun et al., 2026], plus 11 general-control tasks. These domains provide heterogeneous spatial-temporal data and cover forecasting, clustering, transfer, model comparison, gap filling, and stratified evaluation. Full dataset descriptions and source details are provided in Appendix B.1.

Tasks. For each (category, domain) pair, we sourced one or more peer-reviewed publications that operationalized the category’s analytical skill on that domain’s data. Some cells admitted multiple natural instances and yielded several tasks; a few cells had no natural fit and are not represented. For each chosen source paper we wrote a concise prompt, reformulated quantitative success criteria as named metrics with inequality targets, and prepared the staged Parquet files needed to compute those metrics. The eventual count of tasks per category ranges from 4 (Model Distillation) to 11 (Feature and Label Correlation Checking). Table 2 shows one task end-to-end.

Queries A single source paper gives one analysis on one canonical dataset slice. That is sufficient for grading a single-agent attempt, but it leaves an MAS generation method that operate on task-level with no examples to train on. We therefore expand each task into approximately twenty queries by sourcing additional peer-reviewed publications that defined named subsets of the underlying dataset, and attaching one such subset to each query as its scope. A scope record specifies a named geographic, temporal, or sample-size slice together with a short description, a numeric size, and a citation to the publication that defined that slice. The processed data files are identical across queries within a task. The scope changes which subset of those files the system is asked to operate on. Scope variation between train and test queries forces the MAS optimizer to seek workflows that generalize across data samples rather than ones tuned to a specific subset.

Splits. The 2,067 queries are partitioned into 1,184 train, 391 validation, and 492 test queries, stratified by task so that every task has at least three test queries and a non-trivial train and validation pool.

## 3 Evaluation Framework

We evaluate whether MAS generation systems outperform single-agent baselines on real scientific data-driven tasks, and whether any gain justifies the added cost. This requires three components: a test-time grader that rewards real numerical computation rather than defensive defaults, a continuous training-time reward signal for MAS optimizers, and cost-aware metrics that distinguish better metric quality from broader query coverage. Additional details are provided in Appendix C.

## 3.1 Training Protocols

We evaluate every MAS method under two training protocols. In the per-task protocol, a method optimizes separately on each of the 100 tasks using only that task’s train and validation queries, producing 100 task-specific MAS instances for task-level MAS generation system. In the per-category protocol, a method optimizes on the union of train and validation queries from all tasks in a task category, producing 16 category-level MAS instances for task-level MAS generation system. Both protocols are evaluated on the same 492 held-out test queries (§2.2).

## 3.2 Evaluation Metrics

Our evaluation reports one absolute performance score and several comparative metrics relative to the single-agent baseline. Let M denote a generated MAS configuration, SA the single-agent baseline, T the set of tasks, and score<sub>X,t</sub> the per-task score of system X on task t.

Absolute performance score. For each task, we compute a success rate over realistic outputs and a pass rate over named scientific metrics. A query is treated as realistic only if it reports parsable finite metrics, includes at least one strictly positive numeric value, and does not use a −1 sentinel. The per-task score is

$$
\mathrm { s c o r e } _ { t } = \mathrm { s u c c e s s \_ r a t e } _ { t } \cdot \overline { { \mathrm { p a s s \_ r a t e } } } _ { t } ,
$$

where pass\_rate is the mean metric pass rate over realistic test queries, set to zero when no realistic output is produced. The system-level score is the sum over all 100 tasks, with maximum 100.

Win rate and mean score gain. We measure how often and by how much a MAS configuration improves over the single-agent baseline:

$$
\displaystyle \operatorname { w i n } _ { - } \mathrm { r a t e } ( M ) ~ = ~ \frac { 1 } { | T | } \sum _ { t \in T } { \cal k } [ { \tt s c o r e } _ { M , t } > \mathrm { s c o r e } _ { S A , t } ] , \quad \Delta _ { \mathrm { p e r f } } ( M ) ~ = ~ \frac { 1 } { | T | } \sum _ { t \in T } \bigl ( \mathrm { s c o r e } _ { M , t } - \mathrm { s c o r e } _ { S A , t } \bigr ) .
$$

Cost ratio. We report inference cost relative to the single-agent baseline using wall-clock time and, where available, token consumption:

$$
\rho _ { \mathrm { t i m e } } ( M ) = \frac { \bar { c } _ { M } ^ { \mathrm { t i m e } } } { \bar { c } _ { S A } ^ { \mathrm { t i m e } } } , \qquad \rho _ { \mathrm { t o k e n } } ( M ) = \frac { \bar { c } _ { M } ^ { \mathrm { t o k e n } } } { \bar { c } _ { S A } ^ { \mathrm { t o k e n } } } ,
$$

where $\bar { c } _ { X }$ is the mean per-query cost of system $X$

Efficiency. We summarize cost-adjusted improvement as mean score gain per unit of additional inference time:

$$
\eta ( M ) = \frac { \Delta _ { \mathrm { p e r f } } ( M ) } { \mathrm { m a x } \big ( \rho _ { \mathrm { t i m e } } ( M ) - 1 , 0 . 1 \big ) } .
$$

The denominator is clipped at 0.1 to avoid instability when the cost ratio is close to one. The same definition applies to token cost when token accounting is available.

Generalizability. To compare narrow specialization with broader transfer, we measure the mean per-task minus per-category score gap:

$$
\mathrm { g e n } ( M ) = { \frac { 1 } { | T | } } \sum _ { t \in T } \bigl ( \mathrm { s c o r e } _ { M ^ { \mathrm { t a s k } } , t } - \mathrm { s c o r e } _ { M ^ { \mathrm { c a t } } , t } \bigr ) ,
$$

where $M ^ { \mathrm { t a s k } }$ and $M ^ { \mathrm { c a t } }$ are the per-task and per-category variants of the same method. Positive values indicate that narrow task-specific optimization helps; near-zero or negative values indicate that category-level workflows transfer as well or better.

Together, these metrics distinguish absolute performance, breadth of improvement, cost, cost-adjusted efficiency, and training-granularity generalization. Because the 100 tasks span heterogeneous scientific domains and objectives, we also annotate each task with difficulty and four tag axes—capability dimension, pipeline stage, cognitive skill, and MAS challenge—for post-hoc diagnostic cuts. Full label definitions, counts, and examples are provided in Appendix C.4.

## 3.3 Training-time Reward Function

MAS optimizers require a scalar reward per candidate workflow, but scientific data analysis lacks the binary oracle used in coding benchmarks. We therefore use a continuous reference-aware verifier during training. For each reported metric, the reward is

$$
s = 0 . 1 0 \cdot s _ { \mathrm { s a n i t y } } ~ + ~ 0 . 4 0 \cdot s _ { \mathrm { t a r g e t } } ~ + ~ 0 . 5 0 \cdot s _ { \mathrm { r e f e r e n c e } } ,
$$

where $s _ { \mathrm { s a n i t y } }$ checks finite numeric validity, $s _ { \mathrm { t a r g e t } }$ checks whether the metric satisfies its inequality target, and $s _ { \mathrm { r e f e r e n c e } }$ measures closeness to a cached single-agent reference value, clipped to [0, 1] and polarity-adjusted for lower-is-better metrics. To avoid rewarding defensive defaults, zero outputs against non-zero references receive zero reward. When the single-agent reference is unavailable, we drop $s _ { \mathrm { r e f e r e n c e } }$ and renormalize the remaining terms. The reward replaces each method’s usual training-time scoring hook, while the prompt shown to the candidate workflow remains unchanged.

## 4 Experiments and Results

## 4.1 Experimental Setup

Backbone and baselines. All MAS generation methods and the primary single-agent baseline use GPT-5 with identical max-token and temperature settings, so any difference between systems is attributable to the workflow rather than the model. We additionally report baselines on GPT-5.2 and GPT-4o.

Methods and budgets. We evaluate five MAS generation methods selected to span the two major categories in the literature and a range of optimization mechanisms within them. Task-level methods optimize one workflow per problem and reuse it across queries within that problem; we include AFlow [Zhang et al., 2025b], GPTSwarm [Zhuge et al., 2024], and MetaAgent [Zhang et al., 2025c], which apply Monte Carlo tree search over operator graphs, REINFORCE-style edge-realization, and one-shot finite-state-machine generation, respectively. Query-level methods produce a fresh workflow per query through a meta-agent or an in-context controller; we include W4S [Nie et al., 2025] as a low-compute reinforcement-learned meta-agent and AutoAgents as a prompting-only Explorer baseline. Table 3 reports the per-task settings.

Test set. All scores are computed on the 492 held-out test queries with the absolute performance score and realistic filter.

## 4.2 Comparison with the Single-Agent Baseline

Does multi-agent specialization improve over a strong single agent on real scientific data analysis? Figure 2a reports the absolute performance score for all single-agent baselines and reward-function trained MAS configurations on the 492 held-out test queries. The GPT-5 baseline scores 11.0 out of 100. Nine of the ten MAS configurations exceed it, with the strongest reaching 32.0 (GPTSwarm per-category), 30.3 (GPTSwarm per-task), 29.4 (W4S per-task), 26.6 (AFlow per-task), and 26.0 (W4S per-category). The single exception is AutoAgents per-category at 8.2, primarily because that configuration is capped at five test queries per category and leaves most queries unreached.

The number is only partially revealing of the performance. The MAS advantage on the absolute score is not that generated workflows are uniformly more accurate on every scientific metric. It is that some trained workflows produce realistic numerical metrics on a larger fraction of queries. Conditional on producing realistic output, the strongest MAS configurations clear a similar share of inequality targets as the single-agent baseline, typically between 50% and 58%. The MAS gain is therefore primarily a coverage gain: the workflow more often reaches the point at which scientific metrics can be computed at all, rather than producing higher-quality metrics once that point is reached.

![](images/1c154770471f01b77dc63126430c6494b0b09b39175513870bac23910f9e9edf.jpg)

![](images/d608fcb67a1c654c2e6fd349102ed8a5650ebe384a769200fbb602251ae915ad.jpg)

Figure 2: Absolute performance and training-granularity diagnostics. (a) Absolute performance score with the dashed line means the GPT-5 score. (b) Generalization gap, where positive values mean per-task training outperforms per-category training.  
![](images/105ae637e70b72e6255527415b31656db1a61fcbc771b85933fc107f9bd185f8.jpg)

![](images/47e2beceb6450fe2353cb802b273dcbe9f295378eef6fd8b3fa962a52bd9f585.jpg)

![](images/f07f2dc5cf335d5cae42fa658bf2d7d29c52c54fc210d14b17fc6e2a2e7d8b0e.jpg)

![](images/e54361d30f07047a546367fadbdd3e731a96e7ca9036e889cbf521d02e122135.jpg)  
Figure 3: System-level metrics relative to GPT-5 one-turn. Task win rate measures how often the MAS improves a task’s absolute score; mean score gain measures average absolute improvement; time-cost ratio measures wall-clock inference cost; efficiency is gain per unit of extra time.

## 4.3 Cost-Performance Analysis

Does the inter-system gain justify the additional inference cost? The MAS configurations that beat the baseline pay between roughly 1.85 and 4.5 times the inference time, and the relative ordering of methods changes when this cost is taken into account. Figure 3 reports the four primary system-level metrics relative to the reference, and Figure 4a focuses on the cost-performance trade-off.

The strongest absolute-score configuration is GPTSwarm per-category, which gains +21.1 percentage points over the baseline at roughly 4.5× the inference time. GPTSwarm per-task achieves the highest task win rate (51%) at a similar cost ratio. W4S per-task is the most efficient point on the frontier: it gains +18.4 percentage points at only 1.85× the inference time, yielding the highest gain per unit of additional compute. AFlow per-task is the next most efficient configuration. AutoAgents per-category is dominated; its cap on test queries leaves most queries unreached and produces a negative mean improvement despite nonzero quality on the queries it does attempt.

![](images/43ca87aeda23c1f8cec13b9675203c6d9f123909078e843f3f7162055f5b1840.jpg)

![](images/3b833a80bf3cbe6e4d34c601cc31de89b4b6397f5af047d00d412006d4c0da34.jpg)  
Figure 4: Cost and coverage-quality diagnostics. (a) Cost-performance frontier relative to GPT-5 one-turn, where the upper-left region is preferable; orange circles indicate per-category training and green triangles indicate per-task training. (b) Realistic output rate versus mean pass rate conditional on realistic output for each MAS configuration; marker size grows with the number of tasks that have at least one realistic output.

The cost view changes the interpretation of MAS success. If the deployment objective is maximum absolute score and inference cost is not constrained, GPTSwarm is preferred. If the objective is improvement per unit of additional compute, W4S per-task is preferred. No single MAS configuration dominates on both axes, and any production claim of MAS superiority therefore depends on which side of this trade-off the deployment lives on.

## 4.4 Effect of Training Schema on Generalization

Does training a single workflow on the union of sibling tasks generalize as well as training a separate workflow per task? The two protocols of Section 3.1 answer different generalization questions. Per-task training specializes a workflow narrowly while per-category training asks one workflow to transfer across sibling tasks in the same category. Figure 2b reports the mean per-task minus per-category absolute-score gap, which is positive when narrow specialization wins and negative when category-level training transfers as well or better.

GPTSwarm and MetaAgent are near or below zero, meaning their category-level workflows generalize cleanly to held-out sibling tasks. AFlow and AutoAgents gain strongly from per-task specialization, suggesting that their workflows overfit or dilute when trained on a broader category pool. W4S sits between these regimes. The split is informative for MAS design: broader training pools help only when the induced workflow is abstract enough to transfer.

## 4.5 Coverage-Quality Decomposition and Stratified diagnostics.

To understand why MAS improves the absolute score, we decompose performance into realistic output rate and conditional pass rate. Figure 4b shows that conditional pass rates are relatively clustered, typically around 50–58%, while realistic output rates vary substantially across methods. Thus, the main MAS advantage is coverage: stronger workflows more often reach the point of producing realistic scientific metrics, rather than producing much better metrics once realistic outputs are obtained. Detailed method-level patterns and implications for future MAS design are provided in Appendix D.1.

Additional stratified analyses show that MAS gains persist across all three scientific domains and across Easy, Medium, and Hard tasks, with especially large gains on data-preparation, robustness, decomposition, and parallel-search tasks. Full domain-, difficulty-, and tag-level results are provided in Appendix D.2.

## 5 Conclusion

Scientific data analysis changes the rules under which generated multi-agent systems are evaluated. The bottleneck is not only reasoning depth or role decomposition; it is the absence of an oracle that can reward partial, real computation without rewarding trivial defaults. ST-Bench supplies that substrate through paper-grounded query pools, executable spatial-temporal datasets, realistic output filtering, and a reference-aware verifier. Our experiments show cautious upside: several generated workflows beat a one-turn GPT-5 baseline, but only under evaluation machinery that makes real data use visible. The benchmark’s main value is therefore diagnostic. It lets future work ask which MAS designs genuinely compute, which merely format plausible metrics, and which forms of workflow specialization are worth their added cost.

## References

Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, et al. Program synthesis with large language models. arXiv preprint arXiv:2108.07732, 2021.

Daniil A Boiko, Robert MacKnight, Ben Kline, and Gabe Gomes. Autonomous chemical research with large language models. Nature, 624(7992):570–578, 2023.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Josh Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya Sutskever, and Wojciech Zaremba. Evaluating large language models trained on code, 2021. URL https://arxiv.org/abs/2107.03374.

Ziru Chen, Shijie Chen, Yuting Ning, Qianheng Zhang, Boshi Wang, Botao Yu, Yifei Li, Zeyi Liao, Chen Wei, Zitong Lu, Vishal Dey, Mingyi Xue, Frazier N. Baker, Benjamin Burns, Daniel Adu-Ampratwum, Xuhui Huang, Xia Ning, Song Gao, Yu Su, and Huan Sun. Scienceagentbench: Toward rigorous assessment of language agents for data-driven scientific discovery, 2025. URL https://arxiv.org/abs/2410.05080.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

Dheeru Dua, Yizhong Wang, Pradeep Dasigi, Gabriel Stanovsky, Sameer Singh, and Matt Gardner. DROP: A reading comprehension benchmark requiring discrete reasoning over paragraphs. In Proc. ofNAACL, 2019.

Hongcheng Gao, Yue Liu, Yufei He, Longxu Dou, Chao Du, Zhijie Deng, Bryan Hooi, Min Lin, and Tianyu Pang. Flowreasoner: Reinforcing query-level meta-agents, 2025. URL https: //arxiv.org/abs/2504.15257.

Alireza Ghafarollahi and Markus J Buehler. Protagents: protein discovery via large language model multi-agent collaborations combining physics and machine learning. Digital Discovery, 3(7): 1389–1409, 2024.

Juraj Gottweis, Wei-Hung Weng, Alexander Daryin, Tao Tu, Anil Palepu, Petar Sirkovic, Artiom Myaskovsky, Felix Weissenberger, Keran Rong, Ryutaro Tanno, et al. Towards an ai co-scientist. arXiv preprint arXiv:2502.18864, 2025.

Ken Gu, Ruoxi Shang, Ruien Jiang, Keying Kuang, Richard-John Lin, Donghe Lyu, Yue Mao, Youran Pan, Teng Wu, Jiaqian Yu, Yikun Zhang, Tianmai M. Zhang, Lanyi Zhu, Mike A. Merrill, Jeffrey Heer, and Tim Althoff. Blade: Benchmarking language model agents for data-driven science. In The 2024 Conference on Empirical Methods in Natural Language Processing, 2024.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset, 2021. URL https://arxiv.org/abs/2103.03874.

Sirui Hong, Mingchen Zhuge, Jonathan Chen, Xiawu Zheng, Yuheng Cheng, Jinlin Wang, Ceyao Zhang, Zili Wang, Steven Ka Shing Yau, Zijuan Lin, Liyang Zhou, Chenyu Ran, Lingfeng Xiao, Chenglin Wu, and Jürgen Schmidhuber. MetaGPT: Meta programming for a multi-agent collaborative framework. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=VtmBAGCN7o.

Jinhui Hu, Changtao Deng, Qiuwen Zhang, and Aoxuan Pang. Physics-informed neural networks enhanced by data augmentation: a novel framework for robust soil moisture estimation using multi-source data fusion. Journal of Hydrology, page 134320, 2025a.

Shengran Hu, Cong Lu, and Jeff Clune. Automated design of agentic systems, 2025b. URL https://arxiv.org/abs/2408.08435.

Qian Huang, Jian Vora, Percy Liang, and Jure Leskovec. Mlagentbench: Evaluating language agents on machine learning experimentation. arXiv preprint arXiv:2310.03302, 2023.

Xiaowei Jia, Jared Willard, Anuj Karpatne, Jordan S Read, Jacob A Zwart, Michael Steinbach, and Vipin Kumar. Physics-guided machine learning for scientific discovery: An application in simulating lake temperature profiles. ACM/IMS Transactions on Data Science, 2(3):1–26, 2021.

Carlos E Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik Narasimhan. Swe-bench: Can language models resolve real-world github issues? arXiv preprint arXiv:2310.06770, 2023.

Divas Karimanzira. Mass conservative time-series gan for synthetic extreme flood-event generation: Impact on probabilistic forecasting models. Stats, 7(3):808–826, 2024.

Anuj Karpatne, Xiaowei Jia, and Vipin Kumar. Knowledge-guided machine learning: Current trend and future prospects. arXiv preprint arXiv:2403.15989, 2024.

Saeed Khaki, Liang Wang, and Sotirios V. Archontoulis. A cnn-rnn framework for crop yield prediction. Frontiers in Plant Science, 10:1750, 2020. doi: 10.3389/fpls.2019.01750.

F. Kratzert, D. Klotz, C. Brenner, K. Schulz, and M. Herrnegger. Rainfall–runoff modelling using long short-term memory (lstm) networks. Hydrology and Earth System Sciences, 22(11):6005–6022, 2018. doi: 10.5194/hess-22-6005-2018. URL https://hess.copernicus.org/articles/ 22/6005/2018/.

Guohao Li, Hasan Abed Al Kader Hammoud, Hani Itani, Dmitrii Khizbullin, and Bernard Ghanem. Camel: Communicative agents for "mind" exploration of large language model society, 2023. URL https://arxiv.org/abs/2303.17760.

Haoyang Liu, Yijiang Li, and Haohan Wang. Genomas: A multi-agent framework for scientific discovery via code-driven gene expression analysis. arXiv preprint arXiv:2507.21035, 2025.

Licheng Liu, Wang Zhou, Kaiyu Guan, Bin Peng, Shaoming Xu, Jinyun Tang, Qing Zhu, Jessica Till, Xiaowei Jia, Chongya Jiang, et al. Knowledge-guided machine learning can improve carbon cycle quantification in agroecosystems. Nature communications, 15(1):357, 2024.

Xiao Liu, Hao Yu, Hanchen Zhang, Yifan Xu, Xuanyu Lei, Hanyu Lai, Yu Gu, Hangliang Ding, Kaiwen Men, Kejuan Yang, et al. Agentbench: Evaluating llms as agents. arXiv preprint arXiv:2308.03688, 2023.

Chris Lu, Cong Lu, Robert Tjarko Lange, Jakob Foerster, Jeff Clune, and David Ha. The ai scientist: Towards fully automated open-ended scientific discovery. arXiv preprint arXiv:2408.06292, 2024.

Andres M. Bran, Sam Cox, Oliver Schilter, Carlo Baldassari, Andrew D White, and Philippe Schwaller. Augmenting large language models with chemistry tools. Nature machine intelligence, 6(5):525–535, 2024.

Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Shashank Gupta, Bodhisattwa Prasad Majumder, Katherine Hermann, Sean Welleck, Amir Yazdanbakhsh, and Peter Clark. Self-refine: Iterative refinement with self-feedback, 2023. URL https://arxiv.org/abs/2303.17651.

Bodhisattwa Prasad Majumder, Harshit Surana, Dhruv Agarwal, Bhavana Dalvi Mishra, Abhijeetsingh Meena, Aryan Prakhar, Tirth Vora, Tushar Khot, Ashish Sabharwal, and Peter Clark. DiscoveryBench: Towards data-driven discovery with large language models. arXiv, 2024.

Miro Miranda, Marcela Charfuelan, and Andreas Dengel. Exploring physics-informed neural networks for crop yield loss forecasting. arXiv preprint arXiv:2501.00502, 2024.

Andrew J. Newman, Kevin Sampson, Martyn Clark, A. Bock, Roland Viger, D. Blodgett, Nans Addor, and M. Mizukami. Camels: Catchment attributes and meteorology for large-sample studies, June 2022. URL https://doi.org/10.5065/D6MW2F4D.

Tung Nguyen, Johannes Brandstetter, Ashish Kapoor, Jayesh K Gupta, and Aditya Grover. Climax: A foundation model for weather and climate. arXiv preprint arXiv:2301.10343, 2023.

Fan Nie, Lan Feng, Haotian Ye, Weixin Liang, Pan Lu, Huaxiu Yao, Alexandre Alahi, and James Zou. Weak-for-strong: Training weak meta-agent to harness strong executors, 2025. URL https://arxiv.org/abs/2504.04785.

Dilli Paudel, Hilmy Baja, Ron van Bree, Michiel Kallenberg, Stella Ofori-Ampofo, Aike Potze, Pratishtha Poudel, Abdelrahman Saleh, Weston Anderson, Malte von Bloh, Andres Castellano, Oumnia Ennaji, Raed Hamed, Rahel Laudien, Donghoon Lee, Inti Luna, Dainius Masiliunas,¯ Michele Meroni, Janet Mumo Mutuku, Siyabusa Mkuhlani, Jonathan Richetti, Alex C. Ruane, Ritvik Sahajpal, Guanyuan Shuai, Vasileios Sitokonstantinou, Rogerio de Souza Noia Junior, Amit Kumar Srivastava, Robert Strong, Lily-belle Sweet, Petar Vojnovic, Allard de Wit, Maximilian´ Zachow, and Ioannis N. Athanasiadis. CY-Bench: A comprehensive benchmark dataset for subnational crop yield forecasting, 2024.

Markus Reichstein, Gustau Camps-Valls, Bjorn Stevens, Martin Jung, Joachim Denzler, Nuno Carvalhais, and F Prabhat. Deep learning and process understanding for data-driven earth system science. Nature, 566(7743):195–204, 2019.

Chenyang Shao, Dehao Huang, Yu Li, Keyu Zhao, Weiquan Lin, Yining Zhang, Qingbin Zeng, Zhiyu Chen, Tianxing Li, Yifei Huang, et al. Omniscientist: Toward a co-evolving ecosystem of human and ai scientists. arXiv preprint arXiv:2511.16931, 2025.

Yongliang Shen, Kaitao Song, Xu Tan, Wenqi Zhang, Kan Ren, Siyu Yuan, Weiming Lu, Dongsheng Li, and Yueting Zhuang. Taskbench: Benchmarking large language models for task automation. Advances in Neural Information Processing Systems, 37:4540–4574, 2024.

Yiming Sun, Shuo Chen, Shengyu Chen, Chonghao Qiu, Licheng Liu, Youmi Oh, Sparkle L Malone, Gavin McNicol, Qianlai Zhuang, Chris Smith, et al. X-methanewet: A cross-scale global wetland methane emission benchmark dataset for advancing science discovery with ai. In Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 1, pages 2782–2793, 2026.

Daniela Szwarcman, Sujit Roy, Paolo Fraccaro, Orsteinn Elí Gíslason, Benedikt Blumenstiel, Rinki Ghosal, Pedro Henrique De Oliveira, Joao Lucas de Sousa Almeida, Rocco Sedona, Yanghui Kang, et al. Prithvi-eo-2.0: A versatile multi-temporal foundation model for earth observation applications. IEEE Transactions on Geoscience and Remote Sensing, 2025.

Makoto Takamoto, Timothy Praditia, Raphael Leiteritz, Dan MacKinlay, Francesco Alesiani, Dirk Pflüger, and Mathias Niepert. Pdebench: An extensive benchmark for scientific machine learning, 2024. URL https://arxiv.org/abs/2210.07182.

Zhenhailong Wang, Shaoguang Mao, Wenshan Wu, Tao Ge, Furu Wei, and Heng Ji. Unleashing the emergent cognitive synergy in large language models: A task-solving agent through multi-persona self-collaboration, 2024. URL https://arxiv.org/abs/2307.05300.

Jared Willard, Xiaowei Jia, Shaoming Xu, Michael Steinbach, and Vipin Kumar. Integrating scientific knowledge with machine learning for engineering and environmental systems. ACM Computing Surveys, 55(4):1–37, 2022.

Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Beibin Li, Erkang Zhu, Li Jiang, Xiaoyun Zhang, Shaokun Zhang, Jiale Liu, Ahmed Hassan Awadallah, Ryen W White, Doug Burger, and Chi Wang. Autogen: Enabling next-gen llm applications via multi-agent conversation, 2023. URL https://arxiv.org/abs/2308.08155.

Yutaro Yamada, Robert Tjarko Lange, Cong Lu, Shengran Hu, Chris Lu, Jakob Foerster, Jeff Clune, and David Ha. The ai scientist-v2: Workshop-level automated scientific discovery via agentic tree search. arXiv preprint arXiv:2504.08066, 2025.

Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhutdinov, and Christopher D. Manning. HotpotQA: A dataset for diverse, explainable multi-hop question answering. In Conference on Empirical Methods in Natural Language Processing (EMNLP), 2018.

Runlong Yu, Chonghao Qiu, Robert Ladwig, Paul Hanson, Yiqun Xie, and Xiaowei Jia. Physicsguided foundation model for scientific discovery: An application to aquatic science. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 39, pages 28548–28556, 2025.

Guibin Zhang, Luyang Niu, Junfeng Fang, Kun Wang, Lei Bai, and Xiang Wang. Multi-agent architecture search via agentic supernet, 2025a. URL https://arxiv.org/abs/2502.04180.

Jiayi Zhang, Jinyu Xiang, Zhaoyang Yu, Fengwei Teng, Xionghui Chen, Jiaqi Chen, Mingchen Zhuge, Xin Cheng, Sirui Hong, Jinlin Wang, Bingnan Zheng, Bang Liu, Yuyu Luo, and Chenglin Wu. Aflow: Automating agentic workflow generation, 2025b. URL https://arxiv.org/abs/ 2410.10762.

Yaolun Zhang, Xiaogeng Liu, and Chaowei Xiao. Metaagent: Automatically constructing multi-agent systems based on finite state machines, 2025c. URL https://arxiv.org/abs/2507.22606.

Mingchen Zhuge, Wenyi Wang, Louis Kirsch, Francesco Faccio, Dmitrii Khizbullin, and Jürgen Schmidhuber. Gptswarm: Language agents as optimizable graphs. In Forty-first International Conference on Machine Learning, 2024.

## Technical Appendix

## A Related Work

## A.1 AI for Science

Machine learning has been widely adopted across scientific domains Kratzert et al. [2018], Khaki et al. [2020], Liu et al. [2024]. Although purely data-driven approaches can be accurate, they often suffer from limited interpretability, weak generalization, and outputs that violate real-world physical constraints Reichstein et al. [2019]. Knowledge-Guided Machine Learning (KGML) Willard et al. [2022], Karpatne et al. [2024], Jia et al. [2021] addresses these issues by integrating domain-specific knowledge into model design, with representative directions including physics-informed neural networks Hu et al. [2025a], Miranda et al. [2024], physics-constrained generative models Karimanzira [2024], and foundation Earth models Nguyen et al. [2023], Szwarcman et al. [2025], Yu et al. [2025]. These advances center on building individual models for specific problems, leaving open how autonomous agents should orchestrate full data-driven scientific workflows that respect domain knowledge, e.g., feature engineering, model selection, evaluation, uncertainty quantification, and exploratory analysis. ST-Bench targets this gap by curating tasks that probe each stage of the data-driven discovery pipeline across three Earth-science domains.

## A.2 Domain Scientific Agents

Beyond the general MAS frameworks discussed in Section 1, a parallel line of work develops domain-specialized scientific agents. Single-agent systems such as Coscientist Boiko et al. [2023] and ChemCrow M. Bran et al. [2024] pioneered tool-augmented reasoning for chemistry, while more recent multi-agent systems distribute responsibilities across specialized agents, including AI Co-Scientist Gottweis et al. [2025], Shao et al. [2025] for parallel hypothesis evaluation, ProtAgents Ghafarollahi and Buehler [2024] for protein discovery, GenoMAS Liu et al. [2025] for gene expression analysis, and The AI Scientist for end-to-end research automation Lu et al. [2024], Yamada et al. [2025]. These systems demonstrate the promise of role specialization, critical review, and multi-step planning for scientific tasks. However, each is evaluated on narrow case studies tailored to its target domain, leaving the relative contribution of multi-agent coordination unmeasured in any standardized way. ST-Bench complements these efforts by providing a controlled evaluation suite that decouples MAS coordination from domain-specific tooling.

## B Dataset

## B.1 Science Domains

Hydrology. CAMELS [Newman et al., 2022] dataset provides multi-decadal daily streamflow, meteorological forcing, and hundreds of static catchment attributes for over ∼700 minimally disturbed basins across the contiguous United States. We adapt 25 tasks reflecting core challenges studied in ML hydrology, including hydrological signature–based clustering, streamflow forecasting under climate and regional shift, signature regression, cross-basin transfer learning, and uncertainty stratification by basin elevation and humidity. CAMELS is known for strong heterogeneity across basins, sensitivity to nonstationarity, and difficulty of generalization to ungauged regions. These tasks require integrating static and dynamic data, handling long temporal dependencies, and constructing robust end-to-end analysis pipelines, making them a realistic test of whether MAS improves execution reliability beyond single-agent workflows.

Agriculture. We compile a domain we refer to as CropBench from two publicly available sources: a county-level USA Corn Belt yield dataset associated with the CNN-RNN crop yield framework of Khaki et al. [2020], and the multi-region sub-national yield benchmark assembled by AgML CY-Bench [Paudel et al., 2024]. Together they provide yield, management practice, soil composition, and weather records at the county-by-year level across multiple growing regions. We adapt 24 tasks, including yield prediction under management-regime shifts, weather-conditional yield clustering, model comparison across crop types, and stratified performance reporting by climate zone. CropBench tasks require disentangling confounding between management practices and yield outcomes, a recurring failure mode in naive tabular ML.

Wetland methane. X-MethaneWet [Sun et al., 2026] is a multi-site CH<sub>4</sub>-flux compilation drawn from FLUXNET-CH4, with eddy-covariance flux measurements and environmental drivers across northern wetland sites. We adapt 40 tasks, including FLUXNET QC-flag agreement after data cleaning, gap-filling under instrument failure, regional versus pan-Arctic transfer, and site-stratified evaluation. The methane domain is uniquely demanding: fluxes are small relative to instrument noise, quality-control flags are binary expert annotations rather than physical measurements, and the underlying biogeochemistry has strong site-specific behavior that does not transfer through a single global model.

General control set. We additionally include 11 non-domain tasks that operate on synthetic or general tabular data, such as clustering on isotropic Gaussian mixtures or feature-correlation on independent normals. We do not count General as a fourth domain. It serves as a negative control: a system that scores high on General but low on the three Earth science domains is failing the spatial-temporal task rather than the data science task.

## C Evaluation Framework

This appendix provides additional details for the evaluation framework in Section 3, including training protocols, metric definitions, the training-time reward function, and task annotation axes used for diagnostic analysis.

![](images/c75c72ba5e0b4cc2766c0346d4a7c4e3f0618c717771155934902acc22db1cfc.jpg)

![](images/14a9716b652f12195857b714615bae332effa322e2c0856d5b537d6d946fcd4c.jpg)  
Figure 5: ST-Bench distribution. Left: query counts by domain and split, with task counts annotated above each domain. Right: the 16 task categories used to construct the benchmark.

The core question we wish to answer thorough the evaluation is that whether MAS generation systems outperform single-agent baselines on real scientific problems, and if so by how much and at what cost. Answering it requires an evaluation framework that solves three problems specific to our setting. First, candidate systems produce open-ended numerical outputs rather than executable test results, so the test-time grader must reward real computation rather than reward defensive defaults. Second, MAS generation methods rely on optimizers that consume a scalar reward signal per candidate workflow, but scientific data analysis does not naturally provide such a signal. and scientific data analysis does not natively supply one.Third, a single composite score is not sufficient to reveal whether a workflow succeeds because it producesby achieving better metrics, because it coverscovering more queries, or only at a cost that erasesincurring costs that undermine its operational value. We design our evaluation framework around these three problems: a test-time grader with a realistic-output filter, a training-time verifier that delivers a continuous per-task reward signal, and a suite of MAS-specific evaluation metrics that diagnose performance along orthogonal dimensions.

## C.1 Training Protocols

MAS generation methods differ in how they train and produce agentic workflows. Task-level methods optimize one MAS instance per narrow task using a training set of similar queries. Query-level methods maintain broader archives or controllers and produce one workflow per query. To compare both categories fairly, we evaluate every method under two protocols. In the per-task protocol, the method runs its inner-loop optimization separately on each of the 100 tasks, using only that task’s train and validation queries, and produces 100 task-specific MAS instances. In the per-category protocol, the method optimizes on the union of train and validation queries from all tasks in a category, producing 16 category-level MAS instances. Both protocols evaluate on the same 492 held-out test queries from Section 2.2.

## C.2 Evaluation Metrics

A single number per system tells who ranked higher but not why, or at what cost. Our evaluation reports one absolute performance score per system together with six comparative metrics relative to the single-agent baseline. Throughout, M denotes a generated MAS configuration, SA denotes the single-agent baseline, T is the set of tasks, and score<sub>X,t</sub> is the per-task absolute score of system X on task t.

Absolute performance score. For each (task, query, metric) result, we check four conditions: parsability of the reported metrics, name match against a defined task metric, finite numeric validity, and satisfaction of the metric’s inequality target if the metric needs a threshold, such as a minimum number of clusters that defines a successful clustering task outcome. These collect into a per-task success rate and a per-metric pass rate. To prevent defensive defaults from bypassing inequality grading, we apply a realistic filter where a query has realistic output if its reported metrics include at least one strictly positive numeric value and no −1 sentinel. The per-task absolute performance score is

$$
\mathrm { s c o r e } _ { t } = \mathrm { s u c c e s s \_ r a t e } _ { t } \cdot \overline { { \mathrm { p a s s \_ r a t e } } } _ { t } ,
$$

where pass\_rate is the mean per-metric pass rate over the task’s realistic test queries (zero if the task has no realistic queries). The system’s headline absolute score is the sum over all 100 tasks, with maximum 100. The score penalizes systems that fail entirely on a task more harshly than systems that succeed sparsely across many tasks, and ignores defensive zeros entirely.

Win rate and mean score gain. The fraction of tasks on which the MAS configuration improves over the single-agent baseline, paired with the average absolute improvement, is defined as

$$
\operatorname { w i n } _ { - } \mathrm { r a t e } ( M ) = \frac { 1 } { | T | } \sum _ { t \in T } | \mathcal { k } \big [ \mathrm { s c o r e } _ { M , t } > \mathrm { s c o r e } _ { S A , t } \big ] , \qquad \Delta _ { \mathrm { p e r f } } ( M ) = \frac { 1 } { | T | } \sum _ { t \in T } \bigl ( \mathrm { s c o r e } _ { M , t } - \mathrm { s c o r e } _ { S A , t } \bigr ) .
$$

Read together, the two metrics describe the shape of the improvement rather than only its sign. A high win rate paired with a small $\Delta _ { \mathrm { p e r f } }$ identifies a system that improves many tasks by little; the opposite pairing identifies one that lifts a few tasks substantially. Either pattern can be desirable depending on the deployment context.

Cost ratio. The ratio of MAS inference cost to single-agent cost, measured along both wall-clock time and token consumption where the latter is available, is defined as

$$
\rho _ { \mathrm { t i m e } } ( M ) = \frac { \bar { c } _ { M } ^ { \mathrm { t i m e } } } { \bar { c } _ { S A } ^ { \mathrm { t i m e } } } , \qquad \rho _ { \mathrm { t o k e n } } ( M ) = \frac { \bar { c } _ { M } ^ { \mathrm { t o k e n } } } { \bar { c } _ { S A } ^ { \mathrm { t o k e n } } } ,
$$

where ${ \bar { c } } _ { X }$ is the mean per-query cost of system X. We report $\rho _ { \mathrm { t i m e } }$ for every configuration because wall-clock time is logged uniformly across systems. Token accounting is consistently available for the single-agent baseline but not for every MAS adapter in the current release; we therefore report $\rho _ { \mathrm { t o k e n } }$ where it is available and treat it as a partially measured axis.

Efficiency. The mean score gain per unit of additional inference cost is defined as

$$
\eta ( M ) = \frac { \Delta _ { \mathrm { p e r f } } ( M ) } { \operatorname* { m a x } \bigl ( \rho _ { \mathrm { t i m e } } ( M ) - 1 , 0 . 1 \bigr ) } .
$$

The denominator is clipped at 0.1 to avoid division by zero on configurations whose cost ratio approaches one. Efficiency answers a question that single-agent benchmarks rarely have to ask: per unit of additional compute over the cheapest baseline, how much improvement does the MAS deliver? The same definition applies with $\rho _ { \mathrm { t o k e n } }$ when token accounting is available.

Generalizability. The mean per-task minus per-category composite-score gap is defined as

$$
\operatorname { g e n } ( M ) = { \frac { 1 } { | T | } } \sum _ { t \in T } \bigl ( \operatorname { s c o r e } _ { M ^ { \operatorname { t a s k } } , t } - \operatorname { s c o r e } _ { M ^ { \operatorname { c a t } } , t } \bigr ) ,
$$

where $M ^ { \mathrm { t a s k } }$ and $M ^ { \mathrm { c a t } }$ are the per-task and per-category variants of the same method. A near-zero or negative value indicates that the broader category-level workflow transfers to sibling tasks at least as well as a task-specific workflow, while a strongly positive value indicates the method gains substantially from narrow specialization.

Scalability across backbones. This axis evaluates whether the architectural advantages of a generated MAS persist when the underlying LLM is replaced. To operationalize this, we select representative MAS configurations and evaluate their relative gains over single-agent baselines across multiple foundational models. Rather than an exhaustive sweep of all permutations, this representative sampling validates backbone scalability as a core, measurable dimension within the ST-Bench framework, establishing a standardized blueprint for future system audits.

Parallelism. Whether a generated MAS decomposes its work into subtasks that can execute concurrently. Wall-clock inference time relative to total accumulated agent time provides a proxy: a configuration whose wall-clock cost is substantially below its summed agent cost is exploiting parallelism, whereas a configuration whose wall-clock cost approximately equals its summed agent cost is sequential. The current release reports the wall-clock proxy; a full audit of internal subtask parallelization is left to future work.

These metrics are designed to be read together. A configuration with high win rate, high $\Delta _ { \mathrm { p e r f } }$ , low ρ, and high η is unambiguously preferred. A configuration that scores high on $\Delta _ { \mathrm { p e r f } }$ but with $\rho$ in the multiples is preferred only when absolute improvement matters more than its cost. The intraand inter-system comparisons of §4 use this metric suite to support both leaderboard and diagnostic claims.

Because the 100 tasks span heterogeneous scientific domains and data-driven objectives, an aggregate score alone cannot reveal which kinds of tasks benefit most from MAS specialization, which is especially important for domain experts using MAS generation systems. We further annotate each task with difficulty and four orthogonal tag axes—capability dimension, pipeline stage, cognitive skill, and MAS challenge—so that results can be sliced by task structure. Full label definitions, counts, and examples are provided in Appendix C.4.

## C.3 Training-time Reward Function

MAS generation methods drive their optimization with a scalar reward per candidate workflow. Coding benchmarks supply this reward through unit tests; scientific data analysis does not, and the absolute performance score of §C.2 is unsuitable as a training reward because all-zero and −1 defaults clear several inequality targets trivially and the optimizer would converge on them. We therefore define a continuous reference-aware reward function used in place of the absolute score during training.

For each defined metric of a query, the reward is the weighted sum

$$
s = 0 . 1 0 \cdot s _ { \mathrm { s a n i t y } } + 0 . 4 0 \cdot s _ { \mathrm { t a r g e t } } + 0 . 5 0 \cdot s _ { \mathrm { r e f e r e n c e } } ,
$$

where $s _ { \mathrm { s a n i t y } }$ is 1 for a finite numeric value, $s _ { \mathrm { t a r g e t } }$ is 1 if the value satisfies the metric’s inequality target, and s<sub>reference</sub> is the proportion of the cached single-agent reference value the candidate reaches, clipped to [0, 1] and polarity-flipped for lower-is-better metrics. A single override removes the defensive-zero attractor: a candidate value of zero against a non-zero reference returns score zero for that metric, regardless of the other components. For approximately seven percent of queries where the single-agent reference is missing, we drop $s _ { \mathrm { r e f e r e n c e } }$ and renormalize the remaining weights to 0.40 and 0.60. The per-query reward is the average of s across the query’s metrics.

The reward function is wired in evaluator-side: it replaces the per-query scoring hook that each method ordinarily calls during training, while leaving the prompt the candidate sees unchanged. This mirrors how each method consumes a binary oracle on coding benchmarks, and ensures any improvement we observe is attributable to better candidate selection rather than to leaking reference information into the prompt.

## C.4 Per-task Analytical Labels for Evaluation Cuts

The system metrics of §3.2 measure properties of the system; this subsection complements them with axes that slice the task pool itself. Together they form a 2D analysis grid: which property of the system are we measuring, on which subset of tasks. The labels we describe here are not part of the raw task definition (§2); they are evaluation tooling that we add for analysis.

Difficulty. Each task carries a difficulty label set during annotation: 17 Easy, 49 Medium, and 34 Hard. Difficulty reflects the analytical complexity of the source publication. Easy tasks have a single self-contained metric and one canonical method. Hard tasks combine multiple metrics that depend on each other or require iterative refinement of intermediate artifacts. The difficulty mix is deliberately weighted toward Medium and Hard because MAS generation methods are unlikely to add value on Easy tasks where a single agent already succeeds; we want headline numbers dominated by tasks where MAS specialization is more likely to matter.

Four-axis groupings for MAS-targeted analysis. Each task additionally carries a four-axis tag set designed to expose orthogonal MAS-evaluation cuts.

Capability dimension (Modeling 31, Discovery 28, Optimization 23, Robustness 18) records the kind of capability the task exercises: building predictive models, discovering structure in data, solving an explicit objective under constraints, or maintaining performance under noise and shift.

Table 3: Per-task training budgets.
<table><tr><td>Method</td><td>Inner loop</td><td>Per-task budget</td></tr><tr><td>MetaAgent</td><td>one-shot FSM generation</td><td>1 round (no iterative selection)</td></tr><tr><td>AutoAgents</td><td>per-query Explorer</td><td>1 round, 3 test queries cap</td></tr><tr><td>GPTSwarm</td><td>edge-realization optimization</td><td>3 realizations × 2 val queries</td></tr><tr><td>AFlow</td><td>MCTS over operator graphs</td><td>2 rounds, 2 val sample, 1 validation round</td></tr><tr><td>W4S</td><td>iterative meta-agent proposals</td><td>3 iterations × 2 val queries</td></tr></table>

Pipeline stage (Modeling 33, Data Analysis 25, Evaluation 23, Data Preparation 19) records where in the data-science pipeline the task sits, so we can ask which pipeline stages most benefit from MAS role decomposition. This grouping lines up with how MAS designers describe the internal role decomposition of a generated MAS (a “data-prep agent” or a “model-builder agent”), and lets us probe directly whether handing a particular stage to a dedicated agent inside the MAS pays off on real scientific data.

Cognitive skill (Predictive 39, Diagnostic 33, Prescriptive 16, Descriptive 12) records the analytical style the task asks for, following the standard analytics taxonomy: what is happening (Descriptive), why is it happening (Diagnostic), what will happen (Predictive), what should we do (Prescriptive). The skill axis is informative for MAS evaluation because Diagnostic and Prescriptive tasks typically require coordinating multiple sub-analyses, which is exactly what a generated MAS can in principle do better than a single agent.

MAS challenge (Synthesis 35, Parallel Search 23, Long-horizon Planning 23, Error Recovery 14, Decomposition 5) records the kind of MAS-specific competency the task most stresses. Tasks tagged “Decomposition” require breaking the problem into independent subproblems before any agent can act. Tasks tagged “Parallel Search” admit many candidate approaches that benefit from concurrent exploration. Tasks tagged “Long-horizon Planning” require maintaining state and intermediate results across many steps. Tasks tagged “Synthesis” require aggregating partial results from multiple sub-analyses into a coherent final answer. Tasks tagged “Error Recovery” deliberately surface failure modes that the system must catch and re-try. We use the MAS-challenge axis in §4 to ask whether the methods that score well overall do so because they handle a specific type of MAS challenge particularly well, or because they perform uniformly across all five.

## D Experiments and Results

## D.1 Coverage-Quality Decomposition Details

The absolute performance score is the product of two factors that we can examine separately: the realistic output rate, defined as the fraction of test queries on which the system produces a nondefensive metrics dictionary, and the mean pass rate conditional on realistic output, defined as the share of inequality targets cleared on those queries. Figure 4b plots each MAS configuration on these two axes.

Two patterns dominate the figure. First, conditional pass rates cluster within a relatively narrow band, typically between 50% and 58%, while realistic output rates vary much more widely across methods and protocols. Second, the configurations that achieve the highest absolute scores in §4.2 do so by raising the realistic output rate rather than the conditional quality. W4S per-task reaches the highest realistic rate at 38.2%; GPTSwarm and AFlow configurations cluster at intermediate realistic rates; the AutoAgents per-category configuration sits at the lowest realistic rate, consistent with its query-cap-induced coverage shortfall. The narrow band of conditional pass rates indicates that, once a system reaches the point of producing realistic metrics, the share of inequality targets it clears is comparable across methods; the differentiator is whether the system reaches that point at all.

This decomposition refines the finding that MAS specialization improves the absolute score primarily through coverage rather than through higher metric quality. It also identifies where future MAS design effort is most likely to translate directly into absolute-score gains. Improvements that raise the realistic output rate—data-loading robustness, exception-handling discipline that does not collapse to defensive defaults, and end-to-end execution over staged files—move the system along the dominant axis of variation. Improvements aimed at fine-grained metric quality run into the narrow conditional band and yield smaller absolute-score gains.

![](images/b9767532006ba66d3503903a06f5dba516133b5af32c688b6bf07e9f161f1ebd.jpg)  
Figure 6: Task-subset cuts by scientific domain and declared difficulty. Each MAS bar is the best verifier-trained configuration of that training granularity within the group.

## D.2 Stratified Results: Domain, Tags, and Difficulty

The result in Section 4.2 averages over heterogeneous scientific tasks. We therefore stratify the same composite test-split score by three metadata axes: scientific domain, task tags, and declared difficulty. For each group, the GPT-5 score is the mean per-task composite score in that group. The MAS bars and table entries select the best verifier-trained MAS configuration within the group.

Domain. Figure 6a focuses on the three scientific domains in ST-Bench. MAS improves all three, but not uniformly. The largest gain appears on MethaneWet: GPT-5 scores only 8.2 on average, while GPTSwarm reaches 40.1 under per-category training and 40.7 under per-task training. CAMELS also benefits strongly, with GPTSwarm per-category and AFlow per-task both near 29. CropBench is the weakest MAS domain: GPTSwarm per-category reaches 23.6 and W4S per-task reaches 21.5, still roughly doubling the GPT-5 one-turn baseline but with a smaller margin than the other scientific domains. This suggests that MAS generation is especially useful when the task family requires repeated data loading, modeling, and metric computation over similar structured workflows, as in MethaneWet and CAMELS.

Tag system. Table 4 stratifies tasks by the benchmark’s four tag axes: capability dimension, pipeline stage, cognitive skill, and MAS challenge. The strongest gains occur in groups that should naturally reward structured multi-agent decomposition. Data preparation gains 35.2 pp, robustness gains 34.7 pp, decomposition gains 34.0 pp, and parallel search gains 33.0 pp. These are precisely the settings where a generated workflow can separate file inspection, transformation, modeling, and validation steps. The smallest gains are still positive, but more modest: error recovery gains 17.1 pp, pipeline-stage modeling gains 19.0 pp, and synthesis gains 20.3 pp. This pattern is useful for benchmark interpretation: MAS generation is not just "better on average"; it is most valuable on tags that expose coordination, search, and data-handling structure.

Difficulty. Figure 6b shows that MAS gains persist across Easy, Medium, and Hard tasks. The declared difficulty labels are not monotonic with one-turn baseline performance: Easy tasks have the lowest GPT-5 one-turn score at 4.3, while Hard tasks have the highest at 17.4. This is likely because the labels describe scientific task complexity, whereas the one-turn baseline is dominated by execution brittleness and whether the first attempt reaches a realistic metric computation. MAS improves all levels by roughly 22 pp under the best configuration. The best per-category MAS is slightly stronger than the best per-task MAS at every difficulty level in this aggregate, suggesting that broader category-level workflows often capture reusable structure even when individual tasks are labeled difficult.

Table 4: Best MAS configuration by task-tag group. Scores are mean per-task composite score percentages. Gain is best MAS minus GPT-5 one-turn in percentage points; win rate is the fraction of tasks in the group where the selected MAS configuration beats GPT-5.
<table><tr><td>Tag axis</td><td>Tag value</td><td>Tasks</td><td>GPT-5</td><td>Best MAS</td><td>Gain</td><td>Win rate</td></tr><tr><td>Capability</td><td>Discovery</td><td>28</td><td>11.2</td><td>GPTSwarm cat 38.6</td><td>27.4</td><td>57.1%</td></tr><tr><td>Capability</td><td>Modeling</td><td>31</td><td>9.9</td><td>W4S cat 32.4</td><td>22.5</td><td>54.8%</td></tr><tr><td>Capability</td><td>Optimization</td><td>23</td><td>18.0</td><td>GPTSwarm task 41.1</td><td>23.1</td><td>56.5%</td></tr><tr><td>Capability</td><td>Robustness</td><td>18</td><td>3.5</td><td>W4S task 38.2</td><td>34.7</td><td>61.1%</td></tr><tr><td>Cognitive</td><td>Descriptive</td><td>12</td><td>10.0</td><td>GPTSwarm cat 41.7</td><td>31.7</td><td>58.3%</td></tr><tr><td>Cognitive</td><td>Diagnostic</td><td>33</td><td>10.2</td><td>W4S task 35.5</td><td>25.3</td><td>57.6%</td></tr><tr><td>Cognitive</td><td>Predictive</td><td>39</td><td>8.7</td><td>W4S cat 38.5</td><td>29.8</td><td>59.0%</td></tr><tr><td>Cognitive</td><td>Prescriptive</td><td>16</td><td>19.0</td><td>W4S task 44.0</td><td>25.1</td><td>68.8%</td></tr><tr><td>MAS challenge</td><td>Decomposition</td><td>5</td><td>6.0</td><td>AFlow task 40.0</td><td>34.0</td><td>40.0%</td></tr><tr><td>MAS challenge</td><td>Error recovery</td><td>14</td><td>8.0</td><td>MetaAgent cat 25.1</td><td>17.1</td><td>57.1%</td></tr><tr><td>MAS challenge</td><td>Long-horizon planning</td><td>23</td><td>16.2</td><td>W4S cat 47.2</td><td>31.0</td><td>60.9%</td></tr><tr><td>MAS challenge</td><td>Parallel search</td><td>23</td><td>6.1</td><td>GPTSwarm task 39.1</td><td>33.0</td><td>60.9%</td></tr><tr><td>MAS challenge</td><td>Synthesis</td><td>35</td><td>12.7</td><td>W4S task 33.0</td><td>20.3</td><td>57.1%</td></tr><tr><td>Pipeline</td><td>Data analysis</td><td>25</td><td>11.7</td><td>GPTSwarm cat 42.4</td><td>30.7</td><td>60.0%</td></tr><tr><td>Pipeline</td><td>Data preparation</td><td>19</td><td>7.0</td><td>W4S task 42.2</td><td>35.2</td><td>73.7%</td></tr><tr><td>Pipeline</td><td>Evaluation</td><td>23</td><td>7.1</td><td>W4S cat 35.9</td><td>28.8</td><td>60.9%</td></tr><tr><td>Pipeline</td><td>Modeling</td><td>33</td><td>15.4</td><td>GPTSwarm task 34.4</td><td>19.0</td><td>51.5%</td></tr></table>