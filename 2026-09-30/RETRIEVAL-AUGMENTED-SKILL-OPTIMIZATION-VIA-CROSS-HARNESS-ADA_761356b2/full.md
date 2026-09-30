# RETRIEVAL-AUGMENTED SKILL OPTIMIZATION VIA CROSS-HARNESS ADAPTATION

Jaewon Chu<sup>1</sup>, Ji Soo Lee<sup>2</sup>, Jihwan Park<sup>2</sup>, Dohwan Ko<sup>1</sup>, Jeehye Na<sup>2</sup>, Seunghun Lee<sup>2</sup>, Taehoon Lee<sup>2</sup>, Minseo Yoon<sup>2</sup>, Minseok Joo<sup>1</sup>, Yunyang Xiong<sup>3</sup>, Hyunwoo J. Kim<sup>2∗</sup> <sup>1</sup>Korea University, <sup>2</sup>KAIST, <sup>3</sup>Meta AI

allonsy07@korea.ac.kr, hyunwoojkim@kaist.ac.kr

## ABSTRACT

An agent skill is a reusable, actionable natural-language artifact that guides an agent to perform a task effectively under a given harness. Recent studies have explored the optimization of agent skills, contributing to a growing collection of publicly available skills spanning diverse tasks, domains, and harnesses. Despite millions of publicly shared skills, existing skill optimization methods largely overlook this accumulated knowledge, instead relying solely on expensive agent rollouts to iteratively refine skills for a target task. To address this, we propose Retrieval-Augmented Skill Optimization (RASO), a framework that leverages an external skill corpus as prior knowledge throughout skill optimization. RASO retrieves relevant knowledge from existing skills and adapts it to the target task and harness via Cross-Harness Adaptation, accounting for mismatches in both domain and harness. RASO comprises two complementary stages: Retrieval-Augmented Skill Initialization (RASI) constructs a knowledge-grounded initial skill without requiring agent rollouts, while Retrieval-Augmented Skill Update (RASU) iteratively refines the skill by retrieving external knowledge guided by execution feedback. Across four agent benchmarks and two models, extensive experiments show that RASO consistently outperforms baselines without retrieval-augmented skill initialization and updating.

## 1 INTRODUCTION

Large language models (LLMs) are widely deployed as agents within execution harnesses that define available tools, file access, and scoring procedures (Yao et al., 2023; Yang et al., 2024b). In these settings, performance depends not only on the parametric knowledge of the underlying model but also on the procedural policies governing task execution (Wang et al., 2024; 2025), commonly referred to as agent skills (Anthropic, 2025; Li et al., 2026c). An agent skill is a reusable, actionable natural-language artifact that specifies how an agent should accomplish tasks under a given harness. Unlike policies encoded in model weights, agent skills are expressed as text, making them readily inspectable, auditable, and transferable across models without modification.

To automatically construct effective agent skills, skill optimization has recently attracted growing attention (Wang et al., 2026; Xia et al., 2026; Ding et al., 2026; Chen et al., 2026). Existing methods iteratively refine skills based on agent experience collected through training-task rollouts under a given harness. Through such refinement, millions of skills have been publicly shared (Destefanis et al., 2026), collectively encoding procedural knowledge accumulated across diverse tasks, models, and harnesses. Despite this extensive knowledge, existing methods largely rely on the agent’s own experience to optimize each skill (Yang et al., 2026; Alzubi et al., 2026; Ni et al., 2026; Tang et al., 2026), requiring costly rollouts for iterative refinement. As a result, leveraging an external skill corpus as a prior for automatic skill optimization remains largely unexplored.

To this end, we propose Retrieval-Augmented Skill Optimization (RASO), a skill optimization framework that leverages an external skill corpus as prior knowledge. As in Fig. 1, RASO comprises two complementary stages, Retrieval-Augmented Skill Initialization (RASI) and Retrieval-Augmented Skill Update (RASU). RASI leverages prior knowledge from an external skill corpus to construct an effective initial skill directly from task and harness descriptions, without using any expensive agent rollouts. Upon observing a failure during task execution, RASU identifies the missing knowledge underlying the failure and retrieves relevant content from the corpus to address it. However, retrieved skills are originally written for specific tasks and harnesses, potentially encoding domain-specific assumptions or referencing tools unavailable in the target environment. To address this domain and harness mismatch, RASO introduces Cross-Harness Adaptation, a shared operation employed by both RASI and RASU. Rather than directly incorporating retrieved content, this operation abstracts away source-domain-specific instruction and re-expresses the underlying procedure in terms of the objects, commands, and units supported by the target harness.

![](images/40159f3e8de324a893c5f9258bf332aa35b39729d8594f3ca9b40fc0889a9655.jpg)  
Figure 1: Comparison of RASO to prior skill optimization. Existing skill optimization methods (left) either generate an initial skill from the LLM alone or directly reuse retrieved skills, then refine them mainly from execution feedback. However, retrieved skills often come from mismatched harnesses; on SpreadsheetBench 92.9% originate from a different harness (right, top). RASO (middle) addresses this by retrieving relevant external skill knowledge and adapting it to the target domain and execution harness through Cross-Harness Adaptation, supporting both skill initialization (RASI) and skill update (RASU). Overall, our RASO results in substantially stronger performance than both direct retrieval and retrieval-free optimization methods.

We evaluate RASO through extensive experiments on four agent benchmarks, OfficeQA (Opsahl-Ong et al., 2026), SpreadsheetBench (Ma et al., 2024), ALFWorld (Shridhar et al., 2021), and Web-Shop (Yao et al., 2022), with two LLMs, Qwen-3.5-9B (Qwen Team, 2026) and GPT-5.6-Luna (OpenAI, 2026). Specifically, RASI synthesizes strong initial skills that outperform both retrieved skills and retrieval-free initialization without requiring agent rollouts, providing a strong initialization for subsequent optimization. Moreover, RASO with RASU consistently outperforms strong skill optimization algorithms, including TextGrad (Yuksekgonul et al., 2025), GEPA (Agrawal et al., 2026), SkillOpt (Yang et al., 2026), and WikiSkill (Tang et al., 2026).

Our contributions are as follows:

• We propose Retrieval-Augmented Skill Optimization (RASO), a framework that leverages an external skill corpus as prior knowledge for both skill initialization and update. Through its two complementary stages, RASI and RASU, RASO grounds skill construction in retrieved procedural knowledge rather than relying on the optimizer’s parametric knowledge.

• We introduce Cross-Harness Adaptation, a shared operation that adapts retrieved procedural knowledge to the vocabulary of the target domain and harness. This enables knowledge transfer across diverse domains and harnesses without requiring domain- or harnessmatched skills in the corpus.

• Across four benchmarks and two models, we demonstrate that RASI improves performance without requiring agent rollouts, while RASU further refines the skill by retrieving the missing knowledge guided by execution feedback. Our analysis further shows that these gains are associated with adapting diverse external knowledge rather than relying on particular source documents.

## 2 RELATED WORKS

Agent Skills. Agent skills are reusable textual documents that provide procedural guidance for accomplishing tasks within an execution harness (Anthropic, 2025). Represented as text rather than model parameters, skills are easy to inspect, edit, share, and reuse across models without retraining. Prior work has made this knowledge explicit through executable skill libraries (Wang et al., 2024), reusable workflows (Wang et al., 2025), reflections and insights (Shinn et al., 2023; Zhao et al., 2024), or reasoning and procedural memories (Ouyang et al., 2026; Fang et al., 2026), while large public collections such as GitSkills (Destefanis et al., 2026) now make this knowledge available at scale. However, not every skill is useful: its benefit depends on both the procedural knowledge it contains and how well that knowledge matches the target task and harness. Accordingly, two main approaches have emerged: learning procedural knowledge from the agent’s own execution experience and reusing knowledge from external sources.

Skill Optimization from Execution Experience. A major line of work improves agent skill from its own experience, treating the skill as a textual decision variable optimized with execution feedback on the target task. Methods that optimize prompts or contexts using LLM-generated feedback (Pryzant et al., 2023; Yang et al., 2024a; Chu et al., 2026; Zhang et al., 2026) are directly applicable to skill optimization, most notably TextGrad (Yuksekgonul et al., 2025), which backpropagates textual feedback, and GEPA (Agrawal et al., 2026), which evolves prompts by reflecting on execution traces. Skill-specific optimizers follow the same recipe: SkillOpt (Yang et al., 2026) edits a skill from rollout trajectories and accepts only edits that improve validation performance, WikiSkill (Tang et al., 2026) compiles agent experience into persistent knowledge, and others distill or refine skills from trajectories (Ni et al., 2026; Chen et al., 2026; Moll et al., 2026; Wang et al., 2026; Alzubi et al., 2026; Ding et al., 2026). However, these methods primarily optimize skills from observed execution experience, limiting their exploration of procedural knowledge beyond what can be inferred from the agent’s own rollouts. In contrast, our framework supplements execution feedback with procedural knowledge retrieved from an external skill corpus throughout optimization.

Skill Construction from External Knowledge. Procedural knowledge that an agent cannot infer from its own experience often already exists: in shared skills, documentation, and the web, motivating a growing line of work that draws on such external knowledge. Retrieval-based methods, such as SkillRouter (Zheng et al., 2026), select relevant skills from large libraries through improved retrievers (Li et al., 2026a; Miao et al., 2026) or by organizing libraries into graphs and execution structures (Meng et al., 2026; Liu et al., 2026; Fu et al., 2026; Li et al., 2026b; Xia et al., 2026), with dedicated benchmarks for skill retrieval (Kang et al., 2026; Su et al., 2026). Since retrieved skills may refer to different tasks, tools, or actions, other methods adapt external experience to the target interface (Tang et al., 2025) or compile external resources into reusable skills (Pan et al., 2026; Yan et al., 2026). However, these approaches either require repeated retrieval and adaptation for each task instance during test time or use external knowledge only during skill initialization, without further leveraging it as target-task experience accumulates. In contrast, we leverage external knowledge throughout the skill optimization process, using it both to construct the initial skill and to further improve the skill during iterative updates.

## 3 PROBLEM FORMULATION

We consider a frozen language model M acting as an agent through an execution harness $h ,$ which defines the available tools, file access, and observation interface. A skill s is a natural-language artifact provided as the agent’s context to guide task completion under a given harness. Executing the agent on a task instance $x$ with skill s yields a trajectory $h ( { \mathcal { M } } , x , s )$ , which is associated with a reward $r ( \cdot ) \in [ 0 , 1 ]$ by a benchmark-specific evaluator. When a reference answer is available, the evaluator compares the agent’s final output against $\mathbf { i t } ,$ and otherwise uses the environment’s native success criterion. We refer to each agent execution and its corresponding evaluation as a rollout. Given disjoint task splits $\mathcal { D } _ { \mathrm { t r a i n } } , \mathcal { D } _ { \mathrm { v a l } }$ , and $\mathcal { D } _ { \mathrm { t e s t } } .$ , candidate skills are constructed from rollouts on $\mathcal { D } _ { \mathrm { t r a i n } }$ and selected based on performance on $\mathcal { D } _ { \mathrm { v a l } }$ as:

$$
\begin{array} { r } { s ^ { \star } = \underset { s } { \arg \operatorname* { m a x } } \mathbb { E } _ { x \sim \mathcal { D } _ { \mathrm { v a l } } } \left[ r \left( h ( \mathcal { M } , x , s ) \right) \right] . } \end{array}\tag{1}
$$

The selected skill $s ^ { \star }$ is then evaluated on the held-out $\mathcal { D } _ { \mathrm { t e s t } }$

![](images/14c51f0cf1086a397191fe2a911d3e64870e188a876c76c9b47236c9d7133fa5.jpg)  
Figure 2: Overview of RASO. RASI (top, without rollouts) generates retrieval queries from the task and harness descriptions, retrieves the top-K skill sections per query from an external corpus, and adapts them into grounded lessons via Cross-Harness Adaptation to synthesize the initial skill s<sub>0</sub>. RASU (bottom, with rollouts) starts from s<sub>0</sub>, executes the current skill, and generates retrieval queries from rollout results. The retrieved knowledge is adapted into grounded lessons via Cross-Harness Adaptation and incorporated into skill.md to iteratively refine the skill. Each subsequent iteration performs new rollouts using the committed skill.

Since both M and h remain fixed throughout optimization, the only decision variable is the naturallanguage skill s. We include the construction of the initial skill itself in the skill optimization problem, rather than assuming that an initial skill is externally supplied. We therefore define skill optimization to encompass both skill initialization, which constructs an initial skill from task and harness descriptions without requiring any rollouts, and skill update, which refines the skill using results from training rollouts.

## 4 RASO: RETRIEVAL-AUGMENTED SKILL OPTIMIZATION

In this section, we introduce Retrieval-Augmented Skill Optimization (RASO), a framework that evolves agent skills through two complementary stages: skill initialization and skill update, illustrated in Figure 2. Both stages share a common knowledge retrieval and adaptation mechanism but differ in the rollout evidence available to guide skill optimization. Section 4.1 describes the shared mechanism, which retrieves procedural knowledge from a large-scale skill corpus (Destefanis et al., 2026) and adapts it to the target task and harness through cross-harness adaptation. Section 4.2 introduces Retrieval-Augmented Skill Initialization (RASI), which constructs an initial skill using only the target task, harness description, and retrieved knowledge, without requiring agent rollouts. Section 4.3 introduces Retrieval-Augmented Skill Update (RASU), which leverages execution feedback and retrieved knowledge to iteratively refine the skill.

## 4.1 SKILL RETRIEVAL AND CROSS-HARNESS ADAPTATION

We first perform fine-grained, section-level skill retrieval from an external skill corpus to acquire relevant prior knowledge. We then introduce Cross-Harness Adaptation, a mechanism that bridges the gap between source and target domains and harnesses by transforming retrieved knowledge into actionable guidance tailored to the target task and harness. Skill retrieval and adaptation constitute a shared pipeline used by both RASI (Section 4.2) and RASU (Section 4.3).

Section-level skill retrieval. We first divide skill documents into heading-delimited sections to enable fine-grained retrieval. Since external skill documents are developed for diverse tasks and workflows, retrieving entire documents may introduce irrelevant content and favor documents with similar overall objectives over those containing relevant procedural sections. Section-level retrieval instead enables us to identify relevant procedural knowledge while improving the signal-to-noise ratio of retrieved content. Given the resulting section-level corpus C, a query-generation agent formulates a query q and retrieves the top-K most relevant sections using BM25 (Robertson & Zaragoza, 2009), i.e., $\mathbf { \dot { \mathcal { S } } } _ { q } = \mathbf { B } \mathbf { M } 2 5 ( q , \mathcal { C } , K )$

Cross-Harness Adaptation. We employ an adaptation agent, denoted by $\mathcal { A } _ { \mathrm { a d a p t a t i o n } } .$ , to adapt retrieved knowledge to the target task and harness. Let $c _ { i }$ denote a requirement needing external knowledge to be resolved, such as a specific task procedure in RASI or a textual gradient in RASU. Given $c _ { i } ,$ the task description $T ,$ , the harness description H, and the corresponding top- $K$ retrieved sections $\boldsymbol { S } _ { \boldsymbol { q } _ { i } }$ , the agent produces a concise, actionable lesson $\ell _ { i } \colon$

$$
\ell _ { i } = \mathcal { A } _ { \mathrm { a d a p t a t i o n } } \left( c _ { i } , T , H , S _ { q _ { i } } \right) .\tag{2}
$$

Here, the lesson $\ell _ { i }$ serves as a refined knowledge snippet that directly guides the agent to handle the requirement $c _ { i }$ within the target task and harness.

To ensure that each lesson addresses the given requirement and remains valid within the target harness, the adaptation process follows three principles: (1) remove domain-specific nouns and omit procedures without counterparts in the target harness, (2) focus exclusively on requirement $c _ { i } .$ excluding unrelated issues, and (3) preserve specific claims about tool or parameter behavior only when corroborated by H, prioritizing correctness within the target harness over potentially inaccurate specificity. Consequently, each lesson is expressed using the objects, commands, and units of the target task and harness.

## 4.2 RETRIEVAL-AUGMENTED SKILL INITIALIZATION (RASI)

We introduce Retrieval-Augmented Skill Initialization (RASI), which constructs an initial skill for a target task under a given harness without requiring agent rollouts. Performed once at the beginning of skill optimization, RASI comprises four sequential steps: (1) procedure and query generation, (2) section-level skill retrieval, (3) Cross-Harness Adaptation, and (4) skill initialization.

Procedure and query generation. Given the task description $T$ and harness description H, the query-generation agent $\mathcal { A } _ { \mathrm { q u e r y \_ i n i t } }$ generates a set of requirement-query pairs $( c _ { i } , q _ { i } )$

$$
\{ ( c _ { i } , q _ { i } ) \} _ { i = 1 } ^ { M } = \mathcal { A } _ { \mathrm { q u e r y \_ i n i t } } ( T , H ) ,\tag{3}
$$

where M denotes the number of generated pairs. Here, each $c _ { i }$ represents a specific task procedure or harness constraint $( e . g .$ , multi-turn budget management), and $q _ { i }$ is the retrieval query created to search for external skills that address $c _ { i }$

Section-level skill retrieval and Cross-Harness Adaptation. Given the generated requirementquery pairs, RASI applies the shared retrieval and adaptation pipeline described in Section $4 . 1$ For each query $q _ { i }$ , BM25 retrieves the top-K relevant sections $\boldsymbol { S } _ { \boldsymbol { q } _ { i } }$ from the corpus $\mathcal { C } .$ The adaptation agent A<sub>adaptation</sub> then transforms these sections into a grounded lesson $\begin{array} { r l } { \ell _ { i } } & { { } = } \end{array}$ $\bar { \mathcal { A } } _ { \mathrm { a d a p t a t i o n } } ( \bar { c _ { i } } , T , H , S _ { q _ { i } } ^ { ' } )$ . The resulting lessons form $\mathcal { L } _ { \mathrm { i n i t } } = \{ \ell _ { 1 } , \ldots , \bar { \ell } _ { M } \}$ , which are used for subsequent skill initialization.

Skill initialization. Finally, a skill-initializer agent $\mathcal { A } _ { \mathrm { s k i l l \_ i n i t } }$ synthesizes the initial skill $s _ { 0 }$ from the task description T, harness description $H$ , identified procedures and requirements $\{ c _ { i } \} _ { i = 1 } ^ { M }$ , and grounded lessons ${ \mathcal { L } } _ { \mathrm { i n i t } }$ . The agent integrates each lesson $\ell _ { i }$ into the execution step corresponding to $c _ { i } ,$ yielding:

$$
s _ { 0 } = \mathcal { A } _ { \mathrm { s k i l l \_ i n i t } } ( T , H , \{ c _ { i } \} _ { i = 1 } ^ { M } , \mathcal { L } _ { \mathrm { i n i t } } ) .\tag{4}
$$

By incorporating retrieved and adapted knowledge before environment interaction, RASI provide a knowledge-grounded initial skill for subsequent optimization without consuming search rollouts.

## 4.3 RETRIEVAL-AUGMENTED SKILL UPDATE (RASU)

Here, we introduce Retrieval-Augmented Skill Update (RASU) that iteratively refines the current skill $s _ { t }$ using execution feedback from agent rollouts. Complementing the rollout-free initialization of RASI, RASU identifies specific failure modes observed in agent trajectories and retrieves relevant external knowledge to address them. Given trajectories generated by the execution agent using $s _ { t } ,$ each RASU iteration comprises four sequential steps: (1) textual gradient and query generation, (2) section-level skill retrieval, (3) Cross-Harness Adaptation, and (4) skill update.

Textual gradient and query generation. We first sample a minibatch of tasks from $\mathcal { D } _ { \mathrm { t r a i n } }$ and execute agent rollouts using the current skill $s _ { t } . \mathrm { A }$ gradient-and-query generator agent $A _ { \mathrm { q u e r y . } }$ \_update then analyzes the failed trajectories to identify failure mode. For each failure modes, the agent generates a textual gradient $\delta _ { i }$ based on its parametric knowledge and a targeted retrieval query $q _ { i }$ to acquire relevant external knowledge:

$$
\begin{array} { r } { \{ ( q _ { i } , \delta _ { i } ) \} _ { i = 1 } ^ { N } = \mathcal { A } _ { \mathrm { q u e r y } _ { - } \mathrm { u p d a t e } } ( T , H , s _ { t } , \{ \tau _ { j } \} ) , } \end{array}\tag{5}
$$

where $N$ denotes the number of generated query and gradient pairs, and $\{ \tau _ { j } \}$ denotes the rollout trajectories.

Section-level skill retrieval and Cross-Harness Adaptation. Given the failure-driven retrieval queries, RASU applies the shared retrieval and adaptation pipeline described in Section 4.1. This process yields a set of grounded lessons $\mathcal { L } _ { \mathrm { u p d a t e } } = \{ \ell _ { 1 } , \dots , \ell _ { N } \}$ for subsequent skill update.

Skill update. Finally, a skill-updater agent $\mathcal { A } _ { \mathrm { s k i l l } }$ generates a candidate skill $s _ { t + 1 }$ by integrating the current skill $s _ { t } ,$ , textual gradients $\{ \delta _ { i } \} _ { i = 1 } ^ { N }$ , and trajectory-grounded lessons $\mathcal { L } _ { \mathrm { { u p d a t e } } } \mathrm { { : } }$

$$
s _ { t + 1 } = \mathcal { A } _ { \mathrm { s k i l l \_ u p d a t e } } \big ( s _ { t } , \{ \delta _ { i } \} _ { i = 1 } ^ { N } , \mathcal { L } _ { \mathrm { u p d a t e } } \big ) .\tag{6}
$$

The updater refines the current skill $s _ { t }$ to candidate skill $s _ { t + 1 }$ based on the textual gradients and grounded lessons. The candidate skill $s _ { t + 1 }$ is accepted only if it outperforms $s _ { t }$ on the validation set $\mathcal { D } _ { \mathrm { v a l } }$ , with $s _ { t }$ retained otherwise. This pipeline is repeated for a fixed number of iterations, progressively refining the skill through execution feedback and retrieved prior knowledge.

## 5 EXPERIMENT

We evaluate RASO using two target LLMs: GPT-5.6-Luna (OpenAI, 2026) and Qwen-3.5- 9B (Qwen Team, 2026). We refer to the model that executes tasks as the target model and the model that generates or updates skill text as the optimizer model. Unless otherwise specified, we use the same model for both roles across all skill optimization methods. For skill retrieval, we use GitSkills (Destefanis et al., 2026), an external skill corpus spanning diverse domains and harnesses. Our evaluation covers four benchmarks with diverse interaction settings: OfficeQA (Opsahl-Ong et al., 2026), SpreadsheetBench (Ma et al., 2024), ALFWorld (Shridhar et al., 2021), and Web-Shop (Yao et al., 2022). For skill initialization, we compare RASI against three strategies: (1) No Skill, where the agent operates without an initialized skill, (2) SkillRouter (Zheng et al., 2026), which retrieves a relevant skill from the external corpus, and (3) Retrieval-Free Skill Initialization (RFSI), where the target LLM generates an initial skill directly from the task and harness descriptions without access to the external skill corpus. For iterative skill optimization, we compare RASO against TextGrad (Yuksekgonul et al., 2025), GEPA (Agrawal et al., 2026), SkillOpt (Yang et al., 2026), and WikiSkill (Tang et al., 2026). All experiments are conducted with three random seeds, and we report the mean performance over seeds.

## 5.1 MAIN RESULTS

Skill initialization. We first evaluate the quality of skills produced by different initialization methods, i.e., before any subsequent agent rollout or skill optimization, for both GPT-5.6-Luna and Qwen-3.5-9B in Table 1. Across both models and all four benchmarks, RASI consistently achieves the highest performance. With GPT-5.6-Luna, RASI improves over Retrieval-Free Skill Initialization (RFSI) by +5.63 on OfficeQA (45.74 vs. 40.11), +4.77 on Spreadsheet (49.17 vs. 44.40), +3.24 on ALFWorld (72.64 vs. 69.40), and +1.17 on WebShop (45.06 vs. 43.89). The margin is substantially larger over the No Skill and SkillRouter baselines, particularly on OfficeQA, where both achieve only 11.44. This advantage also holds for the smaller Qwen-3.5-9B backbone, where RASI outperforms RFSI by +5.61 on OfficeQA, +2.74 on Spreadsheet, +5.72 on ALFWorld, and +10.73 on WebShop. SkillRouter, which retrieves external skills without Cross-Harness Adaptation, underperforms even the No Skill baseline on several benchmarks, which suggests direct reuse can introduce irrelevant or mismatched procedural knowledge when the retrieved content is not adapted to the target task and harness. Overall, these results show that combining external knowledge retrieval with LLM-based adaptation, as in RASI, yields a more effective initialization than either retrieval alone or retrieval-free LLM-based skill generation, providing a stronger skill initialization for subsequent optimization.

Table 1: Agent performance with GPT-5.6-Luna (OpenAI, 2026) and Qwen-3.5-9B (Qwen Team, 2026). We separate skill initialization from skill update; initialization-only rows report performance at 0 rollouts. RFSI denotes retrieval-free skill initialization. Results are mean ± standard error over three random seeds.
<table><tr><td colspan="4"></td><td colspan="3">Benchmark</td></tr><tr><td>Model</td><td>Method</td><td>OfficeQA</td><td></td><td>Spreadsheet</td><td>ALFWorld</td><td>WebShop</td></tr><tr><td rowspan="11">GPT-5.6-Luna</td><td colspan="5">Skill initialization (without rollout)</td><td></td></tr><tr><td>No skill</td><td> $1 1 . 4 4 \pm 0 . 7 0$ </td><td> $3 2 . 9 8 \pm 0 . 1 2$ </td><td> $6 4 . 4 3 \pm 1 . 3 8$ </td><td></td><td> $4 3 . 5 6 \pm 1 . 0 1$ </td></tr><tr><td>SkillRouter</td><td></td><td> $1 1 . 4 4 \pm 0 . 9 7$ </td><td> $3 3 . 2 1 \pm 0 . 9 4$ </td><td> $5 5 . 9 7 \pm 1 . 1 4$ </td><td> $4 2 . 2 0 \pm 0 . 3 7$ </td></tr><tr><td>RFSI</td><td> $4 0 . 1 1 \pm 0 . 5 8$ </td><td></td><td> $4 4 . 4 0 \pm 2 . 7 4$ </td><td> $6 9 . 4 0 \pm 1 . 5 5$ </td><td> $4 3 . 8 9 \pm 1 . 1 7$ </td></tr><tr><td>RASI</td><td> ${ \bf 4 5 . 7 4 \pm 1 . 4 0 }$ </td><td></td><td> ${ \bf 4 9 . 1 7 \pm 1 . 5 2 }$ </td><td> $7 2 . 6 4 \pm 1 . 6 3$ </td><td> ${ \bf 4 5 . 0 6 \pm 0 . 4 0 }$ </td></tr><tr><td>Skill update (with rollout)</td><td colspan="5"></td></tr><tr><td>TextGrad</td><td></td><td> $4 3 . 8 0 \pm 1 . 2 7$ </td><td> $4 9 . 6 4 \pm 3 . 0 9$ </td><td> $7 0 . 6 5 \pm 0 . 6 6$ </td><td> $4 2 . 3 2 \pm 3 . 1 3$ </td></tr><tr><td>GEPA</td><td></td><td> $4 3 . 9 9 \pm 1 . 4 0$ </td><td> $5 4 . 5 3 \pm 1 . 4 6 $ </td><td> $7 1 . 1 4 \pm 1 . 0 8$ </td><td> $4 5 . 5 4 \pm 0 . 3 5$ </td></tr><tr><td></td><td>SkillOpt</td><td> $4 5 . 5 4 \pm 2 . 0 5$ </td><td> $5 7 . 0 2 \pm 3 . 6 0$ </td><td> $7 2 . 6 4 \pm 0 . 9 0$ </td><td> $4 5 . 1 7 \pm 0 . 2 3$ </td></tr><tr><td>RASO</td><td>WikiSkill</td><td> $4 1 . 4 7 \pm 1 . 1 8$ </td><td> $4 7 . 1 4 \pm 3 . 9 8$ </td><td> $7 1 . 1 4 \pm 0 . 6 6$ </td><td> $4 4 . 4 0 \pm 0 . 6 4$ </td></tr><tr><td></td><td></td><td> ${ \bf 4 9 . 0 3 \pm 1 . 8 5 }$ </td><td> ${ \bf 6 3 . 3 3 \pm 2 . 6 9 }$ </td><td> $\mathbf { 7 4 . 1 3 \pm 1 . 0 0 }$ </td><td> ${ \bf 4 6 . 6 1 \pm 0 . 1 1 }$ </td></tr><tr><td rowspan="11">Qwen-3.5-9B</td><td colspan="5">Skill initialization (without rollout)</td><td></td></tr><tr><td>No skill</td><td> $3 3 . 1 4 \pm 1 . 2 1$ </td><td> $2 8 . 8 1 \pm 0 . 6 0$ </td><td> $3 3 . 0 9 \pm 0 . 6 6$ </td><td></td><td> $1 6 . 2 7 \pm 0 . 2 5$ </td></tr><tr><td>SkillRouter</td><td></td><td> $3 4 . 1 1 \pm 0 . 5 1$ </td><td> $2 3 . 4 5 \pm 0 . 4 8$ </td><td> $3 1 . 3 4 \pm 1 . 2 9$ </td><td> $9 . 4 2 \pm 0 . 4 7$ </td></tr><tr><td>RFSI</td><td> $3 4 . 8 9 \pm 2 . 9 3$ </td><td> $2 7 . 7 4 \pm 2 . 5 6$ </td><td></td><td> $4 2 . 0 4 \pm 2 . 8 7$ </td><td> $1 2 . 5 4 \pm 2 . 5 6$ </td></tr><tr><td>RASI</td><td> ${ \bf 4 0 . 5 0 \pm 0 . 9 7 }$ </td><td></td><td> ${ \bf 3 0 . 4 8 \pm 0 . 7 8 }$ </td><td> $\mathbf { 4 7 . 7 6 \pm 3 . 5 3 }$ </td><td> $2 3 . 2 7 \pm { \bf 0 . 9 8 }$ </td></tr><tr><td>Skill update (with rollout)</td><td colspan="5"></td></tr><tr><td>TextGrad</td><td></td><td> $3 6 . 8 2 \pm 1 . 5 9$ </td><td> $2 5 . 9 5 \pm 0 . 6 3$ </td><td> $4 1 . 0 4 \pm 2 . 2 8$ </td><td> $1 0 . 9 6 \pm 4 . 4 4$ </td></tr><tr><td>GEPA</td><td></td><td> $3 6 . 4 3 \pm 1 . 9 7$ </td><td> $2 4 . 0 5 \pm 1 . 5 6$ </td><td> $4 3 . 5 3 \pm 1 . 9 9$ </td><td> $1 2 . 8 0 \pm 2 . 2 0$ </td></tr><tr><td> $\mathrm { \bf S k i l l O p t }$ </td><td></td><td> $3 7 . 2 1 \pm 1 . 2 1$ </td><td> $2 9 . 5 2 \pm 2 . 2 1$ </td><td> $4 3 . 0 3 \pm 3 . 1 7$ </td><td> $1 3 . 3 6 \pm 1 . 7 0$ </td></tr><tr><td>WikiSkill</td><td></td><td> $3 7 . 4 0 \pm 3 . 0 3$ </td><td> $2 9 . 7 6 \pm 1 . 8 0$ </td><td> $4 2 . 7 9 \pm 2 . 9 3$ </td><td> $1 3 . 4 3 \pm 2 . 0 1$ </td></tr><tr><td>RASO</td><td></td><td> $4 2 . 2 5 \pm 1 . 5 2$ </td><td> ${ \bf 3 1 . 5 5 \pm 0 . 3 1 }$ </td><td> ${ \bf 5 1 . 0 0 \pm } 2 . 4 5$ </td><td> $2 4 . 7 3 \pm 0 . 4 1$ </td></tr></table>

Skill Update We compare RASO with existing skill optimization methods in Table 1. Following the conventional skill optimization setup, existing methods start from RFSI, where an initial skill is LLM-generated without rollouts or access to an external corpus, and subsequently refine it using their respective rollout-based optimization procedures. RASO overall outperforms existing skill optimization methods across both backbones. For GPT-5.6-Luna, RASO improves over the strongest competing method by +3.49 on OfficeQA (49.03 vs. 45.54), +6.31 on SpreadsheetBench (63.33 vs. 57.02), +1.49 on ALFWorld (74.13 vs. 72.64), and +1.07 on WebShop (46.61 vs. 45.54). Similarly, with Qwen-3.5-9B, RASO improves over the strongest competing method by +4.85 on OfficeQA (42.25 vs. 37.40), +1.79 on SpreadsheetBench (31.55 vs. 29.76), +7.47 on ALFWorld (51.00 vs. 43.53), and +11.30 on WebShop (24.73 vs. 13.43). These results show that RASO benefits from both retrieval-augmented initialization and subsequent retrieval-augmented skill refinement.

## 5.2 ANALYSIS

Ablation studies. We conduct an ablation study of the RASO components in Table 2. Relative to the RFSI + RFSU baseline (retrieval-free settings), RASU improves performance from 40.70 to 47.56 (+6.86 points) on OfficeQA and from 51.67 to 61.07 (+9.40 points) on SpreadsheetBench. These gains demonstrate the benefit of incorporating retrieved and adapted external knowledge during iterative skill refinement. Similarly, replacing only the initialization component with RASI improves performance from 40.70 to 45.93 (+5.23 points) on OfficeQA and from 51.67 to 58.45 (+6.78 points) on SpreadsheetBench, demonstrating the benefit of retrieval-augmented initialization under the same subsequent update procedure. Combining both components leads to the strongest performance, reaching 49.03 on OfficeQA and 63.33 on SpreadsheetBench, corresponding to improvements of +8.33 and +11.66 points over the baseline. These results indicate that retrieval-augmented initialization and updating provide complementary gains within RASO.

Table 2: Ablation on the initialization (Init.) and update components with GPT-5.6-Luna.
<table><tr><td colspan="2">Method</td><td colspan="2">Benchmark</td></tr><tr><td>Init.</td><td>Update</td><td>OfficeQA</td><td>Spreadsheet</td></tr><tr><td>RFSI</td><td>RFSU</td><td> $4 0 . 7 0 \pm 1 . 6 7$ </td><td> $5 1 . 6 7 \pm 2 . 0 3$ </td></tr><tr><td>RFSI</td><td>RASU</td><td> $4 7 . 5 6 \pm 1 . 8 5$ </td><td> $6 1 . 0 7 \pm 2 . 1 5$ </td></tr><tr><td>RASI</td><td>RFSU</td><td> $4 5 . 9 3 \pm 1 . 5 8$ </td><td> $5 8 . 4 5 \pm 1 . 7 8$ </td></tr><tr><td>RASI</td><td>RASU</td><td> ${ \bf 4 9 . 0 3 \pm 1 . 8 5 }$ </td><td>一  ${ \bf 6 3 . 3 3 \pm 2 . 6 9 }$  </td></tr></table>

Table 4: Update methods under a fixed RASI initialization with GPT-5.6-Luna. The highlighted row is RASO.
<table><tr><td colspan="2">Method</td><td colspan="2">Benchmark</td></tr><tr><td>Init.</td><td>Update</td><td>OfficeQA</td><td>Spreadsheet</td></tr><tr><td rowspan="4">RASI</td><td>TextGrad</td><td> $4 5 . 7 4 \pm 1 . 4 0$   $4 3 . 4 1 \pm 1 . 9 7$ </td><td> $4 9 . 1 7 \pm 1 . 5 2$   $5 7 . 9 7 \pm 3 . 2 1$ </td></tr><tr><td>GEPA</td><td> $4 5 . 7 4 \pm 1 . 7 2$ </td><td> $5 8 . 6 9 \pm 0 . 8 3$ </td></tr><tr><td>SkillOpt</td><td> $4 6 . 5 1 \pm 2 . 1 0$ </td><td> $6 0 . 2 4 \pm 1 . 5 2$ </td></tr><tr><td>WikiSkill</td><td> $4 7 . 0 9 \pm 2 . 4 2$ </td><td> $5 5 . 0 0 \pm 4 . 7 6$ </td></tr><tr><td rowspan="4"></td><td>RASU</td><td></td><td></td></tr><tr><td></td><td> ${ \bf 4 9 . 0 3 \pm 1 . 8 5 }$ </td><td> ${ \bf 6 3 . 3 3 \pm 2 . 6 9 }$  </td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr></table>

Table 3: Effect of adaptation with GPT-5.6-Luna. RASU used RASI with adaptation as an initial.
<table><tr><td colspan="2"></td><td colspan="2">Benchmark</td></tr><tr><td></td><td>Method Adaptation</td><td></td><td>OfficeQA Spreadsheet</td></tr><tr><td>RASI</td><td>×</td><td> $4 1 . 8 6 \pm { 1 . 5 3 }$ </td><td> $4 1 . 4 3 \pm 2 . 8 3$ </td></tr><tr><td>RASI</td><td>√</td><td> ${ \bf 4 5 . 7 4 \pm 1 . 4 0 }$ </td><td> ${ \bf 4 9 . 1 7 \pm 1 . 5 2 }$ </td></tr><tr><td>RASU</td><td>×</td><td> $4 6 . 7 0 \pm 2 . 0 5$ </td><td> $5 7 . 8 6 \pm 2 . 9 2$ </td></tr><tr><td>RASU</td><td></td><td> ${ \bf 4 9 . 0 3 \pm 1 . 8 5 }$ </td><td> ${ \bf 6 3 . 3 3 \pm 2 . 6 9 }$ </td></tr></table>

![](images/8824393ae4e450e3f60038b03da05e1ae55743b8c974e0b3125852cb640e52a2.jpg)  
Number of retrieved sections (K)  
Figure 3: Effect of the number of retrieved sections K with GPT-5.6-Luna. The dashed line marks our choice of $K = 5 ,$ , and shading denotes standard error.

Effect of Cross-Harness Adaptation. In Table 3, we examine the effect of Cross-Harness Adaptation in both RASI and RASU. For RASI, applying the adaptation improves performance from 41.86 to 45.74 (+3.88 points) on OfficeQA and from 41.43 to 49.17 (+7.74 points) on Spreadsheet-Bench. A similar benefit is observed during skill updating. Under the same RASI initialization, incorporating the adaptation into RASU improves performance from 46.70 to 49.03 (+2.33 points) on OfficeQA and from 57.86 to 63.33 (+5.47 points) on SpreadsheetBench. These consistent gains across both stages demonstrate the importance of adapting retrieved knowledge from diverse domains and harnesses to the target task and execution harness for effective skill initialization and iterative refinement.

Comparison with baselines under RASI initialization. Table 4 compares different skill update methods under the same RASI initialization, thereby isolating the effect of the update procedure. RASU achieves the highest performance on both benchmarks, reaching 49.03 on OfficeQA and 63.33 on SpreadsheetBench. Compared with the strongest alternative update method on each benchmark, RASU improves performance by +1.94 points over WikiSkill on OfficeQA (47.09 to 49.03) and by +3.09 points over SkillOpt on SpreadsheetBench (60.24 to 63.33). Relative to RASI initial ization without subsequent updates, RASU yields gains of +3.29 and +14.16 points on OfficeQA and SpreadsheetBench, respectively. These results show that, under the same retrieval-augmented initialization, RASU provides more effective iterative skill refinement than existing update methods.

Effect of retrieval size. Figure 3 analyzes the effect of the number of retrieved sections K for both RASI and RASU. Performance generally improves as K increases from 1 to 5, with the highest performance achieved at $K \ = \ 5$ on both OfficeQA and SpreadsheetBench. The gains are particularly pronounced for RASU, indicating that access to multiple relevant sections is useful when refining skills based on rollout feedback. Further increasing K to 10 slightly degrades performance for both RASI

![](images/863d74a1d1dd96af666e7075142d234b39db2a787f78f6fb5dc291babbca6c7d.jpg)  
Fraction of skill corpus  
Figure 4: Effect of skill corpus size with GPT-5.6-Luna. Dashed lines mark 0% (no corpus), and shading denotes standard error.

![](images/f86cc65dd21a8a5cf8cf540e5689b1182fa431c66766aff93c08cba8a8b12be7.jpg)  
Figure 5: Qualitative example of Cross-Harness Adaptation. RASI retrieves a skill from a mismatched domain and harness (satellite imagery with numpy/raster arrays ) and adapts its procedure into the target spreadsheet harness. We compare the resulting lessons, initial skills, and generated code with (top) and without (bottom) the top-1 of the five retrieved sections given as input. Highlights mark the differences between the two runs, which are reflected in the generated code and its output. Without the top-1 skill, the model overlooks the missing group, causing an index shift.

and RASU, suggesting that retrieving additional sections may introduce redundant or less relevant information. Accordingly, we set K = 5 for all experiments to balance knowledge coverage against retrieval noise.

Effect of skill corpus size. Figure 4 studies the effect of external skill corpus size for RASI and RASU. Enabling retrieval from only 1% of the corpus already yields a clear performance improvement over no-retrieval counterparts (0%; dashed line), revealing that even a small amount of retrieved external knowledge is beneficial. As the available corpus grows, performance generally continues to improve, with the largest gains observed on SpreadsheetBench. Overall, these results suggest that retrieval itself provides an immediate benefit, while increasing corpus size further im proves the opportunity to retrieve useful knowledge.

Qualitative results. Figure 5 presents a SpreadsheetBench task that requires computing the average of qualifying rows for each department. For the given data, department B has no qualifying rows; hence, the expected output for its cell is blank. The top-1 retrieved section comes from an urban remote-sensing skill whose domain and harness differ from the target. Through Cross-Harness Adaptation, RASI rewrites the source’s rule of skipping empty groups as “treat the output as missing,” and the generated code accordingly leaves B’s cell blank. When this section is removed, the generated code groups only the filtered rows, causing C’s mean to shift into B’s cell. This example suggests that cross-harness adaptation transfers the procedural structure and intent of retrieved sections while adapting them to the target harness format.

## 6 CONCLUSION

We propose Retrieval-Augmented Skill Optimization (RASO), which leverages an external skill corpus as prior knowledge to support both skill initialization and updating. Rather than directly reusing retrieved skills, RASO applies Cross-Harness Adaptation to transfer relevant procedural knowledge to the target task and execution harness. RASO combines Retrieval-Augmented Skill Initialization (RASI), which constructs a strong initial skill without agent rollouts, with Retrieval-Augmented Skill Update (RASU), which further refines the skill by retrieving knowledge guided by execution feedback. Across four agent benchmarks and two model backbones, RASO consistently outperforms retrieval-free skill optimization baselines, demonstrating the value of incorporating external skill knowledge at both initialization and update.

## REFERENCES

Lakshya A Agrawal, Shangyin Tan, Dilara Soylu, Noah Ziems, Rishi Khare, Krista Opsahl-Ong, Arnav Singhvi, Herumb Shandilya, Michael J Ryan, Meng Jiang, Christopher Potts, Koushik Sen, Alex Dimakis, Ion Stoica, Dan Klein, Matei Zaharia, and Omar Khattab. GEPA: Reflective prompt evolution can outperform reinforcement learning. In ICLR, 2026.

Salaheddin Alzubi, Noah Provenzano, Jaydon Bingham, Christian Alexander Calvo, Weiyuan Chen, and Tu Vu. EvoSkill: Automated skill discovery for multi-agent systems. In COLM, 2026.

Anthropic. Equipping agents for the real world with Agent Skills, October 2025. URL https: //www.anthropic.com/engineering/equipping-agents-for-the-real-w orld-with-agent-skills.

Kunfeng Chen, Qihuang Zhong, Juhua Liu, and Bo Du. SkillCAT: Contrastive, assessmentaugmented and topology-aware skill self-evolution for LLM agents. arXiv:2606.13317, 2026.

Jaewon Chu, Jinwoo Seo, Jaewon Cho, Jeehye Na, Yunyang Xiong, Youngdae Kim, and Hyunwoo J. Kim. AgentGrad: Intervention-guided prompt optimization for multi agent systems. In NeurIPS, 2026.

Giuseppe Destefanis, Daniel Graziotin, Matteo Vaccargiu, and Marco Ortu. GitSkills: A dataset of agent skills on GitHub. arXiv:2608.10906, 2026.

Ruomeng Ding, Wei Cheng, Minglai Shao, and Chen Zhao. SkillGen: Learning domain skills for in-context sequential decision making. In AAAI, 2026.

Runnan Fang, Yuan Liang, Xiaobin Wang, Jialong Wu, Shuofei Qiao, Pengjun Xie, Fei Huang, Huajun Chen, and Ningyu Zhang. Memp: Exploring agent procedural memory. In ACL-Findings, 2026.

Dawei Fu, Cheng Jiang, Sitian Qian, Huainan Wang, and Zhongkai Hao. SE-GoS: Self-evolving graph-of-skills for skill library at scale. arXiv:2609.08228, 2026.

Ryangkyung Kang, Hongcheol Cho, and Youngeun Kim. SkillRet: A large-scale benchmark for skill retrieval in LLM agents. arXiv:2605.05726, 2026.

Fangzhou Li, Pagkratios Tagkopoulos, and Ilias Tagkopoulos. SkillFlow: Scalable and efficient agent skill retrieval system. In COLM, 2026a.

Hao Li, Chunjiang Mu, Jianhao Chen, Siyue Ren, Zhiyao Cui, Yiqun Zhang, Lei Bai, and Shuyue Hu. Organizing, orchestrating, and benchmarking agent skills at ecosystem scale. arXiv:2603.02176, 2026b.

Xiangyi Li, Yimin Liu, Wenbo Chen, Bingran You, Zonglin Di, Yifeng He, Shenghan Zheng, Kyoung Whan Choe, Jiankai Sun, Shuyi Wang, et al. SkillsBench: Benchmarking how well agent skills work across diverse tasks. arXiv:2602.12670, 2026c.

Dawei Liu, Zongxia Li, Hongyang Du, Xiyang Wu, Shihang Gui, Yongbei Kuang, and Lichao Sun. Graph-of-Skills: Dependency-aware structural retrieval for massive agent skills. In EMNLP, 2026.

Zeyao Ma, Bohan Zhang, Jing Zhang, Jifan Yu, Xiaokang Zhang, Xiaohan Zhang, Sijia Luo, Xi Wang, and Jie Tang. SpreadsheetBench: Towards challenging real world spreadsheet manipulation. In NeurIPS, 2024.

Xiangcheng Meng, Shu Wang, and Yixiang Fang. SkillRAE: Agent skill-based context compilation for retrieval-augmented execution. arXiv:2605.10114, 2026.

Yongliang Miao, Ziyang Yu, Liang Zhao, Bowen Zhu, and Hasibul Haque. SkillLens: Adaptive multi-granularity skill reuse for cost-efficient LLM agents. arXiv:2605.08386, 2026.

Johannes Moll, Jean-Philippe Corbeil, Jiazhen Pan, Martin Hadamitzky, Daniel Rueckert, Lisa Adams, and Keno Bressem. GRASP: Gated regression-aware skill proposer for self-improving LLM agents. In EMNLP, 2026.

Jingwei Ni, Yihao Liu, Xinpeng Liu, Yutao Sun, Mengyu Zhou, Pengyu Cheng, Dexin Wang, Erchao Zhao, Xiaoxi Jiang, and Guanjun Jiang. Trace2Skill: Distill trajectory-local lessons into transferable agent skills. arXiv:2603.25158, 2026.

OpenAI. GPT-5.6: Frontier intelligence that scales with your ambition, July 2026. URL https: //openai.com/index/gpt-5-6/.

Krista Opsahl-Ong, Arnav Singhvi, Jasmine Collins, Ivan Zhou, Cindy Wang, Ashutosh Baheti, Owen Oertell, Jacob Portes, Sam Havens, Erich Elsen, Michael Bendersky, Matei Zaharia, and Xing Chen. OfficeQA Pro: An enterprise benchmark for end-to-end grounded reasoning. arXiv:2603.08655, 2026.

Siru Ouyang, Jun Yan, I Hsu, Yanfei Chen, Ke Jiang, Zifeng Wang, Rujun Han, Long Le, Samira Daruki, Xiangru Tang, et al. Reasoningbank: Scaling agent self-evolving with reasoning memory. In ICLR, 2026.

Qianjun Pan, Yutao Yang, Junsong Li, Jie Zhou, Kai Chen, Xin Li, Qin Chen, and Liang He. Anything2Skill: Compiling external knowledge into reusable skills for agents. arXiv:2606.09316, 2026.

Reid Pryzant, Dan Iter, Jerry Li, Yin Lee, Chenguang Zhu, and Michael Zeng. Automatic prompt optimization with “gradient descent” and beam search. In EMNLP, 2023.

Qwen Team. Qwen3.5: Towards native multimodal agents, February 2026. URL https://qwen .ai/blog?id=qwen3.5.

Stephen Robertson and Hugo Zaragoza. The probabilistic relevance framework: BM25 and beyond. Foundations and Trends in Information Retrieval, 2009.

Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik R Narasimhan, and Shunyu Yao. Reflexion: language agents with verbal reinforcement learning. In NeurIPS, 2023.

Mohit Shridhar, Xingdi Yuan, Marc-Alexandre Côté, Yonatan Bisk, Adam Trischler, and Matthew Hausknecht. ALFWorld: Aligning text and embodied environments for interactive learning. In ICLR, 2021.

Weihang Su, Jianming Long, Qingyao Ai, Qiaozhi He, Yichen Tang, Changyue Wang, Yiteng Tu, Yingbo Wang, and Yiqun Liu. Skill retrieval augmentation for agentic AI. arXiv:2604.24594, 2026.

Liyan Tang, Cyrus Rashtchian, Chun-Sung Ferng, Andrew Tomkins, Da-Cheng Juan, and Tu Vu. WikiSkill: Compiling agent experience into persistent knowledge for skill evolution. arXiv:2608.27454, 2026.

Xiangru Tang, Tianrui Qin, Tianhao Peng, Ziyang Zhou, Daniel Shao, Tingting Du, Xinming Wei, Peng Xia, Fang Wu, He Zhu, et al. Agent KB: Leveraging cross-domain experience for agentic problem solving. arXiv:2507.06229, 2025.

Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. Voyager: An open-ended embodied agent with large language models. TMLR, 2024.

Hanyu Wang, Yifan Lan, Bochuan Cao, Lu Lin, and Jinghui Chen. SkillGrad: Optimizing agent skills like gradient descent. arXiv:2605.27760, 2026.

Zora Zhiruo Wang, Jiayuan Mao, Daniel Fried, and Graham Neubig. Agent workflow memory. In ICML, 2025.

Tianle Xia, Lingxiang Hu, Yiding Sun, Ming Xu, Lan Xu, Siying Wang, Wei Xu, and Jie Jiang. GraSP: Graph-structured skill compositions for LLM agents. arXiv:2604.17870, 2026.

Zhiling Yan, Dingjie Song, Hanrong Zhang, Wei Liang, Yuxuan Zhang, Yutong Dai, Lifang He, Philip S Yu, Ran Xu, Xiang Li, et al. OpenSkill: Open-world self-evolution for llm agents. In EMNLP, 2026.

Chengrun Yang, Xuezhi Wang, Yifeng Lu, Hanxiao Liu, Quoc V Le, Denny Zhou, and Xinyun Chen. Large language models as optimizers. In ICLR, 2024a.

John Yang, Carlos Jimenez, Alexander Wettig, Kilian Lieret, Shunyu Yao, Karthik Narasimhan, and Ofir Press. SWE-agent: Agent-computer interfaces enable automated software engineering. In NeurIPS, 2024b.

Yifan Yang, Ziyang Gong, Weiquan Huang, Qihao Yang, Ziwei Zhou, Zisu Huang, Yan Li, Xuemei Gao, Qi Dai, Bei Liu, Kai Qiu, Yuqing Yang, Dongdong Chen, Xue Yang, and Chong Luo. SkillOpt: Executive strategy for self-evolving agent skills. arXiv:2605.23904, 2026.

Shunyu Yao, Howard Chen, John Yang, and Karthik Narasimhan. WebShop: Towards scalable real-world web interaction with grounded language agents. In NeurIPS, 2022.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. ReAct: Synergizing reasoning and acting in language models. In ICLR, 2023.

Mert Yuksekgonul, Federico Bianchi, Joseph Boen, Sheng Liu, Pan Lu, Zhi Huang, Carlos Guestrin, and James Zou. Optimizing generative AI by backpropagating language model feedback. Nature, 2025.

Qizheng Zhang, Changran Hu, Shubhangi Upasani, Boyuan Ma, Fenglu Hong, Vamsidhar Kamanuru, Jay Rainton, Chen Wu, Mengmeng Ji, Hanchen Li, et al. Agentic context engineering: Evolving contexts for self-improving language models. In ICLR, 2026.

Andrew Zhao, Daniel Huang, Quentin Xu, Matthieu Lin, Yong-Jin Liu, and Gao Huang. ExpeL: LLM agents are experiential learners. In AAAI, 2024.

Yanzhao Zheng, Zhentao Zhang, Chao Ma, Yuanqiang Yu, Jihuai Zhu, Yong Wu, Tianze Xu, Baohua Dong, Hangcheng Zhu, Ruohui Huang, and Gang Yu. SkillRouter: Skill routing for LLM agents at scale. In COLM, 2026.

## A APPENDIX

## A.1 IMPLEMENTATION DETAILS

For all update methods, we run skill optimization for 2 epochs over $\mathcal { D } _ { \mathrm { t r a i n } }$ with a minibatch size of 40 tasks per iteration. For Qwen-3.5-9B, we use a sampling temperature of 0.0 for agent rollouts and 0.7 for the optimizer model, and serve the model with vLLM on a single NVIDIA RTX A6000 GPU. For GPT-5.6-Luna, we set the reasoning effort to low for both the target and optimizer models. Following SkillOpt (Yang et al., 2026), we process the rollout minibatch in chunks when generating textual gradients and retrieval queries (Eq. 5). The trajectories in each minibatch are split into chunks of size 8, and $\mathcal { A } _ { \mathrm { q u e r y \_ u p d a t e } }$ produces gradient–query pairs for each chunk independently. A single LLM call then merges the chunk-level outputs into the final set $\{ ( q _ { i } , \delta _ { i } ) \} _ { i = 1 } ^ { N }$ , consolidating pairs that describe the same failure mode and removing redundant queries. This design keeps each call within the context limit while allowing the gradients to reflect failure patterns across the entire minibatch. Unless otherwise specified, we retrieve the top-K = 5 sections per query.

Since the external skill corpus is collected from public repositories, it may contain skills written specifically for our evaluation benchmarks. We therefore construct a per-benchmark blocklist and remove it from the corpus before building the retrieval index. For each benchmark, we search the corpus for keywords that identify it. The benchmark name and the names of the datasets it is built on. Whenever any skill matches, we exclude the entire repository containing it rather than the matched skill alone. We choose repository-level exclusion to be conservative. A repository that targets a benchmark often contains companion skills that encode benchmark-specific procedures without naming the benchmark, and such skills would evade keyword matching.

For each benchmark, we generate a task description T and a harness description H once, before any optimization run, and keep both fixed across all methods, models, and seeds. Both descriptions are produced by a coding agent (codex) that reads the benchmark’s harness code and states the task objective and output format (T) and the tools, file access, observation format, interaction budget, and scoring rule exposed by the harness (H). The coding agent was not given access to test instances, and neither description was revised after observing test performance.

To isolate the contribution of retrieval, Retrieval-Free Skill Initialization (RFSI) follows exactly the same pipeline as RASI except for retrieval and adaptation. Specifically, RFSI receives the identical T and H, uses the same procedure and requirement decomposition $\{ c _ { i } \} _ { i = 1 } ^ { M }$ , and synthesizes the skill with the same skill-initialization system prompt; the only difference is that the grounded lessons ${ \mathcal { L } } _ { \mathrm { i n i t } }$ are omitted. Hence, the gap between RASI and RFSI reflects the retrieved and adapted knowledge rather than the procedure decomposition or the prompt. All skill update baselines start from this RFSI skill, so every update method is compared under the same T, H, and initial skill.

## A.2 BASELINE DETAILS

Skill initialization. We compare RASI against three initialization strategies, all evaluated at zero rollouts. No skill runs the agent without an initialized skill. SkillRouter (Zheng et al., 2026) selects skills from a large library with a two-stage retrieve-and-rerank pipeline, in which a bi-encoder retrieves candidate skills and a cross-encoder reranks them using the full skill body. We use it to retrieve a relevant skill from the external skill corpus. Retrieval-Free Skill Initialization (RFSI) generates an initial skill with the target LLM directly from the task and harness descriptions, without access to the external skill corpus.

Skill update. We compare RASU against four update methods. TextGrad (Yuksekgonul et al., 2025) treats the skill as a text variable and optimizes it by backpropagating natural-language feedback, in which an LLM critiques execution results to produce textual gradients that are then used to revise the skill. GEPA (Agrawal et al., 2026) evolves the skill through reflective mutation, in which an LLM reflects on execution traces to propose revisions, and maintains a Pareto front of candidates to preserve diverse strategies during selection. SkillOpt (Yang et al., 2026) converts scored rollouts into bounded add, delete, and replace edits on a skill document and accepts an edit only when it improves a held-out validation score. WikiSkill (Tang et al., 2026) co-evolves skills with a persistent wiki, in which agent trajectories are consolidated into pages documenting failure modes and successful strategies, and skill updates are proposed by consulting this wiki. Unless otherwise specified, we use the same model as both the target and optimizer model for all methods, and none of the baselines accesses the external skill corpus during optimization.

Table 5: Number of tasks in each split.
<table><tr><td>Benchmark</td><td>Train</td><td>Validation</td><td>Test</td></tr><tr><td>OfficeQA</td><td>50</td><td>24</td><td>172</td></tr><tr><td>SpreadsheetBench</td><td>80</td><td>40</td><td>280</td></tr><tr><td>ALFWorld</td><td>39</td><td>18</td><td>134</td></tr><tr><td>WebShop</td><td>80</td><td>40</td><td>500</td></tr></table>

## A.3 BENCHMARK DETAILS

We evaluate on four benchmarks that cover diverse interaction settings, ranging from documentgrounded question answering to spreadsheet manipulation, embodied household tasks, and web navigation. For each benchmark, we use disjoint training, validation, and test splits $( \mathcal { D } _ { \mathrm { t r a i n } } , \mathcal { D } _ { \mathrm { v a l } }$ $\mathcal { D } _ { \mathrm { t e s t } } )$ , whose sizes are summarized in Table 5. Following SkillOpt (Yang et al., 2026), we use the same train/validation/test splits for all methods, so update acceptance and skill selection are performed on an identical validation set.

OfficeQA. OfficeQA (Opsahl-Ong et al., 2026) is a benchmark for end-to-end grounded reasoning over enterprise documents. Questions are posed over a corpus of U.S. Treasury Bulletins spanning nearly 100 years and about 89,000 pages, and require the agent to parse documents, retrieve relevant information across both unstructured text and tables, and perform numerical reasoning. Answers are numerical and are scored as correct when they fall within an allowed relative error tolerance. We use all 246 questions and split them into 50, 24, and 172 questions for training, validation, and testing, respectively.

SpreadsheetBench. SpreadsheetBench (Ma et al., 2024) is a benchmark for real-world spreadsheet manipulation, built from questions collected from online Excel forums. Each instruction requires the agent to read and modify spreadsheet files, and is evaluated in an online-judge style, where a solution is considered correct only if it produces the expected result on multiple spreadsheet files serving as test cases. We use 400 tasks, split into 80, 40, and 280 tasks for training, validation, and testing, respectively.

ALFWorld. ALFWorld (Shridhar et al., 2021) is a text-based embodied environment that aligns the ALFRED household tasks with the TextWorld engine. The agent navigates a simulated household and interacts with objects through textual actions to complete goals from six task types: pick and place, examine in light, clean and place, heat and place, cool and place, and pick two and place. We use 39 and 18 games for training and validation, and evaluate on the 134 out-of-distribution evaluation games.

WebShop. WebShop (Yao et al., 2022) is a simulated e-commerce environment containing about 1.18 million real-world products and 12,087 crowd-sourced instructions. Given a natural-language instruction describing a desired product, the agent searches, navigates product pages, and selects options to make a purchase, and is rewarded according to how well the purchased product matches the requested attributes, options, and price. We use 80 and 40 instructions for training and validation, and evaluate on the 500 test instructions.

## A.4 SKILL UPDATE METHODS UNDER FIXED INITIALIZATION

To isolate the contribution of the skill update procedure, we compare skill update methods while fixing the initial skill. Table 4 in the main paper reports skill update results of GPT-5.6-Luna initialized with RASI. Tables 6, 7, 8 report the remaining three combinations of initialization and LLM(Qwen-3.5-9B and GPT-5.6-Luna) on OfficeQA and SpreadsheetBench. In all four combinations, RASU achieves the highest performance, showing that retrieval-augmented update improves over retrievalfree updates regardless of the initial skill and the target model.

Table 6: Update methods under a fixed RASI initialization with Qwen-3.5-9B. The highlighted row is RASO.  
Table 7: Update methods under a fixed RFSI initialization with Qwen-3.5-9B. The highlighted row is RASU.
<table><tr><td colspan="2">Method</td><td colspan="2">Benchmark</td></tr><tr><td>Init.</td><td>Update</td><td>OfficeQA</td><td>Spreadsheet</td></tr><tr><td rowspan="4">RASI</td><td>TextGrad</td><td> $4 0 . 5 0 \pm 0 . 9 7$ </td><td> $3 0 . 4 8 \pm 0 . 7 8$ </td></tr><tr><td>GEPA</td><td> $3 8 . 7 6 \pm 1 . 4 0$ </td><td> $3 0 . 4 8 \pm 0 . 7 8$ </td></tr><tr><td></td><td> $3 8 . 1 8 \pm 0 . 7 0$ </td><td> $3 1 . 0 7 \pm 0 . 2 1$ </td></tr><tr><td>SkillOpt</td><td> $4 0 . 7 0 \pm 0 . 8 9$ </td><td> $3 0 . 6 0 \pm 0 . 6 6$ </td></tr><tr><td rowspan="4"></td><td>WikiSkill</td><td> $4 0 . 1 2 \pm 0 . 3 4$ </td><td> $3 0 . 4 8 \pm 0 . 7 8$ </td></tr><tr><td>RASU</td><td> $4 2 . 2 5 \pm 1 . 5 2$  </td><td> ${ \bf 3 1 . 5 5 \pm 0 . 3 1 }$  </td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr></table>

<table><tr><td colspan="2">Method</td><td colspan="2">Benchmark</td></tr><tr><td>Init.</td><td>Update</td><td>OfficeQA</td><td>Spreadsheet</td></tr><tr><td>RFSI</td><td>TextGrad GEPA SkillOpt WikiSkill RASU</td><td> $3 4 . 8 9 \pm 2 . 9 3$   $3 6 . 8 2 \pm { 1 . 5 9 }$   $3 6 . 4 3 \pm 1 . 9 7$   $3 7 . 2 1 \pm 1 . 2 1$   $3 7 . 4 0 \pm 3 . 0 3$ </td><td> $2 7 . 7 4 \pm 2 . 5 6$   $2 5 . 9 5 \pm 0 . 6 3$   $2 4 . 0 5 \pm 1 . 5 6$   $2 9 . 5 2 \pm 2 . 2 1$   $2 9 . 7 6 \pm 1 . 8 0$ </td></tr></table>

Table 8: Update methods under a fixed RFSI initialization with GPT-5.6-Luna. The highlighted row is RASU.

<table><tr><td colspan="2">Method</td><td colspan="2">Benchmark</td></tr><tr><td>Initialization</td><td>Update</td><td>OfficeQA</td><td>Spreadsheet</td></tr><tr><td rowspan="6">RFSI</td><td></td><td> $4 0 . 1 1 \pm 0 . 5 8$ </td><td> $4 4 . 4 0 \pm 2 . 7 4$ </td></tr><tr><td>TextGrad</td><td> $4 3 . 8 0 \pm 1 . 2 7$ </td><td> $4 9 . 6 4 \pm 3 . 0 9$ </td></tr><tr><td>GEPA</td><td> $4 3 . 9 9 \pm 1 . 4 0$ </td><td> $5 4 . 5 3 \pm 1 . 4 6$ </td></tr><tr><td>SkillOpt</td><td> $4 5 . 5 4 \pm 2 . 0 5$ </td><td> $5 7 . 0 2 \pm 3 . 6 0$ </td></tr><tr><td>WikiSkill</td><td> $4 1 . 4 7 \pm 1 . 1 8$ </td><td> $4 7 . 1 4 \pm 3 . 9 8$ </td></tr><tr><td>RASU</td><td> ${ \pm 7 . 5 6 \pm 1 . 8 5 }$ </td><td>一  ${ \bf 6 1 . 0 7 \pm 2 . 1 5 }$  </td></tr></table>

Under LLM-init (RFSI). With GPT-5.6-Luna (Table 8), RASU improves the initial skill by +7.45 and +16.67 points on OfficeQA and SpreadsheetBench, respectively, and outperforms the strongest competing baseline, SkillOpt, by +2.02 (47.56 vs. 45.54) and +4.05 (61.07 vs. 57.02). With Qwen-3.5-9B (Table 7), RASU improves the initial skill by +3.59 and +2.86 points and outperforms the strongest competing baseline, WikiSkill, by +1.08 (38.48 vs. 37.40) and +0.84 (30.60 vs. 29.76). In contrast, TextGrad and GEPA degrade the initial skill on SpreadsheetBench (25.95 and 24.05 vs. 27.74).

Under RASI. Table 6 shows retrieval-free baselines struggle to improve upon the retrievalaugmented initial skill with Qwen-3.5-9B under RASI. On OfficeQA, TextGrad, GEPA, and WikiSkill fall below the initial skill (38.76, 38.18, and 40.12 vs. 40.50), and SkillOpt improves it by only +0.20. On SpreadsheetBench, the largest gain among them is +0.59 (GEPA). RASU, in contrast, improves the initial skill by +1.75 and +1.07 points, reaching 42.25 and 31.55. TextGrad and WikiSkill return the initial skill unchanged on SpreadsheetBench, since no candidate update was improved over the validation set.

Complementarity of the two stages. The two stages contribute gains that neither achieves alone. With Qwen-3.5-9B on OfficeQA, RASI without any update (40.50) already outperforms LLM-init (RFSI) followed by RASU (38.48), and combining both stages yields the best performance (42.25).

## A.5 OPTIMIZATION COST

Tables 9 and 10 report the API cost and rollout count of skill optimization with GPT-5.6-Luna. All methods are optimized for the same 2 epochs. The rollout counts differ only because of how each method schedules evaluation. TextGrad, WikiSkill, SkillOpt and RASO update the skill from training rollouts and evaluate the updated skill on the validation set directly. On the other hand, GEPA first re-evaluates each updated skill on a support subset of the training set and proceeds to validation only when this yields an improvement, which incurs additional rollouts. SkillOpt performs an additional slow, meta update at the end of every epoch, which likewise adds rollouts. RASO therefore uses exactly as many rollouts as TextGrad and WikiSkill, and fewer than GEPA and SkillOpt.

Table 9: API cost (USD) of skill optimization with GPT-5.6-Luna. The lowest cost in each column is in bold.
<table><tr><td>Method</td><td>OfficeQA</td><td>Spreadsheet</td><td>ALFWorld</td><td>WebShop</td></tr><tr><td>TextGrad</td><td>1.45</td><td>0.97</td><td>2.04</td><td>41.03</td></tr><tr><td>GEPA</td><td>1.69</td><td>0.58</td><td>1.79</td><td>18.05</td></tr><tr><td>SkillOpt</td><td>2.23</td><td>2.24</td><td>2.75</td><td>17.86</td></tr><tr><td>WikiSkill</td><td>1.06</td><td>0.43</td><td>1.30</td><td>11.15</td></tr><tr><td>RASO</td><td>0.84</td><td>0.49</td><td>0.79</td><td>10.28</td></tr></table>

Table 10: Number of rollouts used for skill optimization, excluding test-set evaluation.

<table><tr><td>Method</td><td>OfficeQA</td><td>Spreadsheet</td><td>ALFWorld</td><td>WebShop</td></tr><tr><td>TextGrad</td><td>196</td><td>320</td><td>114</td><td>360</td></tr><tr><td>GEPA</td><td>272</td><td>360</td><td>174</td><td>370</td></tr><tr><td>SkillOpt</td><td>236</td><td>360</td><td>154</td><td>400</td></tr><tr><td>WikiSkill</td><td>196</td><td>320</td><td>114</td><td>360</td></tr><tr><td>RASO</td><td>196</td><td>320</td><td>114</td><td>360</td></tr></table>

Under this budget, RASO achieves the lowest cost on three of the four benchmarks and remains within \$0.06 of the cheapest method on SpreadsheetBench. Since the budgets are matched, the savings reflect the cost of each rollout and optimization LLM call rather than the number of rollout. At an identical budget, RASO reduces the cost of TextGrad by 42–75%, with the largest gap on WebShop (\$41.03 vs. \$10.28).