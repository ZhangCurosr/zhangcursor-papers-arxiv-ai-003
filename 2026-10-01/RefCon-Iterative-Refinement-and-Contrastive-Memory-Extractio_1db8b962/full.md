# RefCon: Iterative Refinement and Contrastive Memory Extraction for Context-Evolving Agent

Ubaidillah Ariq Prathama<sup>1</sup>, Bo Liu<sup>1</sup>, Yeo Boon Hong<sup>1</sup>, Yu-Xuan Huang<sup>1</sup>, Yangkai Ding<sup>1</sup>, Tao Yu<sup>1</sup> <sup>1</sup>Huawei Technologies, Co., Ltd.

## Abstract

Long-horizon agent interactions generate useful but noisy experience, and retraining models to absorb it is expensive. Context-evolving agents therefore need memory extraction methods that improve with more test-time compute without relying on gold labels. We propose RefCon, which combines sequential selfrefinement with parallel self-contrast to extract higher-quality memories without gold labels. Evaluated on AppWorld and BFCL-V3 across multiple context-evolving agent frameworks, RefCon delivers strong and consistent gains, including relative improvements of 21.6% on ACE and 16.6% on ReMe over no-scaling baselines, while a diversity-focused variant (DivCon) achieves a 35.5% gain on Reasoning-Bank. RefCon consistently outperforms existing baselines without ground-truth labels, and generalizes across model scales and to software engineering tasks, where it surpasses even ground-truth baselines. We further analyze the accuracy-token trade-off and scaling behavior, showing RefCon maintains favorable efficiency and continues to improve as more trajectories are used, unlike diversity-only scaling which saturates earlier.

## 1 Introduction

A few years ago, large language models (LLMs) began to be deployed as agents capable of solving complex tasks through iterative reasoning and tool use. Applications such as web browsing (Wei et al., 2025), interactive tool calling (Patil et al., 2025), and deep research (Phan et al., 2025) require agents to generate long trajectories of actions and observations. However, these trajectories are typically discarded after task completion, despite often containing reusable insights that could benefit future tasks (Shinn et al., 2023; Zhao et al., 2024). While one possible solution is to incorporate such knowledge through model retraining, updating large language models remains computationally expensive due to their growing scale (Hoffmann et al., 2022). This motivates lightweight mechanisms that allow agents to accumulate and reuse experience directly from past interactions without costly retraining.

More recently, self-evolving agent approaches integrate knowledge updates directly into the inference loop. In this paradigm, agents continuously write, refine, and retrieve memories to compose an up-to-date context for each new query, enabling continual learning without expensive model retraining (Park et al., 2023; Wang et al., 2024). These context-evolving agents typically involve three components: memory extraction, memory retrieval, and memory management (Zhang et al., 2025b; Cao et al., 2025; Yan et al., 2025; Zhang et al., 2025a). Memory extraction distills raw trajectories into reusable heuristics; memory retrieval selects relevant experiences for the current task; and memory management maintains a compact memory store by filtering redundancy and irrelevant content. Recent work further explores memoryaware test-time scaling (MaTTS) to improve memory extraction by generating multiple trajectories and selecting higher-quality experiences (Ouyang et al., 2025). This direction builds on broader findings that test-time scaling can improve agent performance (Snell et al., 2025; Zhu et al., 2025).

Despite recent progress, several challenges remain. Some methods rely on gold labels or groundtruth signals to extract memories (Cao et al., 2025; Zhang et al., 2025b; Cai et al., 2025b), which are often unavailable in real-world settings, while replacing them with LLM-as-a-Judge (Zheng et al., 2023) can be unreliable for long-horizon interactions where trajectories appear plausible yet fail to achieve the intended goal (Ouyang et al., 2025; Yang et al., 2025; Liu et al., 2025). Reasoning-Bank (Ouyang et al., 2025) addresses this by introducing self-contrast reasoning over multiple trajectories for parallel scaling (Figure 1 (a)) and self-refinement for sequential scaling (Figure 1 (b)). However, while prior work explores both paradigms (Stein et al., 2025; Wu et al., 2025; Wang et al., 2025a; Yan et al., 2025; Tang et al., 2025), they are typically studied in isolation despite being complementary: parallel scaling improves trajectory diversity while sequential scaling enhances trajectory quality. Evaluating memory extraction effectively requires benchmarks with two key properties: (1) task recurrence, so memories extracted from earlier tasks transfer to later ones, and (2) trajectory ambiguity, where single trajectories may contain misleading signals that contrastive multi-trajectory analysis must disambiguate. These properties enable distinguishing between good and poor extraction methods.

![](images/664021a26a447fa3e548eacebfaedcf869f22f860c00a6b5ba580b8934edfa72.jpg)  
Figure 1: Memory extraction with (a) parallel scaling using self-contrast, and (b) sequential scaling with self-refine, compared to our proposed method (c) RefCon, combining both strength of parallel and sequential scaling.

To address these limitations, we propose RefCon, an extension of memory-aware test-time scaling (MaTTS) for memory extraction that combines the benefits of sequential and parallel scaling. RefCon first applies sequential scaling to generate N trajectories using self-refine (Madaan et al., 2023), as illustrated in Figure 1. These trajectories are then jointly analyzed through parallel scaling, where the memory extractor performs self-contrast reasoning (Chen et al., 2020) to identify improvements across trajectories and distill reusable insights. This design encourages both trajectory diversity and quality, enabling more reliable memory extraction. We further introduce a variant, DivCon, which promotes diversity by instructing the agent to explore alternative strategies instead of refining previous trajectories. For memory retrieval, we adopt a retrieve–rerank–rewrite pipeline that improves relevance by reranking retrieved memories and rewriting them into a concise form. Overall, our contributions are twofold:

• We propose RefCon, a memory-aware testtime scaling framework that unifies parallel and sequential scaling to improve memory extraction quality. We will release it as a plug-in for agentic systems to support contextevolving methods, enabling easier adoption and future research.

• Extensive experiments demonstrate that RefCon consistently outperforms existing baselines in the without-ground-truth setting. Compared with the no-scaling setting, RefCon improves performance by 21.6% on ACE and 16.6% on ReMe, while the variant DivCon achieves a 35.5% gain on ReasoningBank.

## 2 Related Works

Memory Extraction. Modern memory extraction has transitioned from passive logging to active insight distillation. ACE (Zhang et al., 2025b) decouples reflection from curation to synthesize memory updates, while ReasoningBank (Ouyang et al., 2025) highlights the importance of integrating cautionary insights from failed trajectories. Other frameworks distill insights at varying granularities: MUSE (Yang et al., 2025) and FLEX (Cai et al., 2025b) separate high-level heuristics from execution details, EGuR (Stein et al., 2025) learns task-specific workflows, and SAGE (Wang et al., 2025a) and SMITH (Liu et al., 2025) generate reusable code snippets. ReasoningBank (Ouyang et al., 2025) also introduces memory-aware testtime scaling (MaTTS), utilizing self-contrast reasoning (Chen et al., 2020) for parallel scaling and self-refine (Madaan et al., 2023) for sequential scaling, a foundation adopted and extended by many follow-up works in both parallel (Stein et al., 2025; Wu et al., 2025; Cao et al., 2025; Liu et al., 2025; Cai et al., 2025a; Wang et al., 2025a) and sequential (Cao et al., 2025; Yang et al., 2025; Yan et al., 2025; Tang et al., 2025) settings.

Memory Retrieval. Earlier approaches either inserted all memories into the prompt (Suzgun et al., 2025; Zhang et al., 2025b) or relied on basic embedding-similarity retrieval (Ouyang et al., 2025). Since then, retrieval has evolved into dynamic and hierarchical architectures: ReMe (Cao et al., 2025) applies a retrieve–rerank–rewrite pipeline, GAM (Yan et al., 2025) employs "Researcher" agents for deep dives into historical data, G-Memory (Zhang et al., 2025a) leverages graph traversals, AgentKB (Tang et al., 2025) separates workflow patterns from execution details, and EvolveR (Wu et al., 2025) treats retrieval as a learned tool-call via GRPO.

Memory Management. To prevent unbounded memory growth, recent systems treat memory as a self-optimizing component. ACE (Zhang et al., 2025b) and FLEX (Cai et al., 2025b) delegate insert, update, and delete operations to a dedicated agent, while ReMe (Cao et al., 2025) and EvolveR (Wu et al., 2025) prune memories based on retrieval utility scores. Despite its importance, explicit memory management remains underexplored in many current systems.

## 3 Methodology

## 3.1 Problem Formulation

In this work, we focus on the test-time learning (TTL) paradigm (Wu et al., 2024; Wang et al., 2025b), where a model continues to improve its performance during inference without access to ground-truth supervision. Specifically, we consider a setting in which an agent must adapt to a continuous stream of task queries $Q = \{ q _ { 1 } , q _ { 2 } , . . . , q _ { N } \}$ encountered sequentially in an online environment. Unlike traditional supervised learning, this setting precludes access to future queries or ground-truth labels. Instead, the agent must self-evolve by critically analyzing its past trajectories and leveraging self-verification mechanisms. The primary objective is to enable the system to extract and retain useful insights from prior interactions, thereby avoiding redundant rediscovery of successful strategies or the repetition of past failures.

Specifically, we instantiate an LLM-based agent $\pi _ { \mathcal { L } }$ that solves each query through iterative tool use and reasoning in a ReAct loop, while reusing accumulated experience across tasks. To model this cross-task learning process, we define a memory state initialized as $\mathcal { M } _ { 0 } = \emptyset$ . At task i, the agent conditions on the current memory $\mathcal { M } _ { i }$ (appended to the prompt) to produce a trajectory for the query. After the task is completed, the trajectory is distilled into memory updates, yielding the next state $\mathcal { M } _ { i + 1 }$ . Thus, memory evolves at the task level as $\mathcal { M } _ { 0 } , \mathcal { M } _ { 1 } , . . . ,$ and the agent for task i is parameterized by $\pi _ { \mathcal { L } } ( \cdot ; \mathcal { M } _ { i } , \mathcal { A } )$ , where A is the available action/tool space.

## 3.2 RefCon for Memory Extraction

We propose an iterative refinement and contrastive (RefCon) framework, which generates trajectories for our memory extraction process as shown in Figure 2 green zone. The memory extraction process within the test-time learning paradigm is formulated as an iterative refinement procedure that generates a sequence of trajectories. For a given task query $q _ { i } ^ { \phantom { } }$ , the agent first produces an initial trajectory by conditioning its policy $\pi \mathcal { L }$ on the rewritten memory $m _ { i } .$ , which is retrieved from the current memory state $\mathcal { M } _ { i }$ :

$$
\tau _ { i , 1 } = \pi _ { \mathcal { L } } ( q _ { i } , m _ { i } ) .\tag{1}
$$

Subsequently, the agent iteratively refines its previous trajectory to generate improved problemsolving attempts:

$$
\tau _ { i , n } = \pi _ { \mathcal { L } } ^ { \mathrm { r e f i n e } } ( q _ { i } , m _ { i } , \tau _ { i , n - 1 } ) , \quad n \in \{ 2 , \dots , N \} ,\tag{2}
$$

where $\pi _ { \mathcal { L } } ^ { \mathrm { r e f i n e } }$ represents the self-refinement policy that evaluates and improves upon the preceding trajectory. The resulting trajectory set $\{ \tau _ { i , 1 } , \tau _ { i , 2 } , . . . , \tau _ { i , N } \}$ captures a progression of reasoning attempts, evolving from initial exploration to increasingly refined and self-corrected strategies.

Once the N trajectories are generated, a selfcontrast reasoning mechanism is applied to distill high-value insights for memory extraction. Specifically, the agent’s memory extractor compares the differences between successful and unsuccessful steps across the N trajectories, identifying key turning points or reasoning patterns that lead to successful outcomes. Through this contrastive analysis, the agent summarizes these insights into a set of candidate memories $m _ { n e w } .$ formally defined as

Memory Deduplication, Memory Management, and Add Memory  
![](images/5d29c8f9b90e1a087ab059a5f13eb334c84923052ed65dd67cff91cf9dfd1085.jpg)  
Figure 2: Overview of context-evolving agent workflow utilizing RefCon scaling for memory extraction.

$$
m _ { n e w } = \pi _ { \mathcal { L } } ^ { \mathrm { c o n t r a s t } } ( \tau _ { i , 1 } , \dots , \tau _ { i , N } ) ,\tag{3}
$$

where $\pi _ { \mathcal { L } } ^ { \mathrm { c o n t r a s t } } ( \cdot )$ denotes the contrastive memory extraction operator that distills actionable knowledge from multiple trajectories.

Before finalizing the knowledge state $\mathcal { M } _ { i + 1 }$ the system performs a memory deduplication step using embedding-based similarity (Sim) with a threshold ϵ to remove redundant memories. Specifically, the candidate memories $m _ { n e w }$ are compared against both the existing memory $\mathcal { M } _ { i }$ (intermemory comparison) and among themselves (intramemory comparison). Only non-redundant memories are retained, ensuring that memory remains concise and informative. The update rule is defined as:

$$
\begin{array} { c } { { \mathcal { M } _ { i + 1 } = \mathcal { M } _ { i } \cup \{ m \in m _ { \mathrm { n e w } } \mid } }  \\ { { \mathrm { S i m } ( m , \mathcal { M } _ { i } \cup m _ { \mathrm { n e w } } \setminus \{ m \} ) < \epsilon \} . } } \end{array}\tag{4}
$$

The resulting memory $\mathcal { M } _ { i + 1 }$ now contains memories from all previous $i + 1$ task. Each memory entry contains two key components: (i) when to use, a short explanation of when to use the memory, also used as an index; (ii) content, describing the insight from previous trajectory to guide agent in similar situation. These memories also contains other metadata such as frequency and utility.

DivCon. We also propose an alternative trajectory generation strategy named DivCon. Unlike self-refine in RefCon, DivCon uses selfdiversity to enhance memory extraction through self-contrasting highly diverse trajectories. The difference with RefCon relies on Eq. 2, where DivCon prompts the agent to explore other solutions:

$$
\tau _ { i , n } = \pi _ { \mathcal { L } } ^ { \mathrm { d i v e r s i t y } } ( q _ { i } , m _ { i } , \tau _ { i , n - 1 } ) .\tag{5}
$$

## 3.3 Memory Retrieval and Management

The retrieval process serves as the initial phase of the test-time learning cycle, augmenting the agent’s policy $\pi _ { \mathcal { L } }$ with the most relevant historical insights before solving a new task. We adopt a retrieve–rerank–rewrite strategy. Given a query q<sub>i</sub>, the system first retrieves candidate memories from the current memory state $\mathcal { M } _ { i }$ using embedding similarity. A reranking agent then evaluates these candidates and selects the most relevant insights according to the policy’s internal ranking logic. Finally, a rewriting agent consolidates the reranked memories to produce a concise and query-aligned memory:

$$
\begin{array} { r } { m _ { i } = \pi _ { \mathcal { L } } ^ { \mathrm { r e w r i t e } } ( \pi _ { \mathcal { L } } ^ { \mathrm { r e r a n k } } ( \operatorname { S i m } ( \mathcal { M } _ { i } , q _ { i } ) , q _ { i } ) , q _ { i } ) . } \end{array}\tag{6}
$$

The resulting memory $m _ { i }$ summarizes the most relevant prior experience and is used to guide the agent when solving the current task. To maintain a high-quality knowledge base, we implement a utility-based deletion strategy to prune ineffective memories. This ensures that the agent’s behavior is shaped only by high-utility memories. We define the utility of an experience $E \in { \mathcal { M } }$ based on two metrics: the total number of retrievals $f ( E )$ and the historical utility $u ( E )$ , which increments by 1 whenever its recall contributes to a successful task completion. An experience is flagged for removal if its average utility falls below a predefined threshold $\beta ,$ provided it has been retrieved at least α times.Using the established notation, the deletion function ϕ<sub>remove</sub> is formulated as:

$$
\phi _ { r e m o v e } ( E ) = \left\{ \begin{array} { l l } { 1 \left[ \frac { u ( E ) } { f ( E ) } \leq \beta \right] , } & { \mathrm { i f ~ } f ( E ) \geq \alpha , } \\ { 0 , } & { \mathrm { o t h e r w i s e . } } \end{array} \right.\tag{7}
$$

## 4 Experiments

## 4.1 Setup

Datasets. Datasets. Evaluating memory extraction quality requires benchmarks with two properties: recurrence (structurally similar tasks across the sequence so extracted memories are reusable) and trajectory ambiguity (tasks complex enough that single trajectories may contain misleading signals), making contrastive multi-trajectory analysis necessary. We use two datasets emphasizing multi-step agent interaction: AppWorld (Trivedi et al., 2024), which contains app-like tasks requiring tool use across multiple steps, and BFCL-V3 (Patil et al., 2025), which focuses on tool-calling and decision sequences. We use the AppWorld dev split and sample 50 tasks from the BFCL-V3 multi-turn travel subset, which emphasizes recurrence. We report Avg@3 and Pass@3 on BFCL-V3, and Avg@3, Pass@3, Task Goal Completion (TGC) and Scenario Goal Completion (SGC) on AppWorld, with standard deviation across three runs. AppWorld SGC captures recurrence as each scenario contains similar tasks, and further exhibits trajectory ambiguity where similar trajectories may converge on conflicting answers (Table 13, Appendix C). We additionally ablate on SWE-bench Verified (Mini) (Jimenez et al., 2024; Hobbhahn, 2024) to evaluate generalization under weaker feedback and higher trajectory ambiguity.

Models. We use GLM-4.6 (Z.ai, 2024) as the backbone LLM for agent reasoning, tool use, and memory extraction, and OpenAI text-embedding-3-small (OpenAI, 2024) for retrieval indexing and similarity search. GLM-4.6 provides strong instruction following and multi-step reasoning, while text-embedding-3-small offers high-quality semantic representations for effective memory retrieval. We also experiment using Gemma 4 31B (Google DeepMind, 2026) and Qwen3.5 9B (Team, 2026) for generalization to medium and small model.

Baselines & Hyperparameters. We use ReAct as base and evaluate against three context-evolving agents: ACE (Zhang et al., 2025b), which extracts memories via a reflector–curator pipeline; ReasoningBank (Ouyang et al., 2025), which introduces MaTTS with embedding-based retrieval; and ReMe (Cao et al., 2025), which further improves retrieval via a retrieve–rerank–rewrite pipeline. We employ a scaling factor of 3 (except 2 for DivCon). We follow the original implementations for results in Table 1. For Table 2, we disable ground-truth usage and integrate MaTTS for memory extraction. Further details on baseline methods, hyperparameters, and prompt templates are provided in Appendices B, D, and E.

## 4.2 Main Comparison

Table 1 shows that while ground-truth methods remain strong, RefCon achieves the best overall performance without ground-truth labels, surpassing some ground-truth baselines (e.g., ReMe Parallel) and nearly matching the top AppWorld scores. Relative to ReAct, RefCon improves AppWorld Avg@3 by 16.51% (TGC) and 50.06% (SGC), and BFCL-V3 Pass@3 by 32.36%. Against Reasoning-Bank (Sequential), RefCon improves AppWorld Avg@3 by up to 20.06% and BFCL-V3 Pass@3 by 13.89%, though it trails ReasoningBank (Parallel) in BFCL-V3 Avg@3 by 3.06%, likely because BFCL-V3’s narrow distribution favors contrastive diversity over refinement. DivCon consistently ranks second in the without-ground-truth setting except on BFCL-V3. As shown in Figure 3, these trends hold across Gemma 4 31B and Qwen3.5 9B (Tables 4–5, Appendix A), with RefCon remaining competitive with ACE (ground-truth) across all tested LLMs, though gains diminish on Qwen3.5 9B, suggesting memory extraction quality is contingent on base model capability.

Several anomalies are worth noting. On BFCL-V3, self-refinement is less reliable and can degrade correct trajectories, explaining the underperformance of RefCon, DivCon, and ReasoningBank (Sequential). On AppWorld, ACE and Reasoning-Bank without scaling fall below ReAct, confirming that MaTTS is critical — single-trajectory extraction without labels risks reinforcing incorrect behavior. Finally, ReMe’s parallel scaling underperforms even with ground-truth because it only applies self-contrast when trajectory scores differ, leaving the contrastive signal frequently absent. Additional analysis is provided in Appendix C.

Table 1: Performance of context-evolving agent methods on AppWorld and BFCL-V3 Using GLM-4.6.
<table><tr><td rowspan="3">Methods</td><td colspan="4">AppWorld</td><td colspan="2">BFCL-V3</td></tr><tr><td colspan="2">Avg@3</td><td colspan="2">Pass@3</td><td rowspan="2">Avg@3</td><td rowspan="2">Pass@3</td></tr><tr><td>TGC</td><td>SGC</td><td>TGC</td><td>SGC</td></tr><tr><td colspan="7">Baseline</td></tr><tr><td>ReAct</td><td> $6 3 . 7 7 ^ { \pm 4 . 6 0 }$ </td><td> $3 5 . 1 0 ^ { \pm 1 0 . 8 0 }$ </td><td>87.72</td><td>68.42</td><td> $5 2 . 6 7 ^ { \pm 4 . 1 1 }$ </td><td>62.00</td></tr><tr><td colspan="7">With Gold Labels/Ground-Truth</td></tr><tr><td>ACE (with Ground-Truth)</td><td> $\mathbf { 7 8 . 0 0 ^ { \pm 5 . 3 4 } }$ </td><td> $5 0 . 9 ^ { \pm 8 . 9 6 }$ </td><td>96.49</td><td>89.47</td><td> $6 6 . 6 7 ^ { \pm 4 . 7 1 }$ </td><td>82.00</td></tr><tr><td>ReMe (Sequential Scaling)</td><td> $7 5 . 4 3 ^ { \pm 1 0 . 8 1 }$ </td><td> $\mathbf { 5 7 . 9 3 ^ { \pm 1 4 . 9 0 } }$ </td><td>96.49</td><td>89.47</td><td> ${ \bf 6 7 . 3 3 ^ { \pm 0 . 9 4 } }$ </td><td>82.00</td></tr><tr><td>ReMe (Parallel Scaling)</td><td> $6 2 . 0 0 ^ { \pm 0 . 8 5 }$ </td><td> $3 6 . 8 0 ^ { \pm 0 . 0 0 }$ </td><td>84.21</td><td>68.42</td><td> $5 8 . 6 7 ^ { \pm 1 . 8 9 }$ </td><td>68.00</td></tr><tr><td colspan="7">Without Gold Labels/Ground-Truth</td></tr><tr><td>ACE (without Ground-Truth)</td><td> $5 9 . 6 3 ^ { \pm 2 . 9 }$ </td><td> $4 2 . 1 0 ^ { \pm 4 . 3 3 }$ </td><td>77.19</td><td>73.68</td><td> $5 8 . 0 0 ^ { \pm 5 . 8 9 }$ </td><td>80.00</td></tr><tr><td>ReasoningBank</td><td> $5 6 . 1 3 ^ { \pm 1 . 4 3 }$ </td><td> $4 2 . 1 0 ^ { \pm 4 . 3 3 }$ </td><td>80.70</td><td>68.42</td><td> $6 0 . 0 0 ^ { \pm 7 . 1 2 }$ </td><td>82.00</td></tr><tr><td>ReasoningBank (Sequential Scaling)</td><td> $6 9 . 0 3 ^ { \pm 1 . 6 5 }$ </td><td> $4 3 . 8 7 ^ { \pm 2 . 5 0 }$ </td><td>84.21</td><td>73.68</td><td> $6 4 . 0 0 ^ { \pm 1 . 6 3 }$ </td><td>74.00</td></tr><tr><td>ReasoningBank (Parallel Scaling)</td><td> $6 7 . 2 7 ^ { \pm 2 . 2 0 }$ </td><td> $4 5 . 6 3 ^ { \pm 1 0 . 8 1 }$ </td><td>85.96</td><td>78.95</td><td> ${ \bf 6 5 . 3 3 ^ { \pm 4 . 1 1 } }$ </td><td>72.00</td></tr><tr><td>DivCon</td><td> $7 1 . 9 3 ^ { \pm 5 . 7 6 }$ </td><td> $4 7 . 3 7 ^ { \pm 4 . 2 9 }$ </td><td>92.98</td><td>78.95</td><td> $5 8 . 6 7 ^ { \pm 3 . 4 0 }$ </td><td>76.00</td></tr><tr><td>RefCon</td><td> $\mathbf { 7 4 . 3 0 ^ { \pm 0 . 9 4 } }$ </td><td> ${ \pm 2 . 6 7 ^ { \pm 7 . 4 5 } }$ </td><td>92.98</td><td>78.95</td><td> $6 3 . 3 3 ^ { \pm 0 . 9 4 }$ </td><td>82.00</td></tr></table>

![](images/62502ce1541d489eaf69630dac7b9d6be04b67e4d088e621b918456c0b4d223a.jpg)  
Figure 3: Relative improvement of compared to ReAct baseline across different frameworks and LLMs.

complementary strengths: RefCon benefits more when iterative refinement yields progressively better trajectories, whereas DivCon shines when diversity exposes contrasting reasoning patterns. Notably, ACE originally lacks any scaling mechanism, yet adding MaTTS variants substantially improves performance without changing its reflector/curator pipeline, implying gains come from richer trajectory evidence rather than architectural changes and reinforcing that test-time scaling is a broadly effective lever for memory quality across contextevolving agents.

## 4.3 RefCon’s Combined Scaling Advantage

In this subsection, we only adopt the memory format and memory retrieval strategy from each method. However, we modify the memory extraction part to adopt parallel and sequential scaling as introduced by (Ouyang et al., 2025) and also Ref-Con and DivCon as introduced by us. We don’t use any gold-label or ground-truth to ensure fair comparison in this section. We compare the effect of MaTTS methods for each context-evolving agent methods as shown in Table 2 and provide detailed results for each run in Appendix A.

Across agents, RefCon is the strongest variant on ACE (21.6% relative improvement over no scaling) and ReMe (16.6%), while DivCon is the top performer on ReasoningBank (35.5%), with the runner-up alternating between the two, suggesting

DivCon remains competitive with a smaller scaling factor (2), as contrastive extraction benefits most from genuinely diverse trajectories. Temperature-based parallel sampling can be too homogeneous when the model strongly prefers a single solution. DivCon performs particularly well with ReasoningBank due to its higher-level memory format, which encourages exploring alternative trajectories without contradicting existing memories. We still prioritize RefCon in the main results as DivCon is less stable and exhibits weaker scaling behavior, as will be discussed in the scaling law analysis. We further analyze RefCon’s superior ability to resolve recurring problems in Appendix A (Figure 6), showing that self-contrast memory extraction produces the steepest performance gains between consecutive similar tasks, while sequential scaling fails to maintain consistent improvements due to the limitations of extracting memory from a single trajectory.

Table 2: Comparison of RefCon and DivCon against MaTTS methods for context-evolving agents on AppWorld.
<table><tr><td rowspan="3">Methods</td><td colspan="4">AppWorld</td></tr><tr><td colspan="2">Avg@3</td><td colspan="2">Pass@3</td></tr><tr><td>TGC</td><td>SGC</td><td>TGC</td><td>SGC</td></tr><tr><td></td><td>ACE</td><td></td><td></td><td></td></tr><tr><td>ACE (Without Scaling) ACE (Parallel Scaling) ACE (Sequential Scaling)</td><td> $5 9 . 6 3 ^ { \pm 2 . 9 }$   $6 3 . 7 3 ^ { \pm 9 . 5 5 }$   $6 4 . 3 0 ^ { \pm 1 . 5 5 }$ </td><td> $4 2 . 1 0 ^ { \pm 4 . 3 3 }$   $4 3 . 8 7 ^ { \pm 1 3 . 8 3 }$   $4 2 . 1 0 ^ { \pm 7 . 4 5 }$ </td><td>77.19 84.21 84.21</td><td>73.68 78.95 78.95</td></tr><tr><td>ACE (DivCon) ACE (RefCon)</td><td> $6 9 . 0 0 ^ { \pm 2 . 1 6 }$   $7 2 . 5 3 ^ { \pm 5 . 8 2 }$ </td><td> $4 2 . 1 0 ^ { \pm 4 . 3 3 }$   ${ \bf 4 3 . 8 7 ^ { \pm 6 . 5 7 } }$ </td><td>91.23 92.98</td><td>78.95 78.95</td></tr><tr><td></td><td>ReasoningBank  $5 6 . 1 3 ^ { \pm 1 . 4 3 }$ </td><td></td><td></td><td></td></tr><tr><td>ReasoningBank (Without Scaling) ReasoningBank (Parallel Scaling)</td><td> $6 7 . 2 7 ^ { \pm 2 . 2 0 }$ </td><td> $4 2 . 1 0 ^ { \pm 4 . 3 3 }$   $4 5 . 6 3 ^ { \pm 1 0 . 8 1 }$ </td><td>80.70 85.96</td><td>68.42 78.95</td></tr><tr><td>ReasoningBank (Sequential Scaling)</td><td> $6 9 . 0 3 ^ { \pm 1 . 6 5 }$ </td><td> $4 3 . 8 7 ^ { \pm 2 . 5 0 }$ </td><td>84.21</td><td>73.68</td></tr><tr><td>ReasoningBank (DivCon)</td><td> $\mathbf { 7 6 . 0 3 ^ { \pm 2 . 2 1 } }$ </td><td> ${ \pm } 2 . 6 7 ^ { \pm 0 . 0 0 }$ </td><td>94.74</td><td>89.47</td></tr><tr><td>ReasoningBank (RefCon)</td><td> $7 1 . 9 3 ^ { \pm 3 . 8 0 }$ </td><td> $4 9 . 1 3 ^ { \pm 1 3 . 1 3 }$ </td><td>92.98</td><td>84.21</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>ReMe</td><td></td><td></td><td></td></tr><tr><td>ReMe (Without Scaling)</td><td> $6 3 . 7 3 ^ { \pm 3 . 3 0 }$ </td><td> $4 3 . 8 7 ^ { \pm 2 . 5 0 }$ </td><td>85.96</td><td>73.68</td></tr><tr><td>ReMe (Parallel Scaling)</td><td> $7 0 . 7 6 ^ { \pm 3 . 5 7 }$ </td><td> $4 2 . 1 0 ^ { \pm 4 . 3 3 }$ </td><td>91.23</td><td>84.21</td></tr><tr><td>ReMe (Sequential Scaling)</td><td> $6 9 . 6 7 ^ { \pm 3 . 3 0 }$ </td><td> $4 7 . 3 7 ^ { \pm 4 . 2 9 }$ </td><td>91.23</td><td></td></tr><tr><td></td><td> $7 1 . 9 3 ^ { \pm 5 . 7 6 }$ </td><td> $4 7 . 3 7 ^ { \pm 4 . 2 9 }$ </td><td></td><td>78.95</td></tr><tr><td>ReMe (DivCon)</td><td></td><td></td><td>92.98</td><td>78.95</td></tr><tr><td>ReMe (RefCon)</td><td> $\mathbf { 7 4 . 3 0 ^ { \pm 0 . 9 4 } }$ </td><td> ${ \pm 2 . 6 7 ^ { \pm 7 . 4 5 } }$ </td><td>92.98</td><td>78.95</td></tr></table>

Table 3: Performance of context-evolving agent methods on SWE-Bench-Verified (Mini) using GLM-4.6.
<table><tr><td>Method</td><td>Iter 1</td><td>Iter 2</td><td>Iter 3</td></tr><tr><td colspan="4">Methods Without Refinement</td></tr><tr><td>ReAct</td><td> $5 0 . 6 7 ^ { \pm 1 . 1 5 }$ </td><td></td><td></td></tr><tr><td>RB (Par) ACE (w/ GT)</td><td> $5 4 . 0 0 ^ { \pm 2 . 0 0 }$   $5 7 . 3 3 ^ { \pm 1 . 1 5 }$ </td><td></td><td></td></tr><tr><td colspan="4">Methods With Refinement</td></tr><tr><td>Self-Refine</td><td> $5 0 . 6 7 ^ { \pm 1 . 1 5 }$ </td><td> $5 2 . 6 7 ^ { \pm 1 . 1 5 }$ </td><td> $5 4 . 0 0 ^ { \pm 0 . 0 0 }$ </td></tr><tr><td>RB (Seq)</td><td> $5 4 . 0 0 ^ { \pm 0 . 0 0 }$ </td><td> $5 4 . 0 0 ^ { \pm 0 . 0 0 }$ </td><td> $5 6 . 6 7 ^ { \pm 1 . 1 5 }$ </td></tr><tr><td>ReMe (Seq)</td><td> $5 4 . 6 7 ^ { \pm 1 . 1 5 }$ </td><td> $5 6 . 6 7 ^ { \pm 1 . 1 5 }$ </td><td> $5 8 . 6 7 ^ { \pm 1 . 1 5 }$ </td></tr><tr><td>RefCon</td><td> $5 7 . 3 3 ^ { \pm 1 . 1 5 }$ </td><td> $5 8 . 0 0 ^ { \pm 0 . 0 0 }$ </td><td> $\mathbf { 6 0 . 0 0 ^ { \pm 0 . 0 0 } }$ </td></tr></table>

## 4.4 Generalization to Coding Tasks

We evaluate RefCon’s generalizability on SWEbench Verified (Mini), a challenging benchmark where feedback is sparse and trajectory correctness is difficult to assess without executing test cases. Although recurrence is less explicit than in AppWorld, software engineering tasks still involve underlying routines where memory is useful. Using GLM 4.6 with mini-SWE-agent (Yang et al., 2024), Table 3 shows RefCon outperforms all baselines including ACE and ReMe (Sequential) which use ground-truth. We also include vanilla self-refine as a baseline to isolate gains from context-evolving methodology versus standard refinement alone.

While vanilla self-refine shows consistent but slow gains with a low initial success rate matching

ReAct, ReasoningBank (Sequential) and RefCon do not see an immediate increase in the second iteration. Instead, they optimize trajectory efficiency during this stage, which subsequently facilitates higher success rates by the third iteration. Ref-Con’s strong first-iteration performance validates its memory extraction and retrieval mechanisms. Its self-contrast reasoning across multiple trajectories enables it to outperform ACE even when ACE is provided ground-truth code patches.

## 4.5 RefCon Accuracy-Token Tradeoff

As shown in Figure 4, RefCon consistently achieves the highest accuracy across all methods (72.53% on ACE, 71.93% on ReasoningBank, 74.30% on ReMe) with minimal token overhead of only 0.17M–1.18M (≤3%) over sequential scaling, delivering gains up to +8.2%. ACE’s higher total consumption (4.5–4.74M vs. 2.5–3.8M for ReasoningBank and ReMe) stems from its twostage reflection-curation process and injecting all retrieved memories into the ReAct loop, compounding both new and cached token costs.

RefCon’s efficiency comes from where its costs are concentrated. Trajectory generation, the dominant phase by volume, is cache-friendly due to structural redundancy across ReAct turns, keeping new token costs low throughout the generation phase. The additional overhead over sequential scaling is localized almost entirely to the extraction step, which processes three full trajectories of fresh, non-overlapping content and yields predominantly cache misses. This separation makes RefCon’s trade-off favorable: trajectory generation via self-refinement is no more costly than sequential scaling, while the one-time extraction overhead purchases a meaningful jump in memory quality by consolidating all three refined trajectories, spending compute precisely where richer input improves downstream accuracy.

![](images/228a1df99e4936c50b94c21d13279ab8079c38e83a3441c132f155fcd8746ed4.jpg)

![](images/096bc0112a82aaec31ed844070b55d22b74f1ec45936f072677e7e244d59309c.jpg)  
Figure 4: Accuracy (Avg@3) and prompt token tradeoff for each MaTTS methods.

![](images/0fdc7a6d8618cbed131fd524758aad756d6dc104fff12a8b6f708f2a0eca6f44.jpg)  
Figure 5: Scaling of RefCon and DivCon.

## 4.6 Scaling Law

As shown in Figure 5, RefCon and DivCon share a baseline at k = 1 (63.7%) and achieve nearly identical performance at $k = 2 ( \approx 7 1 . 5 \% )$ , but diverge sharply thereafter: RefCon exhibits steady gains reaching 76.0% at k = 5, while DivCon plateaus around 71.0%, suggesting additional diversityseeking trajectories introduce conflicting noise beyond the initial gain. This indicates that while an initial round of diversity exploration provides a useful inductive signal, consistent refinement is more effective at converting increased compute into sustained accuracy. Solution quality proves a more reliable driver than solution diversity as trajectories scale. RefCon is therefore the more scalable strategy under larger compute budgets, as each additional trajectory reliably improves memory consolidation input rather than introducing variance the extractor must filter out, making it particularly well-suited to settings where inference compute can be scaled up.

## 5 Conclusion

We introduced RefCon, a memory-aware testtime scaling framework that unifies sequential self-refinement with parallel self-contrast to extract higher-quality memories for context-evolving agents without relying on gold labels. Notably, Ref-Con’s label-free extraction surpasses some groundtruth baselines and maintains its effectiveness across different model scales. Across AppWorld and BFCL-V3, RefCon consistently delivers the strongest or near-strongest performance and favorable accuracy-token trade-offs, while our DivCon variant highlights when explicit diversity can help but also reveals stability and scaling limits. The successful application of RefCon to the SWEbench benchmark further validates its generalizability to intricate, real-world software engineering environments. By capturing recurring routines and comparing trajectories, our approach enables effective memory extraction even in long-horizon tasks where distinguishing correct from incorrect reasoning is inherently difficult. These findings emphasize that improving memory quality at test time is a practical, compute-efficient path to continual agent improvement, and they motivate further work on more robust diversity generation and adaptive scaling strategies.

## Limitations

Our study has two main limitations. First, to address potential biases in model scale, we conducted experiments using both GLM-4.6 (large), Gemma 4 31B (medium), Qwen3.5 9B (small). Although incorporating three scales reduces the limitation of a single-model study, the generalizability of our results to models with different architectures or tuning styles is not yet fully guaranteed. We aim to broaden the scope of our evaluations to include more diverse model families in future iterations. Second, our test-time scaling analysis only evaluates up to five trajectories (k ≤ 5), which means we cannot yet characterize behavior under larger compute budgets, including whether performance continues to improve, plateaus, or degrades beyond this range.

## Ethical Considerations

This work does not raise specific ethical concerns. Our contributions focus on developing memory extraction frameworks for effective context-evolving agent. All experiments are conducted on publicly available benchmarks with open-source models, without involving human subjects, sensitive data, or privacy-related information. No potential conflicts of interest are present. While there is a potential for generated memories to occasionally degrade model performance, this occurrence is infrequent and can be mitigated through proper extraction or retrieval thresholds.

## References

Yuzheng Cai, Siqi Cai, Yuchen Shi, Zihan Xu, Lichao Chen, Yulei Qin, Xiaoyu Tan, Gang Li, Zongyi Li, Haojia Lin, Yong Mao, Ke Li, and Xing Sun. 2025a. Training-free group relative policy optimization. Preprint, arXiv:2510.08191.

Zhicheng Cai, Xinyuan Guo, Yu Pei, Jiangtao Feng, Ya-Qin Zhang, Wei-Ying Ma, Mingxuan Wang, and Hao Zhou. 2025b. Flex: Inheritable intelligence via forward learning from scaling experience. arXiv preprint.

Zouying Cao, Jiaji Deng, Li Yu, Weikang Zhou, Zhaoyang Liu, Bolin Ding, and Hai Zhao. 2025. Remember me, refine me: A dynamic procedural memory framework for experience-driven agent evolution. Preprint, arXiv:2512.10696.

Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. 2020. A simple framework for contrastive learning of visual representations. In

Proceedings of the 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, pages 1597–1607. PMLR.

Google DeepMind. 2026. Gemma 4: Our most capable open models to date. Accessed: 2026-05-07.

Marius Hobbhahn. 2024. SWEBench-verifiedmini: A size-optimized subset of SWE-bench Verified. https://github.com/mariushobbhahn/ SWEBench-verified-mini. Accessed: 2026-05-07.

Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, Tom Hennigan, Eric Noland, Katherine Millican, George van den Driessche, Bogdan Damoc, Aurelia Guy, Simon Osindero, Karen Simonyan, Erich Elsen, and 3 others. 2022. An empirical analysis of compute-optimal large language model training. In Advances in Neural Information Processing Systems.

Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, and Yejin Choi. 2020. The curious case of neural text degeneration. In International Conference on Learning Representations.

Carlos E Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik R Narasimhan. 2024. SWE-bench: Can language models resolve real-world github issues? In The Twelfth International Conference on Learning Representations.

Jiarun Liu, Shiyue Xu, Yang Li, Shangkun Liu, Yongli Yu, and Peng Cao. 2025. Unifying dynamic tool creation and cross-task experience sharing through cognitive memory architecture. Preprint, arXiv:2512.11303.

Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Shashank Gupta, Bodhisattwa Prasad Majumder, Katherine Hermann, Sean Welleck, Amir Yazdanbakhsh, and Peter Clark. 2023. Self-refine: Iterative refinement with self-feedback. In Thirty-seventh Conference on Neural Information Processing Systems.

OpenAI. 2024. text-embedding-3-small model. OpenAI Developers Documentation. Accessed 2026-02- 26.

Siru Ouyang, Jun Yan, I-Hung Hsu, Yanfei Chen, Ke Jiang, Zifeng Wang, Rujun Han, Long T. Le, Samira Daruki, Xiangru Tang, Vishy Tirumalashetty, George Lee, Mahsan Rofouei, Hangfei Lin, Jiawei Han, Chen-Yu Lee, and Tomas Pfister. 2025. Reasoningbank: Scaling agent self-evolving with reasoning memory. Preprint, arXiv:2509.25140.

Joon Sung Park, Joseph C. O’Brien, Carrie J. Cai, Meredith Ringel Morris, Percy Liang, and Michael S.

Bernstein. 2023. Generative agents: Interactive simulacra of human behavior. In In the 36th Annual ACM Symposium on User Interface Software and Technology (UIST ’23), UIST ’23, New York, NY, USA. Association for Computing Machinery.

Shishir G Patil, Huanzhi Mao, Fanjia Yan, Charlie Cheng-Jie Ji, Vishnu Suresh, Ion Stoica, and Joseph E. Gonzalez. 2025. The berkeley function calling leaderboard (BFCL): From tool use to agentic evaluation of large language models. In Forty-second International Conference on Machine Learning.

Long Phan, Alice Gatti, Ziwen Han, Nathaniel Li, Josephina Hu, Hugh Zhang, Chen Bo Calvin Zhang, Mohamed Shaaban, John Ling, Sean Shi, Michael Choi, Anish Agrawal, Arnav Chopra, Adam Khoja, Ryan Kim, Richard Ren, Jason Hausenloy, Oliver Zhang, Mantas Mazeika, and 1093 others. 2025. Humanity’s last exam. Preprint, arXiv:2501.14249.

Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik R Narasimhan, and Shunyu Yao. 2023. Reflexion: language agents with verbal reinforcement learning. In Thirty-seventh Conference on Neural Information Processing Systems.

Charlie Victor Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. 2025. Scaling LLM test-time compute optimally can be more effective than scaling parameters for reasoning. In The Thirteenth International Conference on Learning Representations.

Adam Stein, Matthew Trager, Benjamin Bowman, Michael Kleinman, Aditya Chattopadhyay, Wei Xia, and Stefano Soatto. 2025. Experience-guided adaptation of inference-time reasoning strategies. Preprint, arXiv:2511.11519.

Mirac Suzgun, Mert Yuksekgonul, Federico Bianchi, Dan Jurafsky, and James Zou. 2025. Dynamic cheatsheet: Test-time learning with adaptive memory.

Xiangru Tang, Tianrui Qin, Tianhao Peng, Ziyang Zhou, Daniel Shao, Tingting Du, Xinming Wei, He Zhu, Ge Zhang, Jiaheng Liu, Xingyao Wang, Sirui Hong, Chenglin Wu, and Wangchunshu Zhou. 2025. AGENT KB: A hierarchical memory framework for cross-domain agentic problem solving. In ICML 2025 Workshop on Collaborative and Federated Agentic Workflows.

Qwen Team. 2026. Qwen3.5: Accelerating productivity with native multimodal agents.

Harsh Trivedi, Tushar Khot, Mareike Hartmann, Ruskin Manku, Vinty Dong, Edward Li, Shashank Gupta, Ashish Sabharwal, and Niranjan Balasubramanian. 2024. AppWorld: A controllable world of apps and people for benchmarking interactive coding agents. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 16022–16076, Bangkok, Thailand. Association for Computational Linguistics.

Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. 2024. Voyager: An open-ended embodied agent with large language models. Transactions on Machine Learning Research.

Jiongxiao Wang, Qiaojing Yan, Yawei Wang, Yijun Tian, Soumya Smruti Mishra, Zhichao Xu, Megha Gandhi, Panpan Xu, and Lin Lee Cheong. 2025a. Reinforcement learning for self-improving agent with skill library. Preprint, arXiv:2512.17102.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V. Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. 2023. Self-consistency improves chain of thought reasoning in language models. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net.

Zora Zhiruo Wang, Apurva Gandhi, Graham Neubig, and Daniel Fried. 2025b. Inducing programmatic skills for agentic tasks. In Second Conference on Language Modeling.

Jason Wei, Zhiqing Sun, Spencer Papay, Scott McKinney, Jeffrey Han, Isa Fulford, Hyung Won Chung, Alex Tachard Passos, William Fedus, and Amelia Glaese. 2025. Browsecomp: A simple yet challenging benchmark for browsing agents. Preprint, arXiv:2504.12516.

Cheng-Kuang Wu, Zhi Rui Tam, Chieh-Yen Lin, Yun-Nung Chen, and Hung-yi Lee. 2024. Streambench: Towards benchmarking continuous improvement of language agents. In Advances in Neural Information Processing Systems, volume 37, pages 107039– 107063. Curran Associates, Inc.

Rong Wu, Xiaoman Wang, Jianbiao Mei, Pinlong Cai, Daocheng Fu, Cheng Yang, Licheng Wen, Xuemeng Yang, Yufan Shen, Yuxin Wang, and Botian Shi. 2025. Evolver: Self-evolving llm agents through an experience-driven lifecycle. Preprint, arXiv:2510.16079.

BY Yan, Chaofan Li, Hongjin Qian, Shuqi Lu, and Zheng Liu. 2025. General agentic memory via deep research. arXiv preprint arXiv:2511.18423.

Cheng Yang, Xuemeng Yang, Licheng Wen, Daocheng Fu, Jianbiao Mei, Rong Wu, Pinlong Cai, Yufan Shen, Nianchen Deng, Botian Shi, Yu Qiao, and Haifeng Li. 2025. Learning on the job: An experience-driven, self-evolving agent for long-horizon tasks. Preprint, arXiv:2510.08002.

John Yang, Carlos E Jimenez, Alexander Wettig, Kilian Lieret, Shunyu Yao, Karthik R Narasimhan, and Ofir Press. 2024. SWE-agent: Agent-computer interfaces enable automated software engineering. In The Thirty-eighth Annual Conference on Neural Information Processing Systems.

Z.ai. 2024. GLM-4.6 model guide. Z.ai Documentation. Accessed 2026-02-26.

Guibin Zhang, Muxin Fu, Kun Wang, Guancheng Wan, Miao Yu, and Shuicheng YAN. 2025a. G-memory: Tracing hierarchical memory for multi-agent systems. In The Thirty-ninth Annual Conference on Neural Information Processing Systems.

Qizheng Zhang, Changran Hu, Shubhangi Upasani, Boyuan Ma, Fenglu Hong, Vamsidhar Kamanuru, Jay Rainton, Chen Wu, Mengmeng Ji, Hanchen Li, Urmish Thakker, James Zou, and Kunle Olukotun. 2025b. Agentic context engineering: Evolving contexts for self-improving language models. Preprint, arXiv:2510.04618.

Andrew Zhao, Daniel Huang, Quentin Xu, Matthieu Lin, Yong-Jin Liu, and Gao Huang. 2024. Expel: Llm agents are experiential learners. In Thirty-Eighth AAAI Conference on Artificial Intelligence, AAAI 2024, Thirty-Sixth Conference on Innovative Applications ofArtificial Intelligence, IAAI 2024, Fourteenth Symposium on Educational Advances in Artificial Intelligence, EAAI 2024, February 20-27, 2024, Vancouver, Canada, pages 19632–19642. AAAI Press.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. 2023. Judging LLM-as-a-judge with MT-bench and chatbot arena. In Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track.

King Zhu, Hanhao Li, Siwei Wu, Tianshun Xing, Dehua Ma, Xiangru Tang, Minghao Liu, Jian Yang, Jiaheng Liu, Yuchen Eleanor Jiang, Changwang Zhang, Chenghua Lin, Jun Wang, Ge Zhang, and Wangchunshu Zhou. 2025. Scaling test-time compute for llm agents. Preprint, arXiv:2506.12928.

## Appendix

## A RefCon and DivCon Detailed Results

Tables 4 and 5 present the performance of contextevolving agents using smaller language model architectures, specifically Gemma 4 31B and Qwen3.5 9B. By replicating our main results from Table 1 (GLM 4.6) at these scales, we provide a focused comparison across the most competitive methods. We observed a significant performance decline on the AppWorld benchmark when using Qwen3.5 9B, whereas the performance drop on BFCL-V3 remained marginal. In settings without ground-truth labels, RefCon consistently outperformed other context-evolving agents across both model scales, even surpassing ACE (with ground-truth) on AppWorld. However, these gains were less pronounced than those achieved with the larger GLM-4.6 model, suggesting that while RefCon’s scaling remains effective, smaller models still struggle with the nuances of high-fidelity memory extraction and reuse. Notably, some contextevolving methods even degraded performance on AppWorld when applied to Qwen3.5 9B. Gemma 4 31B has significantly better performance, similar to GLM 4.6. The benefit of memory reuse is also more aparrent in this model compared to Qwen3.5 9B, showing the importance of having strong base model. The underperformance of DivCon relative to RefCon further suggests that exploring diversity through prompting alone remains a significant challenge for smaller models, reinforcing the necessity of RefCon’s structured refinement and contrastive approach.

In Table 1 and Table 2, we report Avg@3 for both RefCon and DivCon. For RefCon, the reported Avg@3 comes from the third run (after the second refinement), whereas for DivCon it comes from the first run. We execute RefCon three times, yielding first-, second-, and final-run results for each trial; the same setup is used for DivCon across its first and second runs. Because RefCon applies self-refinement, later runs can improve accuracy (Table 6). By contrast, DivCon emphasizes diversity, which does not always lead to better scores (Table 7). Therefore, we report the final run for RefCon and the first run for DivCon for all experiments.

We further analyze the results in Table 2 (ReMe) in Figure 6 to demonstrate RefCon’s superior ability to resolve recurring problems. The AppWorld dataset consists of three related tasks within a single scenario, varying only in their environmental states. By grouping these tasks according to their chronological execution order, we observe that all MaTTS methods demonstrate performance gains in subsequent tasks, indicating successful memory transfer. Notably, RefCon, DivCon, and Parallel scaling exhibit the steepest improvement curves between the first and second tasks, suggesting that the high-quality memory generated via self-contrast is instrumental in solving related tasks. While RefCon and Sequential scaling both achieve high initial performance on the first task due to selfrefinement, Sequential scaling fails to maintain consistent improvements in later tasks, likely due to the limitations of extracting memory from a single trajectory.

Table 4: Performance of context-evolving agent methods on AppWorld and BFCL-V3 Using Gemma 4 31B.
<table><tr><td rowspan="3">Methods</td><td colspan="4">AppWorld</td><td colspan="2">BFCL-V3</td></tr><tr><td colspan="2">Avg@3</td><td colspan="2">Pass@3</td><td rowspan="2">Avg@3</td><td rowspan="2">Pass@3</td></tr><tr><td>TGC</td><td>SGC</td><td>TGC</td><td>SGC</td></tr><tr><td colspan="7">Baseline</td></tr><tr><td>ReAct</td><td> $7 3 . 6 8 ^ { \pm 2 . 0 5 }$ </td><td> $4 5 . 6 1 ^ { \pm 2 . 4 8 }$ </td><td>92.98</td><td>78.95</td><td> $6 1 . 3 3 ^ { \pm 0 . 9 4 }$ </td><td>72.00</td></tr><tr><td colspan="7">With Gold Labels/Ground-Truth</td></tr><tr><td>ACE (with Ground-Truth)</td><td> $\mathbf { 8 0 . 1 2 ^ { \pm 2 . 1 9 } }$ </td><td> $\pm 4 . 3 9 ^ { \pm 4 . 9 6 }$ </td><td>96.49</td><td>89.47</td><td> $\mathbf { 6 9 . 3 3 ^ { \pm 3 . 4 0 } }$ </td><td>84.00</td></tr><tr><td>ReMe (Sequential Scaling)</td><td> $7 8 . 3 6 ^ { \pm 3 . 5 1 }$ </td><td> $5 1 . 4 6 ^ { \pm 3 . 7 2 }$ </td><td>94.74</td><td>84.21</td><td> $6 6 . 6 7 ^ { \pm 2 . 4 9 }$ </td><td>80.00</td></tr><tr><td>ReMe (Parallel Scaling)</td><td> $7 6 . 0 2 ^ { \pm 2 . 9 2 }$ </td><td> $4 8 . 5 4 ^ { \pm 4 . 9 6 }$ </td><td>92.98</td><td>84.21</td><td> $6 4 . 6 7 ^ { \pm 4 . 1 1 }$ </td><td>78.00</td></tr><tr><td colspan="7">Without Gold Labels/Ground-Truth</td></tr><tr><td>ACE (without Ground-Truth)</td><td> $7 4 . 2 7 ^ { \pm 2 . 4 8 }$ </td><td> $4 6 . 2 0 ^ { \pm 2 . 4 8 }$ </td><td>91.23</td><td>73.68</td><td> $6 2 . 0 0 ^ { \pm 2 . 0 5 }$ </td><td>74.00</td></tr><tr><td>ReasoningBank</td><td> $7 3 . 1 0 ^ { \pm 3 . 5 1 }$ </td><td> $4 4 . 7 4 ^ { \pm 3 . 7 2 }$ </td><td>89.47</td><td>73.68</td><td> $6 0 . 6 7 ^ { \pm 2 . 8 3 }$ </td><td>72.00</td></tr><tr><td>ReasoningBank (Sequential Scaling)</td><td> $7 6 . 0 2 ^ { \pm 2 . 6 3 }$ </td><td> $4 8 . 5 4 ^ { \pm 3 . 5 1 }$ </td><td>92.98</td><td>78.95</td><td> $6 4 . 0 0 ^ { \pm 2 . 4 9 }$ </td><td>76.00</td></tr><tr><td>ReasoningBank (Parallel Scaling)</td><td> $7 5 . 2 0 ^ { \pm 3 . 0 7 }$ </td><td> $4 7 . 3 7 ^ { \pm 4 . 3 8 }$ </td><td>91.23</td><td>78.95</td><td> $6 3 . 3 3 ^ { \pm 3 . 3 0 }$ </td><td>74.00</td></tr><tr><td>DivCon</td><td> $7 6 . 6 1 ^ { \pm 3 . 9 4 }$ </td><td> $5 0 . 2 9 ^ { \pm 3 . 7 2 }$ </td><td>94.74</td><td>84.21</td><td> $6 5 . 3 3 ^ { \pm 3 . 3 0 }$ </td><td>80.00</td></tr><tr><td>RefCon</td><td> $\mathbf { 7 9 . 5 3 ^ { \pm 4 . 3 8 } }$ </td><td> ${ \pm } 3 . 2 2 ^ { \pm 5 . 4 2 }$ </td><td>96.49</td><td>89.47</td><td> $\mathbf { 6 8 . 0 0 ^ { \pm 5 . 2 5 } }$ </td><td>84.00</td></tr></table>

Table 5: Performance of context-evolving agent methods on AppWorld and BFCL-V3 Using Qwen3.5 9B.
<table><tr><td rowspan="3">Methods</td><td colspan="4">AppWorld</td><td colspan="2">BFCL-V3</td></tr><tr><td colspan="2">Avg@3</td><td colspan="2">Pass@3</td><td rowspan="2">Avg@3</td><td rowspan="2">Pass@3</td></tr><tr><td>TGC</td><td>SGC</td><td>TGC</td><td>SGC</td></tr><tr><td colspan="7">Baseline</td></tr><tr><td>ReAct</td><td> $1 8 . 7 1 ^ { \pm 2 . 1 9 }$ </td><td> $1 0 . 5 3 ^ { \pm 4 . 3 0 }$ </td><td>29.82</td><td>15.79</td><td> $5 3 . 3 3 ^ { \pm 3 . 4 0 }$ </td><td>72.00</td></tr><tr><td colspan="7">With Gold Labels/Ground-Truth</td></tr><tr><td>ACE (with Ground-Truth)</td><td> $\mathbf { 1 8 . 1 3 ^ { \pm 4 . 6 0 } }$ </td><td> $\mathbf { 8 . 7 7 ^ { \pm 2 . 4 8 } }$ </td><td>29.82</td><td>15.79</td><td> ${ \bf 5 6 . 0 0 } ^ { \pm 4 . 3 2 }$ </td><td>80.00</td></tr><tr><td>ReMe (Sequential Scaling)</td><td> $1 7 . 5 4 ^ { \pm 2 . 4 8 }$ </td><td> $7 . 8 9 ^ { \pm 2 . 4 8 }$ </td><td>26.32</td><td>15.79</td><td> $5 4 . 0 0 ^ { \pm 3 . 2 7 }$ </td><td>76.00</td></tr><tr><td>ReMe (Parallel Scaling)</td><td> $1 6 . 9 6 ^ { \pm 3 . 5 1 }$ </td><td> $7 . 0 2 ^ { \pm 2 . 4 8 }$ </td><td>24.56</td><td>10.53</td><td> $5 2 . 0 0 ^ { \pm 3 . 2 7 }$ </td><td>74.00</td></tr><tr><td colspan="7">Without Gold Labels/Ground-Truth</td></tr><tr><td>ACE (without Ground-Truth)</td><td> $1 5 . 2 0 ^ { \pm 2 . 4 8 }$ </td><td> $5 . 8 5 ^ { \pm 2 . 4 8 }$ </td><td>21.05</td><td>10.53</td><td> $4 0 . 6 7 ^ { \pm 3 . 4 0 }$ </td><td>60.00</td></tr><tr><td>ReasoningBank</td><td> $1 4 . 0 4 ^ { \pm 2 . 4 8 }$ </td><td> $4 . 6 8 ^ { \pm 2 . 4 8 }$ </td><td>19.30</td><td>10.53</td><td> $3 7 . 3 3 ^ { \pm 2 . 4 9 }$ </td><td>54.00</td></tr><tr><td>ReasoningBank (Sequential Scaling)</td><td> $1 6 . 9 6 ^ { \pm 2 . 4 8 }$ </td><td> $8 . 7 7 ^ { \pm 4 . 9 6 }$ </td><td>22.81</td><td>15.79</td><td> $4 5 . 3 3 ^ { \pm 0 . 9 4 }$ </td><td>68.00</td></tr><tr><td>ReasoningBank (Parallel Scaling)</td><td> $1 7 . 5 4 ^ { \pm 2 . 4 8 }$ </td><td> $1 2 . 2 8 ^ { \pm 4 . 9 6 }$ </td><td>22.81</td><td>15.79</td><td> $4 3 . 3 3 ^ { \pm 5 . 2 5 }$ </td><td>62.00</td></tr><tr><td>DivCon</td><td> $1 6 . 9 6 ^ { \pm 4 . 3 8 }$ </td><td> $1 2 . 2 8 ^ { \pm 2 . 4 8 }$ </td><td>26.32</td><td>21.05</td><td> $4 6 . 0 0 ^ { \pm 4 . 9 0 }$ </td><td>72.00</td></tr><tr><td>RefCon</td><td> $\mathbf { 1 9 . 8 8 ^ { \pm 5 . 4 2 } }$ </td><td> $1 2 . 2 8 ^ { \pm 4 . 9 6 }$ </td><td>31.58</td><td>21.05</td><td> ${ \bf 5 0 . 6 7 ^ { \pm 8 . 3 8 } }$ </td><td>80.00</td></tr></table>

## B Memory Examples

We use ReAct framework as base and utilize ACE, ReasoningBank, and ReMe as baseline contextevolving agents. ACE (Zhang et al., 2025b) extracts and curates memories through a reflector–curator agent pipeline but directly uses all stored memories without a dedicated retrieval mechanism. ReasoningBank (Ouyang et al., 2025) introduces the MaTTS paradigm (both parallel and sequential), which performs memory extraction from both successful and failed trajectories and retrieves memories using embedding-based similarity. ReMe (Cao et al., 2025) further improves retrieval by incorporating a retrieve–rerank–rewrite pipeline to filter and refine retrieved memories. These methods also differ in their test-time scaling strategies, including parallel and sequential trajectory refinement. ACE doesn’t have default test-time scaling strategies, while ReMe has different parallel and sequential scaling strategy compared to Reasoning-Bank. For parallel, it extracts memories from best, worst, and comparison of best-worst trajectories. While for sequential scaling, it only retries failed trajectory and use it as cautionary memory for the next generation.

ACE, ReasoningBank, and ReMe use different memory schemas. ACE stores each memory as a section and content pair (Table 8); the section acts as a stable bucket (e.g., strategy, tool usage, or caution) so related memories can be organized and retrieved consistently. ReasoningBank represents memory with title, description, and content (Table 9); this structure separates a short identifier (title), a concise summary for quick screening (description), and the full actionable detail (content), which is useful when comparing many candidate memories during retrieval. ReMe uses when\_to\_use and content (Table 10); this emphasizes direct applicability by explicitly encoding the trigger condition first, followed by the action or rule to execute. Furthermore, ReMe retrieved memories will be rewritten into a coherent step-by-step guide paragraph. All examples are extracted from trajectories generated for the same query: "How many unique songs are there across my Spotify song library, albums library and all playlists?" $( f a c 2 9 I d \_ I )$ The underlying agent frameworks differ, but all examples use the same scaling method, RefCon.

Table 6: RefCon accuracy across runs for each method.
<table><tr><td rowspan="2">Methods</td><td colspan="4">AppWorld</td></tr><tr><td colspan="2">Avg@3</td><td colspan="2">Pass@3</td></tr><tr><td></td><td>TGC</td><td>SGC</td><td>TGC</td><td>SGC</td></tr><tr><td></td><td>ACE</td><td></td><td></td><td></td></tr><tr><td>First Run</td><td> $6 8 . 4 0 ^ { \pm 5 . 7 2 }$ </td><td> ${ \bf 4 3 . 8 7 ^ { \pm 6 . 5 7 } }$ </td><td>85.96</td><td>68.42</td></tr><tr><td>Second Run (First Refinement)</td><td> $7 0 . 1 7 ^ { \pm 6 . 5 7 }$ </td><td> $4 2 . 1 ^ { \pm 8 . 5 7 }$ </td><td>92.98</td><td>84.21</td></tr><tr><td>Third Run (Second Refinement)</td><td> $7 2 . 5 3 ^ { \pm 5 . 8 2 }$ </td><td> ${ \bf 4 3 . 8 7 ^ { \pm 6 . 5 7 } }$ </td><td>92.98</td><td>78.95</td></tr><tr><td colspan="3">ReasoningBank</td><td></td><td></td></tr><tr><td>First Run</td><td> $7 0 . 7 6 ^ { \mp 0 . 8 0 }$ </td><td> $4 7 . 3 7 ^ { \pm 8 . 6 1 }$ </td><td>89.47</td><td>84.21</td></tr><tr><td>Second Run (First Refinement)</td><td> $\mathbf { 7 1 . 9 3 ^ { \pm 2 . 8 6 } }$ </td><td> ${ \bf 5 0 . 8 7 ^ { \pm 1 0 . 8 5 } }$ </td><td>91.23</td><td>84.21</td></tr><tr><td>Third Run (Second Refinement)</td><td> $\mathbf { 7 1 . 9 3 ^ { \pm 3 . 8 0 } }$ </td><td> $4 9 . 1 3 ^ { \pm 1 3 . 1 3 }$ </td><td>92.98</td><td>84.21</td></tr><tr><td colspan="3"></td><td colspan="2"></td></tr><tr><td>First Run</td><td>ReMe  $6 9 . 5 9 ^ { \pm 2 . 9 8 }$ </td><td> $5 0 . 8 7 ^ { \pm 2 . 4 8 }$ </td><td>89.47</td><td>78.95</td></tr><tr><td>Second Run (First Refinement)</td><td> $7 1 . 3 5 ^ { \pm 4 . 3 8 }$ </td><td> $4 9 . 1 3 ^ { \pm 2 . 4 8 }$ </td><td>91.23</td><td>78.95</td></tr><tr><td>Third Run (Second Refinement)</td><td> $\mathbf { 7 4 . 3 0 ^ { \pm 0 . 9 4 } }$ </td><td> ${ \pm 2 . 6 7 ^ { \pm 7 . 4 5 } }$ </td><td>92.98</td><td>78.95</td></tr></table>

Table 7: DivCon accuracy across runs for each method.
<table><tr><td rowspan="2">Methods</td><td colspan="4">AppWorld</td></tr><tr><td colspan="2">Avg@3</td><td colspan="2">Pass@3</td></tr><tr><td></td><td>TGC</td><td>SGC</td><td>TGC</td><td>SGC</td></tr><tr><td colspan="5"> $\mathbf { A C E }$ </td></tr><tr><td>First Run</td><td> $\mathbf { 6 9 . 0 0 ^ { \pm 2 . 1 6 } }$ </td><td> $\pm 2 . 1 0 ^ { \pm 4 . 3 3 }$ </td><td>91.23</td><td>78.95</td></tr><tr><td>Second Run (First Alternative)</td><td> $6 5 . 5 0 ^ { \pm 2 . 9 8 }$ </td><td> $4 0 . 3 7 ^ { \pm 6 . 5 7 }$ </td><td>89.47</td><td>73.68</td></tr><tr><td colspan="3">ReasoningBank</td><td></td><td></td></tr><tr><td>First Run</td><td> $\mathbf { 7 6 . 0 3 ^ { \pm 2 . 2 1 } }$ </td><td> ${ \pm } 2 . 6 7 ^ { \pm 0 . 0 0 }$ </td><td>94.74</td><td>89.47</td></tr><tr><td>Second Run (First Alternative)</td><td> $7 0 . 1 7 ^ { \pm 2 . 5 0 }$ </td><td> $4 7 . 3 3 ^ { \pm 7 . 4 5 }$ </td><td>92.98</td><td>89.47</td></tr><tr><td colspan="3"> $\mathbf { R e M e }$ </td><td></td><td></td></tr><tr><td>First Run Second Run (First Alternative)</td><td> $\mathbf { 7 1 . 9 3 ^ { \pm 5 . 7 6 } }$   $6 7 . 2 5 ^ { \pm 2 . 1 9 }$ </td><td> $\pm 7 . 3 7 ^ { \pm 4 . 2 9 }$   $3 8 . 6 0 ^ { \pm 2 . 4 8 }$ </td><td>92.98 96.49</td><td>78.95 89.47</td></tr></table>

## C Example of Successful RefCon Improvement

This section presents a task that RefCon solves but sequential scaling fails on. For analysis, we compare retrieved memories (Table 11) and trajectory snippets (Table 12). The task is 0d8a4ee\_2 with the query: "Send the following phone message to my siblings and roommates, who do not have a venmo account, "Get on venmo please!". Although each method retrieves five memories, we display only two that differ and materially influence trajectory generation. The memories retrieved by sequential scaling push the agent to skip verification and jump directly to sending messages. In contrast, RefCon retrieves memories that encourage API exploration and case checking. This suggests that RefCon extracts more useful memories by self-contrasting across multiple trajectories, while sequential scaling extracts weak guidance from a single (possibly failed) trajectory. The effect is visible in the rollouts: sequential scaling brute-forces API field combinations, fails, and then sends messages to all\_contact. On the other hand, RefCon first inspects the data structure, then refines the code and correctly identifies contacts\_without\_venmo.

![](images/200e57abaa7603c44f3de4b412ac20da2df66cf8bfec6a3efc3df4ce483028fe.jpg)  
Figure 6: Avg@3 improvement across similar task within the same scenario on AppWorld sorted by chronological run order.

Table 13 illustrates how a single API misuse in Step 12 propagates into a completely wrong final answer. Both trajectories follow identical steps up to and including Step 11, correctly collecting 79 unique song IDs and filtering to 18 R&B candidates. The divergence arises when retrieving play counts: the successful trajectory reads play\_count directly from show\_song, which exposes it as part of the public song schema, while the failed trajectory makes an additional call to show\_song\_privates under the plausible assumption that per-user listening data would reside there. However, show\_song\_privates only returns interaction flags (liked, reviewed, in\_song\_library, downloaded), and contains no play\_count field. The .get(’play\_count’, 0) fallback silently assigns a count of zero to all 18 songs, so the subsequent sort in Step 13 is applied to a uniform list and produces an arbitrary ordering. Crucially, the failed trajectory raises no error at any point, the code is syntactically valid, the API calls succeed, and the output is well-formed. It makes class of mistake particularly difficult to detect without careful cross-referencing of API schemas. This case is where self-contrast reasoning in RefCon shows its benefit.

## D Hyperparameter Details

We generally employ a scaling factor of 3 (except 2 for DivCon). To promote trajectory diversity, we use parallel rollouts with varying temperatures (Holtzman et al., 2020; Wang et al., 2023). Across all methods, we prompt the LLM to extract at most five memories per extraction step. For parallel scaling, we rollout three trajectories at temperatures of 0.7, 0.85, and 1.0 to ensure diversity; otherwise, the default temperature is 0.7. For RefCon and ReMe, we utilize a top-k retrieval and reranking strategy with specific usage and utility pruning thresholds. Method-specific configuration are as follows:

• ReasoningBank: Uses top-1 query retrieval, returning up to five memories for each query.

• ACE: The playbook is capped at 50 memories, and ground-truth usage is disabled during scaling.

• ReMe & RefCon: We use a top-k retrieval of 10, the only choose top-5 after reranking. We set a deduplication similarity threshold ϵ = 0.5, and pruning thresholds α = 5n (usage) and β = 0.5 (utility), where n represents the scaling factor.

## E Prompt Details

Prompt details for memory extraction and memory retrieval for each method: ACE, Reasoning-Bank, and ReMe can be found on the original paper. We only modify the number of trajectory provided in the prompt template for RefCon, DivCon, and parallel scaling, while the instruction remains the same. For memory extraction, RefCon adopts ReasoningBank’s self-contrast reasoning and ReMe’s memory format as presented in Table 14. We also provide the self-refine and self-diversity prompts for AppWorld in Table 15, and the corresponding prompts for BFCL-V3 in Table 16. In our experiments, applying the simpler AppWorld-style refinement prompt to BFCL-V3 was ineffective, since the trajectories showed little to no change. To address this, we include both a critique prompt and an ideation prompt to better guide trajectory refinement, as shown in Table 16. However in some cases, self-refinement can even degrade trajectories that were already correct. We also include prompt for SWE-bench Verified (Mini) refinement in Table 17.

Table 8: Example of memories extracted using ACE with RefCon scaling.  
![](images/4046ac7b3ff6a971e722b9a26b8c13bc9606ada157609800b77ed7c289e4e8ba.jpg)

Table 9: Example of memories extracted using ReasoningBank with RefCon scaling.  
![](images/a51fbb24a71634a98163e0b62c8e19b29823c6ccdec138db437dbd043d7e5211.jpg)

Table 10: Example of memories extracted using ReMe with RefCon scaling.
<table><tr><td>Original Memory</td></tr><tr><td>Content: All trajectories successfully implemented pagination with while loops and page_index incrementing. This consistent pattern across all successful trajectories shows that proper pagination handling is critical for complete data retrieval. The pattern of checking for empty responses to break the loop is reliable. When To Use: When handling authentication tokens in multi-step workflows Content: All trajectories successfully extracted and reused the access_token from login responses across multiple API</td></tr><tr><td>calls. The consistent pattern of storing spotify_access_token = login_result[&#x27;access_token&#x27;] and passing it to subsequent calls demonstrates proper token management. None of the trajectories encountered token expiration issues, suggesting the token lifespan was sufficient for the workflow duration.</td></tr><tr><td>When To Use: When handling data deduplication across multiple sources Content: All trajectories correctly used sets for deduplication, but Trajectory 3 demonstrated the most elegant approach by updating the set with comprehensions within the helper function. Trajectories 1, 2, and 4 used more verbose loop structures. The set-based approach was universally successful, proving it&#x27;s the right pattern for this type of unique counting task across multiple data sources.</td></tr><tr><td>When To Use: When deciding whether to explore API documentation before implementation Content: Trajectory 3 successfully skipped explicit API documentation exploration and went directly to implementation,</td></tr><tr><td>demonstrating that prior knowledge or confidence in API structure can eliminate unnecessary exploration steps. However, Trajectories 1, 2, and 4 took a more cautious approach by examining API docs first, which is safer for unfamiliar APIs but adds overhead. The doc exploration didn&#x27;t change the implementation approach, suggesting it was redundant in this case.</td></tr><tr><td>Rewritten Memory To accurately count unique songs across your Spotify library, you&#x27;ll need to implement a comprehensive data aggregation</td></tr><tr><td>approach that handles multiple library sources with potential overlaps. Start by authenticating with Spotify&#x27;s API using stored credentials, then systematically explore the available endpoints for songs, albums, and playlists. The key is to use a set data structure to track unique song IDs as you paginate through each library source. Implement while loops to handle pagination for songs library, albums library, and all playlists separately, adding each song ID to your set which automatically handles deduplication across all sources. This ensures you capture every unique song exactly once, even if the same track appears in multiple playlists or albums. After processing all paginated results from all three library types,</td></tr></table>

Table 11: Retrieved memory comparison of sequential (failed) and RefCon (success) scaling.
<table><tr><td>Retrieved Memory (Sequential Scaling)</td></tr><tr><td>When To Use: When you encounter verification issue in Venmo Content: If you encounter issues with optional verification steps (like checking Venmo status), proceed with the main task (sending messages) to ensure completion. The primary goal of delivering the message is more important than perfect</td></tr><tr><td>filtering, especially when verification may be unreliable. When To Use: When filtering contacts by a condition and finding no matches, but the user has requested to send a specific</td></tr><tr><td>message Content: If no recipients match the specified filter criteria, consider alternative interpretations of the user&#x27;s request before</td></tr><tr><td>concluding the task. The message content itself may provide clues about the intended audience.</td></tr><tr><td>Retrieved Memory (RefCon) When To Use: When filtering contact on Venmo</td></tr><tr><td>Content: Across trajectories that attempted Venmo filtering, the reliable pattern was to query users by email first, then validate exact email matches (case-insensitive) before excluding a contact from messaging. Trajectories that treated any</td></tr><tr><td>non-empty search response as a match risked false positives, while trajectories that performed exact email verification were more robust. In this setup, an empty search result consistently signaled no Venmo account.</td></tr></table>

Table 12: Trajectory comparison with memory retrieved from sequential (failed) and RefCon (success) scaling.  
Sequential Scaling (Failed) RefCon (Successful)   
Assistant: Assistant:   
contacts\_without\_venmo = [] all\_contacts = siblings + roommates   
for contact in all\_contacts: contacts\_without\_venmo = []   
email = contact[’email’] for contact in all\_contacts:   
search\_result = apis.venmo.search\_users( search\_result = apis.venmo.search\_users(   
access\_token=venmo\_access\_token, access\_token=venmo\_access\_token,   
query=email, page\_limit=1) query=contact[’email’])   
if not search\_result: if not search\_result:   
contacts\_without\_venmo.append(contact) contacts\_without\_venmo.append(contact)   
print(contacts\_without\_venmo) print(contacts\_without\_venmo)   
Environment: [] Environment: []   
Assistant: Assistant:   
contacts\_without\_venmo = [] print(apis.venmo.search\_users(   
for contact in all\_contacts: access\_token=venmo\_access\_token,   
email = contact[’email’] query=’kevin.blac@gmail.com’))   
search\_result = apis.venmo.search\_users( Environment: [{"first\_name": "Kevin", "email":   
access\_token=venmo\_access\_token, "kevin.blac@gmail.com", . . . }]   
query=email) Assistant:   
if not search\_result: all\_contacts = siblings + roommates   
contacts\_without\_venmo.append(contact) contacts\_without\_venmo = []   
print(contacts\_without\_venmo) for contact in all\_contacts:   
Environment: [] search\_result = apis.venmo.search\_users(   
Assistant: access\_token=venmo\_access\_token,   
contacts\_without\_venmo = [] query=contact[’email’])   
for contact in all\_contacts: found = any(user[’email’] == contact[’email’]   
full\_name f"{contact[’first\_name’]} for user in search\_result)   
{contact[’last\_name’]}" if not found:   
search\_result = apis.venmo.search\_users( contacts\_without\_venmo.append(contact)   
access\_token=venmo\_access\_token, print(contacts\_without\_venmo)   
query=full\_name) Environment: [{"contact\_id": 801, "first\_name": "Martin",   
if not search\_result:   
. }]   
contacts\_without\_venmo.append(contact)   
print(contacts\_without\_venmo)   
Environment: []

Table 13: Step-by-step comparison of the successful and failed trajectories. Steps 1–11 are identical in both; the slight divergence occurs at Step 12, propagating to a wrong final answer at Step 13.  
Step Successful Trajectory Failed Trajectory   
1–2 (Identical) Retrieve account passwords; log in to Spotify to obtain access\_token.   
3–7 (Identical) Inspect API schemas for show\_song, show\_song\_privates, show\_song\_library,   
show\_album\_library, show\_playlist\_library.   
8–11 (Identical) Collect 79 unique song IDs from all three libraries; filter to 18 R&B songs via   
show\_song() genre check.   
12 Read play\_count from show\_song Read play\_count from   
show\_song\_privates   
rb\_songs\_with\_play\_count = []   
for song\_id in all\_song\_ids:   
rb\_songs\_with\_play\_count = []   
song\_info = apis.spotify for song in rb\_song\_details:   
.show\_song(song\_id=song\_id)   
song\_id = song['song\_id']   
if song\_info and 'r&b' in private\_info = apis.spotify   
song\_info['genre'].lower(): .show\_song\_privates(   
rb\_songs\_with\_play\_count.append({ song\_id=song\_id,   
'title': song\_info['title'],   
access\_token=spotify\_access\_token)   
'play\_count': song\_info['play\_count'] if private\_info:   
}) play\_count = private\_info   
.get('play\_count', 0)   
show\_song returns a rich public schema rb\_songs\_with\_play\_count.append({   
including play\_count: {song\_id, title, 'title': song['title'],   
'play\_count': play\_count   
album\_id, duration, artists, genre, })   
play\_count, rating, ...}   
Genre filtering and play count retrieval are show\_song\_privates only exposes per  
done in a single loop. All 18 R&B songs user interaction flags: {liked, reviewed,   
in\_song\_library, downloaded}   
receive their correct play counts.   
There is no play\_count field. The   
.get(’play\_count’, 0) fallback silently re  
turns 0 for all 18 songs, producing no error   
and no warning.   
13 Sorted by real play counts: All counts equal 0; order is arbitrary:   
["Mysteries of the Silent Sea", ["Shadows of the Past"   
"Crimson Veil", "When Fate Becomes a Foe",   
"Haunted Memories", "The Curse of Loving You",   
"Fire and Ice"] "Lost in a Moment's Grace"]

Table 14: Memory extraction prompt used for our context-evolving agent utilizing RefCon. We adopt ReMe memory format, but utilize self-contrast reasoning to extract memory.  
Self-Contrast Prompt   
You are an expert AI analyst comparing multiple step sequences which might be successful or failed to extract differential   
insights.   
Your task is to compare and contrast these trajectories to identify the most useful and generalizable strategies as memory   
items using self-contrast reasoning.   
Focus on critical decision points, technique variations, and approach differences.   
COMPARATIVE ANALYSIS FRAMEWORK:   
- DECISION CONTRAST: Compare critical decisions made in success vs failure cases   
- TECHNIQUE VARIATIONS: Identify different approaches and their outcomes   
- TIMING DIFFERENCES: Analyze when certain actions were taken and their impact   
- SUCCESS FACTORS: Extract what specifically made the difference   
EXTRACTION PRINCIPLES:   
- Frame comparisons as PRINCIPLES as well as case-specific SOLUTIONS   
- Identify PATTERNS that differentiate effective vs ineffective approaches   
- Extract RULES that can guide future similar situations   
- Focus on UNDERLYING MECHANISMS rather than surface-level difference   
Trajectory 1:   
{trajectory\_1}   
Trajectory N:   
{trajectory\_n}   
OUTPUT FORMAT:   
Generate up to 5 comparative insights as JSON objects:   
[   
{   
"when\_to\_use": "Specific scenarios where this comparative insight applies",   
"experience": "Detailed comparison highlighting why success approach works better",   
"tags": ["comparative\_analysis", "success\_factors", "relevant\_keywords"],   
"confidence": 0.8,   
"step\_type": "reasoning|action|observation|decision",   
"tools\_used": ["list", "of", "tools"]   
<sup>}</sup><sub>]</sub>

Table 15: RefCon and DivCon trajectory generation prompt for AppWorld.
<table><tr><td>Self-Refine Prompt for AppWorld</td></tr><tr><td>Let&#x27;s carefully re-examine the previous trajectory, including your reasoning steps and action taken. Pay special attention to whether you used the best API sequence and whether you used the API correctly. If you find inconsistencies, correct them.</td></tr><tr><td>If everything seems correct, make it more efficient. Now, solve the same problem again from scratch. Self-Diversity Prompt for AppWorld</td></tr><tr><td>Below is the previous trajectory, the solution might be correct or wrong. Now solve the same problem using a DIFFERENT reasoning approach. Focus on exploring alternative strategies.</td></tr></table>

Table 16: RefCon and DivCon trajectory generation prompt for BFCL-V3.
<table><tr><td rowspan=1 colspan=1>Critique Trajectory Prompt for Self-Refine</td></tr><tr><td rowspan=1 colspan=1>You are an expert reviewer analyzing an AI assistant&#x27;s multi-turn tool-calling trajectory. Your job is to identify mistakes,missed actions, and suboptimal decisions. For each turn in the trajectory, evaluate:1. Did the assistant call the appropriate tools? If a user requested an action (e.g., book, cancel, update), did the assistantactually make a tool call, or did it just respond with text?2. Were the tool arguments correct? Check for wrong parameter values, missing required arguments, or arguments thatcontradict the user&#x27;s request.3. Did the assistant use information from previous tool responses correctly? For example, if a lookup returned an ID, didthe assistant use that ID in subsequent calls?4. Were there any unnecessary or redundant tool calls?5. Did the assistant follow the logical sequence of operations? (e.g., lookup before booking, authenticate before accessingprotected resources)Be specific about which turns have issues and what should be done differently. If a turn looks correct, briefly note it as OK.Focus most of your analysis on turns that seem problematic.IMPORTANT: You must respond ONLY with your critique. Do not attempt to solve the task yourself.</td></tr><tr><td rowspan=1 colspan=1>Self-Refine Prompt for BFCL-V3</td></tr><tr><td rowspan=1 colspan=1>You are an AI assistant tasked with completing a multi-turn tool-calling objective. A reviewer has analyzed your previousattempt and provided a critique.1. Read the critique carefully to understand the mistakes made in the previous trajectory.2. Trust the Schemas: The critique is a helpful guide, but your provided tool schemas are the absolute truth. If the critiquesuggests using a tool or parameter that does not exist in your schemas, ignore it3. Do not ask the user for any additional information/clarification, you are authorized to do everything in behalf of the user.4. Restart the task from the beginning. Your environment state has been reset.5. You are currently at the start of the task, but the provided previous trajectory covers the entire multi-turn interaction. DoNOT execute or simulate future steps ahead of time. Focus ONLY on the current task instruction.6. You can use related memory to guide your reasoning to solve the problem.</td></tr><tr><td rowspan=1 colspan=1>Ideate Solution Prompt for Self-Diversity</td></tr><tr><td rowspan=1 colspan=1>You are an expert AI brainstorming assistant analyzing a previous interaction trajectory. Your job is to read the user&#x27;srequest and the previous approach, and then propose a DIFFERENT but valid reasoning approach or sequence of tool callsthat solves the same problem. Focus your analysis on:1. Understanding the user&#x27;s core intent.2. Identifying the strategy used in the previous trajectory.3. Proposing alternative strategies, using different tools, or a different sequence of operations that could achieve the samegoal successfully.Be specific about the proposed alternative approach.IMPORTANT: You must respond ONLY with your alternative idea. Do not attempt to solve the task yourself.</td></tr><tr><td rowspan=1 colspan=1>Self-Diversity Prompt for BFCL-V3</td></tr><tr><td rowspan=1 colspan=1>You are an AI assistant tasked with completing a multi-turn tool-calling objective. A reviewer has analyzed your previousattempt and provided an alternative idea for a different reasoning approach.1. Read the alternative idea carefully and use it to solve the problem.2. Trust the Schemas: The alternative idea is a helpful guide, but your provided tool schemas are the absolute truth. If thealternative idea suggests using a tool or parameter that does not exist in your schemas, ignore it.3. Do not ask the user for any additional information/clarification, you are authorized to do everything in behalf of the user.4. Restart the task from the beginning. Your environment state has been reset.5. You are currently at the start of the task, but the provided previous trajectory covers the entire multi-turn interaction. DoNOT execute or simulate future steps ahead of time. Focus ONLY on the current task instruction.6. You can use related memory to guide your reasoning to solve the problem.</td></tr></table>

Table 17: RefCon trajectory generation prompt for SWE-bench Verified (Mini).
<table><tr><td>Critique Trajectory Prompt for Self-Refine</td></tr><tr><td>You are an expert reviewer analyzing an AI assistant&#x27;s multi-turn software engineering trajectory. Your job is to identify mistakes, missed files, and suboptimal debugging decisions. For each turn in the trajectory, evaluate: 1. Did the assistant explore the repository effectively? Did it locate the relevant source files and classes, or did it waste turns on unrelated directories? 2. Was the bug localization accurate? Check if the assistant correctly identified the root cause of the issue before attempting</td></tr><tr><td>a fix. 3. Did the assistant use the environment and test tools correctly? For example, if a reproduction script was created, did the assistant analyze the output to guide the patch? 4. Was the generated patch functional and minimal? Identify if the assistant introduced unnecessary changes or failed to</td></tr><tr><td>follow the repository&#x27;s coding style. 5. Did the assistant follow a logical debugging sequence? (e.g., search → reproduce → fix → verify via tests) Be specific about which turns have issues and what should be done differently. If a turn looks correct, briefly note it as OK.</td></tr><tr><td>Focus most of your analysis on turns that seem problematic. IMPORTANT: You must respond ONLY with your critique. Do not attempt to solve the task yourself. Self-Refine Prompt for SWE-bench Verified (Mini)</td></tr><tr><td>You are an AI assistant tasked with resolving a GitHub issue in a complex repository. A reviewer has analyzed your previous attempt and provided a critique.</td></tr><tr><td>1. Read the critique carefully to understand the logic gaps or coding errors made in the previous trajectory. 2. Trust the Codebase: The critique is a helpful guide, but the current file content and test execution results are the absolute truth. If the critique suggests a fix that contradicts the actual code logic, prioritize the codebase.</td></tr></table>