# XferAgent: Cross-Agent Experience Transfer for Self-Improving LLM Agents

Weihao Li<sup>1∗</sup>, Jun Bai<sup>2,3∗</sup>, Shijie Chen<sup>1</sup>, Qincheng Lu<sup>2,3</sup>, Xinyu Wang<sup>2,3</sup>, Ziyang Song<sup>4†</sup>

<sup>1</sup>McCormick School of Engineering, Northwestern University

<sup>2</sup>School of Computer Science, McGill University

<sup>3</sup>Mila – Quebec AI Institute

<sup>4</sup>School of Electrical Engineering and Computer Science, Ohio University

## Abstract

Large language model (LLM) agents can reuse experience from prior task executions to improve performance on subsequent tasks. Although existing methods enable cross-agent experience sharing, differences in agents’ execution histories and task requirements make shared experiences difficult to reuse effectively. We propose XferAgent, a framework for cross-agent experience transfer that organizes reusable experience in local and shared experience graphs. XferAgent reconciles compatible contributions from heterogeneous agents while avoiding unreliable merges. For each new task, it selects shared experiences and integrates them with the target agent’s local experience to support task execution. Experiments across diverse task domains show improvements over representative memory-based methods and effective experience transfer between heterogeneous agent frameworks. Cumulative task success also increases over successive task passes as agents update and reuse their experiences.

## 1 Introduction

Large language model (LLM) agents are increasingly used to solve complex tasks through multistep reasoning, tool use, and interaction with external environments (Yao et al., 2022; Hong et al., 2024; Qin et al., 2024). Solving these tasks often requires iterative execution, exploration, and feedback-driven refinement before an agent reaches a successful outcome (Shinn et al., 2023). Across tasks, agents often encounter recurring reasoning patterns, tool-use procedures, and failure modes. Organizing these execution outcomes as reusable experience can help agents reuse effective strategies and avoid repeating previous failures on subsequent tasks. Recent work has therefore explored experience-driven self-improvement, where agents distill prior executions into reusable experience to improve performance on subsequent tasks (Zhao et al., 2024; Wang et al., 2025; Cao et al., 2026).

Existing memory-based approaches support local self-improvement by organizing experience from an agent’s past executions. Reflexion turns task feedback into self-reflection stored in episodic memory for subsequent attempts (Shinn et al., 2023), while A-MEM dynamically links and updates memory notes as new experience arrives (Xu et al., 2026b). ExpGraph organizes reusable skills and lessons in an experience graph (Feng et al., 2026), while ReasoningBank distills successful and failed executions into reusable reasoning memories for subsequent tasks (Ouyang et al., 2026). These approaches establish that experience accumulated from past executions can improve subsequent behavior. However, they primarily support local experience reuse, where an agent can draw only on the strategies and failure lessons contained in its own execution history. Relevant strategies or failure lessons absent from one agent’s execution history may already have been acquired by other agents, motivating experience reuse beyond a single agent’s local history.

Recent work extends experience-driven selfimprovement across agents by allowing one agent to reuse strategies and failure lessons learned by others. Agent KB aggregates trajectories from heterogeneous agent frameworks into a shared knowledge base (Tang et al., 2025), while FoT aggregates reasoning experience from independently operating agents into a shared insight library (Yao et al., 2026). These studies show that agents can benefit from shared experience, but differences in reasoning protocols, tools, and execution histories make cross-agent experience transfer difficult (Tang et al., 2025; Xu et al., 2026a; Li et al., 2026). The central challenge is therefore to reconcile experience from heterogeneous agents into reusable shared experience and identify which shared experiences are useful for a new task while considering the target agent’s local experience.

In this study, we propose XferAgent, a framework for Cross-Agent Experience Transfer across heterogeneous agents. XferAgent organizes each agent’s reusable experience in a local experience graph and experience contributed across agents in a shared experience graph. Through Cross-Agent Experience Reconciliation, XferAgent aligns compatible contributions while avoiding unreliable merges. For a new task, Task-Conditioned Experience Routing selects a task-relevant subset of the Shared Experience Graph to construct a connected routed experience subgraph. Local-Shared Experience Integration combines the routed subgraph with the target agent’s local experience to select task-relevant guidance. Following task evaluation, Experience Updates support subsequent experience reuse. Our main contributions are:

• We formulate cross-agent experience transfer in experience-driven self-improving LLM agents and introduce XferAgent to support experience reuse across heterogeneous agents.

• We develop Cross-Agent Experience Reconciliation to align compatible contributions from heterogeneous agents while avoiding unreliable merges.

• We combine Task-Conditioned Experience Routing and Local-Shared Experience Integration to select task-relevant experiences using both local and cross-agent outcome evidence.

• Experiments across diverse tasks and LLM backbones demonstrate improvements over representative memory-based methods, with additional evidence of cross-framework experience reuse and increasing cumulative task success across repeated executions.

## 2 Related Work

Experience-Driven Agent Learning and Structured Memory. Experience-driven selfimprovement enables LLM agents to leverage knowledge from prior executions without updating model parameters. Reflexion (Shinn et al., 2023) uses verbal feedback to guide subsequent attempts, ExpeL (Zhao et al., 2024) distills reusable insights from execution trajectories, and Reasoning-Bank (Ouyang et al., 2026) extracts strategies from successful and failed experiences. More structured approaches organize such experiences through evolving memory representations. A-MEM (Xu et al., 2026b) dynamically links related memories, AriGraph (Anokhin et al., 2025) integrates semantic and episodic knowledge into a graph-based world model, and ExpGraph (Feng et al., 2026) organizes skills and lessons through graph-based retrieval and utility-aware ranking. However, these methods primarily focus on experience reuse within individual agents.

Cross-Agent Experience Sharing and Transfer. Recent studies explore cross-agent experience reuse through shared knowledge representations. Agent KB (Tang et al., 2025) aggregates experiences from heterogeneous agent frameworks into a shared knowledge base, while FICAL (Wu et al., 2024) exchanges distilled knowledge for federated in-context learning. FoT (Yao et al., 2026) further distills distributed reasoning experiences into reusable metacognitive insights. These approaches facilitate cross-agent experience sharing, but experiences from heterogeneous agents may be difficult to align, and not all shared experiences are useful for a new task. XferAgent addresses these challenges by reconciling contributed experiences and integrating task-relevant shared experiences with the target agent’s local experience.

## 3 XferAgent

## 3.1 Problem Setup

We consider N heterogeneous agents, indexed by $i \in \{ 1 , \ldots , N \}$ . Agent i maintains a local experience graph $\acute { \mathcal { G } } _ { i } ^ { L , t }$ , while experience contributed across agents is organized in a shared experience graph $\mathcal { G } ^ { \bar { S } , t }$ , where t denotes the task step. Each node v contains experience text $e _ { v } .$ , a type $k _ { v } ~ \in$ {SUCCESS, FAILURE}, an embedding $h _ { v } = \phi ( e _ { v } )$ computed by a text encoder $\phi ,$ provenance, and outcome statistics. Edges represent semantic similarity and typed relations between experiences.

For a new task $x _ { t }$ assigned to agent i at time t, XferAgent retrieves experience from the graph state at t − 1, before the task is executed. After task evaluation produces an outcome score $r _ { t } ,$ XferAgent updates outcome statistics only for experiences included in the agent’s prompt for that task. The completed execution is also converted into new reusable experience for subsequent tasks.

As shown in Figure 1, XferAgent enables crossagent experience transfer through Cross-Agent

![](images/63381a97c995665bce77ead1cdfdfad39c5fc2863716be5e4ddb861901f59a8e.jpg)  
Figure 1: Overview of XferAgent. Agents construct local experience graphs by extracting experiences from successful and failed task executions. Cross-Agent Experience Reconciliation incorporates contributed experiences into a shared experience graph. Task-Conditioned Experience Routing selects task-relevant anchors and constructs a routed experience subgraph. Local–Shared Experience Integration combines local and shared candidates into a temporary fused graph and ranks experiences for task execution, followed by evaluation-based experience updates.

Experience Reconciliation (Section 3.2), Task-Conditioned Experience Routing (Section 3.3), Local-Shared Experience Integration (Section 3.4), and Experience Updates (Section 3.5).

## 3.2 Cross-Agent Experience Reconciliation

XferAgent converts each completed agent execution into reusable experience using a schemaconstrained mapper. A successful execution produces a SUCCESS experience, while an unsuccessful execution produces a FAILURE experience describing a failed strategy or required check rather than the incorrect answer itself. The local experience node may retain richer execution details, such as tool observations and intermediate planning details. When the experience is contributed to the shared experience graph, XferAgent removes task-specific and execution-sensitive information, including the raw task, final answer, and complete trajectory. For a contributed experience, a stable identifier links the local experience node to its representation in the shared experience graph.

To organize reusable experiences into a graph, XferAgent connects each experience to a sparse set of semantically related experiences. For each node, it retains its $K _ { \mathrm { e d g e } }$ most similar experiences whose cosine similarity exceeds $\theta _ { \mathrm { e d g e } }$ . A semantic edge is retained if either node selects the other, with cosine similarity used as its weight.

XferAgent reconciles each contributed experience before incorporating it into the shared experience graph. This is necessary because experience contributions from heterogeneous agents are not naturally aligned and may contain overlapping or semantically similar content. If a stable identifier matches an existing node in the shared experience graph, XferAgent merges the contributed experience with that node directly. Otherwise, XferAgent searches the shared experience graph for potential matches with the same experience type and sufficient semantic similarity. For a contributed experience $v ,$ the candidate set is

$$
\begin{array} { r } { \mathcal { C } ( v ) = \{ u \in \mathcal { V } ^ { S } : k _ { u } = k _ { v } , \cos ( h _ { u } , h _ { v } ) \geq \theta _ { \mathrm { r e c } } \} , } \end{array}\tag{1}
$$

where $\theta _ { \mathrm { r e c } }$ is the minimum cosine similarity for a candidate match. The reconciliation rule considers three cases: (1) If no candidate exists, XferAgent creates a new node. (2) If exactly one candidate exists, XferAgent merges the contributed experience with that node. (3) If multiple candidates exist, XferAgent merges the contributed experience with the most similar candidate only when its cosine similarity exceeds that of the second-most similar candidate by at least a similarity margin $\delta _ { \mathrm { r e c } } $ otherwise, it creates a new node to avoid an unreliable merge. When experiences are merged, their provenance is combined, while outcome statistics remain separate by agent. After reconciliation, XferAgent constructs semantic edges among the resulting nodes in the shared experience graph.

Beyond the undirected semantic edges, XferAgent also preserves typed relations between experiences. XferAgent adds a DERIVED-FROM relation from each newly contributed experience to every shared experience included in that task’s prompt. These relations have a fixed weight of 1.

## 3.3 Task-Conditioned Experience Routing

For a new task $x _ { t }$ , XferAgent constructs a taskrelevant routed experience subgraph instead of delivering the entire shared experience graph. Agent i receives the identifiers and embeddings of available shared experience nodes and computes their task similarity locally as $a ( x _ { t } , v ) = \cos ( \phi ( x _ { t } ) , h _ { v } )$ . It selects up to a fixed number of the highest-scoring positive matches above $\theta _ { \mathrm { { m a t c h } } }$ as anchors, which serve as starting points for graph search. Only the selected identifiers and similarity scores are sent for subgraph construction, while the raw task and its embedding remain with the agent.

Direct cosine similarity may miss experiences that are connected to relevant anchors but have low similarity to the task description. XferAgent therefore propagates task relevance from the anchors over the shared experience graph. The positive anchor scores are normalized to sum to one, defining $\rho _ { i } ( x _ { t } )$ , while other nodes receive zero weight. We use personalized PageRank (PPR) to compute graph relevance scores (Page et al., 1999):

$$
\begin{array} { r } { \mathbf { s } _ { i } ^ { \mathrm { g r a p h } , ( \ell + 1 ) } = \alpha \pmb { \rho } _ { i } ( x _ { t } ) + ( 1 - \alpha ) \boldsymbol { A } ^ { \top } \mathbf { s } _ { i } ^ { \mathrm { g r a p h } , ( \ell ) } , } \end{array}\tag{2}
$$

where A is the row-normalized weighted adjacency matrix of the shared experience graph, α is the restart probability that balances anchor relevance and propagation through graph edges, and ℓ indexes the iterations. After iterations, $s _ { i , v } ^ { \mathrm { { \bar { g r a p h } } } }$ denotes the graph relevance score of experience v, corresponding to its component in the resulting vector $\mathbf { s } _ { i } ^ { \mathrm { { \hat { g r a p h } } } }$

XferAgent also considers previous task outcomes when selecting shared experiences, since graph relevance alone does not capture past task performance. For target agent i, XferAgent computes a cross-agent outcome score from other agents’ past task outcomes. Each experience v in the shared experience graph stores a usage count $n _ { j v } ^ { S }$ and a cumulative task outcome score $z _ { j v } ^ { S }$ for each agent $j .$ These statistics cover only tasks in which experience v was included in agent $j ^ { \prime } \mathbf { s }$ prompt. The leave-one-agent-out (LOAO) calculation excludes agent i’s own outcomes. Therefore, XferAgent aggregates these statistics over all other agents $j \neq i \colon$

$$
\begin{array} { l } { { n _ { - i , v } ^ { S } = \displaystyle \sum _ { j \neq i } n _ { j v } ^ { S } , \quad z _ { - i , v } ^ { S } = \displaystyle \sum _ { j \neq i } z _ { j v } ^ { S } , } } \\ { { s _ { - i , v } ^ { \mathrm { o u t c o m e } } = \displaystyle \frac { z _ { - i , v } ^ { S } + \kappa u _ { 0 } } { n _ { - i , v } ^ { S } + \kappa } , } } \end{array}\tag{3}
$$

where $u _ { 0 }$ is the prior outcome score and κ controls the strength of this prior.

XferAgent also considers contributor support $s _ { - i , v } ^ { \mathrm { s u p p o r t } }$ , which counts the number of distinct agents other than i that contributed experience v. Repeated contributions from the same agent count only once. To favor less-used experiences,

XferAgent adds an exploration bonus to the crossagent outcome score based on other agents’ usage counts. The resulting exploration-enhanced crossagent outcome score is capped at one and denoted by $s _ { - i , v } ^ { \mathrm { e x p - o u t c o m e } }$ . XferAgent combines the graph relevance score, the exploration-enhanced outcome score, and the contributor support score to compute the final routing score:

$$
\begin{array} { r } { s _ { i , v } ^ { \mathrm { r o u t e } } = \beta _ { g } \mathrm { N o r m } ( s _ { i , v } ^ { \mathrm { g r a p h } } ) + \beta _ { o } \mathrm { N o r m } ( s _ { - i , v } ^ { \mathrm { e x p - o u t c o m e } } ) } \\ { + \beta _ { s } \mathrm { N o r m } ( s _ { - i , v } ^ { \mathrm { s u p p o r t } } ) , \qquad ( 4 ) } \end{array}
$$

where Norm applies min-max normalization separately to each term over the available shared experience nodes. The fixed weights $\beta _ { g } , \beta _ { o } .$ and $\beta _ { s }$ control graph relevance, exploration-enhanced outcome, and contributor support, respectively. Computational details appear in Appendix A.

XferAgent uses the routing scores to construct a connected routed experience subgraph. It starts from the anchor with the highest task similarity and repeatedly selects the highest-scoring unselected neighbor of the selected nodes. Each candidate is scored using its routing score plus a fixed-weight bonus for its strongest edge to a selected node. Expansion stops when the node budget B is reached or no eligible neighboring nodes remain. The resulting subgraph contains the selected shared experiences, their connecting edges, embeddings, and cross-agent outcome scores. If no qualifying anchor exists, XferAgent returns an empty subgraph.

## 3.4 Local-Shared Experience Integration

XferAgent integrates the routed experience subgraph with the target agent’s local experience graph to select experiences for the task prompt. XferAgent first selects candidate experiences separately from the local experience graph and the routed experience subgraph, using task-matching nodes as anchors. It applies PPR to each graph using its own edges and retains a bounded number of nodes with the highest positive graph relevance scores. If the routed experience subgraph is empty, the agent can still select experiences from its local experience graph.

XferAgent then builds a temporary fused graph to combine local and shared candidates. For each local candidate, it first checks for a match using its recorded shared identifier. If no identifier match exists, XferAgent merges it with the highestsimilarity experience of the same type already in the pool, provided their cosine similarity meets the fusion threshold. The pool includes previously processed local experiences. If no qualifying match exists, the experience remains separate. When a local experience matches a shared experience, it retains the richer local content while retaining the shared identifier for cross-agent outcome statistics.

XferAgent retains the graph edges among selected candidates and adds semantic links between unmatched local and shared experiences that meet the similarity threshold. It then applies PPR to the fused graph and retains a bounded number of candidates for final ranking. The fused graph is used only for the current task and does not modify either persistent experience graph. Merging matched experiences and adding semantic links can change the structure of the temporary fused graph. Xfer-Agent therefore applies PPR again and retains a bounded number of nodes with the highest positive graph relevance scores for final ranking. The fused graph does not modify the structure of either original graph. Appendix A provides details on edge construction, seed selection, and tie-breaking.

When the target agent has limited outcome history for an experience, outcomes from other agents provide an initial estimate. As local observations accumulate, they receive greater weight. Before executing task $x _ { t } ,$ , agent i reads how often each local experience was included in its prompt, $n _ { i v } ^ { L , t - 1 }$ and the sum of the corresponding task outcome scores, $z _ { i v } ^ { L , t - 1 }$ . For a candidate matched across both graphs, agent i uses the cross-agent outcome score $\bar { s } _ { - i , v } ^ { \mathrm { o u t c o m e , } t - 1 }$ as a shared prior and combines it with these local statistics:

$$
s _ { i , v } ^ { \mathrm { f u s e d } } = \frac { z _ { i v } ^ { L , t - 1 } + \kappa s _ { - i , v } ^ { \mathrm { o u t c o m e } } } { n _ { i v } ^ { L , t - 1 } + \kappa } ,\tag{5}
$$

Here, $s _ { i , v } ^ { \mathrm { f u s e d } }$ denotes the fused outcome score for $x _ { t } .$ , and κ controls the weight of the shared prior relative to local observations. A local-only candidate uses the default prior $u _ { 0 }$ while retaining its local usage count and cumulative outcome score. A shared-only candidate uses the cross-agent outcome score directly. All statistics are taken from the pre-task state $t - 1$

XferAgent ranks the remaining candidates using direct task similarity and the fused outcome score. To favor less-used experiences, it adds the exploration bonus from Section 3.3 to the fused outcome score. The resulting exploration-enhanced fused outcome score is capped at one and denoted by $s _ { i , v } ^ { \mathrm { e x p - f u s e d } , t }$ . For matched candidates, the exploration count combines local usage with other agents’ shared usage. Otherwise, only the available count is used. The final selection score is defined as follows:

$$
\begin{array} { r l } & { s _ { i , v } ^ { \mathrm { s e l e c t } } = \left( 1 - \lambda _ { u } \right) \mathrm { N o r m } ( a ( \boldsymbol { x } _ { t } , \boldsymbol { v } ) ) } \\ & { ~ + \lambda _ { u } \mathrm { N o r m } ( s _ { i , v } ^ { \mathrm { e x p - f u s e d } } ) . } \end{array}\tag{6}
$$

Here, $a ( x _ { t } , v )$ denotes task similarity. When candidates are merged, XferAgent retains the highest similarity score without recomputing it for the selected local content. The two terms are independently min-max normalized over the candidates selected by PPR on the fused graph. If a term has zero range, its values are clipped to the unit interval without rescaling (see Appendix A). The fixed weight $\lambda _ { u }$ controls their balance. XferAgent selects experiences in descending score order, subject to limits on the number of experiences and their total text length. XferAgent selects no experiences when no usable seed exists in the fused graph or the highest score is below a fixed threshold. Selected experiences are included as SUCCESS or FAILURE.

## 3.5 Experience Updates

Before executing task $x _ { t } ,$ XferAgent records the local and shared experiences included in agent i’s prompt. After task evaluation produces an outcome score $r _ { t } .$ , XferAgent updates each recorded experience’s usage count as $n _ { v } ^ { t } = n _ { v } ^ { t - 1 } + 1$ and cumulative outcome score as $z _ { v } ^ { t } = z _ { v } ^ { t - 1 } + r _ { t }$ . XferAgent updates only experiences included in the prompt, modifying agent i’s local statistics and its corresponding records in the shared experience graph. XferAgent then converts the completed execution into reusable experience as described in Section 3.2. The richer experience is added to the local experience graph, while its shareable representation is reconciled before entering the shared experience graph. XferAgent also links the new shared experience to each shared experience included in the task prompt through a DERIVED-FROM relation. These relations have a fixed weight of 1.

## 4 Experiments

## 4.1 Experimental Setup

Datasets and Backbones. We evaluate XferAgent primarily on GAIA (Mialon et al., 2024) and ALFWorld (Shridhar et al., 2020), which assess complementary agent capabilities: multi-step reasoning and tool use versus interactive task execution. GAIA contains 301 test tasks across three difficulty levels, evaluated with Qwen3-Next-80B-A3B-Instruct and GPT-5.6 Luna. ALFWorld comprises 274 tasks (140 seen and 134 unseen), evaluated with Qwen3.8-27B. To examine experience reuse in coding, we include a supplementary evaluation on MBPP (Austin et al., 2021) (60 coding tasks), using Llama-3.1-8B-Instruct.

Agent and Memory Configurations. For GAIA, we use SmolAgents (Roucher et al., 2025) for the main evaluation and additionally consider OWL (Hu et al., 2025) for cross-framework experience transfer. For ALFWorld, agents interact with the environment through its text-based action interface. MBPP employs three logical agents with fixed task assignments and a shared LLM backbone. Across ALFWorld and MBPP, we use a remote-memory budget of B = 8, at most 10 final experience items, and a nominal 2,400-token memory allowance. Memory construction and retrieval settings for these two benchmarks are detailed in Appendix C.

Baselines. We compare XferAgent with No Memory and five representative memory-based methods. No Memory evaluates the same base agent without experience retrieval, providing a reference for assessing the benefits of memory augmentation. The memory-based baselines include AgentKB (Tang et al., 2025) for cross-agent experience sharing, ReasoningBank (Ouyang et al., 2026) for reusable reasoning memories, ExpGraph (Feng et al., 2026) for graph-structured experience retrieval, Mem0 (Chhikara et al., 2025) for persistent memory management, and LightMem (Fang et al., 2026) for efficient memory consolidation. For cross-agent evaluation, we use benchmark-specific AgentKB configurations and adapt the other memory systems to support bounded retrieval across agent-owned memories. Implementation details for the supplementary-benchmark adapters are provided in Appendix C.

Evaluation Protocols and Metrics. For the main benchmark comparisons, we evaluate the reuse of pre-constructed experience memories, keeping memory content and historical outcome statistics fixed during testing. Within each benchmark, methods use the same reasoning backbone and evaluation cohort, with benchmark-specific memory construction and retrieval settings. We separately examine Experience Updates over three task passes on 165 GAIA validation tasks, reporting cumulative task recovery as memories are updated after task evaluation. Performance is measured by accuracy on GAIA, success rate on ALFWorld, and program pass rate on MBPP. Failed or unparseable cases are included in the evaluation denominators. Further details on GAIA Test evaluation and the other two benchmark protocols are provided in Appendices B and C, respectively.

## 4.2 Main Results

The main experiments assess cross-agent experience transfer across diverse tasks and LLM backbones, covering question answering (GAIA), interactive task execution (ALFWorld), and code generation (MBPP). Table 1 presents the overall performance of XferAgent against representative memory-based methods.

GAIA Results. On GAIA (301 test tasks), Xfer-Agent achieves the highest overall accuracy across both backbones. With GPT-5.6 Luna, it reaches 52.16%, exceeding No Memory by 3.99 percentage points and the strongest memory-based baseline, ExpGraph, by 4.65 points. With Qwen3-Next-80B-A3B-Instruct, XferAgent achieves 29.57%, compared with 28.90% for AgentKB and 27.57% for No Memory. Notably, all competing memorybased methods underperform No Memory with Luna, indicating that experience reuse does not necessarily improve task performance. Difficultylevel results are provided in Appendix B.

ALFWorld Task Completion. On 274 ALF-World tasks, XferAgent achieves the highest success rate of 88.7%, outperforming No Memory (84.7%) by 4.0 percentage points and slightly exceeding AgentKB and ExpGraph (both 88.3%). This extends cross-agent experience transfer to multi-step interactive task execution. Implementation details, including AgentKB’s retrieval of up to eight cross-agent candidates, are provided in Appendix C.

MBPP Code Generation. On 60 MBPP tasks, XferAgent achieves the highest pass rate of 53.3%, compared with 50.0% for both No Memory and AgentKB, and 48.3% for ExpGraph. Although the performance margin is relatively modest, the results provide complementary evidence of effective experience reuse in code generation. Together with GAIA and ALFWorld, these comparisons demonstrate the applicability of XferAgent across different task settings, with varying improvements over existing memory-based approaches.

Table 1: Overall performance across GAIA, ALFWorld, and MBPP (%). Metrics are accuracy, task success rate, and program pass rate, respectively. ALFWorld and MBPP follow benchmark-specific single-run protocols (Appendix C). Bold indicates the best result in each column.
<table><tr><td rowspan="2">Method</td><td colspan="2">GAIA</td><td>ALFWorld</td><td>MBPP</td></tr><tr><td>Qwen3-Next-80B A3B-Instruct</td><td>GPT-5.6 Luna</td><td>Qwen3.8-27B</td><td>Llama-3.1-8B Instruct</td></tr><tr><td>No Memory</td><td>27.57</td><td>48.17</td><td>84.7</td><td>50.0</td></tr><tr><td>AgentKB</td><td>28.90</td><td>46.51</td><td>88.3</td><td>50.0</td></tr><tr><td>ReasoningBank</td><td>26.91</td><td>46.18</td><td>85.0</td><td>46.7</td></tr><tr><td>ExpGraph</td><td>27.24</td><td>47.51</td><td>88.3</td><td>48.3</td></tr><tr><td>Mem0</td><td>25.91</td><td>44.85</td><td>84.7</td><td>45.0</td></tr><tr><td>LightMem</td><td>21.59</td><td>41.86</td><td>81.8</td><td>40.0</td></tr><tr><td>XferAgent</td><td>29.57</td><td>52.16</td><td>88.7</td><td>53.3</td></tr></table>

![](images/9127b0d65b23a47c514602666b87b3bcd3e80fa94fd8a82786610a1eb9ee1cb3.jpg)  
Figure 2: Cumulative task performance through experience evolution on GAIA validation (165 tasks). Passes 1–3 report cumulative task success across repeated executions. The independent No Memory baseline is connected to Pass 1 by dashed segments for visual comparison. Overall accuracy is computed over all 165 tasks.

## 4.3 Progressive Self-Improvement through Experience Evolution

To examine progressive self-improvement through experience evolution, we conduct three task passes over 165 GAIA validation tasks. After each execution, agents update their experience memories based on evaluation feedback for subsequent reuse. An independently evaluated No Memory baseline serves as a reference.

Across three task passes, cumulative overall accuracy increases from 56.36% to 67.88% and 74.55%, compared with 52.12% for No Memory (Figure 2). Gains occur across all difficulty levels, with Level 3 improving from 42.31% to 69.23%. These results reflect cumulative success on repeated tasks rather than generalization to unseen tasks, and the independent No Memory baseline does not control for repeated attempts.

## 4.4 Cross-Framework Experience Transfer

We examine cross-framework experience transfer between SmolAgents and OWL with 2, 4, and 8 agents on GAIA validation. The initial two agents use different frameworks, while additional agents are randomly assigned to either framework as the group expands. Each agent builds experiences from 20 questions sampled from a shared pool of 105, yielding 40, 80, and 160 agent–question learning instances with possible overlap. All configurations are evaluated on the same 60 held-out questions (18, 29, and 13 at Levels 1–3).

The macro-average accuracy increases from 40.4% with 2 agents to 47.2% with 4 agents and 55.0% with 8 agents (Figure 3). Gains occur across all difficulty levels, with Level 3 improving from 15.4% to 30.8%. Since agent count and total learning exposure increase together, their individual effects cannot be isolated, nor can the effect of framework heterogeneity. Sampling details are provided in Appendix C.3.

## 4.5 Ablation Studies

We conduct ablation studies on XferAgent using GAIA with GPT-5.6 Luna to assess the contributions of different experience sources and selection mechanisms. As shown in Table 2, Local Experience Only disables retrieval from the Shared Experience Graph, reducing overall accuracy from 52.16% to 46.84%, while Shared Experience Only excludes local experience retrieval and achieves 50.83%. This difference indicates that local and shared experiences provide complementary information, with local experience contributing more strongly in this setting. Furthermore, w/o Outcome-Based Ranking removes outcome-based scoring from experience ranking, decreasing accuracy to 48.17%. w/o Connectivity Constraint relaxes the connectivity requirement during shared experience selection, yielding 46.84%. These findings highlight the contributions of outcome-informed selection and connected subgraph routing to effective experience reuse.

![](images/9befd85f8df1bd2e45b0a00f728203df0a95312a21203f201a7015afb7709807.jpg)  
Figure 3: Cross-framework experience transfer with 2, 4, and 8 agents on 60 held-out GAIA validation questions. Gray bars represent the 2-agent baseline, while light and dark red segments indicate accuracy gains from 2 to 4 and 4 to 8 agents, respectively. Average denotes the unweighted mean of accuracy across the three difficulty levels.

Table 2: Ablation study of XferAgent on GAIA (301 test tasks) using GPT-5.6 Luna. Overall accuracy is reported in percentages, and ∆ denotes the change relative to the full system in percentage points.
<table><tr><td>Variant</td><td>Overall (%)</td><td>∆(pp)</td></tr><tr><td>XferAgent (Full)</td><td>52.16</td><td>1</td></tr><tr><td>Local Experience Only</td><td>46.84</td><td>-5.32</td></tr><tr><td>Shared Experience Only</td><td>50.83</td><td>-1.33</td></tr><tr><td>w/o Outcome-Based Ranking</td><td>48.17</td><td>-3.99</td></tr><tr><td>w/o Connectivity Constraint</td><td>46.84</td><td>-5.32</td></tr></table>

## 4.6 Sensitivity Analysis

We examine four key parameters of XferAgent’s experience reconciliation, routing, and integration on 45 held-out GAIA validation tasks using GPT-5.6 Luna, with experience graphs constructed from the remaining 120 tasks. The default settings are B = 24 (subgraph budget), K = 3 (final experience limit), $\lambda _ { u } ~ = ~ 0 . 4 0$ (ranking weight), and $\delta _ { \mathrm { r e c } } = 0 . 0 3$ (reconciliation margin). Each parameter is varied independently, with the others fixed. Reconciliation variants replay the recorded graph construction trajectory, while other configurations reuse the same constructed memories. Further protocol details are provided in Appendix D.

Increasing the final experience limit to $K = 5$ yields the highest observed accuracy of 62.22% (Table 3), while the default ranking weight $\lambda _ { u } = 0 . 4 0$ outperforms both tested extremes. Changes to B and $\delta _ { \mathrm { r e c } }$ do not alter the logged routing decisions or final experience selections, so their accuracy differences do not establish retrieval-related improvements. These comparisons characterize the observed parameter sensitivity rather than identifying optimal settings.

Table 3: Parameter sensitivity of XferAgent on 45 heldout GAIA validation tasks using GPT-5.6 Luna. One parameter is varied at a time, with others fixed at their default values. $\Delta$ denotes the accuracy change relative to the default in percentage points.
<table><tr><td>Parameter</td><td>Value</td><td>Overall (%)</td><td>∆ (pp)</td></tr><tr><td rowspan="3">B</td><td>8</td><td>51.11</td><td>-6.67</td></tr><tr><td>24 (default)</td><td>57.78</td><td></td></tr><tr><td>40</td><td>46.67</td><td>-11.11</td></tr><tr><td rowspan="3">K</td><td>1</td><td>51.11</td><td>-6.67</td></tr><tr><td>3 (default)</td><td>57.78</td><td></td></tr><tr><td>5</td><td>62.22</td><td>+4.44</td></tr><tr><td rowspan="3"> $\lambda _ { u }$ </td><td>0</td><td>53.33</td><td>-4.44</td></tr><tr><td>0.40 (default)</td><td>57.78</td><td></td></tr><tr><td>1</td><td>55.56</td><td>-2.22</td></tr><tr><td rowspan="3"> $\delta _ { \mathrm { r e c } }$ </td><td>0</td><td>51.11</td><td>-6.67</td></tr><tr><td>0.03 (default)</td><td>57.78</td><td></td></tr><tr><td>0.08</td><td>48.89</td><td>-8.89</td></tr></table>

## 5 Conclusion

We introduced XferAgent, a framework for Cross-Agent Experience Transfer that enables heterogeneous LLM agents to collaboratively reuse experiences beyond their individual memories. Experiments across diverse tasks demonstrate improvements over representative memory-based approaches, while repeated-task evaluations show increasing cumulative success through experience evolution. XferAgent provides a foundation for experience-level collaboration toward selfimproving agent ecosystems.

## Limitations

For cross-agent experience sharing, XferAgent removes task-specific and execution-sensitive information from contributed experiences while keeping raw tasks and task embeddings on the agent side. These measures limit the information shared but do not provide formal privacy guarantees. XferAgent currently focuses on text-based reusable experiences, and future work may extend the framework to more complex multimodal agent settings. Separating source and evaluation tasks does not rule out semantic overlap or exposure during model pretraining.

## References

Petr Anokhin, Nikita Semenov, Artyom Sorokin, Dmitry Evseev, Mikhail Burtsev, and Evgeny Burnaev. 2025. Arigraph: Learning knowledge graph world models with episodic memory for llm agents. Proceedings ofthe Thirty-Fourth International Joint Conference on Artificial Intelligence.

Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, and 1 others. 2021. Program synthesis with large language models. arXiv preprint arXiv:2108.07732.

Zouying Cao, Jiaji Deng, Li Yu, Weikang Zhou, Zhaoyang Liu, Bolin Ding, and Hai Zhao. 2026. Remember me, refine me: A dynamic procedural memory framework for experience-driven agent evolution. In Findings of the Association for Computational Linguistics: ACL 2026, pages 16803–16822.

Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet Singh, and Deshraj Yadav. 2025. Mem0: Building production-ready ai agents with scalable long-term memory. arXiv preprint arXiv:2504.19413.

Jizhan Fang, Xinle Deng, Haoming Xu, Ziyan Jiang, Yuqi Tang, Ziwen Xu, Shumin Deng, Yunzhi Yao, Mengru Wang, Shuofei Qiao, and 1 others. 2026. Lightmem: Lightweight and efficient memoryaugmented generation. In International Conference on Learning Representations, volume 2026, pages 98706–98729.

Tao Feng, Chongrui Ye, Tianyang Luo, Jingjun Xu, Xueqiang Xu, Haozhen Zhang, Zhigang Hua, Yan Xie, Shuang Yang, Ge Liu, and Jiaxuan You. 2026. ExpGraph: Model-agnostic experience learning with graph-structured memory for LLM agents. arXiv preprint arXiv:2605.30712.

Sirui Hong, Mingchen Zhuge, Jonathan Chen, Xiawu Zheng, Yuheng Cheng, Jinlin Wang, Ceyao Zhang, Zili Wang, Steven Ka Shing Yau, Zijuan Lin, Liyang Zhou, Chenyu Ran, Lingfeng Xiao, Chenglin Wu,

and Jürgen Schmidhuber. 2024. MetaGPT: Meta programming for a multi-agent collaborative framework. In The Twelfth International Conference on Learning Representations.

Mengkang Hu, Yuhang Zhou, Wendong Fan, Yuzhou Nie, Ziyu Ye, Bowei Xia, Tao Sun, Zhaoxuan Jin, Yingru Li, Zeyu Zhang, Yifeng Wang, Qianshuo Ye, Bernard Ghanem, Ping Luo, and Guohao Li. 2025. OWL: Optimized workforce learning for general multi-agent assistance in real-world task automation. In The Thirty-ninth Annual Conference on Neural Information Processing Systems.

Weihao Li, Jun Bai, and Ziyang Song. 2026. Fedagentke: Federated semantic knowledge evolution for heterogeneous agents. arXiv preprint arXiv:2607.21361.

Grégoire Mialon, Clémentine Fourrier, Thomas Wolf, Yann LeCun, and Thomas Scialom. 2024. Gaia: a benchmark for general ai assistants. In International Conference on Learning Representations, volume 2024, pages 9025–9049.

Siru Ouyang, Jun Yan, I Hsu, Yanfei Chen, Ke Jiang, Zifeng Wang, Rujun Han, Long Le, Samira Daruki, Xiangru Tang, and 1 others. 2026. Reasoningbank: Scaling agent self-evolving with reasoning memory. In International Conference on Learning Representations, volume 2026, pages 94327–94354.

Lawrence Page, Sergey Brin, Rajeev Motwani, and Terry Winograd. 1999. The pagerank citation ranking : Bringing order to the web. In The Web Conference.

Yujia Qin, Shihao Liang, Yining Ye, Kunlun Zhu, Lan Yan, Yaxi Lu, Yankai Lin, Xin Cong, Xiangru Tang, Bill Qian, and 1 others. 2024. Toolllm: Facilitating large language models to master 16000+ real-world apis. In International Conference on Learning Representations, volume 2024, pages 9695–9717.

Aymeric Roucher, Albert Villanova del Moral, Thomas Wolf, Leandro von Werra, and Erik Kaunismäki. 2025. ‘smolagents‘: a smol library to build great agentic systems. https://github.com/ huggingface/smolagents.

Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. 2023. Reflexion: Language agents with verbal reinforcement learning. Advances in neural information processing systems, 36:8634–8652.

Mohit Shridhar, Xingdi Yuan, Marc-Alexandre Côté, Yonatan Bisk, Adam Trischler, and Matthew Hausknecht. 2020. Alfworld: Aligning text and embodied environments for interactive learning. arXiv preprint arXiv:2010.03768.

Xiangru Tang, Tianrui Qin, Tianhao Peng, Ziyang Zhou, Yanjun Shao, Tingting Du, Xinming Wei, He Zhu, Ge Zhang, Jiaheng Liu, Xingyao Wang, Sirui Hong, Chenglin Wu, and Wangchunshu Zhou.

2025. AGENT KB: A hierarchical memory frame  
work for cross-domain agentic problem solving. In   
ICML 2025 Workshop on Collaborative and Feder  
ated Agentic Workflows.   
Zora Zhiruo Wang, Jiayuan Mao, Daniel Fried, and   
Graham Neubig. 2025. Agent workflow memory. In   
Forty-second International Conference on Machine   
Learning.   
Panlong Wu, Kangshuo Li, Junbao Nan, and Fangxin   
Wang. 2024. Federated in-context llm agent learning.   
arXiv preprint arXiv:2412.08054.   
Jiawei Xu, Arief Koesdwiady, Sisong Bei, Yan   
Han, Baixiang Huang, Dakuo Wang, Yutong Chen,   
Zheshen Wang, Peihao Wang, Pan Li, and 1 others.   
2026a. Rethinking the value of multi-agent work  
flow: A strong single agent baseline. arXiv preprint   
arXiv:2601.12307.   
Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao, Juntao   
Tan, and Yongfeng Zhang. 2026b. A-mem: Agentic   
memory for llm agents. Advances in Neural Informa   
tion Processing Systems, 38:17577–17604.   
Dixi Yao, Tahseen Rabbani, Manzil Zaheer, and Tian Li.   
2026. Federation over text: Insight sharing for multi  
agent reasoning. arXiv preprint arXiv:2604.16778.   
Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak   
Shafran, Karthik Narasimhan, and Yuan Cao. 2022.   
React: Synergizing reasoning and acting in language   
models. arXiv preprint arXiv:2210.03629.   
Andrew Zhao, Daniel Huang, Quentin Xu, Matthieu   
Lin, Yong-Jin Liu, and Gao Huang. 2024. Expel:   
Llm agents are experiential learners. In Proceedings   
of the AAAI Conference on Artificial Intelligence,   
volume 38, pages 19632–19642.

## A Algorithm and Implementation Details

Algorithm 1 describes XferAgent’s routing, integration, execution, and experience updates. REC-ONCILE maps new contributions into the Shared Experience Graph. ROUTECONNECTED applies PPR and grows a connected Routed Experience Subgraph under the node budget. INTEGRATE-ANDSELECT expands the local and routed graphs, constructs a temporary fused graph, and selects experiences for the prompt.

Here, i identifies the target agent for outcome and contributor statistics; $f _ { i }$ identifies its execution framework. $D _ { i }$ maps a completed execution to an experience node, and $P _ { i }$ produces its shareable representation. $\mathcal { T } _ { i }$ contains available node identifiers and embeddings; $\mathbf { \mathcal { A } } _ { i }$ contains anchors and their task similarities. $B _ { i }$ is the Routed Experience Subgraph, and $\mathcal { R } _ { i , t }$ contains the experiences selected for the task prompt.

Algorithm 1 XferAgent with Task-Conditioned   
Experience Routing   
Require: Local Experience Graphs $\{ \mathcal { G } _ { i } ^ { L } \}$ , Shared Experi  
ence Graph $\mathcal { G } ^ { S }$ , node budget B, frozen configuration Γ,   
evaluation mode   
1: for target task $x _ { t }$ assigned to agent i in commit order do   
2: $\mathcal { T } _ { i } \gets$ available node identifiers and embeddings from   
$\mathcal { G } ^ { S }$ under the authorization policy for $f _ { i }$   
3: $\mathbf { \mathcal { A } } _ { i } \gets \mathrm { A G E N T A N C H O R S } ( x _ { t } , \mathbf { \dot { \mathcal { T } } } _ { i } ; \dot { \Gamma } )$   
4: $\boldsymbol { B } _ { i } \gets$ ROUTECONNECTED $( \mathcal { G } ^ { S } , \mathcal { A } _ { i } , i , f _ { i } , B ; \Gamma )$   
5: if the Shared Experience Graph state changed during   
routing then   
$_ { 6 ; }$ Repeat lines $2 { \ - } 4$ using the updated Shared Experi  
ence Graph   
7: end if   
8: $\mathcal { R } _ { i , t } \gets \mathrm { I N T E G R A T E A N D S E L E C T } \left( x _ { t } , \mathcal { G } _ { i } ^ { L } , \mathcal { B } _ { i } , i ; \Gamma \right)$   
9: Record the Routed Experience Subgraph and exact   
local and shared experience nodes represented in $\mathcal { R } _ { i , t }$   
10: $y _ { t } \gets \mathrm { A g e n t } _ { f _ { i } } ( x _ { t } ; \mathcal { R } _ { i , t } )$   
11: $r _ { t } \gets Q ( x _ { t } , y _ { t } )$   
12: if experience updates are enabled then   
13: Update outcome statistics only for experiences in   
the prompt: local observations and agent-i records in the   
Shared Experience Graph   
14: $\tilde { v } _ { i , t } \gets D _ { i } ( x _ { t } , \hat { y } _ { t } , r _ { t } )$   
15: $\begin{array} { r } { \Vec { \mathcal { G } _ { i } ^ { L } } \gets \mathrm { I N S E R T A N D R E B U I L D } ( \mathcal { G } _ { i } ^ { L } , \widetilde { v } _ { i , t } ; \Gamma ) } \end{array}$   
16: $\overline { { v } } _ { i , t }  P _ { i } ( \widetilde { v } _ { i , t } )$   
17: $( \mathcal { G } ^ { S } , v ^ { S } ) \gets \mathrm { R E C O N C I L E } ( \mathcal { G } ^ { S } , \overline { { v } } _ { i , t } , i ; \Gamma )$   
18: Add a DERIVED-FROM relation from $v ^ { S }$ to   
each shared experience node included in the task prompt   
19: end if   
20: end for   
21: return $\{ \mathcal { G } _ { i } ^ { L } \}$ and $\mathcal { G } ^ { S }$

Evaluation and updates. Frozen-memory evaluation runs routing, integration, execution, and grading without updating statistics, constructing experiences, or writing back to the graphs. When updates are enabled, only experiences included in the prompt receive outcome-statistics updates. A failed task may lower their scores and produce a new FAILURE experience. A task whose prompt contains no experiences leaves prior statistics unchanged but may still produce a new experience after grading.

Routing interface. Before routing, the agent receives only available node identifiers and embeddings; experience text, provenance, outcome statistics, contributor identities, and graph structure remain in the Shared Experience Graph until the routed subgraph is delivered. AGENTANCHORS computes the task embedding locally. Positive anchor similarities are normalized to form the PPR restart distribution, and isolated nodes receive selfloops.

Local–Shared integration. Shared candidates initialize the pool. Local candidates are processed in descending source-graph PPR order, matching first by a recorded shared identifier when its candidate is present, otherwise by similarity among all same-type candidates already in the pool. This includes earlier local candidates. Each match preserves any shared identifier but replaces the text, embedding, and local statistics with those of the incoming local candidate. The last local match therefore supplies the representative; local statistics are not pooled. Task similarity retains the maximum pre-merge value rather than being recomputed for the retained text.

Edges are remapped to the fused nodes, discarding self-loops and taking the maximum weight for duplicate edges. Additional semantic edges connect only local-only and shared-only candidates, retaining the most similar neighbors subject to the similarity threshold and neighbor limit. PPR on this temporary graph uses the highest-similarity candidates meeting the task-matching threshold as anchors, up to the anchor limit. Positive similarities supply the normalized restart weights; without them, retrieval returns no experiences. Candidates with positive PPR scores are retained in descending order up to the candidate limit before final ranking. Stable identifiers resolve sorting and matching ties throughout integration.

Scoring and normalization. All outcome scores use pre-task statistics. Candidates with both local and shared records combine local outcome sums and usage counts with the cross-agent outcome score as a prior (Eq. 5). Local-only candidates use $u _ { 0 }$ as the prior; shared-only candidates use the cross-agent score directly.

The exploration bonus is $c \sqrt { \log ( 1 + T _ { \mathrm { p o o l } } ) / \operatorname* { m a x } ( 1 , n _ { v } ) }$ where c is fixed, $n _ { v }$ is the node’s usage count, and $T _ { \mathrm { p o o l } }$ is the total usage count in the relevant pool. Routing uses available shared nodes and excludes the target agent’s own uses. The bonus is added to $s _ { - i , v } ^ { \mathrm { o u t c o m e } }$ and capped at one to obtain $s _ { - i , v } ^ { \mathrm { e x p - } }$ outcome Contributor support $s _ { - i , v } ^ { \mathrm { s u p p o r t } }$ counts distinct other contributing agents, measuring provenance breadth rather than successful uses.

Final ranking uses only candidates retained by fused-graph PPR. Candidates with both local and shared records combine local usage counts with other agents’ shared counts; local-only and sharedonly candidates use their available counts. The bonus is added to $s _ { i , v } ^ { \mathrm { f u s e d } }$ and capped at one to obtain $s _ { i , v } ^ { \mathrm { e x p - f u s e d } }$ . Each score term is min-max normalized within its routing or final-ranking pool. If the range is numerically zero, the original values are retained after clipping to [0, 1].

Benchmark settings are specified with their adapters: Table 7 reports the ALFWorld graph configuration, Appendix C.1 gives MBPP memory and inference limits, and Appendix D defines the GAIA sensitivity defaults.

## B GAIA Evaluation Details

We report GAIA 2023 Test results for two reasoning backbones in Table 1. The Test set contains 301 tasks: 93 at Level 1, 159 at Level 2, and 49 at Level 3. Correct-answer counts come from the official hidden-label evaluation. Memories remain frozen during GAIA Test evaluation.

Memory construction and execution. The validation source pool contains 165 tasks: 53 at Level 1, 86 at Level 2, and 26 at Level 3. The three-client GAIA adapters assign each level to a logical client with its own local memory. Shared retrieval accesses experiences published by other clients. Test tasks do not supply construction feedback, and no model weights are fine-tuned.

Construction is method-specific. Qwen ReasoningBank extracts strategy memories from cached AgentKB validation trajectories generated with the same backbone. Luna ReasoningBank and Mem0 collect fresh validation trajectories. Memory size varies with the construction method and completed source records.

Qwen3-Next-80B-A3B-Instruct is served through Vertex MaaS; GPT-5.6 Luna uses the OpenAI API. The GAIA execution interface uses SmolAgents with search, browser, document, and Python tools. Overall accuracy uses all 301 tasks; level-wise accuracy uses 93, 159, and 49 tasks for Levels 1–3, respectively. Failed tasks remain in their corresponding denominators. Blank answers count as incorrect; nonempty submitted answers are scored by the official evaluator, including answers retained after execution errors.

Table 4 provides the difficulty-level breakdown of the GAIA Test results in Table 1.

Progressive evaluation. Section 4.3 evaluates cumulative recovery over three passes on the same 165 validation tasks (53/86/26 at Levels 1–3). The first pass starts with empty experience memories. Each subsequent pass re-evaluates all 165 tasks, including those solved previously. Experience updates follow task grading, and memories are carried forward between passes. Later attempts may therefore retrieve experiences derived from earlier attempts on the same task. A task is counted as correct if it has been solved at least once by the end of the current pass; each task contributes at most once, and the denominator remains 165. This protocol combines repeated attempts with experience updates. No Memory provides an independent reference rather than a matched three-pass control.

Table 4: GAIA Test accuracy (%) by difficulty. Bold marks the best score in each column within each model block, including ties.
<table><tr><td colspan="5">Qwen3-Next-80B-A3B-Instruct</td></tr><tr><td>Method</td><td>Overall</td><td>L1</td><td>L2</td><td>L3</td></tr><tr><td>No Memory</td><td>27.57</td><td>41.94</td><td>26.42</td><td>4.08</td></tr><tr><td>AgentKB</td><td>28.90</td><td>49.46</td><td>22.64</td><td>10.20</td></tr><tr><td>ReasoningBank</td><td>26.91</td><td>43.01</td><td>23.27</td><td>8.16</td></tr><tr><td>ExpGraph</td><td>27.24</td><td>44.09</td><td>24.53</td><td>4.08</td></tr><tr><td>Mem0</td><td>25.91</td><td>41.94</td><td>23.27</td><td>4.08</td></tr><tr><td>LightMem</td><td>21.59</td><td>35.48</td><td>16.98</td><td>10.20</td></tr><tr><td>XferAgent</td><td>29.57</td><td>47.31</td><td>24.53</td><td>12.24</td></tr></table>

GPT-5.6 Luna
<table><tr><td>Method</td><td>Overall</td><td>L1</td><td>L2</td><td>L3</td></tr><tr><td>No Memory</td><td>48.17</td><td>66.67</td><td>45.91</td><td>20.41</td></tr><tr><td>AgentKB</td><td>46.51</td><td>60.22</td><td>46.54</td><td>20.41</td></tr><tr><td>ReasoningBank</td><td>46.18</td><td>59.14</td><td>45.91</td><td>22.45</td></tr><tr><td>ExpGraph</td><td>47.51</td><td>68.82</td><td>41.51</td><td>26.53</td></tr><tr><td>Mem0</td><td>44.85</td><td>65.59</td><td>40.88</td><td>18.37</td></tr><tr><td>LightMem</td><td>41.86</td><td>52.69</td><td>42.77</td><td>18.37</td></tr><tr><td>XferAgent</td><td>52.16</td><td>72.04</td><td>48.43</td><td>26.53</td></tr></table>

## C Additional Evaluation Protocols

Table 5 summarizes the ALFWorld, MBPP, and cross-framework GAIA protocols. Memory methods use benchmark-specific adapters with agentowned memories and bounded cross-agent retrieval. Memories are constructed from collected trajectories without fine-tuning model weights.

Within each benchmark, models, evaluation tasks, and retrieval limits are matched across methods. ALFWorld uses a fixed cohort across adaptations. ALFWorld and MBPP rates include failed and unparsed cases and are rounded to one decimal place.

## C.1 MBPP: Coding Development Subset

The evaluation cohort in Table 5 contains 60 tasks from a mixed MBPP/MBPPPlus development pool of 100 tasks. Cached construction trajectories have a non-IID three-agent assignment. The memoryconstruction and evaluation sets are disjoint at the task level.

Inference and memory settings. MBPP uses greedy decoding, per-call limits of 4,096 input and 2,048 output tokens, and a single beginning-ofsequence token. Dense retrieval uses the frozen Contriever encoder with a 256-token encoding limit. Agent ownership and evaluation IDs are fixed within each comparison; inference and memory limits appear in Table 5.

Experience-sharing baseline adapters. For MBPP, the memory-based methods use agentowned memories and bounded cross-agent retrieval; No Memory evaluates without retrieval. AgentKB ranks up to 10 local and eight remote candidates by dense similarity and includes at most 10 entries in the prompt. Table 6 summarizes the method-specific adaptations. Agents are logical partitions with in-process routing; network efficiency is outside the evaluation scope.

Evaluation. Each task produces one final program, extracted with the frozen parser and evaluated using the stored check(candidate) harness with a 30-second timeout. Pass rate is computed using this stored harness rather than the fullbenchmark EvalPlus suite.

## C.2 ALFWorld: Embodied Task Completion

ALFWorld evaluates language-guided household tasks through a textual environment. Success rate is pooled over the Seen and Unseen cohort in Table 5, using the final environment state. Thinking is disabled and temperature is zero, with per-call limits of 8,192 input and 256 output tokens. The action interface exposes legal actions. Generated actions are passed to the environment unchanged, including invalid actions. Retrieval uses a training-derived TF-IDF adaptation for the reported memory and graph methods.

The 90 source trajectories were generated with Qwen3.5-9B and reprocessed by the 27B model, yielding 74 AgentKB entries, 90 ReasoningBank entries, and 80 XferAgent entries.

Table 7 lists the key settings. The ALFWorld adapter additionally checks Jaccard overlap between task-facet sets derived from task keywords and capability tags before merging. Nonempty tool sets must overlap, as must nonempty failurecondition sets for FAILURE experiences. During connected expansion, anchor rewards use task similarities, edge rewards use edge weights, and connector costs sum 1 − w along the path, with edge weights clipped to [0, 1]. The disclosure-risk term is inactive: all frozen shared nodes have zero risk scores.

Table 5: Settings for the additional evaluations. Memory content and outcome statistics are frozen during evaluation. Memory allowances specify the token limits for retrieved context.
<table><tr><td>Setting</td><td>ALFWorld</td><td>MBPP</td><td>Cross-framework GAIA</td></tr><tr><td>Backbone</td><td>Qwen3.8-27B (Tinker)</td><td>Llama-3.1-8B-Instruct</td><td>GPT-5.6 Luna</td></tr><tr><td>Construction data</td><td>90 training trajectories</td><td>265 cached training trajectories</td><td>20 questions per agent from a 105-question source pool</td></tr><tr><td>Evaluation cohort</td><td>274 tasks: 140 Seen, 134 Un- 60 development tasks seen</td><td></td><td>60 tasks: 18 / 29 / 13 at Levels 1-3</td></tr><tr><td>Agent arrangement</td><td>Three logical agents</td><td>Three logical agents</td><td>Nested 2 / 4 / 8 contributors; one fixed receiver</td></tr><tr><td>Shared budget B / final 8 / 10 limit K</td><td></td><td>8 / 10</td><td>24/3</td></tr><tr><td>Memory allowance</td><td>2,400 tokens</td><td>2,400 tokens (character esti- 12,000 tokens mate)</td><td></td></tr></table>

Table 6: Memory adapters used for MBPP.
<table><tr><td>Method</td><td>Benchmark-specific adaptation</td></tr><tr><td>AgentKB</td><td>Flat workflow entries; bounded local and cross-client similarity retrieval.</td></tr><tr><td></td><td>ReasoningBank Dense retrieval over cached training ex- periences.</td></tr><tr><td>ExpGraph</td><td>Graph retrieval over cached training ex- periences, frozen during evaluation.</td></tr><tr><td>Mem0</td><td>Source-preserving procedural-memory updates in a persistent store.</td></tr><tr><td>LightMem</td><td>Short-term grouping and offline consoli- dation with source text retained.</td></tr></table>

No Memory uses no retrieved experience. The AgentKB adapter uses a flat knowledge base partitioned across three logical agents. Each episode retrieves once at the start, using at most 10 private candidates from the target agent and eight cross-agent candidates, then at most 10 final entries within 2,400 tokens. It uses neither graph structure nor outcome-based ranking, and its KB is frozen throughout testing.

The other adapters also use the 90 raw training trajectories. ReasoningBank uses local and crossagent TF-IDF similarity over the re-extracted memories. ExpGraph uses diffusion within agent-owned graphs. Mem0 applies ADD/UPDATE/DELETE operations to agent-owned memories; LightMem extracts short-term lessons and consolidates them within each agent.

Table 7: Key settings of the ALFWorld XferAgent adapter.
<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Shared node budget / final limit</td><td>8 / 10</td></tr><tr><td>Memory allowance (tokens)</td><td>2,400</td></tr><tr><td>Experience-text limit (characters)</td><td>5,000</td></tr><tr><td>Reconciliation threshold / margin</td><td>0.92 / 0</td></tr><tr><td>Structural-overlap threshold</td><td>0.50</td></tr><tr><td>Anchor limit / task-match threshold</td><td>10 / 0.08</td></tr><tr><td>Expansion / fused-candidate limits</td><td>20 /20</td></tr><tr><td>Fusion-merge threshold</td><td>0.92</td></tr><tr><td>PPR restart probability</td><td>0.35</td></tr><tr><td>Outcome prior u0 / strength κ</td><td>0.50 / 2</td></tr><tr><td>Exploration weight c</td><td>0.25</td></tr><tr><td>Final outcome weight</td><td>0.40</td></tr><tr><td>Routing: relevance / outcome / support</td><td></td></tr><tr><td>Edge / anchor / connector weights</td><td>0.55 / 0.25 / 0.15 0.10 / 0.25 / 0.10</td></tr></table>

## C.3 Cross-Framework Transfer Protocol

Data and agent groups. The 165 GAIA validation questions are split into the source and evaluation cohorts in Table 5. Table 8 gives the contributor groups. Larger groups retain the smaller groups’ agents and constructed memories. Each agent samples its source questions uniformly without replacement; samples may overlap across agents.

Table 8: Nested contributor groups for cross-framework transfer. Each contributor builds memory from 20 source questions. Totals count agent–question instances; questions may overlap across agents.
<table><tr><td>Agents</td><td>SmolAgents</td><td>OWL</td><td>Total instances</td></tr><tr><td>2</td><td>1</td><td>1</td><td>40</td></tr><tr><td>4</td><td>2</td><td>2</td><td>80</td></tr><tr><td>8</td><td>4</td><td>4</td><td>160</td></tr></table>

Construction and evaluation. Each agent constructs local memory from its selected source tasks;

contributions are imported in a fixed agent order to construct the Shared Experience Graph. The receiver is the first SmolAgents contributor, using the same local memory in every condition. It answers all 60 evaluation questions, producing one final answer per question without multi-agent voting.

Model calls allow up to three API attempts with a 120-second request timeout and no additional SDK retries; the task time limit is 2,700 seconds. The figure’s Average is the unweighted mean of the three level accuracies, not pooled accuracy over the 60 questions. Agent count and aggregate learning exposure increase together.

## D Sensitivity Protocol

The sensitivity study in Section 4.6 uses GPT-5.6 Luna with 120 GAIA validation tasks for memory construction and 45 disjoint held-out tasks for evaluation. Construction contains 38, 63, and 19 questions at Levels $_ { 1 - 3 ; }$ evaluation contains 15, 23, and 7, respectively. One logical agent is assigned to each level.

Each variant changes one factor from the default: $B = 2 4$ , $K = 3 , \lambda _ { u } = 0 . 4 0$ , and reconciliation margin 0.03. K limits the experiences included in the prompt; $\lambda _ { u }$ weights the outcome term in final ranking, not shared graph routing.

Table 3 reports nine configurations: one default and eight variants. The default rows repeat the same 45-task result, not separate evaluations.

Margin variants reconstruct the shared graph by replaying the original construction and feedback events without new solver calls. Feedback is assigned to corresponding reconstructed nodes available at each replay step, counting each mapped node at most once per historical exposure. Local graphs and their identifiers remain unchanged; integration follows Appendix A. Other variants reuse the original graphs. Memories and outcome statistics are frozen during evaluation; all remaining settings are held fixed.

Each task contributes one outcome per configuration; failures count as incorrect in the 45-task denominator.