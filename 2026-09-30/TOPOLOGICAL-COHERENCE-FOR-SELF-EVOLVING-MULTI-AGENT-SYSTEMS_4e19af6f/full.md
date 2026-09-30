# TOPOLOGICAL COHERENCE FOR SELF-EVOLVING MULTI-AGENT SYSTEMS

Sen Zhao<sup>1∗</sup> Ruiqi Kong<sup>1∗</sup> Zuyu Zhang<sup>1</sup> Lifeng Shen<sup>2</sup> Xinyu He<sup>4</sup> Xu Zhang<sup>1†</sup> Qinghua Zhang<sup>3</sup>

<sup>1</sup>Academy of Advanced Interdisciplinary Studies,

Chongqing University of Posts and Telecommunications, Chongqing, China <sup>2</sup>School of Artificial Intelligence,

Chongqing University of Posts and Telecommunications, Chongqing, China <sup>3</sup>School of Computer Science and Technology,

Chongqing University of Posts and Telecommunications, Chongqing, China <sup>4</sup>Towngas, China

## ABSTRACT

Complex tasks inherently couple workflow structure, agent responsibility, collaboration, and memory access: task regions delimit responsibility and tool scope, cross-region dependencies give rise to handoffs, and ownership boundaries delimit private and selectively shared memory. Existing methods can jointly optimize agent and communication structures, yet such optimization does not by itself require responsibility, handoff, and memory boundaries to remain consistent with task dependencies. We term this requirement topological coherence. We introduce TOCOMAS, a Topology-Coherent Multi-Agent System. TOCOMAS grounds a task graph in tool interfaces, organizes compatible task nodes into reusable responsibility domains, and derives dependency-induced and profile-conditioned collaboration together with boundary-regulated memory visibility. During online self-evolution, TOCOMAS proposes coupled changes to agent, collaboration, and memory policies, retaining for subsequent tasks only candidates that satisfy structural constraints and improve evaluated reward. Across BBEH, WorkBench, SWE-Bench-Verified, and CoMemBench, TOCOMAS improves task success over baselines across backbones. CoMemBench also shows gains over the self-evolving baseline in verified progress, handoffs, and memory isolation.

## 1 INTRODUCTION

Large language model (LLM) based multi-agent systems (MAS) provide a general framework for combining specialized reasoning, tool use, and interaction policies (Guo et al., 2024; Li et al., 2024). Conversational and role-based systems such as CAMEL, AutoGen, and AgentVerse (Li et al., 2023; Wu et al., 2024; Chen et al., 2024), together with peer review, debate, MetaGPT, and ChatDev (Xu et al., 2023; Du et al., 2024; Hong et al., 2024; Qian et al., 2024), make increasingly complex tasks accessible. This progress also turns organization into a central design problem: a MAS must decide which task regions each agent owns, which tools and information it may access, where verification is required, and how local outputs compose into a globally executable process.

Recent work automates these choices by searching computation graphs, agentic programs, or workflows (Zhuge et al., 2024; Hu et al., 2025; Zhang et al., 2025c); adapting communication topology or task orchestration (Liu et al., 2024; Zhang et al., 2025b; Yang et al., 2025; Wu et al., 2026; Ke et al., 2026); and retaining structured or decentralized experience (Zhang et al., 2025a; Wang et al., 2026; Yang et al., 2026; Hao et al., 2026). Evolutionary methods further generate agents (Yuan et al., 2025). The most comprehensive system-level formulation jointly evolves agent roles, prompts, models, tools, and communication topology, while retaining a candidate pool and experience memory for cross-task reuse (Hu et al., 2026). Nevertheless, task workflows, agent responsibilities, collaboration, and memory remain separately represented structures, without an explicit constraint that keeps their relations compatible as the system changes.

Figure 1 illustrates this relation on a query whose policy, sales-history, and economicevidence branches feed workbook, reporting, and delivery tasks. The workflow itself supplies a common organizational reference: task regions delimit responsibility and tool scope, cross-region dependencies create handoffs and optional verification, and ownership boundaries delimit private and selectively shared memory. These are different views of one taskconditioned organization. We call their agreement topological coherence: not similarity between graph shapes, but consistency among the relations each layer derives from the task topology (Bodnar et al., 2021).

Building on this formulation, we propose TO-COMAS, a Topology-Coherent Multi-Agent System that carries this constraint through organization and self-evolution. TOCOMAS maps an upstream-provided task graph to a public tool graph under global operation and dependency constraints, then groups structurally and operationally compatible regions into reusable agent

![](images/97ff5217176520ce3abab9f32523a76663ba187266897efc74c286b7bb1a2a70.jpg)  
Figure 1: Topological coherence in ToCoMAS. Task topology jointly induces agent responsibilities, collaboration interfaces, and memory boundaries, which evolve as a coupled organization.

responsibility domains, allowing one agent to own multiple related subgraphs while separating hetero geneous execution semantics. It then derives collaboration and memory coordination from relations among these domains: responsibility ownership delimits local tools and private memory, task-induced adjacency provides dependency-aware handoffs, and boundary policies regulate selective sharing and verifier escalation. Finally, an online controller proposes changes to reusable structural policies and compiles each candidate against the task topology. It reconciles affected ownership, handoff, and memory relations, then retains structurally valid candidates only when their evaluated reward improves on the parent. We evaluate fixed and evolving variants against a direct-call control, prespecified and automatically designed baselines, and a self-evolving baseline across reasoning, tool-use, and software-engineering benchmarks. Our contributions are:

• We formulate topological coherence as a structural constraint that keeps task workflows, agent responsibilities, collaboration interfaces, and memory boundaries consistent during self-evolution.

• We develop a topology-grounded architecture in which task and tool relations induce reusable agent responsibility domains, dependency-aware and profile-conditioned collaboration, and boundary-regulated memory access; execution feedback guides coupled structural proposals.

• We establish controlled comparisons with prespecified, automatically designed, and self-evolving MAS baselines to separate architectural gains from evolutionary search.

## 2 RELATED WORK

Early role-based and conversational MAS organize collaboration through role play, proposal aggregation, debate, and peer review (Li et al., 2023; Wu et al., 2024; Chen et al., 2024; Du et al., 2024; Xu et al., 2023); MetaGPT and ChatDev encode specialized roles in manually specified software-development workflows (Hong et al., 2024; Qian et al., 2024). Automatic design searches optimizable computation graphs (GPTSwarm) (Zhuge et al., 2024), code-defined agentic systems and workflow programs (ADAS and AFlow) (Hu et al., 2025; Zhang et al., 2025c), or agents, prompts, and communication structures (AutoAgents, MASS, and ARG-Designer) (Chen et al., 2023; Zhou et al., 2026; Li et al., 2026b). MAS-Orchestra adds holistic task-conditioned orchestration (Ke et al., 2026). These methods move beyond fixed protocols but favor atomic roles or workflow nodes over reusable task-and-tool responsibility domains.

Dynamic agent networks adapt participating agents at inference time (Liu et al., 2024); G-Designer learns task-aware communication graphs, AgentNet evolves decentralized coordination, and CARD conditions communication topology on dynamic environments (Zhang et al., 2025b; Yang et al., 2025; Wu et al., 2026). Graph-conditioned language models and graph retrieval guide reasoning and information selection (Tang et al., 2024; He et al., 2024), while topological learning studies local-toglobal compatibility through shared incidence and boundaries (Hansen & Ghrist, 2019; Bodnar et al., 2021). In memory organization, G-Memory stores hierarchical collaboration experience, E-mem reconstructs episodic context through local assistants, and PlugMem organizes reusable knowledge in a typed graph (Zhang et al., 2025a; Wang et al., 2026; Yang et al., 2026). Together, these lines support relational reasoning, communication, and memory; they do not, by themselves, jointly derive task ownership, tool authority, required handoffs, and memory visibility from a shared task-and-tool topology.

Recent self-evolution methods revise organizations using feedback: EvoAgent expands agent populations through mutation and crossover, while DecentMem uses decentralized private memories for agent-level self-evolution (Yuan et al., 2025; Hao et al., 2026). EvoMAS, the closest system-level approach, represents a MAS as a structured configuration and jointly evolves agent roles, prompts, model and tool assignments, and communication topology; it also retains a candidate pool and experience memory for cross-task reuse (Hu et al., 2026). Yet co-editing these components does not itself require changes in responsibility to induce compatible agent capabilities, handoffs, and memory permissions after a structural change. TOCOMAS instead enforces topological coherence, deriving agent responsibilities, required handoffs, and memory boundaries from shared task-and-tool regions.

## 3 PROBLEM FORMULATION

Following a configuration-based formulation for evolutionary multi-agent systems (Hu et al., 2026), a multi-agent system is specified by:

$$
C = \left( G , \mathcal { A } , \mathcal { V } _ { \mathrm { i n } } , \mathcal { V } _ { \mathrm { o u t } } \right) ,\tag{1}
$$

where $G = ( \nu , \mathcal { E } )$ is a directed communication graph over agents, $\mathcal { A } = \{ A _ { i } \} _ { i = 1 } ^ { k }$ is the corresponding set of agent configurations, and $\mathcal { V } _ { \mathrm { i n } } , \mathcal { V } _ { \mathrm { o u t } } \subseteq \mathcal { V }$ denote the input and output agents. Each $A _ { i } =$ $( b _ { i } , p _ { i } , \bar { \Gamma _ { i } } )$ contains a backbone model, prompt, and accessible tool set. Given a task $q \in \mathcal { Q }$ with task graph $\boldsymbol { T _ { q } } = ( \mathcal { U } _ { q } , \mathcal { D } _ { q } )$ , let $H _ { q } = ( \bar { \mathcal { U } } _ { q } , \bar { \mathcal { D } } _ { q } , \bar { \Psi } _ { q } )$ be its tool-grounded executable topology. The responsibility map $\pi _ { q } : \mathcal { U } _ { q } \to S _ { q }$ groups task nodes into regions, and the ownership map $\phi _ { q } : S _ { q }  \mathcal { V }$ assigns these regions to capable agents. The task-conditioned configuration is:

$$
C _ { q } = \left( C , H _ { q } , \pi _ { q } , \phi _ { q } , \Lambda _ { q } , \mathcal { M } _ { q } \right) , \qquad \overline { { H } } _ { q } = H _ { q } / \pi _ { q } .\tag{2}
$$

Here $C$ is instantiated with the task-conditioned collaboration graph $G _ { q }$ . The policy $\Lambda _ { q }$ governs memory admission and visibility, whereas $\mathcal { M } _ { q }$ denotes runtime memory records. We call $C _ { q }$ topologycoherent when responsibility regions have capable owners, required cross-region dependencies have compatible handoffs, and memory visibility respects the same ownership boundaries. Executing $C _ { q }$ on q yields an output $y _ { q }$ and trace $\tau _ { q } ;$ configuration quality is evaluated by:

$$
\begin{array} { r l } & { ~ ( y _ { q } , \tau _ { q } ) = \operatorname { E x e c } ( C _ { q } , q ) , } \\ & { R ( q , C _ { q } ) = \operatorname { M e t r i c s } ( q , C _ { q } ) - \beta \cdot \operatorname { C o s t } ( C _ { q } ) . } \end{array}\tag{3}
$$

Here $\operatorname { M e t r i c s } ( q , C _ { q } )$ measures accepted execution quality and may incorporate bounded evidence of useful memory context, while $\operatorname { C o s t } ( C _ { q } )$ denotes the measured resource cost used in candidate selection. The scalar $\beta \geq 0$ controls the performance–efficiency trade-off. The trace provides feedback for structural proposals; candidates are retained only when they pass executable structural checks and improve accepted reward. Accumulated traces inform subsequent evolution and selection.

## 4 THE PROPOSED MODEL

We present TOCOMAS, a topology-coherent framework for constructing and evolving multi-agent systems. Its central principle is that agent organization, collaboration, and memory should not be designed as independent configuration choices. Instead, they are derived from a common task topology and jointly screened as the system evolves. Given a task graph $T _ { q }$ and the public tool graph K, TOCOMAS first grounds every existing task node by a whole-graph map $\Psi _ { q } ,$ without redecomposing the user query. It then forms a quotient topology over responsibility regions, assigns these regions to capable agents, and derives communication and memory boundaries from the same maps. This construction is summarized by:

![](images/7380b40714ec3c1ee7cf51a39eb1577b446c01cc7f80f8996575a99d5b68bcc7.jpg)  
Figure 2: Overview of TOCOMAS. Task-derived responsibility regions organize agents, handoffs, and memory boundaries; online evolution preserves their coherence.

$$
T _ { q } \xrightarrow { \Psi _ { q } } H _ { q } \xrightarrow { \pi _ { q } } \overline { { { H } } } _ { q } , \phi _ { q } : S _ { q } \longrightarrow \mathcal { V } _ { q } .\tag{4}
$$

Here $\pi _ { q }$ groups task nodes into responsibility regions, $\overline { { H } } _ { q }$ preserves their cross-region dependencies, and $\phi _ { q }$ assigns regions to agents. These maps determine required handoffs and constrain the collaboration graph $G _ { q }$ and memory policy $\Lambda _ { q } ;$ additional communication and retrieved records remain subject to their respective policies. As illustrated in Figure 2, the four modules below realize this construction and preserve it during structural evolution.

## 4.1 TOPOLOGICAL AGENT ORGANIZATION

Task topology and responsibility quotient. For a task q, let $\boldsymbol { T _ { q } } = ( \mathcal { U } _ { q } , \mathcal { D } _ { q } )$ be the given task graph and let $K = ( \mathcal { T } , \mathcal { R } )$ be a public graph of tool interfaces and their operated resources. Each node exposes an operation requirement $R ( u )$ —including operations, resources, effects, receipt types, and declared input–output semantics. The grounding map $\Psi _ { q } : T _ { q } \to 2 ^ { \tau }$ selects a minimal valid tool bundle $\Gamma ( u ) = \bar { \Psi _ { q } ( u ) }$ for every existing node jointly, subject to capability coverage, edge-semantic compatibility, side-effect constraints, and provider continuity. Gold trajectories and evaluator state are excluded from $\Psi _ { q } .$ The grounded executable topology is $H _ { q } = ( \mathcal { U } _ { q } , \mathcal { D } _ { q } , \Psi _ { q } )$ Rather than instantiating one agent per subtask, TOCOMAS groups operationally compatible nodes into responsibility regions. We express this organization by a surjective map:

$$
\pi _ { q } : \mathcal { U } _ { q } \longrightarrow S _ { q } , \qquad \mathcal { U } _ { q } = \prod _ { S \in S _ { q } } S , \quad \pi _ { q } ( u ) = S \mathrm { ~ f o r ~ } u \in S ,\tag{5}
$$

where each $S \in S _ { q }$ is a responsibility region and the disjoint union assigns every task node to exactly one such region. Contracting the regions under $\pi _ { q }$ yields the quotient topology:

$$
\overline { { H } } _ { q } = H _ { q } / \pi _ { q } = \left( \mathcal { S } _ { q } , \overline { { D } } _ { q } \right) , \qquad \left( \mathcal { S } _ { i } , \mathcal { S } _ { j } \right) \in \overline { { \mathcal { D } } } _ { q } \Longleftrightarrow \exists ( u , v ) \in \mathcal { D } _ { q } : \pi _ { q } ( u ) = \mathcal { S } _ { i } , \ \pi _ { q } ( v ) = \mathcal { S } _ { j } ,\tag{6}
$$

for $S _ { i } \neq S _ { j }$ . Thus, $\overline { { H } } _ { q }$ preserves dependencies between regions while hiding dependencies internal to a single responsibility domain. To preserve executability, a region merge is admissible only when

the resulting quotient remains acyclic. To construct $\pi _ { q } ,$ we associate each pair of task nodes with the compatibility profile:

$$
\begin{array} { r } { \mathbf { a } _ { q } ( u , v ) = \bigl ( a _ { \Gamma } ( u , v ) , a _ { \mathcal { V } } ( u , v ) , a _ { \mathcal { D } } ( u , v ) , a _ { \mathrm { s } } ( u , v ) \bigr ) , } \end{array}\tag{7}
$$

whose components respectively measure tool overlap, input–output compatibility, dependency adjacency, and semantic similarity. Starting from singleton regions, operationally compatible regions are merged only when the resulting region respects the prescribed task and tool capacities and the quotient remains acyclic. Thus, $\pi _ { q }$ uses complementary structural evidence without introducing a separate free-form role decomposition.

Capability-grounded ownership. Let $\textstyle \Gamma ( S ) = \bigcup _ { u \in S } \Gamma ( u )$ be the capabilities required by a region. A responsibility assignment is a map:

$$
\phi _ { q } : \mathcal { S } _ { q } \longrightarrow \mathcal { V } _ { q } , \qquad \Gamma ( S ) \subseteq \Gamma _ { \phi _ { q } ( S ) } \quad \mathrm { f o r } \mathrm { e v e r y } S \in \mathcal { S } _ { q } ,\tag{8}
$$

where $\mathcal { V } _ { q }$ is the active agent set and $\Gamma _ { A }$ denotes the tools available to agent A. The map need not be injective: one agent may own several compatible regions, while structurally or operationally distinct regions remain assigned to different agents. If no existing agent covers a region, the agent pool supplies a new specialization whose capabilities cover Γ(S). In this way, an agent role is defined by a responsibility domain in the task topology rather than by a prefabricated textual label.

## 4.2 STRUCTURE-ADAPTIVE COLLABORATIVE ORCHESTRATION

The quotient topology determines which responsibility domains must exchange information. Pushing cross-region task dependencies through the ownership map gives the communication edges required for task execution:

$$
\begin{array} { r } { \mathcal { E } _ { q } ^ { \mathrm { r e q } } = \big \{ \big ( \phi _ { q } ( S _ { i } ) , \phi _ { q } ( S _ { j } ) \big ) \big | \exists ( u , v ) \in \mathcal { D } _ { q } , u \in S _ { i } , v \in S _ { j } , \phi _ { q } ( S _ { i } ) \neq \phi _ { q } ( S _ { j } ) \big \} . } \end{array}\tag{9}
$$

The runtime collaboration topology combines these dependency-induced edges with profileconditioned spatial edges:

$$
G _ { q } = \left( \mathcal { V } _ { q } , \mathcal { E } _ { q } ^ { \mathrm { r e q } } \cup \mathcal { E } _ { q } ^ { \mathrm { s p } } \right) .\tag{10}
$$

Here $\mathcal { E } _ { q } ^ { \mathrm { s p } }$ is constructed from agent-profile similarity, query relevance, and tool input–output complementarity, providing additional communication opportunities that may be pruned adaptively.

Let $z _ { i }$ denote the artifact produced for region $S _ { i }$ . Local execution and boundary-aware handoff are summarized by:

$$
\begin{array} { r l } & { \quad z _ { i } = A _ { \phi _ { q } ( S _ { i } ) } \left( q | _ { S _ { i } } , \Gamma ( S _ { i } ) , \mathcal { M } _ { q } ^ { \mathrm { v i s } } ( S _ { i } ) , \{ h _ { j i } \} \right) , } \\ & { \quad h _ { i j } = \mathrm { R o u t e } _ { i j } ( z _ { i } ; \xi _ { i j } ) \in \left\{ \rho _ { i j } ( z _ { i } ) , \emptyset \right\} . } \end{array}\tag{11}
$$

Here $\rho _ { i j }$ exposes only the artifact required by the downstream region, whereas $\mathrm { R o u t e } _ { i j }$ applies the dynamic topology and routing policy; a verifier request may be triggered when required.

## 4.3 BOUNDARY-PRESERVING MEMORY COORDINATION

TOCOMAS associates each memory record with its producing agent, task region, and provenance. The responsibility topology induces a visibility policy $\Lambda _ { q }$ that determines which private records and admitted boundary evidence are available to a region:

$$
\mathcal { M } _ { q } ^ { \mathrm { v i s } } ( S _ { i } ) = \mathrm { F i l t e r } _ { \Lambda _ { q } } \left( \mathcal { M } _ { q } ; S _ { i } \right) .\tag{12}
$$

The filter first enforces ownership and boundary permissions; semantic retrieval subsequently ranks only the admissible records. Specifically, semantic retrieval selects seeds from the admissible view, after which typed graph expansion reconstructs the regional context:

$$
\begin{array} { r } { \mathcal { R } _ { q } ( S _ { i } ) = \mathrm { E x p a n d } _ { L } \left( \mathrm { S e e d } _ { k } \left( q | _ { S _ { i } } , \mathcal { M } _ { q } ^ { \mathrm { v i s } } ( S _ { i } ) \right) ; \mathcal { M } _ { q } ^ { \mathrm { v i s } } ( S _ { i } ) \right) , } \end{array}\tag{13}
$$

where $k$ is the number of retrieval seeds and L is the expansion depth. Semantic relevance can therefore rank admissible memories but cannot override their ownership or sharing boundary. Records may encode episodic, evidence, skill, subgraph, and failure relations.

The association between experience and responsibility is preserved when agent ownership changes. If region S is reassigned, its private memory is transported with the region:

$$
\phi _ { q } ^ { ( t ) } ( S ) \neq \phi _ { q } ^ { ( t + 1 ) } ( S ) \quad \Longrightarrow \quad \mu _ { S } ^ { t  t + 1 } : \mathcal { M } _ { q } ^ { \circ , ( t ) } ( S ) \longrightarrow \mathcal { M } _ { q } ^ { \circ , ( t + 1 ) } ( S ) ,\tag{14}
$$

where $\mu _ { S } ^ { t  t + 1 }$ retains provenance while updating access authority. When task-region ownership changes, records supported by the reassigned tasks are transferred with provenance and their access metadata is updated. This prevents both orphaned experience and unintended global sharing.

## 4.4 COHERENCE-CONSTRAINED STRUCTURAL EVOLUTION

The structural projection of the task-conditioned configuration $C _ { q }$ is:

$$
\Omega _ { q } = \mathrm { S t r u c t } ( C _ { q } ) = ( \pi _ { q } , \phi _ { q } , G _ { q } , \Lambda _ { q } ) .\tag{15}
$$

Here $\Lambda _ { q }$ is the memory admission, visibility, retrieval, and transport policy induced by the responsibility quotient; concrete records $\mathcal { M } _ { q }$ are runtime state governed by $\Lambda _ { q } .$ not genes in the evolutionary configuration. Under the coherence requirement above, $\Omega _ { q }$ is coherent when its ownership, collaboration, and memory policy are supported by the quotient topology $\overline { { H } } _ { q } .$ . The construction in the preceding modules establishes this relation through the shared maps $\pi _ { q }$ and $\phi _ { q }$ , rather than through independent structural penalties.

At evolution step t, execution feedback $\tau _ { q } ^ { ( t ) }$ and accumulated experience $\mathcal { E } ^ { ( t ) }$ guide a structural proposal:

$$
\begin{array} { r } { \widetilde { \Omega } _ { q } ^ { ( t + 1 ) } = \mathrm { P r o p o s e } \left( \Omega _ { q } ^ { ( t ) } , \tau _ { q } ^ { ( t ) } , \mathcal { E } ^ { ( t ) } \right) . } \end{array}\tag{16}
$$

Equation 16 summarizes proposal at the configuration level: evolution changes reusable structural policies, which are compiled into a candidate organization for the current task. A proposal may change only a subset of policies. If it changes task-region ownership, compilation reconciles required handoffs and memory permissions with the new boundaries; other proposals must also preserve those task dependencies. Coupling therefore requires compatible relations, not a change to every component on every step.

A single structural screen checks that each task node has one capable owner and that required crossregion handoffs and memory visibility respect the induced boundaries. A valid candidate is then executed; Accept denotes reward-based selection against its parent using the prescribed margin:

$$
\Omega _ { q } ^ { ( t + 1 ) } = \left\{ \begin{array} { l l } { \widetilde { \Omega } _ { q } ^ { ( t + 1 ) } , } & { \mathrm { V a l i d } ( \widetilde { \Omega } _ { q } ^ { ( t + 1 ) } ) \wedge \mathrm { A c c e p t } ( \widetilde { \Omega } _ { q } ^ { ( t + 1 ) } ) , } \\ { \Omega _ { q } ^ { ( t ) } , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{17}
$$

An accepted organization is available to subsequently launched tasks, while execution traces inform later proposals.

## 5 EXPERIMENTS

We evaluate whether topological coherence across task, agent, collaboration, and memory structures improves task performance, verified workflow progress, handoff reliability, and information isolation. We then disentangle the contributions of the structural components and evolution, and examine parameter sensitivity and token cost conditioned on achieved workflow progress.

## 5.1 EXPERIMENTAL SETUP

Benchmarks. We evaluate reasoning on BBEH (Kazemi et al., 2025), stateful tool use on Work-Bench (Styles et al., 2024), and repository repair on SWE-Bench-Verified (Jimenez et al., 2023), following the general benchmark selection of evolutionary MAS work (Hu et al., 2026). We also use CoMemBench (Zhao et al., 2026) to assess multi-agent workflows with executable dependency graphs, typed handoffs, and paired clean and polluted instances. Its native evaluators separate terminal success from verified progress, information transfer, and isolation. Dataset details appear in Appendix A.

Table 1: Overall comparison on BBEH, WorkBench, SWE-Bench-Verified, and CoMemBench.
<table><tr><td rowspan="2">Backbone</td><td rowspan="2">Organization</td><td rowspan="2">Method</td><td colspan="3">General Task Performance</td><td colspan="4">CoMemBench</td></tr><tr><td></td><td>BBEH ↑ WorkBench ↑ SWE-V ↑</td><td></td><td>SR↑</td><td>VNCR↑</td><td>VHS ↑</td><td>ICS ↑</td></tr><tr><td rowspan="6">Qwen3.8-27B</td><td>Control</td><td>Direct LLM Call</td><td>19.13</td><td>19.28</td><td>9.80</td><td>0.00</td><td>8.81</td><td>34.50</td><td>0.00</td></tr><tr><td>Prespecified</td><td>Peer Review</td><td>46.52</td><td>37.10</td><td>30.50</td><td>1.00</td><td>30.74</td><td>33.37</td><td>19.10</td></tr><tr><td rowspan="2"></td><td>SMoA</td><td>37.17</td><td>37.25</td><td>35.00</td><td>4.00</td><td>38.73</td><td>43.32</td><td>10.10</td></tr><tr><td>ADAS</td><td>32.61</td><td>62.17</td><td>34.20</td><td>0.00</td><td>13.91</td><td>27.73</td><td>7.40</td></tr><tr><td rowspan="2">Automatically designed Self-evolving</td><td>OMAC</td><td>41.09</td><td>55.94</td><td>39.50</td><td>2.00</td><td>32.98</td><td>40.51</td><td>17.30</td></tr><tr><td>EvoMAS</td><td>37.60</td><td>59.13</td><td>40.00</td><td>1.00</td><td>30.42</td><td>30.73</td><td>26.80</td></tr><tr><td rowspan="6">DeepSeek-V4-Flash</td><td rowspan="2">Ours Control</td><td>ToCoMAS</td><td>54.78</td><td>77.68</td><td>48.00</td><td>14.30</td><td>41.60</td><td>51.00</td><td>84.90</td></tr><tr><td>Direct LLM Call</td><td>14.13</td><td>20.43</td><td>12.00</td><td>0.0</td><td>9.03</td><td>27.54</td><td>6.50</td></tr><tr><td rowspan="2">Prespecified</td><td>Peer Review</td><td>49.35</td><td>39.57</td><td>39.60</td><td>0.50</td><td>27.51</td><td>29.45</td><td>17.10</td></tr><tr><td>SMoA</td><td>43.91</td><td>44.06</td><td>43.00</td><td>1.00</td><td>37.92</td><td>41.96</td><td>14.90</td></tr><tr><td rowspan="2">Automatically designed</td><td>ADAS</td><td>30.65</td><td>52.03</td><td>34.20</td><td>3.00</td><td>33.84</td><td>36.17</td><td>14.30</td></tr><tr><td>OMAC</td><td>41.30</td><td>35.36</td><td>43.00</td><td>3.50</td><td>56.88</td><td>55.61</td><td>4.40</td></tr><tr><td>Self-evolving</td><td></td><td>EvoMAS</td><td>50.65</td><td>55.50</td><td>46.00</td><td>1.50</td><td>27.99</td><td>27.17</td><td>9.80</td></tr><tr><td></td><td>Ours</td><td>ToCoMAS</td><td>55.20</td><td>61.45</td><td>49.00</td><td>14.00</td><td>44.36</td><td>59.92</td><td>76.50</td></tr></table>

Evaluation Metrics. We report BBEH accuracy, WorkBench task success, and the official SWE-Bench-Verified resolved rate. For CoMemBench (Zhao et al., 2026), SR measures terminal success, VNCR verified task-node progress, VHS accepted mandatory handoffs, and ICS memory isolation under paired pollution, conditional on clean-target success. Definitions and denominators appear in Appendix A.2. Token cost includes execution, coordination, memory, and recovery calls; infrastructure failures and incomplete telemetry are reported separately.

Backbone Models. The main comparison uses Qwen3.8-27B and DeepSeek-V4-Flash; the separate large-model comparison uses GPT-6-luna. Each run uses a homogeneous task-agent backbone, with model, tool schema, context budget, and decoding policy controlled within each comparison.

Baselines. We compare Direct LLM Call as a control; fixed organizations Peer Review (Xu et al., 2023) and SMoA (Li et al., 2025); automatic design methods ADAS (Hu et al., 2025) and OMAC (Li et al., 2026a); and the self-evolving baseline EvoMAS (Hu et al., 2026). Benchmark adapters record execution and handoff evidence without changing a method’s routing or memory decisions.

Implementation Details. Linux clients access the backbones through OpenAI-compatible APIs; Qwen3.8-27B inference runs on A100 GPUs, while CoMemBench execution uses H800 and A100 workers. Methods share benchmark tasks, tools, and native evaluators; paired CoMemBench variants start from the same source state in isolated workspaces. The fixed TOCOMAS configuration uses organization affinity 0.45 and memory retrieval k = 3; we vary these over {0.30, 0.45, 0.60, 0.75} and {1, 3, 5, 8}, respectively. In online structural evolution, a task’s first attempt uses the genome current at launch and supplies its reported score; feedback then proposes one structural candidate, tested on that same task. Accepted changes affect subsequently launched tasks, not the scored attempt. We retain native evaluator receipts, handoff traces, and model-call ledgers for auditing and cost accounting.

## 5.2 MAIN RESULTS

Table 1 jointly reports general task performance and the four CoMemBench quality measures for the Qwen3.8-27B and DeepSeek-V4-Flash backbones. Figure 3 reports the GPT-6-luna comparison. We draw two observations from the reported results:

Topology-coherent organization improves general task performance. With Qwen3.8-27B, TO-COMAS reaches 54.78% on BBEH, 77.68% on WorkBench, and 48.00% on SWE-Bench-Verified, exceeding the strongest reported baseline by 8.26, 15.51, and 8.00 percentage points, respectively. With DeepSeek-V4-Flash, the corresponding scores are 55.20%, 61.45%, and 49.00%, also leading the reported baselines. These gains are consistent with grounding agent responsibility and tool access in task regions and routing execution across their dependency boundaries.

![](images/b40df5f6a245d49d79cc030caf4e0370a6853cba9bb323ef4e1bbcc73d221704.jpg)

![](images/d571b5f3594e9e524fda69e7e615b32efdf651346736401a0760ce2e8a02b90c.jpg)  
Figure 3: GPT-6-luna results on WorkBench SR and CoMemBench VNCR/VHS.

Table 2: Architectural ablations and fixed-versus-evolving comparison of TOCOMAS.
<table><tr><td rowspan="2">Variant</td><td>WorkBench</td><td colspan="4">CoMemBench</td></tr><tr><td>SR↑</td><td>SR↑</td><td>VNCR↑</td><td>VHS ↑</td><td>ICS ↑</td></tr><tr><td>w/o Topological Agent Organization</td><td>69.86</td><td>4.00</td><td>27.27</td><td>37.97</td><td>67.14</td></tr><tr><td>w/o Structure-Adaptive Collaborative Orchestration</td><td>67.25</td><td>0.00</td><td>28.20</td><td>10.15</td><td>66.70</td></tr><tr><td>w/o Boundary-Preserving Memory Coordination</td><td>68.55</td><td>6.50</td><td>36.30</td><td>42.07</td><td>64.94</td></tr><tr><td>w/o Coherence-Constrained Structural Evolution</td><td>71.16</td><td>10.60</td><td>40.07</td><td>47.80</td><td>80.80</td></tr><tr><td>ToCoMAS</td><td>77.68</td><td>14.30</td><td>41.60</td><td>51.00</td><td>84.90</td></tr></table>

The gains extend to verified handoffs and information isolation. On Qwen3.8-27B CoMemBench, TOCOMAS attains 14.30% SR, 41.60% VNCR, 51.00% VHS, and 84.90% ICS, compared with EvoMAS’s 1.00%, 30.42%, 30.73%, and 26.80%. Improvement is visible not only at the terminal outcome but also in verified node progress, accepted cross-agent artifacts, and resistance to polluted context. This pattern is consistent with deriving collaboration interfaces and memory visibility from the same responsibility topology; component contributions are examined in the ablations below. With GPT-6-luna, TOCOMAS reaches 65.07% WorkBench SR and 49.58% CoMemBench VNCR, and also leads EvoMAS on VHS (54.57% versus 45.80%). CoMemBench SR is 9.00% for TOCOMAS and at most 0.50% for the compared baselines.

## 5.3 ABLATION STUDY

Table 2 compares full TOCOMAS with variants that remove one structural mechanism at a time. Topological Agent Organization assigns connected, tool-compatible task regions to agents; replacing this assignment with canonical-order groups preserves tool coverage but lowers CoMemBench VNCR by 14.33 points. This suggests that the placement of responsibility matters beyond the availability of tools. Structure-Adaptive Collaborative Orchestration derives inter-agent handoffs from cross-region task dependencies and checks whether downstream agents can consume the projected artifacts. Replacing these interfaces with fixed predecessor handoffs reduces VHS by 40.85 points and CoMemBench SR to zero, exposing the cost of a broken consumer boundary even when upstream outputs are verified. Boundary-Preserving Memory Coordination restricts retrieval to private records and evidence admitted across responsibility boundaries; making every verified record globally visible reduces ICS by 19.96 points. Thus, record validity alone does not determine whether it belongs in an agent’s context. Finally, Coherence-Constrained Structural Evolution uses task feedback to update the organization for subsequent tasks; retaining a fixed structure reduces WorkBench SR by 6.52 points. Together, these ablations link the three coherence relations to distinct outcomes: responsibility assignment to verified progress, cross-region handoffs to artifact acceptance, and memory boundaries to isolation; online evolution further improves task success.

## 5.4 PARAMETER SENSITIVITY

Figure 4 varies two controls while holding the remaining settings fixed. The organization-affinity threshold governs how readily related task nodes form an agent region. Both VNCR and VHS improve from 0.30 to 0.60, where they reach 41.6% and 51.3%, before declining at 0.75. Too little selectivity can mix weakly related responsibilities, while too much can fragment useful regions. The memory retrieval budget k limits the records supplied to a task node. Both measures peak at $k = 3$ and decline as retrieval expands to $k = 8 ,$ , consistent with extra context offering no guaranteed benefit once relevant evidence is available. Together, the sweeps favor a bounded task-region and memory scope rather than maximal grouping or retrieval.

![](images/bf865c1ebbcaef0d43f6f40b7df82e1a83fcbc833536f6bbe933b38ad492edc5.jpg)  
(a) Organization affinity.

![](images/1895a1ac0cdb8bcf35da7f7500e7429fcc4b7f32ab6b20c443a927f98526bdba.jpg)  
(b) Memory retrieval budget.

Figure 4: Parameter sensitivity of ToCoMAS.  
![](images/281b4baec7b1a43d596059f614129885915e8354a464199be9d6e6ea850a036a.jpg)

![](images/69b7c2ee0872ec54a4452e088bab53983bf409c645b336f83920e9439e575dc2.jpg)  
Figure 5: Progress-conditioned token cost across WorkBench and CoMemBench.

## 5.5 COST ANALYSIS

We examine token use within bins of observed progress rather than averaging over tasks of unequal length. For task $q ,$ let $n ( q )$ be the number of evaluator-matched actions on WorkBench or receiptverified nodes on CoMemBench, and let $c ( q )$ be its recorded input and output model tokens. Figure 5 plots the mean $c ( q )$ within each observed $n ( q ) = k$ bin, using successful WorkBench tasks and valid CoMemBench cells with complete token telemetry. The WorkBench TOCOMAS curve pairs post-recheck progress with first-pass tokens; the CoMemBench panel uses a separate cohort.

On WorkBench, the plotted TOCOMAS curve lies below Peer Review and SMoA but above Direct LLM Call and ADAS. On CoMemBench, it lies below the other multi-agent curves at most shared progress levels, although Direct LLM Call remains cheaper. This pattern is consistent with the organization in Figure 1: grouping compatible task nodes internalizes their dependencies within an agent, so required handoffs arise only at cross-region edges; projected artifacts and boundary-filtered retrieval limit the context passed to downstream agents. Aligning these decisions with the task topology can therefore reduce coordination overhead without suppressing verified progress.

## 6 CONCLUSION

In this work, we presented TOCOMAS, a topology-coherent architecture for self-evolving multi-agent systems. Rather than specifying agent roles, communication, and memory independently, TOCOMAS grounds task nodes in tool requirements, groups compatible regions into agent responsibilities, and derives handoffs and memory boundaries from the resulting organization. Online evolution compiles structural proposals into task-conditioned organizations and retains improvements that satisfy executable and coherence constraints. Experiments on BBEH, WorkBench, and SWE-Bench Verified show improved task performance over the evaluated baselines; CoMemBench further reveals gains over the self-evolving baseline in verified node completion, accepted handoffs, and isolation from polluted context. Ablations identify distinct contributions from agent organization, collaborative orchestration, memory coordination, and structural evolution. These results support topological coherence as a guiding principle for organizing and evolving multi-agent systems.

## REPRODUCIBILITY STATEMENT

Experimental settings are detailed in Section 5.1. Native evaluators and retained execution receipts, handoff traces, and model-call ledgers support auditing.

## AI USE STATEMENT

Generative AI tools assisted with manuscript drafting and language editing. The authors reviewed AI-assisted material and remain responsible for the scientific claims and reported results.

## REFERENCES

Cristian Bodnar, Fabrizio Frasca, Yuguang Wang, Nina Otter, Guido F. Montufar, Pietro Lio, and Michael Bronstein. Weisfeiler and lehman go topological: Message passing simplicial networks. In Proceedings ofthe 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pp. 1026–1037, 2021.

Guangyao Chen, Siwei Dong, Yu Shu, Ge Zhang, Jaward Sesay, Börje F. Karlsson, Jie Fu, and Yemin Shi. AutoAgents: A framework for automatic agent generation. arXiv preprint arXiv:2309.17288, 2023.

Weize Chen, Yusheng Su, Jingwei Zuo, Cheng Yang, Chenfei Yuan, Chi-Min Chan, Heyang Yu, Yaxi Lu, Yi-Hsin Hung, Chen Qian, et al. AgentVerse: Facilitating multi-agent collaboration and exploring emergent behaviors. In International Conference on Learning Representations, 2024.

Yilun Du, Shuang Li, Antonio Torralba, Joshua B. Tenenbaum, and Igor Mordatch. Improving factuality and reasoning in language models through multiagent debate. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 11733–11763, 2024.

Taicheng Guo, Xiuying Chen, Yaqi Wang, Ruidi Chang, Shichao Pei, Nitesh V. Chawla, Olaf Wiest, and Xiangliang Zhang. Large language model based multi-agents: A survey of progress and challenges. arXiv preprint arXiv:2402.01680, 2024.

Jakob Hansen and Robert Ghrist. Toward a spectral theory of cellular sheaves. Journal ofApplied and Computational Topology, 3:315–358, 2019.

Guangya Hao, Yunbo Long, and Zhuokai Zhao. Self-evolving multi-agent systems via decentralized memory. arXiv preprint arXiv:2605.22721, 2026.

Xiaoxin He, Yijun Tian, Yifei Sun, Nitesh V. Chawla, Thomas Laurent, Yann LeCun, Xavier Bresson, and Bryan Hooi. G-Retriever: Retrieval-augmented generation for textual graph understanding and question answering. In Advances in Neural Information Processing Systems, volume 37, pp. 132876–132907, 2024.

Sirui Hong, Mingchen Zhuge, Jonathan Chen, Xiawu Zheng, Yuheng Cheng, Jinlin Wang, Ceyao Zhang, Zili Wang, Steven Ka Shing Yau, Zijuan Lin, et al. MetaGPT: Meta programming for a multi-agent collaborative framework. In International Conference on Learning Representations, 2024.

Shengran Hu, Cong Lu, and Jeff Clune. Automated design of agentic systems. In International Conference on Learning Representations, 2025.

Yuntong Hu, Yuting Zhang, Matthew Trager, Yi Zhang, Shuo Yang, Wei Xia, and Stefano Soatto. EvoMAS: Evolutionary generation of multi-agent systems. In Proceedings of the 43rd International Conference on Machine Learning, 2026.

Carlos E. Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik Narasimhan. SWE-bench: Can language models resolve real-world github issues? arXiv preprint arXiv:2310.06770, 2023.

Mehran Kazemi, Bahare Fatemi, Hritik Bansal, John Palowitch, Chrysovalantis Anastasiou, Sanket Vaibhav Mehta, Lalit K. Jain, Virginia Aglietti, Disha Jindal, Peter Chen, Nishanth Dikkala, Gladys Tyen, Xin Liu, Uri Shalit, Silvia Chiappa, Kate Olszewska, Yi Tay, Vinh Q. Tran, Quoc V. Le, and Orhan Firat. BIG-Bench Extra Hard. In Proceedings of the 63rd Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pp. 26473–26501, 2025.

Zixuan Ke, Yifei Ming, Austin Xu, Ryan Chin, Xuan-Phi Nguyen, Prathyusha Jwalapuram, Semih Yavuz, Caiming Xiong, and Shafiq Joty. MAS-Orchestra: Understanding and improving multiagent reasoning through holistic orchestration and controlled benchmarks. In Proceedings of the 43rd International Conference on Machine Learning, 2026.

Dapeng Li, Zhen Tan, Peng Qian, Yuheng Li, Kunal Chaudhary, Li Hu, and Jianwei Shen. SMoA: Improving multi-agent large language models with sparse mixture-of-agents. In Pacific-Asia Conference on Knowledge Discovery and Data Mining, pp. 54–65. Springer, 2025.

Guohao Li, Hasan Abed Al Kader Hammoud, Hani Itani, Dmitrii Khizbullin, and Bernard Ghanem. CAMEL: Communicative agents for “mind” exploration of large scale language model society. In Advances in Neural Information Processing Systems, volume 36, pp. 51991–52008, 2023.

Shijun Li, Hilaf Hasson, and Joydeep Ghosh. OMAC: A holistic optimization framework for LLMbased multi-agent collaboration. In Proceedings ofthe 43rd International Conference on Machine Learning, 2026a.

Shiyuan Li, Yixin Liu, Qingsong Wen, Chengqi Zhang, and Shirui Pan. Assemble your crew: Automatic multi-agent communication topology design via autoregressive graph generation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pp. 23142–23150, 2026b.

Xin Li, Shiyi Wang, Siqi Zeng, Yu Wu, and Yi Yang. A survey on llm-based multi-agent systems: Workflow, infrastructure, and challenges. Vicinagearth, 1(1):9, 2024.

Zijun Liu, Yanzhe Zhang, Peng Li, Yang Liu, and Diyi Yang. A dynamic llm-powered agent network for task-oriented agent collaboration. In First Conference on Language Modeling, 2024.

Chen Qian, Wei Liu, Hongzhang Liu, Nuo Chen, Yufan Dang, Jiahao Li, Cheng Yang, Weize Chen, Yusheng Su, Xin Cong, et al. ChatDev: Communicative agents for software development. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 15174–15186, 2024.

Oliver Styles, Sam Miller, Paloma Cerda-Mardini, Tirthankar Guha, Victor Sanchez, and Bertie Vidgen. WorkBench: A benchmark dataset for agents in a realistic workplace setting. arXiv preprint arXiv:2405.00823, 2024.

Jiabin Tang, Yuhao Yang, Wei Wei, Lei Shi, Lixin Su, Suqi Cheng, Dawei Yin, and Chao Huang. GraphGPT: Graph instruction tuning for large language models. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval, pp. 491–500, 2024.

Kaixiang Wang, Yidan Lin, Jiong Lou, Zhaojiacheng Zhou, Bunyod Suvonov, and Jie Li. Emem: Multi-agent based episodic context reconstruction for llm agent memory. arXiv preprint arXiv:2601.21714, 2026.

Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Beibin Li, Erkang Zhu, Li Jiang, Xiaoyun Zhang, Shaokun Zhang, Jiale Liu, et al. AutoGen: Enabling next-gen llm applications via multi-agent conversations. In First Conference on Language Modeling, 2024.

Tongtong Wu, Yanming Li, Ziye Tang, Chen Jiang, Linhao Luo, Guilin Qi, Shirui Pan, and Gholamreza Haffari. CARD: Towards conditional design of multi-agent topological structures. In International Conference on Learning Representations, 2026.

Zhenran Xu, Senbao Shi, Baotian Hu, Jindi Yu, Dongfang Li, Min Zhang, and Yuxiang Wu. Towards reasoning in large language models via multi-agent peer review collaboration. arXiv preprint arXiv:2311.08152, 2023.

Ke Yang, Zixi Chen, Xuan He, Jize Jiang, Michel Galley, Chenglong Wang, Jianfeng Gao, Jiawei Han, and ChengXiang Zhai. PlugMem: A task-agnostic plugin memory module for llm agents. In Proceedings of the 43rd International Conference on Machine Learning, 2026.

Yingxuan Yang, Huacan Chai, Shuai Shao, Yuanyi Song, Siyuan Qi, Renting Rui, and Weinan Zhang. AgentNet: Decentralized evolutionary coordination for llm-based multi-agent systems. In Advances in Neural Information Processing Systems, volume 38, 2025.

Siyu Yuan, Kaitao Song, Jiangjie Chen, Xu Tan, Dongsheng Li, and Deqing Yang. EvoAgent: Towards automatic multi-agent generation via evolutionary algorithms. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pp. 6192–6217, 2025.

Guibin Zhang, Muxin Fu, Kun Wang, Guancheng Wan, Miao Yu, and Shuicheng Yan. G-Memory: Tracing hierarchical memory for multi-agent systems. In Advances in Neural Information Processing Systems, volume 38, 2025a.

Guibin Zhang, Yanwei Yue, Xiangguo Sun, Guancheng Wan, Miao Yu, Junfeng Fang, Kun Wang, Tianlong Chen, and Dawei Cheng. G-Designer: Architecting multi-agent communication topologies via graph neural networks. In Proceedings ofthe 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pp. 76678–76692, 2025b.

Jiayi Zhang, Jinyu Xiang, Zhaoyang Yu, Fengwei Teng, Xionghui Chen, Jiaqi Chen, Mingchen Zhuge, Xin Cheng, Sirui Hong, Jinlin Wang, Bingnan Zheng, Bang Liu, Yuyu Luo, and Chenglin Wu. AFlow: Automating agentic workflow generation. In International Conference on Learning Representations, 2025c.

Sen Zhao, Ruiqi Kong, Zuyu Zhang, Lifeng Shen, Xinyu He, Ding Zou, Xu Zhang, and Qinghua Zhang. CoMemBench: Benchmarking collaborative memory boundaries across multi-agent workflow topologies, 2026. URL https://arxiv.org/abs/2609.32192.

Han Zhou, Xingchen Wan, Ruoxi Sun, Hamid Palangi, Shariq Iqbal, Ivan Vulic, Anna Korhonen, and´ Sercan Ö. Arık. Multi-agent design: Optimizing agents with better prompts and topologies. In International Conference on Learning Representations, 2026.

Mingchen Zhuge, Wenyi Wang, Louis Kirsch, Francesco Faccio, Dmitrii Khizbullin, and Jürgen Schmidhuber. GPTSwarm: Language agents as optimizable graphs. In Proceedings ofthe 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 62743–62767, 2024.

## A BENCHMARK AND METRIC DETAILS

## A.1 BENCHMARK INSTANCES

BBEH (Kazemi et al., 2025) tests difficult reasoning questions against benchmark answers. Work-Bench (Styles et al., 2024) tests workplace tasks involving application tools. SWE-Bench-Verified (Jimenez et al., 2023) tests repository patches against its official resolution criterion. We use each benchmark’s native evaluator.

CoMemBench (Zhao et al., 2026) provides executable workflows in four domains: stateful tool use, repository code, offline retrieval, and formal mathematics. It contains 800 clean workflows (200 per domain) and 800 matched polluted variants. A workflow specifies a request, canonical task nodes and dependency edges, node-local execution requirements, typed handoff contracts, resources, and a domain-native evaluator. Dependency edges represent required task relations, not predefined agent communication edges. Verified node receipts and artifact-exposure records support the progress and handoff metrics below. In a matched polluted variant, one contextually inapplicable artifact is added to a target consumer’s incoming artifact view; the query, graph, initial state, tools, and non-target inputs remain unchanged.

## A.2 METRIC DEFINITIONS AND AGGREGATION

For BBEH, WorkBench, and SWE-Bench-Verified, we report their respective benchmark-level metrics: answer accuracy, task success rate, and official resolved rate. The definitions below apply specifically to CoMemBench.

For CoMemBench, let $\mathcal { C }$ be the evaluated clean workflows. The canonical required nodes of workflow q are $\mathcal { U } _ { q } ,$ consistent with $T _ { q }$ in Section $4 ;$ let $\mathcal { U } _ { q } ^ { \mathrm { v e r } } \subseteq \mathcal { U } _ { q }$ contain nodes with valid native-verifier receipts. Terminal Success Rate (SR) and Verified Node Completion Rate (VNCR) are:

$$
\mathrm { S R } = \frac { 1 } { | { \mathcal C } | } \sum _ { q \in { \mathcal C } } \mathbb { I } [ \mathrm { t e r m i n a l } ( q ) = 1 ] , \qquad \mathrm { V N C R } = \frac { 1 } { | { \mathcal C } | } \sum _ { q \in { \mathcal C } } \frac { | { \mathcal U } _ { q } ^ { \mathrm { v e r } } | } { | { \mathcal U } _ { q } | } .\tag{18}
$$

VNCR averages within-workflow fractions, so workflows receive equal weight; method-private nodes and repeated attempts at a canonical node do not add to the numerator.

Let H contain each non-root consumer whose mandatory direct predecessors have verified. A consumer belongs to $\mathcal { H } ^ { + }$ only if all required predecessor artifacts are exposed at the declared boundary, current, receipt-valid, explicitly consumed, and followed by native-verifier success. Verified Handoff Success (VHS) measures required sharing:

$$
\mathrm { V H S } = \frac { | \mathcal { H } ^ { + } | } { | \mathcal { H } | } .\tag{19}
$$

For isolation, each matched pair z has a target consumer $t _ { z } .$ . Let $\mathcal { P }$ contain pairs whose target verifies in the clean run, and let ${ \mathcal { P } } ^ { + }$ contain those pairs whose target also verifies after the inapplicable artifact is introduced in the polluted run. Isolation Challenge Success (ICS) measures the preservation of a valid target outcome under this memory-boundary intervention:

$$
\mathrm { I C S } = \frac { | \mathcal { P } ^ { + } | } { | \mathcal { P } | } , \qquad \mathcal { P } = \bigcup _ { d \in \mathcal { D } } \mathcal { P } _ { d } ,\tag{20}
$$

where $\mathcal { D }$ is the set of four CoMemBench domains. The aggregate is micro-pooled over eligible target pairs, not averaged over domain percentages; eligibility requires clean target success, not full clean-workflow success. An eligible polluted target that is not reached or fails verification does not enter ${ \mathcal { P } } ^ { + }$ . ICS does not require an explicit rejection message. For VHS or ICS, a cohort with no eligible cases is reported as unavailable rather than zero; infrastructure failures are tracked separately from method outcomes. All four CoMemBench ratios are multiplied by 100 when reported as percentages.