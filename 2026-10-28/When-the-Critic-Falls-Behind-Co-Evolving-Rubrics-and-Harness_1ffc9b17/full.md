# When the Critic Falls Behind: Co-Evolving Rubrics and Harnesses for Continual Agent Self-Improvement

Hao Chen<sup>1</sup>, Yukun Yan<sup>2</sup>\*, Ziyu Han<sup>1</sup>, Ye He<sup>1</sup>, Chuanzhe Guo<sup>1</sup>, Shangzhan Li<sup>1</sup>, Qi Shi<sup>2</sup>, Qingfu Zhu<sup>1</sup>, Maosong Sun<sup>2</sup>, Wanxiang Che<sup>1</sup>\*

<sup>1</sup>Research Center for Social Computing and Interactive Robotics, Harbin Institute of Technology <sup>2</sup>Department of Computer Science and Technology, Institute for AI, Tsinghua University

## Abstract

Autonomous agents increasingly depend on harnesses that govern interaction, memory, tools, and execution around large language models (LLMs). While recent work has begun to evolve these harnesses from execution experience, the supervision used to guide evolution often remains relatively static, making it difficult to track changing behavioral weaknesses across rounds and potentially limiting generalization to unseen tasks. To address this issue, we propose RuHarness, a rubric–harness coevolution framework for continual agent selfimprovement. RuHarness continually discovers and reassesses behavioral criteria from execution trajectories, prioritizes the most relevant unresolved weaknesses, and uses them to guide targeted harness updates while keeping the task objective fixed. Extensive experiments across diverse agent tasks and backbone models show that RuHarness outperforms strong harness-evolution baselines, particularly on unseen held-out tasks. Further analyses show that adaptive rubric prioritization tracks shifting weaknesses and supports more sustained improvement and stable generalization across evolution rounds.

## 1 Introduction

As autonomous agents become increasingly prevalent, their performance depends not only on the capabilities of the underlying large language model (LLM) but also on the surrounding harness that governs interaction, memory, tools, and execution (Luo et al., 2026; Pan et al., 2026). Recent work has begun to evolve the harness itself from execution experience, enabling agents to improve without updating model parameters (Zhang et al., 2026a; Lin et al., 2026b; Lee et al., 2026a). While such approaches can yield substantial gains, recent controlled evaluations suggest that these improvements may weaken on unseen tasks (Wang et al., 2026c), raising questions about whether harness optimization produces generalizable improvement.

![](images/00f49834b3e740388cae947142655b2e43b69ab3b8b9129d359d4a1275181fc1.jpg)  
Figure 1: Illustration of rubric–harness co-evolution. Static rubrics may fall behind as behavioral weaknesses shift during harness evolution, while co-evolving rubrics track emerging failures and guide harness improvement.

One challenge behind this limited generalization is that repeated harness optimization may gradually overfit to task-specific failure patterns exposed during evolution (Srikanth et al., 2026). Existing methods optimize harness against task-level feedback, which indicates whether the agent succeeds but provides limited information about the behavioral deficiencies underlying failure (Zhang et al., 2026a). This limitation becomes increasingly important during continual evolution: as the harness improves, previously dominant failures may be mitigated, while subtler or previously unseen failure patterns emerge (Nie et al., 2026). Figure 1 illustrates that when the guidance used for evolution remains coarse-grained and relatively static, its discriminative and guiding value may gradually diminish (Kim et al., 2025; Tian et al., 2026). Consequently, the system lacks an explicit mechanism for tracking how the importance of different weaknesses changes throughout evolution. Continual harness evolution requires adaptive diagnosis that continually reassesses and reprioritizes behavioral weaknesses while keeping the underlying task objective fixed (Dou et al., 2026; Zhang et al., 2026c).

To this end, we propose RuHarness (Rubric– Harness Co-Evolution), a framework for continual agent self-improvement through joint rubric and harness evolution. RuHarness continually derives behavioral criteria from execution trajectories, reassesses their relevance under the current harness, and prioritizes those that best capture unresolved weaknesses. The resulting rubric frontier guides targeted and coordinated updates to the corresponding harness components, while candidate harnesses are selected using task performance on an independent validation set. By adapting the diagnostic guidance as agent behavior changes while keeping the task objective fixed, RuHarness continually focuses harness optimization on the weaknesses that matter most at each stage of evolution.

Extensive experiments across diverse agent tasks and backbone models demonstrate that RuHarness consistently outperforms strong harness-evolution baselines, with particularly clear gains on unseen held-out tasks. The analyses also show that the rubric frontier adapts to shifting behavioral weaknesses, concentrates on diagnostically relevant criteria, and supports more sustained improvement across evolution rounds. These results indicate that co-evolving rubrics with the harness provides adaptive supervision for continual agent improvement, supporting both continued evolution across evolution rounds and stronger generalization.

## 2 Related Work

Self-Evolving Agents. Agent self-evolution has gradually shifted from storing interaction experience to optimizing persistent external state. Reflection- and memory-based methods refine experience to improve future decisions (Shinn et al., 2023; Zhong et al., 2023), while skill evolution treats reusable skills as editable knowledge that can be refined from execution feedback (Zhu et al., 2026; Yang et al., 2026). These paradigms extend naturally to the broader agent harness, where the optimization target expands from a single external component to the execution infrastructure, including prompts, tools, memory, and control logic (Zhang et al., 2026d). Recent harness-evolution methods seek to diagnose failures from trajectories and iteratively revise this infrastructure (Jiang et al., 2026b), with subsequent work broadening the editable space and making the evolution process more observable and verifiable (Lee et al., 2026b; Lin et al., 2026a). More broadly, recent studies have explored recursive self-improvement and harness–model co-evolution across repeated adaptation rounds (Xia et al., 2026; Yu et al., 2026; Chen et al., 2026b; Fan and Huang, 2026). Yet these approaches still largely treat the diagnostic signals guiding evolution as fixed, leaving how such guidance should adapt to shifting behavioral weaknesses underexplored.

Rubrics for Self-Improvement. Rubrics have evolved from structured evaluation criteria into a flexible interface between evaluation and optimization (Chen et al., 2026a). By decomposing complex objectives into explicit behavioral criteria, they provide fine-grained supervision and have increasingly been incorporated into reward modeling and reinforcement learning (Gunjal et al., 2025; Li et al., 2026a). More recent work makes the rubric itself adaptive: EvoLM (Li et al., 2026b), EvoRubrics (Ding et al., 2026), and DynamicRubric (Wang et al., 2026a) co-evolve evaluative criteria with the optimized policy (Guan et al., 2026), while Wang et al. (2026b) extends this idea to a self-evolving loop over responses, rubrics, and policy. These studies suggest that effective supervision should evolve with the behavior it guides (Jiang et al., 2026a). However, existing work mainly co-evolves rubrics with policies or responses (Li et al., 2026d), leaving rubric adaptation for continual harness evolution largely unexplored.

## 3 Method

In this section, we first present the overall design of RuHarness (§3.1), then describe how rubric evolution tracks the changing behavioral distribution (§3.2) and guides subsequent harness evolution (§3.3), as illustrated in Figure 2.

## 3.1 Overview of the Framework

RuHarness establishes a co-evolution process between the diagnostic rubric and the agent harness through iterative interaction. At evolution round t, the system state is represented by $\boldsymbol { S } _ { t } = \left( \boldsymbol { H } _ { t } , \boldsymbol { R } _ { t } \right)$ where $H _ { t }$ denotes the current harness and $R _ { t }$ the maintained rubric pool. RuHarness updates the diagnostic signal from the behavior exposed by the current harness and uses the resulting signal to guide harness updates, thereby alternating between rubric evolution and harness evolution.

![](images/49c91f75ec08372b3b6059e603dc3fa2c4eda5011c192f0f4ed73f6ae8939313.jpg)  
Figure 2: Overview of the Rubric–Harness Co-Evolution Framework. Rubric evolution adaptively identifies and prioritizes unresolved behavioral weaknesses from trajectory evidence, while harness evolution performs targeted updates accordingly, yielding new behaviors that drive the next round of evolution.

Alternating evolution process. At each round, the current harness $H _ { t }$ is executed on a set of evolution tasks $D _ { \mathrm { e v o } }$ to collect agent trajectories $\mathcal { T } _ { t }$ These trajectories are analyzed to identify remaining failure patterns, which drive the rubric update:

$$
R _ { t + 1 } = \mathcal { E } _ { R } \big ( R _ { t } , \mathcal { T } _ { t } \big ) ,\tag{1}
$$

where $\mathcal { E } _ { R }$ discovers new behavioral criteria and reassesses existing ones from $\mathcal { T } _ { t }$ detailed in §3.2. The updated rubric pool is then prioritized to obtain a set of high-priority criteria $\widehat { R } _ { t + 1 }$ , which captures the behavioral weaknesses most relevant to the current harness. These criteria guide the rubricconditioned harness update:

$$
H _ { t + 1 } = \mathcal { E } _ { H } \big ( H _ { t } , \widehat { R } _ { t + 1 } \big ) ,\tag{2}
$$

where $\mathcal { E } _ { H }$ denotes the harness evolution procedure detailed in §3.3. The updated harness induces a new behavioral distribution in the next round, providing fresh evidence for subsequent rubric evolution and closing the co-evolution loop. The two processes are therefore coupled through agent behavior: the harness shapes the weaknesses that emerge, while the rubric determines which weaknesses are prioritized to guide subsequent evolution.

## 3.2 Rubric Evolution: Adapting to Shifting Agent Weaknesses

Given trajectories from $H _ { t }$ , RuHarness summarizes each failed run into a structured failure report and aggregates these reports to identify recurring cross-task behavioral patterns. These patterns are converted into new criteria $\mathcal { R } _ { t + 1 } ^ { \mathrm { n e w } }$ , which capture recurring weaknesses across tasks while remaining verifiable from trajectory evidence.

Outcome-blind behavioral assessment. Newly discovered criteria are evaluated together with existing criteria on trajectories generated by the current harness. For each criterion c and trajectory τ , an outcome-blind judge assigns:

$$
J ( c , \tau ) \in \{ \mathrm { S a t i s f i e d } , \mathrm { V i o l a t e d } , \mathrm { N } / \mathrm { A } \} ,\tag{3}
$$

where the judge has access only to behavioral evidence in the trajectory and is blinded to the final task outcome. This prevents criterion assessment from directly leaking success or failure labels. Here, N/A indicates that the behavior specified by the criterion does not occur, whereas Satisfied indicates that it occurs and is handled correctly.

Criterion admission and lifecycle. Based on these assessments, newly discovered criteria first undergo a birth check stage before entering the maintained rubric pool. Criteria with insufficient diagnostic value are filtered at this stage, including those that are rarely applicable, fail to identify violations on unsuccessful trajectories, or trigger indiscriminately across outcomes. Criteria that pass this admission stage enter a four-state lifecycle: probation, active, dormant, and retired. Newly admitted criteria enter probation and are promoted to active once they exhibit sufficient diagnostic value. Active criteria become dormant when their associated failures are no longer observed, and may be reactivated upon recurrence, while persistently uninformative criteria are eventually retired. Only active criteria are subsequently prioritized and used to guide harness evolution.

Time-varying diagnostic utility. To determine which active criteria are most useful for the current harness, we assign each criterion a time-dependent diagnostic utility:

$$
\begin{array} { r l } & { U _ { t + 1 } ( c ) = \alpha D _ { t + 1 } ( c ) + \beta C _ { t + 1 } ( c ) + \gamma P _ { t + 1 } ( c ) } \\ & { \qquad + \delta N _ { t + 1 } ( c ) - \lambda \mathrm { R e d } _ { t + 1 } ( c ) , } \end{array}\tag{4}
$$

where $D _ { t + 1 } ( c )$ denotes discrimination between successful and failed trajectories, $C _ { t + 1 } ( c )$ measures the fraction of current failures covered by the criterion, $P _ { t + 1 } ( c )$ measures predictive validity with respect to task failure, $N _ { t + 1 } ( c )$ captures novel failures identified by the criterion, and ${ \mathrm { R e d } } _ { t + 1 } ( c )$ penalizes behavioral redundancy with existing criteria. These statistics are derived from the preceding behavioral assessments and the corresponding task outcomes, and are recomputed for each harness state. Consequently, $\boldsymbol { U } _ { t + 1 } ( c )$ evolves with the harness: criteria associated with mitigated failures decrease in priority, while those capturing newly prominent failures receive higher utility.

Adaptive rubric frontier. Since statistics can be uncertain under limited observations, we rank active criteria using a conservative posterior estimate of diagnostic utility. Specifically, we estimate a Dirichlet posterior from the joint task-outcome and criterion-assessment counts and use the lower q-quantile of the induced utility distribution:

$$
\widetilde { U } _ { t + 1 } ( \boldsymbol { c } ) = Q _ { q } [ p ( U _ { t + 1 } ( \boldsymbol { c } ) \mid \mathcal { T } _ { t } ) ] ,\tag{5}
$$

where $\widetilde U _ { t + 1 } ( c )$ denotes the conservative utility of criterion $c ,$ and $Q _ { q } ( \cdot )$ is the q-th posterior quantile. This reduces the influence of criteria whose high utility is supported by only a few observations. We select the top-K criteria from the active rubric pool:

$$
\widehat { R } _ { t + 1 } = \mathrm { T o p K } _ { c \in R _ { t + 1 } ^ { \mathrm { a c t i v e } } } \widetilde { U } _ { t + 1 } ( c ) ,\tag{6}
$$

where $R _ { t + 1 } ^ { \mathrm { a c t i v e } }$ denotes the active criteria and $\widehat { R } _ { t + 1 }$ the resulting rubric frontier. The resulting frontier prioritizes unresolved behavioral weaknesses that remain both diagnostically informative and statistically reliable under the current harness.

Further details of rubric evolution are provided in Appendix A.2–A.4.

## 3.3 Harness Evolution: Optimizing against the Moving Frontier

To structure the harness search space, we represent the editable harness using six functional coordinates: Equip, Instruct, Act, Observe, Remember, and Terminate. These coordinates respectively capture resource provisioning, behavioral instruction, action policy, observation processing, context management, and completion control. Each coordinate defines a corresponding set of editable components in the agent system.

Frontier-guided targeting. Given the current rubric frontier $\widehat { R } _ { t + 1 }$ , RuHarness identifies the harness coordinates that can directly address each prioritized behavioral weakness. The criterion-tocoordinate mapping is not constrained to be oneto-one, since a single failure pattern may involve multiple parts of the harness. Instead, each criterion may target multiple coordinates, allowing the corresponding interventions to be planned jointly.

Coordinated candidate generation. Based on these criterion-to-coordinate assignments, RuHarness jointly plans a small set of candidate harness updates. Each candidate specifies a concrete hypothesis for addressing the diagnosed weaknesses and may modify multiple related coordinates jointly. We represent each candidate as a coordinated set of coordinate-level modifications:

$$
\Delta _ { j } = \left\{ \delta _ { j } ^ { ( k ) } \right\} _ { k \in M _ { j } } ,\tag{7}
$$

where $M _ { j }$ is the set of modified coordinates and $\delta _ { i } ^ { ( k ) }$ denotes the corresponding modification. The coordinated modifications are applied to the current harness through the update operator ⊕, yielding the candidate harness:

$$
{ \cal H } _ { t } ^ { ( j ) } = { \cal H } _ { t } \oplus \Delta _ { j } .\tag{8}
$$

Each candidate is treated as a single multicoordinate update, preserving the interactions among jointly planned modifications. Before evaluation, each candidate undergoes a lightweight verification step to check whether the proposed modifications are consistent with the diagnosed weaknesses and supporting evidence. The verifier only accepts or rejects the candidate and does not rewrite it or estimate task performance.

Validation-gated update. Candidates that pass verification are evaluated together with the current harness on a separate set of validation tasks $D _ { \mathrm { v a l } }$ , used exclusively for candidate selection and excluded from both evolution and final evaluation. This separation decouples candidate selection from the trajectories used to drive evolution. Let $S ( H , D _ { \mathrm { v a l } } )$ denote the average validation score of harness H. Let $\mathcal { H } _ { t } ^ { \mathrm { c a n d } }$ denote the set of candidate harnesses generated at round t. The candidate with the highest validation score is selected as:

$$
H _ { t } ^ { \star } \in \arg \operatorname* { m a x } _ { H \in \mathcal { H } _ { t } ^ { \mathrm { c a n d } } } S ( H , D _ { \mathrm { v a l } } ) ,\tag{9}
$$

and commit the update only if its improvement over the current harness exceeds a threshold ϵ:

$$
H _ { t + 1 } = \left\{ H _ { t } ^ { \star } , \quad S ( H _ { t } ^ { \star } , D _ { \mathrm { v a l } } ) - S ( H _ { t } , D _ { \mathrm { v a l } } ) > \epsilon , \right.\tag{10}
$$

A candidate is advanced to the next evolution round only if its validation improvement exceeds the acceptance threshold, with the current harness retained otherwise.

## 4 Experimental Methodology

This section presents our experimental setup, covering models and environment, benchmarks, metrics, baselines, and implementation details.

Models and Environment. We conduct experiments with Qwen3.5-35B-A3B (Team, 2026) and GLM-4.7-Flash (Team et al., 2025) as the backbone models. The agent is instantiated with Deep-Agent (Li et al., 2026c), which provides the basic execution framework for tool use, filesystem interaction, and multi-step task solving. Based on this framework, we construct the initial harness and expose its configurable components for subsequent evolution, while keeping the underlying model fixed throughout the experiments. All stages of task execution and evolution are performed using the corresponding backbone model itself.

Benchmarks and Metrics. Evaluation is conducted on EvoAgentBench (Gao et al., 2026), a multi-domain benchmark for long-horizon agent self-evolution. It covers five agent domains: information retrieval based on BrowseComp-Plus (Chen et al., 2025), mathematical reasoning based on OmniMath (Gao et al., 2024), software engineering based on SWE-Bench (Jimenez et al., 2024), code implementation based on LiveCodeBench (Jain et al., 2024), and knowledge work based on GDP-Val (Patwardhan et al., 2025). We use Pass@1 as the primary metric and evaluate each domain with its benchmark evaluator (Tang et al., 2026).

Baselines. For comprehensive comparison, we compare RuHarness against representative selfevolving agent methods that optimize different external states. ACE evolves persistent contextual knowledge from interaction experience (Zhang et al., 2026b), while EvoSkill discovers and refines reusable skills through failure analysis (Alzubi et al., 2026). For harness evolution, we include Self-Harness, which mines model-specific weaknesses and proposes targeted updates (Zhang et al., 2026a); Meta-Harness, which searches executable harness code using accumulated candidates and trajectories (Lee et al., 2026b); and Agentic Harness Engineering (AHE), which improves harnesses through observable and verifiable component updates (Lin et al., 2026a). Together, these baselines cover context, skill, and full-harness evolution.

Implementation Details. We use disjoint evolution (held-in) and validation splits, with the test set (held-out) reserved for final reporting only. At each round, RuHarness generates up to three coordinated harness candidates and accepts at most one update when its validation improvement exceeds $\epsilon \ : = \ : 0 . 0 5$ . For diagnostic utility, we set $\alpha = \gamma = 1 . 0$ and $\beta = \delta = \lambda = 0 . 5$ . Rubric guidance is obtained by ranking criteria with the q-th posterior quantile $( q = 0 . 2 )$ and selecting the Top-K criteria with $K = 1 0$ . Each candidate modifies at most three harness components. More experimental details are provided in the Appendix A.1.

## 5 Results and Analysis

In this section, we first evaluate the performance of RuHarness and conduct ablation studies. We then analyze the evolution of the rubric frontier and how adaptive prioritization guides harness improvement. Finally, we compare evolution trajectories to assess sustained improvement and generalization.

<table><tr><td rowspan="2">Method</td><td colspan="2">Information Retrieval</td><td colspan="2">Knowledge Work</td><td colspan="2">Software Engineering</td><td colspan="2">Code Implementation</td><td colspan="2">Math Reasoning</td><td colspan="2">Average</td></tr><tr><td>Held-in</td><td>Held-out</td><td>Held-in</td><td>Held-out</td><td>Held-in</td><td>Held-out</td><td>Held-in</td><td>Held-out</td><td>Held-in</td><td>Held-out</td><td>Held-in</td><td>Held-out</td></tr><tr><td colspan="10">Qwen3.5-35B-A3B</td><td></td><td></td><td></td></tr><tr><td>Vanilla Agent</td><td>28.0</td><td>20.0</td><td>51.7</td><td>40.0</td><td>40.0</td><td>50.0</td><td>83.6</td><td>65.1</td><td>69.5</td><td>46.0</td><td>54.6</td><td>44.2</td></tr><tr><td>ACE (2026b)</td><td>31.0</td><td>21.5</td><td>56.7</td><td>40.0</td><td>46.7</td><td>55.6</td><td>86.7</td><td>69.8</td><td>67.9</td><td>45.0</td><td>57.8</td><td>46.4</td></tr><tr><td>EvoSkill (2026)</td><td>36.0</td><td>27.7</td><td>53.3</td><td>42.0</td><td>44.4</td><td>47.2</td><td>85.9</td><td>68.6</td><td>70.3</td><td>47.0</td><td>58.0</td><td>46.5</td></tr><tr><td>Self-Harness (2026a)</td><td>34.0</td><td>27.7</td><td>53.3</td><td>44.0</td><td>51.1</td><td>55.6</td><td>90.6</td><td>67.4</td><td>72.9</td><td>47.0</td><td>60.4</td><td>48.3</td></tr><tr><td>AHE (2026a)</td><td>33.0</td><td>24.6</td><td>55.0</td><td>50.0</td><td>53.3</td><td>58.3</td><td>88.3</td><td>66.3</td><td>71.1</td><td>47.0</td><td>60.1</td><td>49.2</td></tr><tr><td>Meta-Harness (2026b)</td><td>34.0</td><td>26.2</td><td>58.3</td><td>36.0</td><td>51.1</td><td>52.8</td><td>89.1</td><td>68.6</td><td>71.1</td><td>49.0</td><td>60.7</td><td>46.5</td></tr><tr><td>RuHarness (ours)</td><td>37.0</td><td>30.8</td><td>60.0</td><td>52.0</td><td>53.3</td><td>63.9</td><td>89.8</td><td>69.8</td><td>72.1</td><td>52.0</td><td>62.4</td><td>53.7</td></tr><tr><td colspan="10">GLM-4.7-Flash</td><td rowspan="2"></td><td rowspan="2">33.6</td><td rowspan="2">24.5</td></tr><tr><td>Vanilla Agent</td><td>13.0</td><td>6.2</td><td>45.0</td><td>32.0</td><td>31.1</td><td>33.3</td><td>49.2</td><td>43.0</td><td>29.5</td></tr><tr><td>ACE (2026b)</td><td>14.0</td><td>4.6</td><td>53.3</td><td>42.0</td><td>37.8</td><td>44.4</td><td>54.7</td><td>47.7</td><td>26.8</td><td>11.0</td><td>37.3</td><td>29.9</td></tr><tr><td>EvoSkill (2026)</td><td>10.0</td><td>9.2</td><td>50.0</td><td>36.0</td><td>44.4</td><td>38.9</td><td>50.0</td><td>46.5</td><td>29.2</td><td>7.0</td><td>36.7</td><td>27.5</td></tr><tr><td>Self-Harness (2026a)</td><td>15.0</td><td>9.2</td><td>50.0</td><td>44.0</td><td>46.7</td><td>47.2</td><td>61.7</td><td>51.2</td><td>29.7</td><td>12.0</td><td>40.6</td><td>32.7</td></tr><tr><td>AHE (2026a)</td><td>15.0</td><td>7.7</td><td>46.7</td><td>36.0</td><td>42.2</td><td>41.7</td><td>60.2</td><td>50.0</td><td>29.2</td><td>10.0</td><td>38.7</td><td>29.1</td></tr><tr><td>Meta-Harness (2026b)</td><td>16.0</td><td>7.7</td><td>56.7</td><td>46.0</td><td>40.0</td><td>38.9</td><td>59.4</td><td>47.7</td><td>30.0</td><td>15.0</td><td>40.4</td><td>31.1</td></tr><tr><td>RuHarness (ours)</td><td>18.0</td><td>13.8</td><td>58.3</td><td>56.0</td><td>48.9</td><td>52.8</td><td>60.9</td><td>52.3</td><td>30.5</td><td>15.0</td><td>43.3</td><td>38.0</td></tr></table>

Table 1: Performance comparison across five domains on the held-in and held-out sets. Average denotes the macro-average across all five domains. The best result is shown in bold, and the second-best result is underlined.

## 5.1 Overall Performance

We evaluate RuHarness across five agent domains on EvoAgentBench using Qwen3.5-35B-A3B and GLM-4.7-Flash, as shown in Table 1. Overall, RuHarness outperforms all the baselines, achieving the best average performance on both held-in and held-out sets across the two backbone models.

Overall Effectiveness. On Qwen3.5-35B-A3B, RuHarness improves the strongest baseline by 1.7 points on held-in and 4.5 points on held-out. The gains are even larger on GLM-4.7-Flash, reaching 2.7 and 5.3 points, respectively. These results indicate that rubric–harness co-evolution provides stable improvements beyond a specific backbone.

Generalization to Unseen Tasks. The advantage of RuHarness is more pronounced on the heldout sets. Although several baselines achieve competitive or even better results on individual held-in tasks, RuHarness attains the highest average heldout performance for both models and exhibits a smaller held-in–held-out gap. This suggests that its improvements are less tied to the evolution tasks and transfer more effectively to unseen tasks.

Robustness across Tasks and Models. RuHarness performs consistently well across heterogeneous tasks and both backbone models, achieving the strongest average held-out performance and competitive results across individual domains. This suggests that its gains are not tied to a specific task type or backbone.

## 5.2 Ablation Study

We further examine the contribution of key components of RuHarness, as summarized in Table 2.

Necessity of the Rubric Module. Removing the rubric module lowers the average held-in score from 62.4 to 57.1 and the held-out score from 53.7 to 45.8. This indicates that directly evolving the harness from task feedback and trajectories provides insufficient behavioral diagnosis, highlighting the role of the rubric in translating task outcomes into actionable weaknesses.

Effect of Rubric Evolution. To isolate the effect of rubric evolution, we retain the initial rubric as a fixed diagnostic rubric throughout the evolution process. Surprisingly, this variant performs even worse than removing the rubric module entirely, suggesting that a static rubric can become counterproductive as the harness evolves. When previously dominant failures are mitigated, outdated criteria may continue to steer optimization toward less relevant behaviors. Further analysis of this experimental process is provided in Appendix A.6.

Role of the Rubric Frontier. Using the entire ranked rubric pool instead of the selected frontier also leads to performance degradation. This suggests that more rubrics do not necessarily provide better guidance, as focusing on relevant criteria reduces interference from low-value signals and yields more effective harness updates. A more detailed examination is provided in Appendix A.7.

<table><tr><td rowspan="2">Method</td><td colspan="2">Information Retrieval</td><td colspan="2">Knowledge Work</td><td colspan="2">Software Engineering</td><td colspan="2">Code Implementation</td><td colspan="2">Math Reasoning</td><td colspan="2">Average</td></tr><tr><td>Held-in</td><td>Held-out</td><td>Held-in</td><td>Held-out</td><td>Held-in</td><td>Held-out</td><td>Held-in</td><td>Held-out</td><td>Held-in</td><td>Held-out</td><td>Held-in</td><td>Held-out</td></tr><tr><td>RuHarness</td><td>37.0</td><td>30.8</td><td>60.0</td><td>52.0</td><td>53.3</td><td>63.9</td><td>89.8</td><td>69.8</td><td>72.1</td><td>52.0</td><td>62.4</td><td>53.7</td></tr><tr><td>w/o Rubric Module</td><td>31.0</td><td>23.1</td><td>51.7</td><td>44.0</td><td>48.9</td><td>52.8</td><td>82.8</td><td>65.1</td><td>71.1</td><td>44.0</td><td>57.1</td><td>45.8</td></tr><tr><td>w/o Rubric Evolution</td><td>35.0</td><td>29.2</td><td>55.0</td><td>34.0</td><td>44.4</td><td>55.6</td><td>76.6</td><td>57.0</td><td>71.3</td><td>46.0</td><td>56.5</td><td>44.4</td></tr><tr><td>w/o Rubric Frontier</td><td>35.0</td><td>30.8</td><td>56.7</td><td>50.0</td><td>46.7</td><td>50.0</td><td>84.4</td><td>67.4</td><td>72.6</td><td>47.0</td><td>59.1</td><td>49.0</td></tr></table>

Table 2: Ablation study of RuHarness on Qwen3.5-35B-A3B across five domains.

![](images/a56686cb026ea1ffaf9a48654a6e1daf667b48199d59e8dc766ad9e3c5ea8d1a.jpg)  
(a) Qwen3.5 / Code

![](images/1f1e79e9c2926aa40f504646468fd3926944e2a622de6eb547d5e242be3f2a57.jpg)  
(b) GLM-4.7 / SWE  
Figure 3: Rubric similarity across evolution rounds.

## 5.3 Analysis of Adaptive Rubric Tracking

To examine how the diagnostic focus changes throughout harness evolution, we measure the Jaccard similarity between rubric frontiers across rounds. Specifically, each frontier is represented by its selected criterion IDs, and we compare the frontier at round t with both the initial frontier $F _ { 0 }$ and the frontier from the preceding round $F _ { t - 1 }$ . We conduct this analysis on two representative settings, Qwen3.5-35B-A3B on Code Implementation and GLM-4.7-Flash on Software Engineering.

As shown in Figure 3, the similarity to the initial frontier decreases sharply in the early rounds and remains low thereafter. By contrast, the similarity between consecutive frontiers first declines, then gradually increases and stabilizes at a relatively high level. These two trends reveal a characteristic evolution pattern: the diagnostic frontier progressively departs from its initial configuration, while changes between adjacent rounds become increasingly gradual. This behavior suggests that early harness updates substantially reshape the set of weaknesses that require attention, whereas later evolution proceeds through more localized adjustments. Overall, the results confirm that the behavioral weaknesses most relevant to harness improvement are not static, supporting the need for an evolving rubric that continuously tracks changes in the agent’s failure landscape. A finer-grained visualization of criterion-level state transitions across evolution rounds is provided in Figure 6.

![](images/91e0e94f87b09ea8e6eb33a6af500ec18c6c744542b3225c914aeecd84aa14cc.jpg)  
(a) Diagnostic changes across frontier transitions.

![](images/9e8d0f98f0151dd4528966e62e3d31dd09aa2b08e2d1830fe7c2eddf8889947a.jpg)

![](images/c33cca86f093a2bbee87f72ee645d744b363b33608b2a38fd779ae301b1dfe32.jpg)  
(b) Violation rates after criteria become prioritized.  
(c) Violation reduction with and without harness updates.  
Figure 4: Analysis of adaptive rubric prioritization and subsequent harness improvement.

## 5.4 Effectiveness of Adaptive Rubric Prioritization on Harness Evolution

To examine whether adaptive rubric prioritization translates into subsequent behavioral improvement, we analyze criterion-level dynamics around frontier transitions and after frontier entry. Specifically, we first compare how the diagnostic components of criteria change when they enter, remain in, exit, or stay outside the frontier. We then align newly prioritized criteria at their frontier-entry round and track how their violation rates evolve over subsequent rounds. To further relate this trend to harness evolution, we additionally distinguish cases with and without committed harness updates.

Frontier transitions track diagnostic relevance. To characterize how diagnostic relevance changes with frontier membership, we group criteria by their transitions between two adjacent evolution rounds: entering the frontier, remaining in the frontier, exiting the frontier, or staying outside the frontier. For each group, we measure the roundto-round changes in the diagnostic utility components defined in Section 3.2, as shown in Figure 4a. As expected, criteria entering the frontier show increases in discrimination, coverage, and predictive validity together with reduced redundancy, while exiting criteria exhibit the opposite pattern. Criteria that remain inside or outside the frontier change substantially less. These results confirm that frontier turnover reflects changes in criterion-level diagnostic relevance rather than arbitrary replacement.

Prioritized weaknesses are progressively mitigated. To assess whether frontier prioritization translates into subsequent behavioral improvement, Figure 4b examines what happens after a criterion enters the frontier. The violation rate remains nearly unchanged in the first round after entry, but decreases noticeably over the following rounds. This delayed reduction suggests that newly prioritized weaknesses are not necessarily resolved immediately, but are progressively mitigated as subsequent harness evolution proceeds. The trend therefore links frontier prioritization with later behavioral improvement, suggesting that the evolving rubric does more than reorganize diagnostic signals and helps focus subsequent optimization on weaknesses that can be progressively reduced.

Committed harness updates yield stronger longer-horizon mitigation. To more directly relate behavioral improvement to harness evolution, Figure 4c compares frontier-entry criteria under committed and non-committed harness updates, where committed denotes a candidate update that passes the validation gate and is adopted for subsequent evolution. The committed group shows a clear and sustained increase in violation reduction over subsequent rounds, whereas the non-committed group fluctuates without a consistent improvement trend. This contrast indicates that frontier-guided prioritization is more likely to translate into persistent behavioral gains when the proposed harness update is successfully validated and adopted.

Taken together, these results show that adaptive rubric prioritization tracks changing diagnostic relevance and helps focus subsequent harness evolution on weaknesses that are progressively mitigated.

## 5.5 Evolution Progress and Generalization

To examine performance across evolution rounds, Figure 5 compares the evolution trajectories of different harness evolution methods on the Qwen3.5- 35B-A3B model with the knowledge work task.

![](images/c2157d1a3fa8f9a3769393b191a8ebe150646949084540c47610b1bd9578afa5.jpg)  
(a) Held-in Task

![](images/db9e60b800f560ea8630ae36b9dba600379b3b4ac8f6fbc9e207e36e09ddcbbe.jpg)  
(b) Held-out Task  
Figure 5: Evolution trajectories of different methods.

We report best-so-far held-in performance to reflect cumulative progress, while held-out performance is evaluated at each round to track generalization.

As shown in Figure 5a, on the held-in evolution tasks, several baselines improve quickly in early rounds but soon plateau, whereas RuHarness continues to gain in later rounds and achieves the highest performance. This suggests that dynamically updating the diagnostic focus helps expose new optimization opportunities after previously dominant weaknesses have been addressed.

The held-out trajectories in Figure 5b show that baseline performance fluctuates more substantially across rounds, while RuHarness exhibits a more stable upward trend in later stages. This suggests that the improvements acquired during evolution transfer more consistently beyond the held-in tasks. Together, these results indicate that adaptive rubric guidance supports both sustained improvement on held-in tasks and more stable generalization to heldout tasks throughout the evolution process.

## 6 Conclusion

This paper presents RuHarness, a rubric–harness co-evolution framework for continual agent selfimprovement. RuHarness dynamically adapts diagnostic criteria to the agent’s evolving behavioral weaknesses and uses the resulting rubric frontier to guide targeted harness optimization. Extensive experiments across diverse tasks and backbone models show that RuHarness achieves strong gains in both held-in performance and generalization to unseen held-out tasks. Further analyses show that the rubric frontier tracks shifting diagnostic relevance and supports sustained harness improvement over repeated evolution rounds. These results highlight the importance of jointly adapting diagnostic supervision and the agent harness for more sustained and generalizable agent evolution.

## Limitations

Although RuHarness is evaluated across multiple tasks and backbone models, all experiments are conducted within a single underlying agent framework. Evaluating the six-coordinate harness abstraction on a broader range of agent architectures would further establish its generality. In addition, due to computational resource constraints, our experiments are conducted on benchmark tasks under a fixed evolution horizon. Examining longer-term and more open-ended adaptation settings would provide a broader view of the resulting co-evolution dynamics.

## References

Salaheddin Alzubi, Noah Provenzano, Jaydon Bingham, Weiyuan Chen, and Tu Vu. 2026. Evoskill: Automated skill discovery for multi-agent systems. Preprint, arXiv:2603.02766.

Hao Chen, Ziyu Han, Yukun Yan, Qingfu Zhu, Maosong Sun, and Wanxiang Che. 2026a. From holistic evaluation to structured criteria: Rubrics across the evolving llm landscape. Preprint, arXiv:2606.08625.

Zhengyu Chen, Teng Xiao, Huaisheng Zhu, Yige Yuan, Luan Zhang, and Jingang Wang. 2026b. Co-harness: Co-evolving harnesses and model weights for llm agents. Preprint, arXiv:2607.22688.

Zijian Chen, Xueguang Ma, Shengyao Zhuang, Ping Nie, Kai Zou, Andrew Liu, Joshua Green, Kshama Patel, Ruoxi Meng, Mingyi Su, Sahel Sharifymoghaddam, Yanxi Li, Haoran Hong, Xinyu Shi, Xuye Liu, Nandan Thakur, Crystina Zhang, Luyu Gao, Wenhu Chen, and Jimmy Lin. 2025. Browsecomp-plus: A more fair and transparent evaluation benchmark of deep-research agent. Preprint, arXiv:2508.06600.

Hongxin Ding, Baixiang Huang, Yue Fang, Weibin Liao, Zheng Li, Jinyang Zhang, Zhijing Wu, Junfeng Zhao, and Yasha Wang. 2026. Evorubrics: Dynamic rubrics as rewards via adversarial co-evolution for llm reinforcement learning. Preprint, arXiv:2606.23038.

Shihan Dou, Haoxiang Jia, Shichun Liu, Feng Chen, Chenhao Huang, Yujiong Shen, Shaofan Liu, Jiayi Chen, Jiahang Lin, Honglin Guo, Qianyu He, Minghao Guo, Ziyi Ye, Pluto Zhou, Tao Gui, Qi Zhang, and Xuanjing Huang. 2026. Agents in the large: Perception-centered architecture for persistent agents. Preprint, arXiv:2608.30478.

Tianyu Fan and Chao Huang. 2026. Helix: Modelharness co-evolution for recursive self-improvement. Preprint, arXiv:2608.13951.

Bofei Gao, Feifan Song, Zhe Yang, Zefan Cai, Yibo Miao, Qingxiu Dong, Lei Li, Chenghao Ma, Liang

Chen, Runxin Xu, Zhengyang Tang, Benyou Wang, Daoguang Zan, Shanghaoran Quan, Ge Zhang, Lei Sha, Yichang Zhang, Xuancheng Ren, Tianyu Liu, and Baobao Chang. 2024. Omni-math: A universal olympiad level mathematic benchmark for large language models. Preprint, arXiv:2410.07985.

Xingze Gao, Chuanrui Hu, Hongda Chen, Pengfei Yao, Zhao Wang, Yi Bai, Zhengwei Wu, Yunyun Han, Xiaofeng Cong, Jie Gui, Yafeng Deng, and Teng Li. 2026. Evoagentbench: Benchmarking agent self-evolution via ability transfer. Preprint, arXiv:2607.05202.

Xin Guan, Xiaomeng Hu, Shen Huang, Zhenyi Wang, Bo Zhang, Zijian Li, Pengjun Xie, Bo Liu, and Jiuxin Cao. 2026. Evorubric: Self-evolving rubricdriven rl for open-ended generation. Preprint, arXiv:2605.29847.

Anisha Gunjal, Anthony Wang, Elaine Lau, Vaskar Nath, Yunzhong He, Bing Liu, and Sean Hendryx. 2025. Rubrics as rewards: Reinforcement learning beyond verifiable domains. Preprint, arXiv:2507.17746.

Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, and Ion Stoica. 2024. Livecodebench: Holistic and contamination free evaluation of large language models for code. Preprint, arXiv:2403.07974.

Haoxiang Jiang, Zihan Dong, Tianci Liu, Wanying Wang, Ran Xu, Tony Yu, Linjun Zhang, and Haoyu Wang. 2026a. Rubric-arrow: Alternating pointwise rubric reward modeling for llm post-training in nonverifiable domains. Preprint, arXiv:2605.29156.

Wen Jiang, Mingmin Chu, Yimeng Tian, Qianxin Zhang, Haofei Yang, Rui Yang, Yang Liu, Tao Lv, and Fangming Li. 2026b. Harnessevolve: Learning from reference trajectories for reliable agent self-evolution. Preprint, arXiv:2609.00829.

Carlos E. Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik Narasimhan. 2024. Swe-bench: Can language models resolve real-world github issues? Preprint, arXiv:2310.06770.

Sunghwan Kim, Dongjin Kang, Taeyoon Kwon, Hyungjoo Chae, Dongha Lee, and Jinyoung Yeo. 2025. Rethinking reward model evaluation through the lens of reward overoptimization. Preprint, arXiv:2505.12763.

Hyunin Lee, Jinglue Xu, Jeffrey Seely, Donghyun Lee, Matei Zaharia, and Yujin Tang. 2026a. Recursive harness self-improvement. Preprint, arXiv:2607.15524.

Yoonho Lee, Roshen Nair, Qizheng Zhang, Kangwook Lee, Omar Khattab, and Chelsea Finn. 2026b. Metaharness: End-to-end optimization of model harnesses. Preprint, arXiv:2603.28052.

Gaotang Li, Bhavana Dalvi Mishra, Zifeng Wang, Jun Yan, Yanfei Chen, Chun-Liang Li, Long T. Le, Rujun Han, George Lee, Hanghang Tong, Chen-Yu Lee, and Tomas Pfister. 2026a. Rubricem: Meta-rl with rubric-guided policy decomposition beyond verifiable rewards. Preprint, arXiv:2605.10899.

Shuyue Stella Li, Rui Xin, Teng Xiao, Yike Wang, Rulin Shao, Zoey Hao, Melanie Sclar, Sewoong Oh, Faeze Brahman, Pang Wei Koh, and Yulia Tsvetkov. 2026b. Evolm: Self-evolving language models through co-evolved discriminative rubrics. Preprint, arXiv:2605.03871.

Xiaoxi Li, Wenxiang Jiao, Jiarui Jin, Guanting Dong, Jiajie Jin, Yinuo Wang, Hao Wang, Yutao Zhu, Ji-Rong Wen, Yuan Lu, and 1 others. 2026c. Deepagent: A general reasoning agent with scalable toolsets. In Proceedings of the ACM Web Conference 2026, pages 2219–2230.

Zhicong Li, Lingjie Jiang, Yulan Hu, Xingchen Zeng, Yixia Li, Xiangwen Zhang, Guanhua Chen, Zheng Pan, Xin Li, and Yong Liu. 2026d. No more stale feedback: Co-evolving critics for open-world agent learning. Preprint, arXiv:2601.06794.

Jiahang Lin, Shichun Liu, Chengjun Pan, Lizhi Lin, Shihan Dou, Zhiheng Xi, Xuanjing Huang, Hang Yan, Zhenhua Han, Tao Gui, and Yu-Gang Jiang. 2026a. Agentic harness engineering: Observabilitydriven automatic evolution of coding-agent harnesses. Preprint, arXiv:2604.25850.

Minhua Lin, Juncheng Wu, Zijun Wang, Zhan Shi, Yisi Sang, Bing He, Zewen Liu, Tianxin Wei, Zongyu Wu, Zhiwei Zhang, Dakuo Wang, Xiang Zhang, Benoit Dumoulin, Cihang Xie, Yuyin Zhou, Suhang Wang, and Hanqing Lu. 2026b. Harness updating is not harness benefit: Disentangling evolution capabilities in self-evolving llm agents. Preprint, arXiv:2605.30621.

Xiaotian Luo, Dizhan Xue, Fengxingyu Wang, Chuanrui Hu, and Yafeng Deng. 2026. Harnessbank: Semantic gene-bank search with gated verification for agent-harness self-evolution. Preprint, arXiv:2607.13683.

Jun Nie, Yonggang Zhang, Jun Song, Qianshu Cai, Dahai Yu, Yike Guo, Xinmei Tian, and Bo Han. 2026. Tthe: Test-time harness evolution. Preprint, arXiv:2607.08124.

Wenbo Pan, Shujie Liu, Chin-Yew Lin, Jingying Zeng, Xianfeng Tang, Xiangyang Zhou, Yan Lu, and Xiaohua Jia. 2026. Evolving agents in the dark: Retrospective harness optimization via self-preference. Preprint, arXiv:2606.05922.

Tejal Patwardhan, Rachel Dias, Elizabeth Proehl, Grace Kim, Michele Wang, Olivia Watkins, Simón Posada Fishman, Marwan Aljubeh, Phoebe Thacker, Laurance Fauconnet, Natalie S. Kim, Patrick Chao, Samuel Miserendino, Gildas Chabot, David Li, Michael Sharman, Alexandra Barr, Amelia Glaese,

and Jerry Tworek. 2025. Gdpval: Evaluating ai model performance on real-world economically valuable tasks. Preprint, arXiv:2510.04374.

Noah Shinn, Federico Cassano, Edward Berman, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. 2023. Reflexion: Language agents with verbal reinforcement learning. Preprint, arXiv:2303.11366.

Dhruv Srikanth, Bingchen Zhao, Dixing Xu, Yuxiang Wu, and Zhengyao Jiang. 2026. Recursive self-improvement of ai research agents. Preprint, arXiv:2609.26457.

Bo Tang, Yang Zhang, Guomian Zhuang, Wenqiang Wei, Gaoyang Zheng, Lindong Xie, Yanchao Tan, Feiyu Xiong, Qingyu Yang, Edward Chung, and Zhiyu li. 2026. From memory to skills: Evidencegrounded co-evolution governance for long-horizon llm agents. Preprint, arXiv:2607.16621.

5 Team, Aohan Zeng, Xin Lv, Qinkai Zheng, Zhenyu Hou, Bin Chen, Chengxing Xie, Cunxiang Wang, Da Yin, Hao Zeng, Jiajie Zhang, Kedong Wang, Lucen Zhong, Mingdao Liu, Rui Lu, Shulin Cao, Xiaohan Zhang, Xuancheng Huang, Yao Wei, and 152 others. 2025. Glm-4.5: Agentic, reasoning, and coding (arc) foundation models. Preprint, arXiv:2508.06471.

Qwen Team. 2026. Qwen3.5: Towards native multimodal agents.

Zihang Tian, Jingsen Zhang, Rui Li, Xiaohe Bo, Yuanzi Li, and Xu Chen. 2026. Arco: Adaptive rubrics with co-evolution for multi-step llm-based agents. Preprint, arXiv:2606.21262.

Beining Wang, Weihang Su, Hongtao Tian, Hao Kong, Tao Yang, Ting Yao, Qingyi Pan, Yueyue Wu, Qingyao Ai, Min Zhang, and Yiqun Liu. 2026a. Co-evolving llm evaluators and policies via dynamicrubric. Preprint, arXiv:2607.20083.

Jianze Wang, Kunwang Zheng, Ying Liu, Yu Cao, Qilong Zhang, Jinlong Chen, Hua Yang, and Qianglong Chen. 2026b. Serpo: Self-evolving rubric policy optimization for open-ended test-time reinforcement learning. Preprint, arXiv:2607.26873.

Yike Wang, Huaisheng Zhu, Zhengyu Hu, Yige Yuan, Zhengyu Chen, Shakti Senthil, Hannaneh Hajishirzi, Yulia Tsvetkov, Noah A. Smith, Pradeep Dasigi, and Teng Xiao. 2026c. Rethinking the evaluation of harness evolution for agents. Preprint, arXiv:2607.12227.

Peng Xia, Rujun Han, Zifeng Wang, Yanfei Chen, Yufan Zhuang, Yoonho Lee, Chengsong Huang, Han Yu, Zhongying CuiZhu, Yifei Ming, Huaxiu Yao, Burak Gokturk, Tomas Pfister, and Chen-Yu Lee. 2026. Rrsi: Regularized recursive self-improvement of agent harnesses. Preprint, arXiv:2609.24972.

Yifan Yang, Ziyang Gong, Weiquan Huang, Qihao Yang, Ziwei Zhou, Zisu Huang, Yan Li, Xuemei Gao, Qi Dai, Bei Liu, Kai Qiu, Yuqing Yang, Dongdong Chen, Xue Yang, and Chong Luo. 2026. Skillopt: Executive strategy for self-evolving agent skills. Preprint, arXiv:2605.23904.

Zhou Yu, Bin Bi, Shiva Kumar Pentyala, Shubham Mehrotra, Sougata Chaudhuri, Shilpa Bhagavath, Zeyuan Chen, Ran Xu, Phil Mui, James Zhu, and Sitaram Asur. 2026. Co-evolving harnesses and models: On-policy correction helps weaker models catch up where imitation fails. Preprint, arXiv:2609.09134.

Hangfan Zhang, Shao Zhang, Kangcong Li, Chen Zhang, Yang Chen, Yiqun Zhang, Lei Bai, and Shuyue Hu. 2026a. Self-harness: Harnesses that improve themselves. Preprint, arXiv:2606.09498.

Qizheng Zhang, Changran Hu, Shubhangi Upasani, Boyuan Ma, Fenglu Hong, Vamsidhar Kamanuru, Jay Rainton, Chen Wu, Mengmeng Ji, Hanchen Li, Urmish Thakker, James Zou, and Kunle Olukotun. 2026b. Agentic context engineering: Evolving contexts for self-improving language models. Preprint, arXiv:2510.04618.

Wenlong Zhang, Zhengbo Jiao, Chenxu Zhang, Lekang Jiang, SiYuan Ma, Qituan Zhang, Guo Chen, and Linfeng Zhang. 2026c. Recursive harness selfimprovement for frontier reasoning data synthesis. Preprint, arXiv:2610.03548.

Xuanliang Zhang, Dingzirui Wang, Keyan Xu, Qingfu Zhu, and Wanxiang Che. 2026d. Scaling laws for agent harnesses via effective feedback compute. Preprint, arXiv:2605.29682.

Wanjun Zhong, Lianghong Guo, Qiqi Gao, He Ye, and Yanlin Wang. 2023. Memorybank: Enhancing large language models with long-term memory. Preprint, arXiv:2305.10250.

Jiayin Zhu, Kelong Mao, Yudong Guo, Dengbo He, Sulong Xu, Simiu Gu, and Yutao Yue. 2026. Skillcoach: Self-evolving rubrics for evaluating and enhancing agentic skill-use. Preprint, arXiv:2607.01874.

## A Appendix

## A.1 Experimental Details

In this subsection, we provide the dataset splits, harness abstraction and editable interfaces, and prompt templates used throughout the experiments.

Dataset Statistics and Settings. We conduct experiments on five agent tasks derived from EvoAgentBench (Gao et al., 2026), covering information retrieval, knowledge work, software engineering, code implementation, and mathematical reasoning. EvoAgentBench reorganizes existing benchmarks into train and test sets to evaluate whether improvements acquired during evolution transfer to unseen tasks. For our experiments, we further split the original training set into evolution and validation subsets, while preserving the original task distribution. The evolution subset is used to drive rubric– harness evolution, whereas the validation subset is used exclusively for candidate selection. We also remove a small number of tasks that cannot be reliably executed in our experimental environment. The resulting dataset statistics are summarized in Table 3.

We report performance on the evolution split as held-in performance. The original test split is never accessed during evolution or candidate selection and is reserved exclusively for held-out evaluation. All splits remain fixed throughout the experiments and are kept identical across all baseline methods to ensure a fair comparison. Evolution rounds, candidate budgets, and validation-based selection protocols are matched across harness-evolution methods unless otherwise specified.

Harness Abstraction and Editable Interfaces. The editable harness space is organized according to the six functional coordinates introduced in Section 3.3. To instantiate this abstraction in the agent framework, each coordinate is aligned with the concrete framework interfaces that control the corresponding functionality, such as skill and tool configuration, instruction prompts, action logic, observation processing, memory handling, and termination behavior. This alignment determines where each proposed harness modification is applied in the underlying agent system, as summarized in Table 4. All methods start from the same initial harness $H _ { 0 }$ for each benchmark and backbone model. The benchmark environment, task specification, tool APIs, and evaluation protocol remain fixed throughout evolution.

Prompt Templates. We provide the key prompt templates used in the co-evolution procedure. The trajectory-analysis prompt is shown in Figure 9, the rubric proposal and outcome-blind judging prompts in Figures 10–11, and the harness intervention planning prompt in Figure 12.

<table><tr><td>Domain</td><td>Evol. (Held-in)</td><td>Val.</td><td>Test (Held-out)</td></tr><tr><td>Information Retrieval</td><td>100</td><td>54</td><td>65</td></tr><tr><td>Knowledge Work</td><td>60</td><td>36</td><td>50</td></tr><tr><td>Software Engineering</td><td>45</td><td>26</td><td>36</td></tr><tr><td>Code Implementation</td><td>128</td><td>54</td><td>86</td></tr><tr><td>Math Reasoning</td><td>380</td><td>98</td><td>100</td></tr></table>

Table 3: Dataset statistics for the evolution, validation, and test splits across five evaluation domains.
<table><tr><td>Coordinate</td><td>Functional Scope</td><td>Editable Content</td></tr><tr><td>Equip</td><td>Capability provisioning</td><td>Skills, tools, and reusable modules</td></tr><tr><td>Instruct</td><td>Behavioral guidance</td><td>System and task instructions</td></tr><tr><td>Act</td><td>Action execution</td><td>Tool-use and action strategies</td></tr><tr><td>Observe</td><td>Observation processing</td><td>Parsing, filtering, and summarization</td></tr><tr><td>Remember</td><td>Context management</td><td>Memory and context handling</td></tr><tr><td>Terminate</td><td>Completion control</td><td>Stopping and finalization</td></tr></table>

Table 4: Unified harness coordinates, their functional scopes, and editable contents.

## A.2 Details of Diagnostic Utility

In this subsection, we provide the detailed definitions of the five criterion-level statistics used to compute the diagnostic utility in Section 3.2.

For each criterion c, the outcome-blind judge assigns every trajectory one of three labels: VIO-LATED, SATISFIED, or N/A. Let $v _ { \mathrm { p a s s } } , s _ { \mathrm { p a s s } } .$ and $n _ { \mathrm { p a s s } }$ denote the numbers of violated, satisfied, and non-applicable trajectories among successful tasks. Similarly, let $v _ { \mathrm { f a i l } } , s _ { \mathrm { f a i l } }$ , and $n _ { \mathrm { f a i l } }$ denote the corresponding counts among unsuccessful tasks. Based on these statistics, the five components are defined as follows.

Discrimination (D): Measures how well a criterion distinguishes unsuccessful trajectories from successful ones by comparing its violation rates across the two groups:

$$
D ( c ) = \frac { v _ { \mathrm { f a i l } } } { v _ { \mathrm { f a i l } } + s _ { \mathrm { f a i l } } } - \frac { v _ { \mathrm { p a s s } } } { v _ { \mathrm { p a s s } } + s _ { \mathrm { p a s s } } } .\tag{11}
$$

Coverage (C): Measures the proportion of unsuccessful trajectories for which the criterion iden-

tifies a violation:

$$
C ( c ) = \frac { v _ { \mathrm { f a i l } } } { v _ { \mathrm { f a i l } } + s _ { \mathrm { f a i l } } + n _ { \mathrm { f a i l } } } .\tag{12}
$$

Predictive Validity (P): Measures whether observing a criterion violation increases the likelihood of task failure relative to the criterion’s applicable population:

$$
P ( c ) = \frac { v _ { \mathrm { f a i l } } } { v _ { \mathrm { f a i l } } + v _ { \mathrm { p a s s } } } { - \frac { v _ { \mathrm { f a i l } } + s _ { \mathrm { f a i l } } } { v _ { \mathrm { p a s s } } + s _ { \mathrm { p a s s } } + v _ { \mathrm { f a i l } } + s _ { \mathrm { f a i l } } } } .\tag{13}
$$

Equivalently,

$$
P ( c ) = { \mathrm { P r } } ( { \mathrm { f a i l ~ | \ v i o l a t e d } } ) { \mathrm { - P r ( f a i l ~ | \ a p p l i c a b l e ) . } }\tag{14}
$$

Novelty (N): Measures the proportion of failed trajectories uniquely identified by criterion $c .$ Let $\mathcal { U } _ { c }$ denote the set of failed trajectories violated only by criterion $^ { c , }$ and let $N _ { F }$ denote the total number of failed trajectories:

$$
N ( c ) = \frac { | \mathcal { U } _ { c } | } { N _ { F } } .\tag{15}
$$

Redundancy (Red): Measures the behavioral overlap between criterion c and the remaining criteria. Let $\mathcal { V } ( c )$ denote the set of trajectories violated by criterion c:

$$
\operatorname { R e d } ( c ) = \operatorname* { m a x } _ { c ^ { \prime } \neq c } \operatorname { J a c c a r d } \left( \mathcal { V } ( c ) , \mathcal { V } ( c ^ { \prime } ) \right) .\tag{16}
$$

A.3 Detail of Rubric Admission and Lifecycle In this subsection, we provide the detailed admission criteria and lifecycle transition rules used to maintain the evolving rubric pool introduced in Section 3.2.

Birth Check. Before entering the maintained rubric pool, each newly discovered criterion undergoes a birth check designed to remove criteria that are unlikely to provide useful diagnostic signals. We reject a criterion under any of the following conditions: (i) never applies, when more than 80% of its assessments are N/A; (ii) neverfires, when it produces no violations on unsuccessful trajectories; or (iii)fires on all, when its violation rate exceeds 90% overall and also exceeds 90% on successful trajectories. These checks remove criteria that are respectively inapplicable, insensitive to actual failures, or insufficiently discriminative. Criteria that pass the birth check enter the lifecycle in the probation state. Since lifecycle transitions are evaluated within the same evolution round, a newly admitted criterion may already appear as active in the round-end visualization if it immediately satisfies the promotion condition.

<table><tr><td>Current State</td><td>Condition</td><td>Next State</td></tr><tr><td>Probation</td><td> $P > 0$ </td><td>Active</td></tr><tr><td>Probation</td><td>Untriggered for 4 rounds</td><td>Retired</td></tr><tr><td>Active</td><td>Trigger rate &lt; 5% for 3 rounds</td><td>Dormant</td></tr><tr><td>Active</td><td>N/A drift for 3 rounds</td><td>Dormant</td></tr><tr><td>Active</td><td> $P < 0 . 1 0 \mathrm { f o r } 3 \mathrm { r o u n d s }$ </td><td>Retired</td></tr><tr><td>Dormant</td><td> ${ \mathrm { T r i g g e r ~ r a t e } } \geq 1 5 \%$ </td><td>Active</td></tr><tr><td>Dormant</td><td>Remains dormant for 4 rounds Retired</td><td></td></tr></table>

Table 5: Transition rules for the rubric lifecycle. Trigger rate is the violation frequency of a criterion, and N/A drift reflects a sustained rise in its not-applicable rate.

Lifecycle Transition Mechanism. Each admitted criterion is maintained in one of four states: probation, active, dormant, or retired. State transitions are evaluated after each evolution round according to the criterion’s observed diagnostic statistics.

Probation. A newly admitted criterion first enters the probation state. It is promoted to active once it exhibits positive predictive validity $( P > 0 )$ . If it remains untriggered for four consecutive rounds, it is instead moved directly to retired.

Active. An active criterion is eligible for frontier construction. It becomes dormant when its trigger rate remains below 5% for three consecutive rounds or when the predefined N/A-drift condition persists for three consecutive rounds. It is directly retired when its predictive validity remains below 0.10 for three consecutive rounds.

Dormant. A dormant criterion is retained for monitoring but excluded from frontier construction. It is reactivated when its trigger rate rises to at least 15%. If it remains dormant for four consecutive rounds without reactivation, it is moved to retired.

Retired. The retired state is terminal. Retired criteria no longer participate in criterion prioritization or frontier construction. Table 5 summarizes all lifecycle transition rules.

Visualization of Lifecycle Dynamics. Figure 6 visualizes the lifecycle dynamics on Qwen3.5-35B-A3B for the Knowledge Work domain. Each column shows criterion states at the end of an evolution round, while the flows trace transitions between states. “Not yet born” denotes criteria introduced only in later rounds. For visualization, the retired category also includes criteria rejected at birth as bookkeeping, although such criteria do not formally enter the lifecycle.

The figure reveals substantial turnover throughout evolution. New criteria continue to enter the lifecycle and become active as they satisfy the promotion condition, while previously active criteria move to dormant or retired states as their diagnostic relevance decreases. Dormant criteria may also return to the active set when the corresponding failures re-emerge. Consequently, the active criterion set is repeatedly reorganized rather than monotonically accumulated.

![](images/d7ef02f6fa93605f54f106a802446d802558d8d04dcab4804e68eca43f468a03.jpg)  
Figure 6: Rubric lifecycle across evolution rounds on the Qwen3.5-35B-A3B model with the knowledge work task. Each column shows criterion states at the end of a round, with flow widths proportional to the number of criteria. “Not yet born” denotes criteria introduced in later rounds; “Retired” is cumulative and includes candidates rejected at birth.

This turnover provides a mechanistic explanation for the frontier dynamics observed in the main analysis. As harness updates alter the behavioral distribution, some criteria lose diagnostic relevance while others become newly informative, continually reshaping the active set available for frontier construction.

## A.4 Details of Posterior Estimation and Conservative Ranking

In this subsection, we provide the details of the conservative ranking strategy introduced in Section 3.2. Directly ranking criteria by their point-estimated utility can be unreliable when only a small number of trajectories support a criterion, since a few observations may produce an artificially high score. We therefore rank criteria using a lower posterior estimate of their diagnostic utility.

For each criterion, we construct a $2 \times 3$ contingency table over task outcome (pass or fail) and criterion assessment (VIOLATED, SATISFIED, or N/A). Since discrimination, coverage, and predictive validity are all computed from this same table, we estimate them jointly. We place a symmetric Jeffreys prior over the six cells, corresponding to adding 0.5 to each observed count, and obtain a Dirichlet posterior over the joint outcome distribution.

We draw B = 2000 samples from this posterior. For each sample, we recompute discrimination, coverage, and predictive validity, while keeping novelty and redundancy fixed because they are computed from trajectory-level set statistics. This produces a posterior distribution over the diagnostic utility. We then use its lower q-quantile as the conservative score:

$$
\widetilde { \cal U } ( c ) = \mathrm { Q u a n t i l e } _ { q } \left( \left\{ U ^ { ( b ) } ( c ) \right\} _ { b = 1 } ^ { B } \right) ,\tag{17}
$$

where $q = 0 . 2$ in all experiments.

This lower-quantile ranking favors criteria whose diagnostic value remains high under uncertainty, while reducing the influence of criteria supported by only a few observations. The resulting scores are used to rank active criteria, from which the Top-K criteria are selected to form the rubric frontier.

## A.5 Robustness to Random Seeds

To assess the robustness of different evolution methods to stochasticity, we repeat all methods with three independent random seeds on two representative settings: Qwen3.5-35B-A3B on Knowledge Work and GLM-4.7-Flash on Software Engineering. Dataset splits, initial harnesses, and all hyperparameters are kept fixed across runs. Table 6 reports the mean and standard deviation over the three runs.

<table><tr><td>Method Held-in</td><td>Held-out</td></tr><tr><td colspan="2">Qwen3.5-35B-A3B on Knowledge Work</td></tr><tr><td>Vanilla Agent</td><td> $5 1 . 1 \pm 1 . 0$   $3 8 . 0 \pm 2 . 0$ </td></tr><tr><td>ACE</td><td> $5 5 . 6 \pm 1 . 0$   $4 2 . 0 \pm 2 . 0$ </td></tr><tr><td>EvoSkill</td><td> $5 4 . 4 \pm 1 . 0$   $4 1 . 3 \pm 1 . 2$ </td></tr><tr><td>Self-Harness</td><td> $5 2 . 2 \pm 1 . 9$   $4 1 . 3 \pm 3 . 1$ </td></tr><tr><td>AHE</td><td> $5 5 . 0 \pm 1 . 7$   $4 7 . 3 \pm 3 . 1$ </td></tr><tr><td>Meta-Harness</td><td> $5 7 . 8 \pm 2 . 5$   $3 8 . 0 \pm 2 . 0$ </td></tr><tr><td>RuHarness</td><td> ${ \bf 5 8 . 9 \pm 1 . 0 }$   ${ \bf 5 0 . 0 \pm 2 . 0 }$ </td></tr></table>

<table><tr><td colspan="2">GLM-4.7-Flash on Software Engineering</td></tr><tr><td>Vanilla Agent</td><td> $3 1 . 1 \pm 2 . 2$   $3 1 . 5 \pm 3 . 2$ </td></tr><tr><td>ACE  $3 7 . 8 \pm 0 . 0$ </td><td> $4 5 . 3 \pm 1 . 6$ </td></tr><tr><td>EvoSkill  $4 2 . 2 \pm 2 . 2$ </td><td> $3 9 . 8 \pm { 1 . 6 }$ </td></tr><tr><td>Self-Harness</td><td> $4 5 . 9 \pm 1 . 3$   $4 5 . 4 \pm 3 . 2$ </td></tr><tr><td>AHE</td><td> $4 2 . 9 \pm 1 . 3$   $3 9 . 8 \pm { 1 . 6 }$ </td></tr><tr><td>Meta-Harness</td><td> $4 0 . 0 \pm 0 . 0$   $3 8 . 0 \pm 1 . 6$ </td></tr><tr><td>RuHarness</td><td> ${ \bf 4 8 . 9 \pm 2 . 2 }$   ${ \bf 5 1 . 9 \pm 1 . 6 }$ </td></tr></table>

Table 6: Robustness to random seeds on two representative settings. Results are reported as mean ± standard deviation over three independent runs.

RuHarness maintains the best average performance across both held-in and held-out evaluations under three independent random seeds. Importantly, the gains remain consistent across different random seeds, while the observed variance is comparable to that of competing methods. This trend is particularly clear on held-out tasks, where RuHarness maintains a clear advantage under both backbone–task combinations. These results indicate that the observed gains are robust to stochasticity in the evolution process.

## A.6 Effectiveness of Dynamic Evolution

In this subsection, we further analyze how dynamic rubric evolution affects generalization during harness optimization. We compare the full RuHarness framework with a static-rubric variant in which the rubric pool is fixed after initialization, while the remaining evolution procedure is kept unchanged. Figure 7 reports the round-wise generalization gap and frontier coverage on the Qwen3.5-35B-A3B Knowledge Work setting.

Dynamic rubrics maintain a smaller generalization gap. We define the generalization gap as held-out performance minus held-in performance at each evolution round. As shown in Figure 7a, although both variants can improve held-in performance during evolution, the static-rubric variant exhibits substantially larger fluctuations and increasingly negative gaps in later rounds. In contrast, the dynamic-rubric variant remains relatively stable across evolution and ends with a markedly smaller discrepancy between held-in and held-out performance. This suggests that harness optimization guided by an evolving rubric transfers more effectively to unseen tasks.

![](images/a3fe76fe5af5316df4b56c47d6b3aca4702ce1933de3a39c4a45b2315b739893.jpg)  
(a) Generalization gap across evolution rounds.

![](images/fafecc80ae9848d65f600f563bfce1099d47339df03d5456564cd32799cbd36b.jpg)  
(b) Frontier coverage across evolution rounds.  
Figure 7: Effectiveness of dynamic rubric evolution on the Qwen3.5-35B-A3B model with the knowledge work task. Compared with a static rubric, the dynamic rubric maintains a smaller held-out–held-in generalization gap and substantially higher frontier coverage over the course of evolution.

Dynamic rubrics preserve coverage of current failures. Figure 7b further shows a clear divergence in frontier coverage, defined as the proportion of failed trajectories captured by at least one criterion in the currently prioritized rubric frontier. As evolution proceeds, the dynamic rubric maintains high coverage across most rounds, whereas the static rubric progressively loses coverage, particularly in later stages. This indicates that a fixed rubric becomes increasingly misaligned with the weaknesses exposed by the evolving harness, while continual rubric updates allow the frontier to track newly emerging failure patterns.

Taken together, these results provide additional evidence for the role of rubric evolution in continual harness optimization. A static rubric can still support improvements on the evolution tasks, but its diagnostic relevance gradually weakens as agent behavior changes. By continually updating and reprioritizing behavioral criteria, RuHarness preserves coverage of unresolved failures and consequently achieves more stable generalization throughout evolution.

## A.7 Effectiveness of Rubric Frontier

To understand why harness evolution benefits from a prioritized rubric frontier rather than the entire rubric pool, we compare the diagnostic composition of the frontier with that of all eligible rubrics. Figure 8 reports the distribution of core diagnos-

![](images/1d4c041951d1ab04d57ff11c06fa005d98e10a00ec782bd0360180515090821f.jpg)  
(a) Distribution of core diagnostic value.

![](images/af11bb9f229ba3574ee53ea47c13ac69cc5f5180336f0390df900231d616e7fc.jpg)  
(b) Rate of non-positive diagnostic value.

Figure 8: Effectiveness of rubric frontier prioritization. Compared with the complete eligible rubric pool, the frontier concentrates criteria with substantially higher core diagnostic value and contains far fewer criteria with non-positive diagnostic value.

tic value and the proportion of criteria with nonpositive diagnostic value. Here, the core diagnostic value is defined as the conservative posterior estimate:

$$
V ( c ) = Q _ { 0 . 2 } [ D ( c ) + 0 . 5 C ( c ) + P ( c ) ] ,\tag{18}
$$

which captures the principal diagnostic signals used for rubric prioritization.

The frontier concentrates diagnostically valuable criteria. As shown in Figure 8a, criteria in the frontier exhibit a substantially higher diagnosticvalue distribution than the complete rubric pool. The median value increases from 0.215 for all eligible rubrics to 0.442 for the frontier. Since the full pool contains the frontier itself, this shift indicates that prioritization substantially increases the concentration of informative criteria presented to harness evolution, rather than simply reducing the number of criteria.

Using all rubrics introduces substantial lowvalue guidance. Figure 8b further shows that only 0.5% of frontier criteria have non-positive diagnostic value, compared with 29.9% when considering the complete rubric pool. Thus, providing all rubrics to the proposer introduces a considerable fraction of weak diagnostic signals alongside the informative ones. This helps explain the degradation observed when frontier selection is removed: the frontier serves as a focused diagnostic filter that preserves high-value guidance while avoiding dilution by criteria that provide limited information about the agent’s current weaknesses.

## A.8 Case Study of Rubric–Harness Co-Evolution

To examine how rubric evolution progressively reshapes harness optimization, we trace two representative evolution trajectories in Table 7. Rather than repeatedly targeting the same failure pattern, the diagnostic focus becomes increasingly specific across rounds, and the corresponding harness updates evolve from coarse checks to more operational, explicit, and structured interventions.

Knowledge Work. The first trajectory exhibits a clear progression in verification requirements. It begins with a basic question of whether generated scripts are actually executed, leading to an OB-SERVE-level rule that requires execution and error checking. The next stage moves beyond detecting failure to asking why thefailure occurred, prompting explicit analysis of returned error messages. The focus then shifts from execution correctness to deliverable correctness: the agent must verify that the expected output artifact exists and matches the required filename. This requirement is subsequently strengthened from a verbal confirmation into programmatic verification, making the check directly executable. In the final stage, verification is moved earlier in the workflow, from post-hoc confirmation to pre-access checking before downstream file use. The trajectory therefore progresses from execution, to diagnosis, to artifact verification, to executable verification, and finally to proactive verification.

Information Retrieval. The second trajectory shows a parallel progression from coarse evidence checking to structured auditing. It starts by requiring full-document retrieval instead of relying on search snippets, establishing a stronger evidence basis for candidate matching. The next refinement moves this requirement closer to the retrieval step itself, preventing premature judgments immediately after observing a snippet. The focus then advances from merely verifying a candidate to making the verification explicit and auditable, requiring the agent to state which constraints are satisfied and provide supporting evidence. Finally, the process becomes systematic: individual evidence statements are organized into a constraint-satisfaction matrix so that all constraints are checked collectively rather than independently. Thus, the diagnostic trajectory evolves from source verification, to immediate verification, to explicit evidence grounding, and finally to structured constraint coverage.

These trajectories illustrate that rubric evolution progressively raises the granularity of behavioral supervision. As earlier requirements are incorporated into the harness, later rubrics shift attention to finer-grained weaknesses, which in turn induce more precise and structured modifications across the relevant harness coordinates. This creates a staged refinement process in which diagnosis and intervention become progressively more operational as evolution proceeds.

<table><tr><td>Round</td><td>Observed Pattern</td><td>Rubric Diagnosis</td><td>Target Coordinate</td><td>Harness Response</td></tr><tr><td colspan="5">Case I: Knowledge Work — From Generic Checking to Executable Verification</td></tr><tr><td>R1</td><td>Execution</td><td>Generated scripts should not be treated as valid merely because they were written; successful execution must be verified.</td><td>OBSERVE</td><td>Require every generated script to be executed and explicitly checked for errors before subsequent actions. Require the agent to inspect the error</td></tr><tr><td>R3</td><td>Error Analysis</td><td>Detecting a tool failure is insufficient if the underlying cause is not examined</td><td>OBSERVE</td><td>output and identify its cause before proceeding with unrelated commands.</td></tr><tr><td>R4</td><td>Deliverable Check</td><td>Successful execution does not guarantee that the expected deliverable has actually been produced.</td><td>+EQUIP</td><td>Add a completion-stage check that confirms the output file exists and matches the required filename.</td></tr><tr><td>R5</td><td>Programmatic Check</td><td>Declarative confirmation remains underspecified for reliable verification.</td><td>TERMINATE</td><td>Strengthen the check by requiring programmatic verification of the output filename, e.g., through os.path.basename.</td></tr><tr><td>R8</td><td>Pre-access Check</td><td>File validity should be established before the file is consumed downstream.</td><td>REMEMBER</td><td>Move verification earlier by requiring file existence to be checked before reading reference files.</td></tr><tr><td colspan="5">Case II: Information Retrieval — From Candidate Retrieval to Auditable Verification Search snippets provide only partial</td></tr><tr><td>R1</td><td>Full-text Check</td><td>evidence and should not be sufficient to establish a candidate match.</td><td>REMEMBER +TERMINATE</td><td>Require the full document to be retrieved before claiming that a candidate satisfies the target constraints.</td></tr><tr><td>R5</td><td>Early Verification</td><td>The agent may form a premature match judgment immediately after seeing a search result.</td><td>OBSERVE</td><td>Explicitly mark search results as snippets and require full-text confirmation before accepting a match.</td></tr><tr><td>R7</td><td>Explicit Evidence</td><td>Verification remains difficult to audit when the response does not expose how individual constraints are satisfied.</td><td>OBSERVE +EQUIP +TERMINATE</td><td>Require the agent to state satisfied constraints and attach supporting evidence for each claim.</td></tr><tr><td>R9</td><td></td><td>Structured Audit Individual evidence statements may still leave overall constraint coverage unclear.</td><td>TERMINATE +REMEMBER</td><td>Require a structured constraint-satisfaction matrix and continue retrieval until all constraints are verified.</td></tr></table>

Table 7: Case studies of rubric–harness co-evolution with Qwen3.5-35B-A3B. Across evolution rounds, diagnostic focus progressively shifts from coarse behavioral checks to more specific and operational requirements, which are translated into targeted modifications of the corresponding harness coordinates.

Failure Trajectory Analysis   
You are analyzing ONE agent trajectory to produce a factual failure report. Another system will   
read many of these reports together and design evaluation criteria from them. Your job is to   
record WHAT HAPPENED, precisely and verifiably -- not to judge the agent, and not to state   
principles.   
VERIFIER RESULT (authoritative):   
outcome: {outcome}   
exception: {exception\_type}   
failed tests: {failed\_tests}   
verifier output (tail): {verifier\_tail}   
steps: {n\_steps}   
wall clock: {wall\_clock\_min} min   
repeated-action ratio: {repetition\_ratio}   
TRAJECTORY:   
{trajectory\_text}   
Produce ONE JSON object:   
{   
"goal": "<what the task required, one sentence>",   
"final\_state": "<where the run actually ended, observable facts only>",   
"divergence\_step": <int>,   
"divergence\_evidence": "<VERBATIM excerpt from that step: the actual action, command, and/or   
error text. Copy it; do not paraphrase.>",   
"failure\_mode": "<exactly one of: {failure\_modes}>",   
"agent\_attributable": <true|false>,   
"attribution\_note": "<one sentence>",   
"missed\_opportunity": "<the specific alternative action available AT divergence\_step -- name the   
action, tool, file, or operation the agent could have used instead. If none, write 'none'.>"   
}   
Rules:   
- Every quote must appear verbatim above. Do not invent output.   
- divergence\_step is where progress STOPPED, usually much earlier than where the run ended.   
- agent\_attributable is false ONLY when no available action could have changed the result --   
infrastructure broke, the agent harness malfunctioned, or the model API failed.   
- An environment obstacle that the agent responded to badly is still attributable (true).   
- missed\_opportunity must be concrete and executable.   
- "Should have been more careful", "should have verified the result", and "should have planned   
better" are rejected -- if that is all you can say, write "none".   
Output ONLY the JSON object.  
Figure 9: Prompt template for failure trajectory analysis.

Rubric Proposal   
You are designing evaluation criteria for an agent rubric.   
A criterion is one sentence describing an OBSERVABLE agent behavior. A judge will later read a   
trajectory and answer satisfied / violated / not\_applicable against it. A criterion is only useful   
if that judge would reach DIFFERENT answers on failing runs and succeeding runs.   
EXISTING RUBRIC POOL -- do not restate these:   
{pool\_summary}   
FAILURE REPORTS ({n\_fail} runs):   
{failure\_reports}   
SUCCESS REPORTS ({n\_pass} runs):   
{success\_reports}   
BLIND SPOTS -- no existing criterion fired on these failures. Prioritize them:   
{uncovered\_list}

![](images/8536ce0096221e9729c165b24484205ec2a2f016c0ab4d4deb27804980987e1e.jpg)  
Figure 10: Prompt template for rubric proposal.

Outcome-Blind Rubric Judge   
You are evaluating whether the following criterion holds for an agent trajectory.   
CRITERION:   
{nl\_description}   
TRAJECTORY (task {task\_id}):   
{trajectory\_text}   
Judge the behavior described by the criterion on its own terms. Do not try to infer whether the   
run ultimately succeeded, and do not let a guess about that influence the verdict.   
Verdicts:   
"satisfied" - the trajectory shows the criterion being met   
"violated" - the trajectory shows the criterion being broken   
"not\_applicable" - the situation the criterion describes never arose   
Output ONLY a single JSON object with these fields:   
{   
"verdict": "satisfied" | "violated" | "not\_applicable",   
"reason": "<one-sentence justification>",   
"step\_index": <0-based int of the offending step, or null if not violated>,   
"quoted\_action": "<VERBATIM excerpt from that step, copied not paraphrased, or null if not   
violated>"   
}   
Do not include any other text - no markdown, no commentary. Just the JSON.  
Figure 11: Prompt template for outcome-blind rubric judging.

![](images/c2bb71a88e2ad8392ca1fa2427149ff9d4e3c29385a776f557cb134c41024abd.jpg)

another way for the bundle to fail to load and another variable that cannot be attributed   
afterwards.   
- state what observable behavior should change if the hypothesis is right, so a reader can tell   
later whether it was.   
Plans must differ from each other in evidence, mechanism, target criteria, or component set. If   
you cannot construct {n\_candidates} genuinely distinct plans, return fewer. A duplicate plan costs   
a full evaluation sweep and returns noise; an empty slot costs nothing.   
Note on self-created obstacles: a common failure is a change that makes the agent's own job harder   
- for example, injecting control text into a query that only an external retrieval system sees,   
or adding a rule that blocks the very calls the agent needs. Before proposing an edit, ask what   
the agent loses by it.   
Output ONLY a single JSON object:   
{   
"plans": [   
{   
"hypothesis": "<falsifiable claim about the failing mechanism>",   
"target\_criterion\_ids": ["<criterion id from the frontier>", ...],   
"evidence": [   
{   
"task\_id": "<task id exactly as shown above>",   
"step\_indices": [<int>, ...],   
"explanation": "<what these steps show>"   
}   
],   
"expected\_behavior": "<what should observably change>",   
"component\_roles": [   
{   
"component": "<component name>",   
"role": "<what this component does in the plan>"   
}   
],   
"risks": "<how this plan could backfire>"   
}   
]   
}   
Do not include any other text - no markdown, no commentary. Just the JSON.  
Figure 12: Prompt template for harness intervention planning.