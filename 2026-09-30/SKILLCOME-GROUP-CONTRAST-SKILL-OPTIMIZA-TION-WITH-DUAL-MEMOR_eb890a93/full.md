# SKILLCOME: GROUP CONTRAST SKILL OPTIMIZA-TION WITH DUAL MEMORY

Haolin Li<sup>♠,♣</sup> , Feng Hong<sup>♢</sup>, Ang Li<sup>♣</sup>, Chilin Fu<sup>♣</sup>, Weichang Wu<sup>♣</sup>, Ya Zhang<sup>♢</sup>, Yanfeng Wang<sup>♢</sup>, Xiaolu Zhang<sup>♣,†</sup>, Jiangchao Yao<sup>♢†</sup> <sup>♠</sup>Fudan University <sup>♢</sup>Shanghai Jiao Tong University <sup>♣</sup>Ant Group

## ABSTRACT

Skill evolution improves the capabilities of large language models by analyzing trajectories generated under a given skill and modifying the skill accordingly. Existing approaches typically generate a single trajectory per question. However, this provides insufficient optimization signals since it requires inferring effective skill edits from a solitary path. It is difficult to pinpoint which actions caused the failure in a failed trajectory, or to determine which actions in a successful one should be incorporated into the skill. Furthermore, they rely on a local batch of trajectories for analysis, making the optimization direction susceptible to noisy evidence. To address these, we propose SkillCome, a Skill-evolution method based on group Contrast optimization with dual memory. For each question, SkillCome generates n trajectories and performs group contrast analysis to precisely identify key behavioral divergences between successful and failed trajectories, offering reliable optimization signals. The dual memory system further accumulates evidence from historical steps to track patterns shared across different groups, leading to more generalized optimization directions. Together, SkillCome builds a systematic optimization process that transforms experience from observed successful trajectories into reusable skills. Extensive experiments on six benchmarks spanning question answering, reasoning, and agentic tasks demonstrate the effectiveness of our method. SkillCome consistently outperforms baselines across five models of varying families and scales, with gains up to +5.69% points.

![](images/b99b57e0082e21604a49bdd7c3ba280738e4917b48425eca49cba5944c8785ad.jpg)

## 1 INTRODUCTION

Large language models (LLMs) can be improved through parametric optimization of model weights or non-parametric optimization of external context (Ouyang et al., 2022; Wang et al., 2023a). Skills (Xu & Yan, 2026; Li et al., 2026), typically derived from human experience or LLM-generated trajectories (Jiang et al., 2026; Bai et al., 2026), provide an effective form of non-parametric optimization. Recent works on skill evolution further optimize skills iteratively (Wang et al., 2025; Lin et al., 2026). Specifically, a target model executes tasks under the current skill, and an optimizer model analyzes the resulting trajectories to propose skill revisions, which are then evaluated on validation sets (Zhang et al., 2026; Zhu et al., 2026).

Despite its effectiveness, skill evolution requires more than an iterative loop of generating and evaluating edits. A sound optimization process requires two key components: a reliable optimization signal that indicates how the current solution should be updated, and an optimizer to organize the signals and guide update directions. Parametric optimization has developed sophisticated mechanisms for both training objectives (Wei et al., 2022a; Rafailov et al., 2023) and optimizer states (Kingma & Ba, 2015; Loshchilov & Hutter, 2019). This motivates us to examine both the construction of optimization signals and the design of optimizer state for skill evolution.

At the optimization signal level, the outcome of a trajectory does not directly reveal how the skill should change. Existing approaches typically analyze one trajectory per question (Alzubi et al., 2026; Wu et al., 2026), leaving it unclear which actions caused failure or contributed to success. At the optimizer level, current methods primarily observe trajectories from a local batch, which can provide noisy or incomplete evidence. A valuable skill pattern can appear in trajectories scattered across different steps. Consequently, a single batch lacks sufficient data to substantiate this recurring pattern, leaving the optimizer to overlook it for inclusion in the skill.

These limitations call for an optimization process with reliable learning signals and a stateful optimizer. In parametric optimization, group relative rewards among responses to the same question have been widely adopted as optimization signals (Shao et al., 2024). For skill evolution, comparing successful and failed trajectories within a group similarly reveals behavioral divergences associated with opposite outcomes. The optimizer can then derive reliable optimization signals from observed successful attempts, rather than inferring a potential solution solely from a failed trajectory. Figure 1 illustrates how same-question contrast reveals a row-deletion error in worksheet-operation tasks and yields a practical skill. We further examine this idea by performing a single post-hoc group contrast skill revision using existing SkillOpt trajectories for the same question. As shown on the right of Figure 1, this already improves performance without additional training rollouts. For such evidence to guide optimization direction, a memory system is necessary to preserve historical observations. Analogous to momentum in gradient descent (Sutskever et al., 2013), the memory can retain patterns learned from previous steps. The optimizer can then combine current findings with accumulated information, continually calibrating its update direction as further evidence becomes available. Such a stateful optimization process helps turn scattered evidence into generalizable skills, preventing noise in individual steps from dominating the optimization process.

In this work, we propose SkillCome, a skill-evolution method that provides more reliable optimization signals while guiding optimization directions with historical evidence. Our method reshapes trajectory generation by introducing grouped rollouts, collecting multiple trajectories for each question. We incorporate within-group contrast between successful and failed trajectories to conventional reflection. These comparisons help identify key behavioral differences and precisely transform trajectory evidence into targeted skill revisions, improving optimization reliability and data efficiency. We further introduce a dual-memory system that connects current observations with historical information. Contrast memory preserves contrast patterns learned between successful and failed trajectories within the same group, while failure memory retains failure patterns shared by different groups. By continuously interacting with the dual memory, the optimizer can discover shared patterns across questions and use them to refine the skill update directions. The two synergistic components establish a comprehensive optimization process for skill evolution. Experiments across diverse model-benchmark pairs demonstrate that SkillCome can discover effective skills that are difficult to learn for single-rollout methods, leading to improved skill evolution and performance. In summary, our contributions are threefold:

• We introduce group rollouts and within-group contrast optimization under the same skill. This provides reliable optimization signals by pinpointing the key divergences between successful and failed trajectories, grounding skill revisions in observed experience.

• To calibrate the direction of skill optimization, we propose a dual-memory system storing historical evidence. The system recognizes recurring patterns from different steps and builds support for generalizable skills, preventing noisy updates from individual batches.

• Extensive experiments on six benchmarks validate the effectiveness of our method across five models with varying scales. SkillCome enables effective skill evolution and delivers consistent improvements, with average gains up to 19.55% compared to no skill.

## 2 RELATED WORK

From Skills to Skill Evolution. Prompt-based methods (Wei et al., 2022b; Yao et al., 2023; Pryzant et al., 2023) have demonstrated that LLMs can be improved by simply refining the textual instructions (Wang et al., 2023b; Liu et al., 2023). Beyond general prompts, agent skills further include task-specific experience, including tool-use strategies and relevant references (Zheng et al., 2025; Shi et al., 2026). They organize experience derived from human expertise or agent trajectories into natural language, optionally accompanied by executable scripts and supporting resources (Shen et al., 2026a). By making such experience editable, skills can also incorporate new lessons from execution feedback (Yang et al., 2026b). This has motivated various research on skill evolution, where agent trajectories are analyzed to iteratively revise skills and improve subsequent task performance (Liu et al., 2026a; Ouyang et al., 2026). They explicitly optimize skill artifacts through repeated generation and revision (Liu et al., 2026a; Ouyang et al., 2026). Trace2Skill (Ni et al., 2026) extracts local lessons from execution traces in parallel and consolidates them into transferable procedural guidance. SkillOpt (Yang et al., 2026a) develops a structured optimization procedure that converts trajectories into bounded skill edits, with edit aggregation and validation-based acceptance. SkillOpt-Lite (Shen et al., 2026b) and OEO (Liu et al., 2026b) delegate parts of the evolution process to an agent rather than a human-implemented pipeline, allowing the agent to automatically organize the evolution. These studies establish generated trajectories as a basis for skill evolution, with different designs for extracting experience and learning edits. While our method focuses on how to better construct the optimization process based on grouped rollouts.

## 3 METHOD

In this section, we first formulate the skill-evolution problem in § 3.1. We then describe grouped rollouts in § 3.2. § 3.3 presents our group contrast skill optimization procedure. The overall framework of SkillCome is illustrated in Figure 2.

## 3.1 PROBLEM FORMULATION

In skill evolution, an external skill document is optimized to improve the performance of LLM agents on given tasks. A skill s consists of a natural-language document together with any accompanying executable scripts. The document is incorporated into the agent context during execution.

We distinguish two model roles: the target model $M _ { \mathrm { t a r } }$ executes tasks under the current skill, while the optimizer model $M _ { \mathrm { o p t } }$ analyzes the resulting trajectories and proposes skill revisions. The two roles can be the same or different LLMs, whose parameters remain frozen throughout skill evolution. Given an input task x and a skill document $s ,$ a trajectory is sampled as $\tau \sim \mathsf { \bar { M } } _ { \mathrm { t a r } } ( \cdot \ | \ x , s )$ . The trajectory τ records model outputs, tool interactions when applicable, and the final answer. A taskspecific evaluator assigns a binary score $R ( x , \tau ) \in \{ 0 , 1 \}$ } according to the task’s success criterion. Given a dataset D, the objective is to find a skill that maximizes expected performance:

$$
\operatorname* { m a x } _ { s } \sum _ { x \in \mathcal { D } } \mathbb { E } _ { \tau \sim M _ { \mathrm { t a r } } ( \cdot | x , s ) } \left[ R ( x , \tau ) \right] .\tag{1}
$$

In practice, we use three disjoint data splits: $\mathcal { D } _ { \mathrm { t r a i n } }$ for collecting training trajectories, $\mathcal { D } _ { \mathrm { v a l } }$ for evaluating the updated skills, and $\mathcal { D } _ { \mathrm { t e s t } }$ for testing the final skill. Starting from an initial skill $s ^ { 0 }$ , the target model executes training tasks under the current skill $s ^ { t }$ at optimization step t. $M _ { \mathrm { o p t } }$ analyzes

![](images/8f34754442b075a28bc96450595b8f138e960cdf6e662d36a8cdc042f801577c.jpg)  
Figure 2: Overview of SkillCome. We first formulate the training as grouped rollouts. The generated trajectories are routed to either “Success Reflection” or “Failure Reflection” based on their outcome correctness. Groups with mixed outcomes undergo “Contrast Reflection”. During reflection, the optimizer model reads and updates failure memory and contrast memory to find generalizable optimization directions. The updated skill is accepted if it improves the validation score.

these trajectories and their evaluation feedback, then proposes and selects edits that add, remove, or replace content from $s ^ { t } .$ . Applying the selected edits yields a new candidate skill $s _ { \mathrm { c a n d } } ^ { t }$

The candidate is evaluated on the validation split using the same target model and execution environment. We denote the measured validation score of a skill s by:

$$
V ( s ) = \frac { 1 } { | \mathcal { D } _ { \mathrm { v a l } } | } \sum _ { x \in \mathcal { D } _ { \mathrm { v a l } } } R ( x , \tau _ { x } ( s ) ) ,\tag{2}
$$

where $\tau _ { x } ( s )$ denotes the validation trajectory generated for task x using skill s. The candidate skill $s _ { \mathrm { c a n d } } ^ { t }$ is accepted only if its score exceeds the recorded validation score of the previous skill:

$$
s ^ { t + 1 } = { \left\{ s _ { \mathrm { c a n d } } ^ { t } , \right. }  & { V ( s _ { \mathrm { c a n d } } ^ { t } ) > V ( s ^ { t } ) , }\tag{3}
$$

After iteratively running the training-updating-validation loop, we retain the skill with the highest validation score as $s ^ { \star }$ and evaluate it on $\mathcal { D } _ { \mathrm { t e s t } }$

## 3.2 GROUPED ROLLOUT

To obtain comparable trajectory evidence for each question, we organize the training process in the form of group rollouts. At optimization step t, we sample m distinct training questions $\{ x _ { i } ^ { t } \} _ { i = 1 } ^ { m } \subseteq$ $\mathcal { D } _ { \mathrm { t r a i n } }$ . For each question, the target model $M _ { \mathrm { t a r } }$ independently generates n trajectories under the same current skill $s ^ { t } \colon$

$$
\tau _ { i j } ^ { t } \sim M _ { \mathrm { t a r } } \bigl ( \cdot \mid x _ { i } ^ { t } , s ^ { t } \bigr ) , \qquad i = 1 , \ldots , m , \quad j = 1 , \ldots , n ,\tag{4}
$$

where the subscripts i and $j$ are the question index and its rollout index, respectively. The skill remains fixed throughout the current optimization step. Thus, the trajectories within a group represent alternative executions of the same question. Each trajectory receives a score $r _ { i j } ^ { t } = \mathbf { \bar { \phi } } R ( x _ { i } ^ { t } , \tau _ { i j } ^ { t } )$ . We organize the scored trajectories for each question into a group $\mathcal { G } _ { i } ^ { t }$ and denote the collection of all groups in the batch by $\bar { \boldsymbol { g } } ^ { t }$ :

$$
\mathcal { G } _ { i } ^ { t } = \{ \left( \tau _ { i j } ^ { t } , r _ { i j } ^ { t } \right) \} _ { j = 1 } ^ { n } , \qquad \mathcal { G } ^ { t } = \{ \mathcal { G } _ { i } ^ { t } \} _ { i = 1 } ^ { m } .\tag{5}
$$

The batch contains m groups and $B = m n$ rollouts in total.

For each group, let $\mathcal { G } _ { i } ^ { t , + }$ and $\mathcal { G } _ { i } ^ { t , - }$ denote the subsets of successful trajectories with $r _ { i j } ^ { t } = 1$ and failed trajectories with $r _ { i j } ^ { t } = 0$ , respectively. Across the batch, we collect these trajectories into success and failure pools:

$$
{ \mathcal { T } } ^ { t , + } = \bigcup _ { i = 1 } ^ { m } { \mathcal { G } } _ { i } ^ { t , + } , \quad \quad { \mathcal { T } } ^ { t , - } = \bigcup _ { i = 1 } ^ { m } { \mathcal { G } } _ { i } ^ { t , - } .\tag{6}
$$

These pools provide evidence for success reflection and failure reflection, respectively. And we collect groups containing both successful and failed trajectories into the mixed-group subset:

$$
\mathcal { G } _ { \mathrm { m i x } } ^ { t } = \left\{ \mathcal { G } _ { i } ^ { t } \in \mathcal { G } ^ { t } \ \middle | \ \mathcal { G } _ { i } ^ { t , + } \neq \varnothing \ \mathrm { a n d } \ \mathcal { G } _ { i } ^ { t , - } \neq \varnothing \right\} .\tag{7}
$$

Each mixed group is passed to the subsequent group contrast reflection, where successful trajectories serve as concrete references for examining failures and guiding skill edits.

## 3.3 GROUP CONTRAST SKILL OPTIMIZATION

## 3.3.1 DUAL MEMORY SYSTEM

Since the optimizer $M _ { \mathrm { o p t } }$ does not have an infinite context length, each optimization step can analyze only a limited set of trajectories. Skill edits derived from one step may overfit to a particular question, while evidence supporting a generalizable pattern may be distributed across multiple steps. To connect these observations, we maintain a failure memory $\dot { \mathcal { M } } _ { \mathrm { f } }$ and a contrast memory $M _ { \mathrm { c } } ,$ both initialized as empty. $\mathcal { M } _ { \mathrm { f } }$ preserves failure patterns extracted through conventional reflection. $\mathcal { M } _ { \mathrm { c } }$ contains contrastive patterns learned from the behavioral differences between successful and failed trajectories within the same group. We keep $\mathcal { M } _ { \mathrm { f } }$ and $\mathcal { M } _ { \mathfrak { c } }$ <sub>c</sub> separate to distinguish failure-only patterns from contrast patterns that provide more reliable optimization signals grounded in observed successful alternatives.

Rather than storing lengthy historical trajectories, each memory consists of concise patterns:

$$
\boldsymbol { \mathcal { M } } = \left\{ ( p _ { k } , \boldsymbol { u } _ { k } ) \right\} _ { k = 1 } ^ { K } ,\tag{8}
$$

where K is the number of learned patterns, $p _ { k }$ is the natural-language description of the k-th pattern, and $u _ { k }$ denotes the set of distinct training groups supporting it. The size of $u _ { k }$ indicates how many different questions support the pattern, rather than how many trajectories. Patterns supported by more questions provide stronger evidence for broadly applicable skill edits.

During trajectory analysis, $M _ { \mathrm { o p t } }$ interacts with the corresponding memory. It compares the current trajectories with the memory and proposes memory updates alongside its skill edits. An update may introduce a new pattern, refine an existing description, or associate additional supporting groups with an existing pattern without changing its description. For the latter two cases, the support set is updated as $u _ { k } \gets u _ { k } \cup u _ { \mathrm { n e w } } .$ , where $u _ { \mathrm { n e w } }$ contains the newly associated group identifiers. We denote the proposed memory updates by $\Delta \mathcal { M }$ , which may be empty when no update is needed. This read-and-update process allows a pattern initially observed in one step to accumulate evidence from different groups across optimization steps, without requiring additional training rollouts.

## 3.3.2 CONVENTIONAL FAILURE AND SUCCESS REFLECTION

At optimization step $t ,$ the trajectory pools $\mathcal { T } ^ { t , - }$ and $\mathcal { T } ^ { t , + }$ obtained from $\ S 3 . 2$ are analyzed separately by $M _ { \mathrm { o p t } }$ to propose skill edits. Failure reflection examines what went wrong and how the skill could prevent similar failures, whereas success reflection extracts useful practices worth retaining. Taking failure reflection as an example, the optimizer uses the current skill, failed trajectories, and failure memory to generate candidate skill edits and optional memory updates:

$$
\left( \mathcal { P } _ { \mathrm { f } } ^ { t } , \Delta \mathcal { M } _ { \mathrm { f } } ^ { t } \right) \gets M _ { \mathrm { o p t } } \left( s ^ { t } , T ^ { t , - } , \mathcal { M } _ { \mathrm { f } } \right) ,\tag{9}
$$

where $\mathcal { P } _ { \mathrm { f } } ^ { t }$ contains the proposed skill edits. Historical failure patterns from $\mathcal { M } _ { \mathrm { f } }$ help $M _ { \mathrm { o p t } }$ relate current mistakes to problems in previous steps and formulate edits supported by different questions. Success reflection is performed through separate optimizer calls and similarly produces candidate edits $\mathcal { P } _ { \mathrm { s } } ^ { t }$ . Note that trajectories from the mixed groups still contribute to conventional reflection, while they are additionally examined through group contrast reflection described next.

## 3.3.3 GROUP CONTRAST REFLECTION

To accurately determine what to correct and what to retain in the skill, we introduce the group contrast reflection. It compares successful and failed trajectories within the same group to track critical divergences that account for their opposite outcomes. For each mixed group $\mathcal { G } _ { i } ^ { t } \in \mathcal { G } _ { \operatorname* { m i x } } ^ { t } ,$ a separate optimizer call generates candidate edits and optional updates to the contrast memory:

$$
\left( \mathcal { P } _ { \mathrm { c } , i } ^ { t } , \Delta \mathcal { M } _ { \mathrm { c } , i } ^ { t } \right) \gets M _ { \mathrm { o p t } } \left( s ^ { t } , \mathcal { G } _ { i } ^ { t , + } , \mathcal { G } _ { i } ^ { t , - } , \mathcal { M } _ { \mathrm { c } } \right) .\tag{10}
$$

Here, $M _ { \mathrm { o p t } }$ is instructed to first locate critical divergences between the successful and failed trajectories. It then examines which actions in the successful trajectories were omitted or executed incorrectly in the failures, and relates these differences to missing or insufficiently actionable guidance in $s ^ { t } .$ . Successful trajectories thus provide concrete references that the target model has actually done, offering reliable optimization signals for skill editing.

The contrast memory $\mathcal { M } _ { \mathrm { c } }$ further connects these same-question comparisons with patterns observed in previous groups. A new comparison can then strengthen or refine an existing pattern instead of being interpreted as an isolated case. This guides skill optimization toward generalizable directions supported by multiple questions, rather than fitting a single case. And the proposed memory updates are integrated after each analysis, making them visible to subsequent steps.

## 3.3.4 EVIDENCE-GUIDED SKILL REVISION

We collect the contrast edits as $\begin{array} { r } { \mathcal { P } _ { \mathrm { c } } ^ { t } = \bigcup _ { \mathcal { G } _ { i } ^ { t } \in \mathcal { G } _ { \operatorname* { m i x } } ^ { t } } \mathcal { P } _ { \mathrm { c } , i } ^ { t } } \end{array}$ and combine them with the failure and success edits. Through dedicated merge calls of the optimizer $M _ { \mathrm { o p t } }$ , we consolidate overlapping edits and resolves conflicting suggestions:

$$
\mathcal { P } ^ { t } = \operatorname { M e r g e } _ { M _ { \mathrm { o p t } } } \left( \mathcal { P } _ { \mathrm { f } } ^ { t } \cup \mathcal { P } _ { \mathrm { s } } ^ { t } \cup \mathcal { P } _ { \mathrm { c } } ^ { t } \right) .\tag{11}
$$

When the number of edits in $\mathcal { P } ^ { t }$ exceeds the given edit budget $L ^ { t } , M _ { \mathrm { o p t } }$ will select at most $L ^ { t }$ edits. Ranking prioritizes contrast edits and failure edits over success edits. It also accounts for the number of supporting groups to find more generalizable skill edits.

Based on the above procedure, the optimizer integrates complementary evidence: Conventional reflection captures shared difficulties and potentially useful practices; group contrast reflection provides practical guidance grounded in grouped rollouts. In addition, the dual memory system further extends the supporting evidence across steps. Applying the selected final edits $\Delta ^ { t } \subseteq { \mathcal { P } } ^ { t }$ produces the candidate skill:

$$
s _ { \mathrm { c a n d } } ^ { t } = \mathrm { A p p l y } ( s ^ { t } , \Delta ^ { t } ) , \qquad | \Delta ^ { t } | \leq L ^ { t } .\tag{12}
$$

The candidate skill $s _ { \mathrm { c a n d } } ^ { t }$ is evaluated and accepted according to Eq. 3, after which the target model $M _ { \mathrm { t a r } }$ collects the next batch of grouped rollouts under the resulting skill. Memory updates are retained even when a candidate is rejected. Since an unsuccessful edit does not invalidate the underlying observations, which may be supported by different groups of rollouts in future steps. Across iterations, group contrast deepens the analysis of individual questions and provides accurate optimization signals. While the dual-memory system plays a momentum-like role to regulate optimization direction by retaining the patterns observed beyond the current batch. For final evaluation, we test the target model on $\mathcal { D } _ { \mathrm { t e s t } }$ using the selected best skill $s ^ { \star }$

## 4 EXPERIMENTS

## 4.1 EXPERIMENTAL SETUP

Datasets. The experiments are conducted on six benchmarks. General question answering tasks include DocVQA (Mathew et al., 2021) and SearchQA (Dunn et al., 2017). Mathematical reasoning task is LiveMathematicianBench (LiveMath; He et al. (2026)). Agentic tasks include Spreadsheet-Bench (Ma et al., 2024), OfficeQA (Opsahl-Ong et al., 2026), and ALFWorld (Shridhar et al., 2021). We adopt the data splits from SkillOpt-Lite (Shen et al., 2026b).

Table 1: Performance across six benchmarks and five models. “–” denotes unsupported VQA evaluation for text-only LLMs. Subscripts in green/red represent the change relative to “No skill”. The best and second-best scores within each model are in bold and underlined, respectively.
<table><tr><td>Model</td><td>Skill</td><td>Spreadsheet</td><td>SearchQA</td><td>LiveMath</td><td>ALFWorld</td><td>OfficeQA</td><td>DocVQA</td><td>Avg</td></tr><tr><td rowspan="6">Qwen3.8 Flash-Next</td><td>No skill</td><td>48.58</td><td>67.50</td><td>33.73</td><td>81.53</td><td>23.48</td><td>90.37</td><td>57.53</td></tr><tr><td>Init skill</td><td> $4 5 . 5 5 _ { - 3 . 0 3 }$ </td><td> $6 7 . 9 7 _ { + 0 . 4 7 }$ </td><td> $3 5 . 1 4 _ { + 1 . 4 1 }$ </td><td> $8 2 . 4 7 _ { + 0 . 9 4 }$ </td><td> $3 0 . 7 5 _ { + 7 . 2 7 }$ </td><td> $9 1 . 3 8 _ { + 1 . 0 1 }$ </td><td> $5 8 . 8 8 _ { + 1 . 3 5 }$ </td></tr><tr><td>Trace2Skill</td><td> $6 9 . 6 6 _ { + 2 1 . 0 8 }$ </td><td> $6 9 . 9 3 _ { + 2 . 4 3 }$ </td><td> $3 1 . 6 0 _ { - 2 . 1 3 }$ </td><td> $8 5 . 0 7 _ { + 3 . 5 4 }$ </td><td> $4 9 . 8 3 _ { + 2 6 . 3 5 }$ </td><td> $9 1 . 3 1 _ { + 0 . 9 4 }$ </td><td> $6 6 . 2 3 _ { + 8 . 7 0 }$ </td></tr><tr><td>SkillOpt-Lite</td><td> $\underline { { 7 3 . 7 5 } } _ { + 2 5 . 1 7 }$ </td><td> $7 2 . 4 3 _ { + 4 . 9 3 }$ </td><td> $4 1 . 7 5 _ { + 8 . 0 2 }$ </td><td> $\underline { { 8 6 . 1 9 } } _ { + 4 . 6 6 }$ </td><td> $\underline { { 6 0 . 3 0 } } _ { + 3 6 . 8 2 }$ </td><td> $\underline { { 9 1 . 7 1 } } _ { + 1 . 3 4 }$ </td><td> $7 1 . 0 2 _ { + 1 3 . 4 9 }$ </td></tr><tr><td>SkillOpt</td><td> $7 2 . 5 1 _ { + 2 3 . 9 3 }$ </td><td> $\underline { { 7 2 . 5 7 } } _ { + 5 . 0 7 }$ </td><td> $\underline { { 4 9 . 5 3 } } _ { + 1 5 . 8 0 }$ </td><td> $8 4 . 7 0 _ { + 3 . 1 7 }$ </td><td> $5 9 . 8 0 _ { + 3 6 . 3 2 }$ </td><td> $9 1 . 2 4 _ { + 0 . 8 7 }$ </td><td> $\underline { { 7 1 . 7 3 } } _ { + 1 4 . 2 0 }$ </td></tr><tr><td>SkillCome</td><td> $7 5 . 7 1 _ { + 2 7 . 1 3 }$ </td><td> $\mathbf { 7 4 . 1 4 _ { \div 6 . 6 4 } }$ </td><td> ${ \bf 6 3 . 6 8 _ { \pm 2 9 . 9 5 } }$ </td><td> $\mathbf { 9 1 . 7 9 _ { + 1 0 . 2 6 } }$ </td><td> ${ \bf 6 3 . 6 8 _ { + 4 0 . 2 0 } }$ </td><td> $\mathbf { 9 2 . 7 8 _ { \div 2 . 4 1 } }$ </td><td> $7 6 . 9 6 _ { \pm 1 9 . 4 3 }$ </td></tr><tr><td rowspan="7"> $\mathrm { D e e p S e e k - V 4 }$  Pro-0813</td><td>No skill</td><td>47.78</td><td>66.57</td><td>28.07</td><td>66.98</td><td>53.71</td><td>一</td><td>52.62</td></tr><tr><td>Init skill</td><td> $4 8 . 4 0 _ { + 0 . 6 2 }$ </td><td> $6 7 . 0 7 _ { + 0 . 5 0 }$ </td><td> $3 1 . 3 7 _ { + 3 . 3 0 }$ </td><td> $7 9 . 1 0 _ { + 1 2 . 1 2 }$ </td><td> $5 4 . 5 6 _ { + 0 . 8 5 }$ </td><td></td><td> $5 6 . 1 0 _ { + 3 . 4 8 }$ </td></tr><tr><td>Trace2Skill</td><td> $5 4 . 9 8 _ { + 7 . 2 0 }$ </td><td> $6 9 . 2 1 _ { + 2 . 6 4 }$ </td><td> $3 3 . 2 5 _ { + 5 . 1 8 }$ </td><td> $8 0 . 0 4 _ { + 1 3 . 0 6 }$ </td><td> $4 6 . 4 5 _ { - 7 . 2 6 }$ </td><td></td><td> $5 6 . 7 9 _ { + 4 . 1 6 }$ </td></tr><tr><td>SkillOpt-Lite</td><td> $5 9 . 0 8 _ { + 1 1 . 3 0 }$ </td><td> $6 9 . 2 9 _ { + 2 . 7 2 }$ </td><td> $3 2 . 0 8 _ { + 4 . 0 1 }$ </td><td> $8 0 . 6 0 _ { + 1 3 . 6 2 }$ </td><td> $5 4 . 7 3 _ { + 1 . 0 2 }$ </td><td></td><td> $5 9 . 1 6 _ { + 6 . 5 3 }$ </td></tr><tr><td>SkillOpt SkillCome</td><td> $5 7 . 3 8 _ { + 9 . 6 0 }$ </td><td> $7 0 . 8 6 _ { + 4 . 2 9 }$ </td><td> $4 8 . 3 5 _ { + 2 0 . 2 8 }$ </td><td> $\underline { { 8 4 . 3 2 } } _ { + 1 7 . 3 4 }$ </td><td> $5 6 . 7 6 _ { + 3 . 0 5 }$ </td><td>一</td><td> $6 3 . 5 3 _ { + 1 0 . 9 1 }$ </td></tr><tr><td></td><td> ${ \bf 6 2 . 1 0 _ { + 1 4 . 3 2 } }$ </td><td> $7 2 . 5 0 _ { + 5 . 9 3 }$ </td><td> $\mathbf { 5 3 . 3 0 _ { + 2 5 . 2 3 } }$ </td><td> $\mathbf { 8 9 . 1 8 _ { \perp 2 2 . 2 0 } }$ </td><td> ${ \bf 5 9 . 2 9 } _ { + 5 . 5 8 }$ </td><td>一</td><td> ${ \bf 6 7 . 2 7 _ { + 1 4 . 6 5 } }$ </td></tr><tr><td>No skill</td><td>45.73</td><td>67.93</td><td>36.08</td><td>86.94</td><td>20.78</td><td>一</td><td>51.49</td></tr><tr><td rowspan="7">GLM-5.3</td><td>Init skill</td><td> $5 2 . 4 4 _ { + 6 . 7 1 }$ </td><td> $6 8 . 5 7 _ { + 0 . 6 4 }$ </td><td> $3 7 . 0 3 _ { + 0 . 9 5 }$ </td><td> $9 1 . 2 3 _ { + 4 . 2 9 }$ </td><td> $2 7 . 8 4 _ { + 7 . 0 6 }$ </td><td></td><td> $5 5 . 4 2 _ { + 3 . 9 3 }$ </td></tr><tr><td>Trace2Skill</td><td> $4 3 . 1 5 _ { - 2 . 5 8 }$ </td><td> $7 0 . 5 0 _ { \scriptstyle + 2 . 5 7 }$ </td><td> $3 6 . 7 9 _ { + 0 . 7 1 }$ </td><td> $8 8 . 8 1 _ { + 1 . 8 7 }$ </td><td> $2 7 . 0 3 _ { + 6 . 2 5 }$ </td><td></td><td> $5 3 . 2 6 _ { + 1 . 7 6 }$ </td></tr><tr><td>SkillOpt-Lite</td><td> $\underline { { 6 8 . 2 3 } } _ { + 2 2 . 5 0 }$ </td><td> $7 7 . 4 3 _ { + 9 . 5 0 }$ </td><td> $\underline { { 6 5 . 0 9 } } _ { + 2 9 . 0 1 }$ </td><td> $8 8 . 4 3 _ { + 1 . 4 9 }$ </td><td> $\underline { { 4 6 . 6 2 } } _ { + 2 5 . 8 4 }$ </td><td></td><td> $\underline { { 6 9 . 1 6 } } _ { + 1 7 . 6 7 }$ </td></tr><tr><td>SkillOpt</td><td> $6 6 . 9 0 _ { + 2 1 . 1 7 }$ </td><td> $7 4 . 2 9 _ { + 6 . 3 6 }$ </td><td> $4 5 . 0 5 _ { + 8 . 9 7 }$ </td><td> $9 2 . 3 5 _ { + 5 . 4 1 }$ </td><td> $4 6 . 2 8 _ { + 2 5 . 5 0 }$ </td><td></td><td> $6 4 . 9 7 _ { + 1 3 . 4 8 }$ </td></tr><tr><td>SkillCome</td><td> ${ \bf 7 1 . 5 3 _ { \pm 2 5 . 8 0 } }$ </td><td> $\underline { { 7 7 . 0 7 } } _ { + 9 . 1 4 }$ </td><td> ${ \bf 6 7 . 9 2 _ { + 3 1 . 8 4 } }$ </td><td> $\mathbf { 9 5 . 9 0 _ { + 8 . 9 6 } }$ </td><td> ${ \bf 4 8 . 6 5 } _ { \substack { + 2 7 . 8 7 } }$ </td><td>一</td><td> ${ 7 2 . 2 1 } _ { \pm 2 0 . 7 2 }$ </td></tr><tr><td>No skill</td><td> $4 7 . 3 3$ </td><td> $6 4 . 8 5 $ </td><td> $3 1 . 6 0 $ </td><td>59.51</td><td> $5 . 4 1$ </td><td>一</td><td>41.74</td></tr><tr><td>Init skill</td><td> $4 8 . 5 8 _ { + 1 . 2 5 }$ </td><td> $6 5 . 4 3 _ { + 0 . 5 8 }$ </td><td> $3 5 . 3 8 _ { + 3 . 7 8 }$ </td><td> $5 6 . 9 1 _ { - 2 . 6 0 }$ </td><td> $1 1 . 6 6 _ { + 6 . 2 5 }$ </td><td></td><td> $4 3 . 5 9 _ { + 1 . 8 5 }$ </td></tr><tr><td rowspan="5">Flash</td><td>DeepSeek-V4 Trace2Skill</td><td> $5 1 . 0 7 _ { + 3 . 7 4 }$ </td><td> $7 1 . 5 0 _ { + 6 . 6 5 }$ </td><td> $3 3 . 2 5 _ { + 1 . 6 5 }$ </td><td> $7 9 . 1 0 _ { + 1 9 . 5 9 }$ </td><td> $1 2 . 8 4 _ { + 7 . 4 3 }$ </td><td></td><td> $4 9 . 5 5 _ { + 7 . 8 1 }$ </td></tr><tr><td>SkillOpt-Lite</td><td> $5 7 . 3 8 _ { + 1 0 . 0 5 }$ </td><td> $6 8 . 0 0 _ { + 3 . 1 5 }$ </td><td> $3 3 . 7 3 _ { + 2 . 1 3 }$ </td><td> $6 1 . 0 1 _ { + 1 . 5 0 }$ </td><td> $2 4 . 6 6 _ { + 1 9 . 2 5 }$ </td><td></td><td> $4 8 . 9 6 _ { + 7 . 2 2 }$ </td></tr><tr><td>SkillOpt</td><td> $\underline { { 5 9 . 1 6 } } _ { + 1 1 . 8 3 }$ </td><td> $\underline { { 7 1 . 5 7 } } _ { + 6 . 7 2 }$ </td><td> $\underline { { 6 8 . 6 3 } } _ { + 3 7 . 0 3 }$ </td><td> $7 4 . 4 4 _ { + 1 4 . 9 3 }$ </td><td> $2 6 . 0 1 _ { + 2 0 . 6 0 }$ </td><td>一</td><td> $5 9 . 9 6 _ { + 1 8 . 2 2 }$ </td></tr><tr><td>SkillCome</td><td> $6 2 . 5 4 _ { + 1 5 . 2 1 }$ </td><td> $7 3 . 3 6 _ { + 8 . 5 1 }$ </td><td> $\mathbf { 7 4 . 2 9 _ { + 4 2 . 6 9 } }$ </td><td> $\mathbf { 8 4 . 1 4 _ { \div 2 4 . 6 3 } }$ </td><td> ${ \bar { \mathbf { 5 0 . 5 1 } } } _ { + 4 5 . 1 0 }$ </td><td>一</td><td> ${ \bf 6 8 . 9 7 _ { + 2 7 . 2 3 } }$ </td></tr><tr><td>No skill</td><td>41.99</td><td>66.00</td><td>26.65</td><td>66.04</td><td>37.16</td><td>89.77</td><td>54.60</td></tr><tr><td rowspan="6">Qwen3.6 35B-A3B</td><td>Init skill</td><td> $4 0 . 7 5 _ { - 1 . 2 4 }$ </td><td></td><td></td><td></td><td></td><td> $\underline { { 9 0 . 1 7 } } _ { + 0 . 4 0 }$ </td><td></td></tr><tr><td>Trace2Skill</td><td> $4 4 . 9 3 _ { + 2 . 9 4 }$ </td><td> $6 6 . 2 3 _ { + 0 . 2 3 }$   $\underline { { 6 9 . 2 1 } } _ { + 3 . 2 1 }$ </td><td> $3 0 . 1 9 _ { + 3 . 5 4 }$   $3 1 . 6 0 _ { + 4 . 9 5 }$ </td><td> $7 9 . 4 8 _ { + 1 3 . 4 4 }$   $7 5 . 5 6 _ { + 9 . 5 2 }$ </td><td> $\underline { { 3 8 . 6 8 } } _ { + 1 . 5 2 }$   $3 5 . 6 4 _ { - 1 . 5 2 }$ </td><td> $8 9 . 1 0 _ { - 0 . 6 7 }$ </td><td> $5 7 . 5 8 _ { + 2 . 9 8 }$   $5 7 . 6 7 _ { + 3 . 0 7 }$ </td></tr><tr><td>SkillOpt-Lite</td><td> $5 6 . 9 3 _ { + 1 4 . 9 4 }$ </td><td> $6 7 . 5 7 _ { + 1 . 5 7 }$ </td><td> $2 9 . 0 1 _ { + 2 . 3 6 }$ </td><td> $8 1 . 3 4 _ { + 1 5 . 3 0 }$ </td><td> $3 7 . 8 4 _ { + 0 . 6 8 }$ </td><td> $9 0 . 1 1 _ { + 0 . 3 4 }$ </td><td> $6 0 . 4 7 _ { + 5 . 8 7 }$ </td></tr><tr><td>SkillOpt</td><td> $\underline { { 5 7 . 6 5 } } _ { + 1 5 . 6 6 }$ </td><td> $6 7 . 3 6 _ { + 1 . 3 6 }$ </td><td> $\underline { { 6 9 . 8 1 } } _ { + 4 3 . 1 6 }$ </td><td> $\underline { { 8 3 . 2 1 } } _ { + 1 7 . 1 7 }$ </td><td> $3 4 . 6 3 _ { - 2 . 5 3 }$ </td><td> $8 9 . 7 7 _ { + 0 . 0 0 }$ </td><td> $\underline { { 6 7 . 0 7 } } _ { + 1 2 . 4 7 }$ </td></tr><tr><td> $\mathbf { S k i l l C o m e }$ </td><td> ${ \bf 6 1 . 4 8 _ { + 1 9 . 4 9 } }$ </td><td> $\mathbf { 7 1 . 2 9 _ { + 5 . 2 9 } }$ </td><td> $\mathbf { 7 1 . 4 6 } _ { \pm 4 4 . 8 1 }$ </td><td> $\mathbf { 8 6 . 3 8 _ { \perp 0 . 3 4 } }$ </td><td> ${ \bf 4 0 . 0 3 _ { + 2 . 8 7 } }$ </td><td> $\mathbf { 9 1 . 2 4 _ { \ell + 1 . 4 7 } }$ </td><td> ${ \bf 7 0 . 3 1 _ { + 1 5 . 7 1 } }$ </td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Models. We employ GLM-5.3 (Zeng et al., 2026), Qwen3.8-Flash-Next (Qwen Team, 2026a), DeepSeek-V4-Pro-0813 (DeepSeek-AI, 2026), DeepSeek-V4-Flash, and Qwen3.6-35B-A3B (Qwen Team, 2026b) as the target model. For the first three models, we use themselves as the optimizer for a self-evolving setting. For the less capable two models, DeepSeek-V4-Pro is employed as the optimizer to evaluate a setting where a stronger model guides a weaker model.

## 4.2 MAIN RESULTS

Settings. We compare our method against the following baselines: running without a skill, the initial skill written by human, skill-evolving methods Trace2Skil (Ni et al., 2026), SkillOpt (Yang et al., 2026a), and SkillOpt-Lite (Shen et al., 2026b). And we follow the training protocol of SkillOpt-Lite. The default number of rollouts n is set to 8. Since all test sets other than SearchQA are on the scale of hundreds of samples, resulting in significant testing volatility, we evaluate these benchmarks over four independent runs and report the average accuracy. More details are provided in Appendix B.

As presented in Table 1, SkillCome achieves the highest scores across 26 model–benchmark pairs, outperforming baselines by a large margin. Compared with the second-best method SkillOpt, our method further improves average accuracy by 5.69 points, demonstrating the benefits of learning from grouped trajectories. On both the self-evolving setting and strong-guide-weak setting, Skill-Come exhibits consistent gains across LLMs from 35B to 1.6T, with average gains up to 27.23 over “No skill”. Notably, the OfficeQA performance of DeepSeek-V4-Flash improves from 5.41% to 50.51% with our method, surpassing the second-best method by 24.50%. This is associated with a recurring format error: the model frequently uses <|DSML|answer>...</|DSML|answer> delimiters instead of the required <answer>...</answer> tags, preventing its responses from being recognized as valid answers. SkillCome retains this repeated pattern in memory and incorporates explicit instructions prohibiting DSML-style output into the skill. This provides a concrete example of how cross-step memory supports the correction of persistent model-specific errors. For more detailed analysis, please refer to Appendix C.1.

Table 2: Ablation experiment of method components on two models.
<table><tr><td>Model</td><td>Method</td><td>Spreadsheet</td><td>SearchQA</td><td>LiveMath</td><td>ALFWorld</td><td>OfficeQA</td><td>DocVQA</td><td>Avg</td></tr><tr><td rowspan="4">DeepSeek-V4 Flash</td><td>SkillOpt</td><td>59.16</td><td>71.57</td><td>68.63</td><td>74.44</td><td>26.01</td><td>一</td><td>59.96</td></tr><tr><td>SkillOpt w/ Memory</td><td>59.43</td><td>71.14</td><td>68.86</td><td>77.80</td><td>47.80</td><td>一</td><td>65.01</td></tr><tr><td>SkillCome w/o Memory</td><td>57.92</td><td>71.43</td><td>63.92</td><td>75.37</td><td>37.84</td><td>一</td><td>61.30</td></tr><tr><td>SkillCome</td><td>62.54</td><td>73.36</td><td>74.29</td><td>84.14</td><td>50.51</td><td>一</td><td>68.97</td></tr><tr><td rowspan="4">Qwen3.6 35B-A3B</td><td>SkillOpt</td><td>57.65</td><td>67.36</td><td>69.81</td><td>83.21</td><td>34.63</td><td>89.77</td><td>67.07</td></tr><tr><td>SkillOpt w/ Memory</td><td>58.63</td><td>69.07</td><td>67.92</td><td>81.16</td><td>35.14</td><td>89.91</td><td>66.97</td></tr><tr><td>SkillCome w/o Memory</td><td>58.72</td><td>68.57</td><td>70.99</td><td>83.40</td><td>34.46</td><td>89.84</td><td>67.66</td></tr><tr><td>SkillCome</td><td>61.48</td><td>71.29</td><td>71.46</td><td>86.38</td><td>40.03</td><td>91.24</td><td>70.31</td></tr></table>

![](images/8fb2e538199ed6140d35ed5520dd764b171ee6aeb7f7e3a1cc6aaa5b982d01a3.jpg)  
Figure 3: A case study of the evolution process of SkillCome on ALFWorld, where the agent needs to manipulate objects within a simulated embodied environment. SkillCome discovers a valuable pattern. Although the initial candidate skill is rejected, the extracted pattern remains in contrast memory. Its support group accumulates from 2 to 14 distinct tasks across optimization steps. During the process, the skill is accepted and refined multiple times.

## 4.3 ABLATION STUDY

To investigate the impact of each component of SkillCome, we conduct two ablations. “SkillOpt w/ Memory” augments the strongest baseline, SkillOpt, with the failure memory in our framework. We also evaluate “SkillCome w/o Memory”, which removes dual memory while retaining group contrast analysis. As shown in Table 2, compared with SkillOpt, “SkillOpt w/ Memory” improves performance in 8 of the 11 model–benchmark pairs and reduces in the remaining three. These mixed results suggest that adding memory to conventional single-rollout optimization yields gains, but not consistently. In contrast, removing memory from SkillCome decreases performance in all pairs. Despite this, “SkillCome w/o Memory” still exhibits improvement over Skillopt. The contrasting effects of memory can be attributed to differences in the evidence within each update. Under a comparable budget, single-rollout methods observe more distinct questions per step, potentially limiting the benefit of memory. Grouped rollouts instead concentrate on repeated trajectories of fewer questions, making cross-step evidence particularly useful for calibrating the optimization direction. Together, the ablations show that the two components of SkillCome are effective and synergistic.

## 5 FURTHER ANALYSIS

## 5.1 CASE STUDY

Figure 3 showcases a detailed evolution process of SkillCome. As illustrated, group contrast reveals a useful search pattern: explore unvisited locations beyond the destination rather than repeatedly checking the same area. With the accumulation of supporting groups in the contrast memory, the corresponding skill edit is selected and accepted at step 3. At step 5, a further revision introduces a complementary rule: deliver the held target before searching for the next. The resulting skill includes practical guidance absent from single-rollout baselines and leads to stronger test performance.

![](images/ea82540b607e159cb8056e210be1950a8270346c91e1d8c3fa02be18501fdc29.jpg)  
(a)

![](images/7a943270ec4c7ca011d35de9a270fc201d4051efd9b442ef38647beb05bbe7de.jpg)  
(b)

![](images/c6e5e4b80bec5edb8a6d895e736a54bc7c6d791f7cdc2e6a62b29e66a0d2c480.jpg)  
(c)  
Figure 4: (a) Cross-model skill transferability on DeepSeek-V4-Flash. (b) Cross-dataset skill transferability on four models (Names are abbreviated). Left are the HotpotQA results, while the right are the Omni-MATH results. (c) Evolution from empty skill on DeepSeek-V4-Flash.

## 5.2 SKILL TRANSFERABILITY

Cross-Model Transferability. To evaluate whether the learned skills remain useful beyond the target model, we transfer the skills between DeepSeek-V4-Flash (Flash) and Qwen3.6-35B-A3B (Qwen3.6). For each benchmark, a skill optimized for Qwen3.6 is directly applied to Flash. Figure 4a shows that transferred skills significantly outperform the initial skill in all five benchmarks. However, they still exhibit gaps from self-evolved skills, with the largest differences on OfficeQA. This is consistent with previous discussion in §4.2, since the skill learned on Qwen3.6 does not contain the DSML-style format issue. Overall, these results indicate that our skills contain both broadly reusable procedures and corrections tailored to the behavioral patterns of individual models.

Cross-Dataset Transferability. We further evaluate the cross-dataset transferability of SkillCome on four models. Specifically, we directly transfer skills learned on SearchQA to HotpotQA (Yang et al., 2018) and those learned on OlympiadBench (He et al., 2024) to Omni-MATH (Gao et al., 2024). As shown in Figure 4b, the transferred skills outperform the no-skill baseline on both benchmarks across all four models, demonstrating their generalizability to new datasets with similar task requirements. More results on transferability are in Appendix C.6 and C.7.

## 5.3 EVOLVING FROM EMPTY SKILL

Since initial skills do not always improve performance, we investigate whether evolving from an empty skill provides a better alternative. We compare evolving from no skill with evolving from initial skills on DeepSeek-V4-Flash. As presented in Figure 4c, evolution from an empty skill improves over “No skill” on all five benchmarks, but evolving from the initial skill achieves higher final performance. This shows that SkillCome can learn useful skills from scratch, while predefined guidance generally provides better starting points, even when their immediate benefits are limited.

## 5.4 EVALUATION IN AGENT HARNESSES

To further evaluate our method in agent harnesses, we apply SkillCome on Qwen3.8-Flash-Next and DeepSeek-V4- Pro-0813 within the Codex CLI harness on three datasets. As illustrated in Figure 5, we compare three settings: Initial skill, where the agent operates through the chat completion mode under the initial skill; Direct chat, where the agent operates through the chat completion mode under the skill learned by SkillCome; and Codex, where the agent operates within the Codex CLI harness under the skill learned by SkillCome. For both models, running under Codex consistently outperforms running under direct chat across all datasets. These results demonstrate the effectiveness of our method within more advanced and complex harness environments, highlighting its practical value in real-world workflows.

![](images/4c3bac80f9b7e06d64a5bf5c2a50279bf18aa43b8f2206da31b6f9aabb524487.jpg)  
Figure 5: Performance comparison between Codex and Direct chat.

## 6 CONCLUSION

In this paper, we identify that existing skill evolution methods lack not only analysis of different trajectories for the same question, but also historical information across training steps. To mitigate these issues, we propose SkillCome, a skill-evolution method based on group contrast analysis and a dual-memory system. Combining the two synergistic components, SkillCome derives precise optimization signals and generalizable optimization directions, learning effective and reusable skills. Comprehensive evaluation across diverse models and benchmarks demonstrates that SkillCome substantially improves agent performance, highlighting its potential for various real-world applications.

## 7 AI USE STATEMENT

In this work, we used generative AI tools for proofreading and polishing the paper. For example, we use LLMs to check the grammar and improve the readability. We also used LLMs for literature discovery. Specifically, they suggested lists of potentially relevant papers, which the authors then independently located the original publications. The citations and descriptions of cited work were prepared by the authors. All AI-generated content was manually reviewed by the authors. We take responsibility for the final content of this work, including text, claims or artifacts produced with the aid of generative AI.

## 8 REPRODUCIBILITY STATEMENT

In this section, we list any related materials that help to reproduce this paper..

1. Datasets: The training and evaluation dataset we used is described in §4.1. The complete description of each dataset and the processing steps are provided in Appendix B.1.

2. Implementation Details: The implementation details, such as training&evaluation settings and hyperparameters are described in §4.1. And their details are provided in Appendix B.2.

3. Code: The code to reproduce our algorithm is provided in the supplementary materials.

## REFERENCES

Salaheddin Alzubi, Noah Provenzano, Jaydon Bingham, Weiyuan Chen, and Tu Vu. Evoskill: Automated skill discovery for multi-agent systems. arXiv preprint arXiv:2603.02766, 2026.

Tong Bai, Zheng-Lin Wan, Pengfei Zhou, Xingrui Yu, Wang-Bo Zhao, Yang You, and Ivor W. Tsang. Skilldag: Self-evolving typed skill graphs for llm skill selection at scale. ArXiv, abs/2606.03056, 2026. URL https://api.semanticscholar.org/CorpusID:288863363.

DeepSeek-AI. Deepseek-v4: Towards highly efficient million-token context intelligence, 2026.

Matthew Dunn, Levent Sagun, Mike Higgins, V Ugur Guney, Volkan Cirik, and Kyunghyun Cho. Searchqa: A new q&a dataset augmented with context from a search engine. arXiv preprint arXiv:1704.05179, 2017.

Bofei Gao, Feifan Song, Zhe Yang, Zefan Cai, Yibo Miao, Qingxiu Dong, Lei Li, Chenghao Ma, Liang Chen, Runxin Xu, Zhengyang Tang, Benyou Wang, Daoguang Zan, Shanghaoran Quan, Ge Zhang, Lei Sha, Yichang Zhang, Xuancheng Ren, Tianyu Liu, and Baobao Chang. Omnimath: A universal olympiad level mathematic benchmark for large language models, 2024. URL https://arxiv.org/abs/2410.07985.

Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, Jie Liu, Lei Qi, Zhiyuan Liu, and Maosong Sun. OlympiadBench: A challenging benchmark for promoting AGI with olympiad-level bilingual multimodal scientific problems. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar (eds.), Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 3828–3850, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long.211. URL https://aclanthology.org/2024. acl-long.211/.

Linyang He, Qiyao Yu, Hanze Dong, Baohao Liao, Xinxing Xu, Micah Goldblum, Jiang Bian, and Nima Mesgarani. Livemathematicianbench: A live benchmark for mathematician-level reasoning with proof sketches, 2026. URL https://arxiv.org/abs/2604.01754.

Yanna Jiang, Delong Li, Haiyu Deng, Baihe Ma, Xu Wang, Qin Wang, and Guangsheng Yu. Sok: Agentic skills–beyond tool use in llm agents. arXiv preprint arXiv:2602.20867, 2026.

Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In International Conference on Learning Representations (ICLR), 2015. URL https://arxiv.org/abs/ 1412.6980.

Xiangyi Li, Wenbo Chen, Yimin Liu, Shenghan Zheng, Xiaokun Chen, Yifeng He, Yubo Li, Bingran You, Haotian Shen, Jiankai Sun, Shuyi Wang, Binxu Li, Qunhong Zeng, Di Wang, Xuandong Zhao, Yuanli Wang, Roey Ben Chaim, Zonglin Di, Yipeng Gao, Junwei He, Liqiang Jing, Luyang Kong, Xin Lan, Jiachen Li, Songlin Li, Yijiang Li, Yueqian Lin, Xinyi Liu, Xuanqing Liu, Ze Ma, Runhui Wang, Tianyu Wang, Wengao Ye, Yue Zhang, Hanwen Xing, Kaixin Li, Yuchen Tian, Shutong Wu, Yixuan Gao, Bo Chen, Litong Liu, Sikai Cheng, Jiajun Bao, Shuaicheng Tong, Shuwen Xu, Terry Yue Zhuo, Tinghan Ye, Qi Qi, Miao Li, Longtai Liao, Zelin Tan, Chang Shi, Xilin Tang, Srinath Tankasala, Boqin Yuan, Yaoyao Qian, Jianhong Tu, Chenguang Wang, Yizhou Sun, Wei Wang, Aaron Taylor, Ziyue Yang, Changkun Guan, Zhikang Dong, Xinyu Zhang, and Dawn Song. Skillsbench: Benchmarking how well agent skills work across diverse tasks. arXiv preprint arXiv:2602.12670, 2026.

Huawei Lin, Peng Li, Jie Song, Fu-Xin Jiang, and Tieying Zhang. Muse-autoskill: Self-evolving agents via skill creation, memory, management, and evaluation. ArXiv, abs/2605.27366, 2026. URL https://api.semanticscholar.org/CorpusID:288671734.

Pengfei Liu, Weizhe Yuan, Jinlan Fu, Zhengbao Jiang, Hiroaki Hayashi, and Graham Neubig. Pretrain, prompt, and predict: A systematic survey of prompting methods in natural language processing. ACM computing surveys, 55(9):1–35, 2023.

Xingyan Liu, Xiyue Luo, Linyu Li, Ganghong Huang, Jianfeng Liu, and Honglin Qiao. Skillforge: Forging domain-specific, self-evolving agent skills in cloud technical support. arXiv preprint arXiv:2604.08618, 2026a.

Yuxuan Liu, Zhaochen Su, Yuhao Zhang, Jiahe Guo, Zhongwei Xie, Huihao Jing, Lingyun Xie, Qing Zong, Yauwai Yim, Zhixiong Zhang, Haoran Li, and Yangqiu Song. Rethinking self-evolving agent skills: Feedback dynamics over multiple rounds. arXiv preprint arXiv:2608.02636, 2026b.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations, 2019. URL https://openreview.net/forum?id= Bkg6RiCqY7.

Zeyao Ma, Bohan Zhang, Jing Zhang, Jifan Yu, Xiaokang Zhang, Xiaohan Zhang, Sijia Luo, Xi Wang, and Jie Tang. Spreadsheetbench: Towards challenging real world spreadsheet manipulation. Advances in Neural Information Processing Systems, 37:94871–94908, 2024.

Minesh Mathew, Dimosthenis Karatzas, and CV Jawahar. Docvqa: A dataset for vqa on document images. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pp. 2200–2209, 2021.

Jingwei Ni, Yihao Liu, Xinpeng Liu, Yutao Sun, Mengyu Zhou, Pengyu Cheng, Dexin Wang, Erchao Zhao, Xiaoxi Jiang, and Guanjun Jiang. Trace2skill: Distill trajectory-local lessons into transferable agent skills. arXiv preprint arXiv:2603.25158, 2026.

Krista Opsahl-Ong, Arnav Singhvi, Jasmine Collins, Ivan Zhou, Cindy Wang, Ashutosh Baheti, Owen Oertell, Jacob Portes, Sam Havens, Erich Elsen, Michael Bendersky, Matei Zaharia, and Xing Chen. Officeqa pro: An enterprise benchmark for end-to-end grounded reasoning. arXiv preprint arXiv:2603.08655, 2026.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, and Ryan Lowe. Training language models to follow instructions with human feedback. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho (eds.), Advances in Neural Information Processing Systems, 2022. URL https://openreview.net/forum?id= TG8KACxEON.

Siru Ouyang, Jun Yan, Yanfei Chen, Rujun Han, Zifeng Wang, Bhavana Dalvi Mishra, Rui Meng, Chun-Liang Li, Yizhu Jiao, Kaiwen Zha, Maohao Shen, Vishy Tirumalashetty, George Lee, Jiawei Han, Tomas Pfister, and Chen-Yu Lee. Skillos: Learning skill curation for self-evolving agents. arXiv preprint arXiv:2605.06614, 2026.

Reid Pryzant, Dan Iter, Jerry Li, Yin Lee, Chenguang Zhu, and Michael Zeng. Automatic prompt optimization with “gradient descent” and beam search. In Proceedings of the 2023 conference on empirical methods in natural language processing, pp. 7957–7968, 2023.

Qwen Team. On the design of Qwen3.8-Next architecture: Evaluation, efficiency, and training stability. Technical report, Alibaba Group, August 2026a.

Qwen Team. Qwen3.6-35B-A3B: Agentic coding power, now open to all, April 2026b. URL https://qwen.ai/blog?id=qwen3.6-35b-a3b.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. URL https: //openreview.net/forum?id=HPuSIXJaa9.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Shuaike Shen, Wenduo Cheng, Mingqian Ma, Alistair Turcan, Martin Jinye Zhang, and Jian Ma. Skillfoundry: Building self-evolving agent skill libraries from heterogeneous scientific resources. arXiv preprint arXiv:2604.03964, 2026a.

Yifei Shen, Bo Li, and Xinjie Zhang. Skillopt-lite: Better and faster agent self-evolution via one line of vibe. arXiv preprint arXiv:2607.03451, 2026b.

Yaorui Shi, Yuxin Chen, Zhengxi Lu, Yuchun Miao, Shugui Liu, Qi Gu, Xunliang Cai, Xiang Wang, and An Zhang. Skill1: Unified evolution of skill-augmented agents via reinforcement learning. arXiv preprint arXiv:2605.06130, 2026.

Mohit Shridhar, Xingdi Yuan, Marc-Alexandre Cote, Yonatan Bisk, Adam Trischler, and Matthew Hausknecht. Alfworld: Aligning text and embodied environments for interactive learning. In International Conference on Learning Representations, 2021. URL https://openreview. net/forum?id=0IOX0YcCdTn.

Ilya Sutskever, James Martens, George Dahl, and Geoffrey Hinton. On the importance of initialization and momentum in deep learning. In Sanjoy Dasgupta and David McAllester (eds.), Proceedings of the 30th International Conference on Machine Learning, volume 28 of Proceedings ofMachine Learning Research, pp. 1139–1147, Atlanta, Georgia, USA, 17–19 Jun 2013. PMLR. URL https://proceedings.mlr.press/v28/sutskever13.html.

Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. Voyager: An open-ended embodied agent with large language models. arXiv preprint arXiv:2305.16291, 2023a.

Jiongxiao Wang, Qiaojing Yan, Ya-Wei Wang, Yijun Tian, Soumya Smruti Mishra, Zhichao Xu, Megha Gandhi, Pan-Pan Xu, and Lin Lee Cheong. Reinforcement learning for selfimproving agent with skill library. ArXiv, abs/2512.17102, 2025. URL https://api. semanticscholar.org/CorpusID:284058483.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. In The Eleventh International Conference on Learning Representations, 2023b. URL https://openreview.net/forum?id=1PL1NIMMrw.

Jason Wei, Maarten Bosma, Vincent Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M. Dai, and Quoc V Le. Finetuned language models are zero-shot learners. In International Conference on Learning Representations, 2022a. URL https://openreview.net/ forum?id=gEZrGCozdqR.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, brian ichter, Fei Xia, Ed Chi, Quoc V Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh (eds.), Advances in Neural Information Processing Systems, volume 35, pp. 24824–24837. Curran Associates, Inc., 2022b. doi: 10.52202/ 068431-1800. URL https://proceedings.neurips.cc/paper\_files/paper/ 2022/file/9d5609613524ecf4f15af0f7b31abca4-Paper-Conference.pdf.

Xiyang Wu, Zongxia Li, Guangyao Shi, Alexander Duffy, Tyler Marques, Matthew Lyle Olson, Tianyi Zhou, and Dinesh Manocha. Co-evolving llm decision and skill bank agents for longhorizon tasks. ArXiv, abs/2604.20987, 2026. URL https://api.semanticscholar. org/CorpusID:287702239.

Renjun Xu and Yang Yan. Agent skills for large language models: Architecture, acquisition, security, and the path forward. arXiv preprint arXiv:2602.12430, 2026.

Yifan Yang, Ziyang Gong, Weiquan Huang, Qihao Yang, Ziwei Zhou, Zisu Huang, Yan Li, Xuemei Gao, Qi Dai, Bei Liu, Kai Qiu, Yuqing Yang, Dongdong Chen, Xue Yang, and Chong Luo. Skillopt: Executive strategy for self-evolving agent skills. arXiv preprint arXiv:2605.23904, 2026a.

Yutao Yang, Junsong Li, Qianjun Pan, Bihao Zhan, Yuxuan Cai, Lin Du, Jie Zhou, Kai Chen, Qin Chen, Xin Li, Bo Zhang, and Liang He. Autoskill: Experience-driven lifelong learning via skill self-evolution. arXiv preprint arXiv:2603.01145, 2026b.

Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhutdinov, and Christopher D. Manning. HotpotQA: A dataset for diverse, explainable multi-hop question answering. In Conference on Empirical Methods in Natural Language Processing (EMNLP), 2018.

Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik Narasimhan. Tree of thoughts: Deliberate problem solving with large language models. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine (eds.), Advances in Neural Information Processing Systems, volume 36, pp. 11809–11822. Curran Associates, Inc., 2023. doi: 10.52202/075280-0517. URL https://proceedings.neurips.cc/paper\_files/paper/2023/file/ 271db9922b8d1f4dd7aaef84ed5ac703-Paper-Conference.pdf.

Aohan Zeng, Xin Lv, Zhenyu Hou, Zhengxiao Du, Qinkai Zheng, Bin Chen, Da Yin, Chendi Ge, Chenghua Huang, Chengxing Xie, et al. Glm-5: from vibe coding to agentic engineering. arXiv preprint arXiv:2602.15763, 2026.

Hanrong Zhang, Shicheng Fan, Henry Peng Zou, Yankai Chen, Zhenting Wang, Jiayu Zhou, Chengze Li, Wei-Chieh Huang, Yifei Yao, Kening Zheng, Xue Liu, Xiaoxiao Li, and Philip S. Yu. Coevoskills: Self-evolving agent skills via co-evolutionary verification. arXiv preprint arXiv:2604.01687, 2026.

Boyuan Zheng, Michael Y. Fatemi, Xiaolong Jin, Zora Zhiruo Wang, Apurva Gandhi, Yueqi Song, Yu Gu, Jayanth Srinivasa, Gao-Wen Liu, Graham Neubig, and Yu Su. Skillweaver: Web agents can self-improve by discovering and honing skills. ArXiv, abs/2504.07079, 2025. URL https: //api.semanticscholar.org/CorpusID:277634081.

Jiayin Zhu, Kelong Mao, Yudong Guo, Deng-Bo He, Sulong Xu, Simiu Gu, and Yutao Yue. Skillcoach: Self-evolving rubrics for evaluating and enhancing agentic skilluse. ArXiv, abs/2607.01874, 2026. URL https://api.semanticscholar.org/ CorpusID:289749220.

## APPENDIX

A Method Details 16   
B Detailed Experimental Setting 18   
B.1 Dataset Details 18   
B.2 Implementation Details 19   
C Additional Experiments 19   
C.1 Notable Findings in the Experiment 19   
C.2 Standard Deviation of Main Results 21   
C.3 Statistics of the Mix Group 21   
C.4 Ablation on the Components of SkillCome . 21   
C.5 Evaluation in Agent Harnesses 22   
C.6 Cross-Model Transferability 22   
C.7 Cross-Dataset Transferability 22   
C.8 Evolving from Empty Skill 23   
C.9 Token Usage of Optimizer Model 23

## A METHOD DETAILS

Prompt A.1: Prompt for Group Contrast Reflection   
You are an expert same-task contrast analyst for AI agent trajectories.   
You receive several valid rollouts of the EXACT SAME task under the same skill. At least   
one rollout succeeded and at least one failed. Analyze them jointly; do not treat them as   
unrelated examples.   
Procedure:   
1. Compare successful and failed trajectories and locate the earliest meaningful   
behavioral divergence.   
2. Explain what the successful rollout did that the failed rollout omitted,   
misunderstood, or executed incorrectly.   
3. Check whether the current skill already contains the needed guidance. If it does,   
sharpen it only when the failed behavior shows the rule is ambiguous or   
insufficiently actionable.   
4. Distinguish skill-relevant reasoning/strategy errors from random wording, evaluator   
noise, tool nondeterminism, and infrastructure behavior.   
5. Propose only generalizable guidance. Never encode the task id, its answer, entity   
names, file names, cell addresses, or other instance-specific facts.   
6. Prefer the smallest effective edit. It is correct to return no edits when the   
contrast does not support a causal, reusable lesson.   
7. Do not edit protected content between SLOW UPDATE markers.   
8. Compare the mechanism with the supplied cross-step hypothesis bank. Link an   
existing ID only when the causal mechanism is genuinely the same. Otherwise leave   
the related ID empty. A causal, generalizable contrast from one task remains   
eligible for an edit; cross-task support is ranking evidence, not an eligibility   
requirement.   
Respond ONLY with a valid JSON object:   
{   
"contrast summary": {   
"first divergence": "<earliest important difference>",   
"success behavior": "<what worked>",   
"failure behavior": "<what went wrong>",   
"causal hypothesis": "<why this changed the outcome>",   
"generalizable": true,   
"confidence": 0.0   
},   
"hypothesis": {   
"related hypothesis id": "<existing bank ID or empty>",   
"mechanism": "<abstract reusable behavioral mechanism>"   
},   
"patch": {   
"reasoning": "<why these edits follow from the same-task contrast>",   
"edits": [   
{   
"op": "append|insert after|replace|delete",   
"target": "<if needed>",   
"content": "<generalizable markdown guidance>"

In this section, we provide the prompt for the core component of SkillCome, group contrast reflec tion. As shown in Prompt A.1, the optimizer jointly analyzes successful and failed trajectories within the same group and returns a structured JSON response containing the contrast analysis, pattern information for contrast memory, and proposed skill edits. Prompt A.2 further presents an example output of group contrast reflection. The example corresponds to the ALFWorld case discussed in Figure 3 of §5.1. It is the output from training step 1, telling the agent to search the target beyond visited places around the destination.

```csv
Prompt A.2: Example of Group Contrast Reflection Output
{
"contrast summary": {
"first divergence": "The successful rollout eventually visited both sidetables after
exhaustively searching dresser-adjacent locations, found the two keychains there,
transported them one at a time to the dresser, and placed them. Most failed rollouts never
visited the sidetables at all, or found the keychains late but then wandered back to
already-searched receptacles instead of delivering them immediately.",
"success behavior": "The successful agent treated the search as open-ended across all
room furniture rather than only around the destination dresser. Once it found the first
keychain, it handled the full inventory constraint by moving that keychain to the
destination before returning for the second keychain, then delivered the second keychain to
the same destination.",
"failure behavior": "Failed trajectories repeatedly cycled through drawers and cabinets
near the dresser, re-examined known-empty receptacles many times, and often never explored
the sidetables containing the goal objects. Even trajectories that found one keychain
frequently continued searching known-empty locations or failed to deliver held goal objects
promptly.",
"causal hypothesis": "Agents assumed the target objects would be near the destination
and exhausted only its neighboring receptacles, repeatedly revisiting them instead of
searching unexplored furniture elsewhere. This caused step-limit failure because the
actual target objects were on unvisited sidetables. In addition, when an agent held one
target object, it needed to place it before picking up the second, because inventory holds
only one object.",
"generalizable": true,
"confidence": 0.82
},
"hypothesis": {
"related hypothesis id": "contrast v2 cc5a59433591",
"mechanism": "Agents fail multi-object search tasks when they limit exploration to
receptacles near the stated destination and repeatedly revisit exhausted locations.
Success requires maintaining unexplored-location coverage across the whole environment and,
when inventory is capacity-limited and the task requires two objects, delivering the first
target object before retrieving the second."
},
"patch": {
"reasoning": "The same-task contrast shows failures not because the agents lacked the
locate/take/place sequence but because they searched only near the destination and
revisited known-empty receptacles. The existing skill already warns about revisiting
searched locations, but it does not explicitly instruct agents to broaden exploration
across the entire environment when local receptacles are empty, nor to deliver a held
target object before attempting to pick up a second required object when inventory is
limited. These are actionable, task-type-general extensions.",
"edits": [
"op": "append",
"target": "Common Mistakes to Avoid",
"content": "- <sub>**</sub>Destination-tunnel vision<sub>**</sub>: Do not search only receptacles
adjacent to the final destination. If nearby places are empty, expand exploration to other
furniture in the room before revisiting anything.\n- <sub>**</sub>Holding one target while trying to
take another<sub>**</sub>: If the task requires two objects and inventory is full, deliver the first
held target to the destination before picking up the second.",
"source type": "contrast",
"support count": 2,
"group support": 2,
"trajectory support": 8,
"memory refs": [
{
"bank type": "contrast",
"hypothesis id": "contrast v2 cc5a59433591"
}
],
"hypothesis id": "contrast v2 cc5a59433591"
}
},
"source type": "contrast",
"batch size": 8,
"trajectory support": 8,
"group support": 2,
"group ids": [
"train/look at obj in light-Pen-None-DeskLamp-305/trial T20190907 115849 734053",
"train/pick clean then place in recep-Lettuce-None-GarbageCan-20/trial T20190909
033324 286989",
"train/pick two obj and place-KeyChain-None-Dresser-322/trial T20190908 051114 312895"
],
"hypothesis id": "contrast v2 cc5a59433591"
```

Table 3: The statistics of the datasets used in our paper. Transfer-only target datasets have no training or validation split.
<table><tr><td>Dataset</td><td>Train</td><td>Validation</td><td>Test</td><td>Total</td></tr><tr><td>SearchQA</td><td>400</td><td>200</td><td>1400</td><td>2000</td></tr><tr><td>SpreadsheetBench</td><td>40</td><td>79</td><td>281</td><td>400</td></tr><tr><td>OfficeQA</td><td>49</td><td>49</td><td>148</td><td>246</td></tr><tr><td>DocVQA LiveMath</td><td>107</td><td>53</td><td>374</td><td>534</td></tr><tr><td>ALFWorld</td><td>36 200</td><td>35 140</td><td>106 134</td><td>177 474</td></tr><tr><td>HotpotQA</td><td>一</td><td>一</td><td>1000</td><td>1000</td></tr><tr><td>OlympiadBench</td><td>40</td><td>100</td><td>60</td><td>200</td></tr><tr><td>Omni-MATH</td><td>一</td><td>一</td><td>1000</td><td>1000</td></tr></table>

## B DETAILED EXPERIMENTAL SETTING

## B.1 DATASET DETAILS

Table 3 summarizes the dataset and splits adopted in our work. The main evaluation covers six benchmarks spanning question answering, mathematical reasoning, and agentic tasks. We mainly follow the splits from SkillOpt-Lite (Shen et al., 2026b). Detailed information on each dataset is as follows:

• SearchQA (Dunn et al., 2017) pairs trivia questions with supporting snippets retrieved from the web. Answering these questions requires extracting relevant facts from potentially noisy or redundant evidence. Here, models answer these questions using the provided snippets rather than performing live web searches.

• SpreadsheetBench (Ma et al., 2024) evaluates spreadsheet manipulation through practical requests originating from real users. Agents must interpret natural-language instructions and implement Python code to perform the corresponding operations on spreadsheet files.

• OfficeQA (Opsahl-Ong et al., 2026) evaluates question answering over lengthy financial documents containing narrative text and tables. We use the 246-question collection in an offline document-tool setting, where agents inspect locally available documents to produce their answers.

• DocVQA (Mathew et al., 2021) evaluates visual question answering over document images. The model must read document content and interpret its spatial organization to locate the information requested by a question.

• LiveMathematicianBench (LiveMath) (He et al., 2026) assesses research-level mathematical reasoning through multiple-choice questions constructed from mathematical research papers.

• ALFWorld (Shridhar et al., 2021) evaluates sequential decision making in household environments through a text-based interaction interface. Agents must search for objects, manipulate them, and coordinate multiple actions to satisfy a specified goal.

We additionally use three datasets for cross-dataset transfer experiments. The search-based benchmark HotpotQA (Yang et al., 2018) is adopted to evaluate the skills evolved on SearchQA. We also transfer the skills evolved on OlympiadBench (He et al., 2024) to Omni-MATH (Gao et al., 2024). OlympiadBench includes training, validation, and test splits, whereas HotpotQA and Omni-MATH are used exclusively for transfer evaluation without training or validation.

• HotpotQA (Yang et al., 2018) is a multi-hop question-answering benchmark based on Wikipedia articles. Its questions require combining evidence from multiple passages, including linking related facts and comparing information about different entities. We randomly select 1000 examples from its distractor validation split to evaluate the direct transfer of skills learned on SearchQA.

• OlympiadBench (He et al., 2024) contains challenging open-end mathematics and physics problems drawn from competitions and examinations. We use a mathematics subset as the source dataset for mathematical skill transfer.

• Omni-MATH (Gao et al., 2024) evaluates Olympiad-level mathematical problem solving across diverse mathematical topics and difficulty levels. We evaluate skills transferred from OlympiadBench on 1000 randomly selected samples on Omni-MATH.

## B.2 IMPLEMENTATION DETAILS

In this subsection, we provide additional implementation details. Trace2Skill (Ni et al., 2026) uses its official implementation, which collects trajectories for one training epoch and analyzes them to construct the skill. Our method uses the same training budget to that of SkillOpt. Following the SkillOpt-Lite training protocol, we run SkillOpt and our method for four epochs or ten batches, whereas SkillOpt-Lite is executed for exactly ten batches.

For all models, the maximum length is set to 16384 tokens per response. For grouped rollouts, we sample n = 8 trajectories per question. In practice, because group rollouts involve repeated rollouts for the same question, we deduplicate the input trajectories for failure and success reflection. Since the test sets of the main benchmarks other than SearchQA contain only a few hundred examples, their scores can fluctuate across evaluations. We therefore evaluate these benchmarks over four independent runs and report the average score to reduce the impact of stochastic variation.

## C ADDITIONAL EXPERIMENTS

## C.1 NOTABLE FINDINGS IN THE EXPERIMENT

A LiveMath Question with a Meta-Option   
Question (excerpt). What is the strongest statement that can be proved about every sufficiently small   
perturbation ψ<sub>0</sub> ∈ E of ϕ measured by ρ<sub>R</sub>?   
Option A. One of the remaining options is correct, but a stronger result can be proven.   
Options B–E (omitted). Mathematical statements about stability with different conditions on the time   
domain, quantifiers, and phase or spatial shifts.   
Ground-truth answer: A.  
Figure 6: An example of the meta-option pattern in LiveMath. Option A is the meta-option and the ground-truth answer. The other options are summarized for brevity.

Table 4: Livemath Performance with the Meta and Non-Meta subsets.
<table><tr><td>Model</td><td>Skill</td><td>All</td><td>Meta</td><td>Non-Meta</td></tr><tr><td rowspan="3">GLM-5.3</td><td>No skill</td><td>36.08</td><td>14.67</td><td>52.50</td></tr><tr><td>Init skill</td><td>38.44</td><td>16.30</td><td>55.42</td></tr><tr><td>SkillCome</td><td>67.92</td><td>71.74</td><td>65.00</td></tr><tr><td rowspan="3">DeepSeek-V4 Flash</td><td>No skill</td><td>31.60</td><td>4.35</td><td>52.50</td></tr><tr><td>Init skill</td><td>35.38</td><td>10.33</td><td>54.58</td></tr><tr><td>SkillCome</td><td>74.29</td><td>96.74</td><td>57.08</td></tr></table>

As shown in Table 1 of §4.2, the three more capable models, GLM-5.3, Qwen3.8-Flash-Next, and DeepSeek-V4-Pro-0813 do not always outperform DeepSeek-V4-Flash and Qwen3.6-35B-A3B after skill evolution, especially on LiveMath. This is because some LiveMath questions include the option “One of the remaining options is correct, but a stronger result can be proven.” We refer to this as the meta-option, since this option is always labeled as correct whenever it appears. Figure 6 provides an example of questions with meta-option. Of the 106 test samples, 46 contain this option and constitute the Meta subset, while we refer to the remaining 60 as the Non-Meta subset. This regularity therefore provides a shortcut for answering the Meta subset without resolving the underlying mathematical problem.

![](images/7d4835ce61545720553f6c8e3301f2a729df6cbd6e23031b363a8d0eec3ae80f.jpg)  
Figure 7: Skills related to the Meta-option pattern on DeepSeek-V4-Flash and GLM-5.3. The skill of DeepSeek-V4-Flash instructs the agent to select the meta-option without further analysis. In contrast, the GLM-5.3 skill prioritizes the meta-option, but still requires the model to analyze the math problem itself. [...] indicates omitted text.

The skills learned for DeepSeek-V4-Flash and Qwen3.6-35B-A3B adopt a strict rule for this pattern: once the meta-option is detected, the model should stop further reasoning and return its corresponding label. By contrast, the skills learned for the stronger models also recognize the pattern but retain instructions to analyze the mathematical content before answering. Figure 7 contrasts the relevant skill learned for DeepSeek-V4-Flash and GLM-5.3. These different strategies explain why the weaker target models achieve higher overall scores on LiveMath, since their gains mostly came from the Meta subset.

Table 4 reports the detailed results for each subset on DeepSeek-V4-Flash and GLM-5.3. Relative to the initial skill, Flash improves from 10.33% to 96.74% on Meta, but only from 54.58% to 57.08% on Non-Meta. GLM-5.3 exhibits a more balanced improvement, reaching 71.74% on Meta and 65.00% on Non-Meta. Although Flash achieves a higher overall score, GLM-5.3 remains stronger on questions without the meta-option. Overall, GLM-5.3 exhibits stronger mathematical reasoning capabilities. The meta-option rule improves accuracy under the benchmark’s labeling pattern, but its benefit should not be interpreted as an equivalent improvement in mathematical reasoning.

Table 5: Standard deviations of SkillCome. Mean scores are provided in Table 1. “–” denotes unsupported evaluation.
<table><tr><td>Model</td><td>Spreadsheet</td><td>LiveMath</td><td>ALFWorld</td><td>OfficeQA</td><td>DocVQA</td></tr><tr><td>Qwen3.8-Flash-Next</td><td>1.78</td><td>1.81</td><td>1.83</td><td>2.09</td><td>0.69</td></tr><tr><td>DeepSeek-V4-Pro-0813</td><td>1.71</td><td>2.50</td><td>0.43</td><td>0.65</td><td></td></tr><tr><td>GLM-5.3</td><td>0.92</td><td>2.83</td><td>0.43</td><td>1.50</td><td></td></tr><tr><td>DeepSeek-V4-Flash</td><td>1.77</td><td>0.90</td><td>1.27</td><td>3.46</td><td></td></tr><tr><td>Qwen3.6-35B-A3B</td><td>3.39</td><td>0.90</td><td>1.65</td><td>2.01</td><td>0.86</td></tr></table>

## C.2 STANDARD DEVIATION OF MAIN RESULTS

Table 5 reports the standard deviations of SkillCome over four independent evaluation runs. SearchQA is excluded because it is evaluated only once. The relatively small standard deviations indicate that the performance of SkillCome remains stable.

## C.3 STATISTICS OF THE MIX GROUP

Table 6: Mixed-group counts and proportions by target model. A mixed group contains both successful and failed rollouts of the same task. Counts are reported as mixed groups / total groups.
<table><tr><td>Target model</td><td>Mixed groups</td><td>Proportion</td></tr><tr><td>DeepSeek-V4-Flash</td><td>142 / 430</td><td>33.02%</td></tr><tr><td>Qwen3.6-35B-A3B</td><td>192 / 480</td><td>40.00%</td></tr><tr><td>GLM-5.3</td><td>107 / 430</td><td>24.88%</td></tr><tr><td>DeepSeek-V4-Pro-0813</td><td>152 / 430</td><td>35.35%</td></tr><tr><td>Qwen3.8-Flash-Next</td><td>138 /480</td><td>28.75%</td></tr><tr><td>Total</td><td>731 /2250</td><td>32.49%</td></tr></table>

Table 6 summarizes the mixed groups collected during skill evolution. The two multimodal LLMs, Qwen3.6-35B-A3B and Qwen3.8-Flash-Next, have more total group counts since their experiments also include DocVQA. Across the five target models, 731 of the 2,250 groups contain both successful and failed trajectories. The remaining non-mixed groups contain either only successful trajectories or only failed trajectories. These statistics show that mixed groups account for a substantial fraction of the collected groups, supporting group contrast reflection across different models.

## C.4 ABLATION ON THE COMPONENTS OF SKILLCOME

In this subsection, we further analyze our ablation experiment on the components of SkillCome discussed in §4.3. Compared with SkillOpt, “SkillOpt w/ Memory” improves performance in 8 of the 11 model–benchmark pairs and reduces in the remaining six. Although the gains are not consistent, it improves DeepSeek-V4-Flash on OfficeQA by 21.79 points over SkillOpt. This can be attributed to the memory system, which helps to learn explicit guidance against the DSML-style answer format discussed in § 4.2.

Meanwhile, “SkillCome w/o Memory” generally shows better performance than Skillopt. Together, the ablations demonstrate that both components of our method are effective. More importantly, combining them leads to the best results on all model-benchmark pairs, suggesting that the two components are synergistic. Group contrast enables deeper per-question analysis, while dual memory connects these findings with evidence from a broader range of samples.

Table 7: Performance comparison of SkillCome under direct chat and the Codex CLI harness.
<table><tr><td>Model</td><td>Skill</td><td>Spreadsheet</td><td>SearchQA</td><td>LiveMath</td></tr><tr><td rowspan="3">Qwen3.8 Flash-Next</td><td>No skill</td><td>48.58</td><td>67.50</td><td>33.73</td></tr><tr><td>Init skill</td><td> $4 5 . 5 5 _ { - 3 . 0 3 }$ </td><td> $6 7 . 9 7 _ { + 0 . 4 7 }$ </td><td> $3 5 . 1 4 _ { + 1 . 4 1 }$ </td></tr><tr><td>SkillCome Codex-SkillCome</td><td> ${ 7 5 . 7 1 } _ { + 2 7 . 1 3 }$   $\mathbf { 8 5 . 1 4 _ { + 3 6 . 5 6 } }$ </td><td> $\underline { { 7 4 . 1 4 } } _ { + 6 . 6 4 }$   $\mathbf { 7 8 . 2 9 _ { + 1 0 . 7 9 } }$ </td><td> $6 3 . 6 8 _ { + 2 9 . 9 5 }$   ${ \bf 6 7 . 9 2 _ { + 3 4 . 1 9 } }$ </td></tr><tr><td rowspan="4">DeepSeek-V4 Pro-0813</td><td>No skill</td><td>47.78</td><td>66.57</td><td>28.07</td></tr><tr><td>Init skill</td><td> $4 8 . 4 0 _ { + 0 . 6 2 }$ </td><td> $6 7 . 0 7 _ { + 0 . 5 0 }$ </td><td> $3 1 . 3 7 _ { + 3 . 3 0 }$ </td></tr><tr><td>SkillCome</td><td> $6 2 . 1 0 _ { + 1 4 . 3 2 }$ </td><td> $\underline { { 7 2 . 5 0 } } _ { + 5 . 9 3 }$ </td><td></td></tr><tr><td>Codex-SkillCome</td><td> $7 5 . 2 7 _ { + 2 7 . 4 9 }$ </td><td> $7 2 . 7 1 _ { + 6 . 1 4 }$ </td><td> $5 3 . 3 0 _ { + 2 5 . 2 3 }$   ${ 7 1 . 2 3 } _ { + 4 3 . 1 6 }$ </td></tr></table>

## C.5 EVALUATION IN AGENT HARNESSES

The detailed results of our evaluation in agent harnesses are presented in Table 7, corresponding to Figure 5 in §5.4. For both models, running under Codex consistently outperforms running under direct chat across all datasets. The performance gains from the harness environment and its tools are particularly pronounced on SpreadsheetBench and Livemath. These results demonstrate the effectiveness of our method within advanced harness environments, highlighting its practical value in real-world workflows.

## C.6 CROSS-MODEL TRANSFERABILITY

Table 8: Cross-target skill transferability experiments of SkillCome. “Qwen3.6 → Flash” denotes transferring the skill learned on Qwen3.6-35B-A3B to DeepSeek-V4-Flash. $\mathrm { ^ { * } F l a s h } \to \mathrm { F l a s h } ^ { \prime \prime }$ is the original performance of SkillCome on the DeepSeek-V4-Flash.
<table><tr><td>Target Model Skill</td><td></td><td>Spreadsheet</td><td>SearchQA</td><td>LiveMath</td><td>ALFWorld</td><td>OfficeQA</td><td>Avg</td></tr><tr><td rowspan="4">DeepSeek-V4 Flash</td><td>No skill</td><td>47.33</td><td>64.85</td><td>31.60</td><td>59.51</td><td>5.41</td><td>41.74</td></tr><tr><td>Init skill</td><td> $4 8 . 5 8 _ { + 1 . 2 5 }$ </td><td> $6 5 . 4 3 _ { + 0 . 5 8 }$ </td><td> $3 5 . 3 8 _ { + 3 . 7 8 }$ </td><td> $5 6 . 9 1 _ { - 2 . 6 0 }$ </td><td> $1 1 . 6 6 _ { + 6 . 2 5 }$ </td><td> $4 3 . 5 9 _ { + 1 . 8 5 }$ </td></tr><tr><td>Qwen3.6 → Flash</td><td> $6 0 . 0 5 _ { + 1 2 . 7 2 }$ </td><td> $7 2 . 2 1 _ { + 7 . 3 6 }$ </td><td> $6 8 . 8 7 _ { + 3 7 . 2 7 }$ </td><td> $7 3 . 5 1 _ { + 1 4 . 0 0 }$ </td><td> $2 2 . 3 0 _ { + 1 6 . 8 9 }$ </td><td> $5 9 . 3 9 _ { + 1 7 . 6 5 }$ </td></tr><tr><td>Flash → Flash</td><td> ${ \bf 6 2 . 5 4 _ { + 1 5 . 2 1 } }$ </td><td> $7 3 . 3 6 _ { + 8 . 5 1 }$ </td><td> ${ \ 7 4 . 2 9 } _ { + 4 2 . 6 9 }$ </td><td> $\mathbf { 8 4 . 1 4 } _ { + 2 4 . 6 3 }$ </td><td> ${ \bar { \mathbf { 5 0 . 5 1 } } } _ { + 4 5 . 1 0 }$ </td><td> ${ \bf 6 8 . 9 7 _ { \pm 2 7 . 2 3 } }$ </td></tr><tr><td rowspan="4"> $\mathrm { Q w e n } 3 . 6$   $3 5 \mathrm { B } { \cdot } \mathrm { A } 3 \mathrm { B }$ </td><td>No skill</td><td>41.99</td><td>66.00</td><td>26.65</td><td>66.04</td><td>37.16</td><td>47.57</td></tr><tr><td> $\mathrm { I n i t s k i l l }$ </td><td> $4 0 . 7 5 _ { - 1 . 2 4 }$ </td><td> $6 6 . 2 3 _ { + 0 . 2 3 }$ </td><td> $3 0 . 1 9 _ { + 3 . 5 4 }$ </td><td> $7 9 . 4 8 _ { + 1 3 . 4 4 }$ </td><td> $3 8 . 6 8 _ { + 1 . 5 2 }$ </td><td> $5 1 . 0 7 _ { + 3 . 5 0 }$ </td></tr><tr><td> $\mathrm { F l a s h }  \mathrm { Q w e n } 3 . 6$ </td><td> $6 0 . 2 3 _ { + 1 8 . 2 4 }$ </td><td> $7 0 . 7 9 _ { + 4 . 7 9 }$ </td><td> $7 2 . 1 7 _ { + 4 5 . 5 2 }$ </td><td> $8 3 . 0 2 _ { + 1 6 . 9 8 }$ </td><td> $\mathbf { 4 0 . 7 1 _ { + 3 . 5 5 } }$ </td><td> $6 5 . 3 8 _ { + 1 7 . 8 1 }$ </td></tr><tr><td> $\mathrm { Q w e n } 3 . 6  \mathrm { Q w e n } 3 . 6$ </td><td> ${ \bf 6 1 . 4 8 _ { \pm 1 9 . 4 9 } }$ </td><td> $\mathbf { 7 1 . 2 9 _ { + 5 . 2 9 } }$ </td><td> $7 1 . 4 6 _ { + 4 4 . 8 1 }$ </td><td> $\mathbf { 8 6 . 3 8 _ { \pm 2 0 . 3 4 } }$ </td><td> $4 0 . 0 3 _ { + 2 . 8 7 }$ </td><td> ${ \bf 6 6 . 1 3 _ { \pm 1 8 . 5 6 } }$ </td></tr></table>

Figure 4a in § 5.2 presents the transfer of skills learned on Qwen3.6-35B-A3B to DeepSeek-V4- Flash. We also have evaluated the reverse direction and report the complete results in Table 8. Skills learned on DeepSeek-V4-Flash transfer well to Qwen3.6-35B-A3B, even outperforming skills evolved specifically for the target model on LiveMath and OfficeQA. Conversely, as discussed in § 5.2, the transferred skills on OfficeQA of DeepSeek-V4-Flash shows substantially lower OfficeQA performance than self-evolved skills, because the transferred skill does not explicitly address Flash’s DSML-style answer format. These results indicate that SkillCome learns not only reusable tasklevel strategies that generalize across different models, but also model-specific rules that correct distinctive behavioral errors.

## C.7 CROSS-DATASET TRANSFERABILITY

Table 9 reports the complete results corresponding to Figure 4b in § 5.2. We transfer the skills learned on SearchQA to HotpotQA, a similar search-based task. Following SkillOpt, we also transfer skills evolved from OlympiadBench to Omni-MATH, another benchmark for open-ended mathematical reasoning. The transferred skills improve performance across all eight model–benchmark pairs, with gains ranging from 0.2 to 4.9 points. Although the magnitude of improvement varies across models, the consistent gains on both target datasets indicate that the learned skills provide reusable guidance beyond the datasets on which they were optimized. These results support the cross-dataset generalizability of SkillCome within similar domains.

Table 9: Full results of cross-dataset skill transferability. $\Delta$ denotes the score improvement of transferred skill over no skill.
<table><tr><td>Source dataset</td><td>Target dataset Model</td><td></td><td>No skill</td><td>Transferred skill</td><td>Δ</td></tr><tr><td rowspan="4">SearchQA</td><td rowspan="4">HotpotQA</td><td>DeepSeek-V4-Pro-0813</td><td>63.20</td><td>64.80</td><td>+1.60</td></tr><tr><td>Qwen3.8-Flash-Next</td><td>66.30</td><td>67.90</td><td>+1.60</td></tr><tr><td>DeepSeek-V4-Flash</td><td>62.70</td><td>64.80</td><td>+2.10</td></tr><tr><td>Qwen3.6-35B-A3B</td><td>64.80</td><td>65.70</td><td>+0.90</td></tr><tr><td rowspan="4">OlympiadBench Omni-MATH</td><td rowspan="4"></td><td>DeepSeek-V4-Pro-0813</td><td>72.30</td><td>74.10</td><td>+1.80</td></tr><tr><td>Qwen3.8-Flash-Next</td><td>86.40</td><td>86.60</td><td>+0.20</td></tr><tr><td>DeepSeek-V4-Flash</td><td>58.90</td><td>60.10</td><td>+1.20</td></tr><tr><td>Qwen3.6-35B-A3B</td><td>66.70</td><td>71.60</td><td>+4.90</td></tr></table>

## C.8 EVOLVING FROM EMPTY SKILL

Table 10: Performance of SkillCome evolving from empty skills.
<table><tr><td>Model</td><td>Skill</td><td>Spreadsheet</td><td>SearchQA</td><td>LiveMath</td><td>ALFWorld</td><td>OfficeQA</td><td>DocVQA</td><td>Avg</td></tr><tr><td rowspan="2">DeepSeek-V4 Flash</td><td>No skill SkillCome</td><td>47.33 58.45 +11.12</td><td>64.85  $\underline { { 7 1 . 7 9 } } _ { + 6 . 9 4 }$ </td><td>31.60 54.95 +23.35</td><td>59.51 79.10+19.59</td><td>5.41  $7 . 0 9 _ { + 1 . 6 8 }$ </td><td>一 一</td><td>41.74 54.28+12.54</td></tr><tr><td>Init skill SkillCome</td><td>48.58 62.54+13.96</td><td>65.43  $7 3 . 3 6 _ { + 7 . 9 3 }$ </td><td>35.38  $7 4 . 2 9 _ { + 3 8 . 9 1 }$ </td><td>56.91 84.14+27.23</td><td>11.66 50.51+38.85</td><td>一 一</td><td>43.59 68.97+25.38</td></tr><tr><td rowspan="2">Qwen3.6 35B-A3B</td><td>No skill SkillCome</td><td>41.99 58.01+16.02</td><td>66.00  $\underline { { 7 0 . 0 7 } } { + 4 . 0 7 }$ </td><td>26.65  $6 4 . 8 6 _ { + 3 8 . 2 1 }$ </td><td>66.04  $\mathbf { 8 7 . 1 3 _ { \perp 2 1 . 0 9 } }$ </td><td>37.16  $3 3 . 2 8 _ { - 3 . 8 8 }$ </td><td>89.77  $9 0 . 0 4 _ { + 0 . 2 7 }$ </td><td>54.60 67.23+12.63</td></tr><tr><td>Init skill SkillCome</td><td>40.75  $\mathbf { 6 1 . 4 8 _ { \div 2 0 . 7 3 } }$ </td><td>66.23  ${ \bf 7 1 . 2 9 _ { + 5 . 0 6 } }$ </td><td>30.19  $7 1 . 4 6 _ { + 4 1 . 2 7 }$ </td><td>79.48  $8 6 . 3 8 _ { + 6 . 9 0 }$ </td><td>38.68  ${ \bf 4 0 . 0 3 _ { + 1 . 3 5 } }$ </td><td>90.17  $\mathbf { 9 1 . 2 4 _ { + 1 . 0 7 } }$ </td><td>57.58  ${ \bf 7 0 . 3 1 _ { \pm 1 2 . 7 3 } }$ </td></tr></table>

As shown in the main Table 1, initial skills do not consistently improve performance over “No skill”. For example, both Qwen models perform worse with the initial skill on Spreadsheet, while several other model–benchmark pairs exhibit only marginal gains. These observations raise the question of whether skill evolution can achieve better results when starting without an initial skill. We therefore investigate the setting of evolving from empty skill on DeepSeek-V4-Flash in Figure 4c of §5.3. Here, we provide more result on Qwen3.6-35B-A3B.

The full results are presented in Table 10. Skill evolution from an empty skill improves over the “No skill” baseline in 10 of the 11 model–benchmark pairs, demonstrating that SkillCome can learn useful guidance without a predefined skill. Nevertheless, evolution from the initial skill achieves higher final accuracy in 10 of the 11 pairs. The only exception is ALFWorld on Qwen3.6-35B-A3B, where starting from an empty skill yields a slight advantage. Notably, on Spreadsheet with Qwen3.6-35B-A3B, the initial skill reduces performance before evolution but leads to a better final result, suggesting that a skill’s immediate effectiveness does not fully reflect its value for evolution.

One explanation is that an initial skill provides task-specific procedures and constraints that offer a useful starting point for revision, even when some instructions are ineffective. Under a limited optimization budget, the optimizer model can refine this existing guidance and correct problematic rules, rather than construct and organize the skill from scratch. Such guidance may also influence the trajectories collected during evolution, providing more informative behavioral references for later updates. Overall, these results suggest that skill initialization affects not only the starting performance but also the subsequent evolution process. Although an empty initialization remains a viable alternative, an initial skill generally provides a more effective basis for evolution.

## C.9 TOKEN USAGE OF OPTIMIZER MODEL

Since the target model rollout budget is matched during training, we compare the optimizer model token usage of SkillCome and SkillOpt. As shown in Table 11, SkillCome consumes fewer optimizer tokens across all evaluated datasets for both target models. This reduction primarily comes from lower input token usage through trajectory deduplication under grouped rollouts, as described in Appendix B.2. These results demonstrate that SkillCome achieves substantial performance gains while reducing optimizer token costs.

Table 11: Optimizer token usage in millions (M). The ratio is computed as SkillCome/ SkillOpt.
<table><tr><td>Target model</td><td>Dataset</td><td>SkillCome (M)</td><td>SkillOpt (M)</td><td>Ratio</td></tr><tr><td rowspan="4">DeepSeek-V4 Flash</td><td>SpreadsheetBench</td><td>0.69</td><td>1.50</td><td>0.460×</td></tr><tr><td>SearchQA</td><td>1.69</td><td>4.64</td><td>0.364×</td></tr><tr><td>OfficeQA</td><td>0.97</td><td>2.20</td><td>0.441×</td></tr><tr><td>LiveMath ALFWorld</td><td>0.56 1.09</td><td>0.90 1.57</td><td>0.622× 0.694×</td></tr><tr><td rowspan="5">Qwen3.6 35B-A3B</td><td>SpreadsheetBench</td><td>0.56</td><td>1.31</td><td>0.427×</td></tr><tr><td>SearchQA</td><td>2.15</td><td>5.07</td><td>0.424×</td></tr><tr><td>OfficeQA</td><td>0.75</td><td>1.86</td><td>0.403×</td></tr><tr><td>DocVQA</td><td>0.16</td><td>0.45</td><td>0.356×</td></tr><tr><td>LiveMath</td><td>0.39</td><td>1.06</td><td>0.368×</td></tr><tr><td></td><td>ALFWorld</td><td>1.05</td><td>1.67</td><td>0.629×</td></tr></table>