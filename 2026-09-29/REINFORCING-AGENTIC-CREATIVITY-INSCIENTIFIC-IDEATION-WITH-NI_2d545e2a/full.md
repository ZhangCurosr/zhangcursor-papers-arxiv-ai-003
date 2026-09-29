![](images/d98b503f14a1c2d6f4e5b5adbf39468cdf894620681a5177a0d505f6d2577aeb.jpg)

# REINFORCING AGENTIC CREATIVITY INSCIENTIFIC IDEATION WITH NIGHT SCIENCE

Priyanka Kargupta<sup>1,\*</sup> Silviu Cucerzan<sup>2</sup> Shweti Mahajan<sup>3</sup> Allen Herring<sup>2</sup> Jiawei Han<sup>1</sup> Ryen W. White<sup>2</sup> Sujay Kumar Jauhar<sup>2,</sup>

<sup>1</sup>University of Illinois Urbana-Champaign <sup>2</sup>Microsoft <sup>3</sup>Microsoft Research Correspondence: pk36@illinois.edu, sjauhar@microsoft.com Blog: pkargupta.github.io/night\_scientist   
Code: microsoft/ai\_night\_scientist   
Work completed while interning at Microsoft.

## ABSTRACT

Large language models (LLMs) excel at structured, verifiable tasks, but their lowentropy bias can produce homogeneous and predictable outputs, limiting their utility for open-ended scientific ideation. Effective discovery, however, spans a broader creative spectrum: from structured day science to loosely structured, serendipitous night science that reaches ideas beyond those typically considered. We introduce AI NIGHT-SCIENTIST, an agentic framework that uses reinforcement learning to teach models when and how to depart from predictable reasoning. Grounded in cognitive science, we model creativity along three axes: action (what to do and how creatively), process (when to explore versus exploit), and outcome (the novelty and usefulness of the resulting idea). We use these axes to train models with GRPO, exposing them to varying degrees and forms of creativity throughout training. This produces substantially more diverse scientific proposals, expanding the range of research directions by 27.8% and contribution types by 14.9% over the base model. It also improves predicted citation impact by up to 32.0 percentage points and originality by 66.2 points. These gains cannot be reproduced by simply increasing decoding temperature; instead, we find that semantic guidance specifying what kind of creativity to pursue is critical. Overall, our results suggest that creativity is a learnable, multi-level ability that can be shaped to help researchers reach ideas beyond those typically explored by LLMs.

## 1 INTRODUCTION

Large language models (LLMs) have excelled at structured, systematic tasks with clear verifiability (e.g., coding and quantitative reasoning), where their performance is often improved with reinforcement learning (RL) (Guo et al., 2025; Wang et al., 2025; Zhong & Wang, 2024; Liu et al., 2024). This success is consistent with a broader tendency toward minimizing token entropy (Agarwal et al., 2025), where models favor high-probability outputs that reflect frequent patterns and expected answers in training data (McCoy et al., 2024). While LLMs have increasingly been applied to scientific ideation (Si et al., 2025; Gottweis et al., 2025), they lack originality (Zhao et al., 2025), tend to generate homogeneous outputs (Wenger & Kenett, 2025), and even plagiarize at nontrivial rates (Gupta & Pruthi, 2025). Ultimately, this directly conflicts with the key attributes of open-ended, creative scientific discovery: novelty, diversity, and serendipity.

Prior methods depict discovery as a highly structured process, involving hypotheses derived from prior observations and/or data, tested against evidence, and refined or rejected accordingly (Gottweis et al., 2025; Agarwal et al., 2026). But this is only \*part\* of the scientific discovery process. They typically view creativity as only an attribute of ideas rather than as part of the process, treating individual actions as fixed behaviors (e.g., search, write) and executing them through scaffolded pipelines (Gu et al., 2024; Lu et al., 2024). While this supports the critical reasoning crucial for validating existing ideas, this overlooks the creative reasoning necessary for discovering new ones.

![](images/4399b3ddcc1166a7eecd2aadea2e3786a7b37e955d24ec90d1a9dd1fcd24e8eb.jpg)  
Figure 1: Given an input task, creativity can be injected into three different axes of reasoning: action, process, and outcome. Moreover, each level can be executed with varying degrees ofcreativity.

Both are complementary and crucial perspectives of scientific discovery, referred to as day and night science (Wechsler et al., 2018; Halpern, 2007; Stent, 1988; Yanai & Lercher, 2019).

Night science captures the often-neglected nature of human-driven discovery: loosely structured and dynamic exploration driven by highly creative actions, such as exploring distant analogies (e.g., biology-inspired technology), considering alternative perspectives after serendipitous encounters (e.g., debates with colleagues from other domains), and acting on partly-formalized intuitions (e.g., abandoning status quo assumptions). Together, day and night science form a spectrum that human researchers have smoothly traversed to uncover breakthroughs (de Chantal & Markovits, 2022; Dwyer et al., 2025), such as chemotherapy and penicillin (Hirsch, 2006; Ligon, 2004).

We hypothesize that LLMs can better traverse this spectrum if given explicit control over when and how to deviate from high-probability output, enabling targeted doses of night science while maintaining day science’s goal-directed behavior. To test this, we introduce AI NIGHT-SCIENTIST, an RL-based agentic framework that incentivizes models to exercise this control effectively. Grounded in cognitive science (Lubart, 2001; Cohen, 1989; Dwyer et al., 2025; Harvey & Berry, 2023), it embeds creativity along three axes of reasoning (Figure 1):

• Action-level (how): Associates each action with a creativity level, allowing the same action to vary from conventional, high-probability behavior to more exploratory and unconventional behavior.

• Process-level (when): Controls when to shift between lower- and higher-creativity actions based on how the multi-step trajectory unfolds.

• Outcome-level (what): Captures the creativity of the resulting idea, favoring outputs that are novel while remaining relevant and useful.

We utilize this framework to train an LLM-based agent using GRPO (Shao et al., 2024) for generating scientific research proposals, where identifying promising ideas often requires long-horizon creative reasoning beyond immediately verifiable evidence. Overall, our work argues that LLM-based support for scientific discovery should span the full day-to-night science spectrum. AI NIGHT-SCIENTIST shows that conventional LLMs do not naturally navigate this spectrum effectively, but targeted reinforcement learning can reshape when and how they depart from structured, predictable reasoning, leading to more creative outcomes. Our contributions can be summarized as:

1. We introduce AI NIGHT-SCIENTIST, an RL-based agentic framework that explicitly represents creativity across actions, reasoning processes, and outcomes for long-horizon scientific ideation.

2. We show that creativity is more than sampling stochasticity: explicit guidance on how to be creative via RL helps an agent learn when creative deviations are useful, while higher temperature does not.

3. Empirically, AI NIGHT-SCIENTIST produces more diverse scientific proposals than its base model, expanding the range of research directions by 27.8% and contribution types by 14.9%, while improving predicted impact by up to 32.0 percentage points and originality by 66.2 points.

## 2 BACKGROUND AND RELATED WORK

Scientific discovery has been characterized as an interplay between structured, hypothesis-driven day science and more exploratory, intuition- and serendipity-driven night science (Stent, 1988; Yanai & Lercher, 2019). Classic theories of creativity characterize creative thought through remote associations between otherwise distant concepts (Mednick, 1962), generative and exploratory modes of cognition (Ward et al., 1999), and outcomes that are both original and useful (Runco & Jaeger, 2012). Creativity can also vary in degree: prior work describes a continuum of creative behavior (Cohen, 1989), with problem-solving strategies ranging from paradigm-preserving to paradigm-stretching and paradigm-breaking (McFadzean, 1998). Related accounts distinguish between exploring existing conceptual spaces and transforming them to enable qualitatively new ideas (Boden, 1998), while the ories of creative ideation show that originality can arise either by flexibly exploring many conceptual directions or by persistently exploring a few in greater depth (Nijstad et al., 2010). Together, these perspectives motivate our view of creativity as varying both where it enters reasoning (within actions, processes, and outcomes) and how strongly it is expressed.

This view contrasts with most current LLM-based approaches to scientific discovery. Existing research agents broaden ideation through retrieval, search, multi-agent interaction, and iterative generation (Gu et al., 2024; Lu et al., 2024; Kargupta et al., 2025a; Gottweis et al., 2025), but generally treat actions such as searching, debating, and writing as fixed behaviors and primarily assess creativity in the resulting idea. This leaves little control over how creatively individual actions are performed or when creative deviations should occur throughout reasoning. Relatedly, Kargupta et al. (2025b) find that LLMs struggle with the metacognitive awareness needed to monitor and adapt their reasoning, further limiting their ability to flexibly shift between critical and creative modes.

Recent work has begun targeting the mechanisms that produce creative scientific ideas. O’Neill et al. (2025) use structured assumption inversion to generate novel hypotheses, inspiring the spark action in our framework, while Kargupta et al. (2025c) and Kargupta et al. (2026) use retrieval to identify research gaps and surface interdisciplinary inspiration. Other work instead learns scientific capabilities directly: GIANTS (He-Yueya et al., 2026) trains smaller models to anticipate scientific insights, while Tong et al. (2026) study whether models can learn scientific taste. Together, these approaches suggest that not only scientific outputs, but also the processes that produce them, can be shaped through structure and supervision. Reinforcement learning offers a way to shape this process without prescribing exactly how discovery should unfold, although open-ended ideation has no single correct answer and must balance qualities such as novelty, relevance, feasibility, and usefulness (Afzal et al., 2025). Moreover, useful creative exploration requires more than simply injecting randomness (Schmidhuber, 2010). Our work builds on these ideas by learning both how creatively individual actions should be performed and when different levels of creativity are useful across a reasoning trajectory.

## 3 AI NIGHT-SCIENTIST: A CREATIVITY-ALIGNED AGENTIC FRAMEWORK

![](images/7999d155821ef7e7b81dff3687f87c3c382c729419635e66e6cef928dd91f3fb.jpg)  
Figure 2: AI NIGHT-SCIENTIST consists of a: (1) rollout phase, where an LLM builds a reasoning trajectory f by iteratively selecting and executing actions with creativity levels, and (2) reward phase during training, where the reward is computed over the final proposal.

We propose AI NIGHT-SCIENTIST, as illustrated in Figure 2, which represents creativity directly within the agent’s reasoning trajectory rather than only in its final output. We extend the ReAct-style reasoning setting (Yao et al., 2023): at each step, the agent selects an action together with a creativity level that specifies how that action should be carried out, executes it, and updates its current idea state. Repeating this process produces a trajectory through the space of possible ideas, allowing the agent to move between more familiar and more unexplored directions as reasoning unfolds. This lets us represent creativity at three levels (Figure 1): action-level creativity captures how an action is performed, process-level creativity captures when to shift between lower- and higher-creativity actions, and outcome-level creativity captures the novelty and usefulness of the resulting idea.

## 3.1 MULTI-LEVEL REPRESENTATION OF CREATIVE REASONING

We define each level below before applying the framework to scientific research proposal generation: Definition 3.1 (Action-Level Creativity). Let $\mathcal { A }$ denote the action space for a task $\tau$ . For each action $a \in { \mathcal { A } }$ , we define an ordered set of creativity levels $\mathcal { C } _ { a } = \{ c _ { a } ^ { ( 1 ) } , \dots , c _ { a } ^ { ( K _ { a } ) } \}$ , where each level gives a natural-language description of how a should be performed. Lower levels describe more conventional, high-probability behavior, while higher levels describe increasingly exploratory or unconventional behavior. The number of levels $K _ { a }$ may vary across actions.

Definition 3.2 (Process-Level Creativity). Let $\tau = \left( ( a _ { 1 } , c _ { 1 } , \hat { o } _ { 1 } ) , \dots , ( a _ { N } , c _ { N } , \hat { o } _ { N } ) \right)$ denote a reasoning trajectory, where $a _ { i } \in { \mathcal { A } }$ is the action selected at step $i , c _ { i } \in \mathcal { C } _ { a _ { i } }$ is its creativity level, and $\hat { o } _ { i }$ is the resulting intermediate output. At each step, the agent selects the next action-level choice $( a _ { i } , c _ { i } )$ based on the task $\tau$ and the preceding trajectory $\tau _ { 1 : i - 1 }$ . Process-level creativity reflects how effectively the agent sequences and adapts these choices over time. Higher process-level creativity means varying the degree of action-level creativity to benefit the evolving reasoning process, rather than consistently favoring either low- or high-creativity behavior.

Definition 3.3 (Outcome-Level Creativity). Let o denote the final outcome produced for a task $\tau$ Outcome-level creativity captures the creativity of o itself, based on two complementary properties: its novelty $\mathcal { N } ( o )$ and its usefulness $\mathcal { U } ( o )$ (Harvey & Berry, 2023). Novelty measures how much o departs from existing or familiar solutions, while usefulness measures how valuable, appropriate, or effective it is for the task. An outcome should exhibit both in order to be considered creative.

Together, this multi-level representation separates how creativity is expressed within an action, when different degrees of creativity are useful across reasoning, and what the process ultimately produces. We represent action-level creativity in natural language to make these choices interpretable and give the user direct semantic control over how and to what extent each action may deviate from its conventional execution. The number and meaning of creativity levels can also vary by action.

These action-level choices accumulate into the reasoning trajectory. Intermediate outputs may not appear directly in the final result, but they can shape later decisions and influence when greater or lesser deviation is useful. Greater creativity may help open new directions, while lower-creativity behavior may be better suited for developing or refining promising ones. Making these choices well requires the model to monitor how its reasoning is progressing and adapt accordingly, a key aspect of metacognitive control that is lacking in existing models (Kargupta et al., 2025b).

## 3.2 TASK FORMULATION: RESEARCH PROPOSAL GENERATION

Given a high-level research problem $p ,$ the agent uses the action space $\mathcal { A }$ in Table 1 to explore and develop a long-horizon research proposal over a multi-step trajectory τ . SEARCH, DEBATE, and SPARK support multiple creativity levels, allowing the agent to vary how broadly it searches, whose perspectives it considers, and how strongly it challenges existing assumptions. WRITE consolidates the trajectory into proposal text, while STOP ends the process and returns the final proposal o. We keep WRITE fixed so that the final proposal primarily reflects the exploration that preceded it.

We focus on proposals with a scope similar to long-term research grants, where ideas are intended to guide work over several years rather than describe an immediately executable experiment. This makes the setting well suited to studying creative reasoning: proposals must remain grounded in existing work, yet many of their central ideas cannot be directly verified at inference time because the required experiments or data may not yet exist. The task therefore rewards reasoning that can move beyond established directions while still producing ideas that are feasible and relevant to $p _ { \cdot }$

## 3.3 REINFORCING ADAPTIVE CREATIVITY

To teach models when and how different degrees of creativity are useful, we leverage reinforcement learning. RL allows the quality of the final idea to shape the reasoning process without prescribing how that process should unfold. Our training pipeline consists of two phases: (i) a rollout phase, where the agent constructs a reasoning trajectory τ using the creativity-aligned action space above, and (ii) a reward phase, where the resulting proposal o is evaluated. We use Group Relative Policy Optimization (GRPO) (Shao et al., 2024), which learns from relative rewards across sampled trajectories without requiring a separate critic. In our primary setting, the reward is applied only to the final proposal, requiring the model to learn which actions to take, how creatively to perform them, and how to sequence them based on their eventual effect on proposal quality.

<table><tr><td rowspan=1 colspan=1>Action A</td><td rowspan=1 colspan=1>Mechanism &amp; Output ô</td><td rowspan=1 colspan=1>Creativity Levels c ∈ Ca</td></tr><tr><td rowspan=1 colspan=1>@ Search</td><td rowspan=1 colspan=1>Generate query → retrieve arXivpapers based on embedding similarity</td><td rowspan=1 colspan=1>LI: Search proposal-specific background using title and core terms.L3: Search tangentially related background for broader context.L5: Search distant domains, alternate perspectives, or broader questions.</td></tr><tr><td rowspan=1 colspan=1>b Debate</td><td rowspan=1 colspan=1>Select participants and topic →retrieve relevant papers → simulatediscussion</td><td rowspan=1 colspan=1>L1: Discuss proposal specifics with a close-domain colleague.L3: Debate with a peer from the same field but a different topic.L5: Explore with experts from distant disciplines in an open-ended debate.</td></tr><tr><td rowspan=1 colspan=1> Spark</td><td rowspan=1 colspan=1>Identify assumption (Bit) → invert it(Flip) → reframe it (Spark)</td><td rowspan=1 colspan=1>LI: Challenge a narrow assumption specific to the current proposal.L3: Challenge a meaningful assumption underlying the approach.L5: Challenge a broad, field-level assumption through radical reframing.</td></tr><tr><td rowspan=1 colspan=1>Write</td><td rowspan=1 colspan=1>Synthesize prior trajectory into aproposal draft or revision</td><td rowspan=1 colspan=1>Fixed: consolidate prior trajectory; preserve credit assignment</td></tr><tr><td rowspan=1 colspan=1> Stop</td><td rowspan=1 colspan=1>End trajectory &amp; return final proposal</td><td rowspan=1 colspan=1>Judge final proposal by P (precedence), F (feasibility), and R (relevance)</td></tr></table>

Table 1: Action space for research proposal generation with low, mid and high creativity levels shown. Table 9 in Appendix E.1 includes all levels; full action prompts provided in Appendix F.

## 3.3.1 REWARDING PROPOSAL QUALITY

Scientific proposals contain multiple ideas whose novelty and feasibility may differ substantially, making a single holistic quality judgment difficult to ground. We therefore first decompose the final proposal into its atomic research ideas, separating what each phase proposes from how it plans to execute it. We then evaluate each component against the original research problem and relevant literature along three dimensions: precedence,feasibility, and relevance (Table 3.3.1).

<table><tr><td rowspan=1 colspan=1>Dimension</td><td rowspan=1 colspan=3>Definition                       Positive Example (Reward ↑)       Negative Example (Reward ↓)</td></tr><tr><td rowspan=1 colspan=1>Precedence</td><td rowspan=1 colspan=1>Distance from prior work and existingapproaches.</td><td rowspan=1 colspan=1>Core idea opens genuinely new techni-cal directions.</td><td rowspan=1 colspan=1>Recombines familiar componentswithout new insight.</td></tr><tr><td rowspan=1 colspan=1>Feasibility</td><td rowspan=1 colspan=1>Credibility &amp; specificity of the execu-tion plan.</td><td rowspan=1 colspan=1>Clear methods with precedent orscoped evaluation strategy.</td><td rowspan=1 colspan=1>Broad, underspecified plan or implau-sible scope.</td></tr><tr><td rowspan=1 colspan=1>Relevance</td><td rowspan=1 colspan=1>Alignment of the proposal with prob-lem p.</td><td rowspan=1 colspan=1>Addresses a critical bottleneck in thetarget domain.</td><td rowspan=1 colspan=1>Drifts off-topic or fails to engage withthe core challenges of p.</td></tr></table>

Table 2: Reward dimensions for proposal generation with examples. Reward prompts in Appendix E.

We average precedence and relevance across the proposal’s atomic ideas, but take the minimum feasibility across experimental plans, since a single infeasible phase may compromise the executability of the overall proposal. The resulting three proposal-level scores are then weighted equally to form the outcome reward: $\begin{array} { r } { R _ { \mathrm { o u t } } = \frac { 1 } { 3 } \left( R _ { \mathcal { P } } + R _ { \mathcal { F } } + \dot { R } _ { \mathcal { R } } \right) } \end{array}$ . These rewards are designed to provide targeted training signals for proposal quality, with atomic ideas evaluated separately and precedence and feasibility grounded in retrieved literature.

Outcome-only training still leaves the model to discover useful creative behaviors through their eventual effect on proposal quality. We therefore explore two additional forms of guidance: exposing the model to occasional serendipitous actions early in training, and directly rewarding properties of the intermediate reasoning process. Neither is required by the framework; we study whether either provides additional benefit beyond the final outcome reward.

## 3.3.2 INCENTIVIZING EXPLORATION THROUGH SERENDIPITOUS ACTIONS

Human discovery is often shaped by unexpected encounters that expose researchers to directions they may not have deliberately pursued (Lubart, 2001). Similarly, an LLM early in training may repeatedly select familiar, low-creativity behaviors simply because it has not experienced useful alternatives. We therefore explore a stochastic intervention that occasionally exposes the model to a different action-level choice. With probability Pr(swap), the selected pair $( a _ { i } , c _ { i } )$ is replaced by a sampled alternative $( a _ { i } ^ { \prime } , c _ { i } ^ { \prime } )$ , and a decay factor $\gamma$ gradually reduces this probability over training. The goal is not to prescribe random exploration, but to expose the model early on to creative behaviors whose value it can later learn from the resulting reward.

## 3.3.3 EXPLORING PROCESS-LEVEL REWARDS

Our primary models receive only the outcome reward above, allowing process-level behavior to emerge from credit assigned to the final proposal. We additionally investigate whether directly reward ing intermediate reasoning provides further benefit. The process reward evaluates each intermediate action along two complementary dimensions. Exploration measures whether $a _ { i }$ introduces a new direction relative to $\tau _ { 1 : i - 1 }$ , while contribution measures how much $a _ { i }$ ultimately contributes to the final proposal o. We score both dimensions for each intermediate action and average them across the trajectory to obtain $R _ { \mathrm { p r o c } } ,$ which then forms the final reward $R _ { \mathrm { p o } } = \textstyle \frac { 1 } { 2 } \left( R _ { \mathrm { p r o c } } + \bar { R } _ { \mathrm { o u t } } \right)$ . Full details and prompts are provided in Appendix L.

## 4 EXPERIMENTS

We train all models using verl (Sheng et al., 2024) with GRPO (Shao et al., 2024) on Qwen3-8B-Base and Qwen3-14B-Base. Each agent trajectory consists of up to 5 actions. For the search action, we index $\mathrm { \ u r X i v ^ { 1 } }$ and use embedding-based retrieval with GPT-4.1 as an external judge within the reward pipeline. Full training details are provided in Appendix M.

## 4.1 DATASET

We construct our dataset from the NSF Awards Database<sup>2</sup>, focusing on Computer Science, Engineering, and Mathematics (CSE) awards from 2018 onward. We use a 90:10 train/test split, yielding 4,414 training and 491 test awards. Our model receives only the award title as $p ,$ which typically describes a broad, long-horizon research problem; we withhold the award abstract, project outcomes, and associated publications so that generation is not anchored to the funded solution. We focus on CSE to ensure broad coverage in open-access arXiv literature, while retaining substantial interdisciplinary breadth: the training set spans 46 research domains, with 23.1% of subfield labels in core AI/ML and over 48% of label occurrences outside AI/ML and core CS and Engineering (Appendix B).

Original NSF grant proposals are not publicly available, so we construct reconstructed reference proposals from evidence surrounding each funded project. For each award, we collect its abstract, project outcomes report, PI information, and award period, then retrieve likely associated arXiv papers by the award PIs. Candidate papers are ranked using PI overlap, topical similarity, publication timing, and explicit funding acknowledgments. We prompt GPT-5.1 to treat these sources as downstream evidence and reconstruct a plausible pre-award research plan that could have led to the observed project and publications. The reference follows the same structured format as generated proposals, including a summary, background, and multi-phase research plan.

These reconstructions are not intended to reproduce the original NSF proposals. Rather, they provide standardized, evidence-grounded references for research directions that were actually funded and subsequently pursued. We use them only as matched references for pairwise evaluation, and the proposal-generating model never receives the oracle information used to construct them. Full reconstruction details and prompts are provided in Appendix K.

## 4.2 EVALUATION METRICS

Scientific creativity is difficult to capture with any single automatic metric, particularly because the long-term value of a research idea may not be observable for years. We therefore evaluate models along three complementary dimensions: predicted citation impact, literature-grounded originality, and research idea diversity. Together, these capture both the quality of individual proposals and the range of research directions explored across the dataset.

Predicted Citation Impact. For each problem, we compare the generated proposal against its matched reconstructed reference in randomized order and report pairwise win rate. We use SciJudge (Tong et al., 2026), a 30B-parameter model trained to predict relative citation impact from large-scale community signals, as a proxy for the potential usefulness and downstream value of the proposed research (prompts in Appendix G). It has an average 82.7% citation prediction accuracy.

Literature-Grounded Originality. We compare each generated proposal against its matched reference for originality. To ground the judgment in prior work, we retrieve the closest paper to each proposal and ask $\mathsf { G P T } \mathrm { - } 5 . 1$ to make a randomized pairwise comparison relative to the retrieved literature (Appendix I).

Research Idea Diversity. Pairwise metrics capture the quality of individual proposals but not whether a model repeatedly produces the same kinds of ideas. Following the annotation format of Chen et al. (2026), we use GPT-5.4-mini to classify each proposal by its primary research idea paradigm and, analogously, by its primary contribution type using categories adapted from prior taxonomies (Wobbrock, 2012; Miles, 2017). For each taxonomy, we report the effective number of categories represented (Appendix H), normalized by the seven available categories; higher values indicate a broader and more balanced range of research directions. The research-paradigm judge Chen et al. (2026) achieves κ = 0.84 agreement with human judgments on 150 samples.

Human-LLM Agreement. As an additional check on our pairwise judges, two human annotators evaluated a subset of proposals. Inter-annotator agreement was 80.0% $( \kappa = 0 . 6 2 5 )$ for impact and 86.7% $( \kappa = 0 . 7 6 6 )$ for originality, while human-LLM agreement was 76.7% for SciJudge on impact and 72.4% for GPT-5.1 on originality (Appendix J).

## 4.3 BASELINES

We compare against baselines that vary how creativity is introduced into proposal generation. Zeroshot models directly generate a structured proposal from the input problem without retrieval, action selection, or creativity levels. Temperature retains the same five creativity levels, but maps each level only to decoding temperature, with $c \in \{ 1 , \ldots , 5 \}$ corresponding to $\dot { T } \in \{ 0 , 0 . 2 5 , 0 . \dot { 5 } , 0 . 7 5 , 1 . 0 \}$ This tests whether increased stochasticity alone can reproduce the benefits of semantic creativity guidance. ReAct (Yao et al., 2023) uses the same action space A with no corresponding $\mathcal { C } _ { a } .$

Creative baselines apply our full creativity-aligned action space and natural-language creativity levels at inference time, but receive no RL training. We also train several RL variants with the same outcome-level reward as AI NIGHT-SCIENTIST: + Temp and + ReAct apply RL to the corresponding baselines above, while + Search + Write retains semantic creativity levels but restricts A to retrieval and writing, similar to retrieval-augmented ideation systems (Si et al., 2025; Wang et al., 2024). Together, these separate the effects of semantic creativity guidance, RL, and the broader action space.

Finally, we compare against GIANTS (He-Yueya et al., 2026), an RL-trained model for scientific insight anticipation. Because GIANTS expects two input papers, we retrieve the two most relevant arXiv papers using the award title and abstract and reformat them with GPT-4.1. GIANTS therefore receives more award-specific information than AI NIGHT-SCIENTIST, which sees only the title.

For closed-source baselines, we use GPT-4.1, a strong general-purpose, non-reasoning model. Model recency does not necessarily imply greater creative diversity, with recent work finding increasing similarity across generations on open-ended tasks (Patel et al., 2026).

## 4.4 EXPERIMENTAL RESULTS & ANALYSIS

## 4.4.1 CREATIVITY-ALIGNED RL IMPROVES PROPOSAL QUALITY

Table 3(a) shows a consistent advantage for AI NIGHT-SCIENTIST over both direct-generation and agentic baselines. Relative to Qwen3-8B, NIGHT-8B improves predicted citation impact by 29.43 percentage points and originality by 53.79 points. At 14B, these gains grow to 32.03 and 66.15 points over Qwen3-14B. In comparison, scaling the zero-shot model from 8B to 14B yields almost no improvement, whereas scaling AI NIGHT-SCIENTIST adds another 3.29 points in citation and

(a) Proposal quality
<table><tr><td rowspan=1 colspan=1>Category</td><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>Citation (%) ↑</td><td rowspan=1 colspan=1>Originality (%) ↑</td></tr><tr><td rowspan=4 colspan=1>Zero-shot</td><td rowspan=1 colspan=1>Llama-3.1-8B</td><td rowspan=1 colspan=1>1.61</td><td rowspan=1 colspan=1>1.15</td></tr><tr><td rowspan=2 colspan=1>Qwen3-8BQwen3-14B</td><td rowspan=1 colspan=1>0.46</td><td rowspan=1 colspan=1>2.53</td></tr><tr><td rowspan=1 colspan=1>1.15</td><td rowspan=1 colspan=1>2.53</td></tr><tr><td rowspan=1 colspan=1>GPT-4.1</td><td rowspan=1 colspan=1>3.90</td><td rowspan=1 colspan=1>12.18</td></tr><tr><td rowspan=2 colspan=1>Temp.</td><td rowspan=2 colspan=1>Qwen3-8B + TempGPT-4.1 + Temp</td><td rowspan=1 colspan=1>2.07</td><td rowspan=1 colspan=1>0.69</td></tr><tr><td rowspan=1 colspan=1>2.99</td><td rowspan=1 colspan=1>9.89</td></tr><tr><td rowspan=2 colspan=1>ReAct</td><td rowspan=2 colspan=1>Qwen3-8B + ReActGPT-4.1 + ReAct</td><td rowspan=1 colspan=1>2.33</td><td rowspan=1 colspan=1>6.54</td></tr><tr><td rowspan=1 colspan=1>3.45</td><td rowspan=1 colspan=1>10.11</td></tr><tr><td rowspan=4 colspan=1>Creative</td><td rowspan=4 colspan=1>Llama-3.1-8B + CreativeQwen3-8B + CreativeQwen3-14B + CreativeGPT-4.1 + Creative</td><td rowspan=1 colspan=1>1.10</td><td rowspan=1 colspan=1>6.52</td></tr><tr><td rowspan=1 colspan=1>1.89</td><td rowspan=1 colspan=1>4.25</td></tr><tr><td rowspan=1 colspan=1>0.71</td><td rowspan=1 colspan=1>4.76</td></tr><tr><td rowspan=1 colspan=1>3.00</td><td rowspan=1 colspan=1>14.02</td></tr><tr><td rowspan=4 colspan=1>W/RL</td><td rowspan=4 colspan=1>GIANTS-4BQwen3-8B + TempQwen3-8B + ReActQwen3-8B + Search + Write</td><td rowspan=1 colspan=1>1.62</td><td rowspan=1 colspan=1>35.57</td></tr><tr><td rowspan=1 colspan=1>11.52</td><td rowspan=1 colspan=1>19.82</td></tr><tr><td rowspan=1 colspan=1>24.94</td><td rowspan=1 colspan=1>46.60</td></tr><tr><td rowspan=1 colspan=1>24.47</td><td rowspan=1 colspan=1>33.33</td></tr><tr><td rowspan=2 colspan=1>Ours</td><td rowspan=2 colspan=1>AI NIGHT-SCIENTIST-8BAI NIGHT-SCIENTIST-14B</td><td rowspan=1 colspan=1>29.89</td><td rowspan=1 colspan=1>56.32</td></tr><tr><td rowspan=1 colspan=1>33.18</td><td rowspan=1 colspan=1>68.68</td></tr></table>

(b) Research-idea diversity
<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>Paradigm ↑</td><td rowspan=1 colspan=1>Contri. ↑</td></tr><tr><td rowspan=1 colspan=1>Qwen3-8B Zero-Shot</td><td rowspan=1 colspan=1>0.738</td><td rowspan=1 colspan=1>0.370</td></tr><tr><td rowspan=1 colspan=1>GPT-4.1 Zero-Shot</td><td rowspan=1 colspan=1>0.785</td><td rowspan=1 colspan=1>0.439</td></tr><tr><td rowspan=1 colspan=1>GIANTS-4B (RL)</td><td rowspan=1 colspan=1>0.773</td><td rowspan=1 colspan=1>0.347</td></tr><tr><td rowspan=1 colspan=1>Qwen3 + Temp (RL)</td><td rowspan=1 colspan=1>0.792</td><td rowspan=1 colspan=1>0.339</td></tr><tr><td rowspan=1 colspan=1>Qwen3 + ReAct (RL)</td><td rowspan=1 colspan=1>0.888</td><td rowspan=1 colspan=1>0.390</td></tr><tr><td rowspan=1 colspan=1>Night-8B (Rpo)</td><td rowspan=1 colspan=1>0.853</td><td rowspan=1 colspan=1>0.362</td></tr><tr><td rowspan=1 colspan=1>+ No swap</td><td rowspan=1 colspan=1>0.844</td><td rowspan=1 colspan=1>0.358</td></tr><tr><td rowspan=1 colspan=1>Night-8B $( R _ { \mathbf { o u t } } )$ </td><td rowspan=1 colspan=1>0.943</td><td rowspan=1 colspan=1>0.425</td></tr></table>

(c) Reward ablation
<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>Reward</td><td rowspan=1 colspan=1>Citation ↑</td><td rowspan=1 colspan=1>Original. ↑</td></tr><tr><td rowspan=1 colspan=1>ReAct</td><td rowspan=1 colspan=1> $R _ { \mathrm { o u t } }$  $R _ { \mathrm { p o } }$ </td><td rowspan=1 colspan=1>24.942.10</td><td rowspan=1 colspan=1>46.606.54</td></tr><tr><td rowspan=1 colspan=1>Temp.</td><td rowspan=1 colspan=1> $R _ { \mathrm { o u t } }$  $R _ { \mathrm { p o } }$ </td><td rowspan=1 colspan=1>11/5210.35</td><td rowspan=1 colspan=1>19.8213.33</td></tr><tr><td rowspan=1 colspan=1>Night-8B</td><td rowspan=1 colspan=1> $R _ { \mathrm { o u t } }$  $R _ { \mathrm { p o } }$ </td><td rowspan=1 colspan=1>29.8926.20</td><td rowspan=1 colspan=1>56.3261.61</td></tr></table>

Table 3: Main results for (a) proposal quality, (b) research-idea diversity, and (c) outcome-only $( R _ { \mathrm { o u t } } )$ vs. process + outcome $( R _ { \mathrm { p o } } )$ training. Best results are bolded and second-best are underlined.

12.36 points in originality. Additional model capacity therefore appears substantially more useful once paired with a learned creative reasoning policy.

The contrast with temperature-based exploration is especially evident after RL. Although both optimize the same proposal-level reward, NIGHT-8B exceeds the temperature-controlled variant by 36.50 points in originality and 18.37 points in citation. ReAct closes much of this gap, showing that a learned multi-step policy already helps, but semantic creativity levels still add 9.72 points in originality and 4.95 points in citation. The useful signal is therefore not simply to “explore more,” but to specify how an action should deviate so RL can learn when those deviations are useful.

The action space matters as well. Restricting the agent to search and write reduces originality by 22.99 points despite using the same creativity levels and outcome reward. Retrieval and iterative writing therefore explain only part of the gain; spark and debate provide additional ways to redirect reasoning rather than simply gather more evidence for an existing direction.

Robustness. To verify that these trends are not specific to reconstructed references, we directly compare NIGHT-8B against GPT-4.1 and Qwen3-8B. It wins 74.95% of citation and 86.96% of originality comparisons against GPT-4.1, increasing to 82.48% and 97.56% against Qwen3-8B. The same advantage therefore holds in direct head-to-head comparisons. We also include the confidence interval results under Appendix A.

## 4.4.2 AI NIGHT-SCIENTIST BROADENS RESEARCH IDEA DIVERSITY

Higher originality would be less meaningful if the model repeatedly relied on the same kind of creative strategy. Table 3(b) suggests otherwise. Relative to zero-shot Qwen3-8B, NIGHT-8B increases normalized research-paradigm coverage from 0.738 to 0.943, corresponding to roughly 1.4 additional effective categories out of seven; contribution-type coverage similarly rises from 0.370 to 0.425. The model therefore produces not only more original proposals, but a broader range of research paradigms and contribution types.

Diversity is also sensitive to early exploration. Removing serendipitous action swaps reduces paradigm coverage by 0.099, or roughly 0.7 effective categories, and contribution coverage by 0.067, or roughly 0.5 categories. This pattern is consistent with early exposure to less familiar behaviors helping the policy discover a broader set of useful trajectories.

The improvements are also not concentrated in a small set of CS topics. NIGHT-8B improves over zero-shot Qwen3-8B on both citation and originality across all 36 evaluated domains, including healthcare, biomedical engineering, geosciences, and quantum science. Full per-domain results are provided in Appendix C.

## 4.4.3 PROCESS REWARDS FAVOR ORIGINALITY OVER BREADTH

Table 3(c) shows a different effect from directly rewarding the reasoning process. For NIGHT-8B, process + outcome training raises originality by 5.29 points but lowers predicted citation impact by 3.69 points. Paradigm and contribution coverage also fall by roughly 0.6 and 0.4 effective categories, respectively. Process supervision can therefore favor more original individual proposals without necessarily producing a broader or more impactful set of ideas.

This effect is specific to the creativity-aligned action space. Adding the same process reward to ReAct sharply reduces both citation and originality, while Temperature also declines on both metrics. Process supervision is therefore not uniformly beneficial; its effect depends on whether the action space provides semantically distinct ways to express different degrees of creativity. We consequently use outcome-only training as our primary setting and treat process rewards as a way to shift the model toward greater proposal-level originality. We detail further analysis of $P _ { \mathtt { p o } }$ in Appendix 4.4.6.

## 4.4.4 QUALITATIVE ANALYSIS

We examine one representative NSF topic to understand how proposal directions differ qualitatively across methods. Table 4.4.4 shows a progression from adapting existing lecture-delivery mechanisms toward changing the underlying learning interaction itself. AI NIGHT-SCIENTIST produces the more substantial reframings, although the process + outcome variant also illustrates that greater creative deviation can come at the cost of grounding and methodological rigor.

<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=2>Key Proposal Direction                              Main Limitation</td></tr><tr><td rowspan=1 colspan=1>GPT-4.1</td><td rowspan=1 colspan=1>Uses AI to segment prerecorded lectures and represent thematerial through conversational agents, with adaptive pacing,clarification, and multimodal accessibility support.</td><td rowspan=1 colspan=1>The proposal is coherent and inclusive, but largely com-bines established lecture-segmentation and chatbot mecha-nisms rather than introducing a distinct technical idea.</td></tr><tr><td rowspan=1 colspan=1>ReAct</td><td rowspan=1 colspan=1>Introduces a multimodal engagement score that combinesvisual, auditory, and physiological signals and uses it to adaptlecture pacing and content granularity in real time.</td><td rowspan=1 colspan=1>The score is a concrete artifact, but the proposed RL-basedadaptation is underspecified: the state, action, and rewardspaces are left unclear for how engagement signals translateinto adaptation decisions.</td></tr><tr><td rowspan=1 colspan=1>NIGHT-8B $( R _ { \mathrm { o u t } } )$ </td><td rowspan=1 colspan=1>Proposes Agentify, which turns a prerecorded lecture into alive, agent-mediated session where an AI interleaves ques-tions, hints, and dialogue based on learner behavior.</td><td rowspan=1 colspan=1>The proposal substantially reframes the interaction, butsome components are only loosely motivated or underdevel-oped, including the cognitive-tier co-evolution mechanismand forced speed controls.</td></tr><tr><td rowspan=1 colspan=1>NIGHT-8B $( R _ { \mathrm { p o } } )$ </td><td rowspan=1 colspan=1>Proposes Progressive Content Enactors (PCEs), which delib-erately inject controlled errors and ambiguities for learners todetect and resolve, shifting the lecture from passive deliverytoward active problem solving.</td><td rowspan=1 colspan=1>The central idea is distinctive, but the proposal also con-tains unverifiable quantitative claims and weakly groundedevaluation measures, reducing methodological credibility.</td></tr></table>

Table 4: Qualitative comparison for the NSF topic “Using AI to Transform Online Video Lectures.” Green highlights novel or well-specified contributions; red highlights incremental, underspecified, or questionable elements. Full proposal excerpts and detailed error analysis are provided in Appendix D.

## 4.4.5 SEMANTIC CREATIVITY LEVELS BETTER ALIGN WITH OUTPUT BEHAVIOR

Our creativity levels are intended to change how an action is executed, not merely label it. Figure 3 measures this relationship using output entropy and semantic similarity (Appendix N). AI NIGHT-SCIENTIST (PO) improves on both measures during training, while Temp (PO) deteriorates toward −0.9. Semantic descriptions therefore provide a more learnable link between the selected creativity level and resulting behavior than temperature alone.

![](images/0996e4a97b4552439d003c8c4cd7cd6d46e86d2db1b769fe29c3b259b131c003.jpg)  
Figure 3: Higher correlation indicates better alignment between selected creativity level and output behavior.

## 4.4.6 PROCESS REWARDS SHIFT THE LEARNED ACTION POLICY

Figure 4 shows how the learned policies differ at test time. GPT-4.1 typically follows a short search → write → stop pattern, while Qwen3-8B-Base uses a broader mix of actions across the full trajectory. AI NIGHT-SCIENTIST-8B trained with process+outcome rewards shows a different preference: it selects spark more frequently across turns and tends to use it at moderate-tohigh creativity levels. This suggests that process supervision shifts the policy toward assumptionchallenging actions that more directly open new research directions.

![](images/bfb69cab95e739ba42a22e88fde7d28d56b562696bba1b9ee0695faae40723c2.jpg)  
Figure 4: Test-time action frequencies across five reasoning turns for three model variants.

Debate is selected relatively rarely. One practical reason may be that it requires more of the trajectory budget, since the agent first identifies participants and relevant evidence before generating the grounded discussion. In general, this suggests that the learned value of an action depends not only on its creative potential, but also on how efficiently it contributes within a limited reasoning horizon.

## 5 CONCLUSION

We introduce AI NIGHT-SCIENTIST, a creativity-aligned agentic framework that represents creativity at the action, process, and outcome levels and uses reinforcement learning to teach models when and how to depart from predictable reasoning. Applied to long-horizon research proposal generation, AI NIGHT-SCIENTIST improves predicted citation impact and originality by up to 32.03 and 66.15 points over its base model, while producing a broader range of research paradigms. These gains are not reproduced by higher decoding temperature or ReAct alone, supporting our central finding that creativity is more than sampling stochasticity. Our ablations further show that outcome-only training provides the strongest overall balance of quality and diversity, while process-level rewards can increase originality at the cost of citation impact and breadth. Together, these results suggest that creativity can be learned as a multi-level reasoning capability, enabling scientific agents to move more flexibly between the structured reasoning of day science and the exploratory reasoning of night science while remaining tools for human-led discovery.

## ETHICS STATEMENT

Supporting human-led research. AI NIGHT-SCIENTIST is designed for early-stage scientific ideation, when research directions are still speculative and not yet fully testable. Its role is to help researchers explore a broader set of possible directions, including connections or assumptions they may not otherwise consider. The system is intended as a creative collaborator rather than an autonomous researcher: generated proposals should be treated as candidate ideas that require critical evaluation, refinement, and validation by domain experts before they can support scientific conclusions.

Broadening research exploration. Human ideation is often shaped by the concepts and examples already available in their local research community (Lubart, 2001). By encouraging proposals that depart from existing work while remaining relevant and feasible, AI NIGHT-SCIENTIST aims to surface directions beyond those most immediately accessible to a researcher or model. This may be especially useful for interdisciplinary exploration, where relevant ideas can be distributed across distant literature and research communities. Importantly, broader exploration does not imply that unfamiliar ideas are inherently better; their scientific value must still be established through expert judgment and empirical validation.

Risks and responsible use. As with other generative systems for scientific writing, AI NIGHT-SCIENTIST can produce plausible but incorrect, infeasible, or insufficiently grounded proposals, and could be misused to generate low-quality scientific content at scale. Its outputs should therefore be presented as AI-generated suggestions rather than validated research plans. Responsible use requires transparent disclosure, independent verification of claims and citations, and meaningful human oversight before ideas are pursued, disseminated, or incorporated into scientific work.

## REPRODUCIBILITY STATEMENT

The main paper specifies the action space, creativity levels, reward formulation, baselines, and evaluation metrics. The appendix further provides the complete action and reward prompts (Appendices E-F), training compute and hyperparameters (Appendix M), NSF dataset filtering and domain construction (Appendix B), reconstructed reference-proposal procedure and prompt (Appendix K), and full evaluation protocols for predicted citation impact, originality, and research-idea diversity (Appendices G-I). We additionally report confidence intervals and human-LLM agreement in Appendices A and J. Together, these materials document the data construction, model training, prompting, retrieval, and evaluation procedures needed to reproduce the reported experiments.

## REFERENCES

Osama Mohammed Afzal, Preslav Nakov, Tom Hope, and Iryna Gurevych. Beyond" not novel enough": Enriching scholarly critique with llm-assisted feedback. arXiv preprint arXiv:2508.10795, 2025.

Dhruv Agarwal, Bodhisattwa Prasad Majumder, Reece Adamson, Megha Chakravorty, Satvika Reddy Gavireddy, Aditya Parashar, Harshit Surana, Bhavana Dalvi Mishra, Andrew McCallum, Ashish Sabharwal, et al. Autodiscovery: Open-ended scientific discovery via bayesian surprise. Advances in Neural Information Processing Systems, 38:25181–25219, 2026.

Shivam Agarwal, Zimin Zhang, Lifan Yuan, Jiawei Han, and Hao Peng. The unreasonable effectiveness of entropy minimization in llm reasoning. arXiv preprint arXiv:2505.15134, 2025.

Margaret A Boden. Creativity and artificial intelligence. Artificial intelligence, 103(1-2):347–356, 1998.

Ziyu Chen, Yilun Zhao, and Arman Cohan. Measuring the gap between human and llm research ideas. arXiv preprint arXiv:2607.01233, 2026.

Leonora M Cohen. A continuum of adaptive creative behaviors. Creativity Research Journal, 2(3): 169–183, 1989.

Pier-Luc de Chantal and Henry Markovits. Reasoning outside the box: Divergent thinking is related to logical reasoning. Cognition, 224:105064, 2022.

Christopher P Dwyer, Deaglán Campbell, and Niall Seery. An evaluation of the relationship between critical thinking and creative thinking: Complementary metacognitive processes or strange bedfellows? Journal ofIntelligence, 13(2):23, 2025.

Juraj Gottweis, Wei-Hung Weng, Alexander Daryin, Tao Tu, Anil Palepu, Petar Sirkovic, Artiom Myaskovsky, Felix Weissenberger, Keran Rong, Ryutaro Tanno, et al. Towards an ai co-scientist. arXiv preprint arXiv:2502.18864, 2025.

Tianyang Gu, Jingjin Wang, Zhihao Zhang, and HaoHong Li. Llms can realize combinatorial creativity: generating creative ideas via llms for scientific research. arXiv preprint arXiv:2412.14141, 2024.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Tarun Gupta and Danish Pruthi. All that glitters is not novel: Plagiarism in ai generated research. arXiv preprint arXiv:2502.16487, 2025.

Diane F Halpern. The nature and nurture of critical thinking. Critical thinking in psychology, (1): 1–14, 2007.

Sarah Harvey and James W Berry. Toward a meta-theory of creativity forms: How novelty and usefulness shape creativity. Academy ofManagement Review, 48(3):504–529, 2023.

Joy He-Yueya, Anikait Singh, Ge Gao, Michael Y Li, Sherry Yang, Chelsea Finn, Emma Brunskill, and Noah D Goodman. Giants: Generative insight anticipation from scientific literature. arXiv preprint arXiv:2604.09793, 2026.

Jules Hirsch. An anniversary for cancer chemotherapy. Jama, 296(12):1518–1520, 2006.

Priyanka Kargupta, Ishika Agarwal, Tal August, and Jiawei Han. Tree-of-debate: Multi-persona debate trees elicit critical thinking for scientific comparative analysis. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 29378–29403, 2025a.

Priyanka Kargupta, Shuyue Stella Li, Haocheng Wang, Jinu Lee, Shan Chen, Orevaoghene Ahia, Dean Light, Thomas L Griffiths, Max Kleiman-Weiner, Jiawei Han, et al. Cognitive foundations for reasoning and their manifestation in llms. arXiv preprint arXiv:2511.16660, 2025b.

Priyanka Kargupta, Runchu Tian, and Jiawei Han. Beyond true or false: Retrieval-augmented hierarchical analysis of nuanced claims. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 29664–29679, 2025c.

Priyanka Kargupta, Shuhaib Mehri, Dilek Hakkani-Tur, and Jiawei Han. Sparking scientific creativity via llm-driven interdisciplinary inspiration. arXiv preprint arXiv:2603.12226, 2026.

B Lee Ligon. Penicillin: its discovery and early development. In Seminars in pediatric infectious diseases, volume 15, pp. 52–57. Elsevier, 2004.

Fang Liu, Yang Liu, Lin Shi, Houkun Huang, Ruifeng Wang, Zhen Yang, Li Zhang, Zhongqi Li, and Yuchi Ma. Exploring and evaluating hallucinations in llm-powered code generation. arXiv preprint arXiv:2404.00971, 2024.

Li-Chun Lu, Shou-Jen Chen, Tsung-Min Pai, Chan-Hung Yu, Hung-yi Lee, and Shao-Hua Sun. Llm discussion: Enhancing the creativity of large language models via discussion framework and role-play. arXiv preprint arXiv:2405.06373, 2024.

Todd I Lubart. Models of the creative process: Past, present and future. Creativity research journal, 13(3-4):295–308, 2001.

R Thomas McCoy, Shunyu Yao, Dan Friedman, Mathew D Hardy, and Thomas L Griffiths. Embers of autoregression show how large language models are shaped by the problem they are trained to solve. Proceedings ofthe National Academy ofSciences, 121(41):e2322420121, 2024.

Elspeth McFadzean. The creativity continuum: Towards a classification of creative problem solving techniques. Creativity and Innovation Management, 7(3):131–139, 1998.

Sarnoff Mednick. The associative basis of the creative process. Psychological review, 69(3):220, 1962.

D Anthony Miles. A taxonomy of research gaps: Identifying and defining the seven research gaps. In Doctoral student workshop: finding research gaps-research methods and strategies, Dallas, Texas, volume 1, pp. 1–10, 2017.

Bernard A Nijstad, Carsten KW De Dreu, Eric F Rietzschel, and Matthijs Baas. The dual pathway to creativity model: Creative ideation as a function of flexibility and persistence. European review of social psychology, 21(1):34–77, 2010.

Charles O’Neill, Tirthankar Ghosal, Roberta Raileanu, Mike Walmsley, Thang Bui, Kevin Schawinski,˘ and Ioana Ciuca. Sparks of science: Hypothesis generation using structured paper data.˘ arXiv preprint arXiv:2504.12976, 2025.

Nirav Patel, Josiah Crossman, Eva Aggarwal, and Emily Wenger. Are llms becoming similarly creative? evidence from three years of models. arXiv preprint arXiv:2608.19437, 2026.

Mark A Runco and Garrett J Jaeger. The standard definition of creativity. Creativity research journal, 24(1):92–96, 2012.

Jürgen Schmidhuber. Formal theory of creativity, fun, and intrinsic motivation (1990–2010). IEEE transactions on autonomous mental development, 2(3):230–247, 2010.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. arXiv preprint arXiv: 2409.19256, 2024.

Chenglei Si, Diyi Yang, and Tatsunori Hashimoto. Can llms generate novel research ideas? a large-scale human study with 100+ nlp researchers. In Y. Yue, A. Garg, N. Peng, F. Sha, and R. Yu (eds.), International Conference on Learning Representations, volume 2025, pp. 94003–94092, 2025. URL https://proceedings.iclr.cc/paper\_files/paper/2025/file/ ea94957d81b1c1caf87ef5319fa6b467-Paper-Conference.pdf.

Gunther S Stent. The statue within: An autobiography. Science, 239(4847):1545–1547, 1988.

Jingqi Tong, Mingzhe Li, Hangcheng Li, Yongzhuo Yang, Yurong Mou, Weijie Ma, Zhiheng Xi, Hongji Chen, Xiaoran Liu, Qinyuan Cheng, et al. Ai can learn scientific taste. arXiv preprint arXiv:2603.14473, 2026.

Qingyun Wang, Doug Downey, Heng Ji, and Tom Hope. Scimon: Scientific inspiration machines optimized for novelty. In Proceedings ofthe 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pp. 279–299, 2024.

Yiping Wang, Qing Yang, Zhiyuan Zeng, Liliang Ren, Liyuan Liu, Baolin Peng, Hao Cheng, Xuehai He, Kuan Wang, Jianfeng Gao, et al. Reinforcement learning for reasoning in large language models with one training example. arXiv preprint arXiv:2504.20571, 2025.

Thomas B Ward, Steven M Smith, and Ronald A Finke. Creative cognition. Handbook ofcreativity, 189:212, 1999.

Solange Muglia Wechsler, Carlos Saiz, Silvia F Rivas, Claudete Maria Medeiros Vendramini, Leandro S Almeida, Maria Celia Mundim, and Amanda Franco. Creative and critical thinking: Independent or overlapping components? Thinking skills and creativity, 27:114–122, 2018.

Emily Wenger and Yoed Kenett. We’re different, we’re the same: Creative homogeneity across llms. arXiv preprint arXiv:2501.19361, 2025.

Jacob O Wobbrock. Seven research contributions in hci. Intelligence, 174(12-13):910–950, 2012.

Itai Yanai and Martin Lercher. Night science. Genome Biology, 20(1):179, 2019.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. ReAct: Synergizing reasoning and acting in language models. In International Conference on Learning Representations (ICLR), 2023.

Yunpu Zhao, Rui Zhang, Wenyi Li, and Ling Li. Assessing and understanding creativity in large language models. Machine Intelligence Research, 22(3):417–436, 2025.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. Judging llm-as-a-judge with mt-bench and chatbot arena. Advances in neural information processing systems, 36:46595–46623, 2023.

Li Zhong and Zilong Wang. Can llm replace stack overflow? a study on robustness and reliability of large language model code generation. In Proceedings of the AAAI conference on artificial intelligence, volume 38, pp. 21841–21849, 2024.

Table 5: Results with 95% confidence intervals. Citation and originality are pairwise win rates against matched reconstructed references; diversity metrics report normalized effective category coverage.
<table><tr><td></td><td colspan="2">Proposal Quality</td><td colspan="2">Research-Idea Diversity</td></tr><tr><td>Method</td><td>Citation (%)</td><td>Originality (%)</td><td>Paradigm</td><td>Contribution</td></tr><tr><td colspan="5">Zero-shot</td></tr><tr><td>Qwen3-8B</td><td>0.46 [0.13, 1.66]</td><td>2.53 [1.42, 4.47]</td><td>0.738 [0.682, 0.785]</td><td>0.370 [0.340, 0.399]</td></tr><tr><td>GPT-4.1</td><td>3.91 [2.45, 6.17]</td><td>12.18 [9.44, 15.59]</td><td>0.785 [0.733, 0.826]</td><td>0.439 [0.401, 0.472]</td></tr><tr><td colspan="5">RL Baselines</td></tr><tr><td>Qwen3-8B + Temp</td><td>11.52 [8.85, 14.87]</td><td>19.82 [16.34, 23.82]</td><td>0.792 [0.742, 0.831]</td><td>0.339 [0.313, 0.364]</td></tr><tr><td>Qwen3-8B + ReAct</td><td>24.94 [20.93, 29.42]</td><td>46.60 [41.75, 51.52]</td><td>0.888 [0.847, 0.916]</td><td>0.390 [0.366, 0.412]</td></tr><tr><td colspan="5">AI NIGHT-SCIENTIST</td></tr><tr><td>NIGHT-8B  $( R _ { \mathrm { p o } } )$ </td><td>26.21 [22.30, 30.53]</td><td>61.61 [56.96, 66.06]</td><td>0.853 [0.807, 0.887]</td><td>0.362 [0.338, 0.384]</td></tr><tr><td>NIGHT-8B  $\left( R _ { \mathbf { 0 u t } } \right)$ </td><td>29.89 [25.77, 34.35]</td><td>56.32 [51.63, 60.91]</td><td>0.943 [0.907, 0.962]</td><td>0.425 [0.412, 0.435]</td></tr></table>

## A CONFIDENCE INTERVALS FOR MAIN RESULTS

Table 5 reports 95% confidence intervals for the primary proposal-quality and research-idea diversity results shown in Table 3. Citation and originality are pairwise win rates against matched reconstructed reference proposals; their intervals are Wilson score intervals over the evaluated proposal pairs, using the same treatment of ties and undecided judgments as the reported win rates. Paradigm and contribution diversity are measured as the normalized effective number of categories represented, exp(H)/K, where H is Shannon entropy and K = 7; their intervals are 95% percentile intervals from 100,000 proposal-level bootstrap resamples within each method. These intervals quantify evaluation-sample uncertainty only.

## B DATASET CONSTRUCTION & DOMAIN COVERAGE

## B.1 FILTERING AND SPLITS

We select NSF CSE awards from 2018 onwards. Non-research grants (e.g., travel and conference grants) are excluded. This yields 4,905 awards in total, split 90:10 into 4,414 training and 491 test instances. We focus on CSE awards to maximize the availability of open-access literature on arXiv, which underpins both the search action and the retrieval steps in the reward pipeline.

## B.2 DOMAIN DISTRIBUTION

To characterize the topical breadth of the training set, we applied an automated three-stage taxonomy pipeline to the 4,414 NSF award titles. In the first stage, GPT-4.1 assigned one to three fine-grained research subfields to each title in batches of 50, yielding 11,071 raw subfield labels (2.5 per title on average) spanning 4,300 unique terms. In the second stage, these labels were grouped into intermediate clusters via a second LLM pass. In the third stage, the intermediate clusters were consolidated into a final canonical taxonomy of 46 research domains (Table 6).

The distribution reflects the intended scope of the NSF CISE directorate: Artificial Intelligence & Machine Learning accounts for 23.1% of all subfield label occurrences, followed by Computer Science Theory (11.2%) and Networks & Communications (5.9%). The concentration in core CS and engineering is expected, especially as open-access literature on arXiv is most comprehensive for these fields, making retrieval-grounded proposal generation most reliable there. Importantly, the training set is not exclusively CS-focused: the remaining ∼20% of labels span applied and interdisciplinary domains, including Biomedical Engineering & Health Informatics (2.7%), Cognitive & Behavioral Sciences (2.2%), Earth & Geosciences (1.0%), Quantum Science & Engineering (0.9%), and policy adjacent areas such as Ethics, Equity & Societal Impacts and Economics, Policy & Law, providing a diverse training signal across 46 domains in total.

Table 6: Distribution of research domains across training proposals, grouped into high-level domain families. Fine-grained labels are derived from a three-stage automated taxonomy pipeline applied to 4,414 NSF award titles; high-level families are used only to organize the labels for presentation.
<table><tr><td>Domain</td><td>Count</td><td>% of Labels</td></tr><tr><td>Computing &amp; AI</td><td>6,418</td><td>58.0%</td></tr><tr><td>Artificial Intelligence &amp; Machine Learning</td><td>2558</td><td>23.1</td></tr><tr><td>Networks &amp; Communications</td><td>657</td><td>5.9</td></tr><tr><td>Computer Systems &amp; Architecture</td><td>626</td><td>5.7</td></tr><tr><td>Data Science &amp; Analytics</td><td>611</td><td>5.5</td></tr><tr><td>Cybersecurity &amp; Privacy</td><td>547</td><td>4.9</td></tr><tr><td>Robotics &amp; Autonomous Systems</td><td>455</td><td>4.1</td></tr><tr><td>Embedded &amp; Hardware Systems</td><td>310</td><td>2.8</td></tr><tr><td>Cloud &amp; Distributed Computing</td><td>277</td><td>2.5</td></tr><tr><td>Software Engineering &amp; Programming Languages</td><td>164</td><td>1.5</td></tr><tr><td>Natural Language Processing &amp; Linguistics</td><td>157</td><td>1.4</td></tr><tr><td>Extended Reality &amp; Immersive Technologies</td><td>34</td><td>0.3</td></tr><tr><td>Sustainable Computing &amp; Infrastructure</td><td>22</td><td>0.2</td></tr><tr><td>Mathematical &amp; Computational Foundations</td><td></td><td></td></tr><tr><td>Computer Science Theory</td><td>1,942</td><td>17.5%</td></tr><tr><td>Operations Research &amp; Optimization</td><td>1239 171</td><td>11.2 1.5</td></tr><tr><td>Modeling, Simulation &amp; Visualization</td><td>152</td><td>1.4</td></tr><tr><td>Mathematics &amp; Theoretical Foundations</td><td>123</td><td>1.1</td></tr><tr><td>Experimental Methods &amp; Evaluation</td><td>100</td><td>0.9</td></tr><tr><td>Statistics &amp; Data Science</td><td>80</td><td>0.7</td></tr><tr><td>Systems Science &amp; Engineering</td><td>77</td><td>0.7</td></tr><tr><td>Human, Social &amp; Educational Research</td><td></td><td></td></tr><tr><td>Human-Computer Interaction</td><td>1,344 538</td><td>12.1% 4.9</td></tr><tr><td>Cognitive &amp; Behavioral Sciences</td><td>241</td><td>2.2</td></tr><tr><td>Education Research &amp; Pedagogy</td><td>236</td><td>2.1</td></tr><tr><td>Computational Social Sciences &amp; Digital Humanities</td><td>173</td><td>1.6</td></tr><tr><td>Career Development &amp; Workforce</td><td>50</td><td>0.5</td></tr><tr><td>Ethics, Equity &amp; Societal Impacts</td><td>47</td><td>0.4</td></tr><tr><td>Organizational Computing &amp; Workflow</td><td>25</td><td>0.2</td></tr><tr><td>Economics, Policy &amp; Law</td><td>22</td><td>0.2</td></tr><tr><td>Legal Informatics &amp; Law</td><td>7</td><td>0.1</td></tr><tr><td>Media, Creativity &amp; Inclusive Technologies</td><td>5</td><td>0.0</td></tr><tr><td>Health &amp; Life Sciences</td><td></td><td></td></tr><tr><td>Biomedical Engineering &amp; Health Informatics</td><td>543</td><td>4.9%</td></tr><tr><td>Biological &amp; Biomedical Sciences</td><td>295 145</td><td>2.7 1.3</td></tr><tr><td>Healthcare &amp; Medical Sciences</td><td>103</td><td>0.9</td></tr><tr><td>Engineering &amp; Physical Sciences</td><td></td><td></td></tr><tr><td>Imaging, Instrumentation &amp; Sensors</td><td>532 122</td><td>4.8% 1.1</td></tr><tr><td>Materials Science &amp; Nanotechnology</td><td>103</td><td>0.9</td></tr><tr><td>Quantum Science &amp; Engineering</td><td>97</td><td>0.9</td></tr><tr><td>Electrical &amp; Electronic Engineering</td><td>63</td><td>0.6</td></tr><tr><td>Energy, Power &amp; Automotive Systems</td><td>46</td><td>0.4</td></tr><tr><td>Safety, Risk &amp; Resilience Engineering</td><td>45</td><td>0.4</td></tr><tr><td>Physical Sciences</td><td>42</td><td>0.4</td></tr><tr><td>Chemistry &amp; Materials Science</td><td>14</td><td>0.1</td></tr><tr><td>Earth, Environment &amp; Urban Systems</td><td>193</td><td>1.7%</td></tr><tr><td>Earth &amp; Geosciences</td><td>116</td><td>1.0</td></tr><tr><td>Agricultural &amp; Environmental Sciences</td><td>31</td><td>0.3</td></tr><tr><td>Urban Informatics &amp; Smart Cities</td><td>26</td><td>0.2</td></tr><tr><td>Environmental Informatics &amp; Sensing</td><td>20</td><td>0.2</td></tr><tr><td></td><td></td><td></td></tr><tr><td>Research Infrastructure &amp; Other Open Science &amp; Research Infrastructure</td><td>99 76</td><td>0.9% 0.7</td></tr><tr><td>Miscellaneous</td><td>23</td><td>0.2</td></tr><tr><td>Total</td><td>11,071</td><td>100.0%</td></tr></table>

## C PERFORMANCE ACROSS RESEARCH DOMAINS

We report per-domain win rates for AI NIGHT-SCIENTIST-8B and Qwen3-8B zero-shot against the reference proposals across 36 canonical research domains (Table 7). The zero-shot baseline achieves near-zero impact win rates in almost every domain (overall 0.5%) and near-zero originality win rates (overall 2.5%), confirming that an untuned 8B model cannot produce proposals competitive with actual NSF awards on either metric. AI NIGHT-SCIENTIST-8B consistently overcomes this gap, delivering positive impact gains in all 34 domains with sufficient decided pairs, ranging from +8 percentage points (pp) (Materials Science & Nanotechnology) to +63 pp (Environmental Informatics & Sensing). Originality gains are uniformly large across all 36 domains, ranging from +37 pp (Natural Language Processing & Linguistics, Energy, Power & Automotive Systems) to +83 pp (Organizational Computing & Workflow), with particularly strong improvements in interdisciplinary and application-oriented fields such as Quantum Science & Engineering $( + 8 0 . 0 \pm 1 7 . 9 \mathrm { p p } )$ , Safety, Risk & Resilience Engineering (+71.4 ± 12.1 pp), Earth & Geosciences (+71.4 ± 17.1 pp), and Computational Social Sciences & Digital Humanities $( + 7 0 . 6 \pm 1 1 . 1 \mathrm { p p } )$

<table><tr><td>Domain</td><td>Citation Gain (pp)</td><td>Originality Gain (pp)</td></tr><tr><td>Computing &amp; AI</td><td></td><td></td></tr><tr><td>Networks &amp; Communications</td><td> $+ 4 0 . 0 \pm 7 . 3$ </td><td> $+ 5 9 . 6 \pm 7 . 2$ </td></tr><tr><td>Robotics &amp; Autonomous Systems</td><td> $+ 3 4 . 0 \pm 6 . 9$ </td><td> $+ 5 7 . 4 \pm 7 . 7$ </td></tr><tr><td>Artificial Intelligence &amp; Machine Learning</td><td> $+ 2 9 . 6 \pm 3 . 4$ </td><td> $+ 5 7 . 0 \pm 3 . 7$ </td></tr><tr><td>Cybersecurity &amp; Privacy</td><td> $+ 4 3 . 2 \pm 7 . 5$ </td><td> $+ 5 5 . 6 \pm 7 . 7$ </td></tr><tr><td>Computer Systems &amp; Architecture</td><td> $+ 3 0 . 6 \pm 7 . 0$ </td><td> $+ 5 4 . 0 \pm 7 . 5$ </td></tr><tr><td>Data Science &amp; Analytics</td><td> $+ 2 8 . 6 \pm 7 . 6$ </td><td> $+ 5 0 . 0 \pm 9 . 1$ </td></tr><tr><td>Cloud &amp; Distributed Computing</td><td> $+ 4 2 . 9 \pm 9 . 4$ </td><td> $+ 5 0 . 0 \pm 9 . 1$ </td></tr><tr><td>Embedded &amp; Hardware Systems</td><td> $+ 3 6 . 1 \pm 1 1 . 5$ </td><td> $+ 4 7 . 8 \pm 1 1 . 3$ </td></tr><tr><td>Software Engineering &amp; Programming Languages</td><td> $+ 2 3 . 1 \pm 1 1 . 7$ </td><td> $+ 4 6 . 2 \pm 1 3 . 8$ </td></tr><tr><td>Natural Language Processing &amp; Linguistics</td><td> $+ 2 2 . 2 \pm 9 . 8$ </td><td> $+ 3 6 . 8 \pm 1 1 . 1$ </td></tr><tr><td>Mathematical &amp; Computational Foundations</td><td></td><td></td></tr><tr><td>Systems Science &amp; Engineering</td><td> $+ 6 0 . 0 \pm 2 1 . 9$ </td><td> $+ 6 0 . 0 \pm 2 5 . 3$ </td></tr><tr><td>Statistics &amp; Data Science</td><td> $+ 1 4 . 3 \pm 1 3 . 2$ </td><td> $+ 5 7 . 1 \pm 1 8 . 7$ </td></tr><tr><td>Mathematics &amp; Theoretical Foundations</td><td> $+ 2 3 . 5 \pm 1 0 . 3$ </td><td> $+ 5 2 . 9 \pm 1 3 . 2$ </td></tr><tr><td>Computer Science Theory</td><td> $+ 2 6 . 3 \pm 4 . 7$ </td><td> $+ 4 9 . 0 \pm 5 . 6$ </td></tr><tr><td>Modeling, Simulation &amp; Visualization</td><td> $+ 3 1 . 2 \pm 1 1 . 6$ </td><td> $+ 4 7 . 1 \pm 1 3 . 4$ </td></tr><tr><td>Experimental Methods &amp; Evaluation</td><td> $+ 3 6 . 4 \pm 1 4 . 5$ </td><td> $+ 4 5 . 5 \pm 1 5 . 0$ </td></tr><tr><td>Operations Research &amp; Optimization</td><td> $+ 2 5 . 0 \pm 1 2 . 5$ </td><td> $+ 3 8 . 5 \pm 1 3 . 5$ </td></tr><tr><td>Human, Social &amp; Educational Research</td><td></td><td></td></tr><tr><td>Organizational Computing &amp; Workflow</td><td> $+ 2 0 . 0 \pm 1 7 . 9$ </td><td> $+ 8 3 . 3 \pm 1 5 . 2$ </td></tr><tr><td>Computational Social Sciences &amp; Digital Humanities</td><td> $+ 1 8 . 8 \pm 9 . 8$ </td><td> $+ 7 0 . 6 \pm 1 1 . 1$ </td></tr><tr><td>Ethics, Equity &amp; Societal Impacts</td><td> $+ 0 . 0 \pm 0 . 0$ </td><td> $+ 6 6 . 7 \pm 1 5 . 7$ </td></tr><tr><td>Education Research &amp; Pedagogy</td><td> $+ 2 1 . 4 \pm 1 1 . 0$ </td><td> $+ 6 4 . 3 \pm 1 3 . 9$ </td></tr><tr><td>Cognitive &amp; Behavioral Sciences</td><td> $+ 4 7 . 1 \pm 1 2 . 1$ </td><td> $+ 5 7 . 9 \pm 1 1 . 3$ </td></tr><tr><td>Human-Computer Interaction</td><td> $+ 2 9 . 3 \pm 7 . 1$ </td><td> $+ 5 1 . 2 \pm 8 . 2$ </td></tr><tr><td>Career Development &amp; Workforce</td><td> $+ 0 . 0 \pm 0 . 0$ </td><td> $+ 5 0 . 0 \pm 2 0 . 4$ </td></tr><tr><td>Health &amp; Life Sciences</td><td></td><td></td></tr><tr><td>Biomedical Engineering &amp; Health Informatics</td><td> $+ 4 1 . 2 \pm 1 1 . 9$ </td><td> $+ 6 1 . 1 \pm 1 1 . 5$ </td></tr><tr><td>Healthcare &amp; Medical Sciences</td><td> $+ 4 3 . 8 \pm 1 2 . 4$ </td><td> $+ 5 6 . 2 \pm 1 3 . 5$ </td></tr><tr><td>Biological &amp; Biomedical Sciences</td><td> $+ 2 5 . 0 \pm 1 2 . 5$ </td><td> $+ 5 0 . 0 \pm 1 4 . 4$ </td></tr><tr><td>Engineering &amp; Physical Sciences</td><td></td><td></td></tr><tr><td>Quantum Science &amp; Engineering</td><td></td><td> $+ 8 0 . 0 \pm 1 7 . 9$ </td></tr><tr><td>Safety, Risk &amp; Resilience Engineering</td><td> $+ 3 5 . 7 \pm 1 2 . 8$ </td><td> $+ 7 1 . 4 \pm 1 2 . 1$ </td></tr><tr><td>Materials Science &amp; Nanotechnology</td><td> $+ 7 . 7 \pm 7 . 4$ </td><td> $+ 6 4 . 3 \pm 1 3 . 9$ </td></tr><tr><td>Imaging, Instrumentation &amp; Sensors</td><td> $+ 3 5 . 3 \pm 1 1 . 6$ </td><td> $+ 6 1 . 1 \pm 1 1 . 5$ </td></tr><tr><td>Chemistry &amp; Materials Science</td><td></td><td> $+ 6 0 . 0 \pm 2 1 . 9$ </td></tr><tr><td>Energy, Power &amp; Automotive Systems</td><td> $+ 5 0 . 0 \pm 1 7 . 7$ </td><td> $+ 3 7 . 5 \pm 1 7 . 1$ </td></tr><tr><td>Earth, Environment &amp; Urban Systems</td><td></td><td></td></tr><tr><td>Earth &amp; Geosciences</td><td> $+ 1 4 . 3 \pm 1 3 . 2$ </td><td> $+ 7 1 . 4 \pm 1 7 . 1$ </td></tr><tr><td>Environmental Informatics &amp; Sensing</td><td> $+ 6 2 . 5 \pm 1 7 . 1$ </td><td> $+ 5 0 . 0 \pm 1 7 . 7$ </td></tr><tr><td>Research Infrastructure &amp; Other</td><td></td><td></td></tr><tr><td>Open Science &amp; Research Infrastructure</td><td> $+ 2 5 . 0 \pm 1 2 . 5$ </td><td> $+ 5 8 . 3 \pm 1 5 . 8$ </td></tr></table>

Table 7: Domain-level gains of NIGHT-8B $( R _ { \mathrm { o u t } } )$ over Qwen3-8B zero-shot, grouped into the same high-level domain families as Table 6. Gains are differences in pairwise win rate (percentage points); uncertainty is propagated as $\sqrt { \sigma _ { \mathrm { N i g h t } } ^ { 2 } + \sigma _ { \mathrm { z s } } ^ { 2 } }$ . Domains within each family are ordered by originality gain. “—” indicates insufficient samples (< 5 pairs) for a domain-level estimate.

Core CS domains (AI & ML, Computer Science Theory, Robotics) show more moderate but still substantial originality gains (+49–+57 pp), possibly reflecting that these are the most prevalent domains within the base model’s pretraining distribution. The broad consistency of gains across all 36 domains (including fields with little CS overlap such as Biomedical Engineering, Healthcare, Materials Science, and Earth Sciences) suggests that the improvements are not confined to a narrow research subdomain.

## D QUALITATIVE ANALYSIS

Table 8: Qualitative comparison for the NSF topic “Using AI to Transform Online Video Lectures.”
<table><tr><td rowspan=1 colspan=2>Method         Key Proposal Excerpt</td><td rowspan=1 colspan=1>Assessment</td></tr><tr><td rowspan=1 colspan=1>GPT-4.1</td><td rowspan=1 colspan=1>&quot;Develop AI algorithms to automatically segment video lectures intocoherent topics and extract key instructional elements. Design andimplement conversational agents that re-present segmented lecturecontent interactively, supporting adaptive pacing, clarifications, andmultimodal delivery (text, visuals, sign language, audio descriptions).Empirically evaluate effectiveness and inclusivity against standardvideo lectures. Assess scalability and real-world deployment chal-lenges within existing platforms.&quot;</td><td rowspan=1 colspan=1>Pros: Well-structured four-phase plan; cov-ers accessibility and inclusivity.Cons: Restates existing segmentation andchatbot approaches without introducing anew mechanism; no novel technical contri-bution beyond integration.</td></tr><tr><td rowspan=1 colspan=1>ReAct</td><td rowspan=1 colspan=1>“Develop a multimodal AI framework for real-time content adap-tation leveraging visual, auditory, and physiological engagementmetrics to dynamically adjust lecture content. Introduces a multi-modal engagement score&#x27; synthesizing data across modalities. Pairedwith a content adaptation engine using reinforcement learning tooptimize lecture pacing and content granularity [no state, action, orreward space defined]. Validated via controlled A/B study with 120participants across three engagement conditions.&quot;</td><td rowspan=1 colspan=1>Pros: Concrete novel artifact (multimodalengagement score); controlled experimentaldesign.Cons: RL formulation is underspecified; ex-perimental design is elaborate relative to thedegree of technical novelty.</td></tr><tr><td rowspan=1 colspan=1>AI    NIGHT-SCIENTIST(Outcome)</td><td rowspan=1 colspan=1>“Agentify&#x27;: a framework that transforms passive lectures into live,agent-mediated sessions where AI agents interweave structured ques-tions, personalized hints, and interactive dialogues based on real-timeparticipant behavior. Proposes co-evolution of agent templates andlecture cognitive tiers. Randomizes 200 learners across 40 lectures(STEM, Social Science, Humanities) into four groups comparingpassive, over-moderated, fixed-tier, and adaptive-tier conditions. In-cludes a forced-speed-slider component with 8 granular speed set-tings (1x-8x) to calibrate attention span variability.&quot;</td><td rowspan=1 colspan=1>Pros: Novel reframing of lectures as co-creative sessions rather than content deliveryartifacts. Central hypothesis is testable. Am-bitious but sensible multi-group experimen-tal design.Cons: Some design elements appear con-trived (speed-slider rationale). Cognitive tierco-evolution mechanism is underdeveloped.</td></tr><tr><td rowspan=1 colspan=1>AI    NIGHT-SCIENTIST(Process     +Outcome)</td><td rowspan=1 colspan=1>“Progressive Content Enactors (PCEs): virtual agents that shiftfrom automated fidelity to intentional pedagogical maladaptation—deliberately embedding controlled errors and ambiguity so that learn-ers are forced to detect and resolve them, promoting active problem-solving over passive reception. Uses a Layered Semantic Trans-formative Stack (LISS) to anchor distortions to content structure.Reports cognitive engagement score (NCIR) of c=0.12, +3.5σ re-tention improvement (TAR-film, 50:1 decay), and a 2.78-σ increasein credibility-based efficacy—none of which are independently veri-fiable or grounded in standard evaluation frameworks.&quot;</td><td rowspan=1 colspan=1>Pros: Most novel direction: purposefulerrors as a pedagogical tool mirrors well-established ideas (e.g., error-based learning,productive failure). PCE is a fresh, action-able concept that is clearly differentiatedfrom prior work.Cons: Background section relies on unverifi-able metrics and internally inconsistent cita-tions. Evaluation framework is not groundedin standard methodology.</td></tr></table>

We qualitatively compare proposals on the NSF topic “Using Artificial Intelligence to Transform Online Video Lectures into Effective and Inclusive Agent-Based Presentations.” GPT-4.1 produces a well-structured but generic proposal that restates existing approaches (e.g., lecture segmentation and conversational agents) without introducing a distinguishing mechanism. ReAct adds a concrete multimodal engagement score but leaves its reinforcement learning formulation underspecified. AI NIGHT-SCIENTIST-8B (outcome-only) introduces the novel Agentify concept—treating lectures as live, co-created sessions in which an AI agent interleaves questions and dialogues in real time, rather than optimizing pre-recorded content after the fact. The process-reward variant proposes Progressive Content Enactors (PCEs), which deliberately introduce controlled errors and ambiguities into lecture to force students to actively resolve them (analogous to how working through a flawed proof teaches more than reading a perfect one), though its background section overreaches with unverifiable metrics. Table 8 provides excerpts and error analysis; key novel or well-specified contributions are highlighted in green, while vague, incremental, or questionable claims are highlighted in red.

## E REWARD IMPLEMENTATION DETAILS

This appendix provides the complete prompts used within the outcome-level reward pipeline described in Section 3.3.1. The reward is computed in three stages: (1) decomposing the final proposal into atomic ideas, (2) scoring each idea for originality and relevance, and (3) scoring the execution plan for feasibility against retrieved literature.

Table 9: Creativity level descriptions for each action in the research proposal generation task. The write and complete actions are not creativity-enabled (fixed at level 1).
<table><tr><td>C</td><td>Description</td></tr><tr><td>search</td><td></td></tr><tr><td>1</td><td>Extremely relevant to the proposal, focusing on background information and related works based on terms extracted from the proposal title.</td></tr><tr><td>2</td><td>Very relevant, focusing on background and related works closely related to the proposal idea.</td></tr><tr><td>3</td><td>Somewhat relevant, focusing on background that is tangentially related to the proposal idea.</td></tr><tr><td>4</td><td>Not very relevant; explores broader topics or concepts that may not be directly related but still provide useful context.</td></tr><tr><td>5</td><td>Very distantly relevant; explores specific alternate perspectives, domains, or philosophical questions of general interest.</td></tr><tr><td colspan="2">debate</td></tr><tr><td>1</td><td>Discussion with a colleague in the same specific area; structured single-turn debate focused tightly on proposal elements.</td></tr><tr><td>2</td><td>Discussion with a colleague on related topics; structured debate focused on the proposal with minor deviation permitted.</td></tr><tr><td>3</td><td>Debate with a peer from the same high-level field but a different topic; open-ended multi-turn format with some deviation.</td></tr><tr><td>4</td><td>Debate with a domain-expert from a different discipline or a mixture of 2–3 experts; open-ended with broader thematic scope.</td></tr><tr><td>5</td><td>Debate with an expert from a completely different discipline or a mixture of domain-experts; unstruc- tured, focused on exploring diverse perspectives and high-level ideas.</td></tr><tr><td colspan="2">spark</td></tr><tr><td></td><td>A minor conventional idea to challenge, very specific to the proposal&#x27;s current state.</td></tr><tr><td>I</td><td>A minor conventional idea to challenge; somewhat broad but still focused on the proposal.</td></tr><tr><td>2</td><td>A moderate conventional idea to challenge; meaningful change questioning existing assumptions.</td></tr><tr><td>3 4</td><td></td></tr><tr><td>5</td><td>A significant conventional idea to challenge; broad and exploratory, with potential field-level impact. A major conventional idea to challenge; very broad, with the potential to revolutionize the field.</td></tr></table>

## E.1 ACTION SPACE AND CREATIVITY LEVELS

Table 9 summarizes the creativity level descriptions $( \mathbb { C } _ { a } )$ for each creativity-enabled action.

## E.2 PRECEDENCE REWARD PROMPT

Each decomposed idea is scored for precedence against a paper retrieved from arXiv using the idea’s keyword phrase.

You are a reviewer for NSF proposals who has been a professor in your   
field for over 20 years. Evaluate the precedence of a proposal’s core   
idea relative to the problem statement and a related paper.   
Problem statement: {problem}   
Related paper: {related\_paper}   
Evaluate the following core idea for its precedence. Consider how   
it introduces new concepts, methodologies, or perspectives that   
differentiate it from the related paper or existing work in the field.   
Idea: {idea}   
Score from 1 to 5:   
1 = Not novel. Rehash of existing work. 2 = Slightly novel. Largely   
derivative.   
3 = Moderately novel. 4 = Very novel. Potential to advance the   
field.   
5 = Highly novel. Groundbreaking; challenges existing paradigms.   
Output JSON: {"score": <int>, "explanation": "<string>"}

## E.3 FEASIBILITY REWARD PROMPT

Each idea’s execution plan is scored for feasibility against a retrieved paper grounding the assessment in existing methodological precedent.

You are a reviewer for NSF proposals who has been a professor in your   
field for over 20 years. Evaluate the feasibility of a given core   
idea from a proposal. A related paper is provided to contextualize   
the assessment.   
Problem statement: {problem}   
Related paper: {related\_paper}   
Evaluate the following implementation idea for its feasibility:   
practicality, resources required, complexity, ethical implications,   
and specificity. A highly feasible proposal includes specific   
technical details; vague ideas should be penalized.   
Implementation idea: {idea}   
Score from 1 to 5:   
1 = Not feasible. 2 = Slightly feasible. 3 = Moderately feasible.   
4 = Very feasible. 5 = Highly feasible and specific.   
Output JSON: {"score": <int>, "explanation": "<string>"}

## E.4 RELEVANCE REWARD PROMPT

Each decomposed idea is scored for relevance to the input problem p.

You are a reviewer for NSF proposals who has been a professor in your   
field for over 20 years. Evaluate the relevance of the following idea   
to the given problem statement: how well it addresses the problem,   
aligns with its objectives, and fits within the proposed methods.   
Problem statement: {problem}   
Idea: {idea}   
Score from 1 to 5:   
1 = Not relevant. 2 = Slightly relevant. 3 = Moderately relevant.   
4 = Very relevant. 5 = Highly relevant; directly aligned and   
comprehensive.   
Output JSON: {"score": <int>, "explanation": "<string>"}

## E.5 ACTION SELECTION PROMPT

The following prompt is provided to the model at each trajectory step to select the next action and creativity level.

You are a researcher selecting the best next action for your proposal   
writing process. The possible actions are:   
search: Generate specific search queries to retrieve information   
(level: 1-5).   
debate: Set up a discussion with one or more participants on a topic   
(level: 1-5).   
spark: Generate a novel research idea that challenges conventional   
thinking (level: 1-5).   
write: Write or revise the proposal (level: 1 only).   
complete: Indicate satisfaction and end the process (level: 1 only).   
Level 1-2 = focused, proposal-specific. Level 4-5 = exploratory,   
creativity-inducing.   
Target problem: {problem}   
Output JSON: {"action": "<string>", "level": <int>}

## F ACTION PROMPTS

Each action in the AI Night-Scientist framework is realized through a structured LLM prompt that takes the current problem p, creativity level c, and the accumulated trajectory context as inputs, and returns a JSON output. Below we document the core prompt structure and output schema for each creativity-enabled action (search, debate, spark) and the write action.

## F.1 SEARCH ACTION

The search action generates up to five arXiv search queries. The creativity level controls query relevance: level 1 produces tightly targeted queries derived from proposal keywords; level 5 produces broad, exploratory queries spanning alternate domains, philosophical perspectives, or unrelated fields. Retrieved papers are parsed and appended to the trajectory context for use by subsequent actions.

You are a researcher starting a literature review for a research   
proposal. Generate specific search queries whose relevance depends   
on the creativity level (1-5):   
Level 1: Extremely relevant; queries based on terms from the proposal   
title.   
Level 2: Very relevant; background and related works closely tied to   
the proposal.   
Level 3: Somewhat relevant; tangentially related background.   
Level 4: Loosely related; broader topics that still provide useful   
context.   
Level 5: Very distantly relevant; alternate domains, perspectives, or   
philosophical questions.   
Problem: {problem} Level: {level} ({level\_description})   
Output JSON: {"search\_queries": ["<query 1>", ..., "<query 5>"]}

## F.2 DEBATE ACTION

The debate action is executed in two sequential steps: (i) generating a debate setup (participants, topics, and structure) and (ii) simulating the debate conversation. The creativity level controls participant diversity and discussion structure, ranging from a focused single-turn discussion with

a same-field colleague (level 1) to an open-ended, multi-turn panel with cross-disciplinary experts (level 5). The conversation is appended verbatim to the trajectory context.

Step 1 -- Debate Setup. You are a researcher designing a debate. The   
creativity level controls who you debate with, what topics you cover,   
and how the debate is structured:   
Level 1: Single senior colleague in the same area; structured   
single-turn debate on proposal specifics.   
Level 2: Peer on related topics; structured, focused on the proposal   
with minor deviation.   
Level 3: Peer from the same field but a different topic; open-ended   
multi-turn, some deviation allowed.   
Level 4: Domain-expert from a different discipline or a 2-3 expert   
panel; open-ended, broader scope.   
Level 5: Expert(s) from a completely different discipline;   
unstructured, high-level philosophical exploration.   
Problem: {problem} Level: {level}   
Output JSON: {"debate\_participants": [{"name": "...", "job": "...",   
"expertise": "..."}],   
"debate\_topics": ["..."], "debate\_structure": "..."}

Step 2 -- Debate Conversation. You are a researcher conducting the   
debate as set up above. Generate a coherent, multi-turn conversation   
where each participant draws on their specific expertise with concrete   
details. Responses should not be surface level.   
Debate setup: {debate\_setup}   
Output JSON: {"conversation\_history": [{"speaker\_name": "...",   
"speaker\_response": "..."}]}

## F.3 SPARK ACTION

The spark action generates a “Bit-Flip” idea (O’Neill et al., 2025): it inverts a commonly held assumption to produce a novel research direction. The creativity level governs the scope of the challenged assumption, from a specific proposal-level constraint (level 1) to a field-reshaping paradigm shift (level 5). The Bit-Flip is appended to the trajectory context and can influence subsequent write and action-selection steps.

You are a researcher who has just had a sudden insight that challenges   
conventional thinking. Generate a Bit-Flip: identify a prevailing   
belief in the field (the Bit), invert or challenge it (the Flip), and   
distill the core conceptual leap into a single phrase (the Spark).   
Creativity level controls the scope:   
Level 1: Minor, proposal-specific assumption to challenge.   
Level 2: Minor but somewhat broader assumption; questions existing   
approaches in the proposal.   
Level 3: Moderate assumption; meaningful challenge to existing norms   
within the proposal.   
Level 4: Significant, field-level assumption; substantial departure   
from the status quo.   
Level 5: Major, paradigm-level assumption; potential to revolutionize   
the field.   
Problem: {problem} Level: {level}   
Output JSON: {"bit": "<2-3 sentences on the status quo and its   
limitation>",   
"flip": "<novel approach or perspective, ≥2 sentences>", "spark":   
"<core phrase>"}

## F.4 WRITE ACTION

The write action produces or revises the complete structured proposal. It takes all accumulated trajectory context as input and outputs a structured JSON proposal. The creativity level is fixed at 1 for this action; creative expression is instead encoded in the trajectory that feeds into it.

```jsonl
You are a researcher writing or revising a research proposal based on
all information gathered throughout the writing process. Create a
coherent, comprehensive proposal. You may deviate from prior ideas if
it improves the proposal.
Problem: {problem} Current proposal: {proposal} Context history:
{context}
Output JSON:
{"proposal_title": "...",
"proposal_summary": "concise statement of what, why, and what
problems it resolves",
"background_and_significance": "historical review of the field, what
remains to be done, and how this proposal advances it",
"research_plan": [{"phase": "...", "idea": "detailed hypothesis and
contribution",
"experimental_plan": "concrete methods, evaluation, pitfalls,
fallback plans"}]}
```

## G SCIJUDGE EVALUATION DETAILS

We evaluate proposal quality using SciJudge (OpenMOSS-Team/SciJudge-30B) (Tong et al., 2026), a 30B-parameter model trained to predict scientific impact via citation forecasting. For each problem instance in the test set, we compare a model-generated proposal against a matched reconstructed reference NSF proposal in a pairwise setting.

Today is {date}. Based on the titles and abstracts of the following   
two proposals A and B, determine which proposal has a higher citation   
count across its eventual papers.   
{timing\_assumption} (e.g., “Assume Proposal B was published earlier   
than Proposal A.”)   
Show your reasoning process in <reason> </reason> tags. Return the   
final answer in <answer> </answer> tags. The final answer should   
contain only ‘A’ or ‘B’.   
Proposal A:   
Title: {title\_a}   
Abstract: {abstract\_a}   
Proposal B:   
Title: {title\_b}   
Abstract: {abstract\_b}

Randomized A/B order. For each pair, proposal order is randomized using a deterministic per-pair seed to eliminate position bias. The model does not know which proposal originated from the system under evaluation versus the reconstructed reference.

Temporal framing. When comparing against the synthetically reconstructed proposal (Appendix K), we instruct the judge to assume the reconstructed reference proposal was published earlier. This grounds the comparison in a realistic temporal context: the model-generated proposal is treated as the “newer” proposal, which must surpass the quality of the funded reference to be preferred.

Output parsing. The model responds with chain-of-thought reasoning inside <reason>...</reason> tags and a final answer (A or B) inside <answer>...</answer> tags. We extract the answer tag; if absent or malformed, we fall back to a regex match on the first standalone ‘A’ or ‘B’ character.

Reported metric. We report pairwise win rate: the fraction of evaluated pairs for which SciJudge predicts the model-generated proposal to yield greater downstream citation impact than the matched reconstructed reference.

## H CONTRIBUTION-TYPE CLASSIFICATION DETAILS

We classify each proposal by its primary scholarly contribution using GPT-5.4-mini and a sevenway taxonomy adapted from Wobbrock (2012). The contribution types are: empirical, which produces new findings from systematically gathered or analyzed data; artifact, which creates a novel instantiated system, tool, process, intervention, or other constructed artifact; methodological, which introduces or refines a reusable method; theoretical, which develops reusable concepts, models, principles, hypotheses, or frameworks; benchmark or dataset, which contributes a reusable data or evaluation resource; survey, which synthesizes existing work into higher-level understanding; and opinion, which advances an evidence-grounded position intended to persuade or redirect discussion.

You are an expert annotator of scholarly contribution types.   
Label the proposal using a domain-general taxonomy of research   
contributions. Do not classify by topic, domain, or technical   
substrate.   
A contribution type is the main form of knowledge or scholarly output   
the proposed work will add.   
Proposal to label:   
Original research topic: {problem}   
Proposal title:   
{proposal\_title}   
Proposal summary:   
{proposal\_summary}   
Background and significance:   
{background\_and\_significance}   
Research plan:   
{research\_plan}   
Identify the proposal’s principal contribution from its proposed   
deliverables and research plan. Classify the central knowledge   
claim or deliverable rather than the topic, incidental methods, or   
technical vocabulary. When several contribution types appear, select   
the one that best captures the proposal’s primary scholarly output.   
Use a secondary label only when a distinct second contribution is   
substantively central.   
Return JSON with exactly this shape:   
{   
"contribution\_type": {   
"primary": "<one contribution type>",   
"secondary": "<one contribution type or none>"   
},   
"confidence": <0.0-1.0>

Distinction from research-idea paradigms. Contribution type captures what form of scholarly output the proposal ultimately contributes, whereas the research-idea paradigm captures what highlevel research move is used to turn an opportunity into a proposed direction. Following the researchidea annotation setup of Chen et al. (2026), our paradigm categories distinguish moves such as assumption relaxation, failure mitigation, formal derivation, empirical mapping, artifact construction, and optimization. The two axes are therefore complementary rather than redundant. For example, an artifact contribution may arise from relaxing an assumption, mitigating a failure, or optimizing resource use; conversely, a measurement-oriented research idea may ultimately contribute either empirical findings or a reusable benchmark.

Primary-label selection. The classifier is instructed to distinguish the proposal’s central contribution from methods or artifacts that merely support it. In particular, empirical contributions are defined by the new findings produced from data, while benchmark or dataset contributions are defined by the reusable resource itself. Similarly, artifact contributions center on a novel instantiated invention, whereas methodological contributions center on a reusable way of conducting research or practice. Theoretical contributions may be empirically evaluated, but the reusable concept, explanation, model, or framework must remain the principal contribution.

Reported diversity. For our diversity analysis, we use only the primary contribution label for each proposal. We measure how broadly and evenly a model distributes its proposals across the seven contribution types using the normalized effective number of categories described in Section 4.2.

## I LLM ORIGINALITY EVALUATION DETAILS

System: You are an experienced NSF proposal reviewer. Return valid   
JSON only.   
User: You are a reviewer for NSF proposals who has been a professor   
in your field for over 20 years.   
You are comparing two proposals that address the same award problem.   
Judge which proposal is more original.   
Originality must be judged relative to: (1) the other proposal;   
(2) the closest retrieved paper for Proposal A; (3) the closest   
retrieved paper for Proposal B. If a proposal appears more novel   
at first glance but is very similar to its closest retrieved paper,   
penalize its novelty accordingly.   
Award title / target problem: {problem\_title}   
Proposal A: {proposal\_a}   
Closest Retrieved Paper for Proposal A:   
Title: {paper\_a\_title} Abstract: {paper\_a\_abstract}   
Proposal B: {proposal\_b}   
Closest Retrieved Paper for Proposal B:   
Title: {paper\_b\_title} Abstract: {paper\_b\_abstract}   
Criteria: Originality of the central concepts; distance from the   
closest retrieved paper; whether the proposal goes beyond recombining   
familiar ideas; depth of novelty rather than superficial novelty;   
whether the proposal opens genuinely new technical directions.   
Calibration examples:   
• Borderline positive: Proposal A may look less flashy but introduces   
a genuinely different technical lever, representation, or problem   
decomposition than both Proposal B and A’s closest paper.   
• Borderline negative: Proposal A is polished but most apparent   
creativity comes from extending the closest paper to a new   
application or evaluation setting without a clearly new core idea.   
• Borderline negative: Proposal A combines familiar components into a   
multi-phase agenda; breadth alone is not novelty if the contribution   
is still an expected recombination of known methods.   
• Borderline negative: Do not reward writing sophistication, dense   
terminology, or detailed milestones if the underlying idea remains   
close to prior work.   
Return JSON only: {"winner": "A or B", "explanation": "brief   
comparative rationale"}

We assess Originality using GPT-5.1 as a judge in a pairwise setting. For each test instance, the model-generated proposal and the matched reconstructed reference proposal are each submitted to our arXiv retrieval system; the top-1 closest paper is retrieved for each using sentence-transformers/all-MiniLM-L6-v2 embedding-based search. The retrieval query is constructed from the proposal title concatenated with the first three sentences of the proposal summary. Proposal order (A vs. B) is randomized per-pair using a deterministic seed to eliminate position bias.

The judge is instructed to assess originality relative to both the competing proposal and the closest retrieved paper for each proposal. This prevents superficially creative proposals from scoring well if their core idea closely mirrors existing literature. The judgment emphasizes: (i) originality of the central concept, (ii) distance from the retrieved nearest neighbor, (iii) whether the proposal recombines familiar ideas or opens a genuinely new technical direction, and (iv) depth of novelty over breadth or terminological density.

We report the fraction of pairs in which the model-generated proposal is preferred over the matched reference (model win rate).

## J HUMAN EVALUATION STUDY

To assess the reliability of our automated evaluation metrics, we conducted a small-scale human evaluation in which two expert annotators independently judged 33 pairwise proposal comparisons (covering all three model variants: Qwen3-8B-Base, GPT-4.1, and AI Night-Scientist-8B). Each item presented annotators with a model-generated proposal alongside a matched reconstructed reference proposal; annotators selected the preferred proposal on two axes (predicted citation impact and precedence) or indicated a tie. Both independent annotators have 5+ and 15+ years of research experience in related fields, respectively.

## J.1 INTER-ANNOTATOR AGREEMENT

Table 10 reports inter-annotator agreement. Agreement is high on citation impact (80.0%, κ=0.625, substantial) and originality (86.7%, κ=0.766, substantial), indicating that both dimensions can be reliably assessed by domain-expert annotators.

Table 10: Inter-annotator agreement across evaluation dimensions (n=15 shared items).
<table><tr><td>Metric</td><td>Raw Agreement</td><td>Cohen&#x27;s κ</td><td>Interpretation</td></tr><tr><td>Impact</td><td>80.0% (12/15)</td><td>0.625</td><td>Substantial</td></tr><tr><td>Originality</td><td>86.7% (13/15)</td><td>0.766</td><td>Substantial</td></tr></table>

## J.2 HUMAN-LLM AGREEMENT

We also measure how well the LLM judges (SciJudge for predicted citation impact, GPT-5.1 for originality) agree with human annotators. For each annotator response, we map the human A/B choice to a model or ground\_truth winner using the answer key, and compare against the corresponding LLM judgment; ties and null LLM outputs are excluded. Table 11 reports the results pooled across both annotators.

Table 11: Human-LLM agreement on predicted citation impact and originality. Ties and unparseable LLM outputs are excluded.
<table><tr><td rowspan=1 colspan=1>Metric</td><td rowspan=1 colspan=1>LLM Judge</td><td rowspan=1 colspan=1>Human-LLM Agreement</td></tr><tr><td rowspan=1 colspan=1>ImpactOriginality</td><td rowspan=1 colspan=1>SciJudge-30BGPT-5.1</td><td rowspan=1 colspan=1>76.7% (46/60)72.4% (43/60)</td></tr></table>

Agreement in the low-to-mid 70s is consistent with prior work reporting LLM-judge alignment with human raters (Zheng et al., 2023), and notably these human–LLM agreement rates provide additional evidence that the automated judges broadly track expert preferences at the proposal level. These results support the use of SciJudge and GPT-5.1 as scalable proxies for human judgment at the proposal level.

## K RECONSTRUCTED REFERENCE PROPOSAL CONSTRUCTION

Original NSF grant proposals are not publicly available. We therefore construct reconstructed reference proposals from evidence surrounding each funded project. These reconstructions are not intended to reproduce the original proposal text; rather, they provide standardized, evidence-grounded references for research directions that were actually funded and subsequently pursued. The proposalgenerating model never receives the award abstracts, project outcomes, or associated publications used in this reconstruction.

1. Collect award evidence. For each award, we collect its title, abstract, project outcomes report (POR), PI names, award ID, and award period from the NSF Awards Database.

2. Retrieve likely associated papers by PI. We use PI-first retrieval over arXiv, issuing up to 12 author-based queries using PI full names, surnames, and combinations of multiple PIs. Results are deduplicated by arXiv ID. This constrains retrieval to papers plausibly authored by the funded investigators before using topical information to determine which papers are most closely associated with the award.

3. Rank candidate papers using award evidence. Candidate papers are first ranked using PIauthor overlap, topical overlap between the award evidence (title, abstract, and POR) and the paper title and abstract, and publication timing relative to the award period. We additionally reward cases in which the award title appears directly in the paper abstract. PI overlap receives the strongest weight so that topical similarity alone cannot make an unrelated paper a strong candidate.

4. Verify candidates using full-text evidence. For the highest-ranked candidates, we retrieve the paper text and extract the abstract, introduction, methods or approach, and experiments or results, excluding references and appendices where possible. We then rescore papers using full-text evidence, including explicit mention of the NSF award ID, NSF funding acknowledgments, PI surnames, and topical overlap with the award. We retain up to three highly ranked papers per award. If no paper passes the grounding threshold, we retain the strongest PI-matched papers with usable text so that reconstruction remains grounded in likely investigator-authored work.

5. Reconstruct a plausible pre-award research plan. We prompt GPT-5.1 as an expert NSF PI using the award metadata, POR, and selected papers. Crucially, the prompt treats publications as downstream evidence of what the funded project ultimately produced and asks the model to infer a plausible pre-award research plan that could have led to those outcomes, rather than summarize or copy the papers. The model is instructed not to mention the reconstruction process and to produce the same structured format used by our generated proposals: a proposal summary, background and significance, and a multi-phase research plan specifying the central idea and experimental plan for each phase.

The resulting references should therefore be interpreted as plausible reconstructions of the funded research direction, not as recovered NSF proposals. We use them only as matched references for pairwise evaluation, providing a consistent comparison point grounded in projects that were funded and subsequently pursued.

You are an expert NSF principal investigator reconstructing a highly   
realistic original NSF proposal. Use the award title, abstract,   
project outcomes report, PI names, and likely resulting papers to   
infer a plausible original proposal. Do not copy text verbatim from   
the papers and do not mention that the proposal is reconstructed or   
generated.   
Award title: {title} PI names: {pi\_names} Award period: {dates}   
NSF award abstract: {abstract}   
NSF project outcomes report: {por}   
Likely papers supported by this award ({N} retrieved):   
{paper\_blocks} (title, authors, arXiv ID, evidence excerpt, processed   
content)   
Write a detailed proposal concrete enough that an evaluator can see   
the tasks, data, algorithms, evaluation metrics, expected results,   
potential pitfalls, and fallback strategies.   
Output JSON: {"proposal\_title": "...", "proposal\_summary": "2-4   
dense paragraphs",   
"background\_and\_significance": "detailed technical review of the   
field",   
"research\_plan": [{"phase": "...", "idea": "...",   
"experimental\_plan": "..."}]}   
Requirements: 3-5 research phases; every phase must include phase,   
idea, and experimental\_plan; content must remain consistent with the   
award metadata and likely papers; no markdown, commentary, or code   
fences.

## L PROCESS-LEVEL REWARD DETAILS

The process-level reward evaluates the quality of the agent’s reasoning trajectory rather than only its final output. Each intermediate action (i.e., all actions before the final write) is scored along two complementary dimensions: exploration and contribution. Both scores are produced by an LLM judge, mapped from the 1–5 integer scale to [−1, 1] via the centering transform $s ^ { \prime } = ( s - \mathrm { \bar { 3 } } ) / 2$ , and averaged equally into a per-action score. The overall process reward is the mean of all per-action scores; actions that were available but unused (i.e., the trajectory is shorter than the maximum allowed) are penalized with a score of −1.

## L.1 EXPLORATION SCORING

The exploration dimension rewards actions that cover genuinely new ground relative to everything already explored in the trajectory. A high exploration score indicates that the action introduces a novel direction, perspective, or information source that prior actions did not cover; a low score indicates redundancy. If no prior actions have been taken, the action is considered maximally exploratory by default.

## L.2 CONTRIBUTION SCORING

The contribution dimension rewards actions that meaningfully shaped the final proposal’s quality. Each intermediate action is scored for how much it contributed to the final proposal’s precedence, feasibility, and overall quality, conditioned on prior actions (to avoid rewarding redundant actions that happen to echo an earlier high-value contribution).

You are a reviewer for NSF proposals who has been a professor in   
your field for over 20 years. Evaluate whether the current action   
is sufficiently different and novel compared to previous actions in   
the research process.   
Problem statement: {problem}   
Current action type: {action\_type}   
Current action output: {action\_output}   
Prior actions taken (ordered): {previous\_actions}   
Assess how meaningfully different the current action is from the   
previous actions. Consider:   
• Does it explore a new direction or perspective not previously   
covered?   
• Would it provide substantially different information from prior   
actions?   
• For search: are the queries fundamentally different from prior   
queries or debate topics?   
• For debate: are the topics/participants substantially different   
from prior debates?   
• For spark: does it challenge different assumptions than prior spark   
actions?   
Assign a low score if this action would likely generate outputs   
similar to prior actions; assign a high score if it explores genuinely   
new ground.   
Score from 1 to 5:   
1 = Not exploratory; highly redundant with prior actions.   
2 = Slightly exploratory; mostly overlaps with prior actions.   
3 = Moderately exploratory; some new perspectives with limited   
novelty.   
4 = Very exploratory; covers substantially new ground.   
5 = Highly exploratory; genuinely novel direction with minimal   
overlap.   
Output JSON: {"score": <int>, "explanation": "<string>"}

You are a reviewer for NSF proposals who has been a professor in your   
field for over 20 years. Evaluate how much a single intermediate   
action contributed to the final proposal across three dimensions:   
precedence, feasibility, and overall quality.   
Problem statement: {problem}   
Final proposal: {final\_proposal}   
Prior actions before this one (ordered): {previous\_actions}   
Intermediate action type: {action\_type}   
Intermediate action output: {action\_output}   
When scoring, condition on prior actions. If this action mostly   
repeats prior actions with similar outputs, assign low contribution   
scores.   
Score each dimension from 1 to 5:   
Precedence contribution: 1 = No contribution or harmful;   
5 = Essential contribution to novelty.   
Feasibility contribution: 1 = No contribution or harmful;   
5 = Essential contribution to feasibility.   
Overall quality contribution: 1 = No contribution or harmful;   
5 = Essential contribution to clarity, coherence, rigor, and alignment   
with the problem.   
Output JSON: {"precedence\_score": <int>, "precedence\_explanation":   
"<string>",   
"feasibility\_score": <int>, "feasibility\_explanation": "<string>",   
"quality\_score": <int>, "quality\_explanation": "<string>"}

## L.3 SCORE AGGREGATION

For each intermediate action i, the combined per-action process score is $r _ { i } ~ = ~ ( \mathrm { e x p l o r e } _ { i } ~ +$ contribute<sub>i</sub>)/2, where each component has been normalized to [−1, 1]. The contribution score is the mean of its three sub-components (novelty, feasibility, overall quality) after normalization. The final process reward for a trajectory of N intermediate actions out of a maximum of $N _ { \mathrm { m a x } }$ is:

$$
R _ { \mathrm { p r o c } } = \frac { 1 } { N _ { \mathrm { m a x } } } \left( \sum _ { i = 1 } ^ { N } r _ { i } ~ + ~ ( N _ { \mathrm { m a x } } - N ) \cdot ( - 1 ) \right)
$$

Missing actions are assigned −1 to penalize overly short trajectories. When combined with outcomelevel rewards (the PO setting), the final reward is $R _ { \mathrm { p o } } = ( \bar { R _ { \mathrm { p r o c } } } + R _ { \mathrm { o u t } } ) / 2$

## M TRAINING HYPERPARAMETERS

We report all key hyperparameters used to train AI NIGHT-SCIENTIST-8B and AI NIGHT-SCIENTIST-14B. Training is performed with Verl (Sheng et al., 2024) on 4 × 8 NVIDIA H100 (80 GB) GPUs (32 GPUs total). We fine-tune Qwen3-8B-Base and Qwen3-14B-Base end-to-end without LoRA adapters, as LoRA is not supported for SGLang-based rollouts in Verl.

Table 12: Training and inference hyperparameters for AI NIGHT-SCIENTIST.
<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Data</td><td></td></tr><tr><td>Train batch size (prompts)</td><td>32</td></tr><tr><td>Max prompt length</td><td>16,384 tokens</td></tr><tr><td>Max response length</td><td>25,000 tokens</td></tr><tr><td>Dynamic batching</td><td>Enabled</td></tr><tr><td>Rollout (SGLang)</td><td></td></tr><tr><td>Rollouts per prompt (n)</td><td>8</td></tr><tr><td>Max agent actions per trajectory</td><td>5 (11 max assistant turns)</td></tr><tr><td>Max tool-response length</td><td>2,048 tokens</td></tr><tr><td>Sampling temperature</td><td>0.7</td></tr><tr><td>Max model context length</td><td>32,768 tokens</td></tr><tr><td>Serendipitous swap probability</td><td>0.5 initially; decay γ = 0.001</td></tr><tr><td>Actor (FSDP2)</td><td></td></tr><tr><td>PPO mini-batch size (prompts)</td><td>32</td></tr><tr><td>Max tokens per GPU (training)</td><td>32,000</td></tr><tr><td>Reference Model (FSDP2)</td><td></td></tr><tr><td>Parameter offload</td><td>Enabled</td></tr><tr><td>Max tokens per GPU (log-prob)</td><td>40,000</td></tr><tr><td>Optimization</td><td></td></tr><tr><td>Algorithm</td><td>GRPO</td></tr><tr><td>KL loss coefficient</td><td>0.001</td></tr><tr><td>KL in reward</td><td>Disabled</td></tr><tr><td>Total training steps</td><td>170</td></tr><tr><td>Retrieval and Reward Infrastructure</td><td></td></tr><tr><td>arXiv index</td><td>Kaggle arXiv snapshot</td></tr><tr><td>Retrieval model</td><td>sentence-transformers/all-MiniLM-L6-v2</td></tr><tr><td>External reward judge</td><td>GPT-4.1</td></tr><tr><td>Judge temperature</td><td>0</td></tr></table>

## N ACTION-LEVEL ENTROPY AND SIMILARITY METRICS

During training, we track two diagnostic metrics, action-level entropy and action-level similarity, that measure whether the model’s actual output behavior is consistent with its selected creativity level. Neither metric is used as a reward signal; they serve purely as interpretability probes.

The core premise of our creativity-aligned framework is that selecting a higher creativity level should demonstrably change how the model acts, producing outputs with higher token entropy (more lexical variety) and outputs that are more distinct from the evolving proposal (higher divergence). If a model selects level 5 but generates outputs indistinguishable from level 1, the creativity selector is decorative rather than functional. These metrics operationalize that sanity check.

Action-level entropy. For each action, we collect the per-token log-probabilities from the model’s generation and compute the mean Shannon entropy across top-k vocabulary candidates. This is averaged over all tokens in the action output to yield a single per-action entropy score. We then compute the Pearson correlation between the action’s selected creativity level (equivalently, its inverse noise level 1 − noise/max\_level) and this entropy value across all actions in a training batch. A positive correlation indicates that higher creativity levels lead to higher-entropy, more diverse token distributions.

Action-level similarity. For each action, we compute the cosine similarity between the action output’s sentence embedding and the embedding of the most recently written proposal draft (or the problem statement if no draft exists yet). We again compute the Pearson correlation between the creativity level and this similarity score across the batch. A negative correlation (similarity decreases as creativity level increases) is desirable: higher creativity should yield actions that cover genuinely new territory rather than paraphrasing what has already been written.

## O LIMITATIONS

Computational cost. Training creativity-aligned agents requires substantial GPU compute. We mitigate this by working with an efficient 8B-parameter architecture, but scaling to larger models would amplify the footprint. Additionally, our reward pipeline relies on LLM judges (GPT-4.1 for feasibility/relevance during training, SciJudge-30B and GPT-5.1 for evaluation). Unlike similaritybased reward signals that compare generated text against existing corpora, we deliberately use LLM judges to assess genuinely new ideas—minimizing data leakage and preventing the model from learning to simply paraphrase known work from our dataset, which spans proposals as far back as 2018. This design choice comes at higher inference cost but is central to the validity of the precedence and originality signal.

Training stability. Given the inherent complexity of our task, particularly the implicit objective of increasing the entropy of a model’s outputs, we have observed that training can become unstable, especially over longer training runs. High-entropy generation is at odds with the stability assumptions underlying standard RL algorithms designed for low-variance, verifiable reward settings. We hope future work will explore more robust RL algorithms that can handle the entropy increases required for creative reasoning. We also note that our infrastructure relies on VERL as the training backend and SGLANG as the inference backend; their current joint support for structured, multi-step agentic frameworks is limited, and tighter integration would enable more scalable and reliable training pipelines.