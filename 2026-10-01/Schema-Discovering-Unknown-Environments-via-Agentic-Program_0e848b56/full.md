# Schema: Discovering Unknown Environments via Agentic Program Induction

Guanning Zeng<sup>3∗</sup> Jiani Wang<sup>1</sup> Wenjie Ma<sup>2</sup> Shaofeng Yin<sup>1</sup> Chenyang Wang<sup>1</sup> Shichen Liu<sup>1</sup> Angjoo Kanazawa<sup>2</sup> Wode Ni<sup>1</sup> Xiuyu Li<sup>1∗</sup> Andrea Zanette<sup>3∗</sup> Haiwen Feng<sup>1,2∗</sup>

<sup>1</sup>Impossible Research <sup>2</sup>UC Berkeley <sup>3</sup>Carnegie Mellon University

Project website: https://schema-harness.github.io/

## Abstract

Learning to complete tasks in unfamiliar environments with unknown rules remains a key challenge for LLM agents. Current LLM agents often record their discoveries in prose, which may not provide a compact, explicit account of how the environment works. Inspired by how scientists organize observations into testable, predictive theories, we introduce Schema, an agent harness that organizes learning and action through interactive program induction. The LLM agent decides what to investigate and how to act, expressing its evolving understanding of the environment as executable programs. The harness consists of a persistent program workspace and a small set of interfaces for checking these programs against the interaction history, planning within them, and executing plans under step-by-step verification. Schema raises ARC-AGI-3 RHAE from 58.7% to 99.2% with the same base model, solves 100% of the public DiG-bench games, and reaches the median performance of the top-50 human players on MazeBench. Extensive analysis shows the efectiveness of Schema in unknown mechanism discovery, and ablations confirm the contribution of each component.

## 1 Introduction

Provided with detailed financial data, an LLM agent can quickly produce a comprehensive summary. Given trip destinations and a budget, it can easily work out a thoughtful itinerary. Armed with established algorithms, it can even beat top human competitors in programming contests (OpenAI, 2025). Modern LLM agents, built on frontier models and equipped with advanced tools, now excel at tasks with familiar settings, known procedures, and well-defined objectives (Jimenez et al., 2024; Merrill et al., 2026; Xie et al., 2024; Yao et al., 2024; El-Kishky et al., 2025; Chan et al., 2025; Paglieri et al., 2025; Kwa et al., 2025). Yet acting in unknown environments where no explicit rules or instructions are given remains a challenge (Chen et al., 2026a,b; Phan et al., 2025). For example, on benchmarks that require LLM agents to learn a completely novel environment from scratch, even the strongest models still fall short of human performance (Warrier et al., 2025; Battleday et al., 2026a; Pappas and Pappas, 2026a).

This gap reflects a diference in how LLM agents and humans discover the mechanisms of an unknown environment. Humans do so through observation, experimentation, and reasoning (Gopnik and Wellman, 2012; Cook et al., 2011), and abstract what they find into explicit, reusable theories (Tenenbaum et al., 2011; Rule et al., 2020). In scientific discovery, for example, researchers derive predictions from theories, test them through experiments, and revise the theories in light of new evidence. Such theories transfer across situations (Lake et al., 2017), letting humans act eficiently and adapt quickly to new challenges (Tsividis et al., 2021; Dubey et al., 2018). LLM agents, by contrast, hold what they learn as prose in a context window or in editable memory. As context is compacted and memory rewritten (Packer et al., 2023), that prose degrades (Liu et al., 2024), and an agent that does not distill it into a persistent, reusable form is left disoriented in a complex, new environment (Laban et al., 2026; Backlund and Petersson, 2025).

Inspired by this observation, we present Schema, the first agentic harness that achieves human-level mechanism discovery in unknown environments. Schema expresses its understanding of environment mechanisms in the formal language of computers: programs. To organize how an agent constructs, tests, refines, and uses this understanding, we propose interactive program induction, a paradigm for agent design that Schema instantiates. Concretely, the LLM agent writes programs that encode environment representations and transitions, checks them against its interaction history, and uses them to guide experimentation and planning (Figure 1). This approach allows Schema to keep and continually update what it learns in a form that is persistent, executable, and verifiable.

![](images/b6dd08c7b2a1fc173f6a948e917e1327ca600ac48b01b1de7b019045ad0eb5fd.jpg)  
Figure 1: Schema observes the environment, encodes discovered mechanisms as executable programs, and uses them to guide planning and action, refining its understanding through further interaction (top). Schema achieves strong performance across all three benchmarks, outperforming baseline harnesses using the same base models (bottom); dashed lines indicate human reference performance. (a) ARC-AGI-3: RHAE over the 25 public games, Claude Fable 5 in Claude Code and in Schema; the human baseline is 100 by construction. (b) DiG-bench: public games won out of 21, GPT-6 Astra. (c) MazeBench: gems collected, GPT-6 Astra in Codex and in Schema, against the top-50 human median.

We evaluate Schema’s generality across environments and base models, eficiency in discovering hidden mechanisms, and ability to sustain progress over long horizons. On ARC-AGI-3 (ARC Prize Foundation, 2026a), Schema improves action eficiency across four model configurations, reaching 99.2% RHAE compared with 58.7% for the same base model in its coding harness. On DiG-bench (Battleday et al., 2026a), Schema discovers hidden rules with fewer failed attempts and solves all 21 public games. On MazeBench (Pappas and Pappas, 2026a), Schema accumulates and reuses knowledge across tens of thousands of interactions, sustaining exploration and reaching the median performance of the top-50 human players. Our contributions are as follows:

• Paradigm. We propose interactive program induction, a paradigm for designing LLM agents that learn and act in unknown environments by continually constructing, testing, and using executable programs.

• Harness. We build Schema, which instantiates this paradigm and reaches human-level performance on various novel environments.

• Analysis. Through ablations and behavioral analyses, we show how Schema’s components support mechanism discovery and sustained exploration.

## 2 Preliminaries

Task Setting. Consider an agent acting in an unfamiliar environment whose available actions are given, but whose rules and completion conditions must be discovered through interaction. At step $t ,$ the agent receives an observation $o _ { t }$ of latent environment state $s _ { t }$ and chooses an action $a _ { t } \in A .$ . The environment evolves according to $s _ { t + 1 } = T ( s _ { t } , a _ { t } )$ and returns $o _ { t + 1 }$ , including task feedback. The unknown transition function T and completion predicate G, with $G ( s ) = 1$ indicating success, together specify the environment’s mechanisms. The agent accumulates the ordered interaction history $\mathcal { H } _ { t } = ( ( o _ { i } , a _ { i } , o _ { i + 1 } ) ) _ { i < t }$

Its objective is to complete the task using as few interactions with the environment as possible (an objective of broad practical relevance, whether imposed by game rules or motivated by the cost of real-world experimentation). Actions therefore serve both to advance the task and to provide evidence about the environment’s mechanisms.

Program Induction. Program induction seeks an executable program P consistent with a set of input–output examples $\boldsymbol { \mathcal { D } } = \{ ( x _ { i } , y _ { i } ) \}$ , so that $P ( x _ { i } ) = y _ { i }$ for each example. For example, from the pairs abc → cba and cab → bac, a learner may infer a program that reverses its input string. Standard counterexample-guided inductive synthesis organizes this search as a dialogue between a synthesizer and a verifier (Solar-Lezama et al., 2006; Solar-Lezama, 2008). The synthesizer proposes a candidate consistent with the current examples. The verifier checks it against a specification and returns a counterexample when it fails. Adding that counterexample to $\mathcal { D }$ constrains the next candidate while retaining the evidence already collected. For environment learning, action outcomes provide examples from which language models can induce programs describing the underlying dynamics (Tang et al., 2024; Dainese et al., 2024). Executing these programs makes their predictions available for both checking and simulation. The choice of which actions to take thus becomes part of the induction process.

## 3 Schema: Agentic Harness with Interactive Program Induction

We propose interactive program induction (IPI), a design paradigm that integrates program induction and action selection into a single LLM agent. Schema instantiates IPI through the four basic operations of hypothesize, certify, plan, and act with verification (Figure 2). Schema’s harness consists of a persistent workspace and a set of interfaces for carrying out these operations (Appendix B). The LLM is free to organize calls to these interfaces, drawing on its broad prior knowledge to make context-dependent decisions that guide its interaction with the environment.

## 3.1 Hypothesize: Encoding Mechanisms as Programs

The Schema agent encodes its current understanding of the environment in a program $P ,$ which typically includes two components. (i) State grounding constructs a structured representation of observed objects, attributes, and relations, together with relevant information carried over from earlier interactions. (ii) Transition rules encode the efects of actions and object interactions on this representation. The harness invokes P through a common prediction interface

$$
\begin{array} { r } { \big ( \hat { o } _ { t + 1 } , z _ { t + 1 } \big ) = P ( z _ { t } , o _ { t } , a _ { t } ) , } \end{array}\tag{1}
$$

where $z _ { t }$ is the program’s internal state and $\hat { o } _ { t + 1 }$ is its prediction of the next observation. The agent uses file management tools to revise the program’s state representation and transition rules, while the harness appends each observed transition to the interaction history.

## 3.2 Certify: Checking Consistency with History

A proposed program is checked by replaying the interaction history. For each recorded action, the harness evaluates Equation 1 using the recorded observation and the internal state reconstructed under $P .$ . Let $\hat { o } _ { i + 1 } ^ { P }$ denote this prediction, and write $\hat { o } _ { i + 1 } ^ { P } \simeq o _ { i + 1 }$ when its observable consequences agree with the environment’s output, including completion or failure feedback. The mismatch set and history-consistency criterion are

$$
\begin{array} { r l } & { \mathcal { M } ( P , \mathcal { H } _ { t } ) = \{ i < t : \hat { o } _ { i + 1 } ^ { P } \neq o _ { i + 1 } \} , } \\ & { \mathrm { C e r t } ( P , \mathcal { H } _ { t } ) \Longleftrightarrow \mathcal { M } ( P , \mathcal { H } _ { t } ) = \emptyset . } \end{array}\tag{2}
$$

In Schema, the harness provides backtesting tools that report mismatches together with predicted and observed outcomes, giving the agent concrete counterexamples from which to revise its hypothesis. Rechecking the full history tests each revision against previously explained behavior, allowing discoveries to accumulate in a coherent program. In practice, the LLM iteratively uses file management and backtesting tools to revise the program and check its consistency with the full interaction history.

## 3.3 Plan: Searching within the Program

The same prediction interface makes P a simulator: feeding predicted observations and internal states back into Equation 1 produces a rollout without interacting with the environment. The agent specifies a target predicate g over these simulated states and observations. The target may express task completion, an intermediate waypoint, or a situation in which an unresolved mechanism can be tested. Planning therefore serves both task progress and experiment design. Given a target, the agent chooses a search procedure $\sigma$ and computational budget b and invokes

![](images/ecb20cf3253dd9fa99f622d56280a91904343a2e0bf94c8133271dd25da12359.jpg)  
Figure 2: Overview of Schema. The LLM agent expresses its understanding as an executable program $P$ in a persistent workspace. The harness checks P against recorded interactions and searches within it using goals and procedures chosen by the agent, without consuming environment actions. Committed plans execute one action at a time under prediction checks. A mismatch interrupts execution and returns new evidence for revising P. The ARC-AGI-3 LS20 level 3 example illustrates an unmodeled color-rotator efect with a simplified prediction: the key is predicted to remain orange but turns blue.

$$
\pi  \mathrm { S E A R C H } _ { \sigma } ( P , z _ { t } , o _ { t } , g ; b ) , \pi = ( a _ { t } , \dotsc , a _ { t + k - 1 } ) .\tag{3}
$$

The Schema harness runs the selected procedure inside $P$ and returns a candidate plan or reports that none was found within the budget. The agent can call built-in planning tools (e.g., breadth-first, depth-first, $\mathrm { A ^ { * } }$ , or greedy search) or write and load any search procedure of its own. It also sets and can adjust search depth, node limits, and runtime in its tool calls. The returned plan and search feedback help the agent decide whether to commit to the actions, refine the search, or investigate further.

## 3.4 Act with Verification: Testing Predictions in the Environment

The agent commits an action sequence of its chosen length, obtained through search or designed directly as an experiment. Before each action ${ { a } _ { t } } ,$ the harness obtains the program’s prediction using Equation 1; it then executes the action and appends the observed transition to $\mathcal { H } _ { t }$ . If prediction and observation agree, execution proceeds without another LLM call. Inspired by model predictive control (Li and Shi, 2014), the harness discards the remaining actions at the first prediction mismatch and returns the unexpected transition to the agent. Each executed action extends the evidence available for induction. The agent is free to use the returned observations to decide whether to revise the program, adjust its plans, or investigate further.

## 4 Experiments

Our experiments assess Schema’s generality across games and base models, eficiency in discovering hidden mechanisms, and ability to sustain progress over long horizons. ARC-AGI-3 establishes performance across base models and uses component ablations to explain how discoveries translate into eficient action (Section 4.1). DiG-bench examines the cost of identifying hidden rules, measured by games won and lives lost at diferent reasoning eforts (Section 4.2). MazeBench examines whether knowledge remains compact and useful as interaction history grows, linking program reuse to continued exploration (Section 4.3). Within each benchmark, we compare Schema with baseline harnesses using the same base model. Further benchmark background and scoring details are given in Appendix C, and gameplay videos are in the supplementary material.

(a) RHAE (%)  
![](images/6ee5262593b596e8f5411db357178419cb7a228f42a20f67afc05c371f7dd97d.jpg)

![](images/b0792bb46479a0d9da7ebc66b5cd018258896741d900482d49b92bb4d0e4d109.jpg)  
Coding harness (Claude Code / Codex)

(c) games at full RHAE (%)  
![](images/125b4727cc4dc593124bfeb3b37062d76d7d9057a5f28197bdf44077ea4c7e76.jpg)  
Figure 3: Schema improves completion and action eficiency across model configurations. Results on the 25 public ARC-AGI-3 games: (a) RHAE, (b) percentage of games won, and (c) percentage of games achieving 100% RHAE. Each model configuration is evaluated with the basic harness, coding harness, and Schema.

## 4.1 ARC-AGI-3: Human-level Action Eficiency

ARC-AGI-3 requires agents to discover unfamiliar game mechanics through interaction and apply them to increasingly complex levels. We assess Schema’s performance relative to human players and baseline harnesses, then use component ablations to analyze the source of its gains.

Task Setup. ARC-AGI-3 (ARC Prize Foundation, March 2026) contains 25 public games, each with six to ten levels presented as 64×64 color grids. The agent is given up to seven available actions and must discover their efects and the game’s objective through interaction. The primary metric, relative human action eficiency (RHAE), combines level completion with action eficiency relative to first-time human players (Appendix C.1). We also report the number of games won and games reaching the maximum score. The benchmark remains challenging: in its oficial basic harness, models whose training-data cutofs precede its public release still only achieve at most 30% RHAE. We evaluate Claude Opus 4.8 (Anthropic, 2026e), Claude Fable 5 (Anthropic, 2026b), and GPT-5.6 Sol (OpenAI, 2026d), with Sol tested at both extra-high and maximum reasoning efort. We compare Schema with two types of baseline harnesses using the same base model: (i) the oficial ARC-AGI-3 basic harness, which provides no auxiliary tools, and (ii) the model’s coding harness (Claude Code (Anthropic, 2026a) for Claude models and Codex (OpenAI, 2026a) for GPT models), which allows agents to maintain persistent notes and execute code and shell commands.

Main Results. Across the four configurations, Schema raises RHAE from 1.2–20.2% in the basic harness to 86.1–99.2% and increases games won from 0–6 to 22–25 (Figure 3). Relative to the coding harnesses, Schema improves RHAE by 34.4–40.7 percentage points (Figure 3a). It reaches 99.2% with Fable 5, 96.7% and 91.8% with Sol at maximum and extra-high efort, and 86.1% with Opus 4.8, compared with 58.7%, 62.3%, 57.4%, and 45.4% in their coding harnesses. With Fable 5 and Sol at maximum efort, Schema clears all 25 games, compared with 14 and 15 for the respective coding harnesses; 19 and 21 games reach the

Table 1: Ablation results.
<table><tr><td></td><td>RHAE (%)</td></tr><tr><td>Schema (full)</td><td>72.9</td></tr><tr><td>prose model</td><td>58.8</td></tr><tr><td>w/o certification</td><td>62.3</td></tr><tr><td>w/o planning</td><td>59.2</td></tr><tr><td>w/o verification</td><td>52.4</td></tr></table>

maximum score, compared with seven for each baseline (Figure 3b,c). With Fable 5, Schema completes every one of the 25 games in fewer actions than the human baseline, even using just 57% of the human action total across all 183 levels. These results suggest that frontier models do not lack the ability to understand unfamiliar environments. However, they need the right harness to put their broad knowledge to use.

Ablation Study & Analysis. To further understand how the harness helps models put their knowledge to use, we replace the executable program with prose and separately remove the interfaces for certification, planning, and verified execution from Schema. We evaluate all variants with Claude Opus 4.8 on the ten hardest games, ranked by the number of actions in their human baselines, covering 75 levels with a budget of 2,000 actions per game. As shown in Table 1, removing any of these components substantially reduces performance, with RHAE falling from 72.9% to 52.4–62.3%.

(d) verification  
![](images/d26199336d62253908bfbc0e32d62daf8d739cebb24d84ee4026b50da4d6a10b.jpg)

![](images/d1784f16ccedb6cba7ed0931176ff504e719587caa90a6fafdaee73ac0c39eb9.jpg)

![](images/bc2502d72273d75dffb041869853591918d2feb58540282a29a32cfbb9057007.jpg)

![](images/a7e6a4bbbe2e449af12b089a406a33e504fc13162f172bf5d7b5c1cf9b78847a.jpg)  
Figure 4: Trace dynamics. (a) Levels cleared versus cumulative steps on CN04 with program and prose representations, alongside humans. (b) Ofline history agreement on AR25 with and without certification, using Schema’s Opus 4.8 main evaluation run. (c) Levels cleared versus cumulative steps on LS20 with and without planning, alongside humans. (d) Steps taken after a prediction mismatch within the same plan without verification.

We further examine the agents’ reasoning and interaction traces to understand how these ablations afect their discovery and use of the environment’s rules, with detailed cases in Appendix D.1. (i) Program induction makes beliefs explicit and reusable. In CN04 level 5, pieces must be moved and rotated to pair their connectors without overlapping, while one piece grows and changes the geometry. The prose agent repeatedly revisits the connection rules; Schema enumerates arrangements under explicit geometric constraints, completing the level in 279 rather than 1,219 actions (Figure 4a). (ii) Certification helps the agent validate hypothesis correctness. In AR25, the agent moves pieces and mirror axes so that reflected shapes match a target. Without certification, a revision to identify the active piece improves predictions on the current level but drops history agreement from 97.7% to 65.2%; Schema’s final model reproduces 99.6% of its history (Figure 4b). (iii) Planning ahead improves action eficiency. Even with a program that fits observed transitions, choosing an eficient action sequence requires reasoning about future states. In LS20 level 5, the agent must transform a carried symbol to match its target, timing contact with a moving rotation marker within an energy budget. Without planning, the model passes all 252 transition checks at entry to the level, but the agent works out routes and transformations manually, taking 233 actions compared with 65 for Schema (Figure 4c). (iv) Runtime verification further boosts adaptation to new scenarios. In LS20 level 2, the agent must rotate a carried symbol and reach the goal using single-use refueling stations. Without verification, it assumes these stations are reusable and continues for 42 actions after the first mismatch, returning to a spent station and running out of energy. Across the ten games in Figure 4d, such continuations account for 58% of actions.

## 4.2 DiG-bench: Eficient Mechanism Discovery

We next turn to a setting where discovering hidden mechanisms is central to success. On DiG-bench, we examine whether Schema can choose informative interactions and eficiently infer the underlying rules.

Task Setup. DiG-bench (Battleday et al., August 2026) evaluates interactive mechanism discovery through text-based games across seven dificulty tiers. Game states are presented as short strings of letters, numbers, and symbols, but both the transition rules and the win conditions must be discovered through interaction. For example, in P-3, the agent queries whether chosen strings satisfy a hidden rule, then classifies new strings, with each incorrect quiz answer costing a life. The agent must choose informative actions and use the resulting feedback to identify hidden rules while completing challenges with very limited lives and per-level step budgets. Some games also ofer a creative mode for experimentation, allowing the agent to construct situations that test its hypotheses. These games remain challenging for frontier models. The benchmark authors report that the strongest model achieves a 71.4% win rate overall and only 40% on the two hardest tiers, while every game was solved by at least one human on their first attempt. Existing agentic harnesses have also shown limited gains over the basic harness on these discovery tasks (Battleday et al., 2026a). We evaluate Schema on the 21 publicly available games, spanning all seven dificulty tiers, using GPT-6 Astra (OpenAI, 2026e) at medium, high, and maximum reasoning efort.

(a) Games won (%)  
![](images/72d0225ea0841b90013c3be6668fd6d2731f6a5dd62d8de768182642ab504392.jpg)

![](images/c1f2b25a78e4a51f3380033d1bf91b0b7830e8e9a273bc0215ba17efcab7070a.jpg)

(c) Levels completed (%, medium)  
![](images/cc5388ce0338db84c88c91ac96307a22cabfe214ff8cbbcb04ba75e371fa3d24.jpg)  
Figure 5: Schema solves more games with fewer failed attempts. GPT-6 Astra on the 21 public DiG-bench games: (a) game win rate and (b) total lives lost at each reasoning efort; (c) percentage of levels completed in each game at medium efort, grouped by dificulty tier.

Results. Compared with the basic harness, Schema improves game completion and reduces failed attempts at all three reasoning eforts (Figure 5). It clears 19, 20, and 21 games at medium, high, and maximum efort, respectively, compared with 11, 16, and 19 for the basic harness. On the two hardest tiers (6–7), Schema raises the win rate at maximum efort from 66.7% to 100%. At medium efort, Schema already matches the basic harness’s maximum-efort win count; compared with the basic harness at medium efort, it also reduces lives lost from 96 to 34. The same trends hold across repeated runs (Appendix E.2). Pooling

![](images/93b3a871370f0486e7fcebdda9763a0675377e7ae76bc46080e4ca2eaafe4e1a.jpg)

![](images/948f68855ccb958195f8bfac08e9f96d3c57c4b23184e49b0220d7085ae89e1b.jpg)  
Figure 6: Token cost on DiG-bench. Average cost (a) per cleared level and (b) by level index, including input and output tokens at OpenAI’s oficial API rates (OpenAI, 2026f).

costs and completed levels across all three reasoning eforts, Schema’s estimated token cost per cleared level is \$3.40, approximately 15% lower than the basic harness’s \$4.00 (Figure 6a). For levels cleared by both harnesses, the cost curve pooled by level index shows higher spending by Schema at early levels but lower mean costs at every index from 9 to 14 (Figure 6b). This pattern is consistent with reusing mechanisms discovered early in the game to solve later levels. Together, these results show that Schema supports eficient mechanism discovery with fewer failed attempts, a lower aggregate token cost per cleared level, and less reasoning efort to reach comparable game completion.

Analysis. We examine the interaction traces and find that Schema is more systematic than the basic harness in testing its hypotheses. When feedback contradicts its current understanding, Schema revises the program and uses further queries to distinguish competing explanations, checking that the new rule also accounts for earlier observations. This helps it resolve uncertainty before risking lives on an untested answer. For example, in P-3 level 3, Schema tests strings with diferent letter orders and counts to discover the acceptance rule that the basic harness misses. See Appendix D.2 for the queries, program revisions, and quiz outcomes in this level.

## 4.3 MazeBench: Knowledge Accumulation over Long Horizons

Finally, we examine whether Schema can build up and draw on its understanding of a large environment over long horizons. We observe sustained knowledge acquisition and agentic exploration.

Task Setup. MazeBench (Pappas and Pappas, July 2026) is a three-dimensional exploration and puzzle-solving environment with 256 interconnected rooms and 100 gems (Figure 7). Its rooms feature mechanisms such as lifts, bridges, springs, and carpets, alongside scenery ranging from forests and icy mountains to palaces and city walls; the rules governing these mechanisms must be discovered through interaction. We evaluate Schema with GPT-6 Astra, measuring exploration progress by gems collected and rooms visited, and compare its trajectory with that of the same model in Codex, using human performance as a reference. At submission, the top-50 human players on the oficial leaderboard have an average recorded playtime of 37.2 hours and an average action count of 65,876; their median gem collection and room coverage are 29.0% and 53.5%, respectively, reflecting the demands of sustained exploration in this environment.

(a) Gems collected  
![](images/7602ebe15e8d99089617af28eeb587a41822ed28702ab1a61d32bfbe8c148cd8.jpg)

(b) Rooms visited  
![](images/9f1d47ac54e191a04425374710c627f9ad1bbe67fb76561da0ba2d1802be6b86.jpg)

(c) Board novelty (%)  
![](images/123e54f4aa8f12e4605f0cc46430249881290c32361127aed2789ed12f057a49.jpg)

(d) Token Cost /gem (USD)  
![](images/61aafa2759db92d583abbc672b2c60342d49462eae7e2a3feb975194af4ae279.jpg)  
Schema (GPT-6 Astra) Codex (GPT-6 Astra) Human top-50 median Codex (GPT-5.6 Sol) Claude Code (Opus 5)

Figure 8: Schema sustains progress over long horizons. (a, b) Gems collected and rooms visited over environment actions; crosses mark the last recorded action of each baseline trajectory, and dotted lines show the top-50 human median. (c) Board-state novelty, with horizontal lines indicating each curve’s mean over the shaded interval after 10,000 actions. (d) Token cost per gem at 23 collected gems.  
(a) Model size  
![](images/4149d76a864c3b8084a19090fc04294061a20f5cb85132b8af397b310b56b17b.jpg)

(b) History explained  
![](images/6317a296f397309f38d47bc3619b9f184b74ff356987c4a61f9b28015b232ec1.jpg)

![](images/ff61723ff30b917b5675c4be8ee577e1f9b88cfc34f778d6fb0026819e2cc084.jpg)

![](images/f39a7a4adafe1b47d0f4cd417cd3b9a0492ccc4eb29ea9383504cbb6af9194f0.jpg)  
Figure 9: Knowledge accumulates in a compact, reusable program. (a) Non-comment program lines in Schema and note lines in the prose-model variant. (b) Recorded transitions explained per program line. (c) Call sites per function. (d) Forward prediction accuracy over the trailing 1,000 actions.

Results. Over a 36-hour run, Schema collects 33 gems and visits 139 rooms in 27,819 actions, exceeding the top-50 human median on both metrics (Figure 8a,b). At the same action count of 20,673, Schema has collected 30 gems and visited 133 rooms, compared with 23 gems and 113 rooms for the Codex baseline, at a similar token cost (Figure 8d). Both agents have visited 95 rooms by action 10,000; over the next 10,000 actions, Schema adds 38 rooms and 13 gems, while Codex adds 18 rooms and five gems. In the plotted interval after action 10,000, Schema also maintains higher mean board-state novelty, at 54% compared with

![](images/e4f0060fee5c7a431eba9cc17ddb35be146ac6ec742deb05c41a3b466d9ee778.jpg)  
Figure 7: MazeBench. The full world of 256 rooms, with close-ups of rooms CxG, FxC, and MxJ.

40% for Codex (Figure 8c). These results show that Schema sustains broader exploration and continues to make progress over long interaction horizons. Appendix E.3 reports every gem’s location and collection step, together with room coverage maps for all four runs in Figure 8.

Analysis. We examine how the agent (i) accumulates knowledge in its program and (ii) reuses that knowledge in new situations during prolonged interaction with an unfamiliar environment. Schema incorporates new experience by revising and extending the program’s shared rules. By approximately 20,000 actions, the program contains 2,012 non-comment lines, compared with 6,133 lines of notes in the prose-model variant (Figure 9a). The number of recorded transitions explained per line and the average number of call sites per function both increase, while forward prediction accuracy remains high (Figure 9b–d). These trends show that growing experience is captured in compact, reusable rules that retain their predictive value. These rules also support planning in new configurations after repeated context compaction. For example, on ice the player keeps sliding until blocked or reaching ordinary ground, so a route to the gem must account for where each slide stops. Schema learns this mechanism in room IxG at around action 2,600. Roughly 25,000 actions later, after at least 20 context compactions, it reuses the rule in a new ice maze in room NxE. After incorporating the new room’s layout into the program, Schema uses A\* search to find a 38-action route to the gem. It executes the entire route, with every observed transition matching the program’s prediction (Appendix D.3).

## 5 Related Work

Agentic Harness. Agent harnesses shape how language models behave and what they can accomplish. ReAct (Yao et al., 2023), CodeAct (Wang et al., 2024c), and SWE-agent (Yang et al., 2024) demonstrate the importance of interaction loops and tool interfaces. Reflexion (Shinn et al., 2023) and ACE (Zhang et al., 2025b) enable agents to learn from experience by turning feedback into reusable reflections and strategies. Recent work further explores self-improving agents and harnesses by revising the agent’s own code (Zhang et al., 2025a) and updating persistent prompts, memories, and skills (Karten et al., August 2026). Schema organizes learning and action around one persistent, evolving executable theory of the environment, continually certified against the full interaction history and directly used for experimentation and planning.

Code World Models. Programs provide interpretable, executable models of environment dynamics, an approach explored in theory-based reinforcement learning before the rise of LLM agents (Tsividis et al., 2021). WorldCoder (Tang et al., 2024) and CWM (Dainese et al., 2024) use LLMs and execution feedback to synthesize and refine Python world models. Subsequent work develops hierarchical planning in TheoryCoder (Ahmed et al., 2025) and compositional probabilistic representations in PoE-World (Piriyakulkij et al., 2025). These methods use LLMs as operators for generating and revising code within predefined learning procedures, such as WorldCoder’s REx search, and typically learn dynamics over supplied state representations. Schema builds on these insights but makes the entire learning process agent-directed, transferring control over model revision, experimentation, and planning to the agent. The agent determines how to learn in environments where state grounding, transition rules, and task goals are all initially unknown.

Concurrent Work. Recent work has explored related approaches on ARC-AGI-3, including EWM (Rodionov, May 2026), OPINE-World (Courtis et al., July 2026), NOOA (Furgale et al., July 2026), Tycho (Lehmann et al., July 2026), and Twin (Skoutnev et al., August 2026). These systems combine executable modeling, interaction feedback, and planning, with some also citing earlier releases of Schema. These concurrent eforts highlight the promise of interactive program induction, which our study explores as a general approach to agent design across diverse unfamiliar environments.

## 6 Conclusion

We introduced interactive program induction, an approach in which an LLM agent learns about an unfamiliar environment by constructing, testing, and using executable models. Schema implements this approach through a persistent program workspace and tools for checking predictions against interaction history, planning, and verifying actions, while the agent directs experimentation and program revision. Across ARC-AGI-3, DiG-bench, and MazeBench, Schema improves task performance with frozen language models, supporting eficient mechanism discovery and sustained exploration over long interaction horizons. Ablations support the contribution of each component, while behavioral analyses illustrate how learned rules can persist across context compactions and be reused in new situations. Together, these findings suggest that persistent executable models provide a way for agents to accumulate knowledge over interaction, turning individual discoveries into explicit hypotheses that can be tested, refined, reused, and ultimately acted upon.

## Acknowledgements

This work was supported in part by the National Science Foundation under Grant CCF-2106778. The authors thank Zhiqi Chen and Emma Chu for helpful discussion and feedback.

## References

Zergham Ahmed, Joshua B. Tenenbaum, Christopher J. Bates, and Samuel J. Gershman. Synthesizing world models for bilevel planning. arXiv preprint arXiv:2503.20124, March 2025.

Zergham Ahmed, Kazuki Irie, Joshua B. Tenenbaum, Christopher J. Bates, and Samuel J. Gershman. Learning abstractions for hierarchical planning in program-synthesis agents. arXiv preprint arXiv:2602.00929, 2026. URL https://arxiv.org/abs/2602.00929.

Anthropic. Claude Code: Overview. https://code.claude.com/docs/en/overview, 2026a. Accessed September 25, 2026.

Anthropic. Claude Fable 5 model documentation. https://platform.claude.com/docs/en/models/ fable-5/overview, 2026b. Released June 9, 2026.

Anthropic. Claude Fable 5.1 model documentation. https://platform.claude.com/docs/en/models/ fable-5-1/overview, 2026c. Released September 1, 2026.

Anthropic. Claude Opus 4.7 model documentation. https://platform.claude.com/docs/en/models/ opus-4-7/overview, 2026d. Released April 16, 2026.

Anthropic. Claude Opus 4.8 model documentation. https://platform.claude.com/docs/en/models/ opus-4-8/overview, 2026e. Released May 28, 2026.

Anthropic. Claude Opus 5 model documentation. https://platform.claude.com/docs/en/models/ opus-5/overview, 2026f. Released July 24, 2026.

ARC Prize Foundation. ARC-AGI-3: A new challenge for frontier agentic intelligence. arXiv preprint arXiv:2603.24621, March 2026a.

ARC Prize Foundation. OpenAI’s GPT-6 Astra on ARC-AGI-3. https://arcprize.org/blog/astra, 2026b. Accessed September 2026.

ARC Prize Foundation. Fable-class models score approximately 20% on the ARC-AGI-3 Public Demo environments. https://x.com/arcprize/status/2080716563448295720, 2026c. Post on X, July 24, 2026.

ARC Prize Foundation. Analyzing GPT-5.5 & Opus 4.7 with ARC-AGI-3. https://arcprize.org/blog/ arc-agi-3-gpt-5-5-opus-4-7-analysis, 2026d. Blog post, May 1, 2026.

ARC Prize Foundation. ARC-AGI leaderboard. https://arcprize.org/leaderboard, 2026e. Accessed September 2026.

ARC Prize Foundation. Claude Opus 5: ARC-AGI results. https://arcprize.org/results/ anthropic-claude-opus-5, 2026f. Accessed September 26, 2026.

ARC Prize Foundation. GPT-5.6 Sol: ARC-AGI results. https://arcprize.org/results/ openai-gpt-5-6-sol, 2026g. Accessed September 26, 2026.

Axel Backlund and Lukas Petersson. Vending-bench: A benchmark for long-term coherence of autonomous agents. arXiv preprint arXiv:2502.15840, 2025.

Ruairidh M. Battleday, Kai Sandbrink, Jimi Cullen-Drohan, Zihan Yan, Timothy Muller, Clare Maguire, Ales Kubicek, Fraser Greenlee-Scott, Sukrit Sumant, Tri Dao, Jürgen Schmidhuber, Michal Valko, Joshua Tenenbaum, Thomas L. Grifiths, Zeb Kurth-Nelson, and James C. R. Whittington. DiG-bench: Discovery in games. arXiv preprint arXiv:2608.12593, August 2026a.

Ruairidh M. Battleday et al. DiG-bench: Discovering unknown rules in text-based games. https: //digbench.ai, 2026b. Accessed September 26, 2026.

Jun Shern Chan, Neil Chowdhury, Oliver Jafe, James Aung, Dane Sherburn, Evan Mays, Giulio Starace, Kevin Liu, Leon Maksin, Tejal Patwardhan, Lilian Weng, and Aleksander Madry. MLE-bench: Evaluating machine learning agents on machine learning engineering. In International Conference on Learning Representations (ICLR), 2025.

Arthur Chen, Zuxin Liu, Jianguo Zhang, Akshara Prabhakar, Zhiwei Liu, Shelby Heinecke, Silvio Savarese, Victor Zhong, and Caiming Xiong. Test-time adaptation for LLM agents via environment interaction. In International Conference on Learning Representations (ICLR), 2026a.

Shiqi Chen, Tongyao Zhu, Zian Wang, Jinghan Zhang, Kangrui Wang, Ruochen Zhou, Siyang Gao, Teng Xiao, Yee Whye Teh, Junxian He, and Manling Li. Why do LLM agents fail in exploring new environments? a world-modeling perspective. In Findings of the Association for Computational Linguistics: EMNLP 2026, 2026b.

Claire Cook, Noah D. Goodman, and Laura E. Schulz. Where science starts: Spontaneous experiments in preschoolers’ exploratory play. Cognition, 120(3):341–349, 2011.

David Courtis, Wenhao Li, and Scott Sanner. OPINE-World: Programmatic world modeling with ontology-error-prioritized interactive exploration for ARC-AGI-3. arXiv preprint arXiv:2607.01531, July 2026.

Nicola Dainese, Matteo Merler, Minttu Alakuijala, and Pekka Marttinen. Generating code world models with large language models guided by Monte Carlo tree search. In Advances in Neural Information Processing Systems (NeurIPS), 2024.

DeepSeek-AI. DeepSeek-V4: Towards highly eficient million-token context intelligence. arXiv preprint arXiv:2606.19348, 2026.

Rachit Dubey, Pulkit Agrawal, Deepak Pathak, Thomas L. Grifiths, and Alexei A. Efros. Investigating human priors for playing video games. In Proceedings of the 35th International Conference on Machine Learning (ICML), volume 80 of Proceedings of Machine Learning Research, pages 1349–1357, 2018.

Ahmed El-Kishky, Alexander Wei, Andre Saraiva, Borys Minaiev, Daniel Selsam, David Dohan, Francis Song, Hunter Lightman, Ignasi Clavera, Jakub Pachocki, Jerry Tworek, Lorenz Kuhn, Lukasz Kaiser, Mark Chen, Max Schwarzer, Mostafa Rohaninejad, Nat McAleese, Oleg Mürk, Rhythm Garg, Rui Shu, Szymon Sidor, Vineet Kosaraju, and Wenda Zhou. Competitive programming with large reasoning models. arXiv preprint arXiv:2502.06807, 2025.

Kevin Ellis, Catherine Wong, Maxwell Nye, Mathias Sablé-Meyer, Lucas Morales, Luke Hewitt, Luc Cary, Armando Solar-Lezama, and Joshua B. Tenenbaum. DreamCoder: Bootstrapping inductive program synthesis with wake-sleep library learning. In Proceedings of the 42nd ACM SIGPLAN International Conference on Programming Language Design and Implementation, pages 835–850, 2021. doi: 10.1145/3453483.3454080. URL https://doi.org/10.1145/3453483.3454080.

Alexis Fox, Junlin Wang, Paul Rosu, and Bhuwan Dhingra. PRO-LONG: Programmatic memory enables long-horizon reasoning. arXiv preprint arXiv:2607.20064, July 2026.

Paul Furgale, Severin Klingler, James Nolan, Matt Staats, Gaia Di Lorenzo, Elisa Martinez Abad, Christian Schüller, Razvan Dinu, Alessio Devoto, Pascal Berard, Gal Kaplun, Elad Sarafian, Riccardo Roveri, Leon Derczynski, and Ricardo Silveira Cabral. NVIDIA-labs OO agents: Native Python object-oriented agents. arXiv preprint arXiv:2607.20709, July 2026.

Kanishk Gandhi, Michael Y. Li, Lyle Goodyear, Agam Bhatia, Louise Li, Aditi Bhaskar, Mohammed Zaman, and Noah D. Goodman. BoxingGym: Benchmarking progress in automated experimental design and model discovery. arXiv preprint arXiv:2501.01540, 2025. URL https://arxiv.org/abs/2501.01540.

Google DeepMind. Gemini 3.1 Pro model card. https://deepmind.google/models/model-cards/ gemini-3-1-pro/, 2026. Published February 19, 2026.

Alison Gopnik and Henry M. Wellman. Reconstructing constructivism: Causal models, Bayesian learning mechanisms, and the theory theory. Psychological Bulletin, 138(6):1085–1108, 2012.

Sumit Gulwani. Automating string processing in spreadsheets using input-output examples. In Proceedings of the 38th Annual ACM SIGPLAN-SIGACT Symposium on Principles of Programming Languages (POPL), pages 317–330. ACM, 2011. doi: 10.1145/1926385.1926423.

Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, and Timothy Lillicrap. Mastering diverse control tasks through world models. Nature, 640(8059):647–653, 2025. doi: 10.1038/s41586-025-08744-2.

Qiushi Han, Keya Hu, Linlu Qiu, Cathy Wu, and Kaiming He. VISTA: A visual harness for reasoning in an interactive world, August 2026. URL https://vista-research.github.io/. Blog post.

Carlos E. Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik Narasimhan. SWE-bench: Can language models resolve real-world GitHub issues? In International Conference on Learning Representations (ICLR), 2024.

Seth Karten, Alex L. Zhang, Kevin Thomas, Sebastian Müller, Elie Bakouch, Daniel Auras, Mika Senghaas, Fares Obeid, Konstantin Dunas, Johannes Hagemann, and Sami Jaghouar. Prime Agent: A self-improving RLM harness. arXiv preprint arXiv:2608.23552, August 2026.

Zaid Khan, Archiki Prasad, Elias Stengel-Eskin, Jaemin Cho, and Mohit Bansal. One life to learn: Inferring symbolic world models for stochastic environments from unguided exploration. arXiv preprint arXiv:2510.12088, October 2025.

Thomas Kwa, Ben West, Joel Becker, Amy Deng, Katharyn Garcia, Max Hasin, Sami Jawhar, Megan Kinniment, Nate Rush, Sydney Von Arx, Ryan Bloom, Thomas Broadley, Haoxing Du, Brian Goodrich, Nikola Jurkovic, Luke Harold Miles, Seraphina Nix, Tao Lin, Chris Painter, Neev Parikh, David Rein, Lucas Jun Koba Sato, Hjalmar Wijk, Daniel M. Ziegler, Elizabeth Barnes, and Lawrence Chan. Measuring AI ability to complete long software tasks. In Advances in Neural Information Processing Systems (NeurIPS), 2025.

Philippe Laban, Hiroaki Hayashi, Yingbo Zhou, and Jennifer Neville. LLMs get lost in multi-turn conversation. In International Conference on Learning Representations (ICLR), 2026.

Brenden M. Lake, Tomer D. Ullman, Joshua B. Tenenbaum, and Samuel J. Gershman. Building machines that learn and think like people. Behavioral and Brain Sciences, 40:e253, 2017.

Jens Lehmann, Andrei Aioanei, and Sahar Vahdati. Tycho: Active abstraction with programmatic world models for ARC-AGI-3. arXiv preprint arXiv:2607.28287, July 2026.

Huiping Li and Yang Shi. Event-triggered robust model predictive control of continuous-time nonlinear systems. Automatica, 50(5):1507–1513, 2014. doi: 10.1016/j.automatica.2014.03.015.

Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. Lost in the middle: How language models use long contexts. Transactions of the Association for Computational Linguistics, 12:157–173, 2024.

Mike A. Merrill, Alexander G. Shaw, Nicholas Carlini, et al. Terminal-bench: Benchmarking agents on hard, realistic tasks in command line interfaces. arXiv preprint arXiv:2601.11868, 2026.

Moonshot AI. Kimi K3 model card. https://huggingface.co/moonshotai/Kimi-K3, 2026. Released July 2026.

OpenAI. OpenAI reasoning models solve all 12 problems at the 2025 ICPC world finals. https: //x.com/OpenAI/status/1968368133024231902, 2025. Accessed September 2026.

OpenAI. Codex CLI. https://learn.chatgpt.com/docs/codex/cli, 2026a. Accessed September 25, 2026.

OpenAI. GPT-5.4 model documentation. https://developers.openai.com/api/docs/models/gpt-5.4, 2026b. Accessed September 2026.

OpenAI. GPT-5.5 model documentation. https://developers.openai.com/api/docs/models/gpt-5.5, 2026c. Accessed September 2026.

OpenAI. GPT-5.6: Frontier intelligence that scales with your ambition. https://openai.com/index/ gpt-5-6/, 2026d. Released July 9, 2026.

OpenAI. GPT-6 Astra: A new generation of intelligence. https://openai.com/index/gpt-6-astra/, 2026e. Accessed September 2026.

OpenAI. GPT-6 Astra model. https://developers.openai.com/api/docs/models/gpt-6-astra, 2026f. API pricing. Accessed September 25, 2026.

Charles Packer, Sarah Wooders, Kevin Lin, Vivian Fang, Shishir G. Patil, Ion Stoica, and Joseph E. Gonzalez. MemGPT: Towards LLMs as operating systems. arXiv preprint arXiv:2310.08560, 2023.

Davide Paglieri, Bartłomiej Cupiał, Samuel Coward, Ulyana Piterbarg, Maciej Wolczyk, Akbir Khan, Eduardo Pignatelli, Łukasz Kuciński, Lerrel Pinto, Rob Fergus, Jakob Nicolaus Foerster, Jack Parker-Holder, and Tim Rocktäschel. BALROG: Benchmarking agentic LLM and VLM reasoning on games. In International Conference on Learning Representations (ICLR), 2025.

Jonathan Pappas and David Pappas. Maze Bench: Visual spatial reasoning in a 3D open world. https://mazebench.com, July 2026a. Project oversight by Florian Brand and Sebastian Müller.

Jonathan Pappas and David Pappas. Introducing MazeBench. https://mazebench.com/blog?post= introducing-mazebench, July 2026b. Published July 28, 2026.

Hanna M. Pasula, Luke S. Zettlemoyer, and Leslie Pack Kaelbling. Learning symbolic models of stochastic domains. Journal of Artificial Intelligence Research, 29:309–352, 2007. doi: 10.1613/jair.2113.

Long Phan, Mantas Mazeika, Andy Zou, and Dan Hendrycks. TextQuests: How good are LLMs at text-based video games? arXiv preprint arXiv:2507.23701, 2025.

Wasu Top Piriyakulkij, Cassidy Langenfeld, Tuan Anh Le, and Kevin Ellis. Doing experiments and revising rules with natural language and probabilistic reasoning. arXiv preprint arXiv:2402.06025, 2024. URL https://arxiv.org/abs/2402.06025.

Wasu Top Piriyakulkij, Yichao Liang, Hao Tang, Adrian Weller, Marta Kryven, and Kevin Ellis. PoE-World: Compositional world modeling with products of programmatic experts. arXiv preprint arXiv:2505.10819, May 2025.

Linlu Qiu, Liwei Jiang, Ximing Lu, Melanie Sclar, Valentina Pyatkin, Chandra Bhagavatula, Bailin Wang, Yoon Kim, Yejin Choi, Nouha Dziri, and Xiang Ren. Phenomenal yet puzzling: Testing inductive reasoning capabilities of language models with hypothesis refinement. In International Conference on Learning Representations, 2024. URL https://arxiv.org/abs/2310.08559.

Qwen Team. Qwen3.6-27B model card. https://huggingface.co/Qwen/Qwen3.6-27B, 2026. Released April 2026.

Sergey Rodionov. Executable world models for ARC-AGI-3 in the era of coding agents. In Artificial General Intelligence: 19th International Conference, AGI 2026, San Francisco, CA, USA, July 27–30, 2026, Proceedings, Part II, Lecture Notes in Computer Science, pages 198–210. Springer, 2026. doi: 10.1007/978-3-032-33195-3\_15. arXiv:2605.05138, May 2026.

Joshua S. Rule, Joshua B. Tenenbaum, and Steven T. Piantadosi. The child as hacker. Trends in Cognitive Sciences, 24(11):900–915, 2020.

Julian Schrittwieser, Ioannis Antonoglou, Thomas Hubert, Karen Simonyan, Laurent Sifre, Simon Schmitt, Arthur Guez, Edward Lockhart, Demis Hassabis, Thore Graepel, Timothy Lillicrap, and David Silver. Mastering Atari, Go, chess and shogi by planning with a learned model. Nature, 588(7839):604–609, 2020. doi: 10.1038/s41586-020-03051-4.

SeungWon Seo, DongHeun Han, SeongRae Noh, and HyeongYeop Kang. Baba in Wonderland: Online self-supervised dynamics discovery for executable world models. arXiv preprint arXiv:2605.16725, 2026. URL https://arxiv.org/abs/2605.16725.

Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning. In Advances in Neural Information Processing Systems, 2023.

Alexy Skoutnev, Kirill Acharya, Gaston Longhitano, Madeleine Udell, Kevin Ellis, and Iddo Drori. Twin: Playing an unknown game with a test-time digital twin. arXiv preprint arXiv:2608.14490, August 2026.

Armando Solar-Lezama. Program Synthesis by Sketching. PhD thesis, University of California, Berkeley, 2008. Technical Report UCB/EECS-2008-177.

Armando Solar-Lezama, Liviu Tancau, Rastislav Bodík, Sanjit Seshia, and Vijay Saraswat. Combinatorial sketching for finite programs. In Proceedings of the 12th International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS), pages 404–415. ACM, 2006. doi: 10.1145/1168857.1168907.

Richard S. Sutton. Integrated architectures for learning, planning, and reacting based on approximating dynamic programming. In Proceedings of the Seventh International Conference on Machine Learning (ICML), pages 216–224, 1990. doi: 10.1016/B978-1-55860-141-3.50030-4.

Hao Tang, Darren Key, and Kevin Ellis. WorldCoder, a model-based LLM agent: Building world models by writing code and interacting with the environment. In Advances in Neural Information Processing Systems (NeurIPS), 2024.

Joshua B. Tenenbaum, Charles Kemp, Thomas L. Grifiths, and Noah D. Goodman. How to grow a mind: Statistics, structure, and abstraction. Science, 331(6022):1279–1285, 2011.

Pedro A. Tsividis, Joao Loula, Jake Burga, Nathan Foss, Andres Campero, Thomas Pouncy, Samuel J. Gershman, and Joshua B. Tenenbaum. Human-level reinforcement learning through theory-based modeling, exploration, and planning. arXiv preprint arXiv:2107.12544, 2021.

Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. Voyager: An open-ended embodied agent with large language models. Transactions on Machine Learning Research, 2024a.

Ruocheng Wang, Eric Zelikman, Gabriel Poesia, Yewen Pu, Nick Haber, and Noah D. Goodman. Hypothesis search: Inductive reasoning with language models. In International Conference on Learning Representations, 2024b. URL https://arxiv.org/abs/2309.05660.

Xingyao Wang, Yangyi Chen, Lifan Yuan, Yizhe Zhang, Yunzhu Li, Hao Peng, and Heng Ji. Executable code actions elicit better LLM agents. In International Conference on Machine Learning, 2024c.

Archana Warrier, Dat Nguyen, Michelangelo Naim, Moksh Jain, Yichao Liang, Karen Schroeder, Cambridge Yang, Joshua B. Tenenbaum, Sebastian Vollmer, Kevin Ellis, and Zenna Tavares. Benchmarking worldmodel learning with environment-level queries. arXiv preprint arXiv:2510.19788, 2025.

xAI. Grok models and pricing. https://docs.x.ai/developers/models, 2026. Accessed September 2026.

Tianbao Xie, Danyang Zhang, Jixuan Chen, Xiaochuan Li, Siheng Zhao, Ruisheng Cao, Toh Jing Hua, Zhoujun Cheng, Dongchan Shin, Fangyu Lei, Yitao Liu, Yiheng Xu, Shuyan Zhou, Silvio Savarese, Caiming Xiong, Victor Zhong, and Tao Yu. OSWorld: Benchmarking multimodal agents for open-ended tasks in real computer environments. In Advances in Neural Information Processing Systems (NeurIPS), 2024.

John Yang, Carlos E. Jimenez, Alexander Wettig, Kilian Lieret, Shunyu Yao, Karthik Narasimhan, and Ofir Press. SWE-agent: Agent-computer interfaces enable automated software engineering. In Advances in Neural Information Processing Systems (NeurIPS), 2024.

Shunyu Yao, Jefrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. ReAct: Synergizing reasoning and acting in language models. In International Conference on Learning Representations, 2023.

Shunyu Yao, Noah Shinn, Pedram Razavi, and Karthik Narasimhan. τ-bench: A benchmark for toolagent-user interaction in real-world domains. arXiv preprint arXiv:2406.12045, 2024.

Z.ai. GLM-5.2: Built for long-horizon tasks. https://huggingface.co/blog/zai-org/glm-52-blog, 2026. Released June 2026.

Jenny Zhang, Shengran Hu, Cong Lu, Robert Lange, and Jef Clune. Darwin Gödel Machine: Open-ended evolution of self-improving agents. arXiv preprint arXiv:2505.22954, May 2025a.

Qizheng Zhang, Changran Hu, Shubhangi Upasani, Boyuan Ma, Fenglu Hong, Vamsidhar Kamanuru, Jay Rainton, Chen Wu, Mengmeng Ji, Hanchen Li, Urmish Thakker, James Zou, and Kunle Olukotun. Agentic context engineering: Evolving contexts for self-improving language models. arXiv preprint arXiv:2510.04618, October 2025b.

A Extended Related Work 16   
A.1 Agent Harnesses and Knowledge Representation . 16   
A.2 Code World Models . . 16   
A.3 Program Induction and Active Experimentation . 17   
B Implementation Details 17   
C Benchmark Details 18   
C.1 ARC-AGI-3 18   
C.2 DiG-bench . 19   
C.3 MazeBench 19   
D Case Studies 22   
D.1 ARC-AGI-3: Cases Behind the Ablations 22   
D.2 DiG-bench: Discovering a Hidden String Rule 24   
D.3 MazeBench: Reusing a Rule After Long Interaction . 25   
E More Results 26   
E.1 ARC-AGI-3 Per-Game Results 26   
E.2 DiG-bench Repeated Runs . 31   
E.3 MazeBench Gem Collection and Room Coverage 33

## A Extended Related Work

## A.1 Agent Harnesses and Knowledge Representation

An agent’s harness shapes both how it interacts with the world and what it carries forward from those interactions. ReAct (Yao et al., 2023) interleaves reasoning and actions, CodeAct (Wang et al., 2024c) uses executable code as an action interface, and SWE-agent (Yang et al., 2024) designs interfaces for navigating and modifying software repositories. Learning from these interactions also requires a representation in which useful discoveries can accumulate. Reflexion (Shinn et al., 2023) retains verbal reflections on task feedback, while ACE (Zhang et al., 2025b) incrementally develops a playbook of strategies and lessons. Voyager (Wang et al., 2024a) stores executable skills in a growing library, allowing previously learned behaviors to support new tasks. Self-improving systems extend what the agent can revise: the Darwin Gödel Machine (Zhang et al., 2025a) modifies its own agent code, and Prime Agent (Karten et al., August 2026) updates persistent prompts, memories, and skills within a programmable harness.

The interaction record itself can also serve as a durable resource. PRO-LONG (Fox et al., July 2026) preserves a complete, structured history that an agent can query with code; its agents sometimes also construct transition models and search them to plan. VISTA (Han et al., August 2026) combines direct visual observations, free-form language reasoning, and lossless visual memory to support interaction. Schema makes an evolving executable theory the persistent object around which learning and action are organized. The history supplies evidence for checking that theory, and the theory supplies predictions for experiments and plans. This gives accumulated knowledge an operational role: a discovered rule can be tested against earlier observations and used to simulate situations the agent has not yet encountered. The program remains available across context compactions, so later decisions can build directly on earlier discoveries.

## A.2 Code World Models

Model-based reinforcement learning uses a model of the environment to support decision-making and learning. Dyna (Sutton, 1990) integrates learning from real experience with updates based on modelgenerated transitions. MuZero (Schrittwieser et al., 2020) combines a learned dynamics model with tree search, predicting rewards, values, and policies for planning, while DreamerV3 (Hafner et al., 2025) learns behavior through trajectories imagined in a neural world model. Schema shares the use of model predictions to make interaction more productive. It accumulates environment knowledge through revisions to an explicit program while keeping the base language model frozen, making each change available for inspection, replay, and subsequent planning.

Explicit representations of environment mechanisms also have a long history in reinforcement learning. Symbolic approaches learn relational rules describing action efects (Pasula et al., 2007), and theory-based reinforcement learning induces structured theories of objects, interactions, and termination conditions for exploration and planning (Tsividis et al., 2021). Programs ofer a representation in which such knowledge can be inspected, revised, and executed to predict future states.

LLMs make it possible to synthesize these models in general-purpose programming languages. World-Coder (Tang et al., 2024) learns Python transition and reward functions, using consistency with experience and an optimistic planning criterion to guide program refinement through REx search. CWM (Dainese et al., 2024) synthesizes simulators from ofline trajectories, using execution feedback within a Monte Carlo tree search over code generation, improvement, and repair. TheoryCoder (Ahmed et al., 2025) combines symbolic high-level planning with simulation in a learned program; TheoryCoder-2 (Ahmed et al., 2026) further learns reusable abstractions from experience. Other work expands the model class and learning setting. PoE-World (Piriyakulkij et al., 2025) combines programmatic experts into a probabilistic transition model, while One Life to Learn (Khan et al., 2025) studies symbolic modeling of stochastic environments from unguided exploration. Alice (Seo et al., 2026) uses model revisions that explain new transitions but fail on earlier ones to identify missing distinctions and guide exploration.

Schema builds on this foundation at the level of agent architecture: the LLM controls when and how to revise its theory, check it against experience, investigate an uncertain mechanism, or plan toward a goal. This control also covers the program’s state grounding, allowing the agent to develop its representation alongside its transition rules and understanding of task completion. Concurrent work on ARC-AGI-3 likewise investigates executable modeling, interaction feedback, and planning (Rodionov, 2026; Courtis et al., 2026; Furgale et al., 2026; Lehmann et al., 2026; Skoutnev et al., 2026). Our study explores this direction as a general approach to agent design across diferent forms of unfamiliar environments.

## A.3 Program Induction and Active Experimentation

Program induction searches for executable explanations of examples, often using a structured program space to make this search tractable. Programming-by-example methods synthesize programs within domainspecific languages; for example, the string-transformation system of Gulwani (2011) infers programs from input–output examples. Sketching (Solar-Lezama et al., 2006; Solar-Lezama, 2008) completes partial programs against a specification, using counterexamples from a verifier to constrain subsequent candidates. DreamCoder (Ellis et al., 2021) learns both a library of reusable program components and a neural search policy, allowing abstractions discovered across tasks to guide later synthesis. These approaches develop complementary ways to construct, check, and reuse programs as representations of knowledge.

Language models provide a further source of candidate hypotheses. Hypothesis Search (Wang et al., 2024b) translates language hypotheses into executable programs and evaluates them on examples. Qiu et al. (2024) similarly study iterative hypothesis refinement, showing how symbolic interpretation helps evaluate and apply rules proposed by language models. Schema places this synthesis-and-checking process within ongoing interaction: recorded transitions provide tests for successive program revisions, and prediction failures supply concrete evidence for further induction. Replaying the full history checks whether a proposed repair also preserves the mechanisms learned earlier.

In an interactive environment, the learner also chooses which evidence to collect. Piriyakulkij et al. (2024) combine language hypotheses with probabilistic inference and information-theoretic experiment design in a rule-discovery task. BoxingGym (Gandhi et al., 2025) studies experimental design and model discovery together in environments defined by generative probabilistic models. Schema connects this process to action in an unfamiliar environment: the agent can use its current program to reach a situation that tests a suspected mechanism, compare the outcome with its prediction, and incorporate the resulting evidence into the same theory used for task planning. Experimentation and task progress therefore draw on a common, continually revised account of how the environment works.

## B Implementation Details

Schema gives a frozen language model a persistent workspace and tools for the four operations in Section 3 (Figure 2). Benchmark adapters supply observations, actions, and task feedback; the agent decides which operation to invoke next. Below, the program contract follows Equation 1, and tools are grouped by function.

Program Workspace. The workspace stores $P ,$ notes, helper scripts, and $\mathcal { H } _ { t }$ , which the harness extends after every executed action. These artifacts persist across turns, levels, and context compactions. The internal state z represents the current situation, including information inferred from past observations. The harness reconstructs it by replaying the relevant history under the current program, with initialization at level entries and resets as defined by the benchmark. Learned rules therefore remain available as the agent grounds them in each new situation.

A turn supplies the current observation, available actions, progress, and remaining budgets where applicable. The agent invokes tools as needed and ends deliberation by committing actions. Execution returns a new observation and a report of the actions taken and any event that interrupted the sequence.

The program defines how observations map to objects, attributes, and relations, and how information needed across successive actions is represented in $z _ { t }$ . Given $\left( z _ { t } , o _ { t } , a _ { t } \right)$ , the program returns $\left( \widehat { o } _ { t + 1 } , z _ { t + 1 } \right)$ as in Equation 1. The predicted observation includes task feedback such as level completion, failure, or game completion. Recorded observations support replay checks; predicted observations support simulated rollouts. A target predicate $g ( z , o )$ specifies a desired outcome: task completion, an intermediate waypoint, or a situation in which an uncertain rule can be tested.

Hypothesize. File tools let the agent read, write, edit, search, and organize its workspace. It can revise the state representation and transition rules, maintain candidate programs, and develop helper scripts for analyzing observations. The read\_history tool retrieves recorded transitions by level, index, range, or event type, with summaries or detailed before-and-after observations. This lets the agent revisit the evidence for a rule and examine transitions that its current program does not explain.

Certify. The backtest tool replays recorded interactions under the program and compares its predictions with observed outcomes. Replay covers the recorded levels by default; a narrower scope helps localize

a mismatch. The agent can also test a candidate file before adopting it. Feedback includes prediction accuracy, mismatched indices, and predicted versus observed outcomes. The model\_predict tool evaluates a selected action from the current observation and reconstructed program state, allowing the agent to inspect a prediction before acting. Replay checks the observations produced during gameplay and the task feedback indicating completion or failure. For ARC-AGI-3, level changes and terminal screens are checked through their corresponding outcome flags. In the color-rotator example of Figure 2, backtesting identifies the unexplained transition and confirms the revised rule against the accumulated history. Abbreviated reports are:

backtest [all transitions]: 138/139 transitions fully correct; 1 mismatch

mismatched transitions (index:kind): #139:observation

#139 action=2 recorded before / after, predicted after ...

backtest [all transitions]: 139/139 transitions fully correct; 0 mismatches

Plan. The agent selects a target and searches over rollouts of P from the current state, without consuming environment actions (Equation 3). It can use built-in procedures such as BFS and $\mathrm { A ^ { * } }$ , or develop planning code in the workspace. The agent chooses the target, candidate actions, and computational budget, and can supply a heuristic or intermediate waypoint. For ARC-AGI-3, it can select click coordinates and allow a plan to begin with a level reset. A successful search returns an action sequence and search statistics; an unsuccessful search reports whether the explored space was exhausted or a budget was reached. The agent uses this feedback to adjust the search, investigate its model, or commit a plan. For the level in Figure 2:

BFS: goal in 15 steps via level\_up; expanded 2496 nodes, 795 distinct states

Plan (-> commit\_actions): [2, 2, 4, 1, 1, 1, 1, 4, 1, 4, 1, 1, 3, 3, 3]

Act with verification. The commit\_actions tool ends deliberation and submits an action sequence, obtained through planning or chosen directly by the agent for exploration. The harness executes committed actions one at a time, records each transition in $\mathcal { H } _ { t }$ , and checks the program’s prediction against the outcome. Matching predictions allow execution to continue without another LLM call; a mismatch discards the remaining actions and returns the unexpected transition to the agent. Task events such as level completion or failure also return control to the agent. Exploration begins with individual actions whose outcomes provide evidence for building the program. An example interruption report is:

world model MISPREDICTED the step just taken (action 2); the rest of the committed plan was dropped.

ARC-AGI-3 observations are grids of hexadecimal color indices, with rendered frames available for inspection; DiG-bench supplies its native text observations; MazeBench supplies an ASCII rendering of the current room. Each adapter exposes the benchmark’s native actions and task feedback, allowing the same operations to support program revision, replay, planning, and verified execution across the three environments.

## C Benchmark Details

We describe the three benchmarks below and summarize their published model results in Tables 2–4.

## C.1 ARC-AGI-3

ARC-AGI-3 (ARC Prize Foundation, 2026a) presents visual games as 64×64 grids, with mechanics and objectives that must be discovered through interaction. Successive levels recombine these mechanics into increasingly dificult puzzles, testing whether an agent can turn its discoveries into eficient plans. The benchmark contains 25 public games, 55 semi-private games for organizer-run evaluations, and 55 private competition games. The latter two sets are not released for general academic use; we evaluate all 25 public games, comprising 183 levels (Figure 10). The ablations in Section 4.1 use the ten games with the most actions in their human baselines, marked in Table 8; together they contain 75 levels.

Relative human action eficiency (RHAE) combines level completion with action eficiency relative to first-time human play. For a game with L levels of which the agent completes the first k, the score is

$$
\mathrm { R H A E } = \operatorname* { m i n } \left( \frac { \sum _ { l = 1 } ^ { k } l } { \sum _ { l = 1 } ^ { L } l } , \ \frac { \sum _ { l = 1 } ^ { k } l \cdot \operatorname* { m i n } \left( 1 . 1 5 , \ ( h _ { l } / a _ { l } ) ^ { 2 } \right) } { \sum _ { l = 1 } ^ { L } l } \right) ,\tag{4}
$$

![](images/561af6e4e396ce45b6d0d7490d2849ef155003f416376dbcf6903f900a0bdecb.jpg)  
Figure 10: The 25 public ARC-AGI-3 games, each shown at the start of its first level.

where $h _ { l }$ and $a _ { l }$ are the human baseline’s and agent’s action counts on level l. Later levels receive greater weight, and exceeding the human action count incurs a quadratic penalty. We report the mean score over games as a percentage.

## C.2 DiG-bench

DiG-bench (Battleday et ${ \mathrm { a l . } }$ , 2026a) comprises 70 text games across seven dificulty tiers, with one to sixteen levels per game. Observations are strings of letters, numbers, and symbols, and actions are individual characters; both transition rules and win conditions must be inferred. Limited lives and per-level step budgets make informative experiments essential: a game is won only when all levels are cleared before these budgets run out. Many games also provide a creative mode for testing hypotheses outside the main level (Battleday et al., 2026b). Three games per tier are public, while the remaining 49 are held privately for evaluation; our experiments cover all 21 public games.

## C.3 MazeBench

MazeBench (Pappas and Pappas, 2026a) is a three-dimensional exploration and puzzle environment with 256 interconnected rooms and 100 gems (Figure 11). Progress requires discovering mechanisms involving pushable blocks, lifts, bridges, and icy surfaces, then applying them across distant rooms. We use the ASCII track, which renders the current room from a rotatable camera. The interface exposes eleven actions for movement, camera control, undo, room reset, and return to visited rooms (Pappas and Pappas, 2026b). Gems collected and rooms visited measure puzzle-solving and exploration over long interactions,

Table 2: Frontier models on ARC-AGI-3, from ARC Prize’s reports and leaderboard as of September 2026 (ARC Prize Foundation, 2026a,e,d,g,f). Model dates follow the cited vendor documentation; empty cells are unreported. <sup>†</sup>Approximate public score for Fable-class models (ARC Prize Foundation, 2026c). <sup>§</sup>Parentheses show the provider-adapter harness (ARC Prize Foundation, 2026b).
<table><tr><td></td><td></td><td></td><td colspan="2">RHAE (%)</td></tr><tr><td>Model</td><td>Cutoff</td><td>Release</td><td>Public</td><td>Semi-private</td></tr><tr><td>Gemini 3.1 Pro (Google DeepMind, 2026)</td><td>Jan 2025</td><td>Feb 2026</td><td></td><td>0.4</td></tr><tr><td>GPT-5.4 (OpenAI, 2026b)</td><td>Aug 2025</td><td>Mar 2026</td><td></td><td>0.2</td></tr><tr><td>Grok 4.20 (xAI, 2026)</td><td>Sep 2025</td><td>Mar 2026</td><td></td><td>0.1</td></tr><tr><td>Claude Opus 4.7 (Anthropic, 2026d)</td><td>Jan 2026</td><td>Apr 2026</td><td></td><td>0.2</td></tr><tr><td>GPT-5.5 (OpenAI, 2026c)</td><td>Dec 2025</td><td>Apr 2026</td><td></td><td>0.4</td></tr><tr><td>Claude Opus 4.8 (Anthropic, 2026e)</td><td>Jan 2026</td><td>May 2026</td><td></td><td>1.5</td></tr><tr><td>Claude Fable 5 (Anthropic, 2026b)</td><td>Jan 2026</td><td>Jun 2026</td><td>20†</td><td></td></tr><tr><td>Grok 4.5 (xAI, 2026)</td><td>Feb 2026</td><td>Jul 2026</td><td>0.3</td><td></td></tr><tr><td>GPT-5.6 Luna (OpenAI, 2026d)</td><td>Feb 2026</td><td>Jul 2026</td><td>0.0</td><td>0.2</td></tr><tr><td>GPT-5.6 Terra (OpenAI, 2026d)</td><td>Feb 2026</td><td>Jul 2026</td><td>2.3</td><td>0.8</td></tr><tr><td>GPT-5.6 Sol (OpenAI, 2026d)</td><td>Feb 2026</td><td>Jul 2026</td><td>13.3</td><td>7.8</td></tr><tr><td>Grok 4.6 (xAI, 2026)</td><td>Feb 2026</td><td>Aug 2026</td><td></td><td>2.1</td></tr><tr><td>Human (ARC Prize Foundation, 2026a)</td><td>Mar 2026</td><td></td><td>100</td><td>100</td></tr><tr><td>Claude Opus 5 (Anthropic, 2026f)</td><td>May 2026</td><td>Jul 2026</td><td></td><td>30.2</td></tr><tr><td>GPT-6 Astra (OpenAI, 2026e)</td><td>Apr 2026</td><td>Sep 2026</td><td></td><td>62.7 (99.9)§</td></tr></table>

Table 3: Frontier models in the basic harness on all 70 DiG-bench games, as reported by the benchmark authors (Battleday et al., 2026a). Win rates average runs within each game, then games within each set; tiers 6–7 contain 20 games. Each game was solved by at least one first-time human player.
<table><tr><td></td><td></td><td></td><td colspan="2">Win rate (%)</td></tr><tr><td>Model</td><td>Cutoff</td><td>Release</td><td>All tiers</td><td>Tiers 6–7</td></tr><tr><td>Gemini 3.1 Pro (Google DeepMind, 2026)</td><td>Jan 2025</td><td>Feb 2026</td><td>16.7</td><td>0.0</td></tr><tr><td>Qwen 3.6 27B (Qwen Team, 2026)</td><td>Oct 2025</td><td>Apr 2026</td><td>1.4</td><td>0.0</td></tr><tr><td>GPT-5.5 (OpenAI, 2026c)</td><td>Dec 2025</td><td>Apr 2026</td><td>25.7</td><td>10.0</td></tr><tr><td>GLM-5.2 (Z.ai, 2026)</td><td>Mar 2026</td><td>Jun 2026</td><td>15.3</td><td>0.0</td></tr><tr><td>Kimi K3 (Moonshot AI, 2026)</td><td></td><td>Jul 2026</td><td>22.9</td><td>0.0</td></tr><tr><td>Claude Opus 5 (Anthropic, 2026f)</td><td>May 2026</td><td>Jul 2026</td><td>71.4</td><td>40.0</td></tr><tr><td>DeepSeek V4 Flash (DeepSeek-AI, 2026)</td><td></td><td>Jul 2026</td><td>11.4</td><td>0.0</td></tr><tr><td>DeepSeek V4 Pro (DeepSeek-AI, 2026)</td><td>Apr 2026</td><td>Aug 2026</td><td>4.3</td><td>0.0</td></tr><tr><td>Human solvability (Battleday et al., 2026a)</td><td>Aug 2026</td><td></td><td>100</td><td>100</td></tr></table>

making this environment useful for studying how agents accumulate and reuse knowledge.

![](images/01598b280cea6ed1d7aa96360235b68bc40d0b07e950e4a52bc1cf39c331982a.jpg)  
Figure 11: The MazeBench environment. The full world of 256 interconnected rooms, with enlarged views of rooms CxG, FxC, and MxJ. Blue circles and connecting lines locate each room within the world.

Table 4: Frontier models on MazeBench’s ASCII track, from the September 14, 2026 leaderboard (Pappas and Pappas, 2026a). Tools permit Python execution. Gem percentages use each run’s available total (71–100); rooms are out of 256. The human row gives the top-50 median, with gems out of 100; individual entries appear in Table 5.
<table><tr><td></td><td></td><td></td><td colspan="2">With tools</td><td colspan="2">Without tools</td></tr><tr><td>Model</td><td>Cutoff</td><td>Release</td><td></td><td>Gems (%) Rooms</td><td>Gems (%)</td><td>Rooms</td></tr><tr><td>Gemini 3.1 Pro (Google DeepMind, 2026)</td><td>Jan 2025</td><td>Feb 2026</td><td>1.1</td><td>5</td><td>0.0</td><td>1</td></tr><tr><td>GPT-5.4 (OpenAI, 2026b)</td><td>Aug 2025</td><td>Mar 2026</td><td></td><td></td><td>0.0</td><td>2</td></tr><tr><td>GPT-5.5 (OpenAI, 2026c)</td><td>Dec 2025</td><td>Apr 2026</td><td></td><td></td><td>0.0</td><td>4</td></tr><tr><td>Claude Opus 4.8 (Anthropic, 2026e)</td><td>Jan 2026</td><td>May 2026</td><td></td><td></td><td>0.0</td><td>4</td></tr><tr><td>GLM-5.2 (Z.ai, 2026)</td><td>Mar 2026</td><td>Jun 2026</td><td></td><td></td><td>0.0</td><td>1</td></tr><tr><td>Claude Fable 5 (Anthropic, 2026b)</td><td>Jan 2026</td><td>Jun 2026</td><td>14.7</td><td>43</td><td>1.4</td><td>10</td></tr><tr><td>GPT-5.6 Luna (OpenAI, 2026d)</td><td>Feb 2026</td><td>Jul 2026</td><td>0.0</td><td>2</td><td>0.0</td><td>1</td></tr><tr><td>GPT-5.6 Terra (OpenAI, 2026d)</td><td>Feb 2026</td><td>Jul 2026</td><td>6.7</td><td>12</td><td>0.0</td><td>2</td></tr><tr><td>GPT-5.6 Sol (OpenAI, 2026d)</td><td>Feb 2026</td><td>Jul 2026</td><td>17.1</td><td>67</td><td>1.4</td><td>4</td></tr><tr><td>Kimi K3 (Moonshot AI, 2026)</td><td></td><td>Jul 2026</td><td>1.3</td><td>4</td><td>0.0</td><td>2</td></tr><tr><td>Claude Opus 5 (Anthropic, 2026f)</td><td>May 2026</td><td>Jul 2026</td><td>16.0</td><td>62</td><td>1.3</td><td>8</td></tr><tr><td>DeepSeek V4 Pro (DeepSeek-AI, 2026)</td><td>Apr 2026</td><td>Aug 2026</td><td>0.0</td><td>2</td><td>0.0</td><td>1</td></tr><tr><td>Grok 4.6 (xAI, 2026)</td><td>Feb 2026</td><td>Aug 2026</td><td>1.1</td><td>6</td><td></td><td></td></tr><tr><td>Claude Fable 5.1 (Anthropic, 2026c)</td><td>Jun 2026</td><td>Sep 2026</td><td>11.1</td><td>33</td><td>2.0</td><td>12</td></tr><tr><td>GPT-6 Astra (OpenAI, 2026e)</td><td>Apr 2026</td><td>Sep 2026</td><td>23.0</td><td>113</td><td>14.0</td><td>103</td></tr><tr><td>Human top-50 (Pappas and Pappas, 2026a)</td><td>Sep 2026</td><td></td><td></td><td></td><td>29.0</td><td>137</td></tr></table>

Table 5: Top-50 human entries used for the MazeBench reference, comprising submissions through September 19, 2026 (Pappas and Pappas, 2026a). Names are omitted; gems and rooms are out of 100 and 256, respectively, and hours denote recorded playtime.
<table><tr><td>Rank</td><td>Gems</td><td>Rooms</td><td>Actions</td><td>Hours</td><td>Rank</td><td>Gems</td><td>Rooms</td><td>Actions</td><td>Hours</td></tr><tr><td>1</td><td>80</td><td>233</td><td>115,060</td><td>55.99</td><td>26</td><td>29</td><td>104</td><td>49,221</td><td>95.95</td></tr><tr><td>2</td><td>80</td><td>233</td><td>151,663</td><td>27.21</td><td>27</td><td>26</td><td>139</td><td>40,112</td><td>12.31</td></tr><tr><td>3</td><td>80</td><td>233</td><td>260,784</td><td>102.35</td><td>28</td><td>26</td><td>126</td><td>68,310</td><td>13.64</td></tr><tr><td>4</td><td>79</td><td>233</td><td>63,543</td><td>10.82</td><td>29</td><td>21</td><td>174</td><td>27,181</td><td>22.36</td></tr><tr><td>5</td><td>79</td><td>233</td><td>206,116</td><td>29.43</td><td>30</td><td>21</td><td>129</td><td>26,261</td><td>5.04</td></tr><tr><td>6</td><td>79</td><td>233</td><td>221,343</td><td>72.30</td><td>31</td><td>21</td><td>105</td><td>31,332</td><td>11.62</td></tr><tr><td>7</td><td>79</td><td>233</td><td>227,236</td><td>286.17</td><td>32</td><td>21</td><td>104</td><td>31,654</td><td>9.79</td></tr><tr><td>8</td><td>77</td><td>233</td><td>169,758</td><td>99.46</td><td>33</td><td>20</td><td>72</td><td>21,845</td><td>23.48</td></tr><tr><td>9</td><td>68</td><td>209</td><td>119,655</td><td>23.30</td><td>34</td><td>17</td><td>96</td><td>5,001</td><td>21.48</td></tr><tr><td>10</td><td>60</td><td>210</td><td>160,616</td><td>77.97</td><td>35</td><td>16</td><td>70</td><td>25,278</td><td>2.80</td></tr><tr><td>11</td><td>59</td><td>233</td><td>120,523</td><td>80.42</td><td>36</td><td>15</td><td>62</td><td>21,428</td><td>24.80</td></tr><tr><td>12</td><td>55</td><td>199</td><td>141,337</td><td>76.70</td><td>37</td><td>14</td><td>96</td><td>30,209</td><td>5.78</td></tr><tr><td>13</td><td>49</td><td>202</td><td>54,560</td><td>11.56</td><td>38</td><td>13</td><td>105</td><td>31,618</td><td>7.97</td></tr><tr><td>14</td><td>49</td><td>173</td><td>47,275</td><td>51.49</td><td>39</td><td>13</td><td>42</td><td>9,311</td><td>4.76</td></tr><tr><td>15</td><td>45</td><td>174</td><td>42,721</td><td>26.39</td><td>40</td><td>12</td><td>62</td><td>40,629</td><td>5.63</td></tr><tr><td>16</td><td>45</td><td>155</td><td>62,784</td><td>225.78</td><td>41</td><td>10</td><td>33</td><td>11,231</td><td>2.93</td></tr><tr><td>17</td><td>40</td><td>162</td><td>94,565</td><td>23.07</td><td>42</td><td>9</td><td>49</td><td>27,759</td><td>8.52</td></tr><tr><td>18</td><td>38</td><td>171</td><td>66,138</td><td>15.01</td><td>43</td><td>9</td><td>47</td><td>8,299</td><td>9.74</td></tr><tr><td>19</td><td>38</td><td>156</td><td>39,231</td><td>72.14</td><td>44</td><td>9</td><td>20</td><td>10,299</td><td>1.59</td></tr><tr><td>20</td><td>37</td><td>153</td><td>34,971</td><td>9.78</td><td>45</td><td>8</td><td>104</td><td>23,064</td><td>10.40</td></tr><tr><td>21</td><td>36</td><td>165</td><td>62,131</td><td>21.71</td><td>46</td><td>8</td><td>86</td><td>11,620</td><td>13.85</td></tr><tr><td>22</td><td>33</td><td>135</td><td>51,825</td><td>15.06</td><td>47</td><td>8</td><td>65</td><td>4,734</td><td>4.94</td></tr><tr><td>23</td><td>32</td><td>164</td><td>58,978</td><td>31.11</td><td>48</td><td>8</td><td>27</td><td>11,690</td><td>12.61</td></tr><tr><td>24</td><td>30</td><td>209</td><td>43,640</td><td>61.86</td><td>49</td><td>7</td><td>57</td><td>38,219</td><td>9.04</td></tr><tr><td>25</td><td>29</td><td>119</td><td>64,195</td><td>9.42</td><td>50</td><td>7</td><td>41</td><td>6,834</td><td>0.89</td></tr></table>

## D Case Studies

## D.1 ARC-AGI-3: Cases Behind the Ablations

The following cases expand the examples in Section 4.1 using recorded interactions and program revisions from the Claude Opus 4.8 runs.

Program representation: assembling changing shapes (CN04, level 5). The agent moves and rotates pieces until their connectors meet, while keeping their bodies from overlapping. One piece grows when acted on, changing both its outline and its connectors. In the final search, Schema represents bodies and connectors as separate sets of grid cells and enumerates rotations and translations. After four growth actions, the growing piece has seven connectors; the computed arrangement pairs all fourteen connectors into seven joints without any body overlap (Figure 12). The run clears the level in 279 actions, compared with 1,219 for the prose-model variant.

![](images/995f070ea527362b751a0e0c123913f1a36c2535e71863e9883343a4176f039b.jpg)  
Figure 12: Recorded CN04 frames before growth, after four growth actions, and at level completion. The final arrangement joins the pieces without overlapping their bodies.

Certification: a new rule can break an old prediction (AR25). AR25 requires moving shapes and mirror axes so that the shapes and their reflections cover the targets. To predict an arrow-key action, the program must identify which object is currently selected. In the run without certification, a revision on level 3 addresses a piece whose selection marker is obscured by a target. But the revised inference also changes predictions for the already completed first level: agreement on the same sixteen recorded transitions falls from 16/16 to 1/16. Across the history available at the two checkpoints, agreement falls from 42/43 (97.7%) to 30/46 (65.2%). Further revisions also introduce an error in the reflection geometry: moving a piece one cell left should move its reflection one cell right, but the later program moves the reflection two cells (Figure 13). With certification, Schema’s final model reproduces 267/268 transitions (99.6%).

![](images/2844e0c53c629c9e8902740ce326a491f32cb8ab3bd6fea4da7a7c476f879d23.jpg)  
Figure 13: The same historical AR25 transition, shown with identical crops. Boxes mark the reflection; dashed lines mark its original position. The turn-15 program predicts the observed one-cell movement exactly, while the turn-39 program predicts two cells.

Planning: coordinating transformations and movement (LS20, level 5). The agent carries a symbol that must match the goal in both shape and color. Contact with a moving marker rotates the symbol, so reaching the right location is insuficient: the player and marker must meet at the right time, with enough energy left to reach the goal. The variant without planning enters this level with a program that passes all 252 historical transition checks, but works out its routes and transformations manually and takes 233 actions to finish. Schema finishes in 65 actions. Its final thirteen-action plan begins with a down–up pair that makes the player and moving marker converge, rotating the carried symbol into the target shape; the remaining eleven actions reach the goal (Figure 14).

![](images/765dfc610c975ddd58b6a3775ed812966b2a19e4717db8bdcf62ff983de59788.jpg)

![](images/9a97a48131d628f83f4faac9c5a1fb6fe737620350d9f1d7ad15cce5956d84f1.jpg)

![](images/968342a09b6ad3764eefc48c885db3da53e2a7348241a21d3265692a78d5c9d6.jpg)  
Figure 14: Recorded LS20 observations before and after the first two actions of the final plan. Enlarged crops show the carried symbol rotating to match the target after the down–up pair. The remaining eleven actions reach the goal.

Verified execution: noticing a spent resource (LS20, level 2). Here the agent must rotate its symbol and reach the goal while replenishing a limited energy supply. The run without verification learns that yellow stations refill energy, but initially treats them as reusable. It commits a 43-action plan that relies on returning to a previously used station. The very first action already exposes the mistake: after the player leaves a station, the model predicts that the station remains, while the observation shows bare floor (Figure 15). Execution continues for all 42 remaining actions, including the return to the spent station, and the energy supply is exhausted before the batch ends. Only afterward does the agent revise the program to consume a station after use and require a still-present station for refueling.

![](images/7066f9f92e52648d2c0929eb2d61f6c11e0ca178f02544f2a86f201fe21088ad.jpg)

![](images/f2793e9072edd2efa0d4cde9b9a1fe9ad32751f0ccaa507a8c474b2affaae792.jpg)

![](images/8151a3a45316239c38a1230bff0559fd0f160887a6ecce6d0b4c79537b340b87.jpg)

![](images/41666b5de95f4f138ce02b2d23aa3df67bccda5dfb95753ce12c8d430f4e4ccc.jpg)  
Figure 15: The first prediction mismatch in the 43-action LS20 batch without verification. Identical crops highlight the station’s location: the prediction retains it, but the observation shows floor. Execution continues through another refill and eventual energy depletion; the rebound after zero is an automatic reset.

## D.2 DiG-bench: Discovering a Hidden String Rule

The puzzle. P-3 asks the agent to discover which strings over a, b, c, and d satisfy a hidden rule. Practice queries return yes or no without costing a life; the agent then takes an eight-question quiz, where each wrong classification costs a life. We follow level 3 of a Schema run with GPT-6 Astra at medium efort.

From letter positions to letter counts. After a is accepted, the agent first encodes the hypothesis that accepted strings end in a. The rejection of ba contradicts this rule, so it adds the requirement that the string also start with a. The next queries expose the problem with that explanation: aba and ac are both accepted, although only the former ends in a. The agent replaces the positional rule with s.count $( \ ' { } a \ ' ) >$ s.count $( ^ { , } { \mathsf { b } } ^ { , } )$ . It then tests baa, abba, and cad: rearranging the letters preserves acceptance, adding a second b removes it, and surrounding a with other letters preserves it (Table 6).

<table><tr><td>Current hypothesis</td><td>New practice queries</td><td>Consequence</td></tr><tr><td>Initial probe</td><td>a: yes</td><td>Propose a rule involving a.</td></tr><tr><td>Ends in a</td><td>b, c, d, ab, ba: no</td><td>ba rules out the current hypothesis.</td></tr><tr><td>Starts and ends in a</td><td>aa, aba, ac: yes</td><td>ac rules out this positional rule.</td></tr><tr><td>More a&#x27;s than b&#x27;s</td><td>baa: yes; abba: no; cad: yes</td><td>All three agree with the count rule.</td></tr></table>

Table 6: All twelve practice queries in P-3 level 3, grouped chronologically by the program hypothesis in use. Hypothesis descriptions summarize the recorded code edits.

Checking the rule and taking the quiz. Each revision is checked against the accumulated interaction history. After the final practice queries, the program reproduces all 107 checkable transitions collected so far, including the earlier levels and the string-entry mechanics. Schema then classifies all eight quiz strings correctly without losing a life (Table 7). Across the full game, this run clears all eight levels with one life lost; the basic harness at medium efort stops at levels 3, 4, and 5 in our three runs.

<table><tr><td>String</td><td>dca</td><td>a</td><td>dbdacb</td><td>bdcaaa</td><td></td><td></td><td>dcbdba acd bcabca adbcc</td><td></td></tr><tr><td>#a - #b</td><td>1</td><td>1</td><td>-1</td><td>2</td><td>-1</td><td>1</td><td>0</td><td>0</td></tr><tr><td>Answer</td><td>yes</td><td>yes</td><td>no</td><td>yes</td><td>no</td><td>yes</td><td>no</td><td>no</td></tr><tr><td>Correct</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td></tr></table>

Table 7: The eight scored questions in order. The count diference is shown to make the learned rule easy to check.

## D.3 MazeBench: Reusing a Rule After Long Interaction

The puzzle. On ice, one directional action can carry the player across several tiles. The player stops when blocked or upon reaching ordinary ground, so solving an ice maze requires planning a sequence of stopping points rather than simply tracing a walkable path to the gem. We follow the same Schema run with GPT-6 Astra from its first encounter with ice in room IxG to a later maze in room NxE.

Learning a reusable movement rule. At action 2,629, a downward move in IxG slides the player across two tiles and stops just before a wall. The agent adds sliding to its movement program and tests further directions. Nine subsequent probes all match its predictions, including a slide that ends on the first ordinary tile beyond the ice (Figure 16a). By action 2,638, the revised program reproduces all 2,638 recorded transitions.

(a) IxG: discovery, action 2,629  
![](images/76fd9b475871647cbfc4b6f273bb2144000966fce7329aec80ea7258dd3a10ff.jpg)

(b) NxE: reuse, action 27,490  
![](images/922416288914bf17ca49f86cbc964c48f6c92969ab507aa040f9008fbf6668b6.jpg)  
Figure 16: Ice-rule discovery and reuse, shown on top-down game-engine views. Arrows trace selected probes in IxG (left), which stop at a wall or upon leaving ice, and the complete 38-action route to the gem in NxE (right).

Applying the rule in a new room. Schema first enters NxE at action 27,486. From several viewpoints and an initial movement, it recovers the wall and ice geometry, including the ordinary entry tile initially hidden by the player. Comparing the program immediately before entering NxE with the version used for the final plan shows that all 58 top-level functions and classes are unchanged: the additions import and register the new room’s geometry. The existing movement code supplies the consequences of sliding through this new layout.

Planning after repeated context compaction. At action 27,490, A\* search finds a 38-action route to the gem, 24,852 actions after the IxG probes (Figure 16b). The saved session records at least 35 context compactions between discovery and this plan. All 38 actions execute, and each complete observation matches its saved forward prediction; the last action collects the gem at action 27,528.

## E More Results

## E.1 ARC-AGI-3 Per-Game Results

Table 8 reports every public game for the four base models, and Figures 17–20 trace each run level by level against the human baseline. With Claude Fable 5, Schema clears all 183 levels in 57% of the human baseline’s actions and uses fewer actions than the human on 163 of them; with GPT-5.6 Sol at maximum efort it also clears every level, in 63% of the human actions.

Table 8: Per-game ARC-AGI-3 results of Schema with each base model. Levels is the number of levels in the game and Human the total actions of the human baseline; for each model, RHAE is the game’s score and Actions counts actions through the last cleared level. Parentheses give completion counts for partially cleared games. <sup>∗</sup>The ten games with the most human actions, used for the ablations in Section 4.1.
<table><tr><td></td><td></td><td></td><td colspan="2">Opus 4.8</td><td colspan="2">Fable 5</td><td colspan="2">Sol xhigh</td><td colspan="2">Sol max</td></tr><tr><td>Game</td><td>Levels</td><td>Human</td><td>RHAE</td><td>Actions</td><td>RHAE</td><td>Actions</td><td>RHAE</td><td>Actions</td><td>RHAE</td><td>Actions</td></tr><tr><td>AR25*</td><td>8</td><td>748</td><td>100.0</td><td>269</td><td>100.0</td><td>298</td><td>100.0</td><td>261</td><td>100.0</td><td>278</td></tr><tr><td>BP35</td><td>9</td><td>651</td><td>62.9</td><td>1,265</td><td>93.5</td><td>566</td><td>28.8</td><td>218 (5/9)</td><td>60.9</td><td>1,347</td></tr><tr><td>CD82</td><td>6</td><td>171</td><td>100.0</td><td>121</td><td>100.0</td><td>144</td><td>100.0</td><td>116</td><td>100.0</td><td>86</td></tr><tr><td>CN04*</td><td>6</td><td>789</td><td>100.0</td><td>479</td><td>100.0</td><td>324</td><td>100.0</td><td>318</td><td>100.0</td><td>241</td></tr><tr><td>DC22*</td><td>6</td><td>1,228</td><td>38.9</td><td>485 (4/6)</td><td>98.7</td><td>1,205</td><td>100.0</td><td>1,018</td><td>100.0</td><td>814</td></tr><tr><td>FT09</td><td>6</td><td>208</td><td>100.0</td><td>94</td><td>100.0</td><td>78</td><td>100.0</td><td>97</td><td>100.0</td><td>92</td></tr><tr><td>G50T*</td><td>7</td><td>879</td><td>100.0</td><td>544</td><td>96.4</td><td>486</td><td>100.0</td><td>457</td><td>100.0</td><td>306</td></tr><tr><td>KA59</td><td>7</td><td>730</td><td>100.0</td><td>431</td><td>100.0</td><td>436</td><td>100.0</td><td>414</td><td>100.0</td><td>430</td></tr><tr><td>LF52*</td><td>10</td><td>1,339</td><td>23.8</td><td>475 (5/10)</td><td>100.0</td><td>1,030</td><td>100.0</td><td>875</td><td>100.0</td><td>844</td></tr><tr><td>LP85</td><td>8</td><td>388</td><td>100.0</td><td>134</td><td>100.0</td><td>99</td><td>100.0</td><td>120</td><td>100.0</td><td>98</td></tr><tr><td>LS20*</td><td>7</td><td>776</td><td>100.0</td><td>642</td><td>100.0</td><td>497</td><td>100.0</td><td>398</td><td>100.0</td><td>479</td></tr><tr><td>M0R0*</td><td>6</td><td>1,107</td><td>100.0</td><td>221</td><td>100.0</td><td>278</td><td>100.0</td><td>271</td><td>100.0</td><td>242</td></tr><tr><td>R11L</td><td>6</td><td>233</td><td>100.0</td><td>83</td><td>100.0</td><td>111</td><td>100.0</td><td>90</td><td>100.0</td><td>85</td></tr><tr><td>RE86*</td><td>8</td><td>1,255</td><td>100.0</td><td>615</td><td>100.0</td><td>635</td><td>100.0</td><td>822</td><td>100.0</td><td>742</td></tr><tr><td>S5I5</td><td>8</td><td>638</td><td>89.9</td><td>643</td><td>100.0</td><td>294</td><td>100.0</td><td>356</td><td>100.0</td><td>318</td></tr><tr><td>SB26</td><td>8</td><td>213</td><td>82.5</td><td>347</td><td>98.6</td><td>135</td><td>100.0</td><td>127</td><td>100.0</td><td>131</td></tr><tr><td>SC25</td><td>6</td><td>350</td><td>93.9</td><td>363</td><td>100.0</td><td>334</td><td>42.5</td><td>1,285</td><td>82.7</td><td>359</td></tr><tr><td>SK48*</td><td>8</td><td>1,070</td><td>85.1</td><td>801</td><td>100.0</td><td>443</td><td>63.9</td><td>1,402</td><td>87.8</td><td>977</td></tr><tr><td>SP80</td><td>6</td><td>518</td><td>56.3</td><td>450 (5/6)</td><td>100.0</td><td>283</td><td>100.0</td><td>301</td><td>100.0</td><td>164</td></tr><tr><td>SU15</td><td>9</td><td>361</td><td>61.5</td><td>444</td><td>100.0</td><td>158</td><td>80.3</td><td>593</td><td>100.0</td><td>160</td></tr><tr><td>TN36</td><td>7</td><td>317</td><td>75.3</td><td>348</td><td>94.7</td><td>210</td><td>80.3</td><td>584</td><td>87.0</td><td>900</td></tr><tr><td>TR87</td><td>6</td><td>414</td><td>100.0</td><td>138</td><td>100.0</td><td>174</td><td>100.0</td><td>208</td><td>100.0</td><td>155</td></tr><tr><td>TU93</td><td>9</td><td>462</td><td>100.0</td><td>243</td><td>100.0</td><td>195</td><td>100.0</td><td>255</td><td>100.0</td><td>246</td></tr><tr><td>VC33</td><td>7</td><td>447</td><td>81.8</td><td>507</td><td>99.1</td><td>342</td><td>100.0</td><td>261</td><td>100.0</td><td>240</td></tr><tr><td>WA30*</td><td>9</td><td>1,843</td><td>100.0</td><td>956</td><td>100.0</td><td>1,080</td><td>100.0</td><td>1,403</td><td>100.0</td><td>980</td></tr><tr><td>Mean total Games won</td><td>183</td><td>17,135</td><td>86.1</td><td>11,098</td><td>99.2</td><td>9,835</td><td>91.8</td><td>12,250</td><td>96.7</td><td>10,714</td></tr></table>

![](images/155aa580255ce5f4777886b2332218982f065b79efbd0609b9ecbbfd6ac9ade0.jpg)  
Figure 17: Per-game progress on ARC-AGI-3 with Claude Opus 4.8 in Schema. Curves connect cumulative action counts at level completion for Schema and the human baseline. The number in each panel is the game’s RHAE.

![](images/50e5969e8097f1b5cfd4df2f59579ccfe8c3b9b18246be4901634c013f1f8274.jpg)  
Figure 18: Per-game progress on ARC-AGI-3 with Claude Fable 5 in Schema, as in Figure 17.

![](images/ec561178cd778763406a63c6f06e23cafd0847c1a16c5c7fca46da11f9bba94f.jpg)  
Figure 19: Per-game progress on ARC-AGI-3 with GPT-5.6 Sol at extra-high efort in Schema, as in Figure 17.

![](images/6f21495cd0eefa66d9a74a5dfe7517f1ee267af11dec98e029fa2be5d35f383f.jpg)  
Figure 20: Per-game progress on ARC-AGI-3 with GPT-5.6 Sol at maximum efort in Schema, as in Figure 17.

## E.2 DiG-bench Repeated Runs

We ran every DiG-bench configuration three times with GPT-6 Astra. Table 9 gives the mean and standard deviation of the games won and the levels completed, and Figures 21 and 22 break them down by reasoning efort, dificulty tier, and game. Schema wins more games than the basic harness in every run at every reasoning efort, and its advantage is largest at medium efort, where it wins 19.3 games on average against 8.7; at medium efort it already matches the basic harness at maximum efort. The gains come from the hardest games: across all eforts and runs, Schema wins 161 of the 162 game runs in tiers 1–6 and 17 of the 27 in tier 7, against 5 for the basic harness.

Table 9: DiG-bench with GPT-6 Astra over three runs: mean ± standard deviation of the games won (of 21) and the levels completed (of 193).
<table><tr><td></td><td></td><td>Medium High</td><td>Max</td></tr><tr><td>Games won</td><td>Basic harness</td><td> $8 . 7 \pm 2 . 5$ </td><td> $1 5 . 0 \pm 1 . 0$   $1 8 . 0 \pm 1 . 0$ </td></tr><tr><td></td><td>Schema  ${ \bf 1 9 . 3 \ : \pm { \ : 0 . 6 } }$ </td><td> ${ \bf 1 9 . 7 \pm 0 . 6 }$ </td><td> ${ \bf 2 0 . 3 \ : \pm { \ : 0 . 6 } }$ </td></tr><tr><td>Levels completed</td><td>Basic harness  $1 1 8 . 0 \pm 1 6 . 7$ </td><td> $1 5 6 . 3 \pm 5 . 5$ </td><td> $1 7 8 . 0 \pm 5 . 2$ </td></tr><tr><td></td><td>Schema</td><td> ${ \bf 1 7 9 . 7 \pm 4 . 6 }$   ${ \bf 1 8 3 . 7 \pm 2 . 3 }$ </td><td> ${ \bf 1 8 7 . 7 \pm 4 . 6 }$ </td></tr></table>

![](images/e674444cba59248c2c67fcd0673acc9ba8949a401296264625fe2334926e783f.jpg)

![](images/a3bd0cdab4704cb499717471598b596a5bfc9a1332fbce367af965d2a4772f68.jpg)

(c) games won by tier, all eforts and runs  
![](images/af241ea3cd2a7bb348989724ae067861f7a5a6e9c0a1cd8b9e76ecf014aa8ff5.jpg)  
Figure 21: DiG-bench with GPT-6 Astra over three runs. (a) Games won and (b) levels completed at each reasoning efort: bars show the mean and error bars one standard deviation. (c) Share of games won in each dificulty tier, over all eforts and runs.

![](images/052eae656d26ce8df574d6eb82edaa9e47fffcf6c478c9aa7e214dc2655282cb.jpg)  
Figure 22: Fraction of levels completed in each DiG-bench game at each reasoning efort, mean and standard deviation over three runs, games ordered by tier.

## E.3 MazeBench Gem Collection and Room Coverage

Figure 23 maps room coverage for the four trajectories in Figure 8. Schema and Codex with GPT-6 Astra share 106 visited rooms: 33 additional rooms appear only in Schema’s trajectory and seven only in Codex’s. Schema also reaches every room visited by each of the other two baselines.

(a) Schema

GPT-6 Astra

![](images/f2cd80a0d1157feda85810747dc959f004041af781ecf6123c7f62d1b400060e.jpg)  
139 rooms (54.3%)

(b) Codex  
GPT-6 Astra  
![](images/efe2b4c73256e444eb0ef1dac9856a3a889c33024735a42fca8dbbaf581ee93e.jpg)  
113 rooms (44.1%)

(c) Codex  
GPT-5.6 Sol  
![](images/5bf3833b0398a2a4f5beb28fe0724a33feffe24fcd5f1fc1eca15e34e3cbabae.jpg)  
67 rooms (26.2%)

(d) Claude Code Opus 5  
![](images/2c7de5a210022dc8350838a023b529b6d55fb21ad683d25c1d6c321b7ad84ac4.jpg)  
62 rooms (24.2%)  
Visited room (panel color)  
Unvisited room Start: HxI  
Figure 23: Room coverage of the four MazeBench runs. Each cell is a room in the same 16 × 16 grid, with columns and rows A–P (e.g., CxG is column C, row G). Panels show each run’s recorded endpoint. Baselines use the September 14, 2026 ASCII tools leaderboard, as in Figure 8.

Table 10 lists all 33 gems, collected across 31 rooms. Coordinates are recovered from recorded top views and the gem’s disappearance at collection. Steps count all issued actions, including camera changes, undo, reset, and room returns. Counts start at one: raw history index i corresponds to step i + 1, so the final index 27,818 represents 27,819 actions.

Table 10: All 33 gems collected by Schema. Read down the left block, then the right. Room-local (x, y) coordinates range from 0 to 15 in the unrotated top view, increasing rightward and downward. Step is the cumulative action count; ∆ counts actions since the previous gem (from the start for gem 1), including exploration elsewhere.
<table><tr><td>Gem</td><td>Room</td><td>(x, y)</td><td>Step</td><td>∆</td><td>Gem</td><td>Room</td><td>(x, y)</td><td>Step</td><td>Δ</td></tr><tr><td>1</td><td>HxH</td><td>(1,3)</td><td>100</td><td>100</td><td>18</td><td>ExF</td><td>(2,6)</td><td>11,173</td><td>730</td></tr><tr><td>2</td><td>GxH</td><td>(10, 1)</td><td>303</td><td>203</td><td>19</td><td>ExC</td><td>(13,12)</td><td>11,478</td><td>305</td></tr><tr><td>3</td><td>HxF</td><td>(1, 1)</td><td>512</td><td>209</td><td>20</td><td>DxG</td><td>(14,2)</td><td>12,469</td><td>991</td></tr><tr><td>4</td><td>GxF</td><td>(2,9)</td><td>749</td><td>237</td><td>21</td><td>IxJ</td><td>(7,14)</td><td>12,643</td><td>174</td></tr><tr><td>5</td><td>FxE</td><td>(11,2)</td><td>1,001</td><td>252</td><td>22</td><td>JxL</td><td>(14, 14)</td><td>13,759</td><td>1,116</td></tr><tr><td>6</td><td>FxF</td><td>(10, 2)</td><td>1,116</td><td>115</td><td>23</td><td>IxE</td><td>(1, 10)</td><td>15,700</td><td>1,941</td></tr><tr><td>7</td><td>IxI</td><td>(1,13)</td><td>2,731</td><td>1,615</td><td>24</td><td>MxN</td><td>(1,15)</td><td>17,168</td><td>1,468</td></tr><tr><td>8</td><td>JxG</td><td>(12,8)</td><td>3,170</td><td>439</td><td>25</td><td>NxN</td><td>(8,8)</td><td>17,815</td><td>647</td></tr><tr><td>9</td><td>JxE</td><td>(1, 1)</td><td>3,463</td><td>293</td><td>26</td><td>MxJ</td><td>(3,13)</td><td>18,779</td><td>964</td></tr><tr><td>10</td><td>MxD</td><td>(12,5)</td><td>4,163</td><td>700</td><td>27</td><td>MxJ</td><td>(12, 11)</td><td>19,535</td><td>756</td></tr><tr><td>11</td><td>JxH</td><td>(4,5)</td><td>4,425</td><td>262</td><td>28</td><td>MxJ</td><td>(5,5)</td><td>19,554</td><td>19</td></tr><tr><td>12</td><td>GxI</td><td>(4,12)</td><td>4,744</td><td>319</td><td>29</td><td>MxI</td><td>(11,5)</td><td>19,742</td><td>188</td></tr><tr><td>13</td><td>CxE</td><td>(2,3)</td><td>5,019</td><td>275</td><td>30</td><td>OxK</td><td>(4, 11)</td><td>20,247</td><td>505</td></tr><tr><td>14</td><td>DxH</td><td>(6,5)</td><td>5,182</td><td>163</td><td>31</td><td>MxL</td><td>(14,8)</td><td>21,732</td><td>1,485</td></tr><tr><td>15</td><td>CxD</td><td>(3,12)</td><td>6,042</td><td>860</td><td>32</td><td>FxB</td><td>(13, 13)</td><td>26,497</td><td>4,765</td></tr><tr><td>16</td><td>FxC</td><td>(0,4)</td><td>7,244</td><td>1,202</td><td>33</td><td>NxE</td><td>(11, 14)</td><td>27,528</td><td>1,031</td></tr><tr><td>17</td><td>NxF</td><td>(13,11)</td><td>10,443</td><td>3,199</td><td></td><td></td><td></td><td></td><td></td></tr></table>