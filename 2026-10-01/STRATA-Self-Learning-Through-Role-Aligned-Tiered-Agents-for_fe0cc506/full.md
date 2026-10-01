# STRATA: Self-Learning Through Role-Aligned Tiered Agents for Real-Time Strategy Games

Xinhe Tian<sup>1</sup>, Xiaoyue Zhang<sup>1</sup>, Ziyou Zhang<sup>2</sup>, Jiacheng Li<sup>3</sup>, Xiaoqiang Jin<sup>2</sup>, Qianchuan Zhao<sup>4</sup>, Gaochen Cui<sup>5,\*</sup>

<sup>1</sup>School of Instrumentation Science and Optoelectronic Engineering, Beijing University of Aeronautics and Astronautics <sup>2</sup>Tsinghua University; <sup>3</sup>University of Chinese Academy of Sciences <sup>4</sup>Department of Automation, Tsinghua University; <sup>5</sup>CFINS, Tsinghua University <sup>\*</sup>Corresponding author

## Abstract

Real-time strategy (RTS) games require agents to coordinate economic development, production and construction, base defense, unit organization, and attack timing over long matches. Existing studies have applied large language models to command decision-making in RTS games, enabling agents to read textual game states and generate high-level plans. However, long inference latency can cause them to miss critical tactical events. The complexity and tactical diversity of full RTS matches also leave existing systems heavily dependent on manually written experience-based prompts, with limited ability to learn continuously from past games. We present STRATA, a role-aligned hierarchical system with cross-game self-learning for Red Alert. STRATA assigns in-game strategic, logistical, and tactical decisions to a Strategic Agent (SA), Logistics Agent (LA), and Tactical Agent (TA), respectively. The SA generates high-level directives based on the global game state and relevant experience cards, while the LA and TA handle logistics and tactical execution. After each match, a Review Agent (RA) derives candidate experience from game traces, validates and revises it using evidence from subsequent matches, and compresses strategic experience supported across multiple games into concise experience cards for SA retrieval. We evaluate STRATA through the formation of experience cards, full-match comparisons before and after learning, and experience learning against AI opponents with diferent play styles. Under a fixed scenario, using the learned experience cards increases the observed win rate from 30% to 100%. Sequential learning against AI opponents with diferent play styles also produces distinct long-term strategic experience.

## 1 Introduction

Real-time strategy (RTS) games require agents to coordinate economic development, production, technology, base defense, and unit operations under partial observability (Ontañón et al. 2013; Vinyals et al. 2017; Samvelyan et al. 2019). Resource investments determine future production capacity. Defensive pressure can disrupt expansion and technology schedules, while attack timing depends on current forces, reinforcements, and the opponent’s state. The trade-of among economy, technology, and military production is therefore central to RTS strategy (Ontañón et al. 2013; Robertson and Watson 2014). Many decisions reveal their efects only minutes later, and opponents continually change the conditions under which earlier plans were made. Agents must therefore revise their strategies over long horizons (Vinyals et al. 2017, 2019). RTS games combine partial observability, large state and action spaces, long decision chains, and delayed feedback (Vinyals et al. 2017; Samvelyan et al. 2019; Vinyals et al. 2019). Their explicit rules, traceable processes, and measurable outcomes provide a controlled setting for evaluating long-horizon planning, online decision-making, and real-time execution (Vinyals et al. 2017; Samvelyan et al. 2019; Andersen, Goodwin, and Granmo 2018). StarCraft II has been widely used for intelligent-agent research (Vinyals et al. 2017), while Red Alert has recently been introduced as a diagnostic benchmark for long-horizon LLM decisionmaking (Li et al. 2026).

Reinforcement learning has long been a major approach for building game-playing agents. Early deep reinforcement learning systems learned control policies directly from pixels (Mnih et al. 2015). Later systems combined search, self-play, and large-scale distributed training for Go, chess, shogi, Dota 2, and StarCraft II (Silver et al. 2016, 2018; Berner et al. 2019; Vinyals et al. 2019). However, these systems often require extensive environment interaction, sustained self-play, and substantial computation (Silver et al. 2018; Berner et al. 2019; Vinyals et al. 2019). Large language models provide another approach. Gato addressed tasks across games, text, and robotic control (Reed et al. 2022), while CICERO combined language communication with strategic reasoning in a long-horizon game that involves both cooperation and competition (Meta Fundamental AI Research Diplomacy Team (FAIR) et al. 2022).

Recent studies use large language models for command decision-making in RTS games. Textual observations, state summaries, and action interfaces allow these models to make high-level economic, technological, and combat decisions (Ma et al. 2024; Li et al. 2025b; Shao et al. 2024). However, resources, production queues, unit positions, and local engagements may all change during a single inference cycle. A model that handles strategy, logistics, and combat must also process a broader input and reasoning scope. TextStar-Craft II, LLM-PySC2, and SwarmBrain improve sequential state summaries, interaction interfaces, and rapid tactical responses, respectively (Ma et al. 2024; Li et al. 2025b; Shao et al. 2024). Even so, model decision cycles can still lag behind real-time game changes.

Existing systems also depend heavily on manually written experience-based prompts. TextStarCraft II identifies the lack of feedback loops and long-term memory as important limitations (Ma et al. 2024). Reflection, language memory, and skill reuse can exploit past feedback, but these methods have mainly been evaluated in open-ended environments such as Minecraft. Compared with full RTS matches, such environments involve less continuous adversarial interference. Their stored experience therefore needs less frequent revision as opponent strategies and game phases change.

![](images/9778f32d4d8ea896b847ce517b6e2b17bb6b753d0e43ecad459e738d225481e0.jpg)  
Figure 1: Overview of STRATA. The system combines role-aligned hierarchical decision-making during each match with a closed loop for post-game experience formation, cross-game validation, and retrieval in later matches.

To address these problems, we present STRATA (Self-Learning Through Role-Aligned Tiered Agents for Real-Time Strategy Games), a role-aligned hierarchical system with cross-game self-learning for full RTS matches. STRATA includes three in-game agents and one post-game review agent. The Strategic Agent (SA) makes strategic decisions and sends high-level directives to the Logistics Agent (LA) and Tactical Agent (TA). The LA and TA use rolespecific observations to execute logistics and tactical actions, respectively. The three in-game agents difer in information scope, decision authority, and update frequency. This separation allows strategic, logistical, and tactical decisions to run independently. It also prevents low-frequency strategic inference from serially blocking logistics and tactical execution. After each match, the Review Agent (RA) examines the game trace and current experience state. It proposes new candidate experience or revises existing experience. Strategic guidance supported across multiple matches is then compressed into concise experience cards for the SA to retrieve in later matches.

We evaluate STRATA in Red Alert through experiencecard formation, full-match comparisons before and after learning, and experience learning against Rush AI and Turtle AI. Across the experiments, the RA continually forms and revises strategic experience from match traces, producing a formal experience library with 16 cards. When the SA uses these cards, the observed win rate increases from 30% to 100%. The system also develops distinct long-term experience for AI opponents with diferent play styles.

In summary, our main contributions are as follows:

1. We propose a role-aligned hierarchical architecture composed of an SA, LA, and TA. The architecture separates strategic, logistical, and tactical decisions by information scope, decision authority, and update frequency.

2. We propose an RA-driven cross-game experiencelearning method for the SA. The RA uses evidence from subsequent matches to validate and revise candidate experience continuously, then accumulates supported strategic experience in experience cards.

3. We evaluate STRATA from three perspectives: experience-card formation, full-match performance before and after learning under a fixed setting, and sequential experience learning against Rush and Turtle opponents. These experiments show how experience cards are formed, how they afect full-match performance, and how they evolve under diferent opponent styles.

## 2 Related Work

LLM agents in interactive environments. Large language model agents extend language-based reasoning to continuous interaction with an environment. Chain-of-Thought exposes intermediate reasoning steps, ReAct alternates reasoning and action, and Tree of Thoughts and LATS search over alternative reasoning and action paths (Wei et al. 2022; Yao et al. 2023a,b; Zhou et al. 2024). These methods support goal decomposition and multi-step planning. However, they are mainly evaluated in textual, web-based, or well-structured tasks in which the environment usually waits for inference to finish.

Research on games also studies long-horizon planning in interactive environments. MineDojo supports open-ended Minecraft tasks, while GITM decomposes long-term goals into subtasks and actions (Fan et al. 2022; Zhu et al. 2023). ProAgent combines independent planning with teammateintention inference for decentralized cooperation (Zhang et al. 2024). Other studies organize multiple language-model agents through hierarchies and specialized roles (Ahn, Kim, and Choi 2025; Deng et al. 2024; Qi et al. 2025). These works demonstrate task allocation and information exchange, but they mainly study open-ended exploration, cooperation, or relatively stable interactions.

LLM agents for RTS games. TextStarCraft II organizes economic, technological, and combat states through singleframe and multi-frame summaries for sequential high-level planning (Ma et al. 2024). $\mathrm { L L M - P y S C } \bar { 2 }$ expands action spaces, multimodal observations, and multi-agent interfaces for complete StarCraft II tasks (Li et al. 2025b). These systems make LLMs easier to integrate with RTS environments. However, state representations still trade information coverage for input length: concise summaries may omit local changes, while richer observations and action spaces increase the reasoning burden. A single model may therefore struggle to operate across diferent decision time scales.

SwarmBrain adds rapid-response mechanisms to highlevel planning so that some local events do not need to wait for the next full inference (Shao et al. 2024). However, its triggers and behaviors require manual design and remain limited by rule coverage. Other work adds domain knowledge through adaptation and expert prompts (Khan and Sukthankar 2024; Li et al. 2025a), preserves state through visua or structured observations (Ma et al. 2026), and improves action executability through hierarchies, role-based decisions, code policies, and state machines (Ahn, Kim, and Choi 2025; Deng et al. 2024; Qi et al. 2025). These approaches improve state understanding, action generation, or response speed. However, major decisions often remain in one core agent or control loop and still require manual maintenance, larger contexts, or predefined rules. Strategy, logistics, and tactics difer in both information needs and rates of change. How to separate their time scales without blocking local execution remains underexplored.

Reflection, memory, and experience learning. Naturallanguage feedback and external memory allow large language models to reuse experience without parameter updates. STaR constructs examples from successful reasoning, Self-Refine repeatedly revises outputs using model feedback (Zelikman et al. 2022; Madaan et al. 2023), and Reflexion stores verbal reflections after failures for later attempts (Shinn et al. 2023). These methods can reduce repeated errors, but their reflections often rely on a limited number of trajectories. Incorrect attribution or changing conditions may therefore cause the agent to rely too heavily on one explanation.

ExpeL extracts reusable experience by comparing successful and failed trajectories (Zhao et al. 2024), while Generative Agents combine retrieval, reflection, and planning over time (Park et al. 2023). GITM, Voyager, and JARVIS-1 use task histories, skill libraries, and multimodal memory for longterm exploration in Minecraft (Zhu et al. 2023; Wang et al. 2024, 2023). These studies show that past experience can guide later decisions, but they usually assume relatively stable goals. In full RTS matches, strategic efects vary with opponent behavior, economy, game phase, and force composition. Conclusions from a single match therefore require continued correction through later supporting evidence and counterexamples.

## 3 STRATA

## 3.1 System Overview

In a full RTS match, strategic judgment, logistics planning, and local combat have diferent information needs and rates of change. STRATA therefore organizes in-game decisionmaking as a role-aligned hierarchical architecture composed of a Strategic Agent (SA), Logistics Agent (LA), and Tactical Agent (TA). The SA uses the global game state and relevant experience to set priorities among economy, technology, force accumulation, and attack timing. It then sends high-level directives to the LA and TA. The LA handles construction, production, repair, and resource allocation, while the TA handles defense, rallying, advances, and local combat. Each agent uses role-specific observations and operates independently at its own time scale. Logistics and tactical execution therefore do not need to wait for every low-frequency strategic inference to finish. Figure 1 summarizes the in-game hierarchy and the cross-game experience-learning loop.

After each match, a Review Agent (RA) reads the complete game trace and current experience state. It organizes strategic evidence and proposes new candidate experience or revisions to existing experience. Experience supported by later matches is compressed into concise experience cards for the SA to retrieve in subsequent matches. The RA updates cross-game experience, the SA uses this experience during a match, and the LA and TA execute logistics and tactical actions. In this way, experience cards transfer findings from completed matches into later strategic decisions.

## 3.2 Role-Aligned Hierarchical In-Game Decision-Making

A full RTS state contains economic, technological, production, force-level, and local-combat information. STRATA derives three role-specific observations from the shared game state according to the needs of each decision type. Each in-game agent can act only within its assigned decision authority. Let $s _ { t }$ denote the shared game state at time t, and let $U _ { t } ^ { \mathrm { T A } }$ denote the set of combat units controlled by the TA. The observations are

$$
\begin{array} { r l } & { o _ { t } ^ { \mathrm { S A } } = \phi _ { \mathrm { S A } } ( s _ { t } ) , \qquad o _ { t } ^ { \mathrm { L A } } = \phi _ { \mathrm { L A } } ( s _ { t } ) , } \\ & { o _ { t } ^ { \mathrm { T A } } = \phi _ { \mathrm { T A } } ( s _ { t } , U _ { t } ^ { \mathrm { T A } } ) . } \end{array}\tag{1}
$$

The SA observes the economic phase, power state, technology progress, total force strength, opponent pressure, relevant experience cards, and compressed logistics and tactical states. The LA observes cash, construction prerequisites, production queues, available production options, repair needs, and current production capacity. The TA receives only its assigned combat units, local opponent information, base positions, and current tactical objective. Each observation contains the information needed for strategic planning, logistics decisions, or local combat. This design keeps each agent’s reasoning within its assigned role.

Role boundaries also constrain agent outputs. The SA issues high-level directives to the LA and TA. The LA manages construction, production, repair, and rally-point settings, while the TA controls its assigned combat units. Their decision process is

$$
\begin{array} { r l } & { ( d _ { t } ^ { \mathrm { L A } } , d _ { t } ^ { \mathrm { T A } } ) = \pi _ { \mathrm { S A } } ( o _ { t } ^ { \mathrm { S A } } , C _ { t } ) , } \\ & { ~ a _ { t } ^ { \mathrm { L A } } = \pi _ { \mathrm { L A } } ( o _ { t } ^ { \mathrm { L A } } , d _ { t } ^ { \mathrm { L A } } ) , } \\ & { ~ a _ { t } ^ { \mathrm { T A } } = \pi _ { \mathrm { T A } } ( o _ { t } ^ { \mathrm { T A } } , d _ { t } ^ { \mathrm { T A } } ) , } \end{array}\tag{2}
$$

where $C _ { t }$ is the set of experience cards relevant to the current situation, and $d _ { t } ^ { \operatorname { L A } }$ and $d _ { t } ^ { \operatorname { T A } }$ are the high-level logistics and tactical directives. The $\mathrm { S A }$ selects the current strategic priority among economy, technology, force accumulation, and attack timing. The LA and TA then choose executable actions from their latest observations.

For example, suppose the SA decides to establish a technology base while withstanding early pressure. It can direct the LA to maintain power and basic production capacity, and direct the TA to protect the base and ore fields. The LA chooses a concrete construction and production order based on current cash, queues, and building conditions. The TA decides whether to defend, rally, or counterattack based on opponent positions and available units. Both agents adapt their execution around the same strategic objective while retaining decision space within their roles. Role-specific observations also reduce the input scope of each model call. The SA, LA, and TA can therefore focus on global strategy, logistics, and the local battlefield, respectively.

The SA, LA, and TA operate asynchronously at diferent time scales. The Red Alert environment continues to advance and publish the latest state. Each agent maintains its current valid output and tracks any pending model request. The SA reassesses the global strategy at a lower frequency. The LA periodically reads the latest economic and production state and updates its plan when it receives a new logistics directive. The TA replans when it receives a new tactical directive, while its current plan remains active until the update is complete.

For each in-game agent $r \in \{ \mathrm { S A } , \mathrm { L A } , \mathrm { T A } \}$ , the decision trigger is summarized as

$$
\mathrm { d e c i d e } _ { r } ( t ) = \left\{ \begin{array} { l l } { 1 , } & { t - \tau _ { r } ^ { \mathrm { l a s t } } \geq \Delta _ { r } , } \\ { 1 , } & { r \in \{ \mathrm { L A } , \mathrm { T A } \} \land \mathrm { n e w } ( d _ { t } ^ { r } ) , } \\ { 0 , } & { \mathrm { o t h e r w i s e } , } \end{array} \right.\tag{3}
$$

where $\tau _ { r } ^ { \mathrm { l a s t } }$ is the time when agent r last initiated a decision, and $\Delta _ { r }$ is its regular decision interval. The SA updates its strategic judgment mainly at regular intervals. The LA and TA can also start a new decision when they receive a new role-specific directive.

Algorithm 1 STRATA In-Game Hierarchical Decision   
Input: Red Alert environment E and formal experience li  
brary M   
Output: Full-match trajectory T   
1: Initialize shared state, directives $( d ^ { \mathrm { L A } } , d ^ { \mathrm { T A } } )$ , and current   
actions $( a ^ { \mathrm { L A } } , a ^ { \mathrm { T A } } )$   
2: while E is not terminated do   
3: Advance $\mathcal { E }$ using current $( a ^ { \mathrm { L A } } , a ^ { \mathrm { T A } } )$ and publish the   
latest state $s _ { t }$   
4: if an SA result is available then   
5: Update high-level directives $( d ^ { \mathrm { L A } } , d ^ { \mathrm { T A } } )$   
6: end if   
7: if an LA result is available then   
8: Update the current logistics action $a ^ { \mathrm { L A } }$   
9: end if   
10: if a TA result is available then   
11: Update the current tactical action $a ^ { \mathrm { T A } }$   
12: end if   
13: if SA is triggered and no SA request is pending then   
14: Construct $o _ { t } ^ { \operatorname { S A } }$ and retrieve relevant experience   
cards $C _ { t }$   
15: LaunchAsync $\pi _ { \mathrm { S A } } ( o _ { t } ^ { \mathrm { S A } } , C _ { t } )$   
16: end if   
17: if LA is triggered and no LA request is pending then   
18: Construct $\mathbf { \Sigma } _ { o _ { t } ^ { \mathrm { L A } } }$   
19: LaunchAsync $\pi _ { \mathrm { L A } } ( o _ { t } ^ { \mathrm { L A } } , d ^ { \mathrm { L A } } )$   
20: end if   
21: if TA is triggered and no TA request is pending then   
22: Construct $\mathbf { \Sigma } _ { O _ { t } ^ { \mathrm { T A } } } ^ { - \mathrm { { T A } } }$   
23: LaunchAsync $\pi _ { \mathrm { T A } } ( o _ { t } ^ { \mathrm { T A } } , d ^ { \mathrm { T A } } )$   
24: end if   
25: Append states, directives, actions, and outcomes to T   
26: end while   
27: return T

Algorithm 1 summarizes the asynchronous in-game decision process.

While one agent waits for a model response, the environment and the other agents continue to run with their current valid outputs. Until a new SA result arrives, the previous high-level directives remain active. The LA continues its logistics plan using the latest economic and production state, while the TA controls units with its current tactical plan. This asynchronous mechanism allows low-frequency SA reasoning to proceed in parallel with logistics and tactical execution. Production, construction, and local combat continue to use recent game states while model requests complete independently.

## 3.3 Review-Agent-Driven Cross-Game Experience Learning

STRATA represents cross-game experience as strategic preferences for the SA and stores these preferences as experience cards in a formal experience library. Each card describes several strategic directions and their relative strengths for a specific situation. An experience card is represented as

$$
\begin{array} { r l } & { c _ { i } = \left. s _ { i } , B _ { i } , e _ { i } , w _ { i } , q _ { i } \right. , } \\ & { B _ { i } = \left\{ ( b _ { i 1 } , \alpha _ { i 1 } ) , \dots , ( b _ { i m } , \alpha _ { i m } ) \right\} . } \end{array}\tag{4}
$$

Here, $s _ { i }$ describes the situation in which the experience applies, $b _ { i j }$ denotes a possible strategic direction, and $\alpha _ { i j }$ gives its preference strength. The field $e _ { i }$ provides a concise example ofa high-level directive. The values $w _ { i }$ and $q _ { i }$ denote the card weight and confidence, respectively.

For example, when the agent already has a war factory, an experience card may recommend maintaining basic vehicle production, reserving resources for technology, and keeping a safe power margin. The SA evaluates these preferences together with the current game state. Under high opponent pressure and insuficient forces, it gives higher priority to basic production and cash preservation. When defensive pressure is low and the economy is stable, it can place more weight on technology development and advanced-unit production. These preferences help the SA choose the strategic priority of the current phase from the current pressure, economy, and force conditions.

During a match, the system selects a small set of relevant experience cards from the formal library M according to applicability, weight, and confidence:

$$
C _ { t } = \mathrm { T o p K } _ { c _ { i } \in M } \left[ \mathrm { M a t c h } ( s _ { i } , \hat { o } _ { t } ^ { \mathrm { S A } } ) \cdot w _ { i } \cdot q _ { i } \right] ,\tag{5}
$$

where $\hat { o } _ { t } ^ { \operatorname { S A } }$ is the global state summary used for experience matching. The RA maintains the cross-game experience that enters the SA’s in-game decision context. The SA combines the current game state with the strategic preferences in $C _ { t }$ and then issues high-level directives to the LA and TA.

After match $e ,$ the RA organizes strategic evidence $E _ { e }$ from the full trajectory $T _ { e }$ . This evidence includes economic and production schedules, opponent pressure, key engagements, changes in high-level directives, experience-card retrieval, and subsequent outcomes. The RA also reads the formal experience library $M _ { e - 1 }$ and candidate pool $P _ { e - 1 }$ saved after the previous match:

$$
R _ { e } = \pi _ { \mathrm { R A } } \left( E _ { e } , M _ { e - 1 } , P _ { e - 1 } \right) ,\tag{6}
$$

where $R _ { e }$ is the review result produced by the RA. The RA assigns each finding to a new strategic theme, an existing candidate, or a formal experience card. For an existing theme, it records supporting evidence, counterexamples, or revision evidence. A new theme enters the candidate pool and may be merged with semantically related candidates. Existing themes continue to accumulate evidence across matches. With suficient support, candidate experience can be promoted to formal experience. Later outcomes can also revise the applicability conditions, strategic preferences, weight, and confidence of formal experience. Repeated high-pressure matches gradually strengthen preferences for basic forces, defense, and cash preservation. Repeated low-pressure matches strengthen preferences for economic expansion, technology, and proactive attacks. Side efects observed in later matches trigger further revisions, allowing the experience library to adapt as game conditions change.

Algorithm 2 Review-Agent-Driven Cross-Game Experience   
Learning   
Input: Completed trajectory $T _ { e } ,$ formal experience library   
M, and candidate pool $P$   
Output: Updated M and $P$   
1: Extract strategic evidence $E _ { e }$ from $T _ { e }$   
2: $R _ { e }  \pi _ { \mathrm { R A } } ( \bar { E } _ { e } , M , P )$   
3: for all finding $r \in R _ { e }$ do   
4: if r introduces a new strategic theme then   
5: Add r to $P ,$ or merge it with a related candidate   
6: else   
7: Accumulate supporting, counterexample, or revi  
sion evidence for the related experience   
8: end if   
9: end for   
10: for all candidate $p \in P$ do   
11: if $p$ has suficient cross-game support then   
12: Promote $p$ into M   
13: end if   
14: end for   
15: for all formal experience $c \in M$ do   
16: if suficient revision evidence is available then   
17: Revise the situation, strategic preferences, weight,   
or confidence of $c$   
18: end if   
19: end for   
20: return M, P

In the next match, the SA retrieves the updated formal experience. It selects relevant cards from the current global state and uses them to issue high-level directives to the LA and TA. After the match, the new trajectory returns to the RA and provides further evidence for candidate and formal experience. STRATA thus turns strategic findings from sequential matches into reusable cross-game experience.

Algorithm 2 summarizes the post-game experience update process.

## 4 Experiments

## 4.1 Experimental Setup

All experiments are conducted in an agent environment built on the OpenRA engine (OpenRA Developers 2026) and its Red Alert mod. The environment provides structured gamestate observations and executable action interfaces. We use GLM-5.1 as the decision model in every experiment and keep all other environment and runtime settings unchanged. The supplementary material reports the map, scenario seed, agent update intervals, and software versions. We evaluate three AI opponents. Normal AI follows a relatively balanced development and combat style. Rush AI emphasizes rapid early attacks, whereas Turtle AI prioritizes base defense, economic growth, and late-game technology.

We first run 80 sequential matches between STRATA and Normal AI. After each match, the RA updates the candidate and formal experience. We then run ten full matches before learning and ten after learning. Before learning, the SA receives no experience cards. After learning, it receives the 16 formal experience cards formed during the 80 matches. The experience state remains fixed during evaluation, and the 80 experience-learning matches are separate from all evaluation matches.

Finally, the Rush and Turtle experiments start from the same 16 formal experience cards and use separate memory spaces. Each condition runs for ten sequential matches, and the RA updates experience after every match. These experiments compare the long-term strategic preferences that emerge under diferent opponent play styles.

## 4.2 Experience-Card Formation

We first examine whether STRATA can form reusable strategic experience across sequential full matches. The system plays 80 matches against Normal AI and conducts a postgame review after each match using the process trace, final outcome, and current experience state. Early matches reveal several recurring problems: delayed opening production, imbalanced harvester counts, insuficient vehicle production capacity, premature technology development, and poorly timed defensive counterattacks. The system initially forms 10 formal experience cards from these recurring problems. The cards cover the main strategic components of economy, production, technology, defense, and ofense. Table 1 presents an abridged example of the experience-card format.

id barracks\_seed\_basic\_   
defense\_001   
when Infantry production is available and fewer than six   
infantry units are currently fielded.   
bias Train Rifle Infantry: 0.35;   
train Rocket Soldiers: 0.40;   
preserve the cash floor: 0.60; pause below 300 cash:   
0.40; . . .   
example Direct the LA to form a small mixed infantry defense   
group, such as three Rifle Infantry and three Rocket   
Soldiers. Keep at least 500 cash after queueing, stop   
below 300 cash, and avoid continuously filled queues   
or additional infantry production buildings.   
weight 0.9   
confidence 1.0

Table 1: An abridged view of an experience card. The complete card representation is provided in the supplementary material.

As more matches are played, early experience is tested in more complex situations. Some rules prove too broad, such as always switching to an attack after completing technology. Other rules identify a useful direction but omit conditions related to the economy, opponent pressure, or current forces. The RA removes or replaces overly general experience and records new problems as candidate cards. Seven candidates accumulate cross-game evidence in later matches and enter the formal experience library. These candidates mainly concern technology transitions, defensive counterattacks, economic expansion, and attack windows. Experience also shifts from isolated action suggestions to longer-term strategic relationships. Examples include limiting early defensive-unit batches to preserve economic resources, advancing technology only after the basic economy and minimum defense become stable, regrouping after defending the base, and expanding ore fields and harvester counts during safe windows.

By the end of the 80-match sequence, the formal experience library contains 16 cards. As the sequence progresses, fewer independent themes require new cards. Updates instead focus on refining the applicability conditions, strategic preferences, and example boundaries of existing cards. These matches reveal more specific side efects, including continuous cash consumption from basic infantry production, technology development that still begins too early, excessive waiting after forces are assembled, and premature attacks with insuficient units. The system revises card applicability, strategic preferences, and example boundaries accordingly. It completes 14 revisions across 9 formal cards. These revisions include limiting production batches, adding cash and queue conditions, delaying technology under low power, and resetting rally and attack conditions.

After 80 matches against Normal AI, STRATA forms 16 formal experience cards covering economy, production, technology, defense, and ofense. Later updates mainly refine their applicability conditions and strategic preferences.

## 4.3 Full-Match Performance Before and After Learning

Table 2 compares full-match results against Normal AI before and after experience learning. STRATA wins 3 of 10 matches before learning and all 10 matches after learning, increasing the observed win rate from 30% to 100%. The mean and median end times decrease from 28,173.1 and 22,800.0 ticks to 20,842.8 and 17,380.5 ticks, respectively. Both conditions use the same environment configuration and fixed experience state; the only diference is whether the SA receives the learned experience cards.

<table><tr><td>Condition</td><td>W/L</td><td>Win rate</td><td></td><td>e Mean end Median end</td></tr><tr><td>Before learning</td><td>3/7</td><td>30%</td><td>28,173.1</td><td>22,800.0</td></tr><tr><td>After learning</td><td>10/0</td><td>100%</td><td>20,842.8</td><td>17,380.5</td></tr></table>

Table 2: Full-match results before and after learning under a fixed scenario; end times are in ticks.

## 4.4 Experience Learning under Diferent Play Styles

Rush AI and Turtle AI expose diferent process-level problems. Against Rush AI, 7 of 10 matches contain long periods with production capacity but insuficient cash, and 6 of 10 show competition between the technology chain and lightunit production. Early pressure therefore forces a limited economy to fund combat, production, and technology at the same time. Against Turtle AI, 7 of 10 matches show vehiclequeue congestion together with idle infantry production, and 7 of 10 retain unspent cash in the middle or late game. Five of 10 miss opportunities for economic expansion under low pressure, and 4 of 10 lack advanced or siege units in the late game. Compared with Rush AI, the Turtle condition makes less efective use of safe windows and converts economic resources into production capacity and attacks less efectively.

These process diferences also change the content of formal experience cards. Both conditions end with 16 formal cards, but three cards develop diferent versions. Table 3 summarizes the main revisions.

<table><tr><td>Experience theme</td><td>Rush condition</td><td>Turtle condition</td></tr><tr><td>Vehicle production</td><td>Lower the production threshold under a thin economy. With fewer than four harvesters, allow cash to fall to about 100 and queue one vehicle at a time to prevent the cash exceeds 1,500 and stop near 800. war factory from idling.</td><td>Use stage-dependent cash floors. Keep about 200 before the second refinery. Afterward, produce vehicles while</td></tr><tr><td>tion expansion</td><td>Economic and produc- Prioritize the second refinery over additional production capacity, radar, and technology to stabilize the basic economy before further development.</td><td>Under low pressure, prioritize a third refinery and another war factory so that economy and production capacity expand together.</td></tr><tr><td></td><td>Post-technology attack Produce advanced units in short batches and wait for an attack window.</td><td>Reduce prolonged rallying and waiting. When at least six combat units are available and few opponents are visible, the SA is more likely to issue a proactive attack directive.</td></tr></table>

Table 3: Main revisions produced from the same initial experience cards under Rush and Turtle conditions.

The RA also proposes new candidates. Against Rush AI, one candidate recommends building the second refinery before advancing technology when pressure is low and only one refinery is available. Against Turtle AI, another candidate recommends using an MCV to establish a second base under sustained low pressure while maintaining a minimum basic force before advancing technology. These candidates address development order under a thin economy, expansion during safe windows, and the balance between technology and basic defense.

At the end of each ten-match sequence, each new candidate has only one supporting observation and therefore remains in the candidate pool. The pool retains single-match strategic proposals until cross-game evidence supports promotion.

![](images/c6e6b62e14303f9f45b338239d00d0c8fd3f86e0f81b15f4cad4e48e0f5179b0.jpg)  
Figure 2: Percentage of SA decisions in which representative formal experience cards were retrieved under Rush and Turtle conditions. One decision may retrieve multiple cards.

Figure 2 compares the retrieval frequencies of representative formal experience cards. Against Rush AI, the SA more often retrieves cards for low-power recovery, short-batch defense, and basic-unit priority. Against Turtle AI, it more often retrieves cards for post-technology quality advances and large-force attacks.

Overall, the experience learned against Rush AI increasingly favors continued production, defense, and economic recovery under a thin economy. The experience learned against Turtle AI emphasizes expansion under low pressure, resource conversion, post-technology quality advances, and proactive attacks. The process diagnoses, formal experience cards, and

SA retrieval distributions point in the same direction. Together, they show that the same initial experience develops into distinct long-term strategic preferences under diferent opponent styles.

## 5 Conclusion and Limitations

We present STRATA, a role-aligned hierarchical system with cross-game self-learning. The SA makes strategic decisions and issues high-level directives, the LA and TA execute logistics and tactical actions, and the RA derives candidate experience from full game traces and continually validates and revises experience cards. In Red Alert, STRATA forms 16 formal cards over 80 matches against Normal AI. Under the fixed scenario, using these cards increases the observed win rate from 30% to 100%. Sequential learning against Rush and Turtle opponents also produces distinct long-term strategic preferences.

The experiments use one map, one fixed seed, one decision model, and a limited number of matches. Future work will expand these settings and evaluate the real-time performance and generalization of alternative role organizations, decision frequencies, and experience-update strategies.

## References

Ahn, D.; Kim, S.; and Choi, J. 2025. Society of Mind Meets Real-Time Strategy: A Hierarchical Multi-Agent Framework for Strategic Reasoning. In Conference on Language Modeling.

Andersen, P.-A.; Goodwin, M.; and Granmo, O.-C. 2018. Deep RTS: A Game Environment for Deep Reinforcement Learning in Real-Time Strategy Games. In 2018 IEEE Conference on Computational Intelligence and Games, 1–8.

Berner, C.; Brockman, G.; Chan, B.; et al. 2019. Dota 2 with Large Scale Deep Reinforcement Learning. arXiv preprint arXiv:1912.06680.

Deng, Y.; Ma, W.; Fan, Y.; Song, R.; Zhang, Y.; Zhang, H.; and Zhao, J. 2024. SMAC-R1: The Emergence of Intelligence in Decision-Making Tasks. arXiv preprint arXiv:2410.16024.

Fan, L.; Wang, G.; Jiang, Y.; et al. 2022. MineDojo: Building Open-Ended Embodied Agents with Internet-Scale Knowledge. In Advances in Neural Information Processing Systems, volume 35, 18343–18362.

Khan, M. J.; and Sukthankar, G. 2024. SC-Phi2: A Fine-Tuned Small Language Model for StarCraft II Build Order Prediction. AI, 5(4): 2338–2352.

Li, J.; Liu, J.; Wang, Y.; Cui, G.; Zhang, X.; Zhao, Q.; Zhang, Z.; and Li, C. 2026. From Winning to Understanding: A Diagnostic Long-Horizon RTS Benchmark for LLMs. In Forty-third International Conference on Machine Learning.

Li, Z.; Lu, C.; Xu, X.; Qi, R.; Ni, Y.; Jiang, L.; Liu, X.; Zhang, X.; Fang, Y.; Huang, K.; and Guo, X. 2025a. Hierarchical Expert Prompt for Large-Language-Model: An Approach Defeat Elite AI in TextStarCraft II for the First Time. arXiv preprint arXiv:2502.11122.

Li, Z.; Ni, Y.; Qi, R.; Lu, C.; Jiang, L.; Xu, X.; Liu, X.; Li, P.; Guo, Y.; Ma, Z.; Li, H.; Wu, H.; Guo, X.; Huang, K.; and Zhang, X. 2025b. LLM-PySC2: StarCraft II Learning Environment for Large Language Models. In Advances in Neural Information Processing Systems, volume 38.

Ma, W.; Fu, Y.; Zhang, Z.; Ghanem, B.; and Li, G. 2026. AVA: Attentive VLM Agent for Mastering StarCraft II. In Findings of the Associationfor Computational Linguistics: ACL 2026, 4270–4290.

Ma, W.; Mi, Q.; Zeng, Y.; Yan, X.; Wu, Y.; Lin, R.; Zhang, H.; and Wang, J. 2024. Large Language Models Play StarCraft II: Benchmarks and A Chain ofSummarization Approach. InAdvances in Neural Information Processing Systems, volume 37.

Madaan, A.; Tandon, N.; Gupta, P.; et al. 2023. Self-Refine: Iterative Refinement with Self-Feedback. In Advances in Neural Information Processing Systems, volume 36.

Meta Fundamental AI Research Diplomacy Team (FAIR); Bakhtin, A.; Brown, N.; Dinan, E.; Farina, G.; Flaherty, C.; Fried, D.; Gof, A.; Gray, J.; Hu, H.; Jacob, A. P.; Komeili, M.; Konath, K.; Kwon, M.; Lerer, A.; Lewis, M.; Miller, A. H.; Mitts, S.; Renduchintala, A.; Roller, S.; Rowe, D.; Shi, W.; Spisak, J.; Wei, A.; Wu, D.; Zhang, H.; and Zijlstra, M. 2022. Human-Level Play in the Game of Diplomacy by Combining Language Models with Strategic Reasoning. Science, 378(6624): 1067–1074.

Mnih, V.; Kavukcuoglu, K.; Silver, D.; et al. 2015. Human-Level Control through Deep Reinforcement Learning. Nature, 518: 529– 533.

Ontañón, S.; Synnaeve, G.; Uriarte, A.; Richoux, F.; Churchill, D.; and Preuss, M. 2013. A Survey of Real-Time Strategy Game AI Research and Competition in StarCraft. IEEE Transactions on Computational Intelligence and AI in Games, 5(4): 293–311.

OpenRA Developers. 2026. OpenRA: What Is OpenRA? https: //www.openra.net/about/. Accessed 2026-07-21.

Park, J. S.; O’Brien, J. C.; Cai, C. J.; Morris, M. R.; Liang, P.; and Bernstein, M. S. 2023. Generative Agents: Interactive Simulacra of Human Behavior. In Proceedings of the 36th Annual ACM Symposium on User Interface Software and Technology.

Qi, R.; Ni, Y.; Jiang, L.; Li, Z.; Huang, K.; and Guo, X. 2025. Memory-Augmented State Machine Prompting: A Novel LLM Agent Framework for Real-Time Strategy Games. arXiv preprint arXiv:2510.18395.

Reed, S.; Zolna, K.; Parisotto, E.; et al. 2022. A Generalist Agent. Transactions on Machine Learning Research.

Robertson, G.; and Watson, I. 2014. A Review of Real-Time Strat egy Game AI. AI Magazine, 35(4): 75–104.

Samvelyan, M.; Rashid, T.; de Witt, C. S.; Farquhar, G.; Nardelli, N.; Rudner, T. G. J.; Hung, C.-M.; Torr, P. H. S.; Foerster, J.; and Whiteson, S. 2019. The StarCraft Multi-Agent Challenge. In Proceedings of the 18th International Conference on Autonomous Agents and MultiAgent Systems, 2186–2188.

Shao, X.; Jiang, W.; Zuo, F.; and Liu, M. 2024. SwarmBrain: Embodied Agent for Real-Time Strategy Game StarCraft II via Large Language Models. arXiv preprint arXiv:2401.17749.

Shinn, N.; Cassano, F.; Gopinath, A.; Narasimhan, K.; and Yao, S. 2023. Reflexion: Language Agents with Verbal Reinforcement Learning. In Advances in Neural Information Processing Systems, volume 36, 8634–8652.

Silver, D.; Huang, A.; Maddison, C. J.; et al. 2016. Mastering the Game of Go with Deep Neural Networks and Tree Search. Nature, 529: 484–489.

Silver, D.; Hubert, T.; Schrittwieser, J.; et al. 2018. A General Reinforcement Learning Algorithm That Masters Chess, Shogi, and Go through Self-Play. Science, 362(6419): 1140–1144.

Vinyals, O.; Babuschkin, I.; Czarnecki, W. M.; et al. 2019. Grandmaster Level in StarCraft II Using Multi-Agent Reinforcement Learning. Nature, 575: 350–354.

Vinyals, O.; Ewalds, T.; Bartunov, S.; et al. 2017. StarCraft II: A New Challenge for Reinforcement Learning. arXiv preprint arXiv:1708.04782.

Wang, G.; Xie, Y.; Jiang, Y.; Mandlekar, A.; Xiao, C.; Zhu, Y.; Fan, L.; and Anandkumar, A. 2024. Voyager: An Open-Ended Embodied Agent with Large Language Models. Transactions on Machine Learning Research.

Wang, Z.; Cai, S.; Liu, A.; et al. 2023. JARVIS-1: Open-World Multi-task Agents with Memory-Augmented Multimodal Language Models. arXiv preprint arXiv:2311.05997.

Wei, J.; Wang, X.; Schuurmans, D.; et al. 2022. Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. In Advances in Neural Information Processing Systems, volume 35, 24824–24837.

Yao, S.; Yu, D.; Zhao, J.; Shafran, I.; Grifiths, T. L.; Cao, Y.; and Narasimhan, K. 2023a. Tree of Thoughts: Deliberate Problem Solving with Large Language Models. In Advances in Neural Information Processing Systems, volume 36.

Yao, S.; Zhao, J.; Yu, D.; Du, N.; Shafran, I.; Narasimhan, K.; and Cao, Y. 2023b. ReAct: Synergizing Reasoning and Acting in Language Models. In International Conference on Learning Representations.

Zelikman, E.; Wu, Y.; Mu, J.; and Goodman, N. D. 2022. STaR: Bootstrapping Reasoning with Reasoning. In Advances in Neural Information Processing Systems, volume 35, 15476–15488.

Zhang, C.; Yang, K.; Hu, S.; et al. 2024. ProAgent: Building Proactive Cooperative Agents with Large Language Models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, 17591–17599.

Zhao, A.; Huang, D.; Xu, Q.; Lin, M.; Liu, Y.-J.; and Huang, G. 2024. ExpeL: LLM Agents Are Experiential Learners. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, 19632–19642.

Zhou, A.; Yan, K.; Shlapentokh-Rothman, M.; Wang, H.; and Wang, Y.-X. 2024. Language Agent Tree Search Unifies Reasoning, Acting, and Planning in Language Models. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings ofMachine Learning Research, 62138–62160.

Zhu, X.; Chen, Y.; Tian, H.; et al. 2023. Ghost in the Minecraft: Generally Capable Agents for Open-World Environments via Large Language Models with Text-Based Knowledge and Memory. arXiv preprint arXiv:2305.17144.