# When Is Smaller Greener? Right-Sizing Language Models for Agentic AI

Sajid Siraj<sup>a,b</sup>, Maria Barbati<sup>b,∗</sup>

<sup>a</sup>Leeds University Business School, University of Leeds, Leeds, United Kingdom

<sup>b</sup>Ca’Foscari University of Venice, Venice, Italy

## Abstract

AI agents use a large language model (LLM) to plan and act over many steps, and a growing number of models compete for this role. This paper asks one question: which model should an agent use, so that its tasks are completed reliably with the least energy, cost and time? Current practice compares models per call, either by choosing the most capable model on a leaderboard or by choosing a smaller and cheaper one. We show that per-call comparisons can mislead for agents. A task succeeds only if all its steps succeed, and failed attempts still use resources. We therefore measure energy, cost and time per successfully completed task, which is the functional unit in the sense of life-cycle assessment. On this basis we develop a general decision model that covers tasks with binary or graded outcomes, and actions that can or cannot be undone. The model gives a simple rule for when a smaller model stops being the greener, cheaper or faster choice. Each resource has its own crossover task length, and the three crossovers divide tasks into three regimes. For short tasks the smaller model is better on every resource, for long tasks the larger model is, and only in between do the preferences of the decision maker matter. We build these results into a two-stage framework that separates eficiency (a preference-free Pareto screen) from preference (a weighted choice within the remaining trade-of). We illustrate the framework with an example in which the same set of models gives three diferent recommendations. A pilot study with 15 open-weight models from five developers, from 0.6 to 27 billion parameters, on two kinds of multi-step agent task with measured GPU energy, finds the predicted structure. The greenest model per completed task changes with task length, from a model of 1 to 3 billion parameters for short tasks to one of 8 or 27 billion for long ones; where it changes depends on the kind of task; and for most pairs of models the observed crossover lengths agree with those predicted from per-step measurements. We set out a full evaluation design to test the framework on agentic benchmarks.

Keywords: Multi-criteria decision analysis; Multi-objective optimisation; Pareto eficiency; Large language models; Agentic AI; Sustainable AI; Functional unit; Satisficing

## 1. Introduction

Large language models (LLMs) are no longer used only to answer questions in a chat window. They now act as the reasoning engine inside AI agents, which plan, call software tools, retrieve information and take actions over many steps (Lei et al., 2025; Zhu, 2026). An agent that handles a customer request, for example, may read a booking record, check a policy, call a payment system and send a confirmation, with the model deciding what to do at each step. At the same time, the number of available models has grown quickly. They range from very large models ofered by a few providers to small open models that can run on modest hardware, and they difer widely in reliability, price, speed and energy use.

This paper asks one question: which language model should an AI agent use, so that its tasks are completed reliably with the least energy, cost and time? We call this the selection question, and it is the thread that runs through the paper. We return to it at the start of each section.

The selection question matters for three reasons. First, the choice is made very often. A deployed agent may run millions of tasks, and each task may involve dozens of model calls. Second, a wrong choice has real consequences. A model that is not reliable enough can send a wrong message or change a record by mistake (Gritta, 2026; Omar, 2026), while a model that is more capable than needed wastes money, time and energy (Yehudai et al., 2025). Third, the energy used by AI inference is now a public concern, and the share used by agents is growing (Oviedo et al., 2026; Elsworth et al., 2025).

Current practice ofers two common answers. The first is to use the most capable model, as judged by a public leaderboard. The second, often presented as the sustainable choice, is to use the smallest or cheapest model that seems good enough (Belcak et al., 2025). Both answers compare models per call: per token, per request or per benchmark question. The same is true of most work on routing requests between models (Chen et al., 2024; Ong et al., 2025) and of recent work on carbon-aware routing (Sheshanarayana et al., 2026).

We argue that per-call comparisons can mislead when the user of the model is an agent. An agent   
completes a task only if every step of the task succeeds. A small diference in the reliability of each step   
therefore becomes a large diference in the success of the whole task, and a failed task must be repeated or   
handed to a person. A model that uses less energy per call may then use more energy per completed task.   
In other words, the smaller model is not always the greener choice, and whether it is depends on the task. This paper makes four contributions to the selection question.

(1) We take the successfully completed task as the unit of comparison, in the sense of the functional unit of life-cycle assessment (International Organization for Standardization, 2006), and build a general decision model on it. The model covers tasks with binary or graded outcomes and actions that can or cannot be undone.

(2) We derive simple results that say when a smaller model stops being the greener, cheaper or faster choice. Each resource has its own crossover task length, and the crossovers divide tasks into three regimes. In two of them the answer needs no preference information at all.

(3) We build these results into a two-stage decision framework. A preference-free Pareto screen removes every model that is worse on all counts, and a weighted choice is made only where a real trade-of remains.

(4) We set out how the framework can be tested, and report a pilot study with 15 open-weight models from five developers on two kinds of agent task, with measured GPU energy. The pilot supports the predicted structure, and for most pairs of models the predicted crossover lengths agree with the observed ones.

The paper speaks to three communities. For AI research, it gives a way to predict when small models are enough for agentic tasks. For operational research, it gives a formal model of a multi-objective choice whose eficient set changes in a predictable way with task length. For sustainability research, it shows how a choice of functional unit changes the environmental comparison of AI systems.

The rest of the paper follows the selection question step by step. Section 2 reviews what is known about it and identifies four gaps. Section 3 uses a simple example to show why per-call comparisons can mislead. Section 4 formalises our answer as a general decision model and a two-stage framework, with an illustrative example. Section 5 asks how the framework can be tested, and Section 6 describes the experiments and reports their results. Section 7 discusses what the results mean, the limits of the approach and the main objections to it. Section 8 returns to the selection question and sets out directions for future work.

## 2. Literature review

To answer the selection question, we need three things: a way to measure how well an agent completes tasks, a way to choose between models, and a way to count the resources that each choice uses. This section reviews the literature on each of them in turn, then the decision-analysis tools we build on, and ends with the gaps that motivate our work.

## 2.1. How AI agents are evaluated

Recent surveys describe LLM agents in terms of reasoning, tool use, coordination between several agents and memory (Lei et al., 2025), and catalogue the protocols used to coordinate agents in practice (Zhu, 2026). Both show that an agent depends on abilities that a single-turn benchmark does not test: planning over many steps, calling tools correctly and staying consistent over a long sequence of actions. Evaluation research has followed this shift. Yehudai et al. (2025) survey agent evaluation and find that cost-eficiency, safety and robustness are rarely measured under realistic conditions. Kapoor et al. (2025) argue that agents should be compared on accuracy and cost together, and show that simple agents can match complex ones at a fraction of the cost. Benchmarks such as τ-bench and its successor (Yao et al., 2024; Barres et al., 2025) test agents in realistic customer-service tasks, and measure how often an agent succeeds on every one of several repeated trials. The Berkeley Function Calling Leaderboard measures how reliably models call tools, from single calls to multi-turn agentic tasks (Patil et al., 2025). Other work shows that outcome-only evaluation can hide failures such as skipped steps or misused tools (Gritta, 2026), and that agents can be misled by unreliable tools (Omar, 2026).

A second line of work studies how success depends on the length of a task. Kwa et al. (2025) measure the length of tasks, in human working time, that models can complete with 50% success, and find that this length has doubled about every seven months. Sinha et al. (2026) show that small gains in the accuracy of single steps lead to large gains in the length of tasks a model can complete, and that errors become more likely when earlier mistakes remain in the context. Mittal (2026) find that task success falls with the number of steps in a way that is well described by a single per-step reliability, which rises with model size but stays below one even for the strongest models. These studies explain why agents fail on long tasks. They do not ask what this means for the choice between models when energy, cost and time are also at stake.

## 2.2. How models are chosen today

A large literature treats model choice as a trade-of between quality and cost, mostly for single requests. FrugalGPT (Chen et al., 2024) and AutoMix (Aggarwal et al., 2023) send a request to a cheap model first and pass it to a stronger model only when needed. Other methods learn to predict which model will answer a request well enough: from past performance (Šakota et al., 2024), from human preference data (Ong et al., 2025), from the predicted dificulty of the request (Ding et al., 2024), or from item response theory (Song et al., 2025). Shah and Shridhar (2025) first narrow a large set of models and then route within it, and RouterBench (Hu et al., 2024) ofers a common benchmark for such methods. More recent work routes the steps of agentic workflows. Fu et al. (2026) show that the true cost of a workflow can be several times the cost of a single call, because of retries and longer reasoning, and Li et al. (2026) choose a model at each step based on the progress of the task. From an operational research view, Yang and Zhu (2026) model the reliability of agentic workflows and allocate a token budget between steps under cost and latency limits. These methods improve the trade-of between quality and cost, but they treat energy and carbon, if at all, only through cost, and they do not tell the decision maker when a smaller model stops being the better choice.

A small number of studies treat model choice as a multi-criteria decision. The closest to ours is Nowak (2026), which frames the routing of business prompts between a cheaper and a stronger model as a managerial decision. Criterion weights are elicited from experts with the Analytic Hierarchy Process (AHP) (Saaty, 1977, 2008), and each prompt is scored on seven criteria before Simple Additive Weighting (SAW) (Hwang and Yoon, 1981; Kaliszewski and Podkopaev, 2016) chooses the model. On 500 business prompts the framework reaches 94.4% response suficiency, against 94.6% for always using the stronger model, at 37.4% lower cost. HugSelect (Joonbakhsh et al., 2026) ranks foundation models from public repositories with a weighted additive model. Both work with single requests, both apply one weighted score to all alternatives, and neither considers energy or the length of a task.

## 2.3. How the resources of AI inference are counted

The energy and carbon cost of AI has been studied since Strubell et al. (2019) drew attention to the cost of training. For inference, Luccioni et al. (2024) show that general-purpose generative models use much more energy than task-specific ones. The ML.ENERGY benchmark (Chung et al., 2025) measures the energy of inference for open models on known hardware, Jegham et al. (2025) estimate the energy, water and carbon of commercial models, and Elsworth et al. (2025) report measured figures for a large commercial service. Oviedo et al. (2026) estimate the energy of inference at scale and note that agentic workflows, with many calls per user request, can have a large efect on the total. Most of this work counts energy per token or per request. Two recent studies move towards the task. Panigrahy and Tyagi (2026) propose energy per successful goal, which counts the energy of failed attempts and retries, and find that agentic workflows can use several times more energy per successful goal than simpler ones. Liu and Li (2026) propose to compare the resources used by an agent with its contribution to the task. On the side of choice, Sheshanarayana et al. (2026) route single requests to minimise carbon subject to accuracy and latency targets, and Belcak et al. (2025) argue that small models are suficient, and more economical, for most agentic tasks. None of these studies links a per-task measure of resources to the choice between models.

## 2.4. Decision-analysis tools

Our framework builds on standard tools of multi-criteria decision analysis. Multi-attribute decision making ranks a given set of alternatives, while multi-objective decision making identifies the alternatives that are eficient, that is, not worse than another alternative on every objective and strictly worse on none (Zeleny, 1982; Miettinen, 1999). Eficient sets can be found by non-dominated sorting (Deb, 2001), and ranked with methods such as SAW and TOPSIS (Hwang and Yoon, 1981; Ciardiello and Genovese, 2023). Weights can be elicited with AHP or, when the decision maker can give only partial information, with the flexible and interactive trade-of method (de Almeida et al., 2016). Three further ideas are central to our model. Satisficing means choosing an alternative that is good enough rather than the best (Simon, 1955). The ε-constraint method turns some objectives into constraints and shows that the solutions found in this way lie on the eficient set (Haimes et al., 1971). Reference-point methods let the decision maker set reservation and aspiration levels for each objective (Wierzbicki, 1980). Finally, the success of a task whose steps must all succeed follows the reliability of a series system, a classical result of reliability theory (Barlow and Proschan, 1975).

## 2.5. Gaps

Returning to the selection question, the review shows that the parts of an answer exist but have not been put together. Table 1 summarises four gaps and how this paper addresses them.

Table 1: Gaps in the literature on the selection question, and how this paper addresses them.
<table><tr><td>Gap</td><td>What is missing</td><td>How we address it</td></tr><tr><td>G1. Unit of compar- ison</td><td>while energy per successful task is only successful task (Section 4.4). measured, not used for choice.</td><td>Models are chosen on per-call measures, A decision model built on resources per</td></tr><tr><td>G2. When is smaller better?</td><td>length, but there is no rule for when (Section 4.5). this reverses the ranking of models on energy, cost or time.</td><td>Task success is known to fall with task Crossover lengths and three regimes</td></tr><tr><td>ences</td><td>no real trade-off exists.</td><td>G3. Role of prefer- Existing multi-criteria methods apply A preference-free screen, with weights weights to all alternatives, even where used only where models are both effi- cient (Sections 4.5 and 4.8).</td></tr><tr><td>G4. Kinds of task</td><td>graded outcomes, or actions that can be (Section 4.6). retried from actions that cannot.</td><td>Methods do not distinguish binary from Four special cases of one general model</td></tr></table>

## 3. Motivation: why per-call comparisons can mislead

Before we formalise our answer to the selection question, we use a simple example to show why the unit of comparison matters.

Suppose model a completes each step of a task correctly 95% of the time and model b 99% of the time. On a single call the gap looks small. Now consider a customer-service task that needs 20 steps and succeeds only if every step succeeds. Model a completes the task with probability 0.95<sup>20</sup> ≈ 0.36 and model b with probability 0.99<sup>20</sup> ≈ 0.82. A failed attempt still uses energy, money and time. If failed tasks are repeated until they succeed, model a needs on average about 2.8 attempts for each success and model b about 1.2. Suppose model a uses one tenth of the energy of model b per call. Per completed task, model a still uses less energy, but its advantage has fallen from ten times to about four times. For a task of 60 steps, model b uses less energy per completed task than model a. The model that looked greener per call is no longer greener.

Life-cycle assessment faces the same problem when it compares products. It solves it with the functional unit, the quantity of service to which all impacts are related (International Organization for Standardization, 2006). A light bulb, for example, is compared per hour of light, not per bulb. We argue that for agentic AI the functional unit should be one successfully completed task. This is in line with recent proposals for goal-level energy accounting (Panigrahy and Tyagi, 2026) and for reporting the cost of agents together with their accuracy (Kapoor et al., 2025). What is new here is that we use this unit to choose between models.

The example raises four questions that a framework for the selection question must answer:

(Q1) Which models can be ruled out for every task, whatever its length?

(Q2) For a task of a given length, which models are eficient, that is, not worse than another model on energy, cost and time together?

(Q3) When does the choice between eficient models depend on the preferences of the decision maker?

(Q4) How does the answer change when actions cannot be undone, or when the quality of the outcome is graded rather than binary?

Section 4 answers these questions in turn.

## 4. A framework for the selection question

This section formalises our answer to the selection question. We first give an overview of the framework and of what it measures. We then present a general decision model, state its main properties, describe its

special cases and relax its assumptions. We close with the choice within a trade-of, an illustrative example and a precise definition of right-sizing.

## 4.1. Overview

Let $\mathcal { A } = \{ a _ { 1 } , \ldots , a _ { m } \}$ denote the set of candidate models. More precisely, each candidate is a model ofered by a provider, since the same model served by two providers can difer in price, speed and terms of use. The set changes over time, as new models appear and old ones are withdrawn. A task profile describes the task for which a model is needed: how many steps it takes, how good the outcome must be, whether a failed attempt can be repeated, how important it is that the task is completed, and how much the deployment cares about energy, cost and time.

The framework maps a set of candidates and a task profile to a recommendation in four steps:

(i) a feasibility screen (Stage 0) removes candidates that break a hard constraint, such as a data-protection rule;

(ii) a per-step screen (Stage 1) removes candidates that are worse than another candidate on every per-step measure; by Observation 2 below, these candidates can never be eficient, for any task;

(iii) a task-level screen finds the candidates that are eficient for the task, measured per successfully completed task, and that meet the required chance of completing it;

(iv) a preference-based choice (Stage 2) picks one of the eficient candidates, but only when more than one remains.

The output is the recommended model together with the eficient set and, where Stage 2 was needed, the weights used. This makes each recommendation open to audit. Figure 1 shows the four steps and the information each one uses.

## 4.2. What we measure

The framework separates properties of a model from properties of a task. Table 2 lists the model quantities. Per-step reliability is measured for a task family, such as customer-service tasks with tool calls, because a model can be reliable in one kind of task and not in another. When no measurement for the task family exists, a general tool-calling score can serve as a first estimate. Context capacity, safety and data governance enter as constraints, not as objectives.

![](images/2be152cb42565bdd6f70065e021842d43eda8abeb56fddba26842496997e4f9d.jpg)  
Figure 1: The framework. Shaded steps use no preference information; Stage 2 (dashed) uses the weights of the task profile and runs only when more than one model is eficient. Dotted arrows show which inputs each step uses.

Table 2: Model quantities used by the framework. Per-step quantities are measured for a task family.
<table><tr><td>Quantity</td><td>Role</td><td>Measurement</td></tr><tr><td> $\rho _ { a } \colon$  per-step reliability</td><td>objective (max)</td><td>share of steps completed correctly in benchmark runs of the task family</td></tr><tr><td> $r _ { a , e } \colon$  energy per step</td><td>objective (min)</td><td>measured on known hardware for open-weight models; estim- ated with bounds for closed models</td></tr><tr><td> $r _ { a , c } .$  cost per step</td><td>objective (min)</td><td>tokens per step multiplied by the provider&#x27;s price</td></tr><tr><td> $r _ { a , t } \mathrm { : }$  time per step</td><td>objective (min)</td><td>measured response time per step</td></tr><tr><td>Context capacity</td><td>constraint</td><td>provider specification</td></tr><tr><td>Safety</td><td>constraint</td><td>published safety evaluations and robustness tests</td></tr><tr><td>Governance</td><td>constraint</td><td>deployment class, data residency, licence</td></tr></table>

Energy is the one quantity that cannot always be measured directly, since providers of closed models do not report it. Each value therefore carries a flag that says whether it was measured or estimated, and estimated values carry a range. Every value also carries a date and a source, because the market changes and the record of where numbers came from is part of the audit trail.

The task profile is described by the parameters in Table 3 below. They can be set in two ways. In the deliberative way, a team sets them once for an agent or a type of task, through a short questionnaire. In the automated way, a small model reads each task description and estimates them at run time, as the router of Nowak (2026) does for its criteria. The rest of the framework is the same in both cases.

## 4.3. Stage 0: feasibility screen

Let Γ denote the hard constraints that follow from the task profile: the context the task needs, the deployment classes allowed by data-protection rules, a minimum safety level, any list of allowed or excluded providers, and any ceiling on cost, time or carbon per task. The feasible set is

$$
{ \cal A } _ { \Gamma } \ = \ \{ a \in { \cal A } : a \mathrm { ~ m e e t s ~ e v e r y ~ c o n s t r a i n t ~ i n ~ } \Gamma \} .\tag{1}
$$

These constraints come first because no amount of reliability or eficiency can make up for a model that may not legally process the data.

## 4.4. A general model of resources per successful task

We now give the general model that answers the selection question for a single task. Consider a candidate model a and a task that needs L steps, where a step is one call to the model. We write $k \in \{ e , c , t \}$ for the three resources, energy, cost and time, so that $r _ { a , k }$ is the amount of resource k that model a uses per step.

The outcome of one attempt at the task may be binary (the task is done correctly or not) or graded, for example a quality score q between 0 and 1. The decision maker sets a quality floor $\underline { { { q } } } ,$ and an attempt counts as a success if its quality reaches the floor. For binary outcomes $\underline { { q } } = 1$ . We write $S _ { a } ( L )$ for the probability that one attempt by model a succeeds. In the basic case, steps succeed or fail independently and any failed step spoils the attempt, as in a series system (Barlow and Proschan, 1975), so that

$$
S _ { a } ( L ) = \rho _ { a } ^ { L } .\tag{2}
$$

Section 4.7 relaxes this assumption.

A failed attempt may be repeated. We allow at most K attempts per task. $K = 1$ describes an irreversible task, such as a payment that cannot be undone. $K = \infty$ describes a task that can be retried freely, such as a data extraction whose result is checked automatically. We assume that each attempt runs all L steps before its outcome is known. This is the cautious case, because it charges every failed attempt in full.

Observation 1 (Resource per successful task). Let $S _ { a } ( L ) > 0$ and $K \geq 1$ . The probability that model a completes the task within K attempts is

$$
P _ { a } ( L , K ) = 1 - \left( 1 - S _ { a } ( L ) \right) ^ { K } ,\tag{3}
$$

and the expected amount of resource k used per successfully completed task is

$$
R _ { a , k } ( L ) = \frac { r _ { a , k } L } { S _ { a } ( L ) } ,\tag{4}
$$

whatever the value of $K$ .

Proof. Attempts succeed independently with probability $S = S _ { a } ( L )$ , which gives (3). The expected number of attempts, with at most K allowed, is $\textstyle \sum _ { i = 0 } ^ { K - 1 } ( 1 - S ) ^ { i } = { \bigl ( } 1 - ( 1 - S ) ^ { K } { \bigr ) } / S$ . Each attempt uses $r _ { a , k } L$ of resource k. Dividing the expected use by the probability of success within K attempts gives (4). □

Observation 1 says that the resource per successful task does not depend on how many retries are allowed. The retry limit changes how often the task is completed at all, but not how eficiently each success is produced. This lets us separate two questions: whether a model is reliable enough for the task, which is a constraint, and how much it uses to deliver each success, which gives the objectives.

The general decision problem follows. For a task with length $L ,$ quality floor $\underline { { q } } ,$ retry limit K and a required probability of completion $\beta ,$ choose a model from the feasible set $\mathcal { A } _ { \Gamma }$ that

$$
\begin{array} { r l } { \mathrm { m i n i m i s e s } } & { { } \left( R _ { a , e } ( L ) , R _ { a , c } ( L ) , R _ { a , t } ( L ) \right) \quad [ \mathrm { a n d , o p t i o n a l l y , m a x i m i s e s } \bar { q } _ { a } ( L ) ] } \end{array}\tag{5}
$$

$$
\mathrm { s u b j e c t ~ t o ~ } P _ { a } ( L , K ) \ge \beta ,
$$

where $\bar { q } _ { a } ( L )$ is the expected quality of a successful output. The problem has several objectives, and we do not combine them at this point. We first find the models that are eficient for (5), and leave the choice among them to Stage 2. The constraint is a chance constraint: it requires the task to be completed with probability at least $\beta . \mathrm { \ B y \ ( 3 ) }$ , it is the same as a lower bound on the success of a single attempt,

$$
S _ { a } ( L ) \geq 1 - ( 1 - \beta ) ^ { 1 / K } .\tag{6}
$$

For example, with $\beta = 0 . 9$ and $K = 3$ , each attempt must succeed with probability at least 0.54. Table 3

lists the parameters of the model, who sets them, and how they relate to the criteria of task profiles used in earlier work (Nowak, 2026).

Table 3: Parameters of the general model (5).
<table><tr><td>Symbol</td><td>Set by</td><td>Meaning and example</td></tr><tr><td> $\rho _ { a }$ </td><td>measurement</td><td>Per-step reliability of model a on the task family.</td></tr><tr><td> $r _ { a , e } , r _ { a , c } , r _ { a , t }$ </td><td>measurement</td><td>Energy, cost and time per step of model  $a .$ </td></tr><tr><td> $L$ </td><td>task profile</td><td>Number of steps in one attempt; follows from the expected length of the task.</td></tr><tr><td>q</td><td>decision maker</td><td>Quality floor; follows from the required accuracy.  $\underline { { q } } = 1$  for tasks that are simply done or not done.</td></tr><tr><td>K</td><td>decision maker</td><td>Maximum number of attempts; follows from whether actions can be undone. K = 1 for irreversible actions,  $K = \infty$  for free retries.</td></tr><tr><td> $\beta$ </td><td>decision maker</td><td>Required probability that the task is completed; follows from the risk of a failed action.</td></tr></table>

## 4.5. Three properties of the general model

We now state three properties of the model in the basic case (2). They answer questions (Q1) to (Q3) of Section 3. Each has a short proof and a plain reading. Throughout, model a dominates model b if a is no worse than b on every objective and strictly better on at least one, and a model is eficient if no other feasible model dominates it.

Observation 2 (Pruning that holds for every task length). If $\rho _ { b } \geq \rho _ { a }$ and $r _ { b , k } \le r _ { a , k }$ for every resource $k ,$ with at least one of these inequalities strict, then b dominates a $f o r$ every task length $L \geq 1$ , and b meets the constraint (6) whenever a does.

Proof. By (2) and (4), $R _ { a , k } ( L ) \ : = \ : r _ { a , k } L / \rho _ { a } ^ { L }$ , which increases with $r _ { a , k }$ and decreases with $\rho _ { a }$ . Hence $R _ { b , k } ( L ) \leq R _ { a , k } ( L )$ for every k. The inequality is strict for every k if $\rho _ { b } > \rho _ { a }$ , and strict for resource k if $r _ { b , k } < r _ { a , k }$ . The constraint depends on the model only through $\rho ^ { L }$ , which is at least as large for b. □

In plain terms, a model that is worse than another on every per-step measure can be removed once, for all tasks. This answers (Q1), and it is the Stage 1 screen of the framework. The screen has two useful properties. First, it depends only on the order of the candidates on each per-step quantity, so no choice of units, scaling or weights can change it. Second, it uses no information about the task beyond the task family, so one screen serves every task in that family. The costly part of the work, measuring the candidates, is shared across all tasks and users.

Observation 3 (One crossover for each resource). Let $\rho _ { a } < \rho _ { b }$ and $r _ { a , k } < r _ { b , k }$ for a resource k. Then

$$
\ln \frac { R _ { a , k } ( L ) } { R _ { b , k } ( L ) } \ : = \ : \ln \frac { r _ { a , k } } { r _ { b , k } } \ : + \ : L \ln \frac { \rho _ { b } } { \rho _ { a } } ,\tag{7}
$$

and model a uses less of resource k per successful task than model b if and only if

$$
L < L _ { k } ^ { * } = \frac { \ln ( r _ { b , k } / r _ { a , k } ) } { \ln ( \rho _ { b } / \rho _ { a } ) } .\tag{8}
$$

Proof. Equation (7) follows directly from (4) and (2). Its right-hand side is negative exactly when $L < L _ { k } ^ { * }$ □

In plain terms, the less reliable model starts with an advantage on each resource, and this advantage shrinks as tasks get longer. Equation (7) is a straight line in $L ,$ and its slope $\ln ( \rho _ { b } / \rho _ { a } )$ is the same for energy, cost and time. Only the starting advantage difers between the three resources. Each resource therefore has its own crossover length $L _ { k } ^ { * } ,$ , and the resource on which the less reliable model has the largest starting advantage is the last to cross.

Observation 4 (Three regimes). Let $\rho _ { a } ~ < ~ \rho _ { b }$ and $r _ { a , k } < r _ { b , k }$ for all three resources, and let $L _ { \mathrm { m i n } } ~ { = }$ min<sub>k</sub> $L _ { k } ^ { * }$ and $L _ { \operatorname* { m a x } } = \operatorname* { m a x } _ { k } L _ { k } ^ { * }$ . Then

(i) for $L < L _ { \mathrm { m i n } }$ , model a dominates model $b ;$

(ii) for $L > L _ { \mathrm { m a x } } .$ , model b dominates model a;

(iii) for $L _ { \mathrm { m i n } } < L < L _ { \mathrm { m a x } }$ , neither model dominates the other.

The middle band is empty when the ratios ${ r } _ { b , k } / { r } _ { a , k }$ are equal for all three resources.

Proof. Parts (i) to (iii) follow from Observation 3 applied to each resource. If the three ratios are equal, the three crossover lengths in (8) are equal and $L _ { \mathrm { m i n } } = L _ { \mathrm { m a x } }$ . □

In plain terms, for short tasks the less reliable model is better on every resource, and for long tasks the more reliable model is. In both cases the choice needs no preference information. Only in the band between them does the choice depend on what the decision maker values, for example low carbon over speed. This answers (Q2) and (Q3). The width of the band depends on how diferently the three resources compare. If energy, cost and time all difer between two models by the same factor, for example because price simply follows energy use, the band disappears and preferences are never needed for that pair. With m candidate models, the eficient set can change only at crossover lengths, of which there are at most $3 m ( m - 1 ) / 2$ Between these lengths it stays the same, so it can be computed once for each interval of task length and stored.

## 4.6. Special cases and variants

We now turn to question (Q4). The general model has four main special cases. They follow from two questions about the task: is the outcome binary or graded, and can a failed attempt be repeated? Table 4 summarises them.

Table 4: Four special cases of the general model (5).
<table><tr><td></td><td>Failures can be retried  $( K = \infty )$ </td><td>Irreversible or limited retries (K fi- nite)</td></tr><tr><td>Binary outcome  $( q = 1 )$ </td><td>per successful task. No constraint. automatic check.</td><td>V1. Minimise energy, cost and time V2. Same objectives, subject to the chance constraint (6). Example: data extraction with an Example: changing an airline book- ing.</td></tr><tr><td rowspan="2">Graded outcome  $( \underline { { q } } < 1 )$ </td><td>V3. Success means reaching the qual- V4. Quality floor as a constraint and ity floor. Quality above the floor has expected quality</td><td> $\bar { q } _ { a }$  as a fourth object-</td></tr><tr><td>no extra value. Example: a summary that must con- tain a set of required facts.</td><td>ive. Example: drafting a report, where better quality has real value.</td></tr></table>

V1: binary outcome, free retries. This is the pure eficiency question. Observations 2 to 4 apply as stated.

V2: binary outcome, irreversible actions. By Observation 1, the objectives are the same as in V1, and only the feasible set changes. The constraint (6) can be strict. With $K = 1$ and $\beta = 0 . 9$ , a model with $\rho = 0 . 9 5$ meets it only for tasks of up to 2 steps, and a model with $\rho = 0 . 9 9$ only for tasks of up to 10 steps. Beyond these lengths neither model is feasible. This result is useful on its own, because it tells the decision maker that the task should be redesigned, for example by adding a human check before each irreversible step, rather than given to a diferent model.

V3: graded outcome with a quality floor. Success now means that the quality of the output reaches the floor, so $S _ { a } ( L )$ depends on q. Quality above the floor has no extra value. This is the satisficing rule of Simon (1955): the decision maker asks for the least resource-intensive model that is good enough.

V4: graded outcome, floor and maximisation. Here the decision maker also values quality above the floor, so expected quality becomes a fourth objective. The floor acts as a reservation level, below which no output is acceptable, and above it quality is traded against resources. This is the logic of reference-point methods (Wierzbicki, 1980). V3 and V4 are not separate frameworks. Following the ε-constraint idea (Haimes et al., 1971), any objective can be turned into a constraint, and the choices made under V3 and V4 both lie on the eficient set of the same general problem. The diference between the V3 and V4 recommendations measures the price of quality: the extra energy, cost and time that the decision maker pays for quality above the floor. Observation 3 still holds for the three resources. The quality objective usually keeps more models eficient, so the bands in which preferences matter become wider.

The efect of the quality floor. Raising the floor lowers the success probability of every model. If it lowers the success of the smaller model more than that of the larger model, which we expect and test in Section 5, the crossover lengths move to shorter tasks. The eficient choice then depends on two features of the task, its length and its quality floor. It can be shown as a map with three regions: the smaller model dominates, both are eficient, and the larger model dominates.

## 4.7. Relaxing the assumptions

The basic case makes three simplifying assumptions. We now relax each one and state what remains true.

Reliability that changes along the task. Per-step reliability need not be constant. Errors can become more likely as a task goes on, for example when earlier mistakes remain in the context (Sinha et al., 2026). In this case we replace $\rho _ { a } ^ { L }$ by the measured success curve $S _ { a } ( L )$ . Observation 1 does not depend on the form of $S _ { a } ( L )$ and holds unchanged. Observation 2 holds if $S _ { b } ( L ) \geq S _ { a } ( L )$ for every $L .$ The single crossover of Observation 3 holds if the ratio $S _ { a } ( L ) / S _ { b } ( L )$ falls as L grows. The lines in (7) are then no longer straight, but they still difer only in their starting values, so the three regimes of Observation 4 remain. The condition on $S _ { a } ( L ) / S _ { b } ( L )$ can be checked directly from benchmark data, and we test it in Section 5.

Failures detected early. If a failed attempt stops at the step that fails, an attempt uses on average (1 − $\rho _ { a } ^ { L } ) / ( 1 - \rho _ { a } )$ steps, and the resource per successful task becomes $\begin{array} { r } { R _ { a , k } ( L ) \ = \ r _ { a , k } \sum _ { j = 1 } ^ { L } \rho _ { a } ^ { - j } } \end{array}$ . This still increases with $r _ { a , k }$ and decreases with $\rho _ { a }$ , so Observation 2 holds. The factor that depends on reliability is still the same for all three resources, so there is still one crossover for each resource, in the same order. The crossover lengths are now found numerically, and they are longer than in the basic case, because failed attempts cost less.

From energy to carbon. Carbon emissions equal energy multiplied by the power usage efectiveness of the data centre and the carbon intensity of the local electricity grid. When two models run in the same data centre at the same time, these factors are the same for both and cancel in (8), so the carbon crossover equals the energy crossover. When the models run in diferent regions, or at diferent times of day, the factors enter the starting advantage and move the carbon crossover. This is where the location and timing of inference, studied by carbon-aware routing (Sheshanarayana et al., 2026) and by measurement at provider level (Elsworth et al., 2025), enter our framework.

Uncertainty in measured values. The values of $\rho _ { a }$ and $r _ { a , k }$ are estimates. Because the denominator ln ${ \left( \rho _ { b } / \rho _ { a } \right) }$ in (8) is small when two models have similar reliabili $\mathrm { { , y , } }$ small errors in reliability can move a crossover by many steps. We therefore report every crossover length with a confidence interval obtained by resampling the benchmark runs. A wide interval is itself useful: it shows the decision maker that, for tasks of that length, the two models cannot be told apart on the evidence available.

## 4.8. Stage 2: choosing within a trade-of

When the task-level screen leaves one model, the framework recommends it, and no preferences are needed. When it leaves more than one, the selection question has no answer without knowing what the decision maker values. Stage 2 provides this answer for the eficient set $E ( L )$ at the task length L.

The values of each objective are rescaled to the range from 0 to 1 within the eficient set,

$$
v _ { k } ( a ) \ = \ { \frac { \operatorname* { m a x } _ { b \in E ( L ) } R _ { b , k } ( L ) - R _ { a , k } ( L ) } { \operatorname* { m a x } _ { b \in E ( L ) } R _ { b , k } ( L ) - \operatorname* { m i n } _ { b \in E ( L ) } R _ { b , k } ( L ) } } , \qquad a \in E ( L ) ,\tag{9}
$$

so that the best model on resource k scores 1 and the worst scores 0. Weights $w _ { e } , w _ { c } , w _ { t }$ , which sum to one, express how much the deployment cares about carbon, cost and time. In the deliberative way, a team sets them once, using AHP pairwise comparisons with the usual consistency check (Saaty, 1977, 2008), or the flexible and interactive trade-of method when the team can give only partial information (de Almeida et al., 2016). In the automated way, they follow from the task profile through a mapping that the organisation can inspect and change. The default rule is SAW,

$$
S ( a ) = w _ { e } v _ { e } ( a ) + w _ { c } v _ { c } ( a ) + w _ { t } v _ { t } ( a ) , \qquad a \in E ( L ) ,\tag{10}
$$

and TOPSIS (Hwang and Yoon, 1981; Ciardiello and Genovese, 2023) is computed alongside it as a check. When the two methods disagree on the top choice, the framework reports the disagreement, because it signals that the choice is close.

Two features keep Stage 2 small and safe. First, it is needed only inside the bands of Observation 4. Outside them, the answer does not depend on the weights, so disputes about weights cannot change it. Second, the chance constraint (6) is applied before Stage 2. A decision maker who puts all the weight on cost can therefore never push an unreliable model to the top for a task whose actions cannot be undone. This plays the role of the risk veto of Nowak (2026), extended from a choice between two models to a choice among many, and derived from the required chance of completing the task rather than set by hand.

## 4.9. An illustrative example

We now bring the parts of the framework together in an example. Table 5 lists six candidate models with illustrative per-step values. The values are not measurements of named commercial models.

Table 5: Illustrative candidate models. Per-step reliability $\rho ;$ energy in Wh per step; cost in US cents per step; time in seconds per step; context in thousands of tokens.
<table><tr><td>Model</td><td> $\rho$ </td><td>Energy</td><td>Cost</td><td>Time</td><td>Context</td></tr><tr><td>A</td><td>0.995</td><td>0.400</td><td>3.00</td><td>2.7</td><td>400</td></tr><tr><td>B</td><td>0.990</td><td>0.220</td><td>1.20</td><td>2.1</td><td>200</td></tr><tr><td>C</td><td>0.970</td><td>0.060</td><td>0.25</td><td>1.5</td><td>128</td></tr><tr><td>D</td><td>0.960</td><td>0.075</td><td>0.30</td><td>1.8</td><td>128</td></tr><tr><td>E</td><td>0.985</td><td>0.320</td><td>1.80</td><td>3.6</td><td>1000</td></tr><tr><td>F</td><td>0.930</td><td>0.030</td><td>0.10</td><td>0.9</td><td>32</td></tr></table>

Stage 1. Model D is worse than model C on every per-step quantity, and model E is worse than model B. By Observation 2, neither D nor E can be eficient for any task length, so both are removed once. Model E would come back only for a task that needs more than 200 thousand tokens of context. Stage 0 would then remove models B, C, D and $\mathrm { F } ,$ and model A does not dominate $\mathrm { E } ,$ because E uses less energy and cost per step.

Eficient sets by task length. For tasks that can be retried freely (V1), Table 6 shows the eficient set as task length grows. It changes at the crossover lengths of Observation 3. For example, model F starts with half the energy of model C per step, but C is more reliable, and the energy crossover between them is at about 17 steps.

Table 6: Eficient models per successful task, by task length, for the candidates in Table 5 (variant V1).
<table><tr><td>Task length L</td><td>Efficient models</td></tr><tr><td>1 to 12</td><td>F</td></tr><tr><td>13 to 16</td><td>C, F</td></tr><tr><td>17 to 21</td><td>B, C, F</td></tr><tr><td>22 to 49</td><td>B, C</td></tr><tr><td>50 to 76</td><td>A, B, C</td></tr><tr><td>77 to 181</td><td>A, B</td></tr><tr><td>182 or more</td><td>A</td></tr></table>

Three task profiles. Consider three tasks given to the same set of models.

• A short triage task with 3 steps, which can be retried freely. Only model F is eficient, so the framework recommends F without asking for weights. Per successful task, F uses about 0.11 Wh against 1.22 Wh for model A, a saving of about 90%.

• A customer-service task with 40 steps, which can be retried freely. Models B and C are both eficient. Per successful task, C uses about 8.1 Wh, 34 cents and 203 seconds, and B uses about 13.2 Wh, 72 cents and 126 seconds. A deployment that cares mostly about carbon or cost chooses C, and one that cares mostly about speed chooses B. This is the only one of the three tasks for which Stage 2 is needed. Note that the smallest model, F, is not eficient here: per successful task it uses about 21.9 Wh, almost three times as much as C, although it uses half as much energy per step.

• The same customer-service task, with an irreversible payment step, allowing at most K = 2 attempts with required completion probability β = 0.95. By (6), each attempt must succeed with probability at least 0.78. At 40 steps only model A meets this, so the framework recommends A. Per successful task, A uses about 19.6 Wh, more than twice as much as C. This is the price of not being able to undo an action.

The same six models thus give three diferent recommendations, and the reason for each one can be read directly from the task profile.

Figure 2 shows why models B and C form a trade-of at 40 steps. On a log scale, the ratio of their resources per successful task is a straight line in task length for each resource, and the three lines are parallel. C is better on time up to about 17 steps, on energy up to about 64 and on cost up to about 77. Between the first and the last crossing, which includes 40 steps, each model is better on some resource.

## 4.10. Right-sizing

The example shows what right-sizing means once the selection question is asked per successful task. Right-sizing is not choosing the smallest model. It is choosing, among the models that meet the required chance of completing the task, a model that is eficient in energy, cost and time per successful task, and, within a trade-of band, the one that best matches the stated preferences. Under this definition right-sizing can point to a larger model, as it did for the irreversible task above.

Right-sizing has an environmental benefit that the framework can measure. For each decision, the saving is the diference in energy, and hence in carbon, per successful task between the recommended model and a reference policy, such as always using the most capable model or always using the smallest. Because agentic workloads contain many short, repeated subtasks, the number of decisions where a smaller model is eficient can be large. The framework also shows the opposite case: tasks where a policy of always using small models costs more energy than it saves. Any figure for total savings across a whole service rests on assumptions about the mix of tasks and the carbon intensity of the grid. We therefore report such figures as scenarios with stated bounds, not as measurements.

![](images/e5c24be38006897c399a68bcbe919f65b2ece146aa858689b936482aa3af24eb.jpg)  
Figure 2: Resources per successful task of model C relative to model B, on a log scale, against task length. Below zero, C uses less. Each line crosses zero at the crossover length of its resource. In the shaded band neither model is better on every resource.

## 5. How can we test the framework?

The framework gives a specific answer to the selection question: it predicts which models are eficient for a task of a given length, and when a smaller model stops being the greener, cheaper or faster choice. This section asks how these predictions can be tested. We first state the claims, then the measurements and the design that test them.

## 5.1. Claims to be tested

The framework rests on four claims, which we state as hypotheses that an unfavourable result would reject.

H1 (structure). On agentic benchmarks, the success of a task falls with its length in a way that meets the single-crossing condition of Section 4.7: for pairs of models, the ratio of their success curves falls as tasks get longer.

H2 (prediction). Crossover lengths predicted from per-step measurements on one set of tasks agree with the crossovers observed on a separate set of tasks, within their confidence intervals.

H3 (decision value). Choices made by the framework have lower regret than choices made by per-call rules. Regret is the extra energy, cost and time per successful task, compared with the best model in hindsight. The per-call rules are: always use the most capable model, always use the smallest model, use the top model of a public leaderboard, a cascade that escalates to a stronger model on failure (Chen et al., 2024), and the single-stage weighted rule of Nowak (2026) applied to all models.

H4 (savings). On a substantial share of tasks, the framework selects a smaller model than the most capable one, with no loss in the required chance of completion. On another share of tasks, it selects a larger model than the smallest one and so avoids the energy cost of failed attempts.

H1 is the foundation of the model, and H3 is the result on which the practical value of the framework depends. If H1 fails, the crossover results do not hold as stated. If H3 fails, per-call rules are good enough, and the extra measurement efort is not worth it.

## 5.2. What is measured and how

The design follows one principle: every value is measured on a common test harness, not taken from figures reported by providers, so that the comparison is fair and can be repeated.

Table 7: Measurements used in the evaluation.
<table><tr><td>Quantity</td><td>How it is measured</td></tr><tr><td>Task success</td><td>Automatic check of the final state of each task against the correct state, as in τ-bench (Yao et al., 2024; Barres et al., 2025).</td></tr><tr><td>Per-step reliability</td><td>Estimated from complete runs, by the step at which each run first fails.</td></tr><tr><td>Success curve  $S _ { a } ( L )$ </td><td>Share of successful runs, grouped by the number of steps a task needs. For open-weight models, measured per inference on known hardware, using pub-</td></tr><tr><td>Energy</td><td>lished measurements (Chung et al., 2025) and checked against an estimate based on measured speed. For closed models, estimated with stated bounds.</td></tr><tr><td>Cost</td><td>Tokens used, multiplied by the provider&#x27;s published price.</td></tr><tr><td>Time</td><td>Measured response time for each step, summed over the task.</td></tr></table>

Each pair of model and task is run several times, so that every measured value carries a confidence interval. Confidence intervals for crossover lengths and for regret are obtained by resampling runs. Tasks are split into two sets: one to estimate per-step values, and one, held out, to test the predictions (H2 and H3). This prevents the framework from being tested on the same data used to set it up.

## 5.3. Benchmarks and models

The benchmarks must vary in task length, so that crossovers can be observed, and must check success automatically. The retail and airline domains of τ-bench meet both needs: tasks difer in the number of tool calls they require, and the airline domain contains actions that cannot be undone, which makes it suitable for variant V2. Function-calling tasks from the Berkeley leaderboard (Patil et al., 2025) add a second task family. For the graded variants V3 and ${ \mathrm { V 4 } } ,$ a task family with a scored outcome is needed; we use a smaller set of such tasks for illustration.

The main study uses open-weight models of diferent sizes, served by one provider through one harness. This has three advantages: their energy use can be measured by third parties, their behaviour is not changed without notice, and all models are run under the same conditions. Closed models are added later as a check of how far the results carry over, using public measurements.

## 5.4. Robustness

We check the results in three ways. First, we vary the weights in Stage 2 and report how often the recommendation changes. Second, we vary the measured values within their confidence intervals and report how often the eficient set changes. Third, we test the framework over time: we fix the measurements at one date, and ask whether the framework ranks models released later correctly from their measured per-step values alone.

## 6. Experiments and results

This section reports a first experiment that tests the framework’s answer to the selection question. Before the full study of Section 5, we ask a narrower question: with real models on real multi-step tasks, does success fall with task length as the model assumes (H1), and do the crossovers fall where per-step measurements predict (H2)? To answer it cleanly, we use tasks whose length we can set exactly, and models whose energy we can measure rather than estimate. We use two kinds of task and models from several developers, so that the results do not rest on one task or one family of models. H3 needs the full study; the pilot gives a first view of H4.

## 6.1. Pilot study design

Tasks. Both tasks are conversations with an agent in which every step has one correct answer that a program can check. The full conversation is kept, so the agent sees its own earlier replies and an error carries forward, as in a real agent. A task of length L succeeds if the first L steps are all correct. Runs are never stopped early, so one run of 150 steps gives the outcome of a task of every length from 1 to 150, together with the energy and time of every step. Every model sees the same 15 tasks of each kind, generated from fixed seeds, and the temperature is set to zero.

• Order task. The agent keeps the running total of a customer order. At each step it receives a tool result describing one order line, to be added or removed, and must reply with the new total. The arithmetic is deliberately easy, because the tool result states the amount of each line: errors come from keeping track over many steps, not from hard sums.

• Tools task. The agent serves a customer whose account holds five orders, each with a few items and a delivery city. At each step the customer asks for one change, and the agent must reply with the one correct tool call: change the delivery city, add an item, remove an item, or refund an item. About half of the requests refer back to earlier ones, such as “add two mugs to that same order”, “send order F4 to the same city as order B3” or “refund every vase in order H9”, which needs the number of vases the order holds now. The tools reply only “ok”, so the agent must keep track of the account itself. A step is correct if the tool and every argument are right.

The two tasks fail in diferent ways. In the order task an error is a wrong number. In the tools task it is a wrong action or a wrong detail, most often a refund of the wrong quantity or a change that the agent has lost track of.

Models and measurement. We use 15 open-weight models from five developers, from 0.6 to 27 billion parameters (Table 8): Qwen (Alibaba), Gemma (Google), Llama (Meta), Ministral (Mistral AI) and GLM (Zhipu AI). They are served through Ollama on a single NVIDIA RTX 5060 Ti graphics card (GPU) with 16 GB of memory. Each model runs alone on the card and fully in its memory, which the test harness checks before every model. Energy is the power of the GPU board, read 20 times a second through the NVIDIA management library and summed over each call. It leaves out the processor, memory and cooling, so it is a lower bound on the energy used. Time is the time per call reported by the server. We do not report money cost, because on one’s own hardware it follows from energy and time.

Estimation. Per-step reliability ρ is estimated from the step at which each run first fails, by the maximum likelihood estimate under a constant per-step rate: the number of steps passed divided by the number of steps passed plus the number of failures. The success curve S(L) is the share of runs whose first failure comes after step L. Confidence intervals come from 1000 resamples of runs within each model.

Reproducibility. The raw logs of every call, the harness and the analysis are in the public repository of the framework. One command recreates every number, table and figure in this section from the committed logs, without a GPU or network access.

## 6.2. Results

Table 8: Pilot results per model and task: estimated per-step reliability, mean GPU energy and time per step, and GPU energy per successfully completed task at three task lengths. ∞: no run completed a task of that length. <sup>†</sup>: removed by the per-step screen, because another model is at least as reliable and uses no more energy or time per step.  
(a) Order task
<table><tr><td>Model</td><td> $\hat { \rho }$ </td><td>95% CI</td><td>J/step</td><td>s/step</td><td colspan="3">kJ per successful task  $L = 1 0$   $L = 2 0$ </td></tr><tr><td>Llama 3.2 1B</td><td>0.4643</td><td>0.2500-0.6053</td><td>7</td><td>0.07</td><td>∞</td><td>∞</td><td>∞</td></tr><tr><td>Gemma 3 1B†</td><td>0.0000</td><td>0.0000-0.0000</td><td>8</td><td>0.15</td><td>8</td><td>8</td><td>∞</td></tr><tr><td>Qwen3 0.6B</td><td>0.9176</td><td>0.8649-0.9436</td><td>29</td><td>0.23</td><td>0.3</td><td>1.7</td><td>∞</td></tr><tr><td>Qwen3 1.7B</td><td>0.9800</td><td>0.9720-0.9846</td><td>30</td><td>0.22</td><td>0.1</td><td>0.5</td><td>2.4</td></tr><tr><td>Llama 3.2 3B</td><td>0.9878</td><td>0.9844-0.9915</td><td>30</td><td>0.21</td><td>0.1</td><td>0.4</td><td>1.8</td></tr><tr><td>Ministral 3 3B</td><td>0.9880</td><td>0.9828-0.9916</td><td>51</td><td>0.45</td><td>0.3</td><td>0.7</td><td>3.1</td></tr><tr><td>Gemma 3 4B</td><td>0.9882</td><td>0.9800-0.9927</td><td>54</td><td>0.53</td><td>0.5</td><td>1.1</td><td>3.5</td></tr><tr><td>Llama 3.1 8B</td><td>0.9905</td><td>0.9884-0.9927</td><td>59</td><td>0.37</td><td>0.4</td><td>0.9</td><td>2.5</td></tr><tr><td>GLM-4 9B†</td><td>0.9444</td><td>0.9250-0.9564</td><td>70</td><td>0.49</td><td>0.6</td><td>3.1</td><td>8</td></tr><tr><td>Qwen3 8B</td><td>0.9980</td><td>0.9956-0.9995</td><td>91</td><td>0.56</td><td>0.6</td><td>1.3</td><td>4.0</td></tr><tr><td>Ministral 3 8B†</td><td>0.9969</td><td>0.9942-0.9986</td><td>124</td><td>0.95</td><td>0.8</td><td>2.1</td><td>6.4</td></tr><tr><td>Ministral 3 14B</td><td>0.9985</td><td>0.9968-1.0000</td><td>136</td><td>0.94</td><td>1.0</td><td>2.1</td><td>5.8</td></tr><tr><td>Qwen3 14B†</td><td>0.9980</td><td>0.9958-0.9995</td><td>153</td><td>0.90</td><td>1.1</td><td>2.4</td><td>7.0</td></tr><tr><td>Gemma 3 12B</td><td>0.9991</td><td>0.9974-1.0000</td><td>154</td><td>1.39</td><td>1.2</td><td>2.4</td><td>9.0</td></tr><tr><td>Qwen3.8 27B</td><td>0.9995</td><td>0.9985-1.0000</td><td>236</td><td>1.61</td><td>2.1</td><td>4.2</td><td>11.0</td></tr></table>

(b) Tools task
<table><tr><td>Model</td><td> $\hat { \rho }$ </td><td>95% CI</td><td>J/step</td><td> $\mathrm { s / s t e p }$ </td><td colspan="4">kJ per successful task  $L = 1 0$   $L = 2 0$   $L = 5 0$ </td></tr><tr><td>Gemma 3 1B</td><td>0.7727</td><td>0.5946–0.8438</td><td>16</td><td>0.24</td><td>1.3</td><td></td><td>∞</td><td>∞</td></tr><tr><td>Llama 3.2 1B</td><td>0.8404</td><td>0.7581-0.8913</td><td>19</td><td>0.17</td><td>0.7</td><td></td><td>8</td><td>8</td></tr><tr><td>Qwen3 0.6B†</td><td>0.8352</td><td>0.6875-0.8958</td><td>23</td><td>0.19</td><td></td><td>0.9</td><td>6.4</td><td>∞</td></tr><tr><td>Qwen3 1.7B</td><td>0.9153</td><td>0.8739-0.9378</td><td>23</td><td>0.18</td><td></td><td>0.2</td><td>1.5</td><td>8</td></tr><tr><td>Llama 3.2 3B</td><td>0.9296</td><td>0.8936-0.9492</td><td>34</td><td>0.23</td><td></td><td>0.3</td><td>3.7</td><td>∞</td></tr><tr><td>Ministral 3 3B</td><td>0.9513</td><td>0.9339-0.9630</td><td>50</td><td>0.45</td><td></td><td>0.3</td><td>1.8</td><td>∞</td></tr><tr><td>Gemma 3 4B†</td><td>0.9312</td><td>0.9080-0.9464</td><td>51</td><td>0.50</td><td></td><td>0.7</td><td>7.0</td><td>∞</td></tr><tr><td>Llama 3.1 8B</td><td>0.9533</td><td>0.9306-0.9648</td><td>70</td><td>0.45</td><td></td><td>0.6</td><td>2.3</td><td>∞</td></tr><tr><td>Qwen3 8B†</td><td>0.9476</td><td>0.9198-0.9620</td><td>73</td><td>0.47</td><td></td><td>0.7</td><td>3.0</td><td>48.4</td></tr><tr><td>Ministral 3 8B</td><td>0.9704</td><td>0.9565-0.9780</td><td>75</td><td>0.60</td><td></td><td>0.6</td><td>1.8</td><td>12.2</td></tr><tr><td>GLM-4 9B†</td><td>0.9470</td><td>0.9272–0.9593</td><td>84</td><td>0.61</td><td></td><td>0.8</td><td>2.8</td><td>8</td></tr><tr><td>Gemma 3 12B</td><td>0.9710</td><td>0.9576-0.9797</td><td>118</td><td>1.13</td><td></td><td>1.1</td><td>3.7</td><td>42.1</td></tr><tr><td>Qwen3 14B†</td><td>0.9689</td><td>0.9603-0.9751</td><td>128</td><td>0.81</td><td></td><td>1.8</td><td>3.6</td><td>31.7</td></tr><tr><td>Ministral 3 14B</td><td>0.9779</td><td>0.9642–0.9846</td><td>129</td><td>0.91</td><td></td><td>1.0</td><td>3.0</td><td>21.8</td></tr><tr><td>Qwen3.8 27B</td><td>0.9878</td><td>0.9833-0.9913</td><td>199</td><td>1.47</td><td></td><td>2.0</td><td>4.2</td><td>12.2</td></tr></table>

Per-step values and the screen. Reliability rises broadly with size, but far from strictly, and so does energy per step (Table 8, Figure 3). The tools task is the harder one: the most reliable model makes an error about once in 80 steps there, against once in 2000 steps in the order task. The per-step screen removes four models in the order task and five in the tools task before task length is considered. Some of these removals are not visible from model size or developer. In the order task, Qwen3 14B and Ministral 3 8B are removed by Qwen3 8B, which is at least as reliable and uses less energy and time per step. In the tools task, Qwen3 8B is itself removed, by Llama 3.1 8B and by Ministral 3 3B, a model of less than half its size. Gemma 3 1B fails at the first step of every order task, because it gives wrong totals from the start, and is removed by Llama 3.2 1B. Smaller models do not always use less energy per step either: Gemma 3 1B uses more than Llama 3.2 1B in the order task, and Qwen3 0.6B uses about as much as Qwen3 1.7B in both tasks. So the smallest model is not always the greener choice even per call, and which models are worth considering depends on the task.

Does success fall with length as assumed? (H1). Success falls with task length for every model and both tasks (Figure 4). A constant per-step rate is only an approximation. For the most reliable models in the order task the success curve stays within 0.1 of the curve that a constant rate implies, but for several mid-sized models it departs by 0.3 to 0.45, because their errors come later in the task than a constant rate predicts. The condition our results need is weaker: that the curves of two models cross at most once for each resource (Section 4.7). We compare energy and time per successful task for every pair of screened models in which the less reliable model uses less per step: 188 comparisons over the two tasks. In 11, all in the order task, the curves do not cross within 150 steps. In 129 they cross once, and in 48 more than once. If we count only lengths at which both curves rest on at least three successful runs, 33 comparisons still cross more than once. Nearly all of these are of two kinds. Either the two curves stay within about 30% of each other over the lengths where they cross back and forth, so that the two models are close to a tie and 15 runs cannot separate them, or an extra crossing falls in the first few steps, because the smaller model sometimes fails at the very first step. The single-crossing condition therefore holds well where the choice matters, and fails mainly where it does not.

Do the crossovers fall where predicted? (H2). Table 9 compares three crossover lengths for energy, for the models shown in Figure 4. The closed form is the formula of Section 4.4, with the mean energy per step. The fitted value applies the same rule to the measured energy step by step, which allows energy per step to grow as the conversation gets longer; in our runs it grows by up to about four times over 150 steps for the smallest models, and by a quarter or less for the largest. The observed value is read from the measured curves.

![](images/d5b355feea76e02a84669a3cc2dfbc513ddb10b6b27bbdee4df3c1db4a4f680a.jpg)

(a) Order task  
![](images/03fb399eb816a29e7b0566f04c5434d1cb679f7d939bd8c0cf40d3b5f5508447.jpg)  
(b) Tools task  
Figure 3: The 15 models on two per-step measures: GPU energy per step, and the expected number of steps before the first error, $1 / ( 1 - \hat { \rho } )$ , both on log scales. Colours and markers show the developer. Hollow markers are models removed by the per-step screen. Better models lie towards the top left.

![](images/3852f0ee54ac57a2e1cd93dcd9d9888b3f227178ac07ac53894bf84d9d3c7343.jpg)

(a) Order task  
![](images/5829aab1f759b9757928391edb3cc19d4900ac036ef2de6efc9b45b0b96c35b7.jpg)  
(b) Tools task  
Figure 4: GPU energy per successfully completed task against task length, on a log scale, for the models that use the least energy per completed task at some length, and for the most reliable model. Lines use the estimated per-step reliability and the measured energy of each step; points are the observed values, every five steps, in the colour and marker of their model. Hollow points rest on fewer than three successful runs out of 15, so they are uncertain.

Over all pairs of screened models, the formula predicts 53 energy crossovers in the order task and 45 in the tools task. In the order task, 8 are predicted beyond 150 steps and none of these is observed. Of the 88 crossovers observed over both tasks, the closed form lies inside the 95% interval of the observed crossover for 72, and the median diference between the two is 9 steps in the order task and 4 in the tools task. The 16 misses have plain causes. Eight involve a model whose observed curve runs out of successful runs before the predicted crossover: Llama 3.2 3B in the order task, with only one of its 15 runs still free of errors after 111 steps, and Qwen3 1.7B in the tools task, with none after 24 steps. The observed crossover then falls where those runs end, a few steps before the prediction. Four involve two models with almost the same reliability, where the formula is very sensitive to small errors in ${ \hat { \rho } } .$ Two involve Llama 3.1 8B, whose errors come late in the order task, and two are diferences of one to three steps for tasks of two to five steps. The full tables for all pairs are in the repository. These predictions are made on the same runs as the observations, so they show that the model describes the data, not yet that it predicts new data. The held-out test of H2 is part of the full study.

Which model is greenest depends on the length and the kind of task (H4). In both tasks the model that uses the least energy per completed task changes as tasks get longer (Figure 4). In the order task, Llama 3.2 1B is greenest for tasks of one or two steps, Llama 3.2 3B from 3 steps to about 100, and Qwen3 8B beyond. Between about 80 and 100 steps the lead passes back and forth between Llama 3.2 3B, Llama 3.1 8B and Ministral 3 3B, whose curves are close there. Taking energy and time together, the three regimes of Section 4.5 appear between Llama 3.2 3B and Qwen3 8B: the time crossover (96 steps by the formula, 83 observed) comes before the energy crossover (107 and 94), and between them Qwen3 8B is faster while Llama 3.2 3B still uses less energy. In the tools task the greenest model moves up in size much sooner: Llama 3.2 1B up to 4 steps, Qwen3 1.7B up to about 20, Ministral 3 3B up to about 30, Ministral 3 8B up to about 50 to 60, and Qwen3.8 27B beyond. Only six of the 15 models completed any tools task of 50 steps. The stakes are large in both directions. At 20 steps in the order task, Llama 3.2 3B used 0.39 kJ per completed task, about a tenth of the 4.2 kJ of the largest model. At 100 steps, Qwen3 1.7B used 39 kJ, almost four times the 10.4 kJ of Qwen3 8B. In the order task the largest model is never eficient within 150 steps when tasks can be retried freely, while in the tools task it becomes the greenest choice at about 50 to 60 steps. The same models give diferent answers for diferent kinds of task, which is why per-step reliability has to be measured for the task family in question, and not read from a general leaderboard.

Table 9: Energy crossover lengths for the models shown in Figure 4: the task length beyond which the more reliable model uses less GPU energy per successfully completed task. Closed form: Section 4.4 with the mean energy per step. Fitted: measured energy per step and the estimated reliability. Observed: from the measured success curves, with a bootstrap interval. A dash means no crossover within 150 steps. Where some resamples show no crossover, their share is given in brackets.  
(a) Order task
<table><tr><td>Less reliable model</td><td>More reliable model</td><td>Energy ratio</td><td colspan="4">Energy crossover L*</td></tr><tr><td></td><td></td><td>per step</td><td>closed form</td><td>fitted</td><td>observed</td><td>observed 95% CI</td></tr><tr><td>Llama 3.2 1B</td><td>Llama 3.2 3B</td><td>0.22</td><td>2</td><td>2</td><td>3</td><td>1-3</td></tr><tr><td>Llama 3.2 1B</td><td>Qwen3 8B</td><td>0.07</td><td>3</td><td>4</td><td>5</td><td>2-5</td></tr><tr><td>Llama 3.2 1B</td><td>Qwen3.8 27B</td><td>0.03</td><td>5</td><td>6</td><td>5</td><td>3-5</td></tr><tr><td>Llama 3.2 3B</td><td>Qwen3 8B</td><td>0.34</td><td>107</td><td>108</td><td>94</td><td>72-112</td></tr><tr><td>Llama 3.2 3B</td><td>Qwen3.8 27B</td><td>0.13</td><td>174</td><td></td><td>112</td><td>83–112 (28% none)</td></tr><tr><td>Qwen3 8B</td><td>Qwen3.8 27B</td><td>0.39</td><td>612</td><td></td><td></td><td>148–148 (100% none)</td></tr></table>

(b) Tools task
<table><tr><td>Less reliable model</td><td>More reliable model</td><td>Energy ratio</td><td colspan="4">Energy crossover L* (steps)</td></tr><tr><td></td><td></td><td>per step</td><td>closed form</td><td>fitted</td><td>observed</td><td>observed 95% CI</td></tr><tr><td>Gemma 3 1B</td><td>Llama 3.2 1B</td><td>0.82</td><td>2</td><td>2</td><td>1</td><td>1–15 (18% none)</td></tr><tr><td>Gemma 3 1B</td><td>Qwen3 1.7B</td><td>0.68</td><td>2</td><td>4</td><td>3</td><td>2-15</td></tr><tr><td>Gemma 3 1B</td><td>Ministral 3 3B</td><td>0.32</td><td>6</td><td>5</td><td>4</td><td>2-10</td></tr><tr><td>Gemma 3 1B</td><td>Ministral 3 8B</td><td>0.21</td><td>7</td><td>9</td><td>7</td><td>4-15</td></tr><tr><td>Gemma 3 1B</td><td>Qwen3.8 27B</td><td>0.08</td><td>10</td><td>13</td><td>15</td><td>6-15</td></tr><tr><td>Llama 3.2 1B</td><td>Qwen3 1.7B</td><td>0.84</td><td>2</td><td>5</td><td>5</td><td>3-20</td></tr><tr><td>Llama 3.2 1B</td><td>Ministral 3 3B</td><td>0.39</td><td>8</td><td>7</td><td>5</td><td>3-20</td></tr><tr><td>Llama 3.2 1B</td><td>Ministral 3 8B</td><td>0.26</td><td>9</td><td>12</td><td>10</td><td>6-20</td></tr><tr><td>Llama 3.2 1B</td><td>Qwen3.8 27B</td><td>0.10</td><td>14</td><td>17</td><td>19</td><td>7-20</td></tr><tr><td>Qwen3 1.7B</td><td>Ministral 3 3B</td><td>0.46</td><td>20</td><td>19</td><td>24</td><td>4-25</td></tr><tr><td>Qwen3 1.7B</td><td>Ministral 3 8B</td><td>0.31</td><td>20</td><td>24</td><td>24</td><td>12-25</td></tr><tr><td>Qwen3 1.7B</td><td>Qwen3.8 27B</td><td>0.12</td><td>28</td><td>32</td><td>24</td><td>17-25</td></tr><tr><td>Ministral 3 3B</td><td>Ministral 38B</td><td>0.66</td><td>21</td><td>32</td><td>31</td><td>15-44</td></tr><tr><td>Ministral 3 3B</td><td>Qwen3.8 27B</td><td>0.25</td><td>37</td><td>45</td><td>34</td><td>18-44</td></tr><tr><td>Ministral 3 8B</td><td>Qwen3.8 27B</td><td>0.38</td><td>55</td><td>61</td><td>49</td><td>23-88</td></tr></table>

Tasks that cannot be undone (variant V2). The picture changes again when a task must succeed at its first attempt. With a single attempt and a required chance of completion of 0.9, the estimated reliabilities allow order tasks of up to about 9 steps for Llama 3.2 3B, 52 for Qwen3 8B, 72 for Ministral 3 14B, 112 for Gemma 3 12B and 230 for the largest model. From about 112 to 230 steps the largest model is then the only feasible choice, although it is never eficient when retries are free. In the tools task no model meets this requirement beyond about 9 steps. For such tasks the framework’s answer is not a model at all: the task must be changed, for example by asking a person to confirm each irreversible change (Section 7). The task profile, not only the models, decides the answer.

What the pilot does not show. The pilot uses two synthetic task families, open-weight models of up to 27 billion parameters, one GPU and 15 runs per model and task. Its energy covers the GPU board only, and its predictions are tested on the same runs that produced them. It does not test H3. The full study of Section 5 addresses each of these limits, using τ -bench tasks, held-out tasks, larger and closed models, and the per-call rules as baselines.

## 7. Discussion

We return to the selection question: which model should an agent use? Our answer is that it depends on the task in a way that can be predicted. For short tasks, and tasks that can be retried freely, a smaller model is often the eficient choice, and right-sizing saves energy, cost and time. For long tasks, and especially for tasks whose actions cannot be undone, a larger and more reliable model can use less of every resource per completed task. Between these regimes lies a band where the choice depends on what the decision maker values. This section discusses what this answer means for practice, its limits, and the main objections to it.

## 7.1. Is smaller greener?

The view that small models are the sustainable choice for agents (Belcak et al., 2025) is right for many tasks, and our framework supports it where tasks are short. But it is a claim about energy per call. Per completed task, it holds only up to a crossover length that depends on the gap in reliability between models. A policy of always using the smallest model can therefore increase energy use. This matches the finding of Panigrahy and Tyagi (2026) that failed attempts and retries can dominate the energy of agentic workflows. Our pilot shows both sides: in the order task, on tasks of 20 steps a 3B model used about a tenth of the energy of a 27B model per completed task, while on tasks of 100 steps a 1.7B model used almost four times as much as an 8B model. In the harder tools task, the 27B model was already the greenest choice from about 50 to 60 steps (Section 6). For sustainability research, the lesson is that the functional unit is not a detail of reporting. It can reverse the ranking of two systems.

## 7.2. What this means for practice

The framework suggests three things that the designers of agents can do beyond choosing a model. First, shorter tasks favour smaller models, so breaking a long task into short parts that can be checked separately can save resources. Second, for tasks with irreversible actions, the chance constraint can rule out every model (variant V2), as it does in our tools task for tasks of more than about nine steps. The right response is then to change the task, for example by asking a person to confirm before an irreversible step, rather than to look for a diferent model. Third, a model can be changed during a task: an agent could start with a small model and switch to a larger one when the remaining part of the task passes a crossover length. We return to this in Section 8.

## 7.3. Limits of the model

The basic case assumes that steps succeed or fail independently and that any failed step spoils the attempt. Real agents sometimes recover from errors, and errors in one step can make errors in later steps more likely. Section 4.7 shows that the main results survive when the success curve is measured directly rather than assumed, provided the single-crossing condition holds, and this condition is tested as H1. The model also treats all attempts at a task as alike. In practice a second attempt may use what was learned in the first, or be handed to a diferent model, which is the idea behind cascades. Finally, the model takes the number of steps as known. In practice it is an estimate, and its uncertainty adds to the uncertainty of the crossover lengths.

## 7.4. Measuring energy and carbon

Energy is the hardest quantity to measure. For open-weight models it can be measured on known hardware; for closed models it must be estimated, and the estimates can be wide (Jegham et al., 2025). We keep measured and estimated values apart and carry the ranges of estimates through to the result. Our measures also cover only the energy used to run the model. They leave out the energy used to make the hardware and to train the model, as well as water use. Finally, cheaper and greener inference may lead to more use overall. This rebound efect means that savings per task do not translate directly into lower total energy use, and we do not claim that they do.

## 7.5. A changing market

Models, prices and hardware change every month, so any eficient set is a snapshot. A model removed by the per-step screen can become eficient again if its price falls, if the model that dominated it is withdrawn, or if a new objective is added. A new model, however, can never bring back a model that was dominated before, since it does not change the values of existing models. For this reason the framework keeps every candidate and its dated values, and recomputes the eficient sets when prices, models or measurements change. This is why we see the framework as a continuously maintained decision-support service rather than a one-of study. In the service, a measurement layer keeps the dated per-step values of each candidate, a core engine computes the eficient sets for each task family and each interval of task length, and an interface takes a task profile and returns the recommendation together with the eficient set, the weights used and the source of every value. Agent frameworks can query the service at run time, and people can use it through a web page built on the same interface.

## 7.6. Relation to routing

Routing methods (Ong et al., 2025; Fu et al., 2026; Li et al., 2026) learn to send each request or step to a suitable model, and often do so well. Our framework does not replace them. It answers a diferent part of the selection question: which models are worth considering for a task, when the choice needs preferences at all, and how energy enters the choice. Its eficient sets can limit the models a router chooses from, and its per-task measures can be used to evaluate routers in the same terms of energy, cost and time.

## 8. Conclusion and future directions

This paper asked which language model an AI agent should use, so that its tasks are completed reliably with the least energy, cost and time. We argued that this selection question must be answered per successfully completed task, not per call. On this basis we built a general decision model and showed that it has a simple structure. Models that are worse on every per-step measure can be removed for all tasks. Between any two remaining models, each resource has one crossover task length, and the crossovers divide tasks into three regimes, only one of which needs the preferences of the decision maker. The same model covers binary and graded outcomes, and tasks whose actions can or cannot be undone. The resulting two-stage framework separates questions of fact, which models are eficient, from questions of value, which trade-of to accept. It also gives right-sizing a precise meaning, which sometimes points to a smaller model and sometimes to a larger one.

Several directions follow. First, the full study set out in Section 5 will test whether the structure seen in the pilot of Section 6 holds on agentic benchmarks, with held-out tasks and per-call rules as baselines. Second, the model can be extended to switching models during a task, which turns the single choice into a policy over the course of the task. Third, agents that work together in teams raise the question of how crossovers combine across agents. Fourth, a study with practitioners can test whether the trade-ofs that the framework shows match the choices that the designers of agents actually make, and how their preferences can best be elicited. Fifth, the measure of resources can be widened to include the energy of hardware and training, and water use. Finally, the framework will be released as an open service with its measurement harness and data, so that its eficient sets can be updated as the market changes.

## References

Aggarwal, P., Madaan, A., Anand, A., Potharaju, S.P., Mishra, S., Zhou, P., Gupta, A., Rajagopal, D., Kappaganthu, K., Yang, Y., Upadhyay, S., Faruqui, M., Mausam, 2023. AutoMix: Automatically mixing language models. arXiv preprint arXiv:2310.12963 .

de Almeida, A.T., de Almeida, J.A., Costa, A.P.C.S., de Almeida-Filho, A.T., 2016. A new method for elicitation of criteria weights in additive models: Flexible and interactive tradeof. European Journal of Operational Research 250, 179–191. doi:10.1016/j.ejor.2015.08.058.

Barlow, R.E., Proschan, F., 1975. Statistical Theory of Reliability and Life Testing: Probability Models. Holt, Rinehart and Winston, New York.

Barres, V., Dong, H., Ray, S., Si, X., Narasimhan, K., 2025. τ<sup>2</sup>-bench: Evaluating conversational agents in a dual-control environment. arXiv preprint arXiv:2506.07982 .

Belcak, P., Heinrich, G., Diao, S., Fu, Y., Dong, X., Muralidharan, S., Lin, Y.C., Molchanov, P., 2025. Small language models are the future of agentic AI. arXiv preprint arXiv:2506.02153 .

Chen, L., Zaharia, M., Zou, J., 2024. FrugalGPT: How to use large language models while reducing cost and improving performance. Transactions on Machine Learning Research .

Chung, J.W., Ma, J.J., Wu, R., Liu, J., Kweon, O.J., Xia, Y., Wu, Z., Chowdhury, M., 2025. The ML.ENERGY benchmark: Toward automated inference energy measurement and optimization, in: Advances in Neural Information Processing Systems (NeurIPS). ArXiv:2505.06371.

Ciardiello, F., Genovese, A., 2023. A comparison between TOPSIS and SAW methods. Annals of Operations Research 325, 967–994.

Deb, K., 2001. Multi-Objective Optimization Using Evolutionary Algorithms. John Wiley & Sons, Chichester.

Ding, D., Mallick, A., Wang, C., Sim, R., Mukherjee, S., Ruhle, V., Lakshmanan, L.V.S., Awadallah, A.H., 2024. Hybrid LLM: Cost-eficient and quality-aware query routing. arXiv preprint arXiv:2404.14618 .

Elsworth, C., Huang, K., Patterson, D., Schneider, I., Sedivy, R., Goodman, S., Townsend, B., Ranganathan, P., Dean, J., Vahdat, A., Gomes, B., Manyika, J., 2025. Measuring the environmental impact of delivering AI at Google scale. arXiv preprint arXiv:2508.15734 .

Fu, H., Lin, S., Xie, Q., Xiong, G., 2026. Not all tokens are equal: Inflation-aware routing for agentic LLM systems. arXiv preprint arXiv:2608.13571 .

Gritta, M., 2026. Process evaluation for agentic systems, in: Findings of the Association for Computational Linguistics: EACL 2026.

Haimes, Y.Y., Lasdon, L.S., Wismer, D.A., 1971. On a bicriterion formulation of the problems of integrated system identification and system optimization. IEEE Transactions on Systems, Man, and Cybernetics SMC-1, 296–297. doi:10.1109/TSMC.1971.4308298.

Hu, Q.J., Bieker, J., Li, X., Jiang, N., Keigwin, B., Ranganath, G., Keutzer, K., Upadhyay, S.K., 2024. RouterBench: A benchmark for multi-LLM routing system. arXiv preprint arXiv:2403.12031 .

Hwang, C.L., Yoon, K., 1981. Multiple Attribute Decision Making: Methods and Applications. Springer-Verlag, Berlin.

International Organization for Standardization, 2006. ISO 14044:2006 environmental management – life cycle assessment – requirements and guidelines. ISO, Geneva.

Jegham, N., Abdelatti, M., Koh, C.Y., Elmoubarki, L., Hendawi, A., 2025. How hungry is AI? Benchmarking energy, water, and carbon footprint of LLM inference. arXiv preprint arXiv:2505.09598 .

Joonbakhsh, A., Adalı, A.C., Jansen, S., Khunjush, F., Farshidi, S., 2026. HugSelect: An explainable multicriteria decision-support framework for foundation-model selection. arXiv preprint arXiv:2608.08069 .

Kaliszewski, I., Podkopaev, D., 2016. Simple additive weighting: A metamodel for multiple criteria decision analysis methods. Expert Systems with Applications 54, 155–161.

Kapoor, S., Stroebl, B., Siegel, Z.S., Nadgir, N., Narayanan, A., 2025. AI agents that matter. Transactions on Machine Learning Research ArXiv:2407.01502.

Kwa, T., West, B., Becker, J., Deng, A., Garcia, K., Hasin, M., Jawhar, S., Kinniment, M., Rush, N., Von Arx, S., Bloom, R., Broadley, T., Du, H., Goodrich, B., Jurkovic, N., Miles, L.H., Nix, S., Lin, T., Painter, C., Parikh, N., Rein, D., Sato, L.J.K., Wijk, H., Ziegler, D.M., Barnes, E., Chan, L., 2025. Measuring AI ability to complete long software tasks, in: Advances in Neural Information Processing Systems (NeurIPS). ArXiv:2503.14499.

Lei, Y., Xu, J., Liang, C.X., Bi, Z., Li, X., Zhang, D., Song, J., Yu, Z., 2025. Large language model agents: A comprehensive survey on architectures, capabilities, and applications Preprint.

Li, S., Abdelmoniem, A.M., Wang, S., 2026. ProgRouter: Online progress-guided orchestration for multiagent LLM workflows under quality-cost tradeofs. arXiv preprint arXiv:2608.25992 .

Liu, D., Li, J., 2026. Cost-utility alignment in LLM agent trajectories: Profiling, attribution, diagnosis, adaptation, and evaluation. arXiv preprint arXiv:2608.26195 .

Luccioni, A.S., Jernite, Y., Strubell, E., 2024. Power hungry processing: Watts driving the cost of AI deployment?, in: Proceedings of the ACM Conference on Fairness, Accountability, and Transparency (FAccT). doi:10.1145/3630106.3658542.

Miettinen, K., 1999. Nonlinear Multiobjective Optimization. Kluwer Academic Publishers, Boston.

Mittal, S., 2026. How fast do agents rot? An empirical study of long-horizon degradation in LLM agents for production decision-making. arXiv preprint arXiv:2609.01660 .

Nowak, M., 2026. A multi-criteria decision framework for enterprise LLM routing. Information 17, 539. doi:10.3390/info17060539.

Omar, M., 2026. When agentic LLMs trust poisoned tools: Vulnerability of clinical LLMs to adversarial guidelines PMC12934906.

Ong, I., Almahairi, A., Wu, V., Chiang, W.L., Wu, T., Gonzalez, J.E., Kadous, M.W., Stoica, I., 2025. RouteLLM: Learning to route LLMs from preference data, in: Proceedings of the International Conference on Learning Representations (ICLR).

Oviedo, F., Kazhamiaka, F., Choukse, E., Kim, A., Luers, A., Nakagawa, M., Bianchini, R., Lavista Ferres, J.M., 2026. Energy use of AI inference, eficiency pathways, and test-time scaling. Joule , 102430doi:10. 1016/j.joule.2026.102430.

Panigrahy, D., Tyagi, A., 2026. Energy per successful goal: Goal-level energy accounting for agentic AI systems. arXiv preprint arXiv:2605.22883 .

Patil, S.G., Mao, H., Yan, F., Ji, C.C.J., Suresh, V., Stoica, I., Gonzalez, J.E., 2025. The Berkeley function calling leaderboard (BFCL): From tool use to agentic evaluation of large language models, in: Proceedings of the 42nd International Conference on Machine Learning (ICML), pp. 48371–48392.

Saaty, T.L., 1977. A scaling method for priorities in hierarchical structures. Journal of Mathematical Psychology 15, 234–281.

Saaty, T.L., 2008. Decision making with the analytic hierarchy process. International Journal of Services Sciences 1, 83–98.

Šakota, M., Peyrard, M., West, R., 2024. Fly-swat or cannon? Cost-efective language model choice via meta-modeling, in: Proceedings of the 17th ACM International Conference on Web Search and Data Mining (WSDM), pp. 606–615.

Shah, S., Shridhar, K., 2025. Select-then-route: Taxonomy guided routing for LLMs, in: Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing: Industry Track, pp. 425–441.

Sheshanarayana, D., Pal, R.S., Sinha, M., Dasgupta, T., 2026. GAR: Carbon-aware routing for LLM inference via constrained optimization. arXiv preprint arXiv:2605.11603 .

Simon, H.A., 1955. A behavioral model of rational choice. The Quarterly Journal of Economics 69, 99–118. doi:10.2307/1884852.

Sinha, A., Arun, A., Goel, S., Staab, S., Geiping, J., 2026. The illusion of diminishing returns: Measuring long horizon execution in LLMs, in: Proceedings of the International Conference on Learning Representations (ICLR). ArXiv:2509.09677.

Song, W., Huang, Z., Cheng, C., Gao, W., Xu, B., Zhao, G., Wang, F., Wu, R., 2025. IRT-Router: Efective and interpretable multi-LLM routing via item response theory, in: Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (ACL), pp. 15629–15644.

Strubell, E., Ganesh, A., McCallum, A., 2019. Energy and policy considerations for deep learning in NLP, in: Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pp. 3645–3650. doi:10.18653/v1/P19-1355.

Wierzbicki, A.P., 1980. The use of reference objectives in multiobjective optimization, in: Fandel, G., Gal, T. (Eds.), Multiple Criteria Decision Making Theory and Application. Springer, Berlin, Heidelberg. volume 177 of Lecture Notes in Economics and Mathematical Systems, pp. 468–486. doi:10.1007/ 978-3-642-48782-8\_32.

Yang, Y.T., Zhu, Q., 2026. Toward reliable design of LLM-enabled agentic workflows: Optimizing latencyreliability-cost tradeofs. arXiv preprint arXiv:2605.23929 .

Yao, S., Shinn, N., Razavi, P., Narasimhan, K., 2024. τ -bench: A benchmark for tool-agent-user interaction in real-world domains. arXiv preprint arXiv:2406.12045 .

Yehudai, A., Eden, L., Li, A., Uziel, G., Zhao, Y., Bar-Haim, R., Cohan, A., Shmueli-Scheuer, M., 2025. Survey on evaluation of LLM-based agents doi:10.48550/arXiv.2503.16416. arXiv:2503.16416.

Zeleny, M., 1982. Multiple Criteria Decision Making. McGraw-Hill, New York.

Zhu, Y., 2026. LLM-based multi-agent orchestration: A survey of frameworks, communication protocols, and emerging patterns. Future Internet 18, 326.