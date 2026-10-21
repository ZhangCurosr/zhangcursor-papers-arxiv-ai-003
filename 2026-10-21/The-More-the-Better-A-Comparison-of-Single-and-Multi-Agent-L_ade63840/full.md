# The More the Better? A Comparison of Single- and Multi-Agent LLM Setups on a Financial Review Task

Pei Liu

School of Engineering and Applied Science, George Washington University

Washington, DC, USA

pei.liu@gwu.edu

ORCID https://orcid.org/0009-0001-9092-1086

## Abstract

Practitioner guidance holds that a large language model reviews a document better when the work is divided among agents. We test a narrow form of it in a controlled comparison of single-agent and multi-agent setups on a financial task, reviewing credit assessment memos against requirements. One agent checks six requirements in one call. A team of three on the same model checks the same memo, two requirements each, in the same words, with no roles or later stage. Labels are known for every requirement on 300 synthetic memos, so misses and false alarms are measured separately. Two primary predictions are tested on four models at a Bonferroni-corrected level. Under an instruction to decide each requirement once, the team catches 0.08 to 0.10 more of the defects that carry an explanation on the three models whose single agent misses some, at the corrected level on two. It raises more false alarms on all four, at the corrected level on three. Balanced accuracy, an exploratory measure, changes by 0.02 or less, and the team uses 2.2 to 2.4 times the tokens. Showing each agent the requirements it was not given reduces the added false alarms on the two Claude models, at the corrected level on one. In exploratory comparisons, sampling the single agent three times and flagging on any draw gives higher balanced accuracy than the team on those models, and the team’s results depend on the instruction.

Keywords. LLM agents, multi-agent systems, single-agent baselines, task decomposition, LLM-as-a-judge, cost-aware evaluation

## 1 Introduction

Checking a document against a list of written requirements is routine work in finance, where a credit file, an audit paper or a disclosure must satisfy each one. A common way to build such a review with a large language model is to split it among agents. One agent checks the figures, another the calculations, a third the judgement calls, and their findings are merged. A practitioner guide states the idea as a rule, that “for complex tasks with multiple considerations, LLMs generally perform better when each consideration is handled by a separate LLM call” (Schluntz and Zhang 2024). The guide ofers no data in support.

Existing tests of the rule leave three things out. Among those that divide the work without adding stages, none reports a reviewer’s miss rate together with a false-alarm rate measured on cases known to be free of defects. None gives the single agent as many calls as the divided reviewer, and none separates dividing the work from withholding what the other parts cover. A team usually difers from the single agent in more than the division of work, through role descriptions, an orchestrator, a filter or a diferent model for one of its agents.

A designer needs to know more than which arrangement scores higher. A review that catches more defects and raises more false alarms is a diferent instrument, and whether it is a better one depends on what each error costs.

We compare two arrangements with only the allocation of requirements to calls varied. A single agent receives six requirements in one call. A team of three agents on the same model receives the same six in identical words, two to a call, with no roles and no later stage. The documents are synthetic credit memos whose labels are known by construction. The comparison is a narrow form of the guide’s rule. We give two considerations to a call where the rule speaks of one. Our task is six checks of one kind on a short document, where the rule speaks of complex tasks. Our main condition also switches of the models’ separate thinking channel and tells the reviewer to decide each requirement once.

Two further arms separate the efect of division from the efects of what accompanies it. One samples the single agent’s prompt three times, so that it uses as many calls as the team. The other, added after the first results had been read, shows each team agent the requirements it was not given.

Our contribution is to supply all three missing elements in one design. The trade we find, more defects caught for more false alarms, reproduces earlier work. The size of its falsealarm side, measured on cells free of defects, is new, and so is the contrast with a team whose agents see what the others cover. Two exploratory results are also new, a comparison of the trade with repeated draws of the single agent and its dependence on the instruction.

## 2 Related work

Teams against single agents. Tran and Kiela (2026) give a single agent and five multi-agent designs the same cap on thinking tokens and find the single agent ahead on average. Zhang et al. (2025) find that no debate framework beats chain-of-thought prompting in more than a fifth of their settings. Kim et al. (2025) relate a team’s benefit to task properties and to the single agent’s own score. These studies score answers as right or wrong and so cannot say which of a reviewer’s two errors a design trades for the other.

Division of a detection task. On financial documents, He et al. (2026) ask a model either for all error types at once or for one type in each query. With the whole document in the prompt, asking for one type at a time raises average recall by 0.08 to 0.10 and cuts precision by more than half [their Table 6]. The pattern reverses when the document is supplied in chunks, and they conclude from F1 that asking for all types at once is better. Liu et al. (2026) find that a team with no filtering stage, on one model, raises recall from 58.7 to 75.0 percent and lowers precision from 91.5 to 74.2 [their Table 2], and that a single agent given the team’s instruction also gains recall and loses precision, by a smaller amount. Their filter restores the precision. With added stages or debate a team can instead raise fewer false alarms and catch less. Li et al. (2026) report a team with a conductor, a judge and a reranker that passes 85 percent of defect-free artifacts against 48 for a single detector. With the reranker removed the team still passes 64 percent, and they test no team with all of those stages removed. Chen et al. (2025) find that two earlier debate protocols mostly raise precision and lower recall, and D’Arcy et al. (2024) that a team given the single agent’s task prompt matches fewer of the human reviewers’ comments at slightly higher precision.

Criteria in one call or several. Two studies vary only the criteria in a judge call, as we do. Peng et al. (2026) find that balanced accuracy falls once a call holds four or five rubrics on long coding-agent trajectories, and changes little on shorter reports. Rao and Callison-Burch (2026) find that on their two binary checklists a whole rubric in one call and one criterion at a time are equally accurate, while 3 to 14 percent of individual verdicts difer. Neither reports the miss rate and the false-alarm rate for each arrangement. Yu et al. (2026) attribute a team’s benefit to isolated contexts and find that a single agent given the team’s divided instructions in one context scores below the single agent with undivided instructions on most tasks [their Table 3]. Work onjudges that assess the quality ofan output, as opposed to detecting defects in a document, gives mixed results on the same question (Lee et al. 2025; Xiong et al. 2025; Sunkavalli 2026), and none of those tasks has defect-free cases.

Repeated calls. A fair baseline lets the single agent make as many calls as the team. Sampling a model several times and taking the most frequent answer goes back to self-consistency decoding (Wang et al. 2023). Li et al. (2024) treat each sample of one model as an agent and report that accuracy under a vote rises with the number of samples, up to 40, on reasoning and code benchmarks. Chen et al. (2024) show that a majority over repeated calls raises accuracy on queries where the correct answer is the single call’s most likely answer and lowers it on the others. Jain et al. (2025) let four of fourteen judge models reject an item. Teo et al. (2026) count a contradiction as detected when any of several reviews finds it, and report no false-alarm rate for the reviewing system.

Framing and gap. We take our framing from Fox et al. (2026). Of every remedy applied to a judge they ask whether it “slides the point along a curve or lifts the curve”, and they find that wording, voting and prompt search slide it. What lifts it in their study is a two-stage pipeline that extracts the facts to check and verifies each in its own call. The pipeline adds a stage and a closed output format, so it is not a division of one task among calls of the kind we test. Two controls in the studies above resemble the scoped team that Section 3.3 defines, and difer from it. Liu et al. give a single agent the team’s instruction, and Yu et al. give a single agent the team’s divided instructions in one context. Both alter the single agent. The scoped team leaves the division in place and restores information to the divided agents. Among the studies we examined, we found none that sets a division without added stages against repeated draws, or that separates division from the loss of that information.

<table><tr><td>Pair</td><td>Requirement</td></tr><tr><td>A</td><td>R1. Every statement of the income used in the affordabil- ity or servicing calculation must give the monthly income stated in Employment.</td></tr><tr><td>A</td><td>R2. Every statement of the approved facility limit must give the same amount, and any stated change to the limit must</td></tr><tr><td>B</td><td>arrive at that amount. R3. Every sum the memo writes out about monthly income must be arithmetically correct, whether it adds an amount, deducts one, or compares the income with an earlier figure.</td></tr><tr><td>B</td><td>R4. Every income multiple quoted for the facility limit must follow from the approved limit and the monthly income stated in Employment.</td></tr><tr><td>C</td><td>R5. Every statement about the repayment record must agree with the six periods shown in the repayment history.</td></tr><tr><td>C</td><td>R6. Every stated period of employment or sector experience must be possible given the applicant&#x27;s age and the age at which education was completed.</td></tr></table>

Table 1: The six requirements and the three pairs in which the team receives them.

## 3 Experimental design

## 3.1 Setting

We study one situation. Several agents built on the same model each receive the same complete document, and each is responsible for part of a single review whose parts draw on overlapping passages. The agents run in parallel, do not communicate and have no tools, and the document fits comfortably in one context window. A team can help for reasons that this design excludes, such as a task that does not fit in one context, subtasks that share no input, or agents with diferent models or tools. Our result says nothing about those cases.

## 3.2 Task and ground truth

The task is the review ofa credit assessment memo against six written requirements. Each memo is written by a program, with no language model, from one record of a public credit card dataset (Yeh and Lien 2009; Yeh 2009). A memo has about 1,100 words in eight sections, and we use 300 of them.

Each requirement is labelled on each memo. We call one requirement on one memo a cell, so there are 1,800 cells. A cell is clean when every statement the requirement governs is correct. Otherwise it has one of three kinds of defect. We write $y _ { i j } = 1$ when memo i has a defect under requirement j and $y _ { i j } = 0$ when the cell is clean. A bare defect is a statement that conflicts with another statement of the same fact, with no reason given. A masked defect comes with its own explanation and is consistent in itself, yet contradicts other facts in the memo. An arithmetic defect is a written-out sum or ratio that is wrong. Under R1, for example, one memo with a bare defect gives monthly income as NT\$23,200 and then computes a ratio “against monthly income of NT\$46,900”. One with a masked defect gives monthly income as NT\$37,000 and then states that the servicing test “was run on monthly income of NT\$31,700, being the stated figure net of NT\$5,300 in non-recurring overtime”. There are 1,136 clean cells, and 205 bare, 229 masked and 230 arithmetic defects. That puts a defect in 37 percent of cells, far more than in practice. The median memo has two defects, and 21 memos have none.

A label was accepted when three sources agreed. The first was the generator that set it. The second was a separate program that recomputed it from the finished text. The third was Claude Opus 5.5, a larger model than any we test, which read the memo without seeing either label. All 1,800 labels passed after one revision to the wording of two requirements, which Appendix A describes. Every defect is an inconsistency inside the document that a program can verify. Omissions, errors against outside facts and matters of judgement are not in the test.

## 3.3 Arms

By “agent” we mean one call of a model with its own instructions and its own context, and by “team” several agents whose verdicts are taken together. Each agent is a single call, so the design can equally be described as a decomposition of a judge prompt, and Schluntz and Zhang (2024) would class it as a parallel workflow. Our team is the simplest kind, with no orchestrator or filter, and in that respect resembles the architecture that Kim et al. (2025) call independent. Four arms review every memo.

Let R be the set of six requirements and $d _ { i }$ memo i. A call $f ( d _ { i } , S , C )$ gives the model the memo, a set $S \subseteq R$ of requirements to judge and a set $C \subseteq R \backslash S$ of requirements shown for context only. It returns a verdict $v _ { i j } \in \{ 0 , 1 \}$ for each $j \in S$ , where 1 is a flag. The model, the memo, the wording of each requirement and the rest of the prompt are the same in every call. The arms difer in $S ,$ in $\dot { C }$ and in the number of calls. They difer otherwise only in the order of the requirement list, given at the end of this section, and in the seeds and, on the Qwen3.5 models, the sampling temperature of the three draws, which Section 3.4 describes.

$$
\begin{array} { r l } { \mathrm { S i n g l e ~ a g e n t ~ } } & { f ( d _ { i } , R , \emptyset ) } \\ { \mathrm { T e a m ~ } } & { f ( d _ { i } , P _ { m } , \emptyset ) , \quad m = 1 , 2 , 3 } \\ { \mathrm { S c o p e d ~ t e a m ~ } } & { f ( d _ { i } , P _ { m } , R \setminus P _ { m } ) , \quad m = 1 , 2 , 3 } \\ { \mathrm { T h r e e ~ d r a w s ~ } } & { v _ { i j } = 1 \mathrm { ~ i f ~ } \sum _ { s = 1 } ^ { 3 } v _ { i j } ^ { ( s ) } \geq t } \end{array}
$$

Here $\{ P _ { 1 } , P _ { 2 } , P _ { 3 } \}$ is the partition of R into the three pairs of Table 1, and a requirement’s verdict comes from the call whose S contains it. The pairs are thematic and fixed, and we test this one partition only. The single agent is the partition with one part, and the guide’s rule of one consideration to a call is the partition with six.

The team changes two things at once. A team agent holds fewer requirements, and it does not see the four it was not given. The scoped team separates the two by setting $C$ to those four, which appear under a heading that says other reviewers judge them.

For the three draws, $v _ { i j } ^ { ( s ) }$ is the verdict of the s-th sample of the single agent’s call and t is the number of flags required. The majority rule is $t = 2 .$ , the any-of-three rule is $t = 1$ and the all-of-three rule, $t = 3 ,$ , is used only for the line in Figure 1. In the team a single call decides each requirement, so the majority rule is stricter for flagging than the team’s.

The order of the requirements in the single agent’s list is rotated across memos and across draws. The order within a team call is fixed.

## 3.4 Models and instruction

We use four models from two families. Qwen3.5 at 9B and 27B parameters run locally, and Claude Haiku 5.5 and Claude Sonnet 5.5 through a batch interface. Every comparison is within one model. Answers are returned under a JSON schema that holds, for each requirement, a reasoning string, a verdict of PASS or FAIL and a quoted evidence sentence. Separate thinking channels are switched of, and the output cap is 2,048 tokens for each requirement in every arm. The Qwen3.5 models are sampled at temperature 0, and at 0.7 for the three draws, which would otherwise be near copies of one another. The Claude models accept no temperature setting and use default sampling in every arm. Comparisons between the three draws and the other arms are therefore confounded with temperature on the Qwen3.5 models and not on the Claude models.

All arms carry the same instruction on how to reason. The reviewer is told to take each requirement once, to say in at most three sentences what it compared, and not to return to a requirement after deciding it. We call this the decideonce instruction. We made it the main condition before the runs reported here, for a measurement reason. Left to reason freely in a preliminary run, which Appendix C describes, Qwen3.5 27B returned no verdict on about 6 percent of calls. A failed single-agent call forfeits six verdicts and a failed team call two, so the losses would have fallen unevenly on the arms. The instruction departs from default practice, and it has efects of its own. On 100 of the 300 memos, selected with a fixed seed, every arm is also run without it, and Section 4.4 reports what changes for the single agent and the team.

## 3.5 Scoring and analysis

A flag is a verdict of FAIL, and a false alarm is a flag on a clean cell. For an arm a with verdicts $v _ { i j } ^ { a } .$ , where a is single, team, scoped, or a rule on the three draws, we report the share of defects caught, $H _ { a }$ , and the false-alarm rate on clean cells, $F _ { a }$

$$
H _ { a } = \frac { \sum _ { i , j } y _ { i j } v _ { i j } ^ { a } } { \sum _ { i , j } y _ { i j } } , \qquad F _ { a } = \frac { \sum _ { i , j } \bigl ( 1 - y _ { i j } \bigr ) v _ { i j } ^ { a } } { \sum _ { i , j } \bigl ( 1 - y _ { i j } \bigr ) }
$$

Within one class of defect, with both sums restricted to that class, the share caught is written $H _ { a } ^ { \mathrm { b a r e } } , H _ { a } ^ { \mathrm { m a s k e d } }$ or

$H _ { a } ^ { \mathrm { a r i t h } }$ . Balanced accuracy is $B _ { a } = ( H _ { a } + 1 - F _ { a } ) / 2$ , which weights the two errors equally whatever their frequency. We write $\Delta H = H _ { \mathrm { t e a m } } - \bar { H _ { \mathrm { s i n g l e } } }$ for the team’s value minus the single agent’s, and $\Delta F$ and ∆B likewise. Other contrasts are written out, as in $F _ { \mathrm { s c o p e d } } - F _ { \mathrm { t e a m } } .$ . An arm returns verdicts and not scores, so each arm gives a single operating point $( F _ { a } , H _ { a } )$ . The sums run over all memos, so rates are pooled over memos and, for all defects, over the three classes. The tokens $K _ { a }$ of an arm are its input and output tokens together.

A missing verdict counts against the arm, and intervals are bootstrap intervals over memos, at 95 percent unless stated. We call a contrast significant when its interval excludes zero. Appendix A gives both rules in full. Each arm was run once, so the intervals cover the sampling of memos and the noise of calls within a run, and do not cover variation between whole runs. Repeated calls exist for the single agent’s prompt only, and Table A8 gives their spread.

The memos, prompts, predictions and analysis code were fixed by a committed hash before each family was run. Five predictions were set from a pilot, an earlier run of the same comparison on a separate memo set that contributes no number reported here. In the pilot the team’s gain appeared on defects that carry an explanation and not on bare ones, which is why the masked contrast is primary. Two were primary, that $\bar { \Delta } H ^ { \mathrm { m a s k e d } } > 0$ and that $\bar { \Delta } F > \bar { 0 }$ . Each model family has two models, so the two primary predictions give four tests per family. Every primary test is reported at the Bonferroni level of 98.75 percent, which we call the corrected level, and the secondary predictions are tested at 95 percent. At that level the interval on the masked contrast has a half-width of about 0.11 on Qwen3.5 9B, 0.08 on Qwen3.5 27B and 0.07 on Haiku 5.5, so an efect near 0.08 could be detected on some models and not on others. Three predictions were secondary. They were that the team and the single agent do not difer on bare defects, that the team catches more masked defects than a majority of three draws, and that its excess of false alarms is smaller on the larger model of a family.

The scoped team was added after the other arms had been read. Its two predictions, fixed before it was run, were that it raises fewer false alarms than the team and that it still catches more masked defects than the single agent. Neither is a primary prediction of the study. The idea came from reading the Sonnet 5.5 team’s false alarms, and the arm is compared with that same team run. We report its contrasts at the corrected level, and Appendix B notes which also hold at 99.375 percent, the level for eight tests per family.

Everything else is exploratory. That covers balanced accuracy, the signal detection analysis of Appendix D, the any-ofthree rule and the comparison of arms with the line through the three-draw rules, the efect of the instruction, the breakeven costs, and the rates by requirement and by memo. Appendix B lists every prediction with its outcome.

## 4 Results

No arm on any model lost more than two calls under the decide-once instruction. Table 2 reports the rates and Figure 1 shows each arm as an operating point. The team uses 2.2 to 2.4 times the single agent’s tokens, and the three draws use more.

## 4.1 The team flags more, and catches more below ceiling

On the three models whose single agent leaves masked defects uncaught, the team’s share caught is higher by 0.08 to 0.10, as Figure 2 shows. The prediction is confirmed at the corrected level on Qwen3.5 27B and Haiku 5.5 and not on Qwen3.5 9B, where the estimate equals that on Qwen3.5 27B and the interval is wider. The team raises more false alarms on every model, at the corrected level on three. The secondary prediction of no diference on bare defects was wrong. On the three models below ceiling the team catches more of them, as Appendix B records. In Figure 1 the segment from single agent to team rises to the right on those models.

With the two errors weighted equally the efects largely cancel. Balanced accuracy changes by 0.02 or less on every model, and all four intervals lie within 0.04 of zero. We set no equivalence margin in advance, and the change is significant on the two Claude models, a rise on Haiku 5.5 and a fall on Sonnet 5.5. Sonnet 5.5’s single agent catches every bare and masked defect, and its team catches two fewer defects and adds 15 false alarms.

As predicted for the larger model of a family, the team’s excess of false alarms is smaller on Qwen3.5 27B than on Qwen3.5 9B, as Appendix B records. In proportion to each model’s own false-alarm rate the two excesses are about the same, and the two Claude models do not difer.

The pooled rates conceal changes in opposite directions. The team’s share of defects caught rises on some requirements and falls on others, as Table A7 shows, so the trade describes the review as a whole and not each part of it. On the pooled rates, an exploratory signal detection analysis in Appendix D is consistent with a lower threshold for flagging and an unchanged ability to tell defective cells from clean ones, and does not establish that account.

## 4.2 Scope information

We read the 15 false alarms of the Sonnet 5.5 team. Most fall on requirement R1. The agent holding R1 flags the sentence that adds a housing allowance to monthly income. That sentence falls under R3, a requirement this agent was not shown. The single agent holds both and does not make the error. By the rates in Table A7, R1 and R3 account for about 0.8 of the team’s added false alarms on Sonnet 5.5 and Qwen3.5 27B. On Haiku 5.5 most of the addition falls on R4, and on Qwen3.5 9B it is spread across five requirements. The scoped team restores to each agent the requirements it was not shown.

The scoped team raises fewer false alarms than the team on Sonnet 5.5, at the corrected level, as Figure 3 shows. The reduction on Haiku 5.5 is the same size and is significant at 95 percent, [−0.021, −0.002]. On Qwen3.5 27B the reduction is not significant, and on Qwen3.5 9B there is none. The scoped team still catches more masked defects than the single agent on Haiku 5.5. That prediction is not confirmed on Qwen3.5 9B, and on Qwen3.5 27B the interval reaches zero.

The reduction may not be free. On Qwen3.5 27B and Haiku 5.5 the scoped team’s share of all defects caught is about 0.02 below the team’s, mostly through arithmetic defects. On those two models each 0.01 taken of the false-alarm rate comes with a fall of about 0.015 in the share of defects caught, which is not significant. Sonnet 5.5 gives up nothing.

<table><tr><td>Model</td><td>Arm</td><td>Hbare</td><td>Hmasked</td><td>Harith</td><td>H</td><td>F</td><td>B</td><td> $K _ { a } / K _ { \mathrm { s i n g l e } }$ </td></tr><tr><td>Qwen3.5 9B</td><td>Single agent</td><td>0.717</td><td>0.559</td><td>0.574</td><td>0.613</td><td>0.160</td><td>0.726</td><td>1.00</td></tr><tr><td></td><td>Team</td><td>0.849</td><td>0.642</td><td>0.722</td><td>0.733</td><td>0.254</td><td>0.740</td><td>2.36</td></tr><tr><td></td><td>Scoped team</td><td>0.898</td><td>0.646</td><td>0.730</td><td>0.753</td><td>0.273</td><td>0.740</td><td>2.55</td></tr><tr><td></td><td>Three draws, majority</td><td>0.756</td><td>0.402</td><td>0.626</td><td>0.589</td><td>0.140</td><td>0.724</td><td>3.11</td></tr><tr><td></td><td>Three draws, any</td><td>0.902</td><td>0.795</td><td>0.870</td><td>0.854</td><td>0.351</td><td>0.751</td><td>3.11</td></tr><tr><td>Qwen3.5 27B Single agent</td><td></td><td>0.971</td><td>0.734</td><td>0.691</td><td>0.792</td><td>0.047</td><td>0.873</td><td>1.00</td></tr><tr><td></td><td>Team</td><td>1.000</td><td>0.817</td><td>0.704</td><td>0.834</td><td>0.076</td><td>0.879</td><td>2.24</td></tr><tr><td></td><td>Scoped team</td><td>0.995</td><td>0.808</td><td>0.661</td><td>0.815</td><td>0.062</td><td>0.876</td><td>2.40</td></tr><tr><td></td><td>Three draws, majority</td><td>1.000</td><td>0.803</td><td>0.735</td><td>0.840</td><td>0.048</td><td>0.896</td><td>3.06</td></tr><tr><td></td><td>Three draws, any</td><td>1.000</td><td>0.961</td><td>0.913</td><td>0.956</td><td>0.174</td><td>0.891</td><td>3.06</td></tr><tr><td>Haiku 5.5</td><td>Single agent</td><td>0.946</td><td>0.790</td><td>0.752</td><td>0.825</td><td>0.013</td><td>0.906</td><td>1.00</td></tr><tr><td></td><td>Team</td><td>1.000</td><td>0.891</td><td>0.752</td><td>0.877</td><td>0.025</td><td>0.926</td><td>2.16</td></tr><tr><td></td><td>Scoped team</td><td>0.990</td><td>0.882</td><td>0.717</td><td>0.858</td><td>0.013</td><td>0.923</td><td>2.30</td></tr><tr><td></td><td>Three draws, majority</td><td>0.966</td><td>0.803</td><td>0.774</td><td>0.843</td><td>0.003</td><td>0.920</td><td>3.00</td></tr><tr><td></td><td>Three draws, any</td><td>0.995</td><td>0.908</td><td>0.891</td><td>0.929</td><td>0.029</td><td>0.950</td><td>3.00</td></tr><tr><td>Sonnet 5.5</td><td>Single agent</td><td>1.000</td><td>1.000</td><td>0.983</td><td>0.994</td><td>0.000</td><td>0.997</td><td>1.00</td></tr><tr><td></td><td>Team</td><td>1.000</td><td>0.996</td><td>0.978</td><td>0.991</td><td>0.013</td><td>0.989</td><td>2.17</td></tr><tr><td></td><td>Scoped team</td><td>1.000</td><td>1.000</td><td>0.983</td><td>0.994</td><td>0.003</td><td>0.996</td><td>2.32</td></tr><tr><td></td><td>Three draws, majority</td><td>1.000</td><td>1.000</td><td>0.991</td><td>0.997</td><td>0.000</td><td>0.998</td><td>3.00</td></tr><tr><td></td><td>Three draws, any</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td><td>0.001</td><td>1.000</td><td>3.00</td></tr></table>

Table 2: Share of defects caught by class and over all defects, H, false alarms on clean cells, F, balanced accuracy, B, and tokens relative to the single agent. The majority rule flags when at least two of three draws flag, and the any rule when at least one does. All 300 memos, decide-once instruction.

![](images/ce7c285d24749f79141195d566725ac280d6ccdcda8bb379906f2d17509cd702.jpg)  
Figure 1: Each arm as an operating point, false alarms on clean cells against the share of all defects caught, with the Claude models enlarged on the right. A thick segment joins a model’s single agent to its team, and a star marks the scoped team. A thin line joins three rules applied to three draws of the single agent’s prompt, flagging when all, at least two or any of the draws flag. Bars on the single agent and the team are 95 percent intervals. Grey lines join points of equal balanced accuracy.

Scope information may act by telling the agent what lies outside its requirements, or by making it more cautious because other reviewers are named. Our prompt does not separate the two. In Table 2, Haiku 5.5’s scoped team has the single agent’s false-alarm rate, catches more masked defects and catches fewer arithmetic defects. The scoped prompt adds about 160 to 250 input tokens a call to the team’s.

![](images/b07e5b3b9daef845484c590c13de43720996f8b4b17b4ba879aea87376e0982c.jpg)  
Figure 2: The team against the single agent, on one scale. The contrasts in masked defects caught, $\Delta H ^ { \mathrm { m a s k e d } }$ , and in false alarms, $\Delta F ,$ , are primary, with intervals at the corrected level. The contrast in balanced accuracy, ∆B, is exploratory, with 95 percent intervals. A filled marker has an interval that excludes zero, and the number beside each row is the estimate. Table A1 gives the bounds.

![](images/8a5de67e30309fb3cdbd16c796ba18be0e39aa5f936c617cff2518be05e64fcb.jpg)  
Figure 3: The scoped team, on the scale of Figure 2. Its contrast with the single agent in masked defects caught and its contrast with the team in false alarms are the two pre-specified contrasts, with intervals at the corrected level. Its contrast with the team in all defects caught is exploratory, with 95 percent intervals. Markers and numbers are as in Figure 2. Table A2 gives the bounds.

## 4.3 Repeated draws

The remaining secondary prediction was that the team catches more masked defects than a majority of three draws. It is confirmed on Haiku 5.5 and not on Qwen3.5 27B or Sonnet 5.5. On Qwen3.5 9B the estimate is above zero, but on the Qwen3.5 models the comparison is confounded with temperature. Appendix B gives the estimates. A draw of the 9B model catches fewer masked defects than its single agent at temperature 0, as Table A8 shows.

Under the exploratory any-of-three rule the three-draw agent catches more defects than the team on all four models, and its balanced accuracy is no lower than the team’s on any. On the two Claude models it is higher by 0.024 and 0.011, both significant.

A further exploratory check, unlike the signal detection analysis of Appendix D, makes no distributional assumption. It compares each arm with the line in Figure 1 that joins the three rules on three draws, taken at the arm’s own false-alarm rate. Haiku 5.5’s team lies 0.038 below the line [−0.065, 0.003], and its single agent 0.052 below [−0.093, −0.018]. On that model repeated draws therefore attain operating points that one call does not, and probably points that the team does not. On the Qwen3.5 models the team is 0.002 above the line on the 9B model and 0.032 below it on the 27B model, neither significant. The check cannot be made on Sonnet 5.5, which is at ceiling.

The draws also show how much repeated calls of one prompt vary. They disagree on 0.39 of cells on Qwen3.5 9B, 0.21 on Qwen3.5 27B, 0.11 on Haiku 5.5 and under 0.01 on Sonnet 5.5. By the repeats in Table A8, the team’s share of masked defects caught on Haiku 5.5 lies outside the range of four calls of the single agent’s prompt, while on Qwen3.5 27B three draws vary as much as the team’s gain. The single agent’s share of defects caught also difers across the three rotations of its list, by up to 0.14 on Qwen3.5 9B, 0.09 on Qwen3.5 27B and 0.05 on Haiku 5.5. Each rotation covers a diferent 100 memos, so the order of the list and the memos are not separated.

## 4.4 The decide-once instruction, exploratory

On the 100 memos run with and without the instruction, removing it raises the single agent’s share of defects caught. With calls that share a prompt and settings combined, masked defects caught rise significantly on the three models below ceiling, as Table A3 shows. Haiku 5.5’s false alarms do not rise with them. For the Qwen3.5 models these combined calls are the draws at temperature 0.7. In the single call at temperature 0, which Table A3 also reports, arithmetic defects caught rise on both models and masked defects caught do not rise significantly on either.

The efect of division also depends on the instruction, as Table 3 shows. Without it the team’s excess of false alarms is larger on all four models, significantly on Qwen3.5 27B, +0.088 [0.013, 0.161], and Sonnet 5.5, +0.013 [0.003, 0.026]. The team’s balanced accuracy is then significantly lower than the single agent’s on Qwen3.5 27B and Sonnet 5.5 and significantly higher on Haiku 5.5. On Qwen3.5 9B the diference is not significant. On Qwen3.5 27B the team’s gain in defects caught disappears while its excess of false alarms grows. Team calls cut of at the output cap account for part of this, as Appendix C records. On completed verdicts alone the team still raises more false alarms and catches almost no more defects.

Part of the team’s gain on the Qwen3.5 models under the instruction may be a recovery of detection that the instruction takes from the single agent. A direct comparison on these 100 memos, the single agent without the instruction against the team with it, does not settle the question, as Table A4 shows.

## 4.5 What the trade costs, exploratory

Table 4 expresses the trade of Section 4.1 as the rise in the false-alarm rate for each unit rise in the share of all defects caught. Whether the exchange is worthwhile depends on how common defects are. Let p be the share of cells that have a defect, $w _ { M }$ the cost of a missed defect and w the cost of a false alarm. With cost taken as linear in the two errors, arm a has an expected cost of errors per cell of

$$
E _ { a } = p \left( 1 - H _ { a } \right) w _ { M } + \left( 1 - p \right) F _ { a } w _ { F } .
$$

With $\Delta H > 0$ over all defects, the team has the lower expected cost when

$$
\frac { w _ { M } } { w _ { F } } > \frac { \Delta F } { \Delta H } \cdot \frac { 1 - p } { p } .
$$

At the defect share of our memos the break-even cost is between 0.4 and 1.3, and at shares of 0.05 to 0.01 it is roughly 10 to 60 times higher. The ratios are imprecise. At a share of 0.02 the interval on Qwen3.5 27B alone gives break-even costs from 10 to 134. Sonnet 5.5 has no row, since its team catches no more than its single agent.

Two cautions apply to Table 4. It uses the false-alarm rate by cell, which probably overstates the team’s cost, as Section 5 explains. It also compares the team with the single agent only. We set the five rows that Table 2 gives for each model side by side at a defect share of 0.02, for costs of a miss from 1 to 200 times that of a false alarm. A majority of three draws has the lowest $E _ { a }$ while a miss costs less than roughly 15 to 55 false alarms, depending on the model. Above that the any-of-three rule has the lowest. The team has the lowest expected cost at none of the ratios we tried, on any model. On Qwen3.5 9B it lies on the line joining the two three-draw rules, so the ranking there is not robust. Table A10 reports the share of flags that would be correct at three defect shares.

The unit of work in practice is the file. The share of the 300 memos with at least one false alarm is higher for the team than for the single agent on every model, by 0.04 to 0.19. Table A6 reports related counts for whole memos.

## 5 Threats to validity and follow-up studies

Four threats could change a conclusion. For the first three we give the study that would address it, and we rank those studies by how far they could change these conclusions.

How well the efects are measured. On Haiku 5.5 most of the team’s flags on clean cells quote a sentence that is defective under another requirement, and one of the single agent’s does. We call these misfiled flags, and Table A9 counts them for every model. The count rests on an exploratory match of text, supported by a hand check of twelve flags. The team’s cost in false alarms is therefore probably overstated on Qwen3.5 27B, Haiku 5.5 and Sonnet 5.5. Its gain on masked defects is less secure on Qwen3.5 27B, where three draws of one prompt vary as much as the gain, than on Haiku 5.5, where the team lies outside the range of four calls. A study that scores flags by what they quote, repeats its arms and samples the three draws at the temperature of the other arms would establish whether the false-alarm cost that defines the trade is real. We rank it second.

The instruction. Every primary test was run under the decide-once instruction, and Section 4.4 shows that on Qwen3.5 27B the team’s balanced accuracy does not differ significantly from the single agent’s with the instruction and is lower without it. The primary results hold for one way of telling the reviewer to reason. Crossing the division with the instruction on the full memo set is the study we rank first, since this is the only threat for which our own data show a result changing.

The division. We test one pairing, two requirements to a call. R1 and R3 fall in diferent pairs, and Section 4.2 traces most of the team’s added false alarms on two models to them. The efect sizes are specific to this pairing, and the guide’s rule as stated, one consideration to a call, is untested. Varying the division, down to one requirement a call, would test it and ranks third.

Generality. The memos are short and synthetic, with internal inconsistencies in 37 percent of cells, and the strongest model is at ceiling. Neither the rates nor the costs should be assumed to carry over to long documents with rare and varied defects, where outcomes should also be reported by file.

## 6 Guidance for agent setup

Whether a division of this kind helps depends on what the single agent misses, on the model, and on how the reviewer is told to reason. Three implications for the design of a review system follow, ordered by the strength of the evidence behind them. All three assume that rates measured on these memos carry over to the designer’s documents, which we have not shown.

The first rests on the primary comparison and holds under the decide-once instruction for the pairing we tested. Division should be treated as a change in what is flagged and not as an improvement, and its cost in false alarms should be worked out before it is adopted. The gain is bounded by what the single agent misses, and the false alarms are not, as the Sonnet 5.5 rows of Table 2 show. A designer therefore needs the single agent’s miss rate and false-alarm rate on cases known to be clean, and the defect share of the setting. With those, the exploratory calculation behind Table 4 gives the cost of a miss above which division is worthwhile relative to a single call.

The second rests on an exploratory comparison that is free of the temperature confound on the two Claude models only.

<table><tr><td>Model</td><td>∆H, with</td><td>∆H, without</td><td> $\Delta F ,$  with</td><td> $\Delta F ,$  without</td><td>∆B, with</td><td> $\Delta B ,$  without</td></tr><tr><td>Qwen3.5 9B</td><td>+0.133 [0.050, 0.214]</td><td>+0.084 [0.000, 0.164]</td><td>+0.110 [0.065, 0.157]</td><td>+0.126 [0.072, 0.179]</td><td>+0.012 [-0.037, 0.060]</td><td>-0.021 [-0.068, 0.026]</td></tr><tr><td>Qwen3.5 27B</td><td>+0.071 [0.017, 0.126]</td><td>+0.004 [-0.046, 0.052]</td><td>+0.035 [0.006, 0.064]</td><td>+0.123 [0.050, 0.192]</td><td>+0.018 [-0.013, 0.049]</td><td>-0.059 [-0.105, -0.013]</td></tr><tr><td>Haiku 5.5</td><td>+0.022 [-0.022, 0.066]</td><td>+0.071 [0.026, 0.118]</td><td>+0.008 [-0.008, 0.026]</td><td>+0.019 [0.000, 0.038]</td><td>+0.007 [-0.016, 0.030]</td><td>+0.026 [0.001, 0.053]</td></tr><tr><td>Sonnet 5.5</td><td>-0.004 [-0.019, 0.009]</td><td>-0.013 [-0.030, 0.000]</td><td>+0.008 [0.000, 0.018]</td><td>+0.021 [0.008, 0.037]</td><td>-0.006 [-0.015, 0.002]</td><td>-0.017 [-0.028, -0.008]</td></tr></table>

Table 3: The contrasts ∆H over all defects, $\Delta F$ and $\Delta B$ on the 100 paired memos, with the decide-once instruction and without it. 95 percent intervals.

<table><tr><td>Model</td><td> $\Delta F / \Delta H$ </td><td> $p = 0 . 3 7$ </td><td> $p = 0 . 0 5$ </td><td> $p = 0 . 0 2$ </td><td> $p = 0 . 0 1$ </td></tr><tr><td>Qwen3.5 9B</td><td>0.77 [0.48, 1.33]</td><td>1.3</td><td>15</td><td>38</td><td>77</td></tr><tr><td>Qwen3.5 27B</td><td>0.69 [0.21, 2.73]</td><td>1.2</td><td>13</td><td>34</td><td>68</td></tr><tr><td>Haiku 5.5</td><td>0.22 [0.01, 0.65]</td><td>0.4</td><td>4</td><td>11</td><td>22</td></tr></table>

Table 4: Team against single agent under the decide-once instruction. The second column is $\Delta F / \Delta H$ , with $\Delta H$ over all defects and a 95 percent interval. The others are the break-even value of $w _ { M } / w _ { F }$ at four values of the defect share $p ,$ assuming the rates carry over.

The single agent sampled three times is the baseline against which a divided review should be judged, since it uses the same number of calls. On those models the any-of-three rule gives higher balanced accuracy than the team, and a majority of three raises no more false alarms than one call, at a higher token cost than the team, as Table 2 shows. Which rule suits a setting depends, as before, on the cost of a miss.

The third rests on 100 memos and is the least secure. How the reviewer is told to reason should be fixed and reported before arrangements are compared, because the instruction changed the efect of division. On Qwen3.5 27B the team’s balanced accuracy does not difer significantly from the single agent’s with the instruction and is lower without it, as Table 3 shows. Removing a constraint on reasoning also risks calls that return no verdict.

We recommend one practice whatever the follow-up studies of Section 5 find. A divided review should be judged on misses and false alarms separately, on cases known to be clean, and against the single agent given the same number of calls.

## 7 Conclusion

Under the decide-once instruction, dividing six requirements among three agents on one model trades false alarms for defects caught, and with the two errors weighted equally the efects largely cancel, as Figure 2 shows. The trade of more defects caught for more false alarms is what He et al. (2026) found with the whole document in the prompt and what Liu et al. (2026) found for a team with no filter, and the smal change in accuracy is what Rao and Callison-Burch (2026) found for binary checklists. In the terms of Fox et al. (2026), the exploratory signal detection analysis of Appendix D is consistent with division sliding the operating point and does not establish it. Repeated draws behave diferently in our data from theirs. On Haiku 5.5 three draws attain operating points that one call does not, as Section 4.3 shows. Their task and their voting rule difer from ours, and we have not tested which of the two explains the disagreement.

Besides the exploratory comparisons, two things are new. The false-alarm rate is measured on cells known to be clean, which lets the trade be sized. The scoped team, an arm added after the first results, raises fewer false alarms than the team on the two Claude models, at the corrected level on one.

## References

Chen, L.; Davis, J. Q.; Hanin, B.; Bailis, P.; Stoica, I.; Zaharia, M.; and Zou, J. 2024. Are More LLM Calls All You Need? Towards the Scaling Properties of Compound AI Systems. In Advances in Neural Information Processing Systems 37, 45767–45790. ArXiv:2403.02419.

Chen, Y.; Niu, G.; Cheng, J.; Han, B.; and Sugiyama, M. 2025. When and Why Does Multi-Agent Debate Fail and Does It Really Underperform? arXiv:2510.20963.

D’Arcy, M.; Hope, T.; Birnbaum, L.; and Downey, D. 2024. MARG: Multi-Agent Review Generation for Scientific Papers. arXiv:2401.04259.

Fox, S.; Markham, L.; Lail, R.; and Karotsieris, M. 2026. LLM Judges Verify Presence, Not Absence: Omission Blindness in AI Clinical Notes and What Recovers It. arXiv:2608.31016.

He, Y.; Gu, Z.; Hu, Z.; Zhou, Y.; Shen, H.; Liang, J.; Dai, Z.; Ma, S.; Yu, F.; Xiao, Y.; and Li, Z. 2026. Are Large Language Models Reliable Reviewers? A Benchmark for Error Detection in Financial Documents. arXiv:2608.12342.

Jain, S.; Ahmed, U. Z.; Sahai, S.; and Leong, B. 2025. Beyond Consensus: Mitigating the Agreeableness Bias in LLM Judge Evaluations. arXiv:2510.11822.

Kim, Y.; Gu, K.; Park, C.; Park, C.; Schmidgall, S.; Heydari, A. A.; Yan, Y.; Zhang, Z.; Zhuang, Y.; Liu, Y.; Malhotra, M.;

Liang, P. P.; Park, H. W.; Yang, Y.; Xu, X.; Du, Y.; Patel, S.; Althof, T.; McDuf, D.; and Liu, X. 2025. Towards a Science of Scaling Agent Systems. arXiv:2512.08296.

Lee, Y.; Kim, J.; Kim, J.; Cho, H.; Kang, J.; Kang, P.; and Kim, N. 2025. CheckEval: A Reliable LLM-as-a-Judge Framework for Evaluating Text Generation Using Checklists. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, 15771–15798.

Li, J.; Zhang, Q.; Yu, Y.; Fu, Q.; and Ye, D. 2024. More Agents Is All You Need. Transactions on Machine Learning Research. ArXiv:2402.05120.

Li, Z.; Guo, Z.; Lu, X.; Wang, J.; Song, J.; Shen, C.; Wu, J.; and Sun, M. 2026. OptArgus: A Multi-Agent System to Detect Hallucinations in LLM-based Optimization Modeling. arXiv:2605.11738.

Liu, W.; Zhao, R.; Yuan, W.; Xu, J.; Xu, Z.; Zhang, X.; and Ngai, E. C.-H. 2026. Dude: A Dual-Detection Multi-Agent System for Paper-Code Discrepancy Detection. arXiv:2609.03416. To appear in Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing.

Peng, Y.; Qi, Y.; Xia, H.; He, G.; Shi, X.; Xuan, R.; Lu, S.; Liu, Y.; Hu, Z.; Liu, Y.; and Peng, H. 2026. Can LLMas-a-Judge Reliably Verify Rubrics in Agentic Scenarios? arXiv:2606.29920. To appear in Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing.

Rao, D.; and Callison-Burch, C. 2026. JEV vs. LLMs as Rubric Judges: Cheaper, Faster, and Wrong in the Same Places. arXiv:2609.29769.

Schluntz, E.; and Zhang, B. 2024. Building Efective Agents. Anthropic Engineering blog, 19 December 2024. https://www.anthropic.com/engineering/buildingefective-agents. Accessed 9 October 2026.

Stanislaw, H.; and Todorov, N. 1999. Calculation of Signal Detection Theory Measures. Behavior Research Methods, Instruments, & Computers, 31(1): 137–149.

Sunkavalli, V. K. 2026. LLM Judges as Raters: A Pre-Registered Audit of Severity, Halo, Reliability, and Version Instability in LLM Essay Scoring on Public Corpora. arXiv:2608.29517.

Teo, R. S. Y.; Yamada, Y.; Kotyan, S.; Imajuku, Y.; and Clanuwat, T. 2026. Beyond Imitation: A Framework and Benchmark for LLM-Assisted Peer Review. Transactions on Machine Learning Research. ArXiv:2610.11087.

Tran, D.; and Kiela, D. 2026. Single-Agent LLMs Outperform Multi-Agent Systems on Multi-Hop Reasoning Under Equal Thinking Token Budgets. arXiv:2604.02460.

Wang, X.; Wei, J.; Schuurmans, D.; Le, Q.; Chi, E. H.; Narang, S.; Chowdhery, A.; and Zhou, D. 2023. Self-Consistency Improves Chain of Thought Reasoning in Language Models. In International Conference on Learning Representations. ArXiv:2203.11171.

Xiong, T.; Ge, Y.; Li, M.; Zhang, Z.; Kulkarni, P.; Wang, K.; He, Q.; Zhu, Z.; Liu, C.; Chen, R.; Zheng, T.; Chen, Y.; Wang, X.; Zhang, R.; Chen, W.; and Huang, H. 2025. Multi-Crit:

Benchmarking Multimodal Judges on Pluralistic Criteria-Following. arXiv:2511.21662. To appear in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition 2026.

Yeh, I.-C. 2009. Default of Credit Card Clients [Dataset]. UCI Machine Learning Repository.

Yeh, I.-C.; and Lien, C.-h. 2009. The Comparisons of Data Mining Techniques for the Predictive Accuracy of Probability of Default of Credit Card Clients. Expert Systems with Applications, 36(2): 2473–2480.

Yu, W.; Zhou, L.; Dong, X.; Barath, S. S.; Staunton, D.; Yoon, B.-J.; Qian, X.; Caverlee, J.; and Ji, S. 2026. When Do Multi-Agent Systems Help? An Information Bottleneck Perspective. arXiv:2607.16133.

Zhang, H.; Cui, Z.; Chen, J.; Wang, X.; Zhang, Q.; Wang, Z.; Wu, D.; and Hu, S. 2025. Stop Overvaluing Multi-Agent Debate—We Must Rethink Evaluation and Embrace Model Heterogeneity. arXiv:2502.08788.

## A Memos, labels and settings

Memos. The record supplies the credit limit, age, education, marital status and six months of repayment status, bills and payments from April to September 2005. The dataset has no income or employment fields. The program draws monthly income, allowances, tenure and sector experience for each record from fixed ranges tied to the credit limit and age, and writes them into sentences from templates. Every figure a requirement tests is stated only by sentences the program controls. A defect changes one sentence, and no other requirement governs that sentence. An agent may still judge the sentence against a requirement of its own. The label of each requirement is drawn independently on each memo.

Labels. In a first pass the third source, Claude Opus 5.5, difered from the other two on five cells. In four it found a figure wrong and doubted whether the sentence fell under the requirement or what the figure referred to, and in the fifth it found no error. The wording of two requirements was made explicit, the memos were regenerated with one sentence changed, and the two afected requirements were read again on all 300 memos, with agreement on every cell. No memo was dropped. Opus 5.5 belongs to the family of two of the tested models and is not one of them.

Models and settings. The Qwen3.5 models ran under Ollama 0.35.1 with 4-bit quantisation, Q4\_K\_M, a context of 16,384 tokens and a fixed seed, and the three draws used three seeds. The Claude models ran through the Batch API with efort set to high. Runs took place between 8 and 10 October 2026. The scoped team was run one day after the other arms. The first requirement in the single agent’s list is rotated by memo index across the three pairs, and each draw advances the rotation by one.

Scoring rules. A missing verdict counts as a miss on a defect and as a flag on a clean cell. A defect counts as caught when the verdict is FAIL, whatever sentence the agent quotes. Intervals are percentile bootstrap intervals from 5,000 resamples of memos, with arms paired by memo.

The decide-once instruction. The sentences below are added to the prompt of every arm.

Take each requirement once. In at most   
three sentences say what you compared   
and what it shows, then give the   
verdict. Do not go back over a   
requirement once you have decided it.

The scoped prompt. A scoped call lists the requirements under two headings and ends the second list with a note. Text in angle brackets stands for the wording of the requirements.

REQUIREMENTS YOU JUDGE   
<the two requirements of this call>   
REQUIREMENTS JUDGED BY OTHER REVIEWERS   
<the other four requirements>   
Do not give a verdict on the   
requirements in this second list. Other   
reviewers judge them. They are listed   
so that you know what they cover.

## B Predictions and outcomes

The seven predictions follow, in three groups, each with its outcome.

## Primary, tested at the 98.75 percent level.

Prediction. The team catches more masked defects than the single agent.

Outcome. Holds on Qwen3.5 27B and Haiku 5.5. Not shown on Qwen3.5 9B. No room on Sonnet 5.5.

Prediction. The team raises more false alarms than the single agent.   
Outcome. Holds on Qwen3.5 9B, Qwen3.5 27B and Sonnet 5.5. Not shown at the corrected level on Haiku 5.5.

## Secondary, tested at the 95 percent level.

Prediction. The team and the single agent do not difer on bare defects.

Outcome. Wrong. The team catches more on three models, by 0.132 [0.062, 0.202], 0.029 [0.009, 0.055] and 0.054 [0.025, 0.086].

Prediction. The team catches more masked defects than a majority of three draws.

Outcome. Holds on Haiku 5.5, 0.087 [0.041, 0.133]. Confounded with temperature on Qwen3.5 9B, 0.240 [0.153, 0.328]. Not shown on Qwen3.5 27B, 0.013 [−0.048, 0.076], or Sonnet 5.5.

Prediction. The team’s excess of false alarms is smaller on the larger model of a family.

Outcome. Holds for Qwen3.5 as a diference, 0.064 [0.030, 0.099]. As a proportion of each model’s rate the excess is about the same. Not shown for Claude, where the diference is −0.002 [−0.015, 0.011].

Pre-specified, added arm, tested at the 98.75 percent level.   
The level for eight tests per family is 99.375 percent.

Prediction. The scoped team raises fewer false alarms than the team.

Outcome. Holds on Sonnet 5.5, also at the eight-test level. At 95 percent only on Haiku 5.5. Not shown on the Qwen3.5 models.

Prediction. The scoped team catches more masked defects than the single agent.

Outcome. Holds on Haiku 5.5, also at the eight-test level. Lower bound 0.000 on Qwen3.5 27B. Not shown on Qwen3.5 9B. No room on Sonnet 5.5.

## C The decide-once instruction

Table A3 gives the change in the single agent’s rates when the instruction is removed. Without the instruction the Qwen3.5

<table><tr><td>Model</td><td>∆Hmasked</td><td>∆F</td><td>∆B</td></tr><tr><td>Qwen3.5 9B</td><td>+0.083</td><td>+0.093</td><td>+0.014</td></tr><tr><td>Qwen3.5 27B</td><td>[-0.028, 0.198] +0.083</td><td>[0.056, 0.128] +0.029</td><td>[-0.013, 0.039] +0.007</td></tr><tr><td>Haiku 5.5</td><td>[0.004, 0.164] +0.100</td><td>[0.006, 0.051] +0.011</td><td>[-0.012, 0.025] +0.020</td></tr><tr><td>Sonnet 5.5</td><td>[0.028, 0.175] -0.004 [-0.019, 0.000]</td><td>[-0.002, 0.026] +0.013 [0.006, 0.022]</td><td>[0.005, 0.036] -0.008 [-0.013, -0.003]</td></tr></table>

Table A1: The contrasts of Figure 2, computed from unrounded rates. $\Delta H ^ { \mathrm { m a s k e d } }$ and ∆F are the two primary contrasts, with intervals at the corrected level. ∆B is exploratory, with 95 percent intervals.

<table><tr><td>Model</td><td> $H _ { \mathrm { s c o p e d } } ^ { \mathrm { m a s k e d } } \mathrm { ~ - ~ }$   $H _ { \mathrm { s i n g l e } } ^ { \mathrm { m a s k e d } }$ </td><td> $F _ { \mathrm { s c o p e d } } -$  Fteam</td><td> $H _ { \mathrm { s c o p e d } } -$  Hteam</td></tr><tr><td>Qwen3.5 9B</td><td>+0.087</td><td>+0.019</td><td>+0.020</td></tr><tr><td>Qwen3.5 27B</td><td>[-0.010,0.196] +0.074</td><td>[-0.018, 0.055] -0.013</td><td>[-0.020, 0.058] -0.020</td></tr><tr><td>Haiku 5.5</td><td>[0.000, 0.144] +0.092</td><td>[-0.034, 0.007] -0.011</td><td>[-0.045, 0.006] -0.018</td></tr><tr><td>Sonnet 5.5</td><td>[0.014, 0.170] at ceiling</td><td>[−0.024, 0.002] -0.011 [-0.019, -0.003]</td><td>[-0.042, 0.006] +0.003 [-0.005, 0.012]</td></tr></table>

Table A2: The contrasts of Figure 3 for the scoped team, computed from unrounded rates. The first two are its prespecified contrasts, with intervals at the corrected level. The last is exploratory, with 95 percent intervals.

27B team lost 17 of300 calls to the output cap, its single agent 2 of 100 and its scoped team 7 of 300. The Qwen3.5 9B team lost 9 of 300. No Claude call was lost. On completed verdicts alone and without the instruction, ∆H is +0.008 and $\Delta F$ is +0.079 on Qwen3.5 27B, and they are +0.094 and +0.095 on Qwen3.5 9B.

Table A4 bears on whether part of the team’s gain under the instruction is a recovery of detection that the instruction takes from the single agent. It sets the single agent without the instruction against the team with it. No diference in defects caught is significant, and the single agent without the instruction raises fewer false alarms than the team with it on Qwen3.5 9B and more on Qwen3.5 27B. One hundred memos give wide intervals.

In the preliminary run that led to the instruction, Qwen3.5 27B kept revising its reasoning until it reached the output cap. The alternative to the instruction was to score completed verdicts only, which would have set aside the rule, fixed in advance, that a missing verdict counts against the arm.

## D Signal detection analysis

From the pooled rates we compute the two measures of equalvariance signal detection theory (Stanislaw and Todorov 1999), whose assumption of equal variance a single operating point cannot test. With z the inverse of the standard normal distribution function, discrimination is $d ^ { \prime } = z ( H ) - z ( F )$ and the threshold is $c = - ( z ( H ) + z ( F ) ) / 2 \mathrm { . }$ A lower c means a reviewer more ready to flag. The measures are not computed for Sonnet 5.5, whose single agent flags no clean cell. Table A5 gives the changes. Against the single agent, the team’s $d ^ { \prime }$ does not change significantly on any of the three models and its c falls on all three. The pooled rates are therefore consistent with a lower threshold and unchanged discrimination, and they do not establish that account. The intervals on $d ^ { \prime }$ are wide. On Haiku 5.5 the upper bound corresponds to roughly the whole of the team’s gain in defects caught, so a real improvement of that size is not excluded. No change in $d ^ { \prime }$ is significant within any defect class either, and the estimates rise in some classes and fall in others. On Qwen3.5 27B and Haiku 5.5 the scoped team’s threshold rises back toward the single agent’s.

<table><tr><td>Model</td><td>Hmasked</td><td> $H ^ { \mathrm { a r i t h } }$ </td><td> $F$ </td></tr><tr><td>Qwen3.5 9B</td><td>+0.179</td><td>+0.163</td><td>+0.046</td></tr><tr><td>Qwen3.5 27B</td><td>[0.105, 0.256] +0.058</td><td>[0.093, 0.230] +0.088</td><td>[0.012, 0.082] +0.020</td></tr><tr><td>Haiku 5.5</td><td>[0.004, 0.112] +0.075</td><td>[0.018, 0.159] +0.044</td><td>[0.000, 0.039] +0.002</td></tr><tr><td>Sonnet 5.5</td><td>[0.027, 0.123] 0.000</td><td>[0.009, 0.079] +0.003</td><td>[-0.005,0.009] 0.000</td></tr><tr><td>Qwen3.5 9B,</td><td>-0.037</td><td>[-0.010, 0.017] +0.188</td><td>+0.005</td></tr><tr><td>single call @Temp=0</td><td>[-0.162, 0.086]</td><td>[0.065, 0.313]</td><td>[-0.039, 0.051]</td></tr><tr><td>Qwen3.5 27B,</td><td>+0.062</td><td>+0.175</td><td>+0.094</td></tr><tr><td>single call @Temp=0</td><td>[-0.045,0.169]</td><td>[0.076, 0.284]</td><td>[0.052, 0.140]</td></tr></table>

Table A3: Change in the single agent’s rates when the instruction is removed, the rate without it minus the rate with it, on the 100 paired memos. For the Claude models the single call and the three draws are pooled, and for the Qwen3.5 models the three draws at temperature 0.7. The last two rows give the single call of the Qwen3.5 models at temperature 0. 95 percent intervals.
<table><tr><td>Model</td><td>All defects caught</td><td>False alarms</td></tr><tr><td>Qwen3.5 9B</td><td>-0.053</td><td>-0.104</td></tr><tr><td>Qwen3.5 27B</td><td>[-0.133,0.024] +0.027</td><td>[-0.158, -0.051] +0.059</td></tr><tr><td>Haiku 5.5 [-0.084, 0.004]</td><td>[-0.026, 0.079] -0.040</td><td>[0.008, 0.113] -0.005 [-0.024, 0.013]</td></tr></table>

Table A4: The single agent without the instruction minus the team with it, one call of each arm on the 100 paired memos, for the three models below ceiling, with 95 percent intervals.

## E Further tables

Tables A6 to A10 give the outcomes for whole memos, the rates by requirement, the repeated calls of the single agent’s prompt, the misfiled flags and the share of flags that would be true.

<table><tr><td></td><td>Qwen3.5 9B</td><td>Qwen3.5 27B</td><td>Haiku 5.5</td></tr><tr><td> $d _ { \mathrm { t e a m } } ^ { \prime } - d _ { \mathrm { s i n g l e } } ^ { \prime } ,$ </td><td>+0.01</td><td>-0.09</td><td>-0.03</td></tr><tr><td>all defects</td><td>[-0.15, 0.17]</td><td>[-0.29, 0.12]</td><td>[-0.33, 0.23]</td></tr><tr><td> $d _ { \mathrm { t e a m } } ^ { \prime } - d _ { \mathrm { s i n g l e } } ^ { \prime } ,$ </td><td>+0.13</td><td>not computed</td><td>not computed</td></tr><tr><td>bare</td><td>[−0.13, 0.40]</td><td></td><td></td></tr><tr><td> $d _ { \mathrm { t e a m } } ^ { \prime } - d _ { \mathrm { s i n g l e } } ^ { \prime } ,$ </td><td>-0.11</td><td>+0.04</td><td>+0.17</td></tr><tr><td>masked</td><td>[-0.37,0.14]</td><td>[-0.23,0.30]</td><td>[-0.19, 0.51]</td></tr><tr><td> $d _ { \mathrm { t e a m } } ^ { \prime } - d _ { \mathrm { s i n g l e } } ^ { \prime } ,$ </td><td>+0.07</td><td>-0.21</td><td>-0.25</td></tr><tr><td>arithmetic</td><td>[-0.16, 0.31]</td><td>[-0.44, 0.04]</td><td>[-0.57, 0.05]</td></tr><tr><td> $d _ { \mathrm { s c o p e d } } ^ { \prime } - d _ { \mathrm { s i n g l e } } ^ { \prime } ,$ </td><td>+0.01</td><td>-0.06</td><td>+0.14</td></tr><tr><td>all defects</td><td>[-0.14, 0.16]</td><td>[−0.26, 0.14]</td><td>[-0.20, 0.49]</td></tr><tr><td> $d _ { \mathrm { s c o p e d } } ^ { \prime } - d _ { \mathrm { t e a m } } ^ { \prime } ,$ </td><td>0.00</td><td>+0.02</td><td>+0.17</td></tr><tr><td>all defects</td><td>[-0.14, 0.15]</td><td>[-0.14,0.19]</td><td>[-0.08, 0.45]</td></tr><tr><td> $c _ { \mathrm { t e a m } } - c _ { \mathrm { s i n g l e } }$ </td><td>-0.33</td><td>-0.20</td><td>-0.24</td></tr><tr><td rowspan="2">Cscoped — Cteam</td><td>[-0.42, -0.24]</td><td>[-0.30, -0.11]</td><td>[-0.39, -0.10]</td></tr><tr><td>-0.06</td><td>+0.09</td><td>+0.17</td></tr><tr><td></td><td>[-0.13, 0.02]</td><td>[0.01, 0.16]</td><td>[0.05, 0.31]</td></tr></table>

Table A5: Changes in $d ^ { \prime }$ and in c under the equal-variance model, with 95 percent intervals, on the three models below ceiling. By class, a common false-alarm rate is used. A class caught in full by either arm is not computed.

<table><tr><td>Model</td><td>Arm</td><td>Clean memos flagged, of 21</td><td>Defective memos fully caught, of 279</td></tr><tr><td rowspan="4">Qwen3.5 9B</td><td>Single agent</td><td>10</td><td>94</td></tr><tr><td>Team</td><td>13</td><td>141</td></tr><tr><td>Scoped team</td><td>18</td><td>151</td></tr><tr><td>Thrèe draws, any</td><td>16</td><td>196</td></tr><tr><td rowspan="4">Qwen3.5 27B</td><td>Single agent</td><td>3</td><td>157</td></tr><tr><td>Team</td><td>5</td><td>186</td></tr><tr><td>Scoped team</td><td>6</td><td>172</td></tr><tr><td>Three draws, any</td><td>16</td><td>251</td></tr><tr><td rowspan="4">Haiku 5.5</td><td>Single agent</td><td>0</td><td>174</td></tr><tr><td>Team</td><td>1</td><td>205</td></tr><tr><td>Scoped team</td><td>1</td><td>197</td></tr><tr><td>Three draws, any</td><td>3</td><td>234</td></tr><tr><td rowspan="4">Sonnet 5.5</td><td>Single agent</td><td>0</td><td>275</td></tr><tr><td>Team</td><td>0</td><td>273</td></tr><tr><td>Scoped team</td><td>0</td><td>275</td></tr><tr><td>Three draws, any</td><td>1</td><td>279</td></tr></table>

Table A6: Whole memos, decide-once instruction. Of the 21 memos with no defect, the number with at least one flag. Of the 279 memos with a defect, the number with every defect caught. The majority rule is omitted. Twenty-one memos with no defect are too few to estimate a rate for such files.

<table><tr><td>Model</td><td>Requirement H</td><td></td><td>F</td></tr><tr><td rowspan="6">Qwen3.5 9B</td><td>R1</td><td></td><td>0.68, 0.84, 0.89 0.143, 0.251, 0.325</td></tr><tr><td>R2</td><td></td><td>0.80, 0.67, 0.70 0.012, 0.106, 0.065</td></tr><tr><td>R3</td><td></td><td>0.38, 0.65, 0.65 0.082, 0.175, 0.219</td></tr><tr><td>R4</td><td></td><td>0.77, 0.80, 0.81–0.561, 0.529, 0.561</td></tr><tr><td>R5</td><td></td><td>0.67, 0.96, 0.90 0.128, 0.346, 0.336</td></tr><tr><td>R6</td><td></td><td>0.38, 0.58, 0.64 0.022, 0.082, 0.093</td></tr><tr><td rowspan="5">Qwen3.5 27B</td><td>R1</td><td></td><td>0.58, 0.80, 0.680.025, 0.094, 0.034</td></tr><tr><td>R2</td><td></td><td>0.93, 0.87, 0.97 0.006, 0.012, 0.000</td></tr><tr><td>R3 R4</td><td></td><td>0.73, 0.74, 0.72 0.022, 0.093, 0.066</td></tr><tr><td>R5</td><td></td><td>0.65, 0.66, 0.600.176, 0.214, 0.225</td></tr><tr><td>R6</td><td></td><td>0.90, 0.97, 0.94 0.024, 0.028, 0.038</td></tr><tr><td rowspan="5">Haiku 5.5</td><td>R1</td><td></td><td>0.93, 0.97, 0.960.027, 0.011, 0.011</td></tr><tr><td>R2</td><td></td><td>0.66, 0.75, 0.73 0.000, 0.005, 0.000 0.99, 1.00, 1.000.018, 0.018, 0.012</td></tr><tr><td>R3</td><td></td><td></td></tr><tr><td>R4</td><td></td><td>0.80, 0.87, 0.78 0.033, 0.049, 0.011</td></tr><tr><td>R5</td><td></td><td>0.70, 0.63, 0.65 0.027, 0.075, 0.059</td></tr><tr><td rowspan="6">Sonnet 5.5</td><td></td><td></td><td>0.80, 0.99, 0.97 0.005, 0.005, 0.000</td></tr><tr><td>R6</td><td></td><td>0.94, 1.00, 1.00 0.000, 0.000, 0.000</td></tr><tr><td>R1</td><td></td><td>1.00, 0.99, 1.00 0.000, 0.049, 0.010</td></tr><tr><td>R2 R3</td><td></td><td>1.00, 1.00, 1.00 0.000, 0.006, 0.000</td></tr><tr><td>R4</td><td></td><td>1.00, 1.00, 0.99 0.000, 0.011, 0.005</td></tr><tr><td>R5</td><td></td><td>0.96, 0.96, 0.97 0.000, 0.011, 0.000 1.00, 1.00, 1.00 0.000, 0.000, 0.000</td></tr><tr><td>R6</td><td></td><td>1.00, 1.00, 1.00 0.000, 0.000, 0.000</td><td></td></tr></table>

Table A7: Share of defects caught, H, and false alarms, $F ,$ by requirement, decide-once instruction, 300 memos. Each entry gives single agent, team and scoped team in that order. The numbers of defective and clean cells are 97 and 203 for R1, 130 and 170 for R2, 117 and 183 for R3, 113 and 187 for R4, 89 and 211 for R5, and 118 and 182 for R6. The Qwen3.5 9B model flags requirement R4 on more than half of its clean cells in every arm. It treats a multiple quoted to one decimal place as wrong because it is rounded.
<table><tr><td></td><td>Qwen3.5 9B</td><td>Qwen3.5 27B</td><td>Haiku 5.5</td><td>Sonnet 5.5</td></tr><tr><td>Hmasked</td><td>0.424</td><td>0.821</td><td>0.790</td><td>1.000</td></tr><tr><td rowspan="4">each repeat</td><td>0.445</td><td>0.764</td><td>0.803</td><td>1.000</td></tr><tr><td>0.432</td><td>0.738</td><td>0.769</td><td>1.000</td></tr><tr><td>0.559</td><td>0.734</td><td>0.786</td><td>1.000</td></tr><tr><td>@Temp=0</td><td>@Temp=0</td><td></td><td></td></tr><tr><td>H, each</td><td>0.586</td><td>0.854</td><td>0.825</td><td>0.994</td></tr><tr><td rowspan="4">repeat</td><td>0.592</td><td>0.800</td><td>0.830</td><td>0.995</td></tr><tr><td>0.556</td><td>0.792</td><td>0.798</td><td>0.994</td></tr><tr><td>0.613</td><td>0.792</td><td>0.830</td><td>0.998</td></tr><tr><td>@Temp=0</td><td>@Temp=0</td><td></td><td></td></tr><tr><td rowspan="6">F, each repeat</td><td>0.191</td><td>0.070</td><td>0.013</td><td>0.000</td></tr><tr><td>0.185</td><td>0.078</td><td>0.011</td><td>0.000</td></tr><tr><td>0.172</td><td>0.085</td><td>0.013</td><td>0.000</td></tr><tr><td>0.160</td><td>0.047</td><td>0.008</td><td>0.001</td></tr><tr><td>@Temp=0</td><td>@Temp=0</td><td></td><td></td></tr><tr><td>0.642</td><td>0.817</td><td>0.891</td><td>0.996</td></tr><tr><td>Team: Hmasked 1, H,</td><td>0.733</td><td>0.834</td><td>0.877</td><td>0.991</td></tr><tr><td>F</td><td>0.254</td><td>0.076</td><td>0.025</td><td>0.013</td></tr></table>

Table A8: Repeated calls of the single agent’s prompt, decideonce instruction, 300 memos. For the Claude models the four repeats are the single call and the three draws, in that order. For the Qwen3.5 models the three draws at temperature 0.7 are shown, with the single call at temperature 0 beside them, marked @Temp=0. On Haiku 5.5 the team’s masked rate and its false-alarm rate both lie outside the range of the four repeats. No repeat of the team or the scoped team exists.

<table><tr><td>Model</td><td>Single agent, flags and misfiled</td><td>Team</td><td>Scoped team</td><td>∆F, misfiled set aside</td></tr><tr><td>Qwen3.5 9B</td><td>182 and 16</td><td>288 and 28</td><td>310 and 25</td><td>+0.083 [0.053, 0.112]</td></tr><tr><td>Qwen3.5 27B</td><td>53 and 5</td><td>86 and 26</td><td>71 and 14</td><td>+0.011 [-0.006, 0.027]</td></tr><tr><td>Haiku 5.5</td><td>15 and 1</td><td>28 and 17</td><td>15 and 2</td><td>-0.003 [-0.011,0.005]</td></tr><tr><td>Sonnet 5.5</td><td>0 and 0</td><td>15 and 8</td><td>3 and 2</td><td>+0.006 [0.002, 0.011]</td></tr></table>

Table A9: Flags on clean cells and, of those, misfiled flags, whose quoted sentence is the defective sentence of a defect under another requirement in the same memo. Decide-once instruction, 1,136 clean cells. The last column is $\Delta F$ with misfiled flags set aside, with a 95 percent interval. Sentences are matched by text after removing case and punctuation. A looser definition that also counts the correct statement of a fact whose other statement is wrong gives the same counts on Qwen3.5 27B and the two Claude models, and 24, 40 and 43 on Qwen3.5 9B.
<table><tr><td></td><td>Qwen3.5 9B</td><td>Qwen3.5 27B</td><td>Haiku 5.5</td><td>Sonnet 5.5</td></tr><tr><td rowspan="3">Single agent</td><td>0.69</td><td>0.91</td><td>0.97</td><td>1.00</td></tr><tr><td>0.17</td><td>0.47</td><td>0.77</td><td>1.00</td></tr><tr><td>0.07</td><td>0.26</td><td>0.56</td><td>1.00</td></tr><tr><td rowspan="3">Team</td><td>0.63</td><td>0.87</td><td>0.95</td><td>0.98</td></tr><tr><td>0.13</td><td>0.37</td><td>0.65</td><td>0.80</td></tr><tr><td>0.06</td><td>0.18</td><td>0.42</td><td>0.60</td></tr><tr><td rowspan="3">Scoped team</td><td>0.62</td><td>0.88</td><td>0.97</td><td>1.00</td></tr><tr><td>0.13</td><td>0.41</td><td>0.77</td><td>0.95</td></tr><tr><td>0.05</td><td>0.21</td><td>0.57</td><td>0.88</td></tr><tr><td rowspan="3">Three draws, majority</td><td>0.71</td><td>0.91</td><td>0.99</td><td>1.00</td></tr><tr><td>0.18</td><td>0.48</td><td>0.94</td><td>1.00</td></tr><tr><td>0.08</td><td>0.27</td><td>0.87</td><td>1.00</td></tr><tr><td rowspan="3">Three draws, any</td><td>0.59</td><td>0.76</td><td>0.95</td><td>1.00</td></tr><tr><td>0.11</td><td>0.22</td><td>0.63</td><td>0.98</td></tr><tr><td>0.05</td><td>0.10</td><td>0.39</td><td>0.96</td></tr></table>

Table A10: Share of flags that would be true at three values of the defect share $p ,$ from the unrounded pooled rates behind Table 2 and assuming they carry over. Each entry gives p of 0.37, 0.05 and 0.02, from top to bottom.