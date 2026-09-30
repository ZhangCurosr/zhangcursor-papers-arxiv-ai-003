# UserProxyBench: Evaluating LLM User Simulators for Agent Benchmarks and Training

Ashish Jain ashish@sarvam.ai

Armaan Sandhu apsandhu@umass.edu

## Abstract

Interactive agent benchmarks and multi-turn reinforcement learning increasingly place a second language model in the role of the user. This simulated user controls what information the agent receives and when, yet current benchmarks score only the agent and do not directly measure whether the user correctly executed its assigned role. We introduce UserProxyBench, an evaluation layer over the τ- bench family, and the User Fidelity Score (UFS), which measures adherence to the benchmark’s private user instructions using task-grounded rubric criteria scored independently of agent success. Holding the agent fixed at GPT-5.5 and varying only the user proxy across 375 enterprise tasks changes mean task reward by 15.2 points, while 24.4% of successful episodes contain a user-specification violation. The dominant failure is premature disclosure: users provide information before it is requested. This behavior has little effect on task reward, yet among successful episodes it causes the agent to make 1.06 fewer tool calls on average, changing the interaction being evaluated while preserving the reward. Finally, across seven proxies we identify an empirical cost–fidelity frontier, enabling practitioners to select the least expensive simulator that satisfies a required fidelity level.

## 1 Introduction

Interactive evaluation has made the “user” part of the benchmark executable. In τ-bench, an LLM user converses with an agent that follows enterprise policy and uses tools (Yao et al., 2025); τ<sup>2</sup>-bench extends this to dual-control settings where the user can also act on a shared environment (Barres et al., 2025). The same pattern is entering training: MUA-RL places an LLM-simulated user inside the reinforcement-learning loop (Zhao et al., 2025), while broader agent-RL systems optimize over long interactive trajectories (Wang et al., 2025; Luo et al., 2025).

This creates an experimental dependency that is easy to overlook. The user proxy controls information timing, observations, and sometimes user-side actions. If it leaks information, fabricates state, or stops early, the agent is no longer solving the intended interaction even when task reward is unchanged. For example, one Telecom proxy opens with “I’m John Smith, by the way. My number is 555-123-2002.” before the agent asks for either field, removing the information-gathering step the task was designed to exercise.

Recent work shows that LLM users differ from real humans and that simulator choice can change measured assistant performance (Dou et al., 2025; Seshadri et al., 2026; Zhou et al., 2026). Purposebuilt user models improve human-likeness over assistant LMs prompted to play users (Naous et al., 2026; Wu et al., 2026). We ask a complementary question: given the benchmark’s own stated user specification, did the proxy execute that role correctly? We call this functional fidelity. It measures contract adherence, not human realism.

We make three contributions. First, we introduce UserProxyBench, an evaluation layer over τ tasks that scores the user independently of the agent. Second, we show that task reward does not subsume user fidelity: nearly one quarter of successful trajectories contain a user-contract violation, and the most common failure is almost invisible to agent reward. Third, we treat user-proxy choice as an explicit operational trade-off, using UFS as a reliability constraint under which user-side inference cost can be minimized for a given evaluation or training pipeline.

## 2 UserProxyBench

User Fidelity Score. Each task provides a private user blueprint (persona, known and unknown information, goal, and task-specific instructions), standing simulator guidelines, and, in some domains, user tools. Let $C _ { i } = \{ c _ { i 1 } , \ldots , c _ { i m } \}$ be the applicable criteria for episode i. We define

$$
f _ { i } = \prod _ { j = 1 } ^ { m } { \bf 1 } \{ c _ { i j } \mathrm { p a s s e s } \} , \qquad \mathrm { U F S } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } f _ { i } .\tag{1}
$$

UFS is therefore the fraction of episodes in which the proxy satisfies every applicable criterion in its own contract. A separate judge-free user-action probe is reported for Telecom, where gold user write actions exist, but is not folded into UFS because the other domains do not expose the same signal.

Task-grounded rubrics. Claude Opus 4.8 writes 4–8 atomic criteria per task from the private blueprint, the exact simulator guidelines, the user tool surface, and one sample interaction. The sample is used only to identify situations that can arise; the blueprint and guidelines remain the authority for correctness. Each criterion must cite a verbatim grounding span, be falsifiable on a trajectory, and concern only the user. An independent verifier (GPT-5.6-Sol, distinct from the generator) reviews each task-level rubric set for grounding, achievability, redundancy, and scope before scoring. Claude Opus 4.8 then grades candidate trajectories without seeing proxy identity; generator and grader are the same model, a limitation we return to in Section 4. The final set contains 2,141 criteria: 634 Telecom, 296 Airline, 656 Retail, and 555 Banking (5.7 per task), spanning four families: groundedness, premature disclosure, goal deviation, and missed information. This follows the broader use of instance-specific rubrics for open-ended evaluation and reward construction (Arora et al., 2025; Gunjal et al., 2026), but targets the user-side environment component rather than the assistant response.

The dominant family derives from an explicit instruction shown to every proxy: “Disclose information progressively. Waitfor the agent to askfor specific information before providing $i t . ^ { \dprime }$ Similar explicit rules prohibit inventing unavailable information, require grounding tool results, and require continuing until the task goal is satisfied. We therefore score stated contract compliance rather than an implicit style preference.

Setup. We evaluate seven reported proxies on the full base split of four enterprise domains: Telecom (114 tasks), Airline (50), Retail (114), and Banking Knowledge (97), for 375 tasks total. GPT-5.5 is frozen as the agent; only the user proxy changes. The reported set contains three open-weight and four hosted models. Each task has one rollout per proxy. We denote the benchmark’s agent-side task reward by $R _ { \tau }$ to distinguish it from the τ benchmark family. Hosted costs use billed user-side spend; for self-hosted models, measured input/output tokens are converted using public serverless prices accessed in August 2026 (Together AI, 2026).

## 3 Results

## 3.1 Fidelity-constrained proxy selection

Once fidelity is measurable, the operational question is which simulator is the least expensive one that meets the pipeline’s reliability requirement. For a required fidelity $q ,$

$$
U ^ { * } ( q ) = \operatorname * { a r g m i n } _ { U } C ( U ) \quad { \mathrm { s . t . } } \quad { \mathrm { U F S } } ( U ) \geq q ,\tag{2}
$$

where $C ( U )$ is user-side inference cost. Figure 1 shows the frontier. $\mathrm { A t } q = . 8 4$ , Gemma-31B is the lowest-cost qualifying proxy: UFS .845 at about \$0.0175 per simulation. Gemini-3.5-Flash gains 1.9 UFS points but costs $3 . 2 \dot { \times }$ more; raising the requirement to $q = . 8 6$ moves the operating point to Gemini, while at $q = . 7 5$ GPT-4.1-mini is cheaper. Three tested proxies are strictly dominated because another model is both cheaper and more faithful. At 64 rollouts over 1,000 prompts, Gemma-31B’s user-side inference is about \$1.1k per epoch versus \$3.6k for Gemini-3.5-Flash.

![](images/df5b4cf50227bafcf1080baef28033bd477665a0ceeea4ac322c58eba44a78f2.jpg)  
User-side cost per simulation (USD, log scale)

Figure 1: Cost–fidelity frontier. Each point is one user proxy; the agent is frozen at GPT-5.5. The dashed line is the efficient frontier (no proxy is both cheaper and higher-UFS). Ringed points are strictly dominated. Because the axis is log-cost, the least-expensive proxy meeting a required UFS q changes with $q \colon$ Gemma-31B at $q = . 8 4$ , Gemini-3.5-Flash at $q = . 8 6$  
![](images/358126fabf2b136166dd8b2e3b343608b7631c4a86c6efb0a190924690187d3f.jpg)  
Figure 2: Only the user simulator changed. Each point is one of seven user proxies; the grey bar spans the $R _ { \tau }$ range induced for the same frozen GPT-5.5 agent within a domain.

## 3.2 The user proxy changes the benchmark result

Nothing about the evaluated agent changes across proxy conditions, yet mean $R _ { \tau }$ ranges from .644 to .796. Excluding Banking, whose absolute reward is uniformly low, per-domain spread ranges from .132 to .298, reaching .280 on Airline and .298 on Retail. Figure 2 makes the experimental control explicit: every point on a row is the same GPT-5.5 agent on the same domain, with only the simulated user changed.

Across all 2,618 scored trajectories (of 2,625: two runs hit an infrastructure error, and five were unscorable because no criterion applied—Appendix B), 24.4% of agent-successful episodes fail UFS; on Telecom, 39.7% of all trajectories are R pass / UFS fail.

## 3.3 Why task reward cannot screen user proxies

Within each proxy–domain cell, we compare agent pass rate when a failure family occurs versus when it does not (at least five episodes per side). Goal deviation is uncommon (134 failures) and reduces pass rate by .515 on average. Premature disclosure is the dominant failure (480), yet changes τ reacts to some user failures and not to the most common one

![](images/660a5617c378b633aa835efb7050184843f4c79dc322db083172f5e4c70b1101.jpg)  
failures observed (all proxies)

![](images/43fc4e1c92e8a323c9f5aae6607a47944e1d4e2e9743feda7aac421e8b54d2cf.jpg)  
Δ in τ pass rate when this family fails  
Figure 3: $R _ { \tau }$ is nearly blind to the dominant failure. Left: how often each failure family occurs across all proxies. Right: the change in agent pass rate when the family is present versus absent, one dot per proxy–domain cell with at least five episodes on both sides (up to 28 cells). Goal deviation costs the agent −.52; premature disclosure, the most common failure, costs −.04.

pass rate by only −.043 on average; its median effect is positive (+.033) and the sign is inconsistent across cells. Figure 3 shows both the frequency of each family and its effect on agent reward: the most common failure is the one $R _ { \tau }$ is least able to detect.

This is visible even for capable models. On Telecom, Sonnet-4-6 and GPT-4.1-mini give the frozen agent the same $R _ { \tau } = . 9 4 7$ , but their UFS scores are .474 and .737. The gap is not only disclosure: in another successful Telecom episode, Gemma-26B says it is turning off Airplane Mode although no corresponding user tool call occurs and the device remains in Airplane Mode (Appendix C).

Removing disclosure criteria substantially reorders models: Qwen3.6-35B moves from sixth to third, Gemma-26B from fourth to last, and disclosure-only versus non-disclosure UFS are almost uncorrelated $( \rho = . 0 7 1 )$ ). UFS is therefore best read together with its failure profile.

## 3.4 Implications for multi-turn RL

For training, the issue is not only whether the final reward changes; it is whether the trajectory being reinforced changes. Restricting to successful episodes $( R _ { \tau } = 1 )$ , over-disclosing users make the frozen agent issue 1.06 fewer tool calls and ask 0.26 fewer questions, consistently across all seven proxies. The agent receives identical task reward despite doing less information gathering.

Across the same 374 tasks completed by all seven proxies, mean agent turns correlate with UFS at $r = . 9 5 5$ , against $r = . 5 9 4$ for R (Appendix E); proxy choice changes mean turns by 26% and questions by 25%. In multi-turn RL, fidelity determines whether rollouts instantiate the intended interaction, while cost determines how many such rollouts fit the budget. Hence the operating rule in Section 3.1: choose the lowest-cost proxy that meets the required fidelity.

## 4 Discussion

UserProxyBench measures contract fidelity, not human realism; the two are complementary. UFS is LLM-judged by the rubric generator, and our manual audit (ten tasks plus 50 criteria/judgments) is single-rater. A judge-free Telecom probe finds 54 missed gold writes among 658 trajectories; of these, 23 pass every applicable criterion and 20 pass a directly relevant one. Limitations include one rollout per task, time-dependent serving prices, a pinned Banking version, and no measurement of downstream trained-policy effects.

## References

Rahul K. Arora, Jason Wei, Rebecca Soskin Hicks, Preston Bowman, Joaquin Quiñonero-Candela, Foivos Tsimpourlas, Michael Sharman, Meghan Shah, Andrea Vallone, Alex Beutel, Johannes Heidecke, and

Karan Singhal. HealthBench: Evaluating Large Language Models Towards Improved Human Health. arXiv:2505.08775, 2025.

Victor Barres, Honghua Dong, Soham Ray, Xujie Si, and Karthik Narasimhan. τ<sup>2</sup>-Bench: Evaluating Conversational Agents in a Dual-Control Environment. arXiv:2506.07982, 2025. Oral at ICML 2026.

Yao Dou, Michel Galley, Baolin Peng, Chris Kedzie, Weixin Cai, Alan Ritter, Chris Quirk, Wei Xu, and Jianfeng Gao. SimulatorArena: Are User Simulators Reliable Proxies for Multi-Turn Evaluation of AI Assistants? EMNLP, 2025.

Anisha Gunjal, Anthony Wang, Elaine Lau, Vaskar Nath, Yunzhong He, Bing Liu, and Sean Hendryx. Rubrics as Rewards: Reinforcement Learning Beyond Verifiable Domains. ICLR, 2026.

Xufang Luo, Yuge Zhang, Zhiyuan He, Zilong Wang, Siyun Zhao, Dongsheng Li, Luna K. Qiu, and Yuqing Yang. Agent Lightning: Train ANY AI Agents with Reinforcement Learning. arXiv:2508.03680, 2025.

Tarek Naous, Philippe Laban, Wei Xu, and Jennifer Neville. Flipping the Dialogue: Training and Evaluating User Language Models. ICLR, 2026.

Preethi Seshadri, Samuel Cahyawijaya, Ayomide Odumakinde, Sameer Singh, and Seraphina Goldfarb-Tarrant. Lost in Simulation: LLM-Simulated Users are Unreliable Proxies for Human Users in Agentic Evaluations. ACL, 2026.

Together AI. Serverless inference pricing. https://www.together.ai/pricing, accessed August 2026.

Zihan Wang, Kangrui Wang, Qineng Wang, Pingyue Zhang, Linjie Li, Zhengyuan Yang, Kefan Yu, Minh Nhat Nguyen, Licheng Liu, Eli Gottlieb, Monica Lam, Yiping Lu, Kyunghyun Cho, Jiajun Wu, Li Fei-Fei, Lijuan Wang, Yejin Choi, and Manling Li. RAGEN: Understanding Self-Evolution in LLM Agents via Multi-Turn Reinforcement Learning. arXiv:2504.20073, 2025.

Shirley Wu, Evelyn Choi, Arpandeep Khatua, Zhanghan Wang, Joy He-Yueya, Tharindu Cyril Weerasooriya, Wei Wei, Diyi Yang, Jure Leskovec, and James Zou. HumanLM: Simulating Users with State Alignment Beats Response Imitation. arXiv:2603.03303, 2026.

Shunyu Yao, Noah Shinn, Pedram Razavi, and Karthik Narasimhan. τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains. ICLR, 2025.

Weikang Zhao, Xili Wang, Chengdi Ma, Lingbin Kong, Zhaohua Yang, Mingxiang Tuo, Xiaowei Shi, Yitao Zhai, and Xunliang Cai. MUA-RL: Multi-turn User-interacting Agent Reinforcement Learning for Agentic Tool Use. arXiv:2508.18669, 2025.

Xuhui Zhou, Weiwei Sun, Qianou Ma, Yiqing Xie, Jiarui Liu, Weihua Du, Sean Welleck, Yiming Yang, Graham Neubig, Sherry Tongshuang Wu, and Maarten Sap. Mind the Sim2Real Gap in User Simulation for Agentic Tasks. In Conference on Language Modeling (COLM), 2026.

## A Full proxy results

<table><tr><td>Proxy</td><td>Hosting</td><td>Cost/sim</td><td>Agent Rτ</td><td>Mean UFS</td><td>Prompt tok</td><td>Completion tok</td></tr><tr><td>Sonnet-4-6†</td><td>hosted</td><td>.1841</td><td>.784</td><td>.778</td><td>57,888</td><td>665</td></tr><tr><td>Gemini-3.5-Flash</td><td>hosted</td><td>.0558</td><td>.796</td><td>.864</td><td>53,111</td><td>2,026</td></tr><tr><td>Qwen3.6-35B†</td><td>open</td><td>.0249</td><td>.779</td><td>.612</td><td>47,590</td><td>7,524</td></tr><tr><td>Gemma-31B</td><td>open</td><td>.0175</td><td>.737</td><td>.845</td><td>43,696</td><td>461</td></tr><tr><td>Gemma-26B</td><td>open</td><td>.0147</td><td>.720</td><td>.759</td><td>36,411</td><td>562</td></tr><tr><td>GPT-5.4-mini†</td><td>hosted</td><td>.0144</td><td>.644</td><td>.599</td><td>26,050</td><td>1,587</td></tr><tr><td>GPT-4.1-mini</td><td>hosted</td><td>.0063</td><td>.720</td><td>.752</td><td>29,574</td><td>449</td></tr></table>

Table 1: Aggregate proxy results, sorted by user-side cost. Agent R and Mean UFS are unweighted means across the four domains for the frozen GPT-5.5 agent under each proxy. <sup>†</sup>Strictly dominated on the cost–UFS plane.

## B Per-domain $R _ { \tau }$ and UFS

<table><tr><td>Proxy</td><td colspan="2">Telecom</td><td colspan="2">Airline</td><td colspan="2">Retail</td><td colspan="2">Banking</td></tr><tr><td></td><td>Rτ</td><td>UFS</td><td>Rτ</td><td>UFS</td><td>Rτ</td><td>UFS</td><td>Rτ</td><td>UFS</td></tr><tr><td>Sonnet-4-6</td><td>.947</td><td>.474</td><td>.880</td><td>.840</td><td>.851</td><td>.912†</td><td>.458</td><td>.885*</td></tr><tr><td>Gemini-3.5-Flash</td><td>.965</td><td>.711</td><td>.880</td><td>.940</td><td>.886</td><td>.929†</td><td>.454</td><td>.876</td></tr><tr><td>Qwen3.6-35B</td><td>.895</td><td>.254</td><td>.920</td><td>.680</td><td>.860</td><td>.646†</td><td>.443</td><td>.866</td></tr><tr><td>Gemma-31B</td><td>.912</td><td>.781</td><td>.840</td><td>.880</td><td>.763</td><td>.886</td><td>.433</td><td>.835</td></tr><tr><td>Gemma-26B</td><td>.833</td><td>.588</td><td>.880</td><td>.800</td><td>.728</td><td>.868</td><td>.438</td><td>.781*</td></tr><tr><td>GPT-5.4-mini</td><td>.895</td><td>.325</td><td>.640</td><td>.580</td><td>.588</td><td>.717†</td><td>.454</td><td>.773</td></tr><tr><td>GPT-4.1-mini</td><td>.947</td><td>.737</td><td>.800</td><td>.740</td><td>.719</td><td>.779†</td><td>.412</td><td>.753</td></tr></table>

Table 2: Per-domain results. $n = 1 1 4 / 5 0 / 1 1 4 / 9 7$ for Telecom/Airline/Retail/Banking, with two exceptions: Banking $n = 9 6$ in the two cells marked <sup>∗</sup> (infrastructure error on one task), and Retail $n = 1 1 3$ for five proxies, marked <sup>†</sup>, where one task yielded no applicable criteria and UFS is undefined. 2,618 of 2,625 trajectories are scored.

## C Failure taxonomy and examples

The taxonomy defines 19 modes under four families; 18 are exercised by the 2,141 retained criteria. groundedness.self\_contradiction receives zero criteria on this task set. Premature disclosure contains volunteers\_unasked\_info, preempts\_diagnosis, and dumps\_blueprint; groundedness covers unsupported personal/device state, unexecuted-action claims, contradictions, and invented unknown information; goal deviation covers premature stopping, accepting forbidden or insufficient outcomes, and abandoning the goal; missed information covers withholding requested information and skipping requested user actions.

Premature disclosure (Qwen3.6-35B, Telecom, $R _ { \tau } = 1 )$

“I’m John Smith, by the way. My number is 555-123-2002.”

This is the opening user turn, before the agent requests identifying information.

Unexecuted action claim (Gemma-26B, Telecom, $R _ { \tau } = 1 )$

“Okay, I’ll try that. Let me see... I’m turning off this Airplane Mode now. Just a moment... okay, I’ve waited a bit.”

No toggle\_airplane\_mode call occurred, and subsequent device state still showed Airplane Mode on.

Fabricated device state (Gemma-26B, Telecom).

“Let me try to send that picture one more time... Oh! It worked! The little bar finished moving and it says it was sent!”

No tool result established MMS success; the last relevant check had returned failure.

## D Rubric construction and judge-free probe

Claude Opus 4.8 generates criteria from the private user blueprint, simulator guidelines, user tool surface, and one sample conversation; the prompt explicitly states that the sample is not an oracle. Criteria must cite an exact grounding span and are batch-reviewed by GPT-5.6-Sol under nine checks covering grounding, falsifiability, scope, user-attribution, achievability, taxonomy fit, and redundancy. Claude Opus 4.8 then grades candidate trajectories one criterion at a time without the proxy identity.

A judge-free Telecom probe compares actual user tool calls against gold user write assertions on 658 scorable trajectories. Fifty-four trajectories miss a gold write; in 23 of these every applicable semantic criterion still passes, and in 20 a directly relevant criterion is applied and still passes. Adding this mechanical signal would change Telecom UFS by .009–.070 per proxy and leaves the ranking unchanged, so we report it as an independent diagnostic rather than changing the cross-domain UFS definition.

![](images/aa032fd218e2ef1b4ef17b13f7233b00591f6cab4387f9e286346d117808980e.jpg)

![](images/1f6d5480ac6d9fe33330cb009e547293718a72ba4f49f6c7f9094244141f45f9.jpg)  
Figure 4: Mean agent turns over the 374 tasks completed by all seven proxies, against UFS (left) and agent $R _ { \tau }$ (right). Points are directly labeled by proxy.

## E Agent effort versus proxy metrics

Figure 4 compares UFS and agent task reward as predictors of the interaction induced by each proxy. Across the 374 tasks completed by all seven proxies, mean agent turns correlate with UFS at $r = . 9 5 5$ compared with $r = . 5 9 4$ for $R _ { \tau }$

## F Limitations and next validation

Our current manual audit consists of ten tasks read end-to-end plus 50 further final criteria/judgments reviewed by one author. It changed the pipeline twice but is not blinded, stratified, or multi-rater. The most important next validation is therefore a blinded human study of criterion validity and judge decisions, with agreement statistics. Repeated rollouts on Telecom, a second frozen agent, and a small RL experiment comparing policies trained against high- and low-UFS proxies would separately test stochasticity, interaction-partner dependence, and downstream training effects.

## G Experimental configuration and assets

All seven reported user proxies use temperature 1.0. The frozen GPT-5.5 agent uses high reasoning effort. GPT-5.4-mini also uses high reasoning effort; GPT-4.1-mini has no thinking mode; Gemini-3.5-Flash uses its default thinking behavior; Qwen3.6-35B-A3B is served locally with thinking enabled; Gemma-4-31B-it and Gemma-4-26B-A4B-it are served locally with thinking disabled; Sonnet-4-6 was run without extended thinking because the trajectory runner did not forward the provider-specific parameter. Local Gemma/Qwen inference used tensor parallelism over two workers (Gemma-4-26B-A4B-it and Qwen also used expert parallelism).

Rubric criteria were generated by Claude Opus 4.8 (temperature 1.0), independently verified by GPT-5.6-Sol, and graded by Claude Opus 4.8. Generator and judge are therefore the same model. We exclude Opus-4-8 and GPT-5.5 from the reported proxy set because their conversations or outputs participate in rubric construction; GPT-5.5 remains the frozen agent in all runs. The benchmark is built on sierra-research/tau2-bench v1.0.0 at commit 8ebb749 (MIT license), with two local engineering patches for judge routing and context-overflow retrieval. Numerical tables and figures are regenerated deterministically from graded trajectory files.

## H Broader impact

More reliable user proxies can make interactive benchmarks and RL environments easier to audit and cheaper to operate. The main risk is over-interpreting contract fidelity as human realism: a high-UFS simulator may still underrepresent real users, dialects, accessibility needs, or adversarial behavior. UFS should therefore complement, not replace, evaluations with real users and population-sensitive simulation metrics. Cost-frontier results are time-dependent and should not be treated as long-lived commercial recommendations.