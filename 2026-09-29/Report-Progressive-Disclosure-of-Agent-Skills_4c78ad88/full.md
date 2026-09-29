# Report: Progressive Disclosure of Agent Skills

Guilin Zhang, Kai Zhao,<sup>∗</sup> Priyanka Mudgal, Waleed Ammar, Xiquan Cui, Xu Chu, Alet Blanken

Workday AI Research

## Abstract

Users of Workday’s deployed LLM-based agents often request features which can be addressed by defining named procedures, also known as skills, in the LLM context, efectively augmenting agents’ capabilities. However, as an agent’s skills library grows in size, so does the agent’s operational cost. Progressive disclosure (lazy-loading) of skills as needed may reduce operational costs, but its impact on overall latency and skill-retrieval quality remains unclear. In this report, we investigate the impact empirically and find that progressive disclosure improves skill-retrieval quality but marginally degrades overall latency.

## Agent Skills in Production

Workday<sup>1</sup> is a cloud-based enterprise platform for managing people, money and agents, serving over 65% of Fortune 500 enterprises. As of August 2026, over 5,500 of our customers are using one or more AI agents deployed in the Workday platform, a 35% increase compared to the previous quarter.

Skills augment agentic capabilities. As more customers adopt our deployed LLM-based agents, it is necessary to adapt their agentic capabilities in response to feature requests and bug reports. A simple and efective approach for enhancing agentic capabilities is to define specialized instructions detailing how a particular feature request or bug report may be addressed. Modern agent frameworks, such as LangChain, package those instructions as skills. Skills package domain expertise, such as workflows, best practices, scripts, reference docs, and templates, into reusable directories. Each skill has a name, a natural-language description, an input/output signature, and often a worked example or usage notes.

Eager loading of skills does not scale. In order to enable an agent’s underlying large language model (LLM) to faithfully follow the instructions in a skill’s definition, the ‘eager loading’ skill management regime fully discloses the skill in the LLM prompt context. However, as an agent’s skills library grows in size, we observe a notable increase in its operational costs, due to the number of tokens used to disclose the skills library in the LLM context, which prompted us to explore alternative regimes for managing agent skills.

Progressive disclosure ofers a viable solution. Lazy loading of skill definitions as needed, also known as progressive disclosure, is an attractive alternative, commonly adopted in agentic frameworks such as LangChain. When progressive disclosure is enabled, agents incrementally disclose skills in the LLM context, pulling in more details only as a task calls for it.

In this report, we investigate the tradeofs of activating progressive disclosure of skills in the agent harness, measuring how it impacts an agent’s overall token-based operational cost, latency and skill-retrieval quality, as the library size grows in a controlled setup. Next, we describe our implementation for progressive disclosure.

## Agent Skills Management in The Harness

A minimal agent consists of an LLM core and its harness.<sup>2</sup> The harness interfaces with the agent’s environment, determines when to invoke the LLM, manages the skills library, prepares the context (and prompt) before invoking the LLM, and processes LLM outputs.

Curating a skills library. An agent’s skills library consists of N structured skills. According to the open standard format for Agent Skills, each skill is a directory containing, at minimum, a SKILL.md file starting with the skill’s frontmatter (a name field and a description field in YAML), followed by the skill’s body content which contains the operative content in Markdown, e.g., step-by-step instructions, input-output examples and edge cases.<sup>3</sup> Instead of manually creating all skill content from scratch, coding agents such as Claude Code can be used to create and modify specialized skills and it is not uncommon for agent developers to share the skills they define in version-controlled repositories in order for them to be reused in multiple agents and by other developers.<sup>4</sup>

Two regimes for operating a skills library. A naïve regime an agent harness may use is to port all details of all

![](images/9027801600031dedd00e7ccf1ba6c1683002499c84ee15a55b017e4f44dafcbd.jpg)

![](images/20c3a084c6e8b9037d1d44b83d5ac7e55911560a1d867845e16f06e5a3a57e40.jpg)

![](images/ebfc7b0e9d2b529afa8aa012afba4160b4703e89e17f3fefb9cb02005a6e2a5e.jpg)  
Figure 1: (a) Under the eager loading regime, skill-retrieval quality degrades as the skills library size N grows (Qwen3-8B). (b) Under the eager loading regime, the larger core LLM (Qwen3-14B) is reliable against easy/medium-dificulty skill distractors, but susceptible to hard distractors. (c) Under the progressive disclosure regime, the LLM core is invoked more often which increases latency, especially for smaller values of the library size N.

N skills in its library to the LLM context, which we call ‘eager loading’ of skills. As the number of skills increase, eager loading substantially increases the number of tokens needed to disclose skills in the LLM context, which impacts the agent’s quality, reliability and cost. An alternative regime loads skills progressively, disclosing minimal information about available skills in the LLM context early on and adding more details about the most relevant skills in subsequent LLM invocations only when a task calls for it, also known as ‘progressive disclosure’. In this report, we contrast eager loading with a regime that loads the frontmatter of all N skills in the LLM context and prompts the LLM core to emit a structured command which determines which skill is most relevant for the task at hand, e.g., {"action":"load\_skill","name":"pptx"}.

The harness then loads the body content of the chosen skill to the LLM context in subsequent LLM invocations.<sup>5</sup>

## Experimental Setup

We estimate the impact of progressive disclosure by measuring skill-retrieval success rate, total tokens used and overall latency in seconds.

Skill retrieval. The impact of progressive disclosure on an agent’s quality is mediated by its ability to select relevant skills, which we quantify using an explicit skill-retrieval task. We augment the body content of each skill with a unique ‘activation code’ identifier, produced by a random number generator. Each instance of the task names a domain intent (e.g., travel) and asks the agent to return the most relevant activation code in a structured response which takes the form: RESULT[<skill-name>]: <activation-code>.

The agent harness delegates the task to the LLM core, which is only able to return the correct activation code if the relevant skill’s body is loaded in its context. We define a

taxonomy of outcomes:

• success (skill name and activation code are both correct),

• wrong\_name (skill name is incorrect),

• wrong\_code (skill name is correct, but the activation code is wrong), and

• crash (context overflow or malformed output).

We experiment with four library sizes: N ∈ {5, 20, 50, 100}, and develop a core set of 24 task instances based on 5 relevant skills. For N > 5, we pad the skills library with distractor skills with three dificulty tiers that vary the directness of the trigger phrasing, drawn from a generator spanning 12 domain families. We use three diferent seeds for each task, and report means over 72 rollouts for each unique combination of library size N and skill management regime (i.e., eager loading vs. progressive disclosure).

LLM cores. We experiment with three LLM cores: Qwen2.5-7B-Instruct (Qwen Team 2024), Qwen3-8B and Qwen3-14B (Qwen Team 2025), with greedy decoding and a 32k context window.<sup>6</sup> For each LLM core, we run 576 agent rollouts, which correspond to 4 library sizes × 2 skill management regimes × 24 task instances × 3 seeds. We use vLLM (Kwon et al. 2023) on a single NVIDIA L40S (48 GB) to serve all LLM cores. Every core LLM invocation passes through a wrapper that records prompt tokens count, completion tokens count, and wall-clock latency from the server’s usage accounting.<sup>7</sup>

## Key Findings

1. Progressive disclosure reduces operational cost and improves reliability. When eager loading is enabled, skill definitions increasingly dominate the context as N grows, eventually breaking the agent’s reliability due to exceeding the maximum allowed prompt size, indicated by crash in all rollouts with N = 100 in Table 1 (Eager tok) and Figure 2 (c). Table 1 (Tok ↓) shows the percentage of tokens saved when progressive disclosure is enabled, demonstrating savings across all values of N, as intended, as a direct result of only disclosing the full body of the most relevant skill in progressive disclosure. Progressive disclosure demonstrates increasingly bigger token savings as N grows, up to 81.7% of tokens originally used in the eager loading regime. Figure 2 (a) visually demonstrates how token usage grows as a function of library size in both regimes. With the prevalent token-based pricing of LLMs, these savings translate to substantial savings of agents’ operational cost at scale.

![](images/ec558b1aede70ea2024e2eeaf980be96344cf94a460ff1dfe945028ac69153a4.jpg)

![](images/7deb0c759096a34b0523c196344b28b1bb2fd8898807cb8428108c818797fb97.jpg)

![](images/c4ddd99cca72bf1ff12a9813cfcc12492f2e0efe921290b169340cf3f4f4122c.jpg)  
Figure 2: (a) Prompt token usage grows faster in the eager loading regime. (b) Skill-retrieval quality degrades faster in the eager loading regime, as N grows. (c) The rate at which each skill management regime overflows the maximum context size at library size N=100.

2. Progressive disclosure may increase overall latency. As we described earlier, the progressive disclosure regime for skill management entails an additional LLM invocation dedicated to identifying relevant skills, which often increases the agent’s mean overall latency, as illustrated in Figure 1 (c). For example, using progressive disclosure instead of eager loading corresponds to a 15% increase in the agent’s overall latency (from 1.75 to 2.01 seconds) when using Qwen3-8B with a library of size N=50.<sup>8</sup>

3. Progressive disclosure improves skill-retrieval quality, on average. While progressive disclosure is the undisputed winner on average skill retrieval quality, neither regime consistently outperforms the other, as shown in Figure 2 (b). Progressive disclosure either matches or outperforms eager loading on average skill-retrieval quality in all but one experimental setup, as shown in Table 1 (Skill retrieval). As the library size N grows, distractor skills dominate the LLM context, and the skill-retrieval quality for the eager loading regime falls sharply (e.g., from 1.00 at N=5 to 0.125 at N=50 for Qwen3-14B), exposing retrieval of the wrong skill name as the dominant failure as shown in Figure 1 (a).<sup>9</sup> We note that the largest LLM core (Qwen3-14B) is resilient against easy/medium distractors but remains susceptible to hard distractors, as shown in Figure 1 (b). This finding emphasizes the importance of developing skill management regimes which do not only optimize for token usage but also skill retrieval quality.

## Conclusion

Skills have emerged as an efective way of augmenting agentic capabilities in response to increasing adoption of production agents. Progressive disclosure declares the frontmatter (a skill’s name and brief description) of all skills available to an agent in the initial LLM context then incrementally loads the full definition of relevant skills as needed, instead of naïvely loading the full definitions of all skills. In this report, we found that progressive disclosure of agent skills reduces token usage and improve reliability as well as skill-retrieval quality, at the expense of greater latency.

## Open Questions

How many skills do we need? The experimental setup discussed earlier limits skill retrieval to one skill for each task. In practice, the agent has no prior knowledge about the number of skills needed for a given task. How do we balance the verbosity of skill definitions with the complexity of the task at hand remains an open research question in agent skill management with critical consequences.

Which skill files do we need? The experimental setup discussed earlier limits the definition of each skill in the library to its SKILL.md file. In practice, a skill’s directory may also contain additional files (e.g., scripts, references or assets) which would take up even more room in the LLM context. What strategies do we use to determine which files to load? How do we evaluate the eficacy of diferent strategies in practice?

<table><tr><td rowspan="2">LLM core</td><td rowspan="2">N</td><td colspan="3">Token usage</td><td colspan="2">Skill retrieval</td><td colspan="2">Reliability</td></tr><tr><td>EL tok</td><td>PD tok</td><td>Tok↓</td><td>EL succ</td><td>PD succ</td><td>EL crash</td><td>PD crash</td></tr><tr><td rowspan="4">Qwen2.5-7B</td><td>5</td><td>1648</td><td>1215</td><td>26.3%</td><td>1.00</td><td>1.00</td><td>0.00</td><td>0.00</td></tr><tr><td>20</td><td>6057</td><td>1943</td><td>67.9%</td><td>0.17</td><td>1.00</td><td>0.00</td><td>0.00</td></tr><tr><td>50</td><td>14871</td><td>3481</td><td>76.6%</td><td>0.40</td><td>0.67</td><td>0.00</td><td>0.00</td></tr><tr><td>100</td><td>crash</td><td>6223</td><td></td><td>0.00</td><td>1.00</td><td>1.00</td><td>0.00</td></tr><tr><td rowspan="4">Qwen3-8B</td><td>5 20</td><td>1652</td><td>1644</td><td>0.5%</td><td>1.00</td><td>1.00</td><td>0.00</td><td>0.00</td></tr><tr><td></td><td>6060</td><td>2519</td><td>58.4%</td><td>0.68</td><td>0.38</td><td>0.00</td><td>0.00</td></tr><tr><td>50</td><td>14876</td><td>4262</td><td>71.3%</td><td>0.29</td><td>0.64</td><td>0.00</td><td>0.00</td></tr><tr><td>100</td><td>crash</td><td>5184</td><td></td><td>0.00</td><td>0.79</td><td>1.00</td><td>0.00</td></tr><tr><td rowspan="4">Qwen3-14B</td><td>5 20</td><td>1652</td><td>973</td><td>41.1%</td><td>1.00</td><td>1.00</td><td>0.00</td><td>0.00</td></tr><tr><td></td><td>6061</td><td>1553</td><td>74.4%</td><td>0.42</td><td>0.96</td><td>0.00</td><td>0.00</td></tr><tr><td>50</td><td>14875</td><td>2719</td><td>81.7%</td><td>0.13</td><td>0.72</td><td>0.00</td><td>0.00</td></tr><tr><td>100</td><td>crash</td><td>4661</td><td></td><td>0.00</td><td>1.00</td><td>1.00</td><td>0.00</td></tr></table>

Table 1: Eager loading (EL) vs. progressive disclosure (PD) results as a function of library size N. Tokens are per-task means over 72 rollouts. crash denotes context overflow. Token reductions at $N \geq 2 0$ are significant at $p < 1 0 ^ { - 2 5 }$ , based on one-sided Mann–Whitney U test on per-rollout totals.

Which skills are safe to include? With increased adoption of skill libraries, agent developers may accidentally include serious vulnerabilities hidden in skill definitions. How do we efectively monitor an agent’s risk exposure as a result of including a skill in its library?

## References

Kwon, W.; Li, Z.; Zhuang, S.; et al. 2023. Eficient Memory Management for Large Language Model Serving with PagedAttention. In SOSP.

Levy, M.; Jacoby, A.; and Goldberg, Y. 2024. Same Task, More Tokens: the Impact of Input Length on the Reasoning Performance of Large Language Models. In ACL.

Qwen Team. 2024. Qwen2.5 Technical Report. arXiv preprint arXiv:2412.15115.

Qwen Team. 2025. Qwen3 Technical Report. arXiv preprint arXiv:2505.09388.