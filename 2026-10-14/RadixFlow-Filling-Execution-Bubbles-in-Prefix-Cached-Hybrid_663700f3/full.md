# RadixFlow: Filling Execution Bubbles in Prefix-Cached Hybrid LLM Serving

Huihao Jing , Wenbin Hu , Shaojin Chen , Haochen Shi , Haoyu Huang , Sirui Zhang , Yuxuan Liu , Zhongwei Xie , Haoran Li <sup>†</sup>

HKUST

hjingaa@connect.ust.hk

## Abstract

Prefix reuse improves LLM inference by sharing the KV cache of common prompt prefixes. This helps only if the scheduler can execute a matched branch in the current worker round. Schedulers assume that a matched prefix is ready, which holds for full attention but fails for hybrid LLMs. There, a branch (the consumer) may need recurrent state that another branch (the producer) is still computing in the same round. The native scheduler stops its waitingqueue scan at this blocked consumer, and the resulting head-of-line blocking creates execution bubbles: idle batch slots that independent work could fill. We introduce RadixFlow, which decouples prefix matching from state readiness. Its state-ready traversal turns a global scan barrier into a local wait: only the consumer is blocked until the next round, while feasible independent branches keep filling the batch under the native state check. On five hybrid models, RadixFlow lowers mean latency by 3% to 24% and worker rounds by 11% to 44% at almost unchanged reuse. On the larger Qwen3.5 models, ablations show that most of the gain comes from traversal, which adds little on a full-attention control. Heavier rounds limit how much of this saving reaches latency.

All techniques have been implemented and integrated into vllm-jev.

## 1 Introduction

Efficient LLM inference in online settings increasingly involves long shared contexts. Retrievalaugmented generation, multi-turn dialogue, and agent pipelines send queries that repeat the same documents, system prompts, or histories, placing inference in a prefill-dominated regime (Dexter et al., 2025; Patel et al., 2023). An important approach to improving efficiency in this regime is prefix reuse, which keeps the KV cache of shared prefixes (Kwon et al., 2023; Zheng et al., 2024;

Gim et al., 2024; Yu et al., 2023). Under a stream of queries, however, its benefit depends on how the scheduler groups the remaining work into worker rounds (Zhong et al., 2024).

Existing schedulers take one step for granted: a branch that matches a cached prefix can run. This holds for full attention, where a matched prefix needs only its KV pages (Kwon et al., 2023; Zheng et al., 2024). Hybrid LLMs break it, because resuming a branch also requires recurrent and convolutional state at the match boundary (Gu and Dao, 2023; Dao and Gu, 2024; Yang et al., 2024; Pan et al., 2025a; Zhang and SGLang Team, 2026). Following operating-system terminology, we call the branch that computes this state the producer and a branch that must read it the consumer. When both fall in the same worker round, the consumer has a same-round dependency: its prefix is known, but its state is readable only after the producer executes. In the evaluated vLLM path, the consumer’s allocation failure ends the waiting-queue scan (vLLM Contributors, 2026c,b), causing head-of-line blocking. In Figure 1, A must wait for P, so the scan also excludes the independent branch D, although D passes the native state check and fits the remaining budget. We call such legally fillable idle capacity an avoidable execution bubble.

We introduce RadixFlow, a batch-construction policy that separates a prefix match from state readiness. Its state-ready traversal blocks only the consumer, which waits in a wait queue and is woken for a recheck in the next round, while independent branches join the current batch. Other allocation outcomes keep their native handling. Flexible admission control briefly collects arrived work before the scan. In any fixed round, RadixFlow selects a superset of the native batch. On five hybrid models, RadixFlow lowers mean client latency by 3% to 24% while prefix reuse stays within 0.03 percentage points of Radix. Traversal contributes most of the gain on the larger Qwen3.5 models and little on a full-attention control. The remaining rounds carry more work, so their cost decides how much of the 11% to 44% round reduction reaches latency (Agrawal et al., 2024; Pan et al., 2025b).

![](images/a6a962b787c55c596aa5e6d37c1fd568824b5fb6f84e035a06cf3ba32169bee9.jpg)  
Figure 1: Consumer A waits for P, while independent D can still enter the round. Both panels share the same prefix structure and fifteen work fragments. The baseline stops at consumer A, whose state depends on producer P; RadixFlow blocks only A and keeps scanning. Blue outlines mark first-round selections, dashed cells are empty, and shaded lengths show token load. Blocked consumers are woken for a recheck in the next round. These idealized schedules process the same 12,970 tokens in six versus four rounds.

Our contributions are:

• A cache hit is not a runnable branch. We identify same-round dependencies in hybrid LLM serving, formalize readiness and avoidable execution bubbles, and show how headof-line blocking leaves feasible work unscheduled despite a prefix match and spare capacity.

• Block the consumer, not the queue. We develop state-ready traversal, which blocks only the consumer and keeps selecting independent work, and prove its fixed-round feasibility and selection properties without changing cache storage or native state checks.

• Fewer rounds, faster serving. RadixFlow lowers mean latency by 3% to 24% on five hybrid models at nearly unchanged reuse. Ablations, round-occupancy traces, and workercost accounting relate the gains to batch construction.

## 2 Related Work

Prefix caching decides what state can be reused. PagedAttention shares KV pages, RadixAttention indexes exact prefixes, and hybrid caches retain the recurrent components needed at a reuse boundary (Kwon et al., 2023; Zheng et al., 2024; Pan et al., 2025a; Zhang and SGLang Team, 2026). RadixFlow preserves that storage policy and uses the native state check to select executable work.

Scheduling systems balance reuse, load, and waiting through request ordering or batch formation (Dexter et al., 2025; Srivatsa et al., 2025; Agrawal et al., 2024; Zheng et al., 2026). Bounded dynamic batching motivates our timer controls (NVIDIA, 2026b). RadixFlow acts inside a round: a same-round dependency changes which branches are selected, without changing their native order or bypassing a state read. Kernel and tree-aware batching work also motivates measuring per-round cost (Dao et al., 2022; Pan et al., 2025b). Appendix K gives the broader comparison.

## 3 Problem Formulation

We study batch construction for a prefix-cached worker. A branch must match the intended prefix, have usable state at its resume boundary, and fit the remaining resources. These requirements determine which arrived branches can execute together. We follow the finite-stream setting of Dexter et al. (2025).

## 3.1 How a branch executes

A branch can reuse state only after publication. Figure 2 links prefix lookup and the native state check to batch construction, restoration, and prefill. Restoration precedes the resumed forward; publication precedes dependent reuse. Independent CPU control can overlap GPU work (NVIDIA, 2026a), and several branches share one worker invocation (Yu et al., 2022; Agrawal et al., 2024).

## 3.2 Streams, state, and rounds

Notation. A worker round holds at most $S$ rows and $T$ scheduled tokens; $x _ { \le h }$ denotes the first $h$ tokens of input $x .$ . Request $q _ { i }$ arrives at offered time $a _ { i }$ and contains input branches $x _ { i j }$ . Let R be all branches; for $\boldsymbol { r } = ( i , j )$ , write $x _ { r } = x _ { i j }$ . The policy observes only arrived inputs. In Figure 1, $P$ and A are branches sharing a prefix. Full stream notation appears in Appendix D.

State readiness. Let $\mathcal { C } _ { t }$ contain published state that passes the native state check. A hybrid reuse boundary needs $\mathrm { K V , }$ recurrent, and convolutional components. Its ready length $h _ { t } ( r )$ is the largest usable boundary, while $\hat { h } _ { t } ( \boldsymbol { r } )$ is the longest token match. For A, $\hat { h } _ { t } ( A ) \ > \ h _ { t } ( A )$ when $P$ is still producing the matched state. That state enters $\mathcal { C } _ { t + 1 }$ after execution. Appendix D gives Definition 3.

Let $\boldsymbol { A } _ { t }$ contain arrived, admitted branches with unscheduled work. A feasible partial batch $B \subseteq A _ { t }$ fits both budgets and resumes each branch from valid published state or its own earlier state. This permits D to join $P$ , but prevents A from reading state that $P$ is still producing.

Definition 1 (Readiness outcome). Fix round t and a partial batch $B \subseteq A _ { t } .$ . The native state check maps each $r \in \mathcal { A } _ { t } \setminus B$ to an outcome $\sigma _ { t } ( r$ | $\begin{array} { l l } { \displaystyle { B } ) } & { \displaystyle \in \mathrm { ~  ~ \Omega ~ } . } \end{array}$ {READY, BLOCKED, OTHER}. READY means $B \cup \{ r \}$ is feasible. BLOCKED means the check rejects r because a component at $\hat { h } _ { t } ( \boldsymbol r )$ is still being written by some producer $r ^ { \prime } \in { \boldsymbol { B } }$ , that ${ \mathrm { i } } \mathbf { s } ,$ r is a consumer with a same-round dependency. OTHER covers all remaining outcomes, including budget overflow and unrecognized allocation failures.

READY is stronger than a ready boundary: it also requires the branch to fit the round. With P selected, A is BLOCKED, while an independent branch such as D can be READY.

Definition 2 (Execution bubble). Unused rows are counted by $b _ { t } = S - \vert B _ { t } \vert$ . An avoidable execution bubble exists when $b _ { t } > 0$ and some $r \in \mathcal { A } _ { t } \setminus \mathcal { B } _ { t }$ has $\sigma _ { t } ( r \ | \ B _ { t } ) =$ READY: the batch leaves out arrived work that could legally fill its capacity.

In the first prefix-stop round of Figure 1, $P$ occupies one of four rows, leaving three rows empty. This row-based measure differs from pipeline bubbles (Agrawal et al., 2024). Our target is legally fillable capacity; a full token budget or an empty executable set can also leave rows unused.

Problem 1 (Online batch construction). Let $C _ { i } ( \pi )$ be the completion time of the last branch of logical request $q _ { i }$ . Given arrived work, state $\mathcal { C } _ { t }$ , and the native outcomes $\sigma _ { t } ,$ construct feasible batches online to minimize mean flow time,

$$
J ( \pi ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \bigl ( C _ { i } ( \pi ) - a _ { i } \bigr ) .
$$

Round count and round cost determine service work, while arrivals and queueing determine when each request receives it. Section 5.4 relates the service quantities; the experiments measure both offered-arrival and post-submission latency.

## 3.3 Where bubbles arise

The evaluated vLLM waiting loop ends its scan when native allocation returns no blocks. Its hybridstate manager produces this outcome for matched current-step state, even with available pool capacity (vLLM Contributors, 2026c,b). Appendix F.1 identifies the source path. We model this path as scanning $\boldsymbol { A } _ { t }$ in native order ≺ and adding each READY branch. We call this rule prefix-stop when the scan ends at the first branch whose outcome is not READY. Consider a producer $P$ and a consumer $A$ that share a prefix. Once $P$ is selected, $\sigma _ { t } ( A { \it \Delta \phi } | \ B ) \ = \ \mathrm { \bf { B } } \mathrm { { L O C K E D } }$ . A prefix-stop scan ends at A, so every READY branch $r \succ A$ is left out, which is head-of-line blocking. Continuing the scan pays off when arrived independent work passes the native state check and fits the remaining capacity. In that case, the empty row is avoidable by Definition 2.

![](images/2f357548610e1ee9eba955fb9683ff173d59403e3d938dff87177f0a3553dc16.jpg)  
Figure 2: RadixFlow changes selection, not native state checks or cache updates. A checkmark selects feasible work for $B _ { t } ; \mathbf { a }$ pause blocks a consumer with a same-round dependency in the wait queue $D _ { t } .$ , and a cross preserves native handling. Solid state components are published. The outlined recurrent component is still in flight and is not part of $\mathcal { C } _ { t } ;$ blocked consumers are woken for a recheck in round t + 1. The queue inset contrasts head-of-line blocking with continuing past the blocked consumer.

Figure 1 shows the effect. Both schedules contain the same fifteen work fragments and 12,970 tokens. Stopping at A takes six four-slot rounds and leaves nine slots empty. Continuing past A takes four rounds and leaves one slot empty.

Packing reference. For fixed work fragments with W tokens, V row visits, and dependency depth D, at least max $\{ \lceil W / T \rceil , \lceil V / S \rceil , D \}$ rounds are required. The four-round example attains this capacity bound for its 12,970 tokens and fifteen fragments. Proposition 1 and its proof appear in Appendices D and E.

## 4 RadixFlow

RadixFlow recovers executable capacity within a worker round. Its main rule is state-ready traversal: block only the consumer of a same-round dependency and keep selecting independent work, removing head-of-line blocking. Flexible admission determines when arrived branches become eligible before that scan. Both stages retain the native state check and cache management.

## 4.1 State-ready traversal

Traversal replaces the prefix-stop rule with one change. It scans eligible branches in the native order ≺ and keeps a partial batch B and a wait queue $D _ { t }$ . A branch with $\sigma _ { t } ( r \mid B )$ = READY is selected through the native allocator. A branch with outcome BLOCKED joins $D _ { t }$ and stays registered, and the scan continues. An OTHER outcome keeps the native action (Algorithm 1). The outcome of a branch is read at its scan position, against the batch built so far.

Algorithm 1 RadixFlow constructs a feasible   
round by blocking only consumers of same-round   
dependencies.   
1: Apply bounded admission to arrived branches.   
2: $\Vec { B ^ { \prime } } \gets \Vec { \boldsymbol { \emptyset } } ; D _ { t } \gets \boldsymbol { \emptyset }$   
3: for eligible branch r in native order ≺ do   
4: $\sigma \gets \sigma _ { t } ( r \mid B )$ ▷ Native state check   
5: if $\sigma =$ READY then   
6: Select r through the native allocator.   
7: $B  B \cup \{ r \}$   
8: else if $\sigma =$ BLOCKED then   
9: $D _ { t } \gets D _ { t } \cup \{ r \} ;$ keep r registered.   
10: Continue scanning.   
11: else   
12: Apply the native action. ▷ May stop   
13: end if   
14: end for   
15: Execute $\begin{array} { r } { B _ { t } = B ; } \end{array}$ publish to $\mathcal { C } _ { t + 1 }$   
16: Wake the consumers in $D _ { t }$ for a recheck in round t + 1.

Continuing after every allocation failure would mix different causes. A BLOCKED consumer waits for an already selected producer; an OTHER result can indicate a resource limit or another failure. RadixFlow continues in the former case and preserves native handling in the latter. Thus A waits for P while the scan visits D.

Properties. For fixed state and native order, RadixFlow constructs a feasible batch containing every branch selected by prefix-stop. If readiness is monotone as the batch grows and no OTHER outcome stops the scan, no avoidable bubble remains. Proposition 2 in Appendix D states the assumptions and Appendix E proves the result. This is a fixed-round guarantee; stream-wide savings are measured below.

Cost. Continuing the scan makes additional queue visits and repeats the native state check. Each blocked consumer is visited once per pass, so the extra work grows linearly with the number of inspected branches.

## 4.2 Flexible admission

Admission control trades batch size against waiting time. The admission deadline is 5 ms, about 9% of the 52.95 ms native mean worker-call interval on Qwen3.5-0.8B. We fix this short bound across models rather than tune it on evaluation results. A branch is released earlier in three cases: the worker already has running work, the held branches fill the available rows, or their residual token demand fills the token budget. Residual demand is computed from current native cache availability. A branch does not wait for the rest of its request group. Flex denotes flexible admission. The Radix+Flex variant keeps this rule and removes state-ready traversal, so it isolates the effect of admission. The timer controls differ from Radix+Flex because they release work only at their deadline or when rows are full.

## 4.3 Fragmentation extension

A small residual budget can split a branch into an extra fragment. RadixFlow+Frag defers such a branch when the split would shorten its next aligned segment, and then looks for another fit. This can reduce fragmentation, but it can also delay progress. We keep RadixFlow as the main method and test RadixFlow+Frag as an ablation.

Design choices. RadixFlow changes selection rather than checkpoint placement. Additional recurrent snapshots can enable more reuse but also incur copying and retention costs; waiting for every branch can postpone work that is already executable. The policy instead uses published native state and releases each arrived branch independently. Appendix A details these choices.

## 5 Experiments

We ask: (Q1) Does RadixFlow improve serving performance? (Q2) Where does the gain come from? (Q3) How does per-round cost shape the latency improvement?

## 5.1 Setup

Workload. We need heavy prefix sharing and little decode, so we build the workload from QuAL-ITY and RACE reading comprehension (Pang et al., 2022; Lai et al., 2017): 1034 QuALITY questions from 62 articles and 517 RACE-M questions from 128 articles. Each question becomes one logical request with four branches. Each branch holds the document, the question, and one candidate answer, and generates one token. The shared-document, candidate-specific structure models the prefill pattern of multi-candidate scoring and candidate-comparison judging. We measure serving performance rather than answer accuracy. Each model uses its own tokenizer, and overlength inputs are rejected rather than truncated.

Cold streams represent unrelated questions that share no cache across requests, while reuse streams represent repeated questions about a shared document. Mixed streams interleave documents. We implement these regimes with cache namespaces and fixed request orders; branches within a question still share its prefix. Three regimes and two input orders give 9306 logical requests per policy.

Backend and models. All policies run in vLLM 0.29 on A800 GPUs with eager execution, asynchronous scheduling disabled, and allocator utilization 0.8. They share the prefix index, the native state check, model weights, and cache primitives, so only the scheduling rule changes. We test four hybrid Qwen3.5 models (0.8B, 4B, 9B, and 27B), including Qwen3.5-0.8B-Base, shortened to Qwen3.5-0.8B. NVIDIA-Nemotron-Nano-9B-v2 is a Mamba2 hybrid, shortened to Nemotron-9B (NVIDIA Corporation, 2025). Qwen3-8B is the full-attention control, denoted Qwen3-8B (FA) (Qwen Team, 2025, 2026).

Calibration. To avoid favoring any policy, we tune the row and token budgets once for the native scheduler on 64 held-out questions, following vLLM’s automatic tuner (vLLM Contributors, 2026a), and then freeze them. The chosen budgets appear in Table 1. All Qwen runs use an offered rate of five requests per second, which overloads the larger models. Nemotron-9B runs at 1.5 requests per second, which is 80% of its calibrated native throughput. Appendix B gives the full protocol.

Policies. The six policies compare scheduling components on a common backend. Radix retains native vLLM admission and traversal within the shared experimental overlay. The overlay adds registration, indexing, tracing, and the common copy guard; Radix is not an untouched vLLM build. Radix+Timer5 and Radix+Timer10 add bounded waiting with two deadline settings (NVIDIA, 2026b). Radix+Flex adds our admission rule. RadixFlow adds state-ready traversal on top, and RadixFlow+Frag adds the fragmentation guard. So Radix versus Radix+Flex tests admission, and Radix+Flex versus RadixFlow tests traversal.

<table><tr><td>Method</td><td>Latency (s) ↓</td><td>Replay (req/s) ↑</td><td>Rounds K↓</td><td>Reuse (%) ↑</td></tr><tr><td>Qwen3.5-0.8B</td><td></td><td></td><td>S=16, T=8192, 5.0 req/s</td><td></td></tr><tr><td>Radix</td><td>0.256</td><td>4.806</td><td>19,803</td><td>81.362</td></tr><tr><td>Radix+Timer5</td><td>0.260</td><td>4.805</td><td>19,371</td><td>81.367</td></tr><tr><td>Radix+Timer10</td><td>0.264</td><td>4.805</td><td>19,181</td><td>81.373</td></tr><tr><td>Radix+Flex</td><td>0.249</td><td>4.805</td><td>18,409</td><td>81.364</td></tr><tr><td>RadixFlow</td><td>0.232</td><td>4.806</td><td>17,610</td><td>81.366</td></tr><tr><td>RadixFlow+Frag</td><td>0.228</td><td>4.806</td><td>17,744</td><td>81.371</td></tr><tr><td>Qwen3.5-4B</td><td></td><td></td><td>S=64, T=8192, 5.0 req/s</td><td></td></tr><tr><td>Radix</td><td>13.1</td><td>4.210</td><td>13,424</td><td>80.405</td></tr><tr><td>Radix+Timer5</td><td>11.8</td><td>4.269</td><td>12,698</td><td>80.341</td></tr><tr><td>Radix+Timer10</td><td>11.5</td><td>4.274</td><td>12,526</td><td>80.377</td></tr><tr><td>Radix+Flex</td><td>12.8</td><td>4.206</td><td>12,942</td><td>80.433</td></tr><tr><td>RadixFlow</td><td>10.0</td><td>4.387</td><td>10,146</td><td>80.431</td></tr><tr><td>RadixFlow+Frag</td><td>10.1</td><td>4.345</td><td>10,296</td><td>80.450</td></tr><tr><td>Qwen3.5-9B</td><td></td><td></td><td>S=64, T=8192, 5.0 req/s</td><td></td></tr><tr><td>Radix</td><td>26.7</td><td>3.240</td><td>11,042</td><td>79.686</td></tr><tr><td>Radix+Timer5</td><td>26.5</td><td>3.245</td><td>10,365</td><td>79.461</td></tr><tr><td>Radix+Timer10</td><td>26.5</td><td>3.257</td><td>10,279</td><td>79.471</td></tr><tr><td>Radix+Flex</td><td>26.5</td><td>3.252</td><td>10,770</td><td>79.757</td></tr><tr><td>RadixFlow</td><td>24.8</td><td>3.368</td><td>7,444</td><td>79.696</td></tr><tr><td>RadixFlow+Frag</td><td>24.9</td><td>3.352</td><td>7,534</td><td>79.755</td></tr></table>

<table><tr><td>Method</td><td>Latency (s) ↓</td><td>Replay (req/s) ↑</td><td>Rounds K↓</td><td>Reuse (%) ↑</td></tr><tr><td>Qwen3.5-27B</td><td></td><td></td><td>S=64, T=8192, 5.0 req/s</td><td></td></tr><tr><td>Radix</td><td>164</td><td>0.743</td><td>11,748</td><td>68.497</td></tr><tr><td>Radix+Timer5</td><td>168</td><td>0.728</td><td>10,524</td><td>67.426</td></tr><tr><td>Radix+Timer10</td><td>169</td><td>0.726</td><td>10,453</td><td>67.334</td></tr><tr><td>Radix+Flex</td><td>163</td><td>0.746</td><td>11,555</td><td>68.567</td></tr><tr><td>RadixFlow</td><td>159</td><td>0.766</td><td>6,529</td><td>68.508</td></tr><tr><td>RadixFlow+Frag</td><td>151</td><td>0.805</td><td>6,293</td><td>70.166</td></tr><tr><td>Nemotron-9B</td><td></td><td></td><td>S=16, T=8192, 1.5 req/s</td><td></td></tr><tr><td>Radix</td><td>2.44</td><td>1.442</td><td>14,338</td><td>75.939</td></tr><tr><td>Radix+Timer5</td><td>2.41</td><td>1.442</td><td>10,539</td><td>75.947</td></tr><tr><td>Radix+Timer10</td><td>2.38</td><td>1.442</td><td>10,471</td><td>75.950</td></tr><tr><td>Radix+Flex</td><td>2.43</td><td>1.442</td><td>10,455</td><td>75.941</td></tr><tr><td>RadixFlow</td><td>2.15</td><td>1.442</td><td>9,613</td><td>75.950</td></tr><tr><td>RadixFlow+Frag</td><td>2.11</td><td>1.442</td><td>9,861</td><td>75.959</td></tr><tr><td>Qwen3-8B (FA)</td><td></td><td></td><td>S=16, T=8192, 5.0 req/s</td><td></td></tr><tr><td>Radix</td><td>16.4</td><td>4.000</td><td>9,223</td><td>85.243</td></tr><tr><td>Radix+Timer5</td><td>16.7</td><td>3.996</td><td>6,779</td><td>85.209</td></tr><tr><td>Radix+Timer10</td><td>16.4</td><td>4.001</td><td>6,536</td><td>85.265</td></tr><tr><td>Radix+Flex</td><td>16.3</td><td>4.002</td><td>6,782</td><td>85.250</td></tr><tr><td>RadixFlow</td><td>16.4</td><td>4.000</td><td>6,746</td><td>85.250</td></tr><tr><td>RadixFlow+Frag</td><td>16.4</td><td>4.002</td><td>6,761</td><td>85.254</td></tr></table>

Table 1: RadixFlow lowers mean client latency on the hybrid models while preserving almost the same reused input. The first three methods are baselines; the last three are our ablation, main method, and extension. Each block covers 9306 requests across three regimes and two orders. Replay throughput divides completions by arm wall time, including arrivals and drain; it is not a saturation limit. Bold marks the best latency, round count, and reuse. FA denotes full attention.
<table><tr><td>Model</td><td colspan="3">Rows / round ↑</td><td colspan="3">First selection (s) ↓</td></tr><tr><td></td><td>Radix</td><td>RadixFlow</td><td>Gain (%)</td><td>Radix</td><td>RadixFlow</td><td>Reduction (%)</td></tr><tr><td>Qwen3.5-0.8B</td><td>3.63</td><td>4.10</td><td>12.9</td><td>0.057</td><td>0.049</td><td>14.0</td></tr><tr><td>Qwen3.5-4B</td><td>5.42</td><td>7.31</td><td>34.9</td><td>12.576</td><td>9.273</td><td>26.3</td></tr><tr><td>Qwen3.5-9B</td><td>6.61</td><td>10.02</td><td>51.6</td><td>25.682</td><td>23.255</td><td>9.5</td></tr><tr><td>Qwen3.5-27B</td><td>6.51</td><td>12.11</td><td>86.0</td><td>160.538</td><td>153.314</td><td>4.5</td></tr><tr><td>Nemotron-9B</td><td>2.91</td><td>4.38</td><td>50.5</td><td>1.898</td><td>1.582</td><td>16.6</td></tr><tr><td>Qwen3-8B (FA)</td><td>4.28</td><td>5.86</td><td>36.9</td><td>15.817</td><td>15.813</td><td>0.0</td></tr></table>

Table 2: Denser batches accompany earlier selection on hybrids. First selection measures registration-to-selection waiting; percentage changes use displayed values. See Appendix M for output sensitivity.

Metrics. We report completed-replay throughput and post-submission latency, with sender lag separately tabulated in Appendix J. Rounds count nonempty global worker calls; reuse counts native reported hit tokens. Worker time is the elapsed host-side call interval (Appendix C).

## 5.2 Q1: Serving performance

RadixFlow reduces mean client latency by 9.3%, 23.6%, 7.1%, and 3.1% on the Qwen3.5 scale range, and by 12.0% on Nemotron-9B (Table 1). Reused input stays within 0.03 percentage points of Radix on each hybrid. Completed-replay throughput rises by 4.2%, 4.0%, and 3.0% on Qwen3.5-4B, 9B, and 27B. On the lighter Qwen3.5-0.8B and Nemotron runs, completion throughput tracks the offered arrivals while request latency falls. Each improvement compares policies under the same model, frozen budgets, and arrival trace.

![](images/839973b27d5929dc9157fd586cf3b29fcb5259ff74724a18b9575b694339c3b9.jpg)

![](images/079fc3c4141796e2aa40ffb892de9011b7545c4c5aa8ae0d21ab8e935d5de390.jpg)  
Figure 3: Round savings and latency gains differ across the measured configurations. Left: reductions relative to Radix. Right: blue is net host-time credit and orange is the cost shift of remaining rounds; together they equal the gross round credit under Equation 2. The panels share model order.

## 5.3 Q2: Admission and state-ready traversal

The incremental comparison separates admission from traversal. On Qwen3.5-27B, Radix+Flex saves 1.6% of the rounds, whereas RadixFlow saves 44%. The difference persists in each stream: traversal adds a reduction of 44%, 43%, and 44% relative to Radix+Flex in cold, mixed, and reuse streams. Here cold means no sharing across questions; the four branches of a question still share a prefix. A same-round dependency can therefore arise even in a cold stream.

Figure 4(a)–(c) and Table 2 show denser rounds and earlier selection. On the full-attention control, Flex uses 6782 rounds and RadixFlow 6746. This small traversal increment is consistent with its recurrent-state check remaining inactive, while admission still groups work. Nemotron-9B gains chiefly from admission: it removes 27% of rounds, followed by another 8% from traversal. Its native occupancy of 2.91 rows out of 16 leaves room for admission to pack more work at the lower offered load.

## 5.4 Q3: Round savings and their cost

A denser round processes more work, so round count and elapsed time change by different amounts. Figure 3 measures this conversion.

Worker-time accounting. Let $I _ { r } = | x _ { r } |$ and let $H _ { r }$ be the cache-hit tokens that the engine reports for branch $^ { r } \cdot$ To measure how expensive each worker call is, we use its elapsed host-wrapper interval, which can include GPU waiting as well as

CPU work. With this round cost $c ( \boldsymbol { B } _ { t } )$ , we define

$$
\rho = \frac { \sum _ { r } H _ { r } } { \sum _ { r } I _ { r } } , \quad T _ { \mathrm { h o s t } } = \sum _ { t = 1 } ^ { K } c ( \mathcal { B } _ { t } ) , \quad \bar { c } = \frac { T _ { \mathrm { h o s t } } } { K } .\tag{1}
$$

Token reuse $\rho$ measures how much input is skipped. K and c¯ measure how the rest is executed. For a baseline (index 0) and a policy (index 1), Equation 1 gives the exact identity

$$
T _ { \mathrm { h o s t , 0 } } - T _ { \mathrm { h o s t , 1 } } = \underbrace { \left( K _ { 0 } - K _ { 1 } \right) \bar { c } _ { 0 } } _ { \mathrm { r o u n d c r e d i t } } - \underbrace { K _ { 1 } ( \bar { c } _ { 1 } - \bar { c } _ { 0 } ) } _ { \mathrm { c o s t s h i f t } } .\tag{2}
$$

The identity separates saved invocations from the changed cost of the remaining work. We interpret it alongside client latency and completed-replay throughput. Packing replaces many small invocations with fewer larger ones. On Qwen3.5-27B, tokens per round rise by 80%, but host cost per thousand scheduled tokens falls from 183 to 175 ms. Total host-wrapper time falls by 3.8%, alongside the 3.0% replay-throughput gain. Figure 3 separates round credit from the retained-round cost shift; on Qwen3.5-9B, that shift absorbs about 60% of the credit.

Waiting and tail latency. Table 2 shows earlier native selection on every hybrid. On Qwen3.5- 9B, p95 falls from 73.3 to 70.6 s; the full-attention control has nearly unchanged tails. Figure 4(d)–(f) complements the mean latency with the request distribution.

Selection evidence. Holding admission fixed, the Flex–RadixFlow comparison isolates the traversal increment. The higher occupancy and earlier selection in Table 2 provide complementary evidence for improved batch construction. Appendix J specifies the trace scope, and Appendix M reports output sensitivity.

![](images/af76144a2929097d0eb5ffe561e3e227742b3964d0163e45aeb3797a1b233737.jpg)  
Figure 4: (a)–(c) Round fullness: RadixFlow shifts rounds toward fuller budgets; rightward is better. (d)–(f) Request latency: the benefit reaches requests only partly and varies by model; leftward is faster. Curves pool three regimes and two orders per model. Latency is measured after submission and displayed through the largest p99 among the three policies.

On Qwen3.5-27B, +Frag lowers latency by another 5.0% relative to RadixFlow, with 3.6% fewer rounds and 1.66 percentage points more reuse. Deferring a shortened aligned fragment changes both batch packing and the cache boundaries later consumers encounter, coupling fragmentation control with the realized cache trajectory.

The hybrid ablations and full-attention control separate traversal from admission. Occupancy rises as rounds fall, and worker-time accounting quantifies how their cost affects completion. The observed request benefit is therefore interpreted through selection and execution cost, rather than through cache hits alone.

Arrival order and the operating point. Arrival order changes both the cache trajectory and the branches available for packing. RadixFlow lowers mean latency in both orders on every hybrid model (Appendix J). Offered load determines how those packing opportunities reach the client. With a small queue, earlier selection can shorten waiting even when completed throughput follows the arrival rate. Under sustained overload, packing instead helps drain accumulated work. Qwen3.5-27B illustrates the latter setting: its five-request/s offered rate exceeds its replay completion rate, so queueing dominates client latency. Throughput and total worker time quantify how efficiently this backlog is processed.

The full-attention control clarifies the role of state readiness. Admission can still consolidate branches that share a prefix, but traversal contributes little when the recurrent-state check is inactive. Hybrid models expose an additional opportunity: preserve the producer’s work, block its consumer, and use remaining capacity for an independent branch.

## 6 Conclusion

In hybrid LLM serving, a cache match is not yet a runnable branch. RadixFlow blocks only the consumer of a same-round dependency and keeps selecting independent work that fits. Schedulers should therefore separate such dependencies from memory pressure: a blocked consumer needs only a recheck in the next round.

## Limitations

Our evaluation focuses on candidate-conditioned prefill with one generated token under frozen model, budget, and arrival settings. Worker-call timing and decoded-output agreement are the reported measures. Load sweeps, production scheduling, and finer dependency traces extend this setting; Appendix N records the comparison scope.

## Ethical Considerations

Experiments use public benchmark inputs and publicly available model weights under their terms. Exact-prefix namespaces remain isolated, and no approximate semantic cache reuse is introduced. Our scripts keep failed and incomplete artifacts. Final summaries are produced only after identifier, ownership, and token-accounting checks. Appendix F documents the native state contract and shared implementation primitives. AI assistance was used for implementation and manuscript revision. The authors remain responsible for the reported evidence and claims.

## References

Amey Agrawal, Nitin Kedia, Ashish Panwar, Jayashree Mohan, Nipun Kwatra, Bhargav Gulavani, Alexey Tumanov, and Ramachandran Ramjee. 2024. Taming throughput-latency tradeoff in LLM inference with Sarathi-Serve. In 18th USENIX Symposium on Operating Systems Design and Implementation, pages 117–134. USENIX Association.

Tri Dao. 2023. FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. Preprint, arXiv:2307.08691. ArXiv preprint.

Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. 2022. FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. In Advances in Neural Information Processing Systems, volume 35.

Tri Dao and Albert Gu. 2024. Transformers are SSMs: Generalized models and efficient algorithms through structured state space duality. In Proceedings ofthe 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 10041–10071. PMLR.

Gregory Dexter, Shao Tang, Ata Fatahi, Qingquan Song, Tejas Dharamsi, and Aman Gupta. 2025. LLM query scheduling with prefix reuse and latency constraints. In Advances in Neural Information Processing Systems, volume 38.

Bin Gao, Zhuomin He, Puru Sharma, Qingxuan Kang, Djordje Jevdjic, Junbo Deng, Xingkun Yang, Zhou

Yu, and Pengfei Zuo. 2024. Cost-Efficient Large Language Model Serving for Multi-turn Conversations with CachedAttention. Preprint, arXiv:2403.19708. ArXiv preprint.

In Gim, Guojun Chen, Seung-seob Lee, Nikhil Sarda, Anurag Khandelwal, and Lin Zhong. 2024. Prompt Cache: Modular Attention Reuse for Low-Latency Inference. In Proceedings ofMachine Learning and Systems, volume 6.

Albert Gu and Tri Dao. 2023. Mamba: Linear-Time Sequence Modeling with Selective State Spaces. Preprint, arXiv:2312.00752. ArXiv preprint.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient memory management for large language model serving with PagedAttention. In Proceedings of the 29th Symposium on Operating Systems Principles.

Guokun Lai, Qizhe Xie, Hanxiao Liu, Yiming Yang, and Eduard Hovy. 2017. RACE: Large-scale ReAding comprehension dataset from examinations. In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, pages 785– 794. Association for Computational Linguistics.

Yuhan Liu, Yihua Cheng, Jiayi Yao, Yuwei An, Xiaokun Chen, Shaoting Feng, Yuyang Huang, Samuel Shen, Rui Zhang, Kuntai Du, and Junchen Jiang. 2025. LM-Cache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference. Preprint, arXiv:2510.09665. ArXiv preprint.

NVIDIA. 2026a. CUDA C++ Best Practices Guide: Timing. Official technical documentation. Accessed October 2026; year denotes access year.

NVIDIA. 2026b. Triton inference server: Dynamic batcher. Archived version 2.58.0 documentation. Accessed October 2026; year denotes access year.

NVIDIA Corporation. 2025. NVIDIA-Nemotron-Nano-9B-v2: Official model card. Model released 18 August 2025; accessed 12 October 2026.

Rui Pan, Zhuang Wang, Zhen Jia, Can Karakus, Luca Zancato, Tri Dao, Yida Wang, and Ravi Netravali. 2025a. Marconi: Prefix caching for the era of hybrid LLMs. In Proceedings of Machine Learning and Systems, volume 7.

Zaifeng Pan, Yitong Ding, Yue Guan, Zheng Wang, Zhongkai Yu, Xulong Tang, Yida Wang, and Yufei Ding. 2025b. FastTree: Optimizing attention kernel and runtime for tree-structured LLM inference. In Proceedings of Machine Learning and Systems, volume 7.

Richard Yuanzhe Pang, Alicia Parrish, Nitish Joshi, Nikita Nangia, Jason Phang, Angelica Chen, Vishakh Padmakumar, Johnny Ma, Jana Thompson, He He, and Samuel R. Bowman. 2022. QuALITY: Question answering with long input texts, yes! In Proceedings

ofthe 2022 Conference ofthe North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, pages 5336–5358. Association for Computational Linguistics.

Pratyush Patel, Esha Choukse, Chaojie Zhang, Aashaka Shah, Íñigo Goiri, Saeed Maleki, and Ricardo Bianchini. 2023. Splitwise: Efficient generative LLM inference using phase splitting. Preprint, arXiv:2311.18677. ArXiv preprint.

Ramya Prabhu, Ajay Nayak, Jayashree Mohan, Ramachandran Ramjee, and Ashish Panwar. 2024. vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention. Preprint, arXiv:2405.04437. ArXiv preprint.

Qwen Team. 2025. Qwen3-8B: Official Model Card. Official model card. Accessed October 2026.

Qwen Team. 2026. Qwen3.5-0.8B-Base: Official Model Card. Official model card. Accessed October 2026.

Vikranth Srivatsa, Zijian He, Reyna Abhyankar, Dongming Li, and Yiying Zhang. 2025. Preble: Efficient Distributed Prompt Scheduling for LLM Serving. In International Conference on Learning Representations.

Biao Sun, Ziming Huang, Hanyu Zhao, Wencong Xiao, Xinyi Zhang, Yong Li, and Wei Lin. 2024. Llumnix: Dynamic scheduling for large language model serving. In 18th USENIX Symposium on Operating Systems Design and Implementation (OSDI 24), pages 173–191. USENIX Association.

vLLM Contributors. 2026a. Automated vllm server parameter tuning. Technical documentation, source snapshot retained. Accessed October 2026; year denotes access year.

vLLM Contributors. 2026b. vLLM 0.29.0: Native scheduler. Versioned implementation (v0.29.0). Accessed October 2026; year denotes access year.

vLLM Contributors. 2026c. vLLM 0.29.0: Singletype KV cache manager. Versioned implementation (v0.29.0). Accessed October 2026; year denotes access year.

Bingyang Wu, Yinmin Zhong, Zili Zhang, Shengyu Liu, Fangyue Liu, Yuanhang Sun, Gang Huang, Xuanzhe Liu, and Xin Jin. 2023. Fast Distributed Inference Serving for Large Language Models. Preprint, arXiv:2305.05920. ArXiv preprint.

Songlin Yang, Jan Kautz, and Ali Hatamizadeh. 2024. Gated Delta Networks: Improving Mamba2 with Delta Rule. Preprint, arXiv:2412.06464. ArXiv preprint.

Zihao Ye, Lequn Chen, Ruihang Lai, Wuwei Lin, Yineng Zhang, Stephanie Wang, Tianqi Chen, Baris Kasikci, Vinod Grover, Arvind Krishnamurthy, and

Luis Ceze. 2025. FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving. In Proceedings of Machine Learning and Systems, volume 7.

Gyeong-In Yu, Joo Seong Jeong, Geon-Woo Kim, Soojeong Kim, and Byung-Gon Chun. 2022. Orca: A distributed serving system for Transformer-Based generative models. In 16th USENIX Symposium on Operating Systems Design and Implementation, pages 521–538. USENIX Association.

Lingfan Yu, Jinkun Lin, and Jinyang Li. 2023. Stateful Large Language Model Serving with Pensieve. Preprint, arXiv:2312.05516. ArXiv preprint.

Yichi Zhang and SGLang Team. 2026. Unified radix cache in SGLang: One tree for hybrid model prefix caching. Technical documentation. Accessed October 2026; year denotes access year.

Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Chuyue Sun, Jeff Huang, Cody Hao Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph E. Gonzalez, Clark Barrett, and Ying Sheng. 2024. SGLang: Efficient execution of structured language model programs. In Advances in Neural Information Processing Systems, volume 37.

Zhen Zheng, Xin Ji, Taosong Fang, Fanghao Zhou, Chuanjie Liu, and Gang Peng. 2026. BatchLLM: Optimizing large batched LLM inference with global prefix sharing and throughput-oriented token batching. In Proceedings ofMachine Learning and Systems, volume 8.

Yinmin Zhong, Shengyu Liu, Junda Chen, Jianbo Hu, Yibo Zhu, Xuanzhe Liu, Xin Jin, and Hao Zhang. 2024. DistServe: Disaggregating prefill and decoding for goodput-optimized large language model serving. In 18th USENIX Symposium on Operating Systems Design and Implementation (OSDI 24), pages 193–210. USENIX Association.

## A Design Rationale and Alternatives

This appendix explains the decisions that led to the reported method. The observations come from development prototypes and implementation audits. They are not additional benchmark results. Our final evaluation does not pool exploratory runs with the frozen comparisons.

## A.1 From prefix matches to executable work

Our first question was whether a more precise index could turn more exact matches into useful cached computation. A radix index can locate a shared token boundary, but a hybrid model may lack the recurrent state at that boundary. Counting these matches as executable hits would overstate reuse. This led us to separate three events: finding a prefix, having a complete valid state, and placing the work into a legal worker round.

This separation changed the optimization target. Saving tokens is useful, but it does not show that a global round disappears. A shorter branch may still need the same number of rounds because of row limits, aligned chunks, the progress of other requests, or a dependency that resolves only after an existing round. We therefore kept cached tokens as an accounting quantity and added worker rounds as a scheduling quantity. Neither replaces elapsed time.

## A.2 Why not save a checkpoint at every branch?

A checkpoint at a newly observed branch helps future reuse only if its complete state is available. Building it with an extra serial prefill may cost the time it is meant to save. Capturing it during an ordinary forward pass avoids that extra producer. It still costs state copies, publication, temporary buffers, and retention. A split in the token tree cannot recover a recurrent state that was never saved.

Our prototypes explored opportunistic capture and chained publication. They also revealed unused checkpoints and memory that stayed resident after a logical node became invalid. These observations led to a conservative choice. The reported method leaves checkpoint construction and eviction to the native backend and changes only the response to a recognized same-round dependency. State capture at arbitrary boundaries remains a separate design problem.

## A.3 Why not wait for every group?

Waiting for a whole group can build a denser batch. It can also hold a cold request while its other branches arrive, even though no useful shared state is ready. Later revalidation protects correctness, but it cannot refund the waiting. A cache-aware gate solves this only if its readiness information stays valid and held requests keep normal cancellation and timeout behavior.

We therefore separate admission from state readiness. Flexible admission uses only arrived work and a short waiting bound, and releases work earlier when runnable work or resource demand justifies it. State-ready traversal responds to one specific native rejection rather than making all branches of a group wait together. The timer controls test whether ordinary bounded waiting explains the gain without the same-round dependency signal.

## A.4 Why not minimize rounds alone?

A round with more rows or tokens can cost more than the rounds it replaces. A small change in packing can also move execution to a different kernel shape. In development, fewer tokens or rounds sometimes did not shorten completion time. This led us to keep round cost and client latency as separate measurements. The full-attention control matters here because an inactive recurrent rule should not receive credit for admission gains.

The same reasoning motivates the fragmentation extension. Deferring a branch can avoid a short aligned segment, but it can also delay useful progress. RadixFlow+Frag is therefore an ablation, not an assumed improvement. We do not pick the main method separately for each dataset after seeing which variant wins.

## B Replay and Calibration Protocol

Figure 5 connects the frozen comparison protocol to the request timestamps used in the analysis.

## B.1 Input construction

The main replay uses the same 1551 question identifiers for every policy of a model. A branch concatenates the source context, the question, one candidate answer, and a short correctness prompt. The four candidates produce four input encodings, and each generates one token. This construction keeps realistic shared documents and suffix differences while focusing measurement on prefill. It does not reproduce the full answer-scoring protocols of the benchmarks.

Each model’s own tokenizer is applied before evaluation. We record the input identifier and tokensequence hash, and check positive lengths and the declared context limit. Token lengths can differ across models. No method receives labels, future arrivals, precomputed prefix cuts, or another policy’s state. Overlength inputs are rejected rather than silently truncated or run with an unreported context extension.

## B.2 Cache isolation and arrival traces

Cold requests receive question-specific cache namespaces. Shared-prefix conditions use the article namespace. Each policy run uses separate namespaces, so one arm cannot warm another. A logical request identifier is distinct from its cache namespace. Several concurrent requests can share an article without becoming one request group.

![](images/c1b3ccd315144afcc64849b1b89576ad3bc780f109a2519f2df595f89f18a49b.jpg)

![](images/1d5ac92463f9337ba4325ca732d0208915230ec4781a2d5c4188876a7ca50113.jpg)  
Figure 5: Frozen budgets and matched inputs isolate the scheduling comparison within each model. (a) Native-only calibration fixes the row and token budgets before evaluation. Each policy receives the same 1551 questions in three regimes and two input orders. (b) Sender lag precedes actual submission; client latency starts at submission. The timeline is schematic, with no time scale.

The Qwen series uses a common offered trace of five requests per second with forward and reversed assignments. Reversal changes the assignment order. It does not let any policy inspect future inputs. The client has a finite submission pool, so actual sends can lag offered timestamps under overload. We record offered time, actual send time, and completion time to expose this backpressure. A cache hit does not remove queueing, and prior serving studies examine load imbalance and tail latency directly (Sun et al., 2024; Zhong et al., 2024).

## B.3 Native-only parameter selection

Calibration uses 64 question identifiers disjoint from evaluation, under three reuse conditions. All candidate measurements use native scheduling. RadixFlow results are never used to choose the configuration. The selected configuration, precision, model revision, tokenizer, and cache geometry are then fixed across all six policies.

The search adapts the official tuning flow (vLLM Contributors, 2026a). It uses real inputs instead of synthetic lengths and p95 instead of p99. We report this adaptation because a configuration tuned on a proxy workload can behave differently on long shared-prefix requests. A configuration that meets the calibration constraint may still exceed the same p95 on the longer replay.

## B.4 Pairing and exclusions

Within a model, all policies use the same GPU and workload definition. Across models, services may share host resources. A foreign GPU process entering a private experiment invalidates that attempt for timing comparisons. Only experiment-owned processes are stopped, and partial outputs are kept separately.

Completed arms are checked for identifiers and expected counts before aggregation. A failed or partial arm is never filled in from another run. Repeated traffic conditions are not additional independent questions.

## C Accounting and Timing Semantics

## C.1 Tokens and global rounds

For branch r, let I<sub>r</sub> be its prompt tokens and $H _ { r }$ its native reported cache-hit tokens, as in Equation 1. We report

$$
\rho = \frac { \sum _ { r } H _ { r } } { \sum _ { r } I _ { r } } , \qquad W = \sum _ { r } ( I _ { r } - H _ { r } ) + E .
$$

Here E is additional scheduled work identified from per-request intervals. It is zero in the earlier validated runs and is kept explicitly in the 27B ledger.

A worker round is one global worker invocation with scheduled model work. Empty executor calls for completion or cleanup are counted separately. Summing per-request visits or tensor-parallel rank calls would overcount global rounds. Cleanup calls are not deleted merely because they carry zero tokens, since they may have lifecycle duties. For a fixed row capacity, the unweighted bubble count is $\textstyle \sum _ { t } ( S - | B _ { t } | )$ . It counts slots rather than tokens or hardware use, so we use it only for the illustration in Figure 1.

## C.2 Three distinct clocks

Client latency measures the completion of the required branches after actual submission. Offeredto-completion time also captures sender lateness. Host-wrapper time measures a CPU-side call around worker execution, and asynchronous returns and enqueue gaps can separate it from GPU work. CUDA event intervals, where available, measure stream intervals and may include launch gaps.

These clocks are not interchangeable. CUDA timing guidance distinguishes asynchronous execution from properly ordered timing (NVIDIA, 2026a). We never add a CPU enqueue duration to a CUDA interval as if the two were disjoint. A drop in host-wrapper time is not a direct estimate of saved GPU time, especially when batch shapes or queue occupancy change.

## C.3 First-token edge cases

A generated token may decode to an empty string or a special symbol. A successful one-token response can therefore lack a first nonempty text event. Such a response is neither a timeout nor a zero-latency observation. Missing text-event times stay missing, and conditional TTFT summaries report their coverage. Client completion latency and token usage remain available and are checked independently.

## D Full Stream Model and Traversal Properties

Let V be the token vocabulary and $\begin{array} { r l } { [ n ] } & { { } = } \end{array}$ $\{ 1 , \ldots , n \}$ . The four-round layout is as short as capacity allows. The following bound makes this precise.

Proposition 1 (Capacity bound). Fix a hit assignment and a partition of the remaining work into fragments. Let W be their total tokens, V their total row visits, and D the length of their longest dependency chain in rounds. Every schedule π that realizes them satisfies

$$
K ( \pi ) \ge \operatorname * { m a x } \{ \left\lceil W / T \right\rceil , \left\lceil V / S \right\rceil , D \} .\tag{3}
$$

For Figure 1, $\begin{array} { l } { T \ = \ 4 0 9 6 } \end{array}$ and $S \ = \ 4 .$ so $\lceil 1 2 , 9 7 0 / T \rceil = 4$ and $\lceil 1 5 / S \rceil = 4$ . The longest illustrated dependency chain has two fragments. The feasible four-round layout therefore attains the bound for this fixed work. The proof is in $\mathsf { A p - }$ pendix E; general online streams also face arrival constraints.

Definition 3 (State readiness). A model fixes a set Σ of state components. Full attention has $\Sigma =$ {kv}, and a hybrid model has $\Sigma = \{ \mathrm { k v , r e c , c o n v } \}$ Let $\mathcal { C } _ { t }$ be the set of pairs $( p , \varsigma )$ such that component $\varsigma \in \Sigma$ of prefix p is published and passes the native state check at the start of round t. A boundary h is ready for branch r in round t if $( x _ { r , \leq h } , \varsigma ) \in$ $\mathcal { C } _ { t }$ for all $\varsigma \in \Sigma$ . The ready length $h _ { t } ( r )$ is the largest ready boundary, and $h _ { t } ( r ) = 0$ means cold execution.

The prefix index reports the longest token match $\hat { h } _ { t } ( r ) \geq h _ { t } ( r )$ . For full attention, the KV pages of a matched prefix are usually present, so $\hat { h } _ { t } ( r ) =$ $h _ { t } ( r )$ . In a hybrid model, $\hat { h } _ { t } ( r ) > h _ { t } ( r )$ whenever the recurrent component at the matched boundary is missing. State produced in round t enters $\mathcal { C } _ { t + 1 }$ , not $\mathcal { C } _ { t } .$ . For A, the gap between matched length $\hat { h } _ { t } ( A )$ and ready length $h _ { t } ( A )$ separates prefix identity from published state. It creates an avoidable bubble when the scan stops despite other feasible work.

Definition 4 (Prefill request stream). A stream is a finite sequence ${ \mathcal Q } \ : = \ : ( ( q _ { i } , a _ { i } ) ) _ { i \in [ n ] }$ with offered arrival times $a _ { 1 } \ \leq \ \cdots \ \leq \ a _ { n }$ . Request $q _ { i } ~ = ~ ( x _ { i 1 } , \ldots , x _ { i m _ { i } } )$ holds $m _ { i }$ input branches $x _ { i j } \in \mathcal { V } ^ { * }$ . Let $\mathcal { R } = \{ ( i , j ) : i \in [ n ] , j \in [ m _ { i } ] \}$ be the set of branches. For $r = ( i , j )$ we write $x _ { r } = x _ { i j }$ and $a _ { r } = a _ { i }$ . An online policy observes branch r only after time $a _ { r }$

A branch is one input encoding. In Figure 1, P and A are branches with a shared prefix; our replay groups four branches into each logical request.

Definition 5 (Feasible batch and schedule). Let $\mathcal { A } _ { t } \subseteq \mathcal { R }$ be the branches that have arrived, have been released by admission by round $t ,$ and still have unscheduled tokens. A batch $B _ { t } \subseteq A _ { t }$ assigns each $r \in B _ { t }$ a token count $n _ { t } ( r ) \geq 1$ . It is feasible if (i) $\begin{array} { r } { | \mathcal { B } _ { t } | \le S , \mathrm { ( i i ) } \sum _ { r \in \mathcal { B } _ { t } } n _ { t } ( r ) \le T } \end{array}$ , and (iii) every $r \in B _ { t }$ resumes from a boundary that is ready in round t or from its own state of an earlier round. A schedule $\pi = ( B _ { 1 } , \ldots , B _ { K } )$ is a sequence of nonempty feasible batches that covers every token of every branch. Its round count is $K ( \pi )$

Condition (iii) prevents A from using state that $P$ is still producing. An independent branch such as $D$ can join P if the row and token budgets allow it.

Properties. Within a round, $\mathcal { C } _ { t }$ is fixed and the residual budgets only shrink as $\boldsymbol { B }$ grows. Adding a branch can also create a new producer. The evaluated synchronous scheduler publishes no completed worker state during this scan, and its currentstep rejection remains active until a later step. Thus a blocked A does not become selectable as more branches join the partial batch. We state the corresponding monotonicity requirement explicitly below; additional backend outcomes must satisfy it for part (iii) of the proposition.

Assumption 1 (Monotone readiness). For any round t and partial batches $B \subseteq B ^ { \prime }$ , if $\sigma _ { t } ( r \mid B ^ { \prime } ) =$ READY, then $\sigma _ { t } ( r \mid \ r _ { B } ) = \mathrm { R E A D Y }$

Proposition 2 (Properties of state-ready traversal). Fix round $t ,$ the state $\mathcal { C } _ { t } ,$ the eligible set, and the order ≺. Let $\boldsymbol B _ { t } ^ { \mathrm { s t o p } }$ and $B _ { t } ^ { \mathrm { R F } }$ be the batches built by prefix-stop and by RadixFlow. Then:

(i) $B _ { t } ^ { \mathrm { R F } }$ is feasible.

(ii) $\boldsymbol { B } _ { t } ^ { \mathrm { s t o p } } \subseteq \boldsymbol { B } _ { t } ^ { \mathrm { R F } }$ , hence $b _ { t } ^ { \mathrm { R F } } \leq b _ { t } ^ { \mathrm { s t o p } }$

(iii) Under Assumption 1, if no OTHER outcome ends the scan, round t has no avoidable bubble under RadixFlow.

The proof is in Appendix E. Part (i) is the safety guarantee: blocking a consumer changes selection without accepting state rejected by the native state check. Parts (ii) and (iii) characterize the bubbles recovered by continuing past BLOCKED outcomes, under the stated conditions. The comparison holds for a fixed state. Over a stream, the two policies reach different states, so the total round count is measured in Section 5.

## E Proofs

Proof of Proposition 1. Each round schedules at most $T$ tokens, so $K ( \pi ) T \geq W$ and $K ( \pi ) \geq$ $\lceil W / T \rceil$ . Each row visit occupies one of at most S rows in one round, so $K ( \pi ) ~ \ge ~ \lceil V / S \rceil$ . Consider two consecutive fragments on a dependency chain. The later one resumes from state that the earlier one produces. By Definition 3, that state enters $\mathcal { C } _ { t + 1 }$ when the earlier fragment runs in round t. Condition (iii) of Definition 5 then places the later fragment in round $t + 1$ or after. A chain of D fragments thus needs D distinct rounds, so $K ( \pi ) \geq D$ □

Proof of Proposition 2. (i) RadixFlow starts from $B = \varnothing$ , which is feasible. Algorithm 1 adds a branch only at line 7; line 9 changes the wait queue without adding work. It adds r only when $\sigma _ { t } ( r \mid \ r _ { B } ) = \mathrm { R E A D Y }$ , which by Definition 1 means that $B \cup \{ r \}$ is feasible. A BLOCKED branch is never added, and the native action for OTHER never adds a branch that fails the native state check. By induction over the scan, every partial batch is feasible, and so is $B _ { t } ^ { \mathrm { R F } }$

(ii) Let $r ^ { \star }$ be the first branch in ≺ whose outcome under prefix-stop is not READY. Before $r ^ { \star }$ both scans see the same partial batch at every step, obtain the same outcomes, and receive the same token counts from the native allocator. They therefore hold the same partial batch $B ^ { \star }$ when they reach $r ^ { \star }$ , and prefix-stop returns $\boldsymbol { B } _ { t } ^ { \mathrm { s t o p } } = \boldsymbol { B } ^ { \star }$ . If the outcome of $r ^ { \star }$ is OTHER, RadixFlow takes the same native action, and the two batches coincide. If it is BLOCKED, RadixFlow continues and only adds branches, so $B ^ { \star } \subseteq B _ { t } ^ { \mathrm { R F } }$ . If no such $r ^ { \star }$ exists, both scans select every eligible branch. In all cases $B _ { t } ^ { \mathrm { s t o p } } \subseteq B _ { t } ^ { \mathrm { R F } }$ , and $b _ { t } = S - | B _ { t } |$ gives $b _ { t } ^ { \mathrm { R F } } \leq b _ { t } ^ { \mathrm { s t o p } }$

(iii) Assume no OTHER outcome ends the scan, so RadixFlow visits every branch of $\boldsymbol { A } _ { t }$ . Take any $r \in \mathcal { A } _ { t } \backslash B _ { t } ^ { \mathrm { R F } }$ , and let $B _ { r }$ be the partial batch when r was visited. Since r was not added, $\sigma _ { t } ( r \mid B _ { r } ) \neq$ READY. Partial batches only grow, so $B _ { r } \subseteq B _ { t } ^ { \mathrm { R F } }$ By Assumption $1 , \sigma _ { t } ( r \mid \bar { \mathcal { B } } _ { t } ^ { \mathrm { R F } } ) \neq \mathrm { R }$ EADY. No branch outside the batch is READY, so by Definition 2 the round has no avoidable bubble. □

## F Native State Contract

## F.1 Native prefix-stop path

The reference path is vLLM 0.29.0, not a synthetic queue policy. In Scheduler.schedule, lines 991– 1011 of vllm/v1/core/sched/scheduler.py allocate waiting-request blocks and end the waiting-queue scan when allocation fails (vLLM Contributors, 2026b). In MambaManager.get\_num\_blocks\_to\_allocate, lines 1476–1485 reject reuse of a block hash recorded in cached\_blocks\_this\_step by reporting a requirement above pool capacity (vLLM Contributors, 2026c). The same set is cleared at the next step in lines 1729–1730. This is the specific same-round dependency RadixFlow recognizes. Other native paths can already skip work for different reasons, and remain unchanged. All policies share eager execution, disabled asynchronous scheduling, prefix caching, and

native state checks.

All policies share request registration, cache namespace isolation, component-generation checks, and the same verified copy primitive. The state check distinguishes published components from a producer’s in-flight state; a blocked consumer stays registered and is woken for a later native recheck.

## G Anonymous Reproducibility Record

The review manuscript uses the public model names, benchmark sources, and hardware descriptions needed to interpret the experiments. It omits author identities, affiliations, private repository URLs, hostnames, account names, absolute paths, and internal experiment identifiers. Development failures are described by mechanism rather than by chronology.

An anonymized release will include pinned model and data revisions, input and source hashes, frozen configurations, policy definitions, offered traces, per-arm identifier counts, worker and cleanup-call counts, and token-accounting summaries. Logs will be scrubbed of credentials and machine-specific paths before release.

## H Timing and Extension Protocol

Component profiling aligns request IDs with worker rounds and separates CPU lookup, selection, dispatch, GPU restoration, prefill, and publication. Exclusive intervals preserve overlap rather than double-counting stages. Load sweeps at matched fractions of native capacity, fixed-row scale comparisons, and wider branching are natural extensions of the reported frozen replay.

## I 27B Token-Accounting Ledger

All six 27B policies completed 9306 logical requests each. The initial summary rejected the identity $\begin{array} { r } { W = \sum _ { r } ( I _ { r } - H _ { r } ) } \end{array}$ because it omitted scheduled revisits. We kept the original outputs and rebuilt the interval ledger from actual executor-call traces.

Each request’s first recorded start equals its reported cache-hit length. Its intervals stay within the prompt and reach the final prompt position without gaps. The union of intervals measures uniquely scheduled input positions. Overlapping intervals measure repeated scheduled work. Two Radix+Timer10 branches first start after a cached prefix and then return to position zero. Their reset work accounts for the remaining discrepancy. For every request and every arm, we verify

<table><tr><td>Policy</td><td>Overlap</td><td>Reset</td><td>Total extra</td></tr><tr><td>Radix</td><td>66,944</td><td>0</td><td>66,944</td></tr><tr><td>Radix+Timer5</td><td>49,584</td><td>0</td><td>49,584</td></tr><tr><td>Radix+Timer10</td><td>53,760</td><td>512</td><td>54,272</td></tr><tr><td>Radix+Flex</td><td>48,864</td><td>0</td><td>48,864</td></tr><tr><td>RadixFlow</td><td>182,384</td><td>0</td><td>182,384</td></tr><tr><td>RadixFlow+Frag</td><td>186,096</td><td>0</td><td>186,096</td></tr></table>

Table 3: Additional scheduled tokens in the 27B replay. Worker calls and token totals are kept rather than deduplicated.

$$
W = \sum _ { r } ( I _ { r } - H _ { r } ) + E _ { \mathrm { o v e r l a p } } + E _ { \mathrm { r e s e t } } .
$$

This ledger reconciles reported hits with actual scheduled intervals (Table 3). RadixFlow’s overlap accounts for 0.39% of its scheduled tokens, versus 0.14% for Radix. Thus reported prefix reuse and additional scheduled work remain separate quantities. The traces identify repeated intervals and the two reset prefixes; attributing their cause to preemption would require native preemption events. Native reported token reuse, uniquely skipped input positions, and repeated scheduling are recorded separately. Identifier coverage, one-token completions, interval continuity, and GPU ownership checks pass for all arms.

## J Saved-Trace Mechanism and Cost Analysis

We reanalyze the completed runs without additional model inference. Each worker record is joined to measured requests by branch identifier. Functionality smoke requests are excluded. We verify the row and token budgets for every joined round. Logical requests are counted once per policy, regime, and input order. Table 4 separates admission from traversal; Table 5 reports all policies’ latency distributions. Figure 6 makes the regime dependence visible: the largest latency gain need not coincide with the largest round reduction.

Percentiles in this appendix use linear interpolation of the empirical order statistics. Client latency is submission to completion of the four branches. Sender lag is offered arrival to submission. Offeredarrival latency is their sum for each request, before computing a percentile. Percentiles of the two parts are not additive.

![](images/f827650ec38d29235254ef8098f4c29c2bc2b77501c2e951687870ce5ba5cb9b.jpg)  
Figure 6: Round reductions and latency gains respond differently to the reuse regime. Each cell compares RadixFlow with Radix on the same model and stream, pooling both input orders. Blue indicates a reduction; orange indicates a regression. The panels use the same percentage scale; cold prevents sharing across questions, not within one question. Values summarize the frozen replays in Table 4.

## J.1 Batch shape and cost association

We fit $\begin{array} { r c l } { t _ { k } } & { = } & { \alpha \ + \ \beta ( n _ { k } / 1 0 0 0 ) \ + \ \gamma v _ { k } \ + } \end{array}$ δ1[RadixFlow] to nonempty rounds from Radix, Radix+Flex, and RadixFlow. Here $n _ { k }$ is scheduled tokens and $v _ { k }$ is row visits. Table 6 reports the fit alongside mean batch shape. Cache lengths, copied bytes, kernel shapes, and temporal host load are omitted variables. The adjusted coefficient δ summarizes policy association under this batch-shape model; component profiling separates its underlying costs.

The native timestamps provide a second view of waiting (Table 7). Figure 7 places the two interval changes on a common within-model scale. On Qwen3.5-27B, mean registration-to-first-selection time falls by 7.22 s, while the remaining interval grows by 1.26 s. On Qwen3.5-9B, the corresponding changes are a 2.43 s reduction and a 0.35 s increase. This supports a concrete trade-off: requests enter execution sooner, but denser rounds extend their subsequent service interval. The latter includes waiting as well as execution and cannot isolate restoration cost.

## J.2 Capacity gap and counter scope

For a fixed policy, stream, and order, $L _ { \mathrm { c a p } } ~ =$ max $( \lceil W / T \rceil , \lceil V / S \rceil )$ relaxes arrivals, alignment, and state dependencies. W includes repeated scheduled tokens and V includes every scheduled fragment. These quantities can change with the policy, so this diagnostic is not a shared global optimum. The bound becomes loose when highly reused short fragments cannot arrive or become ready together. Likewise, unused row slots and token budget measure residual capacity, not automatically avoidable bubbles or GPU idle time.

The saved state-check counters are cumulative visits, including smoke traffic and repeated scans. They establish that the recurrent check is active on hybrid models and absent on the full-attention control. They do not identify the blocked and recovered rows in each worker round. Per-request blocking counts are not recorded. A future trace must capture the dependency, the skip, and the recovered row in the same batch before attributing a specific saved round to a specific bypass or analyzing starvation.

## J.3 Scale and load sensitivity

A plausible scale effect is that longer worker calls let more requests accumulate, exposing more consumers of current-round producers. However, row capacity also changes from 16 to 64 and the offered load queues larger models more heavily. Per-round dependency and recovered-row traces would distinguish these factors.

## J.4 Dataset and order sensitivity

The workload uses article-level sharing, so arrival order changes both realized reuse and the executable set. We retain both orders in every primary comparison rather than selecting the faster one. Dataset-level latency summaries can be obtained by request identifiers, but global rounds cannot be assigned uniquely to QuALITY or RACE when a batch contains both. We therefore report global round counts at the stream level, where batch membership is unambiguous. Figure 8 shows the latency reduction in each stratum. Both orders favor RadixFlow on each hybrid model. The fullattention control changes sign across orders, with small effects in both datasets, consistent with its small aggregate difference.

<table><tr><td>Model</td><td>Stream</td><td>Radix K</td><td>Flex K</td><td>RadixFlow K</td><td>Latency reduction (%)</td><td> $K / L _ { \mathrm { c a p } }$ </td></tr><tr><td rowspan="3">Qwen3.5-0.8B</td><td>Cold</td><td>6,849</td><td>6,563</td><td>6,152</td><td>7.1</td><td>3.33</td></tr><tr><td>Mixed</td><td>6,603</td><td>6,101</td><td>5,884</td><td>8.7</td><td>3.94</td></tr><tr><td>Reuse</td><td>6,351</td><td>5,745</td><td>5,574</td><td>13.2</td><td>3.88</td></tr><tr><td rowspan="3">Qwen3.5-4B</td><td>Cold</td><td>4,477</td><td>4,382</td><td>2,753</td><td>10.7</td><td>1.49</td></tr><tr><td>Mixed</td><td>3,866</td><td>3,805</td><td>2,905</td><td>61.2</td><td>2.56</td></tr><tr><td>Reuse</td><td>5,081</td><td>4,755</td><td>4,488</td><td>12.6</td><td>7.71</td></tr><tr><td rowspan="3">Qwen3.5-9B</td><td>Cold</td><td>4,389</td><td>4,318</td><td>2,666</td><td>4.8</td><td>1.44</td></tr><tr><td>Mixed</td><td>3,336</td><td>3,192</td><td>1,984</td><td>7.1</td><td>1.68</td></tr><tr><td>Reuse</td><td>3,317</td><td>3,260</td><td>2,794</td><td>35.5</td><td>4.18</td></tr><tr><td rowspan="3">Qwen3.5-27B</td><td>Cold</td><td>4,044</td><td>3,998</td><td>2,256</td><td>2.8</td><td>1.13</td></tr><tr><td>Mixed</td><td>3,874</td><td>3,820</td><td>2,172</td><td>2.9</td><td>1.14</td></tr><tr><td>Reuse</td><td>3,830</td><td>3,737</td><td>2,101</td><td>3.5</td><td>1.14</td></tr><tr><td rowspan="3">Nemotron-9B</td><td>Cold</td><td>4,821</td><td>4,045</td><td>3,332</td><td>13.6</td><td>1.59</td></tr><tr><td>Mixed</td><td>4,576</td><td>3,360</td><td>3,248</td><td>1.9</td><td>2.37</td></tr><tr><td>Reuse</td><td>4,941</td><td>3,050</td><td>3,033</td><td>5.9</td><td>3.47</td></tr><tr><td rowspan="3">Qwen3-8B (FA)</td><td>Cold</td><td>2,449</td><td>2,091</td><td>2,090</td><td>-0.2</td><td>1.39</td></tr><tr><td>Mixed</td><td>2,268</td><td>1,732</td><td>1,723</td><td>0.6</td><td>1.89</td></tr><tr><td>Reuse</td><td>4,506</td><td>2,959</td><td>2,933</td><td>5.1</td><td>3.76</td></tr></table>

Table 4: Regime analysis, pooling both orders. Latency reduction compares RadixFlow with Radix. Flex denotes Radix+Flex. $L _ { \mathrm { c a p } }$ sums the separate capacity bounds for the two orders using RadixFlow’s observed work. Cold prevents sharing across questions, not within a question.

## K Extended Related Work

Serving research organizes four decisions: what state to keep, where to place requests, when to execute them, and how to compute a batch. Our focus is eligibility within batch construction: which matched branches can run now?

Keeping reusable state. Prompt Cache reuses predefined prompt modules (Gim et al., 2024), while Pensieve preserves conversation state across memory tiers (Yu et al., 2023). CachedAttention overlaps cache transfer with computation (Gao et al., 2024), and LMCache supports cache movement between tiers (Liu et al., 2025). These approaches make reuse more available. RadixFlow instead schedules the work left after the native state check.

Keeping complete hybrid state. Mamba, Mamba-2, and Gated Delta Networks require recurrent state at each reuse boundary (Gu and Dao, 2023; Dao and Gu, 2024; Yang et al., 2024). Marconi balances expected reuse and compute saving against hybrid-state memory cost (Pan et al., 2025a); SGLang’s unified cache tracks the components needed at a boundary (Zhang and SGLang Team, 2026). These policies concern state retention. RadixFlow uses the native state check (vLLM Contributors, 2026c) to distinguish published state from a same-round dependency during selection.

Placing and ordering requests. Orca schedules at iteration level (Yu et al., 2022), and Sarathi-Serve balances prefill and decode through chunking (Agrawal et al., 2024). DistServe and Splitwise place the two phases on separate resources (Zhong et al., 2024; Patel et al., 2023); FastServe and Llumnix adapt execution through preemption or migration (Wu et al., 2023; Sun et al., 2024). Prefixaware Preble and k-LPM balance reuse against load and waiting (Srivatsa et al., 2025; Dexter et al., 2025). RadixFlow preserves the native request order and changes selection within one round when a same-round dependency is encountered.

Forming and computing batches. BatchLLM combines prefix sharing with token batching (Zheng et al., 2026), while FastTree couples grouping with tree-aware kernels (Pan et al., 2025b).

<table><tr><td>Model</td><td>Policy</td><td>p50 (s)</td><td>p95 (s)</td><td>p99 (s)</td><td>Sender lag (s)</td><td>Offered latency (s)</td></tr><tr><td>Qwen3.5-0.8B</td><td>Radix</td><td>0.217</td><td>0.503</td><td>0.767</td><td>0.000</td><td>0.256</td></tr><tr><td rowspan="12">Qwen3.5-4B</td><td>Radix+Timer5</td><td>0.225</td><td>0.512</td><td>0.842</td><td>0.000</td><td>0.261</td></tr><tr><td>Radix+Timer10</td><td>0.227</td><td>0.526</td><td>0.828</td><td>0.000</td><td>0.265</td></tr><tr><td>Radix+Flex</td><td>0.210</td><td>0.521</td><td>0.798</td><td>0.000</td><td>0.249</td></tr><tr><td>RadixFlow</td><td>0.201</td><td>0.455</td><td>0.630</td><td>0.000</td><td>0.232</td></tr><tr><td>RadixFlow+Frag</td><td>0.200</td><td>0.443</td><td>0.606</td><td>0.000</td><td>0.228</td></tr><tr><td>Radix</td><td>4.525</td><td>47.300</td><td>48.576</td><td>15.943</td><td>29.067</td></tr><tr><td>Radix+Timer5</td><td>0.901</td><td>96.417</td><td>150.829</td><td>14.059</td><td>25.819</td></tr><tr><td>Radix+Timer10</td><td>0.900</td><td>93.226</td><td>151.225</td><td>14.008</td><td>25.519</td></tr><tr><td>Radix+Flex</td><td>2.656</td><td>47.658</td><td>48.840</td><td>16.205</td><td>28.996</td></tr><tr><td>RadixFlow</td><td>1.666</td><td>43.718</td><td>44.928</td><td>11.875</td><td>21.907</td></tr><tr><td>RadixFlow+Frag</td><td>1.274</td><td>45.873</td><td>47.405</td><td>13.131</td><td>23.265</td></tr><tr><td>Qwen3.5-9B Radix</td><td>12.849</td><td>73.326</td><td>75.169</td><td>60.581</td><td>87.248</td></tr><tr><td></td><td>Radix+Timer5</td><td>2.316</td><td>155.450</td><td>254.673</td><td>60.875</td><td>87.404</td></tr><tr><td></td><td>Radix+Timer10</td><td>2.290</td><td>158.283</td><td>251.954</td><td>59.667</td><td>86.191</td></tr><tr><td></td><td>Radix+Flex</td><td>14.205</td><td>73.509</td><td>75.208</td><td>59.995</td><td>86.461</td></tr><tr><td></td><td>RadixFlow</td><td>9.978</td><td>70.605</td><td>72.574</td><td>54.651</td><td>79.425</td></tr><tr><td>Qwen3.5-27B</td><td>RadixFlow+Frag</td><td>10.045</td><td>72.476</td><td>74.510</td><td>55.190</td><td>80.047</td></tr><tr><td></td><td>Radix</td><td>229.697</td><td>252.283</td><td>256.890</td><td>721.476</td><td>885.663</td></tr><tr><td></td><td>Radix+Timer5</td><td>11.554</td><td>769.029</td><td>970.131</td><td>742.837</td><td>911.084</td></tr><tr><td></td><td>Radix+Timer10</td><td>11.273</td><td>911.466</td><td>1014.259</td><td>743.882</td><td>912.454</td></tr><tr><td></td><td>Radix+Flex</td><td>230.947</td><td>252.283</td><td>256.511</td><td>717.981</td><td>881.401</td></tr><tr><td></td><td>RadixFlow</td><td>223.046</td><td>246.640</td><td>251.870</td><td>696.686</td><td>855.808</td></tr><tr><td>Nemotron-9B</td><td>RadixFlow+Frag</td><td>199.937</td><td>253.765</td><td>262.483</td><td>653.534</td><td>804.627</td></tr><tr><td></td><td>Radix</td><td>0.420</td><td>15.818</td><td>18.789</td><td>0.000</td><td>2.438</td></tr><tr><td></td><td>Radix+Timer5</td><td>0.393</td><td>12.128</td><td>40.586</td><td>0.000</td><td>2.407</td></tr><tr><td></td><td>Radix+Timer10</td><td>0.396</td><td>11.799</td><td>39.126</td><td>0.000</td><td>2.378</td></tr><tr><td></td><td>Radix+Flex RadixFlow</td><td>0.395</td><td>15.962</td><td>18.916</td><td>0.000</td><td>2.432</td></tr><tr><td></td><td></td><td>0.391</td><td>13.275</td><td>16.133</td><td>0.000</td><td>2.146</td></tr><tr><td>Qwen3-8B (FA)</td><td>RadixFlow+Frag</td><td>0.392</td><td>13.492</td><td>16.949</td><td>0.000</td><td>2.111</td></tr><tr><td></td><td>Radix</td><td>3.235</td><td>54.603</td><td>55.425</td><td>24.775</td><td>41.198</td></tr><tr><td></td><td>Radix+Timer5</td><td>1.754</td><td>94.325</td><td>117.654</td><td>25.229</td><td>41.910</td></tr><tr><td></td><td>Radix+Timer10</td><td>1.657</td><td>97.192</td><td>104.945</td><td>24.484</td><td>40.843</td></tr><tr><td></td><td>Radix+Flex</td><td>3.594</td><td>54.596</td><td>55.342 55.402</td><td>24.743</td><td>41.086</td></tr><tr><td></td><td>RadixFlow RadixFlow+Frag</td><td>3.631 3.579</td><td>54.638 54.588</td><td>55.420</td><td>24.868 24.882</td><td>41.278</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>41.291</td></tr></table>

Table 5: Latency distribution and mean sender lag on all measured requests. The last two columns are means, not tail quantiles. The offered rate differs between Qwen and Nemotron-9B.

Dynamic batching collects work under a waiting bound (NVIDIA, 2026b), which our timer controls isolate. FlashAttention, FlashAttention-2, Flash-Infer, and vAttention improve the computation or layout of the resulting batch (Dao et al., 2022; Dao, 2023; Ye et al., 2025; Prabhu et al., 2024). Because these decisions change per-round cost, we measure that cost rather than equating fewer rounds with proportional speedup.

## L Additional Waiting and Capacity Analysis

The full-attention Qwen3-8B (FA) isolates admission from the recurrent check. The check never fires, and Radix+Flex and RadixFlow have almost the same round count in every stream. Their aggregate reduction relative to Radix is about 27%, with little change in mean latency. Its p50 instead rises from 3.24 to 3.63 s under RadixFlow, so fewer rounds are not uniformly better for requests. Nemotron-9B exposes a different balance: admission removes 27% of the rounds, and traversal removes another 8% relative to Radix+Flex. At its lower offered load, Radix starts with only 2.91 rows per round out of 16; Flex raises this to 3.99 before traversal raises it to 4.38. This pattern is consistent with admission filling sparsely occupied rounds. Its host cost stays nearly constant, which allows the reduced round count to translate more directly into time saving. Appendix J gives the regime and distribution summaries.

Equation 2 separates the credit for eliminating rounds from the cost shift in the remaining rounds. We use Radix as index 0 and RadixFlow as index 1. On Qwen3.5-9B, the cost shift absorbs about 60% of the gross credit (Figure 3). On Nemotron-9B, per-round cost changes little, so more of the round reduction survives as a reduction in total host-wrapper time. A shape-adjusted analysis of tokens and rows follows the same pattern: batch shape explains much of the Qwen variation, with smaller residual policy associations (Appendix J). Direct attribution to state restoration or queue scanning requires the component measurements in Appendix H.

<table><tr><td>Model</td><td colspan="2">Tokens/round</td><td colspan="2">Rows/round</td><td>δ(ms)</td><td> $R ^ { 2 }$ </td></tr><tr><td></td><td>Radix</td><td>RadixFlow</td><td>Radix</td><td>RadixFlow</td><td></td><td></td></tr><tr><td>Qwen3.5-0.8B</td><td>1401</td><td>1575</td><td>3.63</td><td>4.10</td><td>0.51</td><td>0.328</td></tr><tr><td>Qwen3.5-4B</td><td>2173</td><td>2871</td><td>5.42</td><td>7.31</td><td>2.70</td><td>0.911</td></tr><tr><td>Qwen3.5-9B</td><td>2739</td><td>4060</td><td>6.61</td><td>10.02</td><td>3.05</td><td>0.957</td></tr><tr><td>Qwen3.5-27B</td><td>3998</td><td>7208</td><td>6.51</td><td>12.11</td><td>18.41</td><td>0.997</td></tr><tr><td>Nemotron-9B</td><td>2482</td><td>3701</td><td>2.91</td><td>4.38</td><td>-0.61</td><td>0.238</td></tr><tr><td>Qwen3-8B (FA)</td><td>2328</td><td>3181</td><td>4.28</td><td>5.86</td><td>0.12</td><td>0.232</td></tr></table>

Table 6: Batch-shape changes and descriptive host-cost regression. Separate fits are used for each model.
<table><tr><td>Model</td><td colspan="2">Registration to first selection (s)</td><td colspan="2">After first selection (s)</td></tr><tr><td></td><td>Radix</td><td>RadixFlow</td><td>Radix</td><td>RadixFlow</td></tr><tr><td>Qwen3.5-0.8B</td><td>0.057</td><td>0.049</td><td>0.120</td><td>0.124</td></tr><tr><td>Qwen3.5-4B</td><td>12.576</td><td>9.273</td><td>0.373</td><td>0.528</td></tr><tr><td>Qwen3.5-9B</td><td>25.682</td><td>23.255</td><td>0.664</td><td>1.013</td></tr><tr><td>Qwen3.5-27B</td><td>160.538</td><td>153.314</td><td>2.457</td><td>3.718</td></tr><tr><td>Nemotron-9B</td><td>1.898</td><td>1.582</td><td>0.347</td><td>0.396</td></tr><tr><td>Qwen3-8B (FA)</td><td>15.817</td><td>15.813</td><td>0.322</td><td>0.325</td></tr></table>

Table 7: Mean native branch intervals on all 37,224 measured branches per policy. First selection is the first scheduler selection, not GPU execution start. The second interval includes subsequent rounds, intervening waiting, and completion processing. These native branch intervals are not directly additive to logical client latency.

Waiting and tail latency. Client latency includes service-side waiting. Offered-arrival latency also includes sender lag when the client pool is full. For Qwen3.5-27B, RadixFlow lowers mean sender lag from 721 to 697 s, in addition to lowering postsend latency from 164 to 159 s. These are distinct waiting intervals, and their tail quantiles are measured on each request rather than added together. On Qwen3.5-9B, p95 falls from 73.3 to 70.6 s and p99 from 75.2 to 72.6 s; on Nemotron-9B, they fall from 15.8 to 13.3 s and from 18.8 to 16.1 s. The full-attention control has nearly unchanged tails; Appendix J covers all six policies.

How close are the rounds to the capacity bound? For each stream and order, we compare actual rounds with max(⌈W/T⌉, ⌈V /S⌉), using that policy’s observed scheduled tokens W and row visits V . On Qwen3.5-27B, RadixFlow reaches 1.13– 1.14 times this capacity-only bound across the three streams. On Qwen3.5-9B, the ratio is 1.44 in cold traffic but 4.18 in reuse traffic. High reuse leaves little token work, yet alignment, arrival times, and state dependencies still require rounds. The bound relaxes these constraints rather than describe an attainable online schedule.

<table><tr><td>Model Decoded agreement (%)</td></tr><tr><td>Qwen3.5-0.8B 98.477</td></tr><tr><td>Qwen3.5-4B 99.850</td></tr><tr><td>Qwen3.5-9B 99.680</td></tr><tr><td>Qwen3.5-27B 99.995</td></tr><tr><td>Nemotron-9B 99.688</td></tr><tr><td>Qwen3-8B (FA) 99.441</td></tr></table>

Table 8: Exact decoded first-token agreement between RadixFlow and Radix on 37,224 paired branches per model.

## M Worker Cost and Output Audit

The saved HTTP responses record one completed output token per branch. We compare decoded text exactly, with no normalization of whitespace or special-token text. Every policy has 37,224 paired branches per model. Agreement with Radix ranges from 98.477% to 99.995% for RadixFlow; the audit retains differing strings. The audit quantifies sensitivity to the scheduling configuration. Repeatversus-repeat controls would separate native variability from policy effects. A downstream quality comparison uses candidate likelihoods, a fixed tie rule, and gold answers; the present text-agreement metric measures serving-output stability.

Earlier selection offsets a longer remaining interval  
![](images/bfe42d2857edaf643031ac681672652359da70e334d7181565ef3ce5140eb2a6.jpg)

Figure 7: Earlier selection offsets a longer interval after selection on the hybrid models. Points show the RadixFlowminus-Radix change in each interval, normalized by the sum of the two Radix mean intervals. Labels give absolute changes in seconds, calculated from the rounded means in Table 7. The later interval includes intervening waiting and completion processing as well as execution; these are native branch intervals, not logical client latency.
<table><tr><td>Model</td><td>Radix</td><td>Timer5</td><td>Timer10</td><td>Flex</td><td>RadixFlow</td><td>+Frag</td></tr><tr><td>Qwen3.5-0.8B</td><td>52.95</td><td>52.99</td><td>53.09</td><td>53.64</td><td>54.24</td><td>53.94</td></tr><tr><td>Qwen3.5-4B</td><td>82.63</td><td>83.26</td><td>84.14</td><td>83.42</td><td>92.95</td><td>91.69</td></tr><tr><td>Qwen3.5-9B</td><td>112.85</td><td>116.78</td><td>116.67</td><td>113.38</td><td>145.50</td><td>144.16</td></tr><tr><td>Qwen3.5-27B</td><td>729.94</td><td>829.01</td><td>836.45</td><td>738.22</td><td>1262.95</td><td>1246.66</td></tr><tr><td>Nemotron-9B</td><td>53.06</td><td>53.82</td><td>53.10</td><td>53.69</td><td>53.87</td><td>54.31</td></tr><tr><td>Qwen3-8B (FA)</td><td>26.60</td><td>27.55</td><td>27.73</td><td>27.42</td><td>27.54</td><td>27.42</td></tr></table>

Table 9: Mean host-wrapper milliseconds per nonempty global worker round, retained for all six methods.

## N Baseline Audit and Replay Sensitivity

Common overlay versus stock vLLM. Radix is the native-selection arm of the common experimental backend. All six arms execute registration/index hooks and tracing, and use the same copy-coverage guard. Their paired differences measure scheduling on a common backend. A stock vLLM+APC arm must remove the overlay entirely while retaining the same model, input tokens, arrival trace, and resource settings. The table reports the six common-backend policies named in its rows.

Alternative execution structures. Producer-first grouping executes a common producer before admitting its consumers, exchanging online packing freedom for an explicit group dependency. SGLang’s prefix/fork programming model is another relevant comparison (Zheng et al., 2024; Zhang and SGLang Team, 2026). These alternatives require matched producer cost, cache capacity, and request-completion clocks, rather than comparing only consumer latency. These execution structures form a separate comparison axis from the six scheduling controls.

Timer eligibility and order. Timer5/10 register each branch before gating. A branch carries a deadline; peers are released when their earliest deadline expires or the row capacity is reached. The scheduler temporarily removes held branches and prepends them back after scanning executable work. In addition, the hold test can release peer timers while constructing the held list. Thus the control changes effective eligibility and can let a later branch execute before an earlier held one. The timer control therefore implements bounded admission with dynamic eligibility. An order-preserving timer control should snapshot eligibility before selection and record overtakes and deferral counts.

The 27B Timer10 tail split appears within every stream and both orders: p50 ranges from 10.53 to 11.80 s and p99 from 910.83 to 1104.77 s. It is therefore not created solely by pooling different streams. All requests completed. The code audit identifies eligibility changes consistent with the split; starvation attribution would require perrequest deferral events. The reported Timer measurements use this eligibility rule.

![](images/7053a5f9b9d5da2215dee0af73e0a586f9afc162e667b61013fd9f0f389b89ec.jpg)

![](images/72876ca2449b57c485ad29bfbd49dc4b0c1c300ad299306cd58ce87bb816bbef.jpg)  
Figure 8: Latency improves in both input orders and both datasets on every hybrid model. Points show the percentage reduction in mean client latency for RadixFlow relative to Radix, pooling the other conditions. Connectors pair the two strata within a model; they are not uncertainty intervals. Forward and reverse are different input assignments within the frozen evaluation.

The 4B mixed-stream result. The 61.2% mean reduction persists in both assignment orders. Forward latency changes from 10.31 to 3.30 s (68.0%); reverse changes from 9.78 to 4.50 s (54.0%). The two orders are not independent timing repetitions. The benefit appears in both assignments, with its magnitude changing as the workload order changes.

Operating points and uncertainty. The 27B rate of 5 requests/s exceeds its observed draininclusive completion rate near 0.74 requests/s, producing sustained overload and sender backpressure. We use saturation here to describe that operating condition; the replay rate is not a loadsweep estimate of a throughput ceiling. Post-send and offered-to-completion latency are separate outcomes. Between-model comparisons also change row capacity and sometimes offered rate. The figure presents configurations, not a causal model-size trend.

Each frozen condition is run once. Forward/reverse order comparisons measure assignment sensitivity, not run-to-run variance. Host resources may be shared across services. Fixedrow, isolated-host repetitions at 50/80/95% of independently measured native capacity would provide uncertainty for the small effects and establish a production operating envelope. CUDA-graph and asynchronous scheduling require separate matched runs; an asynchronous dependency must be identified by producer generation and completion fence, rather than a synchronous step number alone.