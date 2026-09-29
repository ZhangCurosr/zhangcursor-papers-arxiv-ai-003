# TMCS: Tool-Grounded Multi-Agent Reasoning for Compositional Chemical Problem Solving

Shengqin Wang<sup>∗</sup>, Jie Jin<sup>∗</sup>, Yu Cheng, Yihang Chen, Weilin Luo, Yuan Xie, Zhizhong Zhang

East China Normal University, Shanghai Innovation Institute, University College London, Huawei Noah’s Ark Lab

## Abstract

Despite the promise of Large Language Models (LLMs) in computational chemistry, rigorous combinatorial chemistry problems remain dificult because they require quantitatively constrained molecular modification, candidate validation, and systematic revision after failed attempts. Existing tool-augmented chemical agents demonstrate useful planning and tool use, but they rarely provide a unified loop for property-driven molecular optimization and workflow-level composition. To bridge this gap, we propose Tool-Grounded Multi-Agent Reasoning for Compositional Chemical Problem Solving (TMCS), a step-by-step multi-agent framework that formalizes chemical problem solving as an interpretable, tool-augmented workflow. At the task level, specialized agents leverage external tools, few-shot trajectory memory, and structured reflection to iteratively refine solutions. At the workflow level, TMCS chains generation, understanding, editing, description, and optimization into a closed-loop pipeline. Evaluations across multiple chemical tasks demonstrate that TMCS consistently enhances chemical reasoning across both openand closed-source base models, achieving state-of-the-art performance.

## Introduction

Large Language Models (LLMs) and specialized Chemical Language Models (CLMs) have recently emerged as a transformative force in accelerating drug discovery and materials science (Bai et al. 2025; Ma et al. 2024; Shojaee et al. 2025; Hatakeyama-Sato et al. 2023; Xia et al. 2025; Han et al. 2025; Zhang et al. 2025; Xia et al. 2022; Lv et al. 2024; Zhang et al. 2024; Tan et al. 2025; Zhao et al. 2024; Jiang et al. 2025; Ross et al. 2022; Chithrananda, Grand, and Ramsundar 2020; Ahmad et al. 2022). However, rigorous chemical problem-solving—such as drug design—is inherently procedural rather than instantaneous. Such tasks are rarely resolved through a one-shot prediction; instead, they necessitate a logical sequence of operations, including structural understanding, targeted modification, and iterative refinement (Ock et al. 2026). This stepwise complexity underscores a critical yet overlooked requirement: true chemical intelligence must be assessed not solely by the final output, but by the system’s capacity for transparent, intermediate logical reasoning (Hao et al. 2026). Existing paradigms face bottlenecks that hinder this procedural reasoning (Wang et al. 2025; Bai et al. 2025; Li et al. 2026; Solovev et al. 2025; Lee et al. 2026; Liu et al. 2024). First, at the individual task level, most methods treat chemical problems as direct, end-to-end predictions. Large language models (LLMs) and difusionbased approaches map an input directly to an output relying on parameterized, black-box knowledge. This lacks reasoning transparency: structural changes are not visible step-bystep, making it dificult to interpret why a candidate succeeds or fails under fine-grained chemical constraints. Second, at the execution level, standard models exhibit what we term execution rigidity. Even when basic external tools are introduced, they rarely revise their strategy after a failed attempt, lack explicit reflection, and cannot reuse prior successful trajectories. Finally, at the workflow level, existing models often operate in knowledge isolation, executing tasks like generation, understanding, and optimization as disjointed processes rather than a continuous, cohesive pipeline.

![](images/ad691a52226b2a6e5f4e7288f4269905d515e62123d85e26a852aff46494a62d.jpg)

![](images/2427a5737320364b7883415b015097c56fc88ac389a1b5b4f8e186e850646b68.jpg)  
Figure 1: Mean property improvement (∆) across six molecular optimization objectives. The grouped bar charts compare the proposed TMCS framework against its corresponding generative baselines: Llama-3.3-70B-Instruct (left) and GPT-5.4 (right). TMCS consistently achieves superior ∆ values across both physicochemical constraints and bioactivity targets. Higher positive values indicate successful property enhancements, whereas values near or below zero signify property degradations.

The core dificulty is particularly acute for molecular optimization. A useful drug-design assistant must not only generate a syntactically valid molecule, but also determine whether a local edit actually improves logP, QED, solubility, or targetspecific bioactivity while preserving a meaningful scafold. Pure LLM prompting can express chemical intuition, yet it cannot reliably observe the quantitative consequence of each edit. Conversely, standalone tools can score or validate candidates but do not decide how to revise a failed design.

This gap motivates a framework in which the LLM proposes chemically plausible actions, external tools provide objective feedback, and a bounded reflection loop revises the strategy only when the current trajectory is invalid or insuficiently improved.

To address these limitations, we propose TMCS, an interpretable, stepwise multi-agent framework for chemical problem solving. Built on the principle that chemical reasoning requires structured execution rather than simple black-box scaling, TMCS shifts the paradigm from one-shot generation to a closed-loop, tool-grounded refinement process. To achieve this, we introduce a dual-level optimization strategy. First, we introduce the Task-Level Iterative Refinement: rather than stopping after a single prediction, TMCS decomposes individual tasks into executable stages. Specialized agents handle understanding, editing, and description, while external tools provide deterministic property validation. A reflection module monitors intermediate outputs and triggers strategy revision upon failure, guided by a few-shot trajectory memory that preserves reusable patterns from successful cases. As shown in Figure 1, our framework achieves significant improvements in property optimization for both open-source and closed-source models.

Second, we introduce the Workflow-Level Composition: moving beyond isolated execution, TMCS chains these diverse, optimized tasks into a unified pipeline. By passing context sequentially–from molecular generation, to structural understanding, and finally to targeted optimization–the framework ensures that insights gained in early stages directly inform downstream reasoning. This dual-level design improves interpretability, as every edit is linked to a specific structural rationale and validation result.

Extensive evaluations show that TMCS substantially improves both open-source and closed-source base models, achieving state-of-the-art performance. Component, initialization, robustness, and cost analyses further indicate that the gains arise from verifiable tool-grounded feedback and bounded reflection rather than from unconstrained extra computation alone.

Our main contributions are summarized as follows:

• We propose TMCS, a stepwise multi-agent reasoning framework that transitions chemical problem solving from opaque, one-shot predictions to a transparent, closed-loop refinement process.

• We design a dual-level execution architecture: at the task level, it enhances structural understanding and optimization through tool-use, few-shot reasoning trajectories, and reflection; at the workflow level, it composes these tasks into a unified, sequential pipeline.

• We conduct extensive experiments showing that TMCS broadly enhances both open and closed-source LLMs. The framework yields substantial performance enhancements, achieving state-of-the-art results in chemical reasoning tasks and highlighting the superiority of structured execution.

## Related work

## Chemical Reasoning Models and Evaluation

Prior work shows that many traditional evaluations emphasize factual recall and short-answer prediction, making it hard to distinguish knowledge retrieval from genuine chemical reasoning (Hao et al. 2026; Guo et al. 2025) . To address this gap, ChemCoTBench formulates molecular add/delete/- substitute operations as reasoning primitives, promoting a modular evaluation paradigm that better matches chemist workflows (Hao et al. 2026). Meanwhile, chemistry foundation models such as ChemLLM, ChemDFM, and Llas-Mol have improved performance on molecular and reaction tasks (Zhang et al. 2024; Zhao et al. 2025; Yu et al. 2024), but they still face limitations in cross-task robustness, reasoning consistency, and interpretability(Wang et al. 2025). More recent methods, such as Chem-R, introduce protocol-guided distillation, process-level supervision, and reinforcement-style optimization to improve the reliability of long-horizon chemical reasoning (Wang et al. 2025; Bai et al. 2025)

## Multi-Agent Systems for Drug Discovery

Another major direction is tool-augmented multi-agent systems for drug discovery. Frameworks such as ChemCrow, ChemAgent, CACTUS, and DrugAgent combine LLM planning with chemistry tools, establishing increasingly mature workflows for molecular understanding, reaction analysis, and candidate generation (Bran et al. 2023; Tang et al. 2025; Liu et al. 2024). MADD further extends this paradigm toward end-to-end hit identification through task decomposition and agent orchestration (Solovev et al. 2025). However, closed-loop molecular optimization remains underexplored. It requires quantitatively constrained structural edits, multiproperty balancing, candidate validation, and systematic revision after failed modifications. Existing systems provide limited support and sufer from scarce optimization trajectories, tool-chain error accumulation, and cost and privacy constraints (Solovev et al. 2025; Bran et al. 2023). ChemCRAFT partially addresses these issues by delegating deterministic computation to external tools while retaining high-level decisions within the model (Li et al. 2026). This gap motivates TMCS, which integrates structured reasoning, tool-grounded validation, and multi-agent execution for reliable molecular optimization.

Compared with these systems, TMCS focuses specifically on property-driven molecular optimization and its composition with upstream generation and understanding tasks. As summarized in Table 1, TMCS integrates tool use, memory, and reflection into an optimization-oriented protocol in which candidate molecules are generated, externally validated, scored by property oracles, and revised through bounded feedback.

## Method

## Overview of the TMCS Framework

We formulate chemical problem solving as a structured duallevel reasoning process that maps an optional combined query $x \ : = \ : ( x _ { \mathrm { t e x t } } , x _ { \mathrm { m o l } } )$ to an optimized output y. Rather than relying on a single black-box prediction, our framework decomposes complex tasks through a holistic architecture. As illustrated in Figure 2, the TMCS framework is supported by three foundational pillars: the Task Routing Agent, the Trajectory Memory Bank, and the Chemical Tool Service.

![](images/a01b5c6c306f2cefcc2a0642c2b622f93dd82d7b0214ac4f5848d5bcc6f66424.jpg)  
Figure 2: The overall macroscopic architecture of the TMCS framework. It illustrates the compositional chemical pipeline supported by the Task Router, Memory Bank, and Tool Service.

![](images/2b0b3b16b9679f0fc80045863d48ccfceeb694774258ac09f83ddda1456da16b.jpg)  
Figure 3: The task-level tool-grounded Agent Loop and memory construction process. It details how raw chemical data is transformed into few-shot memory to guide the self-evolving refinement cycle.

Given an input x, the framework operates across two granularities. At the Workflow-Level, macro-tasks are sequentially chained (e.g., generation → analysis → optimization). At the Task-Level, each specialized agent assigned by the Task Router executes a self-evolving loop. Formally, the endto-end execution can be denoted as the composition of K specialized agents:

$$
y _ { \mathrm { f i n a l } } = \left( \prod _ { k = 1 } ^ { K } A _ { k } \right) ( x ; \mathcal { M } , \mathcal { T } )\tag{1}
$$

where M represents the dynamic trajectory memory and T represents the integrated tool suite. This compositional design guarantees both macroscopic task coherence across diverse reasoning stages and microscopic structural precision at the atomic level.

<table><tr><td>Method</td><td>Bounded reflection</td><td>Task routing</td><td>Closed-loop optimization</td><td>Multi task</td></tr><tr><td>ChemCrow</td><td>X</td><td>△</td><td>x</td><td>x</td></tr><tr><td>DrugAgent</td><td>x</td><td>△</td><td>△</td><td>X</td></tr><tr><td>MADD</td><td>△</td><td>√</td><td>△</td><td>x</td></tr><tr><td>ChemCRAFT</td><td>△</td><td>x</td><td>△</td><td>x</td></tr><tr><td>TMCS</td><td>√</td><td>√</td><td>√</td><td>√</td></tr></table>

Table 1: Capability comparison with representative toolaugmented chemical agents. ✓: full support; △: partial support; ✗: absent or not a primary focus.

## Chemical Tool Service

To ground LLM reasoning in executable chemistry, TMCS instantiates a hybrid tool stack T that normalizes all tool invocations into JSON-compatible program calls (Li et al. 2026; Wang et al. 2025). The stack supports structural validation, deterministic property calculation, and constrained molecular editing. Candidate molecules must be chemically parseable, while editing operations are additionally verified through SMARTS-count changes to ensure that the requested functional group is added, removed, or replaced exactly as specified. Physicochemical descriptors, molecular similarities, structural patterns, and pharmacological properties are computed using deterministic backends or aligned predictive models rather than inferred directly by the LLM.

For molecular design and captioning, TMCS employs a specialized model derived from Chem-R as a replaceable generation tool. Generated structures and guarded LLM fallback outputs are subjected to the same validity and editconsistency checks before being accepted into the work flow. This design combines deterministic chemical execution with flexible generation while maintaining a strict validation boundary. A complete description of the tools, interfaces, and functional roles is provided in supplementary materials.

## Trajectory Memory Construction and Retrieval

To mitigate zero-shot instability during complex chemical reasoning, we introduce an ofline-constructed Few-Shot Trajectory Memory Bank, denoted by M, which supplies executable priors during inference. As shown in supplementary materials, we provide additional specifications regarding this section.

Trajectory Construction. As shown in Figure 3, we construct our instruction dataset following (Li et al. 2026) through a three-step process. First, we collect high-fidelity molecular pairs, property annotations, and functional descriptions from databases like ChEBI and PubChem, standardizing them into a uniform SMILES-centric format based on task types. Second, we integrate these data instances with strategy templates that explicitly decompose chemical transformations into logical, tool-executable cognitive steps (i.e., structural analysis, optimization strategy, and validation). Finally, we employ large language models and external tools to infer initial reasoning trajectories.

Structured Memory and Deterministic Retrieval. We construct a task-bounded memory bank that encapsulates structured, deterministically executable exemplars (e.g., toolcall records) across molecular understanding, editing, and optimization tasks. To ensure rigorous structural alignment, we employ a deterministic, symbolic indexing mechanism based on exact key matching directed by task types. By directly matching exemplars to the specific task category, we guarantee that the retrieved trajectories are strictly compatible with the active query and immediately applicable to the current chemical problem. Each specific task is associated with its corresponding few-shot demonstrations. Furthermore, to rigorously prevent train-test contamination, the memory bank is constructed entirely ofline, remains frozen during evaluation, and enforces strict structural disjointness from the test sets through canonical de-duplication and topological distance thresholding.

## Task-Level: Tool-Grounded Agent Loop

Upon receiving a specific task assignment from the Router Agent, the system initiates the core Agent Loop. As shown in Figure 3, taking the property optimization task as a representative paradigm, the internal loop dynamically orchestrates a sequence of actions based on the current context.

Chemical Agent Loop. The chemical agent loop proceeds through four integrated phases: (1) Feature Calculation, where the agent invokes $\tau _ { \mathrm { c a l c } }$ to extract physicochemical features (e.g., functional group counts and molecular weight) from the current molecule $\bar { y ^ { ( t - 1 ) } }$ ; (2) Strategy Formulation, where guided by the extracted features, target constraints, and retrieved memory M, the agent articulates a targeted structural modification strategy $z ^ { ( t ) } ; ( 3 )$ Generation & Editing, where the agent utilizes $\tau _ { \mathrm { e d i t } }$ or $\tau _ { \mathrm { g e n } }$ to actualize the proposed strategy, yielding a candidate molecule $y ^ { ( t ) }$ along with its generation rationale; and (4) Analysis & Validation, where the candidate is rigorously evaluated via $\mathcal { T } _ { \mathrm { v a l i d } }$ and $\tau _ { \mathrm { c a l c } }$ to return an objective feedback vector $\mathbf { r } ^ { ( t ) }$

![](images/b76224629025dd9e1ce2a329c101765adeb68f5527c50ee1397d54a041bfdb29.jpg)  
Figure 4: Example of QED attribute optimization.

Refinement and Summarization. Upon receiving the feedback vector, the agent performs a reflection operation to analyze the current solution’s eficacy. In our implementation, molecular optimization uses at most three refinement rounds and generates three candidates per round. The best valid candidate is selected according to property improvement and scafold similarity; if no candidate improves the objective, the next prompt receives explicit feedback describing the best failed attempt and whether more conservative or more aggressive edits are needed. The reflection module classifies errors into format, structural, counting, logic, hallucination, or unknown categories, then appends a targeted correction prompt rather than an open-ended instruction to “think again.” This self-evaluation drives further iterative refinement, ultimately culminating in the final task solution and a comprehensive analytical summary report.

## Workflow-Level: Compositional Chemical Pipeline

For multifaceted, real-world chemical challenges (e.g., designing a targeted drug from a raw textual description), a single autonomous loop is insuficient due to context window dilution and error accumulation. At the workflow level, TMCS chains diverse, specialized agents into a macroscopic pipeline.

As shown in Figure 2 , the workflow initiates with a Molecule Generation Agent translating textual descriptions (e.g., “an indole alkaloid derivative”) into initial structural scafolds. The finalized summary of this agent is seamlessly handed over as the initial context to a Molecule Analysis Agent, which isolates modifiable substituents. Finally, the Property Optimization Agent receives this focused analysis to perform the targeted refinement loop discussed above. This compositional architecture ensures that the cognitive load is compartmentalized, preventing the cascading of hallucinations while preserving high-fidelity reasoning across complex chemical workflows.

## Experiments

## Experimental Setup

Benchmark. To comprehensively validate our framework, we conducted evaluations across a broad spectrum of tasks. At the task level, we evaluated drug molecule design and molecular description from ChemLLMBench (Guo et al. 2023), alongside molecular understanding, molecular editing, and property optimization tasks from ChemCoT-Bench (Hao et al. 2026). At the workflow level, we utilized the molecular design task (Guo et al. 2023) as an initialization point to sequentially drive downstream optimization tasks and evaluate the optimization performance across six distinct molecular properties.

Evaluation Metrics. For rigorous and fair benchmarking, we follow standard protocols from prior work (Guo et al. 2023; Hao et al. 2026). Molecular design is evaluated by exact SMILES match, while molecule captioning uses BLEU-4. Molecular understanding evaluates functional-group and ring counting with Mean Absolute Error (MAE), Murcko scafold extraction with Tanimoto similarity, complex-ring detection with accuracy, and SMILES equivalence as a binary classification task using accuracy. Molecular editing is measured by Pass@1, indicating whether the edited structure strictly follows the instruction, whereas molecular optimization is assessed by average property improvement. Direct LLM baselines receive task-specific optimization prompts and the same property oracles as TMCS; complementary ablations and cost analyses further distinguish architectural gains from those due to additional inference budget. The LLM parameters are set to ‘temperature = 0.3‘ and max tokens are 512, with up to eight retries upon failure.

Baselines. To establish a rigorous and comprehensive comparison, we categorize our baselines into three distinct paradigms. For direct reasoning, we adopt the direct inference baselines established in ChemCoTBench to evaluate standard prompting capabilities (Grattafiori et al. 2024; Hurst et al. 2024; Guo et al. 2025; Comanici et al. 2025; Singh et al. 2025). For domain-specific systems, we select Chem-R, ether0 (Narayanan et al. 2026), and Chem-CRAFT as our primary comparators. Finally, to validate the generalizability of our proposed framework at both the task and workflow levels, we deploy TMCS on top of leading foundational models, utilizing both open-source (specifically Llama-3.3-70B-Instruct) and closed-source (specifically GPT-5.4) LLMs as our base models.

## Main Results

Task-Level Benchmark Results. Table 2 shows that TMCS achieves strong gains across molecular understanding, editing, generation, and description, outperforming strong LLM baselines particularly on tasks requiring multi-step reasoning, precise structural manipulation, or description generation. These results demonstrate that the same stepwise reasoning architecture generalizes beyond a single optimization objective. They also justify a dedicated molecular optimization module: because text-to-molecule generation may produce plausible scafolds but cannot ensure property improvement under structural constraints, TMCS uses generation only for initialization and applies tool-grounded optimization to obtain quantitatively improved candidates.

Table 3 summarizes the task-level benchmark performance on multiple optimization objectives. TMCS consistently improves property scores across LogP, solubility, QED, DRD2, JNK3, and GSK3-β. Compared with both open and closed baselines, our method exhibits stronger optimization ability and more stable success rates.

While the Llama-3.3-70B-Instruct model exhibits limited reasoning capabilities in direct inference tasks, the integration of the TMCS framework leads to substantial performance gains. Similarly, although powerful proprietary models like GPT-5.4 demonstrate baseline efectiveness in zero-shot settings, their performance is further amplified by TMCS. These results underscore that the TMCS framework provides a robust, universal mechanism for unlocking hightier chemical reasoning, regardless of the underlying model’s initial architecture.

Workflow-Level Pipeline Performance. To test whether the individual task gains can be composed into a stronger end-to-end system, we start from drug molecule generation and then perform downstream optimization. We compare the tool-enhanced pipeline against a single-turn, task-specific LLM baseline.

Table 4 presents the core results of this workflow-level optimization stage, whose initialization setting difers from the standalone task-level benchmark in Table 3. We compare the Tool-enhanced TMCS framework against a Pure LLM Baseline using task-specific prompts but no external feedback during generation. The results demonstrate a significant performance gap, particularly in properties like DRD2 and JNK3, where the lack of quantitative feedback causes the pure LLM to struggle (∼15%–42% success). In contrast, TMCS leverages tool-grounded feedback to achieve a substantial performance boost, reaching success rates as high as 98%.

The table reports two complementary workflow-level measures over all 100 molecules. SR is the fraction with a positive change, whereas ∆ is the net mean change after summing both positive and negative changes. TMCS improves SR for all six objectives on both backbones, and its all-sample ∆ is higher on all six objectives for 70B and on five of six objectives for GPT-5.4. This combination shows that TMCS is more stable across molecules rather than relying on a small number of large gains.

For GPT-5.4, the stability advantage is especially clear on the black-box objectives: TMCS raises SR from 42.0% to 80.0% on DRD2, from 30.0% to 96.0% on JNK3, and from 35.0% to 90.0% on GSK3-β; their net mean changes increase from 0.002 to 0.148, 0.004 to 0.054, and 0.003 to 0.124, respectively. Solubility is the only ∆ exception: the baseline reaches 1.305 versus TMCS’s 1.062, but TMCS still improves SR from 80.0% to 92.0%. Thus, the baseline’s larger solubility magnitude does not overturn the broader evidence that tool-grounded feedback yields more reliable molecule-level improvements.

## Ablation Study

Component Ablation Table 6 shows the contribution of each component on two representative optimization objectives, LogP and DRD2. The full model achieves the best performance on both metrics. Removing memory or tools consistently degrades performance, confirming that the gains cannot be attributed only to extra LLM calls.

The results reveal three important trends. First, removing few-shot memory reduces both improvement and success rate, indicating that reusable trajectory patterns are helpful for selecting better edits. Second, removing tools causes a larger drop in success rate, suggesting that deterministic validation is important for producing reliable candidates.

<table><tr><td>Methods</td><td>FG↓</td><td>Ring↓</td><td>Murcko↑</td><td>Ring-sys↑</td><td>Eq↑</td><td>Add</td><td>Delete</td><td>Sub</td><td>Design</td><td>Capt</td></tr><tr><td colspan="9">General Models</td><td></td></tr><tr><td>GPT-40</td><td>0.17</td><td>1.35</td><td>0.21</td><td>80.0</td><td>72</td><td>80</td><td>80</td><td>65.0</td><td>0.07</td><td>0.01</td></tr><tr><td>DeepSeek-R1</td><td>0.27</td><td>1.55</td><td>0.34</td><td>45.0</td><td>65</td><td>70</td><td>70</td><td>68.3</td><td>0.22</td><td>0.04</td></tr><tr><td>GPT 5.4</td><td>0.29</td><td>1.05</td><td>0.42</td><td>45.0</td><td>84</td><td>80</td><td>80</td><td>88.3</td><td>0.19</td><td>0.13</td></tr><tr><td>Gemini-2.5-pro</td><td>0.11</td><td>0.60</td><td>0.51</td><td>87.5</td><td>82</td><td>100</td><td>85</td><td>81.7</td><td>0.29</td><td>0.04</td></tr><tr><td>Llama-3.3-70B-Instruct</td><td>0.52</td><td>1.80</td><td>0.12</td><td>68.3</td><td>67</td><td>60</td><td>80</td><td>50.0</td><td>0.03</td><td>0.02</td></tr><tr><td colspan="9">Chemical Models</td><td></td><td></td></tr><tr><td>ChemCRAFT</td><td>0.03</td><td>0.15</td><td>0.57</td><td>100.0</td><td>97</td><td>85</td><td>95</td><td>80.0</td><td></td><td></td></tr><tr><td>ether0</td><td></td><td>0.35</td><td></td><td></td><td>63</td><td>94</td><td>76</td><td>78.0</td><td>0.30</td><td>0.03</td></tr><tr><td>Chem-R</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.42</td><td>0.41</td></tr><tr><td>BioMedGPT</td><td>1.60</td><td>2.43</td><td>0.18</td><td>53.3</td><td>39</td><td>10</td><td>12</td><td>10.0</td><td></td><td></td></tr><tr><td>BioMistral</td><td>1.00</td><td>1.85</td><td>0.04</td><td>32.5</td><td>50</td><td>0</td><td>10</td><td>0.0</td><td></td><td></td></tr><tr><td>TMCS (70B)</td><td>0.31</td><td>0.00</td><td>1.00</td><td>100.0</td><td>99</td><td>90</td><td>90</td><td>95.0</td><td>0.44</td><td>0.48</td></tr><tr><td>TMCS (GPT 5.4)</td><td>0.00</td><td>0.00</td><td>1.00</td><td>100.0</td><td>99</td><td>95</td><td>100</td><td>96.7</td><td>0.47</td><td>0.49</td></tr></table>

Table 2: Results on individual chemical tasks. 70B represents Llama-3.3-70B-Instruct, ↓ indicates lower is better, while ↑ indicates higher is better.
<table><tr><td rowspan="2">Methods</td><td colspan="2">LogP</td><td colspan="2">Solubility</td><td colspan="2">QED</td><td colspan="2">DRD2</td><td colspan="2">JNK3</td><td colspan="2">GSK3-β</td></tr><tr><td>∆</td><td>SR</td><td>∆</td><td>SR</td><td>∆</td><td>SR</td><td>∆</td><td>SR</td><td>∆</td><td>SR</td><td>∆</td><td>SR</td></tr><tr><td colspan="10">General Models</td><td></td><td></td><td></td><td></td></tr><tr><td>GPT-40</td><td>-0.09</td><td>37</td><td>0.92</td><td>80</td><td>0.13</td><td>70</td><td>0.07</td><td>48</td><td>-0.02</td><td>30</td><td>-0.00</td><td>39</td></tr><tr><td>DeepSeek-R1</td><td>0.47</td><td>69</td><td>0.80</td><td>80</td><td>0.17</td><td>72</td><td>0.12</td><td>62</td><td>-0.02</td><td>29</td><td>0.01</td><td>41</td></tr><tr><td>Gemini-2.5-pro</td><td>-0.22</td><td>76</td><td>1.06</td><td>70</td><td>0.28</td><td>84</td><td>0.36</td><td>74</td><td>-0.02</td><td>35</td><td>0.06</td><td>68</td></tr><tr><td>Llama-3.3-70B-Instruct</td><td>0.02</td><td>35</td><td>0.72</td><td>81</td><td>0.15</td><td>61</td><td>0.00</td><td>31</td><td>-0.01</td><td>30</td><td>0.01</td><td>40</td></tr><tr><td>GPT-5.4</td><td>0.54</td><td>83</td><td>1.00</td><td>90</td><td>0.18</td><td>91</td><td>0.00</td><td>57</td><td>0.00</td><td>39</td><td>0.04</td><td>69</td></tr><tr><td colspan="10">Chemical Models</td><td></td><td></td><td></td><td></td></tr><tr><td>ether0</td><td>0.00</td><td>0</td><td>0.00</td><td>0</td><td>0.00</td><td>0</td><td>0.00</td><td>0</td><td>0.00</td><td>0</td><td>0.00</td><td>0</td></tr><tr><td>Chem-R</td><td></td><td>1</td><td>0.34</td><td>83</td><td></td><td>=</td><td>0.01</td><td>36</td><td>-0.02</td><td>24</td><td>-0.01</td><td>29</td></tr><tr><td>BioMistral</td><td>0.01</td><td>1</td><td>0.24</td><td>6</td><td>0.00</td><td>0</td><td>0.00</td><td>1</td><td>-0.01</td><td>1</td><td>-0.01</td><td>0</td></tr><tr><td>BioMedGPT</td><td>-0.36</td><td>17</td><td>0.25</td><td>63</td><td>-0.29</td><td>7</td><td>-0.09</td><td>5</td><td>-0.11</td><td>6</td><td>-0.08</td><td>1</td></tr><tr><td>TMCS (70B)</td><td>1.10</td><td>96</td><td>0.92</td><td>73</td><td>0.20</td><td>80</td><td>0.30</td><td>90</td><td>0.12</td><td>72</td><td>0.07</td><td>49</td></tr><tr><td>TMCS (GPT 5.4)</td><td>0.90</td><td>98</td><td>1.09</td><td>99</td><td>0.25</td><td>98</td><td>0.42</td><td>92</td><td>0.07</td><td>61</td><td>0.16</td><td>79</td></tr></table>

Table 3: Optimization results on multiple molecular objectives. ∆ is the mean property improvement, where a negative ∆ indicates that most optimizations are property degradations. SR denotes success rate (%).

Regarding the ablation of reflection frequency, as shown in Table 6, removing reflection weakens the system’s ability to escape local plateaus, which is especially visible in the optimization tasks that require iterative refinement. One round already improves over the direct baseline, while five rounds only marginally improve solubility ∆ from 0.92 to 0.93 compared with the three-round TMCS setting. We therefore use three rounds as the default because it captures nearly all of the performance gain while avoiding the additional latency and token cost of longer reflection. For tasks such as molecular understanding, a single reflection step is suficient to achieve optimal performance, so we restrict the reflection mechanism to a single iteration for these tasks.

Additionally, we provide a detailed statistical significance and robustness analysis in supplementary materials, followed by a computational cost and eficiency analysis of our framework in supplementary materials.

Impact of Initialization Source. To further understand the robustness of TMCS, we investigate the impact of the initialization source on optimization success. As shown in Table 7, we split the cases into those starting from generated molecules (Generated-Start) and those falling back to ground-truth molecules (GT-Start).

A key finding is that pure LLM optimization is highly sensitive to the starting source: performance varies substantially between generated scafolds and ground-truth scafolds across DRD2, GSK3-β, and JNK3. TMCS remains consistently strong in both settings. When the initial molecule is produced by a general molecular generator, TMCS can still further optimize it; when the system starts from a groundtruth scafold, TMCS also improves it. Thus, Chem-R-style initialization is useful but not the sole source of the fina gain; the tool-grounded optimization loop supplies the decisive navigation signal.

<table><tr><td rowspan="2">Model</td><td rowspan="2">Strategy</td><td colspan="2">LogP</td><td colspan="2">Solubility</td><td colspan="2">QED</td><td colspan="2">DRD2</td><td colspan="2">JNK3</td><td colspan="2">GSK3-β</td></tr><tr><td> $\Delta$ </td><td>SR</td><td> $\Delta$ </td><td>SR</td><td> $\Delta$ </td><td>SR</td><td>∆</td><td>SR</td><td>∆</td><td>SR</td><td>∆</td><td>SR</td></tr><tr><td>70B</td><td>Baseline TMCS (Ours)</td><td>0.152 1.056</td><td>41.0 89.0</td><td>0.004 0.416</td><td>15.0 65.0</td><td>-0.003 0.123</td><td>16.0 81.0</td><td>0.000 0.094</td><td>17.0</td><td>0.001</td><td>15.0 68.0</td><td>0.002 0.085</td><td>14.0 74.0</td></tr><tr><td>GPT-5.4</td><td>Baseline TMCS (Ours)</td><td>-0.417 1.101</td><td>82.0 98.0</td><td>1.305</td><td>80.0</td><td>0.095 0.178</td><td>78.0 94.0</td><td>0.002 0.148</td><td>69.0 42.0 80.0</td><td>0.028 0.004</td><td>30.0 0.054</td><td>0.003 96.0 0.124</td><td>35.0 90.0</td></tr></table>

Table 4: Workflow-level optimization-stage results: comparison between the task-specific Pure LLM Baseline and TMCS across six molecular objectives. We report ∆ (mean net property change over all 100 cases, including positive and negative changes) and SR (success rate, %). 70B represents Llama-3.3-70B-Instruct.

<table><tr><td rowspan="2">Method</td><td colspan="2">LogP</td><td colspan="2">DRD2</td></tr><tr><td> $\Delta$ </td><td>SR</td><td>Δ</td><td>SR</td></tr><tr><td>Ours</td><td>1.10</td><td>96</td><td>0.30</td><td>90</td></tr><tr><td>w/o few-shot</td><td>0.96</td><td>88</td><td>0.27</td><td>84</td></tr><tr><td>w/o tool</td><td>0.96</td><td>86</td><td>0.26</td><td>80</td></tr></table>

Table 5: Ablation results on molecular optimization tasks. ∆ is the mean property improvement, where a negative ∆ indicates that most optimizations are property degradations, and SR denotes success rate.

<table><tr><td>Setting</td><td>Solubility ∆</td></tr><tr><td>Baseline</td><td>0.72</td></tr><tr><td>Ablation on reflection rounds</td><td></td></tr><tr><td>w/o reflection 1 round</td><td>0.87 0.90</td></tr><tr><td>5 rounds</td><td>0.93</td></tr><tr><td>3 rounds (Ours)</td><td>0.92</td></tr></table>

Table 6: Ablation results on reflection rounds for molecular optimization. ∆ denotes mean property improvement; TMCS uses three rounds by default.

Eficiency Trade-of. The multi-agent framework intentionally spends more inference than single-turn prompting because it performs candidate generation, tool scoring, and reflection. However, this overhead is controlled rather than open-ended: molecular optimization is capped at three rounds with three candidates per round, deterministic tasks are routed directly to tools, and short tasks use at most one reflection. Supplementary materials quantifies this trade-of. Direct inference is cheaper, but it lacks the feedback channel needed for long-step optimization; TMCS pays additional tokens only for modules that expose intermediate chemical evidence. The reflection-round ablation further shows that extending the loop beyond the default budget produces negligible gains, supporting the chosen quality-eficiency balance.

<table><tr><td>Model</td><td>Start Source Strategy</td><td></td><td>DRD2</td><td>GSK3-β JNK3</td><td></td></tr><tr><td rowspan="2">GPT-5.4</td><td>Gen-Start</td><td>Baseline TMCS</td><td>40.9 78.8</td><td>31.9 94.2</td><td>32.9 95.7</td></tr><tr><td>GT-Start</td><td>Baseline TMCS</td><td>44.1 82.4</td><td>41.9 80.6</td><td>23.3 96.7</td></tr></table>

Table 7: Ablation results on initialization source for bioactivity tasks (DRD2, GSK3-β, JNK3).

Case Study. To provide an implementation-level qualitative analysis, Figure 4 illustrates a three-round QED optimization trajectory driven by the proposed pipeline. As depicted, the system integrates the task prompt, deterministic tool feedback, and few-shot structural rationales to initialize the generation cycle.

Starting from the initial compound $( s _ { 0 } , \mathrm { Q E D } = 0 . 3 3 5 4 ) .$ Round 1 executes a localized structural edit—replacing the PAINS-flagged hydroxamic acid liability with an acetyl group. When the trajectory plateaus in Round 2, the ReflectionAgent actively intervenes. By analyzing the updated tool profiles, it explicitly diagnoses that although the QED improved, the naphthalene ring system remained overly bulky.

Conditioned on these tool-grounded diagnostics, the agent dynamically refreshes its strategy for Round 3. It shifts from localized editing to a deeper scafold simplification (e.g., replacing the fused bicyclic naphthalene motif with a simpler para-substituted variant). This case concretely demonstrates the eficacy of the multi-round reflective logic: it detects stagnation, performs precise credit assignment via explicit diagnostics, and successfully escapes local optima that singlepass generative models typically fail to overcome.

## Conclusion

TMCS is a tool-grounded multi-agent framework that reformulates chemical problem-solving into an interpretable, iterative workflow, integrating generative modeling with deterministic chemical logic and achieving SOTA that lets open-weight models match or surpass closed-source ones.

## References

Ahmad, W.; Simon, E.; Chithrananda, S.; Grand, G.; and Ramsundar, B. 2022. Chemberta-2: Towards chemical foundation models. arXiv preprint arXiv:2209.01712.

Bai, L.; Cai, Z.; Cao, Y.; Cao, M.; Cao, W.; Chen, C.; Chen, H.; Chen, K.; Chen, P.; Chen, Y.; et al. 2025. Intern-s1: A scientific multimodal foundation model. arXiv preprint arXiv:2508.15763.

Bran, A. M.; Cox, S.; Schilter, O.; Baldassari, C.; White, A. D.; and Schwaller, P. 2023. Chemcrow: Augmenting large-language models with chemistry tools. arXiv preprint arXiv:2304.05376.

Chithrananda, S.; Grand, G.; and Ramsundar, B. 2020. ChemBERTa: large-scale self-supervised pretraining for molecular property prediction. arXiv preprint arXiv:2010.09885.

Comanici, G.; Bieber, E.; Schaekermann, M.; Pasupat, I.; Sachdeva, N.; Dhillon, I.; Blistein, M.; Ram, O.; Zhang, D.; Rosen, E.; et al. 2025. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261.

Grattafiori, A.; Dubey, A.; Jauhri, A.; Pandey, A.; Kadian, A.; Al-Dahle, A.; Letman, A.; Mathur, A.; Schelten, A.; Vaughan, A.; et al. 2024. The llama 3 herd of models. arXiv preprint arXiv:2407.21783.

Guo, D.; Yang, D.; Zhang, H.; Song, J.; Wang, P.; Zhu, Q.; Xu, R.; Zhang, R.; Ma, S.; Bi, X.; et al. 2025. DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning. Nature, 645(8081): 633–638.

Guo, T.; Guo, K.; Nan, B.; Liang, Z.; Guo, Z.; Chawla, N. V.; Wiest, O.; and Zhang, X. 2023. What can Large Language Models do in chemistry? A comprehensive benchmark on eight tasks. In Oh, A.; Naumann, T.; Globerson, A.; Saenko, K.; Hardt, M.; and Levine, S., eds., Advances in Neural Information Processing Systems 36:Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023.

Han, Y.; Wan, Z.; Chen, L.; Yu, K.; and Chen, X. 2025. From generalist to specialist: A survey of large language models for chemistry. In Proceedings of the 31st International Conference on Computational Linguistics, 1106–1123.

Hao, L.; Cao, H.; Feng, B.; Shao, D.; Tang, R.; Yan, Z.; Tian, Y.; Yuan, L.; and Li, Y. 2026. Beyond chemical qa: Evaluating llm’s chemical reasoning with modular chemical operations. Advances in Neural Information Processing Systems, 38.

Hatakeyama-Sato, K.; Yamane, N.; Igarashi, Y.; Nabae, Y.; and Hayakawa, T. 2023. Prompt engineering of GPT-4 for chemical research: what can/cannot be done? Science and Technology ofAdvanced Materials: Methods, 3(1): 2260300.

Hurst, A.; Lerer, A.; Goucher, A. P.; Perelman, A.; Ramesh, A.; Clark, A.; Ostrow, A.; Welihinda, A.; Hayes, A.; Radford, A.; et al. 2024. Gpt-4o system card. arXiv preprint arXiv:2410.21276.

Jiang, L.; Sun, S.; Qi, B.; Fu, Y.; Xu, X.; Li, Y.; Zhou, D.; and Fu, T. 2025. Chem3dllm: 3d multimodal large language models for chemistry. arXiv preprint arXiv:2508.10696.

Lee, N.; De Brouwer, E.; Hajiramezanali, E.; Biancalani, T.; Park, C.; and Scalia, G. 2026. Rag-enhanced collaborative

llm agents for drug discovery. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, 561–569.

Li, H.; Cao, H.; Peng, S.; Liu, Z.; Feng, B.; Wang, Y.; Yan, Z.; Tian, Y.; Li, Y.; and Yuan, L. 2026. Agentic reinforcement learning empowers next-generation chemical language models for molecular design and synthesis. arXiv preprint arXiv:2601.17687.

Liu, S.; Lu, Y.; Chen, S.; Hu, X.; Zhao, J.; Lu, Y.; and Zhao, Y. 2024. Drugagent: Automating ai-aided drug discovery programming through llm multi-agent collaboration. arXiv preprint arXiv:2411.15692.

Lv, L.; Li, H.; Wang, Y.; Yan, Z.; Chen, Z.; Lin, Z.; Yuan, L.; and Tian, Y. 2024. Navigating chemical-linguistic sharing space with heterogeneous molecular encoding. arXiv preprint arXiv:2412.20888.

Ma, P.; Wang, T.-H.; Guo, M.; Sun, Z.; Tenenbaum, J. B.; Rus, D.; Gan, C.; and Matusik, W. 2024. Llm and simulation as bilevel optimizers: A new paradigm to advance physical scientific discovery. arXiv preprint arXiv:2405.09783.

Narayanan, S.; Braza, J.; Grifiths, R.-R.; Bou, A.; Wellawatte, G.; Caldas Ramos, M.; Mitchener, L.; Pieler, M.; Rodriques, S.; and White, A. 2026. Training a scientific reasoning model for chemistry. Advances in Neural Information Processing Systems, 38: 157671–157710.

Ock, J.; Meda, R. S.; Badrinarayanan, S.; Aluru, N. S.; Chandrasekhar, A.; and Barati Farimani, A. 2026. Large language model agent for modular task execution in drug discovery. Journal of Chemical Information and Modeling, 66(4): 2055–2068.

Ross, J.; Belgodere, B.; Chenthamarakshan, V.; Padhi, I.; Mroueh, Y.; and Das, P. 2022. Large-scale chemical language representations capture molecular structure and properties. Nature Machine Intelligence, 4(12): 1256–1264.

Shojaee, P.; Meidani, K.; Gupta, S.; Barati Farimani, A.; and Reddy, C. 2025. Llm-sr: Scientific equation discovery via programming with large language models. In International Conference on Learning Representations, volume 2025, 16054–16085.

Singh, A.; Fry, A.; Perelman, A.; Tart, A.; Ganesh, A.; El-Kishky, A.; McLaughlin, A.; Low, A.; Ostrow, A.; Ananthram, A.; et al. 2025. Openai gpt-5 system card. arXiv preprint arXiv:2601.03267.

Solovev, G. V.; Zhidkovskaya, A. B.; Orlova, A.; Gubina, N.; Vepreva, A.; Golovinskii, R.; Tonkii, I.; Dubrovsky, I.; Gurev, I.; Gilemkhanov, D.; Chistiakov, D.; Aliev, T. A.; Poddiakov, I.; Zubkova, G.; Skorb, E. V.; Vinogradov, V.; Boukhanovsky, A.; Nikitin, N.; Dmitrenko, A.; Kalyuzhnaya, A.; and Savchenko, A. 2025. MADD: Multi-Agent Drug Discovery Orchestra. In Christodoulopoulos, C.; Chakraborty, T.; Rose, C.; and Peng, V., eds., Findings of the Association for Computational Linguistics: EMNLP 2025, 6956–6998. Suzhou, China: Association for Computational Linguistics. ISBN 979-8-89176-335-7.

Tan, Q.; Zhou, D.; Xia, P.; Liu, W.; Ouyang, W.; Bai, L.; Li, Y.; and Fu, T. 2025. Chemmllm: Chemical multimodal large language model. arXiv preprint arXiv:2505.16326.

Tang, X.; Hu, T.; Ye, M.; Shao, Y.; Yin, X.; Ouyang, S.; Zhou, W.; Lu, P.; Zhang, Z.; Zhao, Y.; Cohan, A.; and Gerstein, M. 2025. ChemAgent: Self-updating Library in Large Language Models Improves Chemical Reasoning. arXiv:2501.06590.

Wang, W.; Chen, B.; Zhang, D.; Liu, W.; Pu, S.; Gao, B.; Zeng, J.; Wei, X.; Yu, T.; Sun, S.; Fu, T.; Ouyang, W.; Bai, L.; Li, J.; Wang, Z.; Li, Y.; and Zhang, S. 2025. Chem-R: Learning to Reason as a Chemist. arXiv:2510.16880.

Xia, J.; Zhu, Y.; Du, Y.; and Li, S. Z. 2022. A systematic survey of chemical pre-trained models. arXiv preprint arXiv:2210.16484.

Xia, Y.; Jin, P.; Xie, S.; He, L.; Cao, C.; Luo, R.; Liu, G.; Wang, Y.; Liu, Z.; Chen, Y.-J.; Guo, Z.; Bai, Y.; Deng, P.; Min, Y.; Lu, Z.; Hao, H.; Yang, H.; Li, J.; Liu, C.; Zhang, J.; Zhu, J.; Bi, R.; Wu, K.; Zhang, W.; Gao, K.; Pei, Q.; Wang, Q.; Liu, X.; Li, Y.; Zhu, H.; Lu, Y.; Ma, M.; Wang, Z.; Xie, T.; Maziarz, K.; Segler, M.; Yang, Z.; Chen, Z.; Shi, Y.; Zheng, S.; Wu, L.; Hu, C.; Dai, P.; Liu, T.-Y.; Liu, H.; and Qin, T. 2025. Nature Language Model: Deciphering the Language of Nature for Scientific Discovery. arXiv:2502.07527.

Yu, B.; Baker, F. N.; Chen, Z.; Ning, X.; and Sun, H. 2024. Llasmol: Advancing large language models for chemistry with a large-scale, comprehensive, high-quality instruction tuning dataset. arXiv preprint arXiv:2402.09391.

Zhang, D.; Liu, W.; Tan, Q.; Chen, J.; Yan, H.; Yan, Y.; Li, J.; Huang, W.; Yue, X.; Ouyang, W.; Zhou, D.; Zhang, S.; Su, M.; Zhong, H.-S.; and Li, Y. 2024. ChemLLM: A Chemical Large Language Model. arXiv:2402.06852.

Zhang, Q.; Ding, K.; Lv, T.; Wang, X.; Yin, Q.; Zhang, Y.; Yu, J.; Wang, Y.; Li, X.; Xiang, Z.; Zhuang, X.; Wang, Z.; Qin, M.; Zhang, M.; Zhang, J.; Cui, J.; Xu, R.; Chen, H.; Fan, X.; Xing, H.; and Chen, H. 2025. Scientific Large Language Models: A Survey on Biological & Chemical Domains. ACM Comput. Surv., 57(6).

Zhao, Z.; Chen, B.; Li, J.; Chen, L.; Wen, L.; Wang, P.; Zhu, Z.; Zhang, D.; Li, Y.; Dai, Z.; Chen, X.; and Yu, K. 2024. ChemDFM-X: towards large multimodal model for chemistry. Science China Information Sciences, 67(12).

Zhao, Z.; Ma, D.; Chen, L.; Sun, L.; Li, Z.; Xia, Y.; Chen, B.; Xu, H.; Zhu, Z.; Zhu, S.; Fan, S.; Shen, G.; Yu, K.; and Chen, X. 2025. Developing ChemDFM as a large language foundation model for chemistry. Cell Reports Physical Science, 6(4): 102523.