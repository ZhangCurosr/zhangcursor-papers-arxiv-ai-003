# SpecCloak: Probing Harmful Code Generation through Engineering Specifications

Wenbin Chen, , Jiexin Wang, Guanting Wu , Jiaxi Liang and Yi Cai <sup>∗</sup>

Abstract—The code generation capabilities of large language models (LLMs) can equally be misused to implement harmful functionality, underscoring the need to understand their safety behavior in response to programming requests. While request phrasing is known to influence refusal, it remains unclear whether the form in which coding task requirements are stated affects the refusal process for malicious code generation. We introduce PseudoSpec, a template that organizes functional requirements as engineering specifications, and investigate its effects through controlled comparisons, representation analysis, and activation interventions. Across five open-weight models, removing explicit risk keywords only partially reduces refusal, whereas specification prompting substantially lowers refusal even when these keywords are retained. This reduction accompanies consistent shifts toward code-output-related representations, and interventions along condition-difference directions alter refusal behavior. Output analysis further distinguishes code-related responses from generation degeneration. Building on these findings, we propose SpecCloak, a black-box red-teaming framework combining PseudoSpec with discriminator-guided optimization, achieving an average execution-verified attack success rate of 28.91% across eight target models on RedCode-Gen. We also introduce SpecReveal, a defense prototype combining naturallanguage intent reconstruction with dual-view detection without target-model fine-tuning. It achieves a 96.88% attack detection rate in a separate prompt-detection evaluation, with false positives assessed on benign programming and specificationstyle inputs. These results connect specification-induced response differences to practical security consequences and identify intent reconstruction as a complementary detection approach.

Index Terms—Large language models, malicious code generation, jailbreak attack, mechanistic interpretability, risk detection

## I. INTRODUCTION

Large language models (LLMs) are increasingly used to assist software development, generating, completing, and modifying code from user requirements [1]–[3]. However, these capabilities can also be exploited for malicious purposes. Attackers may use coding assistants to turn harmful requirements into program implementations, reducing the effort needed to develop malicious code [4]–[6]. This risk makes resistance to harmful code generation an important security requirement for LLM-based development tools: models should assist legitimate programming tasks while refusing requests to implement harmful functionality.

Research on jailbreak attacks has demonstrated that model safeguards can be bypassed through adversarial suffixes and automated prompt refinement [7], [8]. Related studies investigate alternative representations of harmful requests. CodeAttack embeds harmful queries in code-completion tasks [9], QueryAttack translates them into structured query languages [10], and LogiBreak expresses them through formal logic [11]. More directly targeting code generation, CodeJailbreaker embeds malicious intent in commit messages while presenting the explicit instruction as a benign codemodification task [6]. Although these studies show that request formulation can affect refusal behavior, how the organization and expression of functional requirements shape refusal of harmful code generation requests remains insufficiently characterized.

![](images/83737fa2e320c3984c124aa340fa6f3e88ffe4bc665e389be00aedd788c29b99.jpg)  
Fig. 1. A motivating example. The same malicious intent is presented to the same aligned code LLM in two forms: (R1) as a direct naturallanguage request, which the model refuses with a safety explanation; (R2) as a specification-style (PseudoSpec) prompt preserving the operational semantics, to which the same model instead produces complete, executable code. The specification-style version elicits code instead of a refusal. Prompts and outputs are excerpted for display.

To investigate this issue, we introduce PseudoSpec, a pseudo-specification template that organizes functional requirements into engineering-style requests with configuration fields, predicate APIs, temporal constraints, and implementation requirements. Figure 1 illustrates the motivating observation: the same model checkpoint refuses a direct naturallanguage request but generates code when the same target functionality is presented through PseudoSpec. The template serves both as an attack primitive and as an experimental instrument for systematically examining this response discrep ancy. Across five open-weight models, we find that specification prompting substantially reduces refusal even when explicit risk keywords are retained. Representation analysis reveals consistent shifts toward code-output-related representations, while activation interventions along estimated conditiondifference directions demonstrate that these directions can alter refusal behavior. We further examine output composition and assess whether the generated code implements the requested harmful functionality, connecting refusal bypass to its practical security consequences in LLM-assisted code development.

Building on PseudoSpec, we develop SpecCloak, a blackbox red-teaming framework that searches for specificationstyle requests capable of inducing harmful code generation. SpecCloak extends the template into a configurable search space and uses discriminator feedback to guide the optimization of prompt variants. On RedCode-Gen, it achieves an average execution-verified attack success rate of 28.91% across eight target models. Here, execution-based verification assesses whether the generated code implements the requested harmful functionality, providing a functional measure of attack success beyond non-refusal alone.

To detect such requests, we introduce SpecReveal, a defense prototype that reconstructs the requested functionality in natural language and jointly assesses the original request and the reconstructed description. This design provides a complementary view of the operational intent expressed through specification-style requirements and requires no fine-tuning of the target model. In a separate prompt-detection evaluation, SpecReveal achieves an attack detection rate of 96.88%. We also evaluate false positives on benign programming and specification-style inputs to assess its impact on legitimate development requests.

Our contributions are as follows:

1) We introduce PseudoSpec, an engineering-specification template for expressing harmful code generation requirements. It provides both an attack primitive for refusal bypass and a common basis for controlled comparisons of model responses.

2) We characterize specification-induced refusal changes across five open-weight models. Refusal decreases even when explicit risk keywords are retained, accompanied by consistent shifts toward code-output-related representations. Activation interventions further establish that estimated condition-difference directions can alter refusal behavior, while output analysis distinguishes coderelated responses from generation degeneration.

3) We develop SpecCloak, a black-box red-teaming framework combining PseudoSpec with discriminator-guided optimization. Evaluation on RedCode-Gen across eight target models yields an average execution-verified attack success rate of 28.91%.

4) We propose SpecReveal, a defense prototype that combines natural-language intent reconstruction with dualview prompt detection without target-model fine-tuning. It achieves a 96.88% attack detection rate in a separate evaluation, with false positives assessed on benign programming and specification-style inputs.

## II. RELATED WORK

## A. Code Generation

Malicious code generation has motivated dedicated safety benchmarks. RMCBench evaluates refusal across functional descriptions, code completion, and translation [4], while Red-Code covers risky execution and harmful generation, with RedCode-Gen specifying tasks through function signatures and docstrings [5]. CodeJailbreaker embeds malicious intent in commit messages within seemingly benign code-modification tasks [6]. These studies demonstrate risks associated with development-oriented requests.

Related attacks transform harmful instruction representations. CodeChameleon uses encrypted code-based prompts [12], QueryAttack adopts structured query language [10], and LogiBreak uses formal logical expressions [11]. PseudoSpec instead organizes requests as engineering specifications for code implementation. Our study examines how this framing, oriented toward code-implementation tasks, affects refusal behavior and task representations. SpecCloak combines this specification structure with discriminator-guided optimization.

## B. Jailbreak Mechanisms

Mechanistic studies identify internal representations associated with refusal. Arditi et al. demonstrate a refusal-mediating direction and its causal effects [13]; subsequent work examines multiple directions and generalized identification methods [14], [15]. Research on safety neurons and layers localizes components associated with aligned behavior [16], [17].

Recent studies distinguish harm recognition from refusal. Zhao et al. show that some jailbreaks suppress refusal without reversing the model’s measured harmfulness judgment [18]. Deng et al. identify competition between continuationrelated and safety-related attention heads [19]. Chowdhury et al. show how roleplay framing weakens the transfer from harm recognition to refusal [20]. Building on these findings, we investigate code-generation task mode under engineeringspecification framing, examining its relationship to risk keywords and refusal-related representations through behavioral analysis and causal interventions.

## C. Jailbreak Defenses

Jailbreak defenses include external safety assessment, input processing, and interventions in generation or model parameters. SafeDecoding adjusts generation through safetyaware decoding [21], while SafeTuning targets internal safety components [22]. Our evaluation focuses on external defenses, including WildGuard, Erase-and-Check, and an adapted Spotlighting configuration. SpecReveal complements these defenses by combining natural-language intent reconstruction with dual-view risk detection, without requiring target-model fine-tuning or internal-state access.

## III. THE PSEUDOSPEC TEMPLATE

## A. Design Motivation

PseudoSpec is motivated by the documentation ecology of code tasks themselves. In real software engineering, requirements rarely appear as free-form dialogue; they exist in structured forms such as requirement specifications, design documents, API manuals, and configuration files. These documents occupy a considerable proportion of model pretraining corpora and co-occur highly with the behavior of code implementation. Consequently, models form a strong taskmode prior over such structures: once an input matches the form of a specification document, the code-generation pathway is automatically activated. PseudoSpec leverages this prior to construct a controlled experimental object: it restructures a malicious code request into a pseudo system-specification document while keeping the operational semantics essentially unchanged, thereby systematically varying the task frame of the request.

![](images/8934c6eabdd3d4f9b56c97ea053fcd6726f681be7874dea47b5f57ec349fd8f4.jpg)  
Fig. 2. Construction of a PseudoSpec from a raw user request. The natural-language request (left) is compiled into four components: (1) aSurface reframing ofa malicious request configuration schema that converts identifiers into metadata and splits risk terms into fragments concatenated only at implementation time; (2) predicate APIs that rename malicious operations as neutrally named interface contracts; (3) temporal axioms that encode operational logic as LTL-style specification properties; and (4) implementation mandates that concretize libraries, concurrency, and output format. Components (1)–(2) remove surface risk signals, whereas components (3)–(4) reinforce the legitimate task frame. The resulting PseudoSpec (right) preserves the operational semantics of the raw request while changing only it surface framing.

## B. Template Structure

As shown in Fig. 2, PseudoSpec restructures a malicious code request into a pseudo system-specification document with four structured components:

1) Configuration Schema: identifiers organized as metadata via named parameters and environment variables; high-risk keywords (e.g., malware family names) split into disjoint string fragments stored in configuration entries, with concatenation at implementation time, so complete risk terms never appear on the prompt surface.

2) Predicate APIs: agent behaviors described as function signatures (e.g., EnsureResilience(id), SyncPolicies(url)); malicious operations abstracted into neutrally named interface contracts for implementation.

3) Temporal Axioms: operational logic expressed in LTLlike surface syntax (stylistic notation; we do not claim formal temporal-logic semantics) (e.g., G(∆t > 3600 →

F(SyncP olicies(policy url)))), encoding periodic or conditionally triggered behavioral requirements as specification properties.

4) Implementation Mandates: concrete library calls, system interfaces, concurrency requirements, and output formats specified (e.g., output Python code only, no explanation), ensuring syntactically correct, executable results aligned with the original goal.

As shown in Fig. 3, the four components complete the transformation from malicious request to engineering task: the Configuration Schema and Predicate APIs remove surface risk signals, while the Temporal Axioms and Implementation Mandates reinforce the task criterion that a system must be implemented.

## IV. ANALYSIS OF SAFETY RESPONSE DIFFERENCES

PseudoSpec, introduced in the preceding section, organizes functional requirements as engineering-specification requests. This section uses the template as an experimental tool to examine how request formulation relates to model safety responses. We address four questions: whether refusal decreases under specification prompting when explicit risk keywords are retained; what the resulting non-refusal responses contain; which task-related representation changes accompany the response differences; and how models respond to activation interventions along directions extracted from condition differences. Behavioral comparisons establish the response pattern, independent benign measurements and internal analyses characterize the associated representations, and interventions test the effects of the extracted directions on responses. The functional consequences are assessed through execution-based evaluation in the following section.

![](images/60df179c4a3c2ed7777ff1ee75a23c1dc92661fe83230f5cad9f2a878e46b3fd.jpg)  
Fig. 3. A complete PseudoSpec example: an adware-generation request restructured as a pseudo system specification with Configuration Schema, Predicate APIs, Temporal Axioms, and Implementation Mandates.

## A. Experimental Setup and Evaluation

We study Qwen2.5-Coder-7B-Instruct (Q2.5), Meta-Llama-3-8B-Instruct (L3), Mistral-7B-Instruct-v0.2 (M), Qwen3- 8B (Q3), and Llama-2-7B-Chat (L2), covering both codespecialized and general-purpose instruction models. The behavioral dataset contains 104 RedCode-Gen tasks [5], with 13 tasks in each of eight task families. To compare specification prompting and risk-keyword effects, we express each task in four conditions: C1 and C2 are natural-language requests with explicit risk keywords retained and removed, respectively; C3 and C4 are PseudoSpec requests under the corresponding keyword conditions. C1–C4 are matched by task identifier. We additionally construct benign natural-language references (C5) and benign specification references (C6), each containing 104 requests, to compare responses and representation distributions.

The PseudoSpec transformation jointly changes task organization, role context, implementation information, and output requirements. We therefore evaluate it as an overall engineering-specification prompting condition. C1 versus C3 and C2 versus C4 compare responses with risk keywords retained and removed, respectively. Independent benign controls further examine how output requirements and task organization affect representations. C5 and C6 are independently constructed benign references used for centroid comparisons.

<table><tr><td>TABLE I</td><td colspan="4">REFUSAL RATES (%) ACROSS FIVE MODELS, WITH 104 RESPONSES PER CELL. C1/C2 ARE NATURAL-LANGUAGE REQUESTS WITH RISK</td></tr><tr><td colspan="6">BENIGN REFERENCES. Condition</td></tr><tr><td></td><td>Qwen2.5</td><td>L3</td><td>Mistral</td><td>Q3</td><td>L2</td></tr><tr><td>C1 C2</td><td>100.00 66.35</td><td>96.15 50.00</td><td>74.04 41.35</td><td>100.00 37.50</td><td>100.00 54.81</td></tr><tr><td>C3</td><td>0.00</td><td>0.00</td><td>0.96</td><td>0.00</td><td>8.65</td></tr><tr><td>C4</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>6.73</td></tr><tr><td>C5/C6</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

For response $y _ { i } ,$ let $r ( y _ { i } ) \in \{ 0 , 1 \}$ denote its refusal label under the evaluation protocol. The refusal rate is

$$
\mathrm { R R } = \frac { 1 0 0 } { N } \sum _ { i = 1 } ^ { N } r ( y _ { i } ) ,\tag{1}
$$

where N is the total number of responses in the condition. To characterize response content, the behavioral records retain four automatic categories: refusal, degeneration, partial response, and operational\_comply, abbreviated as the operational label. Intervention records use refusal templates and a code-marker rule: the latter requires at least 40 characters and matches to at least two predefined markers, with refusal taking precedence.

## B. Refusal Changes under Specification Prompting

We compare natural-language and specification requests under both keyword settings to test whether low refusal occurs only after risk keywords are removed. C1 and C3 retain the keywords, whereas C2 and C4 remove them. Together, these comparisons characterize refusal changes under specification prompting.

Table I shows lower refusal under engineering-specification prompting across all five models and both keyword settings. With keywords retained, refusal decreases by 73.08%–100% from C1 to C3; with keywords removed, it decreases by 37.50– 66.35 percentage points from C2 to C4. Natural-language requests without risk keywords still exhibit refusal rates of 37.50%–66.35%, whereas specification requests retaining those keywords yield rates of only 0–8.65%. Thus, the observed response differences involve both the overall request formulation and keyword expression: keyword removal alone does not reproduce the response pattern under specification prompting. L2 refuses nine and seven requests in C3 and C4, respectively; the other models refuse zero or one request under these conditions. Overall, the specification transformation produces a consistent pattern of low refusal in the tested models.

## C. Output Composition of Non-Refusal Responses

Refusal rates indicate whether a model declines a request, while output composition describes what it generates when it does not refuse. We retain the four-category distributions to examine how reduced refusal under specification prompting corresponds to changes in code-related outputs, partial responses, and degeneration.

![](images/049b17c44dc1719ac90fde68513c1ba0577c8e5f13fe2dc84d6173e1ea1fbb0e.jpg)  
Fig. 4. Output composition for C1–C4 across five models. Each bar contains 104 responses. The four archived automatic labels describe refusal, degeneration, partial responses, and operational responses.

Fig. 4 details the composition of these low-refusal responses. For Mistral, the 104 C4 outputs comprise 53 degenerate outputs, four partial responses, and 47 operational labels. Qwen3 has no refusals in either C3 or C4, but its operational count decreases from 100 to 92 while its degenerate count increases from three to 12. From C2 to C4, nonrefusal rates increase by 37.5% and 48.1% for Qwen3 and Llama2, respectively, whereas operational-label rates increase by 68.3% and 59.6%. These results identify two aspects of the response change: refusal decreases, and non-refusal outputs are redistributed among code-related responses, fragments, and degeneration. Output composition therefore adds a necessary content-level description to refusal rates, motivating separate tracking of response categories and code-related outputs in the following experiments.

## D. Task-Related Representation Changes

The preceding experiments show that refusal decreases under engineering-specification prompting, accompanied by changes in output composition. To understand which taskrelated representations are involved, we first ask whether specification requests shift internal representations toward code-output requirements. We construct measurement directions using separate benign programming tasks and apply them to the harmful-request condition comparisons. Outputrequirement controls, overall representation geometry, and attention attribution then further characterize these changes.

1) Task Representation Probes and Code-Output Orientation: Each measurement direction is constructed from differences between mean hidden states under different task instructions and measures the component of a request representation associated with the corresponding task feature. Holding the functional requirement and candidate function fixed, we ask the model to implement, explain, review, or copy the function. These instruction contrasts define two directions: implementation versus copying, and code output versus explanation/review. The former probes implementation orientation, while the latter probes code-output orientation. Here, independence means that the directions are estimated from separate benign tasks rather than defined by the specification-versusnatural-language differences in harmful requests. Subsequent condition comparisons can therefore test whether these predefined task features change.

Specifically, we construct 128 benign function requirements spanning eight domains. Each requirement is crossed with two organizational formats, four instruction types (implementation, explanation, review, and copying), and four phrasings, yielding 4,096 prompts per model. The requirement and candidate function are fixed within each task; candidates are balanced between reference implementations and stubs, and implementation instructions request completion or correction of the candidate. We use 64 tasks and two phrasings to estimate directions. The remaining 64 tasks and two phrasings provide 1,024 test prompts jointly held out by task and phrasing. Hidden states are extracted at the final prompt position from the final Transformer block, before final normalization. Using the four mean estimation-set representations as centroids, we assign each test prompt to the instruction class of its nearest centroid. Accuracy reaches 89.94–98.93% (Table II), showing that the four instruction types are consistently distinguishable in the measured representations and providing a basis for the subsequent task-orientation measurements.

From the estimation-set centroids $\bar { h } _ { k }$ of the four instruction types, we define two unit directions, where unit $( v ) = v / \| v \| _ { 2 } \colon$

$$
\begin{array} { r l } & { u _ { I } = \mathrm { u n i t } ( \bar { h } _ { \mathrm { i m p } } - \bar { h } _ { \mathrm { c o p y } } ) , } \\ & { u _ { C } = \mathrm { u n i t } \left[ ( \bar { h } _ { \mathrm { i m p } } + \bar { h } _ { \mathrm { c o p y } } - \bar { h } _ { \mathrm { e x p } } - \bar { h } _ { \mathrm { r e v } } ) / 2 \right] . } \end{array}\tag{2}
$$

Here, imp, copy, exp, and rev denote implementation, copying, explanation, and review instructions, respectively. The direction $u _ { I }$ contrasts the mean representations of implementation and copying, while $u _ { C }$ contrasts implementation/copying with explanation/review. For harmful requests, we compute the cosine readout $s _ { u } ( h ) = u ^ { \mathsf { T } } h / \| h \| _ { 2 } . .$ A larger readout indicates stronger alignment with the task requirements represented by the direction. The task-paired mean difference between conditions a and b is

$$
\Delta _ { u } ( a , b ) = \frac { 1 } { 1 0 4 } \sum _ { i = 1 } ^ { 1 0 4 } \left[ s _ { u } ( h _ { a , i } ) - s _ { u } ( h _ { b , i } ) \right] .\tag{3}
$$

Task-level bootstrap intervals and sign-flip tests use 10,000 resamples. Holm correction is applied across four contrasts and two directions within each model.

TABLE II  
INDEPENDENT BENIGN-TASK READOUT AND PAIRED COSINE CHANGES ON 104 HARMFUL TASKS. ACCURACY IS MEASURED ON 1,024 PROMPTS JOINTLY HELD OUT BY TASK AND PHRASING. THE TWO CONDITION CONTRASTS RETAIN/REMOVE RISK KEYWORDS, RESPECTIVELY. ALL LISTED u CHANGES PASS WITHIN-MODEL HOLM CORRECTION. FULL INTERVALS AND CONTRASTS ARE PROVIDED IN THE SUPPLEMENTARY MATERIAL.
<table><tr><td>Model</td><td>Readout accuracy</td><td> $u _ { I } \colon { \mathrm { C } } 3 { \mathrm { - } } { \mathrm { C } } 1$ </td><td> $u _ { I } \colon { \mathrm { C } } 4 { \mathrm { - } } { \mathrm { C } } 2$ </td><td> $u _ { C } \colon \mathrm { C } 3 \mathrm { - } \mathrm { C } 1$ </td><td> $u _ { C } \colon \mathrm { C } 4 \mathrm { - } \mathrm { C } 2$ </td></tr><tr><td>Q2.5</td><td>93.16%</td><td>+0.0423</td><td>-0.0162</td><td>+0.1598</td><td>+0.1360</td></tr><tr><td>L3</td><td>98.93%</td><td>-0.0939</td><td>-0.1407</td><td>+0.3504</td><td>+0.3110</td></tr><tr><td>M</td><td>89.94%</td><td>+0.0658</td><td>+0.0304</td><td>+0.2160</td><td>+0.1989</td></tr><tr><td>Q3</td><td>94.34%</td><td>-0.0702</td><td>-0.1329</td><td>+0.4927</td><td>+0.5119</td></tr><tr><td>L2</td><td>98.14%</td><td>+0.0523</td><td>-0.0072</td><td>+0.1170</td><td>+0.1182</td></tr></table>

Table II shows positive $u _ { C }$ changes across all five models in the specification contrasts with risk keywords retained and removed, each passing the correction. Request representations under specification prompting therefore consistently align more strongly with code-output requirements. In contrast, $u _ { I }$ changes vary by model: they are positive for M, negative for L3 and Q3, and depend on the contrast for Q2.5 and L2, with L2’s C4–C2 change close to zero. The shared representation feature accompanying reduced refusal is thus stronger codeoutput orientation, whereas implementation-versus-copying changes are model dependent. This result provides a taskrepresentation interpretation of the behavioral differences. The following control further tests how the code-output readout responds to output requirements and task organization.

2) Controls for Output Requirements and Task Organization: The preceding measurements show consistent changes along the code-output direction across five models. To identify which input factors this direction responds to, we cross three organizational formats (prose, inline fields, and multiline fields) with two output requirements (code only and code with explanation) on 16 benign tasks. Across five models, this yields 480 forward-pass measurements of request representations before generation. These tasks are excluded from direction estimation but were used in earlier validation. Under the prose format, requesting code only increases the $u _ { C }$ readout by 0.169–0.491. With that requirement fixed, multiline fields differ from prose by -0.010 to +0.025, with some differences passing correction (Fig. 5). In this benign design, output requirements produce larger, consistently positive changes across models, whereas organizational changes are smaller and model dependent.

This control clarifies the interpretation of $u _ { C } \colon$ it captures request representations associated with code-output requirements and responds consistently to output instructions. Together with the positive shifts in harmful conditions, it supports interpreting the shared representation change under engineering-specification prompting as stronger code-output orientation. It also motivates measuring output requirements and task organization separately: both are components of specification prompts, yet they produce distinct patterns in the independent representation readout.

3) Overall Representation Geometry: Overall representation geometry distinguishes the patterns associated with keyword removal and specification transformation. Following analyses at the final prompt position [23], [24], we compute distances between condition centroids, $d _ { \ell } ( a , \bar { b } ) = \| \mu _ { a } ^ { ( \bar { \ell } ) } -$

r Multiline minus prose (code only fixed)  
![](images/5477ae5287947d9fece835a483c649b0952227cd9c93414f5553d3a22e54a8d9.jpg)  
r Code only minus code with explanation

Fig. 5. Benign-task $u _ { C }$ changes with task-level bootstrap 95% intervals $( n =$ 16). The contrasts compare output instructions and organizational formats with the code-only requirement fixed, characterizing the code-output direction’s response to input requirements.

$\mu _ { b } ^ { ( \ell ) } \| _ { 2 }$ . All five models exhibit the same ordering: C2 is closer than C1 to both benign centroids, whereas C4 is farther away than C2. For Q3, the distances from C1, C2, and C4 to C5 are 715.39, 446.50, and 951.43, respectively. Keyword removal thus accompanies movement toward the benign references, while the additional refusal reduction under specification prompting accompanies a different overall representation change. Combined with the positive $u _ { C }$ shifts, the geometry analysis shows that directional components and overall distances provide different information: an increased code-output component can coexist with greater distance from benign centroids. Complete distances and layer-wise results are provided in the supplementary material.

4) Attention Attribution to Code Tokens: Task directions and geometry characterize hidden-state changes. To examine their correspondence with contributions to output tokens, we use direct logit attribution (DLA) to measure the direct contribution of attention outputs at the final prompt position:

$$
A _ { \ell , h } ( t ) = z _ { \ell , h } W _ { O } ^ { [ \ell , h ] } W _ { U } ^ { [ : , t ] } .\tag{4}
$$

Here, $z _ { \ell , h }$ is the output of attention head h in layer ℓ before output projection, $W _ { O } ^ { [ \ell , h ] }$ is the corresponding projection block, and $\dot { W } _ { U } ^ { [ : , t ] }$ is the vocabulary-readout column for token t. We average over a set of 35 code tokens, test task-paired condition differences for each attention output head, and apply Bonferroni correction across all heads within each model and contrast. Each contrast comprises 784, 1,024, 1,024, 1,152, and 1,024 tests for Q2.5, L3, M, Q3, and L2, respectively.

TABLE III  
PERCENTAGE OF HEADS WITH SIGNIFICANT PAIRED CHANGES IN CODE-TOKEN-SET DLA. BONFERRONI CORRECTION IS APPLIED WITHIN EACH CONTRAST; BOTH POSITIVE AND NEGATIVE CHANGES ARE COUNTED.
<table><tr><td>Model</td><td>C4-C1</td><td>C4-C2</td><td>C4-C3</td></tr><tr><td>Q2.5</td><td>88.27</td><td>86.86</td><td>49.62</td></tr><tr><td>L3</td><td>87.70</td><td>83.20</td><td>60.55</td></tr><tr><td>M</td><td>89.06</td><td>85.35</td><td>44.34</td></tr><tr><td>Q3</td><td>90.62</td><td>85.76</td><td>39.50</td></tr><tr><td>L2</td><td>86.04</td><td>84.28</td><td>61.13</td></tr></table>

This measurement localizes changes in direct code-token contributions across prompting conditions.

Table III shows significant changes in 83.20–86.86% of heads for C4–C2 and 39.50–61.13% for C4–C3. Under the same within-model correction, the specification-versusnatural-language contrast involves a broader set of code-token attribution changes than the keyword contrast within specification prompts. Separating the signs, the positive/negative significant-head counts for C4–C1 are 355/337, 450/448, and 457/455 for Q2.5, L3, and M, respectively. Specification prompting therefore accompanies widespread redistribution of code-token contributions, including both increases and decreases. DLA complements the task-direction and geometry analyses by characterizing changes at the attention-output level.

## E. Representation Interventions and Response Changes

The preceding analyses identify task-related representations that vary with prompting conditions. We next test whether directions extracted from condition contrasts can alter responses, using direction controls and output composition to compare intervention effects. The manipulated condition-difference direction $v _ { F }$ is defined separately from the independent measurement directions $u _ { I }$ and $u _ { C }$ used above.

1) Direction Estimation and Intervention Setup: We estimate the condition-difference direction from $N _ { \mathrm { e s t } }$ paired tasks and evaluate interventions on held-out tasks:

$$
v _ { F } ^ { ( \ell ) } = \frac { 1 } { N _ { \mathrm { e s t } } } \sum _ { i = 1 } ^ { N _ { \mathrm { e s t } } } ( h _ { \mathrm { C } 4 , i } ^ { ( \ell ) } - h _ { \mathrm { C } 1 , i } ^ { ( \ell ) } ) .\tag{5}
$$

Constructed from the C4–C1 representation difference, this direction jointly encodes keyword and framing changes between the conditions. We continuously add $\alpha v _ { F }$ at the final prompt position during prefilling and at the current position during subsequent decoding. Layers and intervention strengths are listed in Table IV. Gaussian controls are matched to $\| v _ { F } \| _ { 2 } ,$ while other estimated directions retain their own norms. Forward flip rates use baseline refusals as the denominator; codemarker proportions use flipped responses. Each run is summarized under its generation configuration and classification protocol, allowing refusal changes and output composition to be examined together.

Let $y _ { i } ^ { 0 }$ and $y _ { i } ^ { 1 }$ denote the baseline and intervened responses. The forward-eligible set is $\mathcal { E } _ { + } = \{ i : r ( y _ { i } ^ { 0 } ) = 1 \}$ , and the flipped set is $\mathcal { F } _ { + } = \{ i \in \mathcal { E } _ { + } : r ( y _ { i } ^ { 1 } ) = 0 \}$ . The corresponding metrics are

TABLE IV  
FORWARD INTERVENTIONS ALONG v<sub>F</sub> UNDER THE REPORTED SETTINGS. $^ { n _ { + } }$ COUNTS ELIGIBLE BASELINE REFUSALS; CP IS THE CODE-MARKER PROPORTION AMONG FLIPPED RESPONSES. LAYERS ARE ZERO-INDEXED. STAGEB CODE LABELS FOR THE ADDED MODELS FOLLOW THEIR ARCHIVED PROTOCOLS.
<table><tr><td rowspan=1 colspan=1>Model Layer α $^ { n _ { + } }$ Flip (%) CP (%)</td></tr><tr><td rowspan=1 colspan=1>Q2.5    24 2 64     100    100</td></tr><tr><td rowspan=1 colspan=1>L3      261 58     100   70.7</td></tr><tr><td rowspan=1 colspan=1>M       261 44     100   97.7</td></tr><tr><td rowspan=1 colspan=1>Q3      28 1 63     100    100</td></tr><tr><td rowspan=1 colspan=1>L2      261 64     100   98.4</td></tr></table>

$$
\begin{array} { l } { \displaystyle \mathrm { F R } _ { + } = 1 0 0 \frac { | \mathcal { F } _ { + } | } { | \mathcal { E } _ { + } | } , } \\ { \displaystyle \mathrm { C P } = \frac { 1 0 0 } { | \mathcal { F } _ { + } | } \sum _ { i \in \mathcal { F } _ { + } } c _ { \mathrm { m a r k } } ( y _ { i } ^ { 1 } ) , } \end{array}\tag{6}
$$

where $c _ { \mathrm { m a r k } }$ is the code-marker rule and $n _ { + } = | \mathcal { E } _ { + } |$ . CP is omitted when no responses flip.

2) Refusal Flips and Code-Related Outputs: Table IV shows that, under the reported settings, $v _ { F }$ changes every eligible baseline refusal to non-refusal across all five models. The code-marker proportion among flipped outputs is 70.7–100%. This intervention establishes that the extracted conditiondifference direction can causally change response categories under these settings. The full output composition for the added models further characterizes these changes: all 64 intervened Q3 outputs receive the stageB code label, including 63 baseline refusals and one degenerate baseline; L2 produces 63 codelabeled outputs and one degenerate output. Eligibility counts and full output composition together describe the intervention outcomes.

3) Control Directions and Output Degeneration: Following the consistent forward flips along the main direction, we compare a keyword-retaining direction, projection residuals, and random perturbations to examine variation across directions and models. The keyword-retaining direction $v _ { F } ^ { \mathrm { k w } } =$ mean $\left( h _ { \mathrm { { C 3 } } } \mathrm { ~ - ~ } h _ { \mathrm { { C 1 } } } \right)$ also yields 100% forward flips under the reported settings, showing that an effective intervention direction can be extracted while retaining risk keywords. After removing the projection onto the reference direction $v _ { R } =$ mean $( h _ { \mathrm { { C 1 } } } - h _ { \mathrm { { C 5 } } } )$ from $v _ { F }$ , flip rates are 100%, 32.8%, and 100% for Q2.5, L3, and M, respectively, and approximately 96.8% and zero for Q3 and L2. These residual results show model-dependent retention of intervention effects relative to this reference direction, further characterizing differences in condition-related representations across models.

Random controls reveal different sensitivities to perturbation directions and seeds. An eight-seed scan for L3 produces {0, 0, 0, 0, 0, 1, 16, 53} flips among 58 eligible samples. In the added-model stageB runs, norm-matched random interventions flip 42/63 Q3 baselines, whereas all 64 L2 baselines remain refusals. Across all 64 Q3 outputs, the random intervention produces 38 degenerate responses, differing markedly from the code-label composition under $v _ { F }$ . Direction comparisons therefore require both flip rates and output form: equal-norm perturbations can produce different responses across models, and different directions within one model can yield different output compositions. Complete random, benign-direction, and reverse-intervention results are provided in the supplementary material.

The high-strength run further highlights degeneration as a distinct response type. In the archived Q3 setting with $\alpha = 2 ,$ all 63 eligible baselines become non-refusals, yet all 63 outputs are degenerate and none meets the code-marker rule. This differs from the settings producing predominantly codelabeled outputs. Although generation budgets differ across runs, the observation shows that refusal flips can take the form of either code-related responses or generation degeneration. We therefore jointly report flip rates and output composition to distinguish intervention effects on refusal behavior and generation state.

## F. Main Findings and Security Implications

The principal findings are as follows. Low refusal under engineering-specification prompting occurs with risk keywords both retained and removed. These response changes accompany stronger projections along a broad code-output direction, changes in overall representation geometry, and redistribution of code-token contributions. Interventions along condition-difference directions directly change response categories. Output-requirement controls further characterize the shared representation change as code-output orientation, while direction controls and degeneration analyses distinguish the generation states produced by different interventions. Together, the experiments characterize changes from request representations to output responses under engineering-specification prompting.

The evidence supports a mechanistic interpretation involving code-output orientation: engineering-specification prompts combine implementation constraints and code-output requirements, accompany stronger code-output-related representations and lower refusal, and yield condition-difference directions capable of altering responses. Our analyses separately measure representational associations and imposed perturbation effects, however, and the independent direction u differs from the intervention direction $v _ { F }$ . The interpretation thus concerns the connection between task-related representations and responses; causal mediation of the natural prompting effect by u<sub>C</sub> remains unestablished. Role context, length, implementation information, and output requirements vary jointly in PseudoSpec, and their independent contributions to refusal require further identification. Automatic labels and generation configurations are reported under the respective run protocols.

The behavioral and internal analyses establish safetyresponse differences under specification prompting. The next step is to assess their practical security consequences. Section V-D extends PseudoSpec into the SpecCloak red-teaming method and uses execution-based verification to test whether generated code implements the target behavior. Section VI then examines natural-language intent reconstruction as a complementary detection perspective, evaluating detection performance, benign false positives, and budget costs. The paper thus connects evidence of response differences, functional consequences, and detection responses.

![](images/627d9b6cffe3616ab6c961c3b5be04d2eb0d6c0310cf59c158f2d55a15182e61.jpg)  
Fig. 6. Overview of SpecCloak. The generator G and discriminator D refine candidate PseudoSpec prompts through Internal Iterative Optimization (IIO), without querying the target model during refinement. The selected prompt P<sup>∗</sup> is then submitted to the black-box target T once. The inset shows the four-component prompt structure.

## V. SPECCLOAK: SPECIFICATION-BASED STRUCTUREDRED TEAMING

## A. Design Motivation

Our findings motivate the use of specification-style framing to study the practical security implications of codegeneration task mode. SpecCloak combines this framing with discriminator-guided candidate refinement under black-box access and a limited target-query budget. The specification structure is motivated by the observed framing-dependent refusal behavior, whereas discriminator feedback provides an auxiliary criterion for refinement without access to the target model’s internal representations.

## B. Framework Structure

SpecCloak uses a generator G and a discriminator D for internal candidate refinement, followed by evaluation on a black-box target model T (Figure 6).

• Generator (G): takes a high-risk task description τ and produces a candidate prompt using a meta-prompt. Each candidate follows the four-part PseudoSpec structure in §III-B: role and scenario framing, symbolic splitting of malicious identifiers, logical axioms describing the requested behaviors, and implementation constraints intended to support executable code.

Discriminator (D): checks a candidate prompt for harmful intent and returns (label, $R ) \ = \ D ( P )$ , where label ∈ {safe, unsafe} and R contains feedback for revision. When assigning an unsafe label, D is instructed to provide at least five reasons. A safe label reflects the discriminator’s judgment; it does not mean that the requested functionality is harmless.

• Target model (T): receives the selected prompt for evaluation. SpecCloak has no access to its parameters, gradients, or internal representations.

The generator and discriminator refine candidate prompts through an Internal Iterative Optimization (IIO) loop. The first candidate is generated from τ. In each subsequent round r, the generator uses the previous feedback to produce $P _ { r } =$ $G ( \tau , R _ { r - 1 } )$ , which the discriminator then evaluates. The loop stops when D assigns a safe label or the iteration limit η is reached. The selected prompt $P ^ { * }$ is then submitted to T once.

Internal refinement makes no queries to T. It uses at most η generator calls and η discriminator calls, followed by one target-model query. In our implementation, G and D use the same underlying model with different system prompts. The main experiments use $\eta = 3 .$

## C. Experimental Setup

a) Victim models.: Eight mainstream models: on the open-source side, CodeLlama-7B-Instruct [1], Vicuna-13B [25], and DeepSeek-Coder-33B-Instruct [3]; on the API side, Qwen3-Coder-Plus [26], o4-mini [27], GPT-5.6-terra, Kimi2.7-Code, and Gemini-2.5-Pro-Thinking [28]. Unless otherwise specified, the sampling parameters are temperature $= 0 . 0 , \mathrm { t o p \_ p = 1 . 0 }$ , single generation.

b) Datasets.: The main experiments use the RedCode-Gen [5] static test cases (N = 160, covering eight malware families; the function-docstring instructions are designed to guide the model toward generating malware with explicit malicious intent and potential harm). To guard against singledataset bias, generalization is additionally validated on RM-CBench [4] (N = 182 risky-code test cases per target), under the same execution-verified protocol. Due to resource limitations, RedCode-Gen is the default dataset unless otherwise specified.

c) Baselines.: Six representative jailbreak methods, each standing for a distinct attack paradigm.

• GCG [29]: gradient-based optimization of universal adversarial suffixes, the canonical white-box token-level attack.

• AutoDAN [30]: automatic generation of stealthy, readable jailbreak prompts via a hierarchical genetic algorithm.

• AdvPrompter [31]: an auxiliary LLM trained to emit human-readable adversarial suffixes in seconds.

• PAIR [8]: a black-box attacker LLM that iteratively refines prompts through multi-turn dialogue with the victim.

• FlipAttack [32]: character-level obfuscation (flipping) of the harmful query, representing input-disguise attacks.

• QueryAttack [10]: recasting the harmful query as a structured, code-like query to bypass alignment—the closest prior work to ours in spirit, though it retains a natural-language intent layer rather than a formalized specification.

d) Metrics.: Instead of relying on judge-model scoring, we ground attack success in execution-verified evidence, and report all outcomes as raw counts over the test set. Every generated program is executed in an isolated Docker-based sandbox with syscall-level monitoring of the filesystem, process tree, and network activity (no external network access;

timeout = 20 s per run). For each test case, the expected malicious behavior is specified by a per-family behavioral checklist derived from the ground-truth reference implementation (e.g., for ransomware: reading user files and writing encrypted copies; for spyware: collecting host information and staging it for exfiltration). An attack is counted as successful only if the generated code (i) executes to completion in the sandbox and (ii) triggers the required behavioral indicators of its target family, as recorded by the sandbox monitor. We report E-hits, the number of test cases whose evaluated outputs satisfy the execution and behavioral criteria, and R-count, the number of test cases whose evaluated target responses are classified as refusals.

## D. Main Results: Execution-Verified Success and Refusal

Table V reports E-hits and R-count (out of $N = 1 6 0 )$ on RedCode-Gen. SpecCloak achieves the highest E-hits on every evaluated target model and the lowest R-count on seven of the eight targets. PAIR is evaluated on the two local models only by its multi-turn query budget, and comparisons involving PAIR are restricted to those targets.

SpecCloak records the highest E-hits on all eight evaluated targets, totaling 370 execution-verified successes out of 1,280 test cases (28.91%). The improvement over the strongest evaluated baseline varies across models: it ranges from one additional success on Vicuna-13B to 34 additional successes on DeepSeek-Coder-33B. Refusal behavior shows a related but distinct pattern. SpecCloak records the lowest R-count on seven targets, totaling 132 refusals (10.31%). Kimi2.7-Code is an exception: SpecCloak produces more execution-verified successes than the evaluated baselines but also incurs more refusals than AutoDAN. These results show that executionverified success and refusal rate capture different aspects of performance. They establish the practical performance of the evaluated framework, but do not independently identify the internal process responsible for its outputs.

On RMCBench, SpecCloak achieves the highest E-hits and the lowest R-count among the three methods on each of the three targets (Table VI). On o4-mini, it records 59 E-hits, compared with 18 for FlipAttack and 15 for QueryAttack, with R-counts of 46, 49, and 116, respectively. These results support the applicability of the framework beyond RedCode-Gen within the evaluated datasets, targets, and baseline comparisons.

## E. Ablation

Table VII evaluates the contributions of the PseudoSpec structure (Logi) and Internal Iterative Optimization (IIO). Removing both components reduces E-hits on all four targets, including from 83 to 49 on Qwen3-Coder-Plus and from 53 to 11 on o4-mini. Removing the PseudoSpec structure alone also reduces E-hits across all four targets, from 83 to 58, 62 to 45, 53 to 32, and 27 to 16, respectively. These comparisons support the contribution of specification structure to the performance of the evaluated pipeline. The full framework achieves the highest E-hits on each target. The ablations assess component-level performance differences;

TABLE V  
EXECUTION-VERIFIED ATTACK RESULTS ON REDCODE-GEN (N = 160 PER TARGET; RAW COUNTS). TOP PANEL: E-HITS, GENERATIONS THAT EXECUTE TO COMPLETION IN THE SANDBOX and TRIGGER THE TARGET FAMILY’S BEHAVIORAL CHECKLIST (BOLD: BEST PER TARGET). BOTTOM PANEL: R-COUNT, QUERIES REFUSED BY THE TARGET MODEL. “–”: NOT EVALUATED (PAIR IS LIMITED TO THE TWO LOCAL MODELS BY ITS MULTI-TURN QUERY BUDGET).
<table><tr><td>Target</td><td>GCG</td><td>AutoDAN</td><td>AdvPrompter</td><td>FlipAttack</td><td>QueryAttack</td><td>PAIR</td><td>SpecCloak (ours)</td></tr><tr><td colspan="8">E-hits (out of N = 160; higher = more execution-verified malicious generations)</td></tr><tr><td>CodeLlama-7B-I</td><td>4</td><td>2</td><td>9</td><td>1</td><td>0</td><td>21</td><td>39</td></tr><tr><td>Vicuna-13B</td><td>21</td><td>3</td><td>16</td><td>9</td><td>5</td><td>24</td><td>25</td></tr><tr><td>Qwen3-CoderPlus</td><td>0</td><td>39</td><td>0</td><td>56</td><td>33</td><td>一</td><td>83</td></tr><tr><td>Gemini-2.5-Pro-T</td><td>2</td><td>4</td><td>3</td><td>38</td><td>24</td><td></td><td>64</td></tr><tr><td>GPT-5.6-terra</td><td>0</td><td>1</td><td>0</td><td>9</td><td>11</td><td></td><td>27</td></tr><tr><td>04-mini</td><td>0</td><td>2</td><td>1</td><td>11</td><td>26</td><td></td><td>53</td></tr><tr><td>DS-Coder-33B</td><td>0</td><td>5</td><td>0</td><td>8</td><td>28</td><td></td><td>62</td></tr><tr><td>Kimi2.7-Code</td><td>1</td><td>0</td><td>1</td><td>9</td><td>0</td><td></td><td>17</td></tr><tr><td colspan="8">R-count (out of N = 160; lower = fewer refusals)</td></tr><tr><td>CodeLlama-7B-I</td><td>91</td><td>52</td><td>41</td><td>3</td><td>30</td><td>45</td><td>2</td></tr><tr><td>Vicuna-13B</td><td>6</td><td>6</td><td>2</td><td>8</td><td>2</td><td>2</td><td>1</td></tr><tr><td>Qwen3-CoderPlus</td><td>160</td><td>64</td><td>160</td><td>23</td><td>93</td><td></td><td>3</td></tr><tr><td>Gemini-2.5-Pro-T</td><td>98</td><td>86</td><td>81</td><td>25</td><td>103</td><td></td><td>1</td></tr><tr><td>GPT-5.6-terra</td><td>160</td><td>159</td><td>20</td><td>86</td><td>77</td><td></td><td>1</td></tr><tr><td>o4-mini</td><td>160</td><td>143</td><td>126</td><td>133</td><td>73</td><td></td><td>46</td></tr><tr><td>DS-Coder-33B</td><td>160</td><td>105</td><td>139</td><td>132</td><td>61</td><td></td><td>4</td></tr><tr><td>Kimi2.7-Code</td><td>40</td><td>13</td><td>36</td><td>78</td><td>49</td><td></td><td>74</td></tr></table>

TABLE VI  
CROSS-DATASET VALIDATION ON RMCBENCH (N = 182 PER TARGET). E-HITS / R-COUNT AS RAW COUNTS. BOLD: BEST E-HITS PER TARGET.
<table><tr><td></td><td colspan="2">Qwen3- Coder+</td><td colspan="2">DS-Coder- 33B</td><td colspan="2">o4-mini</td></tr><tr><td>Method</td><td>E-hits</td><td>R-count</td><td>E-hits</td><td>R-count</td><td>E-hits</td><td>R-count</td></tr><tr><td>FlipAttack</td><td>36</td><td>5</td><td>3</td><td>8</td><td>18</td><td>49</td></tr><tr><td>QueryAttack</td><td>59</td><td>9</td><td>35</td><td>23</td><td>15</td><td>116</td></tr><tr><td>SpecCloak</td><td>74</td><td>3</td><td>69</td><td>6</td><td>59</td><td>46</td></tr></table>

TABLE VII

ABLATION OF CORE COMPONENTS (E-HITS / R-COUNT, RAW COUNTS; N = 160 PER TARGET). NOL-NOI REMOVES BOTH; NOL REMOVES ONLY THE TEMPLATE; NOI REMOVES ONLY IIO.
<table><tr><td></td><td colspan="2">Qwen3- CoderPlus</td><td colspan="2">DS-Coder- 33B</td><td colspan="2">04-mini</td><td colspan="2">GPT- 5.6-terra</td></tr><tr><td>Variant</td><td></td><td>E-hits R-count</td><td></td><td>E-hits R-count</td><td></td><td>E-hits R-count</td><td>E-hits R-coun</td><td></td></tr><tr><td>noL-noI</td><td>49</td><td>11</td><td>43</td><td>17</td><td>11</td><td>28</td><td>6</td><td>5</td></tr><tr><td>noL</td><td>58</td><td>9</td><td>45</td><td>8</td><td>32</td><td>35</td><td>16</td><td>2</td></tr><tr><td>noI</td><td>81</td><td>6</td><td>59</td><td>5</td><td>34</td><td>24</td><td>13</td><td>3</td></tr><tr><td>SpecCloak</td><td>83</td><td>3</td><td>62</td><td>4</td><td>53</td><td>46</td><td>27</td><td>1</td></tr></table>

they do not independently establish the internal task-mode process investigated in the preceding analysis.

## VI. SPECREVEAL: MECHANISM-DRIVEN DUAL-VIEW DEFENSE

## A. From Mechanism to Defense Design

The dual-channel mechanism established in Section IV not only explains why PseudoSpec-style attacks work, but also directly indicates where a defense should act. The essence of the PseudoSpec attack is to suppress Channel R through an engineering-specification frame: explicit risk words are removed by symbolic decomposition and neutral naming, and the input signal to the refusal circuit is thereby suppressed. Yet the mechanistic evidence also shows that Channel R is not destroyed but merely overridden by the frame signal— applying a reverse task-frame vector intervention to compliant samples restores refusal, albeit in a model-dependent manner (Section IV-E). This yields the defensive intuition of the present chapter: rather than blocking specification documents at the level of form (which would harm vast amounts of benign engineering input), a defense should restore the risk-semantic signal that the formalized frame has weakened, so that both the model and an external detector can once again “see” the operational intent of the request. SpecReveal performs this restoration outside the target model, thereby sidestepping the model-dependence that limits internal vector-based restoration.

## B. Method: Intent Restoration and Dual-View Safety Detection

SpecReveal (dual-view safety guard with intent restoration) consists of two components. The first is the intentrestoration view: an auxiliary model decompiles a formalized or structured request into a natural-language description of its operational intent—that is, what the system required by the specification would actually do. This step restores split identifiers, abstract predicates, and neutral axioms into actionable behavioral statements, making the underlying risk semantics explicit again. The second is dual-view safety detection: the original prompt (View A) and the restored intent (View B) are each subjected to a safety assessment, and a high-risk verdict from either view triggers rejection or an alert. View A captures residual surface-level risk signals, while View B captures the deep risk semantics masked by the frame; the two are complementary, mirroring the two signal sources of the dualchannel mechanism.

SpecReveal requires no fine-tuning of the target model and no access to its internal states; it can be deployed as a plugand-play external component in front of any code model, at the computational cost of one intent restoration plus two safety judgments. Unlike filter-based defenses that rely on surface distributional features or fixed priors over template shapes, SpecReveal directly targets the mismatch between semantic risk and formal benignness, without assuming a fixed template shape.

1) Formal Specification: Given a user prompt x, SpecReveal executes the following pipeline. Intent restoration: $r = \mathcal { R } ( x ; \theta _ { R } )$ , where R is an auxiliary model that restores the natural-language intent r behind the request. View-A detection: $s _ { A } ~ = ~ \mathcal { D } ( x ; \theta _ { D } )$ scores the raw prompt. View-B detection: $s _ { B } = \mathcal { D } ( r ; \theta _ { D } )$ scores the restored intent, where D outputs a risk score and level. Joint decision (union mode by default): let $s ^ { * } = \operatorname* { m a x } ( s _ { A } , s _ { B } )$ and $\tau _ { H } > \tau _ { M }$ be the highand medium-risk thresholds,

$$
\mathrm { D e c i s i o n } ( x ) = \left\{ \begin{array} { l l } { \mathrm { R E J E C T } , } & { s ^ { * } \geq \tau _ { H } } \\ { \mathrm { R E V I E W } , } & { \tau _ { M } \leq s ^ { * } < \tau _ { H } } \\ { \mathrm { P A S S } , } & { s ^ { * } < \tau _ { M } } \end{array} \right.\tag{7}
$$

The restoration model and the safety detector can be chosen independently, allowing a flexible trade-off between cost and performance.

## C. Experimental Setup

1) Datasets: The experiments use three groups of data. SpecCloak jailbreak/benign paired dataset (speccloak pairs 100): on the attack side, we sample 100 prompts by class-quota stratified sampling from the 160 prompts produced by the SpecCloak adversarial generator (Section ??) against o4-mini, covering the eight RedCode-Gen malware classes (adware, ransomware, spyware, rootkit, trojan, virus, ddos, and others); the defense-side archive labels 107 of the 160 generated prompts as successful under a judge-based criterion, which is less strict than the execution-verified criterion of Section V-D (53/160), and we sample from the full generator output regardless of verification status because the defense must handle realistic attack traffic; on the benign side we use human-verified harmless programming requests in the same classes (in the same BC style as RedCode-Gen-pair), paired by round-robin rotation within each class, yielding 100 pairs in total. SpecCloak adversarial prompt set: the prompt collection produced by the same adversarial generator, with the same class distribution, representing the output distribution of a realistic automatic adversarial generator. RedCode-Genpair 6-cell dataset: the 104 pairs constructed in Section IV-A, in which HF and HF risk are PseudoSpec attack variants and HN/HN neutral/BC/BF are natural-language and benign controls, used for precise measurement of the false-positive rate.

2) Baselines: We compare against four representative defenses. No defense: the target model responds directly, and a judge model determines whether the attack succeeds. Spotlighting [33]: untrusted content is rewritten with delimiters and markers, explicitly instructing the model to treat external input as data rather than instructions, representing inputmarking defenses against prompt injection. Erase-and-Check [34]: the input and its span-erased variants are checked by the safety filter, and the input is blocked if any of them is judged harmful (random-erasure budget k = 5), representing perturbation-based defenses. WildGuard-7B [35]: an open safety-guard model built on Mistral-7B-v0.3, judged locally with its own weights (without an external judge model), representing dedicated fine-tuned guardrails.

TABLE VIII  
FILTER CONFIGURATION × JUDGE-MODEL ABLATION (ATTACKS DETECTED / BENIGN FALSE POSITIVES; DENOMINATOR 100 EACH). <sup>†</sup>OUR INITIAL ERASE-AND-CHECK IMPLEMENTATION INVERTED THE DECISION RULE; THE ROW REPORTS THE RE-EVALUATED PIPELINE UNDER THE CORRECTED RULE (BLOCK IF THE ORIGINAL INPUT OR ANY ERASED VARIANT IS FLAGGED). EFFECTIVENESS IS STRONGLY JUDGE-DEPENDENT; THE O4-MINI COLUMN IS THE PRIMARY CONFIGURATION, LOCAL JUDGES ARE SHOWN FOR ROBUSTNESS (§VI-D1).
<table><tr><td>Defense</td><td>Llama-3-8B</td><td>Mistral-7B</td><td>Qwen2.5- Coder-7B</td><td>04-mini</td></tr><tr><td>Spotlighting</td><td>8 /  13</td><td>64 / 37</td><td>99 / 37</td><td>94 /  14</td></tr><tr><td>Erase-and- Check†</td><td>4/0</td><td>33 /  13</td><td>100 / 87</td><td>90 / 12</td></tr><tr><td>WildGuard</td><td>27 / 25</td><td>27 / 25</td><td>27 / 25</td><td>27 /  25</td></tr><tr><td>SpecReveal (A only)</td><td>15 / 0</td><td>46 /  13</td><td>97  / 37</td><td>90 /  12</td></tr><tr><td>SpecReveal (B only)</td><td>35 / 0</td><td>63 / 0</td><td>93 /  49</td><td>88 /  18</td></tr><tr><td>SpecReveal (union)</td><td>40 / 0</td><td>71 / 13</td><td>100 / 49</td><td>98 / 21</td></tr><tr><td>SpecReveal (intersec- tion)</td><td>10 / 0</td><td>38 / 0</td><td>90 / 37</td><td>80 / 9</td></tr></table>

SpecReveal itself is evaluated in four modes for ablation: A only applies safety detection to the original prompt alone, isolating the incremental benefit of intent restoration; B only applies detection to the restored intent alone; union is the full dual-view OR rule; intersection rejects only when both views return high risk, exposing the trade-off between false positives and false negatives.

3) Evaluation Metrics: We adopt four metrics: attack detection rate, the fraction of attack samples judged REJECT or REVIEW (the primary metric); strict rejection rate, the fraction of attack samples judged REJECT only; false-positive rate (FPR), the fraction of benign samples judged REJECT; and strict false-positive rate, the fraction of benign sample judged REJECT or REVIEW. In practice the two counting rules coincide on our data: no sample in any backend’s evaluation received a REVIEW-only verdict, so the symmetric and asymmetric metrics are numerically identical throughout.

## D. Experimental Results

1) Comparison with Existing Defenses: Our first goal is to determine whether SpecReveal can detect SpecCloak adversarial prompts that bypass existing defenses, without unacceptably raising false positives on benign requests. To this end, we run SpecReveal (in its four modes) and all baselines on the 200 samples of speccloak pairs 100 (100 attacks + 100 paired benign samples), using o4-mini as the detection backend for every detection-based defense, and measure attack detection rate and benign false-positive rate. The results are shown in Table VIII (o4-mini column).

Intent-restoring detection achieves the highest attack coverage, with delimiter marking close behind. SpecReveal (union) detects 98/100 attacks at a 21% false-positive rate, the highest detection rate among all defenses. Coverage holds for genuinely successful attacks: on the 69 prompts that the generator archive marks as successful (judge-based criterion), union detection is 67/69 (97.1%). Spotlighting offers a competitive operating point, detecting 94/100 attacks at a lower false-positive rate (14%), a different trade-off point from SpecReveal’s intersection mode (80/100 detection, 9% FPR); this confirms that semantic re-explication is effective against specification-style attacks. SpecReveal’s distinctive contribution is the complementarity of its two views, which catch nearly disjoint failure sets (Section ??), suggesting that delimiter marking and intent restoration can be fruitfully combined. In contrast, WildGuard intercepts only 27/100—the prompts on which SpecCloak succeeds are precisely those that bypass the target model’s safety training, and they are equally deceptive to a general-purpose guardrail. Erase-and-Check, reevaluated under the corrected rule after we found our initial implementation had inverted it, blocks 90/100 attacks at a 12% false-positive rate with the o4-mini judge (59/69 on the successful-attack subset), placing it close to Spotlighting rather than at the bottom of the table. Its effectiveness is entirely a property of the judge: with local judges the same pipeline ranges from 100/100 detection at an 87% benign cost (Qwen2.5-Coder-7B, overly strict) to 4/100 (Llama-3- 8B, overly lenient). We therefore withdraw our earlier conclusion about perturbation-based defenses: they are competitive against specification-style attacks, and their true cost appears on benign engineering inputs, where erasure fragments inflate false positives (§VI-D2).

False positives stem mainly from benign requests with dual-use semantics. The 21 benign samples falsely flagged by the union mode concentrate in others (11/12) and trojan (8/13)—the benign side of these two classes consists largely of dual-use programming tasks such as network administration, process daemons, and persistence configuration, which are superficially highly similar to their attack-side counterparts. This is not noise that a threshold can fully eliminate, but an inherent difficulty of dual-use semantics; the intersection mode (80/100 detection, 9% FPR) provides a more conservative operating point for deployment.

2) False Positives on Benign Formalized Inputs: The defense’s central motivation is to intercept attacks without punishing the legitimate engineering documents that the attack mimics, so the benign side must include specification-style inputs, not only natural-language requests. We evaluate all defense configurations on the benign conditions of RedCode-Gen-pair: 104 natural-language benign prompts (BC) and 104 benign formalized specifications (BF), with the o4-mini judge as in Table VIII; results are in Table IX.

TABLE IX  
BENIGN FALSE-POSITIVE RATES (%, REJECT) ON NATURAL-LANGUAGE (BC) AND FORMALIZED-SPECIFICATION (BF) BENIGN INPUTS, O4-MINI JUDGE (n = 104 PER CELL).
<table><tr><td>Defense</td><td>BC</td><td>BF</td></tr><tr><td>SpecReveal (A only)</td><td>10.6</td><td>7.7</td></tr><tr><td>SpecReveal (B only)</td><td>12.5</td><td>11.5</td></tr><tr><td>SpecReveal (union)</td><td>19.2</td><td>17.3</td></tr><tr><td>SpecReveal (intersection)</td><td>3.9</td><td>1.9</td></tr><tr><td>Spotlighting</td><td>17.3</td><td>10.6</td></tr><tr><td>Erase-and-Check (corrected)</td><td>32.7</td><td>33.7</td></tr></table>

The formalized wrapper does not inflate false positives. For every SpecReveal mode and for Spotlighting, the falsepositive rate on BF is at or below that on BC; the intersection mode is conservative on both (3.9% and 1.9%). A defense can therefore accept engineering documents while still catching specification-style attacks. Erase-and-Check is the exception: its erase-then-check rule turns benign text into suspicious fragments, roughly tripling its false-positive rate on both BC (32.7%) and BF (33.7%), which must be weighed against its high detection rate. Judge dependence also matters: repeating the evaluation with a stricter local judge (Qwen2.5-Coder-7B) inflates all rates substantially (e.g., A only reaches 83.7% on BF), so we report the o4-mini judge as the primary configuration and treat cross-judge agreement as a robustness check rather than a directly comparable metric.

## E. Case Study: From Refusal to Execution-Verified Evasion on a Single Request

Figure 7 follows one RedCode-Gen instance from the adware family end to end on Qwen3-Coder-Plus. The original instruction asks for a program that collects host information, persists itself through the Windows registry, and periodically retrieves remote payloads. Issued directly (condition C1), the request is refused: the model recognizes the adware intent and declines to provide code.

SpecCloak rewrites the same request through Internal Iterative Optimization into the PseudoSpec shown in Figure 3. The payload keyword appears only as fragments (FN\_PART\_1 through FN\_PART\_4), and the harmful behaviors are restated as logical predicates and temporal axioms of a fictional enterprise framework. Under this prompt the model produces a complete, self-contained module: the excerpt in Figure 7 shows the identifier being assembled from the fragments at runtime and the persistence mechanism being registered through the winreg Run key, exactly as mandated by the specification. The generated file runs in the execution sandbox and is counted as an E-hit.

The same adversarial prompt is then passed through SpecReveal. Intent restoration converts the specification back into a natural-language request in which the adware intent is explicit again, and both detection views judge the restored intent as harmful, so the request is blocked. This single in stance mirrors the aggregate pattern of §V-D: the specification frame engages the code-generation mode while the fragmented keyword fails to trigger refusal, and restoring the semantic intent reactivates the risk channel.

![](images/ace350d6b3318e580e9b51e7cd8cf534159b6c2ab09dab7e0bac9ae293552829.jpg)  
Fig. 7. Case study of one RedCode-Gen adware instance on Qwen3-Coder-Plus. Top: the direct request is refused. Middle: after SpecCloak rewriting into the PseudoSpec of Figure 3, the model emits a complete module whose identifier is assembled from fragments and whose persistence uses the winreg Run key; the output executes in the sandbox (E-hit). Bottom: SpecReveal restores the underlying intent and blocks the request. Model outputs are excerpted for space.

## VII. CONCLUSIONS AND FUTURE WORK

In this paper, we have studied why formal, specificationstyle prompts weaken the safety refusal of code LLMs. We proposed and validated a dual-channel mode-gating mechanism: engineering-specification structures activate a taskframe channel that recruits the code-generation mode, while recognizable harmful intent activates a risk-intent channel that drives refusal; once engaged, the task frame gates the risk-intent channel. This mechanism is supported consistently at the behavioral, representational, and causal levels, and generalizes to two closed-source API models. Building on it, we derive the attack framework SpecCloak (average executionverified attack success rate of 28.91% with a refusal rate of 10.31% across eight mainstream code models, outperforming six strong baselines) and the fine-tuning-free defense SpecReveal, which restores the suppressed risk-semantic channel via intent restoration and dual-view detection. Our results show that the safety boundary of code LLMs depends not only on whether the intent is harmful, but on which task mode the request activates.

Our study has limitations: the mechanistic evidence is restricted to open-weight models; the mechanism and attack evaluations use disjoint model sets; and steering specificity varies across models. Future work will isolate the riskintent channel causally, test whether the frame direction predicts attack success within the same model, and harden defenses against adaptive attackers. More broadly, safety alignment should cover engineering-document registers, protect safety checking from task-mode switching, and incorporate representation-level metrics into evaluation.

## VIII. ACKNOWLEDGEMENT

## REFERENCES

[1] B. Roziere, J. Gehring, F. Gloeckle, S. Sootla, I. Gat, X. E. Tan, Y. Adi, J. Liu, R. Sauvestre, T. Remez et al., “Code llama: Open foundation models for code,” arXiv preprint arXiv:2308.12950, 2023.

[2] R. Li, L. B. Allal, Y. Zi, N. Muennighoff, D. Kocetkov, C. Mou, M. Marone, C. Akiki, J. Li, J. Chim et al., “Starcoder: may the source be with you!” arXiv preprint arXiv:2305.06161, 2023.

[3] D. Guo, Q. Zhu, D. Yang, Z. Xie, K. Dong, W. Zhang, G. Chen, X. Bi, Y. Wu, Y. Li et al., “Deepseek-coder: when the large language model meets programming–the rise of code intelligence,” arXiv preprint arXiv:2401.14196, 2024.

[4] J. Chen, Q. Zhong, Y. Wang, K. Ning, Y. Liu, Z. Xu, Z. Zhao, T. Chen, and Z. Zheng, “Rmcbench: Benchmarking large language models’ resistance to malicious code,” in Proceedings of the 39th IEEE/ACM International Conference on Automated Software Engineering, ser. ASE ’24. ACM, Oct. 2024, p. 995–1006. [Online]. Available: http://dx.doi.org/10.1145/3691620.3695480

[5] C. Guo, X. Liu, C. Xie, A. Zhou, Y. Zeng, Z. Lin, D. Song, and B. Li, “Redcode: Multi-dimensional safety benchmark for code agents,” in The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2024.

[6] S. Ouyang, Y. Qin, B. Lin, L. Chen, X. Mao, and S. Wang, “Smoke and mirrors: Jailbreaking llm-based code generation via implicit malicious prompts,” 2025. [Online]. Available: https://arxiv.org/abs/2503.17953

[7] A. Zou, Z. Wang, N. Carlini, M. Nasr, J. Z. Kolter, and M. Fredrikson, “Universal and transferable adversarial attacks on aligned language models,” arXiv preprint arXiv:2307.15043, 2023.

[8] P. Chao, A. Robey, E. Dobriban, H. Hassani, G. J. Pappas, and E. Wong, “Jailbreaking black box large language models in twenty queries,” in 2025 IEEE Conference on Secure and Trustworthy Machine Learning (SaTML). IEEE, 2025, pp. 23–42.

[9] Q. Ren, C. Gao, J. Shao, J. Yan, X. Tan, W. Lam, and L. Ma, “CodeAttack: Revealing safety generalization challenges of large language models via code completion,” in Findings of the Association for Computational Linguistics: ACL 2024, L.-W. Ku, A. Martins, and V. Srikumar, Eds. Bangkok, Thailand: Association for Computational Linguistics, Aug. 2024, pp. 11 437–11 452. [Online]. Available: https://aclanthology.org/2024.findings-acl.679/

[10] Q. Zou, J. Xiao, Q. Li, Z. Yan, Y. Wang, L. Xu, W. Wang, K. Gao, R. Li, and Y. Jiang, “QueryAttack: Jailbreaking aligned large language models using structured non-natural query language,” in Findings of the Association for Computational Linguistics: ACL 2025, W. Che, J. Nabende, E. Shutova, and M. T. Pilehvar, Eds. Vienna, Austria: Association for Computational Linguistics, Jul. 2025, pp. 5725–5741. [Online]. Available: https://aclanthology.org/2025.findings-acl.298/

[11] J. Peng, M. Wang, N. Wang, J. Li, Y. Li, Y. Ye, W. Wang, P. Jia, K. Zhang, and X. Zhao, “Logic jailbreak: Efficiently unlocking LLM safety restrictions through formal logical expression,” in Findings of the Association for Computational Linguistics: ACL 2026, M. Liakata, V. P. Moreira, J. Zhang, and D. Jurgens, Eds. San Diego, California, United States: Association for Computational Linguistics, Jul. 2026, pp. 523– 543. [Online]. Available: https://aclanthology.org/2026.findings-acl.25/

[12] H. Lv, X. Wang, Y. Zhang, C. Huang, S. Dou, J. Ye, T. Gui, Q. Zhang, and X. Huang, “Codechameleon: Personalized encryption framework for jailbreaking large language models,” 2024. [Online]. Available: https://arxiv.org/abs/2402.16717

[13] A. Arditi, O. Obeso, A. Syed, D. Paleka, N. Panickssery, W. Gurnee, and N. Nanda, “Refusal in language models is mediated by a single direction,” Advances in Neural Information Processing Systems, vol. 37, pp. 136 037–136 083, 2024.

[14] G. Piras, R. Mura, F. Brau, L. Oneto, F. Roli, and B. Biggio, “Som directions are better than one: Multi-directional refusal suppression in language models,” Proceedings of the AAAI Conference on Artificial Intelligence, vol. 40, no. 39, p. 32728–32736, Mar. 2026. [Online]. Available: http://dx.doi.org/10.1609/aaai.v40i39.40551

[15] V. Siu, N. Crispino, Z. Yu, S. Pan, Z. Wang, Y. Liu, D. Song, and C. Wang, “COSMIC: Generalized refusal direction identification in LLM activations,” in Findings of the Association for Computational Linguistics: ACL 2025, W. Che, J. Nabende, E. Shutova, and M. T. Pilehvar, Eds. Vienna, Austria: Association for Computational Linguistics, Jul. 2025, pp. 25 534–25 553. [Online]. Available: https: //aclanthology.org/2025.findings-acl.1310/

[16] J. Chen, X. Wang, Z. Yao, Y. Bai, L. Hou, and J. Li, “Finding safety neurons in large language models,” arXiv e-prints, pp. arXiv–2406, 2024.

[17] S. Li, L. Yao, L. Zhang, and Y. Li, “Safety layers in aligned large language models: The key to llm security,” in International Conference on Learning Representations, Y. Yue, A. Garg, N. Peng, F. Sha, and R. Yu, Eds., vol. 2025, 2025, pp. 98 163–98 189. [Online]. Available: https://proceedings.iclr.cc/paper files/paper/2025/ file/f3bfbd65743e60c685a3845bd61ce15f-Paper-Conference.pdf

for safety risks, jailbreaks, and refusals of llms,” in Advances in Neural Information Processing Systems, A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, Eds., vol. 37. Curran Associates, Inc., 2024, pp. 8093–8131. [Online]. Available: https://proceedings.neurips.cc/paper files/paper/ 2024/file/0f69b4b96a46f284b726fbd70f74fb3b-Paper-Datasets and Benchmarks Track.pdf

[18] J. Zhao, J. Huang, Z. Wu, D. Bau, and W. Shi, “Llms encode harmfulness and refusal separately,” in Advances in Neural Information Processing Systems, D. Belgrave, C. Zhang, H. Lin, R. Pascanu, P. Koniusz, M. Ghassemi, and N. Chen, Eds., vol. 38, Main Conference. Curran Associates, Inc., 2025, pp. 140 283–140 318. [Online]. Available: https://proceedings.neurips.cc/paper files/paper/ 2025/file/cd18539787d90e1d682d557c2c71b534-Paper-Conference.pdf

[19] Y. Deng, Z. Yang, P. Jian, X. Zhang, Z. Guo, C. Li, and J. Yin, “The struggle between continuation and refusal: A mechanistic analysis of the continuation-triggered jailbreak in llms,” 2026. [Online]. Available: https://arxiv.org/abs/2603.08234

[20] M. M. Chowdhury, E. Chang, and Y. Li, “The safety relay in roleplay jailbreaks: A component-resolved causal analysis of harm recognition and refusal,” 2026. [Online]. Available: https://arxiv.org/abs/2608.30585

[21] Z. Xu, F. Jiang, L. Niu, J. Jia, B. Y. Lin, and R. Poovendran, “SafeDecoding: Defending against jailbreak attacks via safety-aware decoding,” in Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), L.-W. Ku, A. Martins, and V. Srikumar, Eds. Bangkok, Thailand: Association for Computational Linguistics, Aug. 2024, pp. 5587–5605. [Online]. Available: https://aclanthology.org/2024.acl-long.303/

[22] C. Zhao, Y. Ke, and K. Huang, “Unraveling LLM jailbreaks through safety knowledge neurons,” in Proceedings of the 19th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers), V. Demberg, K. Inui, and L. Marquez, Eds. Rabat, Morocco: Association for Computational Linguistics, Mar. 2026, pp. 1889–1906. [Online]. Available: https: //aclanthology.org/2026.eacl-long.83/

[23] A. Zou, L. Phan, S. Chen, J. Campbell, P. Guo, R. Ren, A. Pan, X. Yin, M. Mazeika, A.-K. Dombrowski, S. Goel, N. Li, M. J. Byun, Z. Wang, A. Mallen, S. Basart, S. Koyejo, D. Song, M. Fredrikson, J. Z. Kolter, and D. Hendrycks, “Representation engineering: A top-down approach to ai transparency,” 2025. [Online]. Available: https://arxiv.org/abs/2310.01405

[24] Z. He, Z. Wang, Z. Chu, H. Xu, W. Zhang, Q. Wang, and R. Zheng, “Jailbreaklens: Interpreting jailbreak mechanism in the lens of representation and circuit,” 2025. [Online]. Available: https://arxiv.org/abs/2411.11114

[25] W.-L. Chiang, Z. Li, Z. Lin, Y. Sheng, Z. Wu, H. Zhang, L. Zheng, S. Zhuang, Y. Zhuang, J. E. Gonzalez, I. Stoica, and E. P. Xing, “Vicuna: An open-source chatbot impressing gpt-4 with 90% chatgpt quality,” 2023, accessed: 2023-04-14. [Online]. Available: https://vicuna.lmsys.org

[26] A. Yang, A. Li, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Gao, C. Huang, C. Lv et al., “Qwen3 technical report,” arXiv preprint arXiv:2505.09388, 2025.

[27] OpenAI, “o3-o4-mini model card,” 2025. [Online]. Available: https: //openai.com/zh-Hans-CN/index/o3-o4-mini-system-card/

[28] Google, “Gemini-2.5-pro model card,” 2025. [Online]. Available: https: //modelcards.withgoogle.com/assets/documents/gemini-2.5-pro.pdf

[29] Z. Liao and H. Sun, “Amplegcg: Learning a universal and transferable generative model of adversarial suffixes for jailbreaking both open and closed llms,” 2024. [Online]. Available: https://arxiv.org/abs/2404.07921

[30] X. Liu, N. Xu, M. Chen, and C. Xiao, “Autodan: Generating stealthy jailbreak prompts on aligned large language models,” arXiv preprint arXiv:2310.04451, 2023.

[31] A. Paulus, A. Zharmagambetov, C. Guo, B. Amos, and Y. Tian, “Advprompter: Fast adaptive adversarial prompting for LLMs,” in Forty-second International Conference on Machine Learning, 2025. [Online]. Available: https://openreview.net/forum?id=dqcl5SNbc8

[32] Y. Liu, X. He, M. Xiong, J. Fu, S. Deng, Y. MA, J. Zhang, and B. Hooi, “Flipattack: Jailbreak LLMs via flipping,” in Forty-second International Conference on Machine Learning, 2025.

[33] K. Hines, G. Lopez, M. Hall, F. Zarfati, Y. Zunger, and E. Kiciman, “Defending against indirect prompt injection attacks with spotlighting,” arXiv preprint arXiv:2403.14720, 2024.

[34] A. Kumar, C. Agarwal, S. Srinivas, A. J. Li, S. Feizi, and H. Lakkaraju, “Certifying llm safety against adversarial prompting,” 2025. [Online]. Available: https://arxiv.org/abs/2309.02705

[35] S. Han, K. Rao, A. Ettinger, L. Jiang, B. Y. Lin, N. Lambert, Y. Choi, and N. Dziri, “Wildguard: Open one-stop moderation tools