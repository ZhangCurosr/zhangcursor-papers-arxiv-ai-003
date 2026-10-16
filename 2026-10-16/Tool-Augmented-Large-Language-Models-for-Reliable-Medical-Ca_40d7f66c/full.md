# Tool-Augmented Large Language Models for Reliable Medical Calculation from Patient Notes

Yang Zhong <sup>1</sup>, Ling Luo<sup>1,\*</sup>, Xuanlyu Zhu<sup>1</sup>, Yanhua Wang<sup>2</sup>, Yuanyuan Sun<sup>1</sup>, Jian Wang<sup>1</sup>, Hongfei Lin<sup>1</sup>

<sup>1</sup>College of Computer Science and Technology, Dalian University of Technology, Dalian, China

<sup>2</sup>Air Force Communications NCO Academy, Dalian, China

To whom correspondence should be addressed: lingluo@dlut.edu.cn

Abstract—Medical calculators translate raw patient observations into clinically actionable values and scores. However, applying these calculators directly to free-form patient notes requires precise variable extraction combined with the exact execution of complex formulas, scoring rules, and output conventions. Relying solely on direct large language model (LLM) prompting is unreliable in this context, as rule recall, parameter extraction, arithmetic reasoning, and formatting become entangled within a single generative step. To address these limitations, we propose a tool-augmented LLM framework for reliable medical calculation from patient notes. The framework retrieves and structures calculator specifications, generates reusable executable tools, and uses a constrained extractor to map each patient note to calculatorready inputs. The tools then perform deterministic calculation, separating clinical language understanding from rule execution and arithmetic. Evaluated on the MedCalc-Bench dataset, our method achieves 86.73% final-answer accuracy, outperforming the one-shot prompting and supervised fine tuning-only baselines by 24.0 and 17.27 percentage points. Ablation and error analyses show that these performance gains are primarily driven by rule-grounded tool synthesis and schema-constrained extraction, whereas remaining failures are concentrated in clinicalsemantic extraction and benchmark-specific output contracts. By grounding calculator logic in executable tools, the proposed approach alleviates arithmetic hallucinations and improves the computational reliability of the final clinical outputs.

Index Terms—Medical Calculation, Large Language models, Tool Augmentation, Parameter Extraction, Patient Notes

## I. INTRODUCTION

Medical calculators convert patient observations into clinically meaningful values, risk scores, or treatment-relevant estimates. Clinicians use calculator resources such as MDCalc [1] to estimate renal function, stratify cardiovascular and thromboembolic risk, grade disease severity, estimate gestational age, and support medication dosing. However, performing these calculations manually can be time-consuming and prone to error, particularly in settings with limited clinical capacity. The projected global shortage of health workers further underscores the need for reliable methods that can reduce routine computational burden while preserving transparency and clinical oversight [2]. Automating medical calculations from free-form patient notes aims to provide reliable computational support in settings where clinician time is limited.

This task differs fundamentally from conventional medical question answering. A system must identify every variable required by the target calculator, including numerical measurements, categorical variables, Boolean criteria, dates, time intervals, and units. These variables may be expressed indirectly, distributed across a note, negated, or implied by diagnoses, medications, and clinical history. The system must then apply the appropriate formula, scoring rule, threshold, boundary condition, and output format. A missed risk factor, an incorrectly interpreted time window, or an improperly normalized unit can change the final result. Reliable medical calculation therefore requires both accurate clinical information extraction and deterministic rule execution, together with a trace that makes errors inspectable.

Large language models (LLMs) are promising interfaces for this task because they can interpret unstructured clinical text, follow instructions, generate structured representations, and interact with external tools. Nevertheless, direct prompting remains unreliable for medical calculation [3]. When formula recall, parameter extraction, mathematical arithmetic, and output formatting are entangled within a single generative step, a hallucinated formula or a minor arithmetic error can silently corrupt the final answer without revealing the source of failure. Prior work has sought to address these limitations through retrieval-augmented prompting, supervised or reinforcement learning, program synthesis, and tool-using agents. In particular, tool-augmented systems can improve numerical reliability by delegating arithmetic to validated calculators [4], [5]. However, existing approaches commonly rely on fixed, manually implemented calculator libraries, which limits coverage to calculators already available as tools. In addition, many systems still combine parameter extraction and calculation within the same reasoning loop, making it difficult to distinguish errors in clinical understanding from errors in calculator implementation or output formatting.

Taiyi demonstrates bilingual biomedical multitask adaptation through supervised fine-tuning [6]. A recent survey reviews collaboration between LLMs, symbolic solvers, and verifiers for mathematical problem solving [7].

To address these limitations, we propose a tool-augmented method that separates clinical language understanding from deterministic medical calculation. Our method first retrieves evidence for a target calculator and structures its specification, including required inputs, formulas or scoring rules, boundary conditions, and expected outputs. It then synthesizes a reusable executable calculator tool and validates the tool on examples from the same calculator, using execution feedback to identify and repair implementation errors. At inference time, a schema-constrained extraction model maps each patient note to calculator-ready structured inputs, and the tool performs the final computation deterministically.

Our contributions are as follows.

• We propose a tool-augmented framework for medical calculation from patient notes that combines rule-grounded retrieval, on-demand executable tool synthesis with validation and repair, schema-constrained parameter extraction, and deterministic tool execution. This decoupled architecture effectively mitigates formula hallucinations and arithmetic errors, and supports tool construction beyond predefined tool libraries.

• We establish a standardized evaluation on the corrected MedCalc-Bench benchmark using a unified data split and official metrics. We compare prompting baselines, multiple tool-writing systems, and extraction models under the same evaluation setting, avoiding comparisons aggregated across different benchmark versions.

• Experimental results show that the best configuration achieves 86.73% final-answer accuracy and a 97.47% parameter matching rate on 1,100 test cases, outperforming one-shot prompting baselines by 24.0 percentage points. Tool-level evaluation, extraction-model comparisons, category-level analyses, ablations, and case studies further characterize the sources of improvement and remaining errors.

The code for our framework is available online.

## II. RELATED WORK

## A. Medical Calculation Benchmarks

MedCalc-Bench [3] established medical calculation as an evaluation target distinct from knowledge-oriented medical question answering. Unlike MedQA [8], PubMedQA [9], MedMCQA [10], Med-PaLM [11], and HealthBench [12], a medical calculator is defined by an explicit equation, a scoring rule, or branch logic. Consequently, a model needs to accurately extract the required patient variables from free text and then apply that specific rule deterministically. This task therefore jointly evaluates structured clinical extraction and exact execution rather than medical knowledge alone. The MedCalc-Bench baselines still err frequently even when the target calculator is given, exposing a persistent gap between recalling a formula and applying it to a specific note [3]. The original MedCalc-Bench results show that LLMs still fail frequently even when the target calculator is given, with errors arising from entity extraction, rule selection, and arithmetic execution [3]. Subsequent audits further show that many original labels are erroneous, ambiguous, or incomputable, and that open-book access to calculator rules changes the interpretation of model performance [13], [14]. These findings motivate our use of the corrected MedCalc-Bench release as the primary benchmark and our emphasis on evaluation within one fixed dataset version and answer contract.

## B. LLMs for Medical Calculation

Existing approaches for LLM-based medical calculation can be grouped into three main lines: retrieval- and promptingbased methods, supervised training-based methods, and toolcalling methods [15]–[22]. First, retrieval and prompting methods keep the model frozen and restructure the input. MedRaC [23], for example, decomposes medical calculation into formula selection, entity extraction, calculation, and final-answer checking, and argues that final-answer accuracy alone cannot localize failures; however, its retrieval and Python-execution steps are still performed within the per-case inference pipeline, so formula-selection, code-execution, and answer-formatting errors can still propagate to the final answer. Second, supervised training-based methods encode calculator knowledge in model weights. MedCalc-R1 [24] combines supervised finetuning with a knowledge-guided GRPO reward, and MedCalc-Eval/MedCalc-Env build training data and an interactive environment [25]. These methods can improve accuracy, but calculator updates, revised rules, or changed answer contracts may require additional training. Third, tool-calling methods delegate computation to external calculators or tool libraries. Prior work has shown that tool use can substantially reduce numerical errors in clinical calculation settings [4], [5]. Related works such as RiskAgent [26], MeNTi [27], and MedMCP Calc [28] connect agents to calculators through retrieval, nested tool calling, and Model Context Protocol integration. These approaches improve numerical stability, but they often assume that the target calculator already exists as a manually implemented tool. Many also keep parameter extraction and tool invocation within the same agent loop, which makes it difficult to separate errors in clinical interpretation from errors in tool selection, tool implementation, or output formatting.

Our method differs from these methods in two respects. First, it does not assume a fixed, hand-built calculator library. Instead, it synthesizes an executable calculator tool from retrieved rule evidence and improves the tool through executionbased validation and repair. Second, it separates schemaconstrained parameter extraction from deterministic tool execution. This design reduces the coupling between clinical understanding and computation, while producing an auditable trace of the retrieved rule evidence, generated tool, extracted parameters, tool input, and tool output. As a result, errors can be localized more clearly to rule retrieval, tool construction, parameter extraction, or benchmark-specific output handling.

## III. METHOD

Our approach formulates medical calculation from patient notes as a tool-augmented workflow with three main steps shown in Figure 1: (1) rule-grounded calculator specification retrieval, (2) executable calculator tool synthesis and validation, and (3) schema-constrained parameter extraction and tool execution. Next, we will first describe the task definition and then detail each step.

![](images/3b2d410cc3f2e74f6eb6f58b3107b49287560a49af696bec56b95eb468bdd6f7.jpg)  
Fig. 1. Overview of the proposed method. It is illustrated with the Cockcroft–Gault creatinine-clearance calculator. The workflow retrieves and structures calculator rules, synthesizes and validates an executable calculator tool, extracts calculator inputs from a patient note, and executes the validated tool deterministically.

## A. Task Definition

Let an input be denoted by $x = ( n , q )$ , where n is a patient note, q is the calculation question. The goal is to produce an answer y that is consistent with the reference calculator c and the benchmark-specific output contract. Our workflow can be expressed as:

$$
( c , r _ { c } , S _ { c } ) = \mathrm { R e t r i e v e A n d S t r u c t u r e } ( q ) ,\tag{1}
$$

$$
\tau _ { c } = \mathrm { G e n e r a t e T o o l } ( c , r _ { c } , S _ { c } ) ,\tag{2}
$$

$$
z = \mathrm { E x t r a c t } ( n , q , c , S _ { c } , K _ { c } ) ,\tag{3}
$$

$$
y = \mathrm { E x e c u t e } ( \tau _ { c } , z )\tag{4}
$$

where $r _ { c }$ is the rule specification for calculator $c , \ S _ { c }$ is the calculator parameter schema, $\tau _ { c }$ is the generated executable tool, $K _ { c }$ is a set of same-calculator demonstrations, and $z$ is the structured parameter object extracted from the patient note. The LLM is responsible for organizing calculator rules, generating tools, and interpreting clinical language, whereas the validated executable tool performs the final medical calculation.

## B. Rule-Grounded Calculator Specification Retrieval

The first stage retrieves and structures the specification of the target calculator. First, based on the input question $q ,$ the system searches medical calculation resource websites such as MDCalc to find the corresponding target calculator $c .$ When a calculator specification has already been stored in memory, the workflow reuses the cached version. Otherwise, the knowledge retriever searches for calculator evidence together with benchmark-provided metadata where available. A rule-curation LLM converts the retrieved evidence into a structured calculator specification (Figure 2 (1) shows the specific prompt):

$$
r _ { c } = \mathrm { \{ n a m e , i n p u t s , f o r m u l a , c o n d i t i o n s , o u t p u t \} } .\tag{5}
$$

The corresponding calculator parameter schema $S _ { c }$ specifies the complete set of calculator inputs, their value types, allowed units, and permitted categorical values. The structured specification is stored with provenance information so that later cases involving the same calculator can reuse the retrieved evidence without repeated search.

## C. Executable Calculator Tool Synthesis and Validation

The second stage transforms the retrieved calculator specification into reusable executable code. Medical calculators may involve numerical formulas, categorical scoring rules, Boolean criteria, conditional branches, unit conversions, boundary conditions, rounding rules, and calculator-specific output conventions. Directly asking an LLM to recall and apply all of these requirements during answer generation can lead to incorrect formulas, omitted conditions, arithmetic errors, or format mismatches. We therefore convert the rule into Python code before test-time inference.

Given the structured rule specification $r _ { c }$ and parameter schema $S _ { c }$ , the tool-writing LLM generates a Python calculator function calc(relevant). The function validates and parses inputs, implements the required formula or scoring logic, handles relevant branches and boundary cases, and returns an output consistent with the calculator’s expected contract. Figure 2 (2) shows the tool-builder prompt.

Tool generation is iterative. Each candidate tool is evaluated on eight training examples associated with the same calculator. Runtime failures, missing-field errors, and output mismatches are returned to the tool-writing model together with the expected and observed results. The model may revise the implementation for up to four validation-and-repair rounds. Once validated, the tool is cached and reused for subsequent cases involving the same calculator. This design reduces repeated tool-generation calls and promotes consistent behavior across patient notes.

![](images/7f2623df08ef73430c0a9e3da512b687ad866c28e96077b79829b3bbdd82a890.jpg)  
Fig. 2. Condensed prompts used for calculator rule curation, executable tool synthesis, and schema-constrained parameter extraction.

Validation samples are grouped by calculator ID and eight are randomly drawn without replacement from the historical training set for each calculator. They are not selected by difficulty or field coverage. All repair rounds reuse the same eight samples, executing each tool with annotated parameters and comparing its outputs with reference answers. This sampling differs from the coverage-based selection of the three extraction demonstrations. Eight random checks do not guarantee branch or boundary coverage, and deterministic execution does not guarantee a correct tool implementation.

## D. Schema-Constrained Parameter Extraction and Tool Execution

The third stage operates on individual patient notes. The calculator tools are generated once per calculator and then reused, whereas the repeated inference-time task is parameter extraction. Therefore, we fine-tuned a smaller Qwen3- 8B model for this high-frequency stage, while reserving the tool-writing LLM for calculator construction and validation. This division can reduce computational cost and enable the extraction component to be deployed locally when required by data-governance constraints.

Before extraction, a case retriever selects $k \ = \ 3$ samecalculator examples, denoted by $K _ { c } ,$ and provides them together with the calculator parameter schema $S _ { c } .$ Calculatorspecific schemas are necessary because medical calculation requires more than generic entity recognition. Depending on the calculator, the required inputs may include continuous measurements, Boolean risk factors, disease categories, dates, gestational ages, time intervals, or values paired with units.

The extraction model receives the patient note $n ,$ the question $q ,$ the calculator c, the examples $K _ { c } ,$ , and the schema $S _ { c } .$ Figure 2 (3) shows the extraction task prompt. The constrained JSON object is returned:

$$
\begin{array} { r } { z = \{ s _ { 1 } : v _ { 1 } , \ldots , s _ { m } : v _ { m } \} , } \end{array}\tag{6}
$$

where each value may be a number, Boolean value, categorical label, date, time interval, or value-unit pair. The structured parameter object is then sent to the tool-serving interface, which loads the cached tool $\tau _ { c } ,$ executes the calculator using calc(z), and returns the final calculation result. The LLM does not perform the final formula application or arithmetic at this stage; these operations are executed deterministically by the executable tool.

We derive schemas from calculator specifications rather than from individual annotated examples. Across the 55 calculators in the benchmark, the schemas contain 328 required fields. For few-shot extraction, we begin with the public exemplar for each calculator and select two additional training examples greedily to cover required fields not represented in the initial example. This procedure yields three demonstrations per calculator and covers all 328 required fields without using test cases as demonstrations.

Memory serves as an auxiliary mechanism rather than the main methodological contribution. It stores calculator specifications, schemas, generated tools, selected examples, execution traces, and historical errors. This supports reuse, reduces repeated retrieval, and improves auditability.

## IV. EXPERIMENTS

## A. Dataset and Experimental Settings

We conduct controlled experiments on MedCalc-Bench Verified v1.0.7 [29], a corrected version of MedCalc-Bench that includes revised labels, answer contracts, calculator specifications, and executable reference implementations for 55 medical calculators. The dataset contains 10,538 training instances and 1,100 test instances. We do not use the original MedCalc-Bench v1.0 for evaluation because physician-supervised audits have identified erroneous, ambiguous, and incomputable cases in the original benchmark [13], [14].

We report three evaluation metrics for different modules. Final-answer accuracy is the proportion of test cases judged correct by the official benchmark evaluator. Parameter matching rate (PMR) is the proportion of required calculator parameters for which the extracted value matches the gold annotation. Tool-level accuracy evaluates the generated calculator tools using gold parameters, thereby isolating calculator implementation quality from parameter-extraction quality.

Unless otherwise specified, GPT-5.5 is used to generate calculator tools, and Qwen3-8B [30] is used for parameter extraction. Table I summarizes the main configuration for supervised fine-tuning of the extraction models.

TABLE I  
MAIN SUPERVISED FINE-TUNING CONFIGURATION FOR THE PARAMETER-EXTRACTION MODEL.
<table><tr><td>Item</td><td>Setting</td></tr><tr><td>Base models</td><td>Qwen3-8B</td></tr><tr><td>Tuning method</td><td>LoRA adapter on frozen base model</td></tr><tr><td>LoRA rank / alpha / dropout</td><td>16 / 32 / 0.05</td></tr><tr><td>Learning rate</td><td>5×10−5</td></tr><tr><td>Epochs</td><td>5</td></tr><tr><td>global batch size</td><td>8</td></tr><tr><td>Max sequence length</td><td>8,192</td></tr></table>

## B. Main Results

Table II presents the main results on the MedCalc-Bench Verified test set. The baseline results show that direct SFT and prompting-based reasoning strategies provide limited improvements when clinical parameter extraction, calculator-rule application, arithmetic, and output formatting must be handled within a single generation. Our method achieves 86.73% final-answer accuracy, outperforming the strongest one-shot prompting baseline and the SFT-only baseline based on the same Qwen3-8B backbone by 24.00 and 17.27 percentage points, respectively. These results indicate that decomposing the task and delegating rule execution and arithmetic to medical calculator tools substantially improves end-to-end performance.

The table also includes previously reported results as references. MedRaC is a training-free Formula-RAG and Pythonexecution pipeline [23]. The original paper reports an overall accuracy of 64.68% in its ablation section. Based on the surrounding discussion, this result follows the LLaMA3.1- 8B-Instruct analysis, and it is evaluated on the authors cleaned MedCalc-Bench set rather than the full MedCalc-Bench Verified v1.0.7 test set. The open-book rows provide calculator specifications at inference time, with the guided variant adding task-specific instructions [14]. Claude Sonnet 4.6 is reported under a code-augmented agent loop that uses a Python REPL for computation, averaged over five runs on the latest verified dataset [31]. Compared to these methods, our approach achieves competitive performance. Notably, some reported results were evaluated on a subset rather than the full 1,100-case Verified test split. Our workflow delegates calculator-rule interpretation, branch logic, rounding behavior, and output conventions to a reusable, validated tool. Consequently, the extraction model is responsible only for mapping the patient note to the required calculator schema. This separation prevents the need for a single generative step to be simultaneously clinically accurate, mathematically exact, and compliant with complex output conventions.

The external rows in Table II are contextual comparisons, whereas the rows from Direct prompting through Ours use our controlled evaluation. The 0.49-percentage-point difference from the externally reported 86.24% does not establish a statistically significant advantage.

TABLE II  
MAIN RESULTS ON MEDCALC-BENCH VERIFIED V1.0.7.
<table><tr><td>Method</td><td>Model</td><td>Score (%)</td></tr><tr><td>MedRaC [23]</td><td>LLaMA3.1-8B-Instruct</td><td>64.68</td></tr><tr><td>Open-book guided prompting [14]</td><td>GLM-4.6V</td><td>82.00</td></tr><tr><td>Open-book prompting* [14]</td><td>GLM-4.7</td><td>85.50</td></tr><tr><td>Code agent (Python REPL) [31]</td><td>Claude Sonnet 4.6</td><td>86.24</td></tr><tr><td>Direct prompting</td><td>Qwen3-8B</td><td>14.00</td></tr><tr><td>Zero-shot CoT</td><td>Qwen3-8B</td><td>31.55</td></tr><tr><td>One-shot prompting</td><td>Qwen3-8B</td><td>62.73</td></tr><tr><td>One-shot prompting</td><td>DeepSeek-R1-Distill-Qwen-7B</td><td>55.45</td></tr><tr><td>One-shot prompting</td><td>Llama-3.1-8B</td><td>38.36</td></tr><tr><td>SFT-only</td><td>Qwen3-8B</td><td>69.46</td></tr><tr><td>Ours</td><td>GPT-5.5 tool generation +</td><td>86.73</td></tr></table>

Note: <sup>\*</sup>denotes the results reported on the sub test set.

C. Effect of Generated Tool Quality on Medical Calculation Performance

Table III evaluates the effect of generated calculator-tool quality on final medical-calculation performance. Tool-level accuracy is measured by providing each generated tool with gold parameters. The final-answer accuracy then evaluates the same tool set with a fixed Qwen3-8B-SFT extractor. The

Per-category gap from PMR to final accuracy (Qwen3-8B-SFT, GPT-5.5 tools)

results show that deterministic execution improves reliability when the generated calculator tool faithfully implements the relevant calculator rule. GPT-5.5 tools generated with rulegrounded retrieval and validation-and-repair achieve the highest tool-level accuracy and final-answer accuracy. Compared with tools generated by Qwen3.5-Plus, the GPT-5.5 tools show a clear performance advantage and produce no runtime errors. Moreover, removing web retrieval and iterative repair reduces both tool-level accuracy and final-answer accuracy by more than 10 percentage points, despite using the same extraction model.

With GPT-5.5 and the Qwen3-8B-SFT extractor fixed, retrieval plus repair raises final accuracy from 73.27% to 86.73% (+13.46 percentage points), supporting their joint contribution beyond the tool-writing model alone. The comparison does not separate retrieval from repair.

TABLE III  
GENERATED TOOL QUALITY AND END-TO-END CALCULATION PERFORMANCE.
<table><tr><td>Tool setting</td><td>Model</td><td>Runtime errors</td><td>Tool acc. (%)</td><td>Final acc. (%)</td></tr><tr><td>Direct generation</td><td>GPT-5.5</td><td>0</td><td>74.45</td><td>73.27</td></tr><tr><td>Retrieval + repair</td><td>Qwen3.5-Plus</td><td>21</td><td>84.45</td><td>76.18</td></tr><tr><td>Retrieval + repair</td><td>GPT-5.3-Codex</td><td>4</td><td>89.00</td><td>82.82</td></tr><tr><td>Retrieval + repair</td><td>GPT-5.5</td><td>0</td><td>94.91</td><td>86.73</td></tr></table>

Note: The last column pairs each tool set with the same Qwen3-8B-SFT extractor.

## D. Parameter-Extraction Model Comparison

Figure 3 compares parameter-extraction models while holding the GPT-5.5-generated tool set fixed. The evaluated extractors include models from the Qwen3, Qwen2.5, Llama, and GLM families [30], [32]–[34]. Once formula execution and arithmetic are delegated to tools, parameter extraction becomes the primary language-understanding challenge. The extractor must determine which clinical facts correspond to calculator fields and serialize them according to the required schema.

![](images/12e60f5bfb770a2c11332032f0889ad9813bddd4e52e4fa99f6206d938e4cf94.jpg)  
Fig. 3. Parameter matching rate and final-answer accuracy across base extractor families with a fixed GPT-5.5-generated tool set.

Across the evaluated models, higher PMR is generally associated with higher final-answer accuracy. The Qwen3 model achieved the best performance. However, while GLM-4 demonstrated a high Parameter Matching Rate (PMR), its final answer accuracy did not improve. This discrepancy arises because the two metrics measure performance at different levels. PMR is calculated over individual parameters, whereas final-answer accuracy is calculated at the case level. A case may have a high proportion of correctly extracted parameters but still receive an incorrect final answer when one clinically decisive field is missed or misclassified.

This discrepancy is especially important for calculators with Boolean risk factors, categorical inputs, date boundaries, or unit-sensitive measurements. For example, a missed comorbidity, an incorrectly inferred medication-related condition, or an incorrect date interpretation can alter the final score even when most other fields are correct. The case analysis presented later illustrates this issue using the HEART Score, where failure to infer hypercholesterolemia from atorvastatin use changes the final score despite otherwise accurate extraction.

## E. Category-Level Analysis

Figure 4 compares PMR and final-answer accuracy across calculator categories for the best configuration.

![](images/c7b6d11d127d70182acd81278b24e92f7f2066ec294a79bc6330af102909308d.jpg)  
Fig. 4. Parameter matching rate and final-answer accuracy by calculator category for the best configuration.

Low PMR primarily reflects errors in clinical interpretation and parameter extraction, whereas a large gap between PMR and final-answer accuracy motivates case-level inspection of possible errors after extraction, including boundary handling, unit conversion, rounding, or benchmark-specific output conventions. Physical-measurement and laboratory-test calculators achieve the strongest performance because their inputs are often explicit measurements and their rules are relatively direct. Once the correct values are extracted, deterministic execution resolves most of the remaining computation. Date and dosage calculators exhibit a different failure profile. Their input variables are often recoverable, but final correctness remains sensitive to date boundaries, unit conversions, rounding requirements, and output formatting. The date category shows the clearest discrepancy between PMR and final-answer accuracy, requiring case-level inspection to distinguish calculator or benchmark output-contract errors from failures to interpret the patient note. Risk, diagnosis, and severity calculators present the complementary challenge. These calculators often require clinical-semantic judgments involving implied comorbidities, symptom timing, disease categories, negation, and historical context. In such cases, a single Boolean or categorical error can change the final score.

PMR is parameter-level, whereas final-answer accuracy is case-level; their difference alone cannot localize failures to the generated tool or output contract.

## F. Ablation Analysis

All ablation experiments use the same corrected test set and the official benchmark evaluator. The full workflow combines GPT-5.5-generated tools with Qwen3-8B-SFT parameter extraction. The No extractor SFT condition uses the same GPT-5.5-generated tools but replaces the fine-tuned extractor with the original Qwen3-8B model. The No schema/fewshot condition removes the calculator schema and calculator demonstrations from the extraction prompt. The No tool condition removes the tool-augmented execution and relies on the fine-tuned Qwen3-8B model to generate the final answer directly.

TABLE IV  
COMPONENT ABLATION RESULTS.
<table><tr><td>Variant</td><td>Final acc. (%)</td><td> $\Delta$ </td></tr><tr><td>Full workflow</td><td>86.73</td><td></td></tr><tr><td>No extractor SFT</td><td>83.73</td><td>-3.00</td></tr><tr><td>No schema/few-shot</td><td>79.18</td><td>-7.55</td></tr><tr><td>No tool</td><td>69.46</td><td>-17.27</td></tr></table>

The results in Table IV show that both deterministic tool execution and constrained parameter extraction contribute substantially to final performance. First, the No tool condition results in the largest performance degradation (-17.27 percentage points), confirming that even a highly specialized fine-tuned model cannot reliably replicate the deterministic accuracy of an executable calculator. Second, the No schema/few-shot variant shows a substantial drop of 7.55 percentage points, highlighting that calculator-specific schemas (e.g., field definitions, unit requirements, and Boolean defaults) are fundamental to grounding the extraction process. Finally, the No extractor SFT condition results in a moderate decrease. This suggests that while inference-time schema constraints provide the necessary structure for the task, supervised fine-tuning helps the model handle complex clinical conventions, such as handling missing data, interpreting implied conditions, and ensuring consistent categorical labeling.

## G. Construction Usage and Inference Time

Table V summarizes the model calls and token consumption required to construct the 55 calculator tools using retrieval and iterative repair. Table VI compares inference efficiency across four tool configurations on the 1,100-case test set using two NVIDIA RTX 5090 GPUs, with Qwen3-8B-SFT held fixed as the parameter extractor. We report the total inference time and the average time per case, computed by dividing the total runtime by the number of test cases. These measurements exclude tool construction and characterize average processing efficiency over the full test set rather than individual-request latency.

TABLE V  
CONSTRUCTION CALLS AND TOKEN USAGE.
<table><tr><td>Builder</td><td>Calls</td><td>Input</td><td>Output</td><td>Total</td></tr><tr><td>Qwen3.5-Plus</td><td>144</td><td>243,571</td><td>147,350</td><td>390,921</td></tr><tr><td>GPT-5.3-Codex</td><td>159</td><td>272,884</td><td>124,773</td><td>397,657</td></tr><tr><td>GPT-5.5</td><td>147</td><td>227,454</td><td>371,758</td><td>599,212</td></tr></table>

TABLE VI  
INFERENCE TIME WITH THE SAME SFT EXTRACTOR.
<table><tr><td>Tool library</td><td>Total (min)</td><td>Seconds/case</td></tr><tr><td>Qwen3.5-Plus: retrieval + repair</td><td>48.15</td><td>2.626</td></tr><tr><td>GPT-5.3-Codex: retrieval + repair</td><td>48.47</td><td>2.644</td></tr><tr><td>GPT-5.5: retrieval + repair</td><td>47.17</td><td>2.573</td></tr><tr><td>GPT-5.5: direct generation</td><td>47.28</td><td>2.579</td></tr></table>

## V. CASE STUDY AND ERROR ANALYSIS

To characterize the remaining failure modes, we analyze the 146 incorrect final answers produced by the best configuration on the test set. We categorize the errors into two types. The Extraction error means that at least one required calculator field extracted from the note mismatches the gold entity. The Generated-tool error means that all required fields are correctly extracted, but the final result remains incorrect because of an error in the generated tool, the rounding rule, the unit convention, the date boundary, or the answer string, which still violates the benchmark contract.

Table VII presents one successful case and the two representative error cases. In Case A, the Cockcroft-Gault calculation succeeds because all required variables are explicitly extracted and the tool computes the correct creatinine-clearance value. Case B illustrates an extraction error. The HEART Score tool correctly applies the scoring rule to the parameters it receives. However, the extraction model fails to infer hypercholesterolemia from the patient’s atorvastatin use. Case C illustrates a tool error. The extractor correctly identifies the input steroid, dose, route, and target steroid: Hydrocortisone PO, 162.935 mg, and Dexamethasone PO. The tool should convert the dose using the equivalence between oral hydrocortisone and oral dexamethasone, yielding the benchmark answer of 6.029 mg. However, the GPT-5.5-generated normalization logic searches for the substring cortisone before hydrocortisone, causing Hydrocortisone PO to be incorrectly normalized as cortisone. The resulting computation, $1 6 2 . 9 3 5 \times 0 . 7 5 / 2 5 = 4 . 8 8 8 0 5 .$ is therefore based on the wrong source-steroid equivalence and produces an incorrect final answer of 4.89 mg. This error originates in the generated tool rather than in parameter extraction or answer formatting.

Figure 5(a) summarizes the overall error distribution: 80 errors result from extraction, whereas 66 result from the generated tool. Extraction errors are concentrated in calculators that require clinical-semantic interpretation, including diagnosis identification, temporal reasoning, implied comorbidities, and risk-factor assessment. Generated-tool errors occur more often when the necessary parameters are relatively easy to identify but correct evaluation depends on a specific formula branch, normalization rule, rounding procedure, unit convention, date boundary, or answer representation.

TABLE VII REPRESENTATIVE CASES OF OUR METHOD FOR MEDICAL CALCULATION.  
Case A Case B Case C   
Calculator Creatinine Clearance, Cockcroft-Gault (lab test) HEART Score (risk) Steroid Conversion Calculator (dosage)   
Patient note (key facts) 87-year-old man; height 163 cm; weight 48 kg; 47-year-old man in the ED; diabetes, hyperten- Patient receives Hydrocortisone PO 162.935 mg;   
serum creatinine 1.4 mg/dL; the question asks sion, atorvastatin use, normal ECG, troponin > the question asks for the equivalent Dexametha-  
Cockcroft-Gault clearance with weight selection. 3× upper limit, no prior atherosclerotic disease sone PO dose, with the oral route preserved   
slightly suspicious history.   
Question Creatinine clearance, mL/min, using adjusted HEART Score (0–10) at first admission, before Equivalent Dexamethasone PO dose in mg.   
body weight if it exceeds ideal. treatment.   
Generated tool pseu- ibw = ideal\_weight(sex, risk\_count = diabetes if "cortisone" in name: return   
docode (written once height); w = actual if actual + hypertension + cortisone; elif "hydrocortisone"   
by GPT-5.5) < ibw else adjusted; crcl = hypercholesterolemia + ...; in name: return hydrocortisone;   
((140-age)<sub>\*</sub>w)/(72<sub>\*</sub>scr). risk\_points = 2 if risk\_count then convert by steroid-equivalence table.   
>= 3 else 1 if risk\_count >= 1   
else 0.   
Extracted parameters sex=Male, age=87 y, weight=48 kg, height=163 age=47, hypertension=True, diabetes=True, input steroid=Hydrocortisone PO, dose=162.935   
(Qwen3-8B-SFT) cm, creatinine=1.4 mg/dL. hypercholesterolemia=False, ECG=Normal, mg; target steroid=Dexamethasone PO.   
troponin=> 3×, atherosclerotic=False,   
suspicion=Slightly suspicious.   
Tool computation weight 48 < ideal 59.6, so uses actual weight; only 2 risk factors counted, so risk block scores Hydrocortisone is misread as cortisone:   
(140 − 87) × 48/(72 × 1.4) = 25.24. 1 and total = 4. 162.935 × 0.75/25 = 4.88805 mg.   
Model answer 25.24 mL/min 4 4.89 mg   
Gold answer 25.2381 5 6.029 mg   
Cause The validated tool executes the correct formula Extraction error: the extractor missed that ator- Generated-tool error: the normalizer matches   
from correctly extracted inputs. vastatin implies hypercholesterolemia. cortisone before hydrocortisone.

![](images/7f026f8af37a4db5c4895b14439d884a59b079f81877b0550d88d8e228272743.jpg)  
Fig. 5. Error analysis of the best run. Extraction errors come from incorrectly extracted calculator fields, while Generated-tool errors occur after the required fields have been extracted correctly.

Figure 5(b) presents this distinction for the seven calculators with the largest numbers of errors. Each listed calculator contributes 20 test cases. Estimated Gestational Age is primarily affected by date-handling and output-contract errors, whereas Steroid Conversion is mainly affected by tool-rule errors, including normalization failures such as Case C. Wells PE, FeverPAIN, the Charlson Comorbidity Index, and the HEART Score are primarily affected by input-extraction errors because they require clinical judgments involving symptoms, timing, risk factors, and implied medical history.

## VI. CONCLUSION

This work presents a tool-augmented framework for reliable medical calculation from free-text patient notes. The framework separates calculator-rule retrieval, executable tool generation, schema-constrained parameter extraction, and deterministic tool execution. By grounding calculator tools in retrieved evidence and validating them through execution feedback, the approach reduces reliance on LLM formula recall and minimizes arithmetic errors. The separation between clinical interpretation and deterministic calculation also improves transparency. On the corrected MedCalc-Bench Verified test set, our method achieves the competitive performance, substantially outperforming prompting and SFT-based baselines. The analyses further show that reliable performance depends on both high-quality generated calculator tools and accurate schema-constrained extraction.

Despite these advantages, the study faces limitations that define its current scope. Extraction errors persist when the system must infer implicit information like medication-related conditions, negation, or temporality. Generated-tool errors arise when inputs are processed using incorrect normalization rules, formula branches, or output formats. Future work will address these challenges through reinforcement learning that rewards both accurate extraction of critical fields and correct final calculations. Furthermore, intelligent agents could improve reliability by verifying uncertain parameters, testing generated tools on boundary cases, and requesting clarification when key information is missing.

Evaluation concerns new patient notes for the same 55 calculators used during tool construction. Generalization to previously unseen calculators and clinical deployment remain untested.

## ACKNOWLEDGMENT

This work was supported by the National Natural Science Foundation of China (No. 62302076, 62276043, 62676065) and the Fundamental Research Funds for the Central Universities (No. DUT25YG108).

[1] MDCalc, “MDCalc: Medical calculators, equations, scores, and guidelines,” accessed: Oct. 15, 2026.

[2] World Health Organization, Global Strategy on Human Resources for Health: Workforce 2030. Geneva, Switzerland: World Health Organization, 2016.

[3] N. Khandekar, Q. Jin, G. Xiong, S. Dunn, S. S. Applebaum, Z. Anwar, M. Sarfo-Gyamfi, C. W. Safranek, A. A. Anwar, A. Zhang, A. Gilson, M. B. Singer, A. Dave, A. Taylor, A. Zhang, Q. Chen, and Z. Lu, “MedCalc-Bench: Evaluating large language models for medical calculations,” in Advances in Neural Information Processing Systems, vol. 37, 2024, pp. 84 730–84 745.

[4] Q. Jin, Z. Wang, Y. Yang, Q. Zhu, D. Wright, T. Huang, W. J. Wilbur, Z. He, A. Taylor, Q. Chen, and Z. Lu, “AgentMD: Empowering language agents for risk prediction with large-scale clinical tool learning,” arXiv preprint arXiv:2402.13225, 2024.

[5] A. J. Goodell, S. N. Chu, D. Rouholiman, and L. F. Chu, “Large language model agents can use tools to perform clinical calculations,” npj Digital Medicine, vol. 8, no. 1, p. 163, 2025.

[6] L. Luo, J. Ning, Y. Zhao, Z. Wang, Z. Ding, P. Chen, W. Fu, Q. Han, G. Xu, Y. Qiu, D. Pan, J. Li, H. Li, W. Feng, S. Tu, Y. Liu, Z. Yang, J. Wang, Y. Sun, and H. Lin, “Taiyi: A bilingual fine-tuned large language model for diverse biomedical tasks,” Journal of the American Medical Informatics Association, vol. 31, no. 9, pp. 1865–1874, 2024.

[7] F. He, H. Lai, J. Liu, B. Wang, H. Chen, H. Liu, and C. Zhang, “Solving mathematical problems using large language models: A survey,” Data Intelligence, vol. 7, no. 4, pp. 907–946, 2025.

[8] D. Jin, E. Pan, N. Oufattole, W.-H. Weng, H. Fang, and P. Szolovits, “What disease does this patient have? a large-scale open domain question answering dataset from medical exams,” Applied Sciences, vol. 11, no. 14, p. 6421, 2021.

[9] Q. Jin, B. Dhingra, Z. Liu, W. Cohen, and X. Lu, “PubMedQA: A dataset for biomedical research question answering,” in Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP). Association for Computational Linguistics, 2019, pp. 2567–2577.

[10] A. Pal, L. K. Umapathi, and M. Sankarasubbu, “MedMCQA: A largescale multi-subject multi-choice dataset for medical domain question answering,” in Proceedings of the Conference on Health, Inference, and Learning, ser. Proceedings of Machine Learning Research, vol. 174. PMLR, 2022, pp. 248–260.

[11] K. Singhal, S. Azizi, T. Tu, S. S. Mahdavi, J. Wei, H. W. Chung, N. Scales, A. Tanwani, H. Cole-Lewis, S. Pfohl et al., “Large language models encode clinical knowledge,” Nature, vol. 620, no. 7972, pp. 172– 180, 2023.

[12] R. K. Arora, J. Wei, R. S. Hicks, P. Bowman, J. Quinonero-Candela,˜ F. Tsimpourlas, M. Sharman, M. Shah, A. Vallone, A. Beutel, J. Heidecke, and K. Singhal, “HealthBench: Evaluating large language models towards improved human health,” arXiv preprint arXiv:2505.08775, 2025.

[13] J. Ye, D. Tawfik, A. J. Goodell, N. V. Kotha, M. K. Buyyounouski, and M. Bayati, “Scalable stewardship of an LLM-assisted clinical benchmark with physician oversight,” arXiv preprint arXiv:2512.19691v3, 2026.

[14] A. Krohn-Grimberghe, “MedCalc-Bench doesn’t measure what you think: A benchmark audit and the case for open-book evaluation,” arXiv preprint arXiv:2603.02222, 2026.

[15] J. Wei, X. Wang, D. Schuurmans, M. Bosma, B. Ichter, F. Xia, E. Chi, Q. V. Le, and D. Zhou, “Chain-of-thought prompting elicits reasoning in large language models,” in Advances in Neural Information Processing Systems, vol. 35, 2022, pp. 24 824–24 837.

[16] X. Wang, J. Wei, D. Schuurmans, Q. Le, E. Chi, S. Narang, A. Chowdhery, and D. Zhou, “Self-consistency improves chain of thought reasoning in language models,” arXiv preprint arXiv:2203.11171, 2022.

[17] P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. Kuttler, M. Lewis, W.-t. Yih, T. Rockt¨ aschel, S. Riedel, and D. Kiela,¨ “Retrieval-augmented generation for knowledge-intensive NLP tasks,” in Advances in Neural Information Processing Systems, vol. 33, 2020, pp. 9459–9474.

[18] L. Gao, A. Madaan, S. Zhou, U. Alon, P. Liu, Y. Yang, J. Callan, and G. Neubig, “PAL: Program-aided language models,” in Proceedings of the 40th International Conference on Machine Learning, ser. Proceed-

ings of Machine Learning Research, vol. 202. PMLR, 2023, pp. 10 764– 10 799.

[19] W. Chen, X. Ma, X. Wang, and W. W. Cohen, “Program of thoughts prompting: Disentangling computation from reasoning for numerical reasoning tasks,” arXiv preprint arXiv:2211.12588, 2022.

[20] S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. Narasimhan, and Y. Cao, “ReAct: Synergizing reasoning and acting in language models,” arXiv preprint arXiv:2210.03629, 2022.

[21] T. Schick, J. Dwivedi-Yu, R. Dess\`ı, R. Raileanu, M. Lomeli, E. Hambro, L. Zettlemoyer, N. Cancedda, and T. Scialom, “Toolformer: Language models can teach themselves to use tools,” in Advances in Neural Information Processing Systems, vol. 36, 2023, pp. 68 539–68 551.

[22] Y. Qin, S. Hu, Y. Lin, W. Chen, N. Ding, G. Cui, Z. Zeng, X. Zhou, Y. Huang, C. Xiao et al., “Tool learning with foundation models,” ACM Computing Surveys, vol. 57, no. 4, pp. 1–40, 2024.

[23] B. Wang, I. Xia, Y. Zhang, J. Wang, F. Ouyang, S. Han, A. Cohan, H. Yu, and Z. Yao, “From scores to steps: Diagnosing and improving LLM performance in evidence-based medical calculations,” in Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics, 2025, pp. 10 809–10 833.

[24] H. Wang, L. Yan, X. Yao, F. Meng, Y. He, J. Jiang, and Y. Guan, “MedCalc-R1: Knowledge-guided reward framework for medical mathematical reasoning,” arXiv preprint arXiv:2608.08623, 2026.

[25] K. Mao, J. Ding, J. Chen, M. Bian, R. Chen, X. Peng, S. Ren, L. Li, and J. Xu, “MedCalc-Eval and MedCalc-Env: Advancing medical calculation capabilities of large language models,” arXiv preprint arXiv:2510.27267, 2025.

[26] F. Liu, J. Wu, H. Zhou, X. Gu, S. Molaei, A. Thakur, L. Clifton, H. Wu, and D. A. Clifton, “RiskAgent: Autonomous medical AI copilot for generalist risk prediction,” medRxiv, 2025, preprint.

[27] Y. Zhu, S. Wei, X. Wang, K. Xue, S. Zhang, and X. Zhang, “MeNTi: Bridging medical calculator and LLM agent with nested tool calling,” in Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers). Association for Computational Linguistics, 2025, pp. 5097–5116.

[28] Y. Zhu, Y. Huang, S. Qin, Z. Huang, S. Zhang, and X. Zhang, “MedMCP-Calc: Benchmarking LLMs for realistic medical calculator scenarios via MCP integration,” arXiv preprint arXiv:2601.23049, 2026.

[29] nikhilk7153, “MedCalc-Bench-Verified,” GitHub repository, accessed: Oct. 15, 2026.

[30] A. Yang, A. Li, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Gao, C. Huang, C. Lv et al., “Qwen3 technical report,” arXiv preprint arXiv:2505.09388, 2025.

[31] Anthropic, “System card: Claude Sonnet 4.6,” Feb. 2026, published February 17, 2026.

[32] Qwen, A. Yang, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Li, D. Liu, F. Huang, H. Wei et al., “Qwen2.5 technical report,” arXiv preprint arXiv:2412.15115v2, 2025.

[33] A. Grattafiori, A. Dubey, A. Jauhri, A. Pandey, A. Kadian, A. Al-Dahle, A. Letman, A. Mathur, A. Schelten, A. Vaughan et al., “The Llama 3 herd of models,” arXiv preprint arXiv:2407.21783, 2024.

[34] Team GLM, A. Zeng, B. Xu, B. Wang, C. Zhang, D. Yin, D. Zhang, D. Rojas, G. Feng, H. Zhao et al., “ChatGLM: A family of large language models from GLM-130B to GLM-4 all tools,” arXiv preprint arXiv:2406.12793, 2024.