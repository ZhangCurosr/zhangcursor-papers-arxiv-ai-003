# Self-Guided LLM Reasoning for Error Diagnosis in Vehicle Routing Problems

Ni Zhang<sup>1</sup>, Jianan Zhou<sup>2</sup>, Yue-Jiao Gong<sup>3</sup>, Zhiguang Cao<sup>1</sup>

<sup>1</sup>Singapore Management University <sup>2</sup>Nanyang Technological University <sup>3</sup>South China University of Technology

ni.zhang.2025@phdcs.smu.edu.sg, jianan004@e.ntu.edu.sg, gongyuejiao@gmail.com, zgcao@smu.edu.sg

## Abstract

Vehicle Routing Problems (VRPs) are fundamental to many real-world applications. Recent advances in large language models (LLMs) have spurred progress in solving these NPhard problems. However, most existing approaches assume that input instances are error-free, neglecting the presence of faults commonly encountered in real-world data. In this work, we systematically investigate the capability of LLMs to detect (and correct) errors in VRP instances, and identify key limitations, particularly in handling complex instances or errors that substantially increase the LLM’s reasoning burden. To overcome these challenges, we propose a self-guided reasoning framework in which LLMs autonomously generate code that provides structured insights to aid in error diagnosis. Experiments across diverse problem settings and LLMs demonstrate that our framework significantly improves error detection performance, achieving near-perfect accuracy.

## 1 Introduction

Vehicle Routing Problems (VRPs) are widely deployed in real-world decision-making pipelines across last-mile delivery (Tiwari and Sharma 2023; Tilk, Olkis, and Irnich 2021; Kitjacharoenchai and Lee 2019), transportation (Gayialis, Konstantakopoulos, and Tatsiopoulos 2019; Wang et al. 2024; Xu et al. 2023), and ride-sharing (Lin et al. 2012; Guo et al. 2022; Chen et al. 2019). While decades of research have produced highly eficient optimization algorithms, the reliability of these pipelines remains fragile: subtle inconsistencies or latent errors in problem instances can lead to solver failures or infeasible solutions. Most VRP solvers assume that input instances are well-formed and feasible. In practice, however, instances may contain errors such as formatting issues, missing fields, or violated feasibility constraints. Importantly, many of these faults are latent as they are not immediately evident from the instance specification and require additional analysis to be identified. Given the diversity of VRP formulations and input formats across problem settings, designing general-purpose programs for automatic error detection is highly challenging. As a result, prior approaches typically rely on domain expertise or post hoc analysis of solver outputs to infer potential instance-level issues, which is both time-consuming and expertise-dependent.

Recent advances in large language models (LLMs) ofer a promising new direction. With strong language understanding and reasoning capabilities, LLMs can interpret diverse inputs and reason about structured information. These capabilities naturally prompt following questions: a) Can LLMs be enabled to identify and correct errors in VRP instances? b) How and why does such reasoning succeed or fail? c) Under what conditions do LLMs struggle to identify errors, and how can these limitations be efectively mitigated?

Recent work has begun to explore this direction. For example, MOID (Li et al. 2025) attempts to identify potential instance-level issues by examining the feasibility of solutions produced by a solver. However, it heavily relies on solver outputs and thus inherits their limitations, since infeasible solutions may arise from solver deficiencies rather than errors in the input instance. Moreover, a comprehensive evaluation of LLMs’ capability to directly identify and correct errors within VRP instances remains absent to date.

In this paper, we systematically investigate the capability of LLMs to diagnose errors in VRP instances. To address the aforementioned research questions, we evaluate: (i) LLMs ability to handle increased error complexity, (ii) their ability to cope with increased instance complexity, and (iii) the efect of iterative reasoning on error detection performance. Our findings show that (i) LLMs perform well in detecting formatting errors and relatively simple constraint violations, such as capacity and demand constraints. However, their ability to identify more complex constraint violations, including time windows and distance limits, as well as scenarios with larger numbers of errors, is substantially weaker. (ii) Detection performance degrades as instance scale and constraint counts increases, indicating limited robustness under increased instance complexity. (iii) While iterative reasoning improves the detection of formatting issues and simple constraint violations, it provides limited benefit for identifying more complex errors.

In summary, we find that LLMs are efective at identifying and correcting errors in relatively simple settings, whereas their performance degrades when confronted with instances or errors of higher complexity. To address limitations, we propose a self-guided reasoning framework in which the LLM generates analysis code as an intermediate reasoning tool. This code enables the model to extract structured insights that facilitate error detection while alleviating overall reasoning burden. Our contributions are summarized as follows:

• We present a comprehensive study of LLMs’ capability to detect errors in VRP instances, including a systematic analysis of how error complexity, instance complexity and reasoning depth afect detection performance.

• We propose a self-guided reasoning framework in which LLMs generate code to derive intermediate reasoning insights, addressing their limitations in handling complex errors while reducing the overall reasoning burden.

• Extensive experiments across diverse problem settings, including vehicle routing and bin packing, demonstrate that our framework significantly enhances error detection performance, achieving near-perfect accuracy.

## 2 Related work

Existing uses of LLMs in VRPs broadly fall into two categories. The first line of work focuses on using LLMs to directly generate solutions. For example, OPRO (Yang et al. 2024) employs LLMs to produce solutions directly, LEMA (Liu et al. 2024b) leverages LLMs to perform genetic search, and Jiang et al. (Jiang et al. 2025) fine-tune LLMs for solution generation. The second line of work instead leverages LLMs to generate heuristic code or optimization operators for solving VRPs. FunSearch (Romera-Paredes et al. 2024) adopts an island-based evolutionary framework to guide LLMs in generating heuristic programs, while EOH (Liu et al. 2024a) integrates LLMs with evolutionary algorithms to automatically discover improved heuristics. ReEvo (Ye et al. 2024) further augments the evolutionary loop with reflective self-feedback, enabling the LLM to critique and refine its own proposals, and HSEvo (Dat, Doan, and Binh 2025) introduces a hierarchical evolutionary structure with diversity-aware selection over modular heuristic components. AFL (Zhang et al. 2025) extends this paradigm with an agentic framework for solving complex VRPs through multi-stage reasoning and coding. Overall, these works demonstrate the growing potential of LLMs.

However, all of these methods assume that input instances are feasible and error-free, and therefore lack the capability to handle instance-level errors that frequently arise in practice. MOID (Li et al. 2025) takes a step toward error detection by analyzing solver execution, but its reliance on potentially unreliable solvers makes it dificult to distinguish between instance-level errors and solver-induced failures. As such, it serves as an initial attempt rather than a comprehensive solution. In this work, we bridge this gap by enabling LLMs to directly identify and reason about instance-level errors.

## 3 A Comprehensive Study

In this section, we conduct a comprehensive study on LLMs capability to detect error in VRP instances, examining performance across varying levels of error complexity, instance complexity, and reasoning depth. To enable a systematic evaluation, we construct error-injected VRP datasets by introducing 200 errors into 20 instances for each VRP variant. These errors include both formatting errors (e.g., value errors or missing entries) and feasibility-related errors (e.g., constraint violation). Detailed descriptions are provided in Appendix B.

![](images/7c18f78ba6d09f4aba0920427f03d099bad0d509e4d858b347fe01de9cb66a22.jpg)

<table><tr><td>Variant</td><td>Format</td><td>Capacity</td><td>Backhaul</td><td>Distance Limit</td><td>Time Window</td></tr><tr><td>CVRP</td><td>91.38</td><td>84.28</td><td></td><td></td><td></td></tr><tr><td>OCVRP</td><td>90.32</td><td>92.38</td><td></td><td></td><td></td></tr><tr><td>MDCVRP</td><td>94.44</td><td>87.67</td><td></td><td>一</td><td></td></tr><tr><td>CVRPB</td><td>98.90</td><td></td><td>61.19</td><td>1</td><td></td></tr><tr><td>CVRPMB</td><td>91.18</td><td></td><td>64.29</td><td></td><td></td></tr><tr><td>CVRPTW</td><td>95.38</td><td>90.00</td><td></td><td></td><td>0.00</td></tr><tr><td>CVRPL</td><td>97.53</td><td>82.97</td><td></td><td>0.00</td><td></td></tr></table>

Figure 1: Overall error detection performance across diferent VRP variants, with a breakdown by error type shown below (higher is better).

## 3.1 Error Complexity

We analyze LLMs’ ability to handle varying levels of error complexity, including constraint violations across diferent problem variants and varying numbers of errors.

Problem Variant. We analyze error detection performance across 7 simple VRP variants, each containing one or two types of constraints. As shown in Fig. 1, LLMs achieve consistently high detection rates for formatting errors and faults associated with simple constraints (e.g., capacity (C)), indicated by the red bars. These results suggest that LLMs are efective at identifying errors that can be detected through basic inspection or straightforward logical checks.

However, detection rates degrade substantially as constraint complexity increases. For variants involving backhaul (B) constraints, accuracy declines noticeably. For more complex constraints, such as time windows (TW) and distance limits (L), detection rates drop sharply and approach zero. This highlights a key limitation: while LLMs perform well on formatting errors and simple constraint violations, they struggle with errors that require complex calculations or numerical reasoning.

Error Number. We further evaluate the impact of error number on LLM-based error detection by varying the number of injected errors and measuring detection performance across diferent VRP variants. To provide a more comprehensive assessment of its impact, we select variants with increasing constraint complexity, ranging from CVRP, which involves a single type of constraint, to MDOCVRPMBLTW, which incorporates six types of constraints.

Fig. 2 shows that detection performance consistently declines as the number of injected errors increases across all variants. This trend holds regardless of the underlying problem type, indicating that LLMs are highly sensitive to the accumulation of errors within an instance. These results suggest that a larger number of errors imposes a substantially greater reasoning burden on LLMs, as they must simultaneously identify more errors.

![](images/8b2977fc496e20aef0365f2a6b499c9cdbe3e4d5506d33acb0f31728b291b3ba.jpg)

Effect of Error Number on 100-Node Instances  
![](images/a6023f815d6e51fe882e478be986613dc2579df6fcca06f70ba6daffb920850d.jpg)  
Figure 2: Efect of fault number and instance scale.

## 3.2 Instance Complexity

Beyond error complexity, intrinsic instance complexity may also impact LLMs’ ability to detect errors. To study this, we conduct experiments across varying instance scales and constraint counts.

Instance Scale. We examine the efect of instance scale on error detection performance by comparing instances of diferent sizes. As shown in Fig. 2, detection rates on 100- node instances are uniformly lower than those on 50-node instances across all VRP variants and error settings. This performance degradation indicates that larger instance significantly increase the complexity of instance-level reasoning. As the number of nodes grows, LLMs reason over a larger set of constraints and dependency relationships, which exacerbates the reasoning burden and makes accurate error identification increasingly dificult.

Count of Constraint Type. We evaluate LLM-based error detection across 48 VRP variants with constraint counts ranging from one to six, and report the average detection rate for each constraint level. As shown in Table 1, error detection performance exhibits a clear decreasing trend as the number of constraints types increases. While LLMs achieve relatively high detection accuracy when only one or two constraint types are present, performance degrades substantially as more constraint types are introduced, with a sharp drop observed for instances involving five or six constraint types. These results indicate that an increasing number of constraint types significantly challenges LLMs’ ability to identify errors, highlighting a key limitation of direct LLM-based error detection in highly constrained VRPs.

![](images/ac493c507b198f1c237325528beb5730909d96e28281ebf1dc81f69d1bbc3c55.jpg)  
Figure 3: Efect of reasoning depth on error detection performance across diferent VRP variants.

Table 1: Average error detection rates of LLMs under diferent numbers of constraint types. (higher is better).
<table><tr><td>Constraint Type</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td></tr><tr><td>LLM (%)</td><td>87.27</td><td>79.42</td><td>72.42</td><td>57.42</td><td>53.97</td><td>22.30</td></tr></table>

## 3.3 Reasoning Depth

Iterative reasoning is an efective approach to improve error detection. To evaluate its impact, we analyze the role of reasoning depth through an iterative process, in which the LLM identifies errors, the instance is repaired accordingly, and the LLM then reasons again on the updated version. As shown in Fig. 3, increasing reasoning depth consistently improves detection rates across VRP variants. For variants with relatively simple constraints (e.g., CVRP and CVRPB), repeated reasoning yields performance gains, with detection rates approaching 100% after several iterations.

However, for problems with more complex constraints, the benefits of deeper reasoning diminish. In particular, for instances with constraints such as time windows and combinations of multiple constraints, detection performance remains far from perfect. These results suggest that while iterative reasoning can facilitate error detection by LLMs, it remains insuficient for handling complex errors, such as violations involving intricate constraints or combinations of constraints that require sophisticated feasibility reasoning.

## 4 Self-Guided Reasoning Framework

Our analysis of LLMs’ ability to detect instance-level errors reveals a key limitation: their dificulty in handling instances or errors of higher complexity which may require advanced reasoning. To address this, we propose a self-guided reasoning framework in which the LLM first constructs a reasoning chain for error detection and then uses it to guide the generation of analysis code. The generated code yields structured insights that the LLM can reference, enhancing its ability to identify errors while alleviating the overall reasoning burden.

![](images/466f41387e5e31087ddd8084a261f2a97d54d615de050a5dbe724346fa4135cc.jpg)  
Figure 4: Overall framework of Self-Guided Reasoning.

As illustrated in Fig. 4, the framework begins with instance analysis, where the LLM analyzes the input instance by identifying the problem type, required section names, and associated constraints. Based on this analysis, the LLM derives a reasoning path that outlines how to analyze potential errors in the instance and generates corresponding function-level prompts for each step. Guided by these prompts, executable code is generated and run to extract relevant insights. Finally, using both the original instance and the extracted insights as references, the LLM produces a final report.

## 4.1 Instance Analysis

Given an input instance, the instance analysis stage aims to establish a concise and structured understanding of the problem specification. The LLM first parses the raw instance to identify its essential characteristics, which are summarized as follows:

• Problem type, indicating the underlying optimization problem (e.g., CVRP) implied by the instance.

• Required section names, specifying the mandatory sections that a valid instance of the identified problem type should contain.

• Associated constraints, describing the feasibility conditions in the identified problem, such as capacity, time window and so on.

Based on this structured information, the LLM performs a self-validation step to identify potential inconsistencies among the problem type, required sections, associated constraints, and the instance content, and revises its analysis accordingly. Finally, the LLM produces an instance analysis summary, providing an intuitive and reliable reference that helps reduce the reasoning burden in subsequent stages. The detailed prompt used for instance analysis is provided in Appendix D.1.

## 4.2 Insight Generation

The framework then proceeds to the insight generation stage, where the LLM derives structured insights for error analysis. A detailed example of the prompt used at this stage is provided in Appendix D.2. Based on the previously generated instance analysis summary, the LLM infers which parts of the instance are likely to contain errors. More specifically, it reasons about the potential locations of formatting errors, such as missing sections or invalid values, as well as feasibility errors, such as capacity violations, and then specifies a corresponding reasoning path.

Following this reasoning path, the LLM generates a set of function-level prompts, each corresponding to a specific analysis step. These prompts clearly describe the intended functionality, required inputs, and expected outputs, and are used to guide code generation. During this process, the LLM is instructed to ensure that the generated code reports detailed diagnostic information at the point of error detection, allowing for precise localization and subsequent revision.

The generated code is then executed to perform structured instance analysis and extract concrete evidence of potential errors. If execution failures or runtime bugs occur, the resulting bug messages are fed back to the LLM, which revises the code accordingly. Once execution succeeds, the analysis outputs are collected and summarized as error insights, which form the basis for subsequent reporting and decision-making.

Overall, in this stage, the LLM guides code generation to convert its reasoning into executable procedures. This design ofers two key advantages: (i) complex constraint calculations are ofloaded to code execution, addressing the limitations of LLMs in precise numerical reasoning; and (ii) programmatic analysis enables both eficient and comprehensive evaluation of the instance, substantially reducing the reasoning burden on the LLM.

## 4.3 Report Generation

In the final stage, the framework performs report generation by synthesizing the derived insights with the original instance information. Guided by the extracted error insights, the LLM systematically aligns the identified errors with the corresponding components of the input instance. This enables the generation of a comprehensive and structured report that clearly outlines the types, locations, and potential causes of the detected errors. Based on this report, the LLM is further instructed to revise the instance accordingly, correcting the identified errors to obtain a valid and consistent instance specification. The prompt used for this stage is provided in Appendix D.3.

Table 2: Error Detection Rate (EDR) comparison across 48 VRP variants (higher is better).
<table><tr><td></td><td>EDR(%)</td><td>Time(m)</td><td>EDR(%)</td><td>Time(m)</td><td>EDR(%)</td><td>Time(m)</td><td>EDR(%)</td><td>Time(m)</td><td>EDR(%)</td><td>Time(m)</td><td>EDR(%)</td><td>Time(m)</td></tr><tr><td></td><td colspan="2">CVRP</td><td colspan="2">OCVRP</td><td colspan="2">CVRPTW</td><td colspan="2">CVRPL</td><td colspan="2">MDCVRP</td><td colspan="2">CVRPB</td></tr><tr><td>LLM</td><td>87.25</td><td>6.35</td><td>91.88</td><td>6.20</td><td>66.50</td><td>7.00</td><td>59.50</td><td>7.38</td><td>94.42</td><td>8.15</td><td>78.50</td><td>8.25</td></tr><tr><td>Self-Guided</td><td>99.49</td><td>6.39</td><td>97.98</td><td>7.20</td><td>95.00</td><td>7.22</td><td>100.00</td><td>8.56</td><td>97.47</td><td>9.54</td><td>94.00</td><td>9.03</td></tr><tr><td></td><td>CVRPMB</td><td></td><td>OCVRPTW</td><td></td><td></td><td>MDOCVRP</td><td colspan="2">OCVRPB</td><td colspan="2">CVRPLTW</td><td colspan="2">MDCVRPL</td></tr><tr><td>LLM</td><td>77.95</td><td>9.45</td><td>79.50</td><td>10.96</td><td>94.50</td><td>8.17</td><td>84.00</td><td>8.55</td><td>59.61</td><td>9.21</td><td colspan="2">59.11</td></tr><tr><td>Self-Guided</td><td>96.92</td><td>9.97</td><td>98.00</td><td>10.87</td><td>98.50</td><td>8.32</td><td>94.00</td><td>8.52</td><td>99.51</td><td>8.93</td><td colspan="2">8.50 94.09 9.45</td></tr><tr><td></td><td>CVRPBL</td><td></td><td>CVRPBTW</td><td></td><td>MDCVRPTW</td><td></td><td>MDCVRPB</td><td></td><td>OCVRPMB</td><td></td><td colspan="2">CVRPMBL</td></tr><tr><td>LLM</td><td>56.50</td><td>7.04</td><td>64.00</td><td>13.96</td><td>76.00</td><td>15.48</td><td>71.50</td><td>10.57</td><td>86.00</td><td>9.27</td><td colspan="2">65.00 7.63</td></tr><tr><td>Self-Guided</td><td>100.00</td><td>8.10</td><td>95.00</td><td>12.36</td><td>95.50</td><td>17.29</td><td>95.50</td><td>11.35</td><td>95.00</td><td>10.55</td><td colspan="2">91.50 7.93</td></tr><tr><td></td><td>OCVRPL</td><td></td><td>CVRPMBTW</td><td></td><td>MDCVRPMB</td><td></td><td>MDCVRPMBL</td><td></td><td>CVRPMBLTW</td><td></td><td colspan="2">MDCVRPBTW</td></tr><tr><td>LLM</td><td>75.25</td><td>7.90</td><td>64.00</td><td>10.46</td><td>78.00</td><td>8.64</td><td>44.50</td><td>9.55</td><td>61.39</td><td>12.43</td><td colspan="2">51.00 12.50</td></tr><tr><td>Self-Guided</td><td>99.50</td><td>7.38</td><td>96.50</td><td>11.95</td><td>96.00</td><td>8.88</td><td>96.65</td><td>11.22</td><td>98.51</td><td>12.17</td><td colspan="2">90.50 13.08</td></tr><tr><td></td><td>OCVRPLTW</td><td></td><td>MDOCVRPL</td><td></td><td>OCVRPBL</td><td></td><td>OCVRPMBL</td><td></td><td>MDOCVRPTW</td><td></td><td colspan="2">OCVRPBTW</td></tr><tr><td>LLM</td><td>56.59</td><td>10.87</td><td>43.69</td><td>7.68</td><td>62.00</td><td>7.95</td><td>63.00</td><td>8.63</td><td>68.50</td><td>13.25</td><td colspan="2">71.50</td></tr><tr><td>Self-Guided</td><td>96.10</td><td>12.95</td><td>93.00</td><td>8.97</td><td>96.50</td><td>8.56</td><td>94.50</td><td>9.56</td><td>98.50</td><td>14.12</td><td colspan="2">94.00</td></tr><tr><td></td><td></td><td>OCVRPMBTW</td><td>MDOCVRPB</td><td></td><td>MDOCVRPMB</td><td></td><td>MDCVRPLTW</td><td></td><td>MDCVRPBL</td><td></td><td colspan="2">CVRPBLTW</td></tr><tr><td>LLM</td><td>60.00</td><td>11.83</td><td>86.50</td><td>10.28</td><td>67.50</td><td>9.66</td><td>23.08</td><td>11.32</td><td>44.00</td><td>8.37</td><td colspan="2">56.00</td></tr><tr><td>Self-Guided</td><td>99.50</td><td>13.35</td><td>92.00</td><td>9.73</td><td>97.00</td><td>8.80</td><td>98.56</td><td>12.38</td><td>90.00</td><td>10.62</td><td colspan="2">96.59</td></tr><tr><td></td><td></td><td>MDVRPMBTW</td><td>MDOVRPLTW</td><td></td><td>OCVRPBLTW</td><td></td><td>OCVRPMBLTW</td><td></td><td></td><td>MDOCVRPBTW</td><td colspan="2">MDOCVRPMBTW</td></tr><tr><td>LLM</td><td>59.50</td><td>14.42</td><td>43.75</td><td>8.50</td><td>56.96</td><td>10.10</td><td>66.07</td><td>10.28</td><td>60.00</td><td>10.53</td><td colspan="2">62.50</td></tr><tr><td>Self-Guided</td><td>90.00</td><td>15.20</td><td>96.67</td><td>9.83</td><td>97.83</td><td>11.22</td><td>98.55</td><td>12.53</td><td>94.50</td><td>11.27</td><td colspan="2">99.00</td></tr><tr><td></td><td></td><td>MDOCVRPBL</td><td>MDOCVRPMBL</td><td></td><td>MDCVRPBLTW</td><td></td><td>MDCVRPMBLTW</td><td></td><td>MDOCVRPBLTW</td><td></td><td></td><td>MDOCVRPMBLTW</td></tr><tr><td>LLM Self-Guided</td><td>42.45</td><td>9.95</td><td>60.00</td><td>8.12</td><td>31.02</td><td>9.78</td><td>61.95</td><td>12.39</td><td>14.03</td><td>13.53</td><td>30.56</td><td>13.13</td></tr><tr><td></td><td>100.00</td><td>10.78</td><td>99.52</td><td>10.23</td><td>98.61</td><td>12.43</td><td>100.00</td><td>11.60</td><td>99.10</td><td>15.28</td><td>98.46</td><td>14.52</td></tr></table>

## 5 Experiment

To evaluate the efectiveness of the proposed self-guided framework, we compare it against LLM across 48 VRP variants. Then we conduct a comprehensive analysis along multiple dimensions, including reasoning depth, error number, instance scale, and constraint complexity, to assess the performance improvements introduced by our framework. Furthermore, we provide comparisons with several prompting strategies to further validate the advantages of our approach. Finally, we extend our evaluation to other combinatorial optimization problems and LLMs to demonstrate the generality of the proposed framework.

To provide a more complete understanding of the framework’s behavior, additional analyses are presented in the appendix, including false positive analysis on valid instances (Appendix C.2), comparison of error detection accuracy on instance-level (Appendix C.3), and downstream solver validation on repaired instances (Appendix C.4). And details on hyperparameters and the experimental setup are provided in Appendix C.1.

## 5.1 Comparison of Error Detection Rate

We conduct experiments on 48 VRP variants to assess error detection performance. The comparison of Error Detection Rate (EDR) is reported in Table 2. As shown by these results, directly applying LLMs exhibits limited error detection capability on many variants, especially those involving multiple constraints or complex combinations, where the EDR often falls below 70% and in some cases even below 50%.

By contrast, the proposed self-guide approach consistently achieves improvements in error detection performance across all variants. In the most cases, self-guide raises the EDR to above 95%, and frequently reaches near-perfect detection rates, demonstrating its robustness under diverse problem settings and constraint combinations. Importantly, these performance gains are obtained without introducing significant additional computational overhead. As shown by the time statistics, self-guide only incurs a marginal increase in runtime compared with direct LLM inference, while delivering dramatically higher error detection accuracy.

Table 3: Impact of instance scale and fault number on EDR for LLM and the Self-Guided framework.
<table><tr><td>Scale</td><td colspan="6">50</td><td colspan="6">100</td></tr><tr><td>Fault Number</td><td colspan="2">5</td><td colspan="2">10</td><td colspan="2">15</td><td colspan="2">5</td><td colspan="2">10</td><td colspan="2">15</td></tr><tr><td>EDR(%)</td><td>LLM</td><td>Self-Guided</td><td>LLM</td><td>Self-Guided</td><td>LLM</td><td>Self-Guided</td><td>LLM</td><td>Self-Guided</td><td>LLM</td><td>Self-Guided</td><td>LLM</td><td>Self-Guided</td></tr><tr><td>CVRP</td><td>97.00</td><td>100.00</td><td>87.25</td><td>99.49</td><td>86.82</td><td>98.99</td><td>94.00</td><td>98.00</td><td>89.80</td><td>97.96</td><td>82.33</td><td>97.90</td></tr><tr><td>CVRPTW</td><td>79.00</td><td>97.00</td><td>66.50</td><td>95.00</td><td>63.67</td><td>90.33</td><td>68.00</td><td>98.00</td><td>64.00</td><td>93.00</td><td>62.33</td><td>87.67</td></tr><tr><td>CVRPB</td><td>82.00</td><td>94.00</td><td>78.50</td><td>94.00</td><td>56.02</td><td>91.73</td><td>76.00</td><td>95.00</td><td>77.00</td><td>92.00</td><td>53.33</td><td>92.33</td></tr><tr><td>OCVRPB</td><td>93.00</td><td>95.00</td><td>84.00</td><td>94.00</td><td>71.62</td><td>88.85</td><td>85.00</td><td>99.00</td><td>79.00</td><td>89.00</td><td>64.33</td><td>87.00</td></tr><tr><td>CVRPMBL</td><td>70.05</td><td>99.00</td><td>65.00</td><td>91.50</td><td>47.52</td><td>88.37</td><td>67.74</td><td>100.00</td><td>60.34</td><td>93.50</td><td>45.97</td><td>87.58</td></tr><tr><td>MDCVRPMBTW</td><td>70.00</td><td>92.00</td><td>59.50</td><td>90.00</td><td>54.67</td><td>88.33</td><td>69.00</td><td>98.00</td><td>56.50</td><td>91.00</td><td>52.67</td><td>87.67</td></tr><tr><td>MDOCVRPBTW</td><td>66.00</td><td>96.00</td><td>60.00</td><td>94.50</td><td>49.67</td><td>86.67</td><td>58.00</td><td>100.00</td><td>55.50</td><td>95.50</td><td>48.33</td><td>94.66</td></tr><tr><td>MDOCVRPMBLTW</td><td>72.59</td><td>98.56</td><td>69.44</td><td>98.46</td><td>44.67</td><td>96.00</td><td>54.98</td><td>97.26</td><td>44.46</td><td>97.85</td><td>39.87</td><td>96.46</td></tr></table>

Table 4: Impact of Reasoning Depth on EDR for LLM and the Self-Guided Framework

<table><tr><td rowspan=1 colspan=1>Reasoning Depth</td><td rowspan=1 colspan=2>1</td><td rowspan=1 colspan=2>2</td><td rowspan=1 colspan=2>3</td><td rowspan=1 colspan=2>4</td><td rowspan=1 colspan=2>5</td></tr><tr><td rowspan=1 colspan=1>EDR(%)</td><td rowspan=1 colspan=2>LLM Self-Guided</td><td rowspan=1 colspan=2>LLM Self-Guided</td><td rowspan=1 colspan=2>LLM Self-Guided</td><td rowspan=1 colspan=2>LLM Self-Guided</td><td rowspan=1 colspan=2>LLM Self-Guided</td></tr><tr><td rowspan=1 colspan=1>CVRP</td><td rowspan=1 colspan=2>87.25    99.49</td><td rowspan=1 colspan=2>94.89    100.00</td><td rowspan=1 colspan=2>96.42    100.00</td><td rowspan=1 colspan=2>97.45    100.00</td><td rowspan=1 colspan=1>97.96</td><td rowspan=1 colspan=1>100.00</td></tr><tr><td rowspan=1 colspan=1>CVRPTW</td><td rowspan=1 colspan=1>66.50</td><td rowspan=1 colspan=1>95.00</td><td rowspan=1 colspan=2>68.00    98.00</td><td rowspan=1 colspan=1>69.00</td><td rowspan=1 colspan=1>98.00</td><td rowspan=1 colspan=1>70.00</td><td rowspan=1 colspan=1>99.00</td><td rowspan=1 colspan=1>72.50</td><td rowspan=1 colspan=1>100.00</td></tr><tr><td rowspan=1 colspan=1>CVRPB</td><td rowspan=1 colspan=1>78.50</td><td rowspan=1 colspan=1>94.00</td><td rowspan=1 colspan=1>85.00</td><td rowspan=1 colspan=1>96.00</td><td rowspan=1 colspan=1>90.00</td><td rowspan=1 colspan=1>96.00</td><td rowspan=1 colspan=1>94.00</td><td rowspan=1 colspan=1>99.00</td><td rowspan=1 colspan=1>99.00</td><td rowspan=1 colspan=1>100.00</td></tr><tr><td rowspan=1 colspan=1>OCVRPB</td><td rowspan=1 colspan=1>84.00</td><td rowspan=1 colspan=1>94.00</td><td rowspan=1 colspan=1>88.50</td><td rowspan=1 colspan=1>99.50</td><td rowspan=1 colspan=1>90.50</td><td rowspan=1 colspan=1>100.00</td><td rowspan=1 colspan=1>93.00</td><td rowspan=1 colspan=1>100.00</td><td rowspan=1 colspan=1>94.50</td><td rowspan=1 colspan=1>100.00</td></tr><tr><td rowspan=1 colspan=1>CVRPMBL</td><td rowspan=1 colspan=1>65.00</td><td rowspan=1 colspan=1>91.50</td><td rowspan=1 colspan=1>67.50</td><td rowspan=1 colspan=1>95.50</td><td rowspan=1 colspan=1>69.00</td><td rowspan=1 colspan=1>100.00</td><td rowspan=1 colspan=1>70.00</td><td rowspan=1 colspan=1>100.00</td><td rowspan=1 colspan=1>70.50</td><td rowspan=1 colspan=1>100.00</td></tr><tr><td rowspan=1 colspan=1>MDCVRPMBTW</td><td rowspan=1 colspan=1>59.50</td><td rowspan=1 colspan=1>90.00</td><td rowspan=1 colspan=1>60.00</td><td rowspan=1 colspan=1>99.00</td><td rowspan=1 colspan=1>60.00</td><td rowspan=1 colspan=1>99.50</td><td rowspan=1 colspan=1>61.00</td><td rowspan=1 colspan=1>100.00</td><td rowspan=1 colspan=1>65.50</td><td rowspan=1 colspan=1>100.00</td></tr><tr><td rowspan=1 colspan=1>MDOCVRPBTW</td><td rowspan=1 colspan=1>60.00</td><td rowspan=1 colspan=1>94.50</td><td rowspan=1 colspan=1>61.50</td><td rowspan=1 colspan=1>98.50</td><td rowspan=1 colspan=1>62.00</td><td rowspan=1 colspan=1>99.00</td><td rowspan=1 colspan=1>62.50</td><td rowspan=1 colspan=1>99.00</td><td rowspan=1 colspan=1>64.00</td><td rowspan=1 colspan=1>100.00</td></tr><tr><td rowspan=1 colspan=1>MDOCVRPMBLTW</td><td rowspan=1 colspan=1>30.56</td><td rowspan=1 colspan=1>98.46</td><td rowspan=1 colspan=1>31.00</td><td rowspan=1 colspan=1>99.54</td><td rowspan=1 colspan=1>34.93</td><td rowspan=1 colspan=1>100.00</td><td rowspan=1 colspan=2>35.26    100.00</td><td rowspan=1 colspan=1>36.24</td><td rowspan=1 colspan=1>100.00</td></tr></table>

## 5.2 Performance Analysis

Impact of Error Number. Table 3 shows that increasing the number of injected errors significantly degrades the error detection performance of direct LLM inference, especially for instances with complex constraints such as CVRPMBL and MDOCVRPMBLTW. In contrast, the self-guided framework remains robust to error number variations, consistently achieving substantially higher EDR and often maintaining over 90% detection accuracy even when the error number reaches 15.

Impact of Instance Scale. The impact of instance scale is also evident from Table 3. When moving from 50-node to 100-node instances, the error detection performance of direct LLM inference generally deteriorates, especially under larger error numbers. This performance drop highlights the dificulty of handling larger problem instances with more complex interactions using pure LLM-based reasoning. In contrast, the self-guided framework exhibits much better scalability. While instance scale increases the overall difficulty of the task, self-guided framework maintains consistently high EDR across most variants, with only moderate performance degradation in the most challenging settings. This demonstrates that the proposed framework scales more efectively with instance size, enabling robust error detection even on larger and more complex VRP instances.

Efect of Reasoning Depth. Table 4 reports the impact of reasoning depth on error detection performance for both direct LLM inference and the proposed self-guided framework. As the reasoning depth increases, direct LLM inference shows only limited and gradual improvements, and fails to consistently achieve perfect error detection across diferent VRP variants. In contrast, the self-guided framework exhibits clear and stable convergence behavior. Under our framework, a reasoning depth of five is suficient to guarantee a 100% error detection rate across all evaluated VRP variants.

A detailed analysis of the impact of constraints type count is provided in Appendix C.5.

## 5.3 Comparison with Prompt Strategies

To demonstrate the advantage of self-guided framework, we compare it with self-refine (Madaan et al. 2023), criticguided refinement (Bai et al. 2022) and COT (Wei et al. 2022). All methods are evaluated under the same experimental settings to ensure a fair comparison.

As shown in Table 5, while Self-Refine, critic-guided refinement, and CoT consistently improve upon direct LLM inference, their gains remain limited across complex VRP variants. In contrast, the self-guided framework achieves substantially higher EDR across all tested instances, particularly on more challenging variants. Moreover, this performance improvement is obtained with only modestly increased runtime, demonstrating that the self-guided framework ofers a more efective and robust solution for error detection.

## 5.4 Evaluation on Other Combinatorial Optimization Problems

We additionally apply our framework to the Bin Packing Problem (BPP). BPP is a classical combinatorial optimization problem that aims to pack a set of items into the minimum number of bins subject to capacity constraints. In the one-dimensional setting (1D BPP), each item is characterized by a single size, whereas in the two-dimensional setting (2D BPP), items have both width and height, introducing additional geometric constraints and increasing problem complexity.

Table 5: Comparison with Diferent Prompt Strategies.
<table><tr><td></td><td colspan="2">LLM</td><td colspan="2">Self-Refine</td><td colspan="2">Critic-Guided Refinement</td><td colspan="2">CoT</td><td colspan="2">Self-Guided</td></tr><tr><td></td><td>EDR(%)</td><td>Time(m)</td><td>EDR(%)</td><td>Time(m)</td><td>EDR(%)</td><td>Time(m)</td><td>EDR(%)</td><td>Time(m)</td><td>EDR(%)</td><td>Time(m)</td></tr><tr><td>CVRP</td><td>87.25</td><td>6.35</td><td>92.86</td><td>9.60</td><td>92.35</td><td>9.12</td><td>91.33</td><td>7.13</td><td>99.49</td><td>6.39</td></tr><tr><td>CVRPTW</td><td>66.50</td><td>7.00</td><td>69.00</td><td>11.66</td><td>68.50</td><td>13.92</td><td>69.00</td><td>8.10</td><td>95.00</td><td>7.22</td></tr><tr><td>CVRPB</td><td>78.50</td><td>8.25</td><td>82.00</td><td>11.43</td><td>83.50</td><td>10.52</td><td>83.00</td><td>7.60</td><td>94.00</td><td>9.03</td></tr><tr><td>OCVRPB</td><td>84.00</td><td>8.55</td><td>84.50</td><td>11.33</td><td>86.00</td><td>11.08</td><td>85.00</td><td>6.78</td><td>94.00</td><td>8.52</td></tr><tr><td>CVRPMBL</td><td>65.00</td><td>7.63</td><td>69.00</td><td>10.12</td><td>70.50</td><td>10.50</td><td>67.00</td><td>5.88</td><td>91.50</td><td>7.93</td></tr><tr><td>MDCVRPMBTW</td><td>59.50</td><td>14.42</td><td>65.50</td><td>12.60</td><td>60.00</td><td>14.73</td><td>62.00</td><td>9.05</td><td>90.00</td><td>15.20</td></tr><tr><td>MDOCVRPBTW</td><td>60.00</td><td>10.53</td><td>62.00</td><td>11.61</td><td>61.50</td><td>14.96</td><td>60.50</td><td>9.83</td><td>94.50</td><td>11.27</td></tr><tr><td>MDOCVRPMBLTW</td><td>30.56</td><td>10.13</td><td>34.89</td><td>13.20</td><td>31.88</td><td>16.00</td><td>33.18</td><td>9.15</td><td>98.46</td><td>14.52</td></tr></table>

Table 6: EDR and Runtime on Bin Packing Problems.
<table><tr><td></td><td colspan="2">LLM</td><td colspan="2">Code</td></tr><tr><td></td><td>EDR(%)</td><td>Time(m)</td><td>EDR(%)</td><td>Time(m)</td></tr><tr><td>1D BPP</td><td>94.50</td><td>5.88</td><td>97.50</td><td>6.00</td></tr><tr><td>2D BPP</td><td>90.00</td><td>6.97</td><td>94.50</td><td>7.35</td></tr></table>

As shown in Table 6, the proposed self-guided framework consistently outperforms direct LLM inference on both 1D and 2D BPP. While the LLM already achieves relatively high error detection rates, our framework further improves performance, reaching 97.50% on 1D BPP and 94.50% on 2D BPP. These improvements are obtained with only a marginal increase in runtime, demonstrating that the self-guided framework can efectively extend to other combinatorial optimization problems.

## 5.5 Application to Other LLMs

Table 7: Comparison across diferent LLMs with and without the Self-Guided framework on EDR (%).
<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>GPT-4.1</td><td rowspan=1 colspan=2>GPT-40</td><td rowspan=1 colspan=2>Claude-4</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=6>LLM Self-GuidedLLM Self-GuidedLLM Self-Guided</td></tr><tr><td rowspan=1 colspan=1>CVRP</td><td rowspan=1 colspan=1>87.25</td><td rowspan=1 colspan=1>99.49</td><td rowspan=1 colspan=1>85.71</td><td rowspan=1 colspan=1>98.99</td><td rowspan=1 colspan=1>91.84</td><td rowspan=1 colspan=1>100.00</td></tr><tr><td rowspan=1 colspan=1>CVRPTW</td><td rowspan=1 colspan=1>66.50</td><td rowspan=1 colspan=1>95.00</td><td rowspan=1 colspan=1>60.00</td><td rowspan=1 colspan=1>93.50</td><td rowspan=1 colspan=1>68.50</td><td rowspan=1 colspan=1>96.00</td></tr><tr><td rowspan=1 colspan=1>CVRPB</td><td rowspan=1 colspan=1>78.50</td><td rowspan=1 colspan=1>94.00</td><td rowspan=1 colspan=1>75.00</td><td rowspan=1 colspan=1>93.00</td><td rowspan=1 colspan=1>80.00</td><td rowspan=1 colspan=1>95.50</td></tr><tr><td rowspan=1 colspan=1>OCVRPB</td><td rowspan=1 colspan=1>84.00</td><td rowspan=1 colspan=1>94.00</td><td rowspan=1 colspan=1>85.50</td><td rowspan=1 colspan=1>95.00</td><td rowspan=1 colspan=1>86.50</td><td rowspan=1 colspan=1>95.50</td></tr><tr><td rowspan=1 colspan=1>CVRPMBL</td><td rowspan=1 colspan=1>65.00</td><td rowspan=1 colspan=1>91.50</td><td rowspan=1 colspan=1>63.00</td><td rowspan=1 colspan=1>90.00</td><td rowspan=1 colspan=1>66.50</td><td rowspan=1 colspan=1>92.00</td></tr><tr><td rowspan=1 colspan=1>MDCVRPMBTW</td><td rowspan=1 colspan=1>59.50</td><td rowspan=1 colspan=1>90.00</td><td rowspan=1 colspan=1>55.00</td><td rowspan=1 colspan=1>88.50</td><td rowspan=1 colspan=1>61.00</td><td rowspan=1 colspan=1>92.00</td></tr><tr><td rowspan=1 colspan=1>MDOCVRPBTW</td><td rowspan=1 colspan=1>60.00</td><td rowspan=1 colspan=1>94.50</td><td rowspan=1 colspan=1>58.50</td><td rowspan=1 colspan=1>93.50</td><td rowspan=1 colspan=1>61.50</td><td rowspan=1 colspan=1>95.00</td></tr><tr><td rowspan=1 colspan=1>MDOCVRPMBLTW</td><td rowspan=1 colspan=1>30.56</td><td rowspan=1 colspan=1>98.46</td><td rowspan=1 colspan=1>26.85</td><td rowspan=1 colspan=1>94.45</td><td rowspan=1 colspan=1>34.07</td><td rowspan=1 colspan=1>99.54</td></tr></table>

We further evaluate the proposed self-guided framework on GPT-4o (OpenAI 2024) and Claude Opus 4 (Anthropic 2025) to assess its generality across LLM backends. As shown in Table 7, direct inference degrades substantially on complex VRP variants across GPT-4.1, GPT-4o, and Claude Opus 4, especially under multiple coupled constraints and long-horizon reasoning, suggesting that this limitation is not model-specific.

In contrast, the proposed self-guided framework consistently yields substantial improvements in EDR across all models and problem variants. Notably, even on the most challenging instances, self-guided inference raises the EDR to above 90% in most cases, demonstrating stable gains across diferent LLM families. These results suggest that the proposed framework is largely model-agnostic and can efectively enhance error detection reliability regardless of the underlying LLM backbone.

## 6 Conclusion

In this work, we investigate error detection in vehicle routing problem instances and identifies fundamental limitations of large language models (LLMs) when confronted with instances or errors of higher complexity. To address these challenges, we introduce a self-guided reasoning framework that enables LLMs to generate reasoning paths to guide code generation and produce additional insights, thus enhancing their own reasoning capabilities. Extensive experiments across diverse problem settings and LLM backbones demonstrate that the proposed approach consistently and substantially improves error detection performance, achieving near-perfect accuracy and underscoring its efectiveness and generality.

Despite its efectiveness, the proposed self-guided framework has certain limitations. While we evaluate the framework on a wide range of COPs and LLM backbones, further validation on additional problem domains and real-world noisy datasets would be valuable to fully assess its generalization. Moreover, some errors in problem instances may interfere with the detection of other errors; for example, missing or malformed sections can obscure subsequent error detection. Although our framework alleviates this issue through iterative reasoning, such cascading efects cannot be entirely eliminated in the initial reasoning round.

Anthropic. 2025. Introducing Claude 4. https://www. anthropic.com/news/claude-4. Accessed: 2026-01-05.

Bai, Y.; Kadavath, S.; Kundu, S.; Askell, A.; Kernion, J.; Jones, A.; Chen, A.; Goldie, A.; Mirhoseini, A.; McKinnon, C.; et al. 2022. Constitutional ai: Harmlessness from ai feedback. arXiv preprint arXiv:2212.08073.

Berto, F.; Hua, C.; Zepeda, N. G.; Hottung, A.; Wouda, N.; Lan, L.; Park, J.; Tierney, K.; and Park, J. 2025. RouteFinder: Towards Foundation Models for Vehicle Routing Problems. Transactions on Machine Learning Research.

Chen, W.; Mes, M.; Schutten, M.; and Quint, J. 2019. A ridesharing problem with meeting points and return restrictions. Transportation science, 53(2): 401–426.

Dat, P. V. T.; Doan, L.; and Binh, H. T. T. 2025. HSEvo: Elevating automatic heuristic design with diversity-driven harmony search and genetic algorithm using llms. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, 26931–26938.

Gayialis, S. P.; Konstantakopoulos, G. D.; and Tatsiopoulos, I. P. 2019. Vehicle routing problem for urban freight transportation: A review of the recent literature. Operational research in the digital era–ICT challenges, 89–104.

Guo, J.; Long, J.; Xu, X.; Yu, M.; and Yuan, K. 2022. The vehicle routing problem of intercity ride-sharing between two cities. Transportation Research Part B: Methodological, 158: 113–139.

Jiang, X.; Wu, Y.; Li, M.; Cao, Z.; and Zhang, Y. 2025. Large Language Models as End-to-end Combinatorial Optimization Solvers. In The Thirty-ninth Annual Conference on Neural Information Processing Systems.

Kitjacharoenchai, P.; and Lee, S. 2019. Vehicle routing problem with drones for last mile delivery. Procedia Manufacturing, 39: 314–324.

Li, K.; Zheng, R.; Hao, X.; and Wang, Z. 2025. Multiobjective infeasibility diagnosis for routing problems using large language models. arXiv preprint arXiv:2508.03406.

Lin, Y.; Li, W.; Qiu, F.; and Xu, H. 2012. Research on optimization of vehicle routing problem for ride-sharing taxi. Procedia-Social and Behavioral Sciences, 43: 494–502.

Liu, F.; Tong, X.; Yuan, M.; Lin, X.; Luo, F.; Wang, Z.; Lu, Z.; and Zhang, Q. 2024a. Evolution of Heuristics: Towards Eficient Automatic Algorithm Design Using Large Language Model. In 41st International Conference on Machine Learning (ICML 2024), 32201–32223. ML Research Press.

Liu, S.; Chen, C.; Qu, X.; Tang, K.; and Ong, Y.-S. 2024b. Large language models as evolutionary optimizers. In 2024 IEEE Congress on Evolutionary Computation (CEC), 1–8. IEEE.

Madaan, A.; Tandon, N.; Gupta, P.; Hallinan, S.; Gao, L.; Wiegrefe, S.; Alon, U.; Dziri, N.; Prabhumoye, S.; Yang, Y.; et al. 2023. Self-refine: Iterative refinement with selffeedback. Advances in Neural Information Processing Systems, 36: 46534–46594.

OpenAI. 2023. GPT-4 Technical Report. arXiv preprint arXiv:2303.08774.

OpenAI. 2024. Hello GPT-4o. https://openai.com/index/ hello-gpt-4o/. Accessed: 2026-01-05.

Romera-Paredes, B.; Barekatain, M.; Novikov, A.; Balog, M.; Kumar, M. P.; Dupont, E.; Ruiz, F. J.; Ellenberg, J. S.; Wang, P.; Fawzi, O.; et al. 2024. Mathematical discoveries from program search with large language models. Nature, 625(7995): 468–475.

Tilk, C.; Olkis, K.; and Irnich, S. 2021. The last-mile vehicle routing problem with delivery options. Or Spectrum, 43(4): 877–904.

Tiwari, K. V.; and Sharma, S. K. 2023. An optimization model for vehicle routing problem in last-mile delivery. Expert Systems with Applications, 222: 119789.

Wang, Q.; Kong, L.; Xu, J.; Zhou, B.; Liu, X.; Lin, Z.; Shi, S.; Zhang, X.; and Li, L. 2024. Dual-driven interfaces of a CoP/CoO cocatalyst on a host photocatalyst for rapid charge transport in solar-driven H2 evolution. ACS Sustainable Chemistry & Engineering, 12(31): 11717–11727.

Wei, J.; Wang, X.; Schuurmans, D.; Bosma, M.; Xia, F.; Chi, E.; Le, Q. V.; Zhou, D.; et al. 2022. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35: 24824–24837.

Xu, D.; Liu, S.; Zhang, M.; Xu, L.; Gao, H.; and Yao, J. 2023. Manipulating the Dynamic Self-Reconstruction of CoP Electrocatalyst Driven by Charge Transport and Ion Leaching. Small, 19(33): 2300201.

Yang, C.; Wang, X.; Lu, Y.; Liu, H.; Le, Q. V.; Zhou, D.; and Chen, X. 2024. Large Language Models as Optimizers. In International Conference on Learning Representations.

Ye, H.; Wang, J.; Cao, Z.; Berto, F.; Hua, C.; Kim, H.; Park, J.; and Song, G. 2024. ReEvo: Large language models as hyper-heuristics with reflective evolution. Advances in neural information processing systems, 37: 43571–43608.

Zhang, N.; Cao, Z.; Zhou, J.; Zhang, C.; and Ong, Y.-S. 2025. An Agentic Framework with LLMs for Solving Complex Vehicle Routing Problems. arXiv preprint arXiv:2510.16701.

## A Vehicle Routing Problems

Vehicle Routing Problems (VRPs) are a fundamental class of combinatorial optimization problems (COPs) that aim to design a set of vehicle routes to serve a collection of customers while minimizing a given objective, typically the total travel distance or cost. In a standard VRP setting, all routes start and end at a depot, each customer must be visited exactly once, and various operational constraints must be satisfied. Due to the rich combination of complex constraints and diverse real-world application scenarios, VRPs are commonly used as a representative benchmark for COPs.

Several constraints are commonly studied in VRPs, including capacity, open route, and multi-depot. The details of these constraints are summarized in Table 8, and the relationships between constraints and VRP variants are illustrated in Table 9.

Table 8: Summary of VRP constraints considered in this work.
<table><tr><td rowspan=1 colspan=1>Constraint</td><td rowspan=1 colspan=1>Description</td></tr><tr><td rowspan=1 colspan=1>C</td><td rowspan=1 colspan=1>Capacity constraint, which limits the total demand served byeach vehicle so that it does not exceed its carrying capacity.</td></tr><tr><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>Open route constraint, where vehicles are not required toreturn to the depot after completing their routes.</td></tr><tr><td rowspan=1 colspan=1>B</td><td rowspan=1 colspan=1>Backhaul constraint, where customers are partitioned intolinehaul and backhaul sets, and vehicles are required to serveall linehaul customers before visiting any backhaul customerson each route.</td></tr><tr><td rowspan=1 colspan=1>MB</td><td rowspan=1 colspan=1>Mixed backhaul constraint, which relaxes strict service order-ing and allows limited interleaving of linehaul and backhaulcustomers.</td></tr><tr><td rowspan=1 colspan=1>L</td><td rowspan=1 colspan=1>Distance limit constraint, which restricts the maximum traveldistance of each route.</td></tr><tr><td rowspan=1 colspan=1>TW</td><td rowspan=1 colspan=1>Time window constraint, requiring each customer to beserved within a specified time interval.</td></tr><tr><td rowspan=1 colspan=1>MD</td><td rowspan=1 colspan=1>Multi-depot constraint, which allows vehicles to start frommultiple depots rather than a single centralized one.</td></tr></table>

## B Dataset

## B.1 Assumption

We construct our error detection dataset based on benchmark instances from the Route Finder dataset (Berto et al. 2025). To evaluate the capability of LLMs in identifying instance-level errors, we systematically inject synthetic errors into valid VRP instances. To ensure that each faulty instance admits a unique and unambiguous interpretation, we impose several assumptions during dataset construction, as described below.

a) Capacity and Distance Limit Correctness. Capacity values and distance limits are assumed to be correct and are not subject to corruption. This assumption enables us to disambiguate between potential error sources, such as distinguishing demand-related errors from capacity violations, and separating node coordinate errors from distance limit misconfigurations.

b) Single Error Per Node. At most one error is injected per node. For example, if a node is assigned missing or invalid coordinates, no additional errors are introduced for that node. This constraint prevents overlapping errors that could lead to ambiguous error attribution.

c) Depot Error Exclusion. When distance limit or time window errors are present, no errors are injected at the depot. This restriction is imposed because depot errors would interfere with the computation of these constraints.

These assumptions enable controlled error injection while avoiding cascading or interacting errors, facilitating a reliable and interpretable evaluation of error detection performance.

## B.2 Format Errors

(i) Value Error. Value errors refer to numerical values that violate basic validity constraints, such as identifiers being out of range or numerical fields taking negative values where only non-negative values are permitted. For example, a customer ID exceeding the defined index range or a negative demand value constitutes a value error in the CVRP dataset.

(ii) Value Missing. Value missing errors occur when required numerical fields are absent in an instance. Typical examples include missing demand values, service times, or coordinates for a customer node.

(iii) Section Missing. Section missing errors arise when the name of a required section in the instance description is absent, even though the corresponding data entries may still be present.

(vi) Duplicate Node. Duplicate node errors occur when the same node appears multiple times in an instance.

## B.3 Feasibility Errors

(i) Capacity (C). This constraint is associated with exceed capacity errors, which occur when the total demand assigned to a vehicle exceeds its capacity limit. For LLMs, detecting such errors mainly requires a straightforward comparison between the demand and the capacity value. As a result, this error type is relatively easy to identify, and LLMs typically achieve high detection accuracy.

(ii) Backhaul (B) and Mixed Backhaul (MB). For backhaul and mixed backhaul constraints, customer demands may take positive (linehaul) or negative (backhaul) values, and feasibility requires that the absolute value of the accumulated demand along a route does not exceed the vehicle capacity. Compared with the standard capacity constraint, error detection under these settings involves an additional absolute-value operation, which slightly increases the reasoning dificulty and results in lower detection accuracy.

(iii) Distance Limit (L). For the distance limit constraint, we compute the maximum round-trip distance between the depot and all customer nodes, and subtract a small margin (0.1) to obtain the distance limit value. We then check whether the round-trip distance of each node to the depot falls within this limit and record related violations.

Table 9: VRP variants and their associated constraints.
<table><tr><td>VRP Variant</td><td>C Capacity</td><td>0 Open</td><td>B Backhaul</td><td>MB Mixed Backhaul</td><td>L Distance Limit</td><td>TW Time Windows</td><td>MD Multi-Depot</td></tr><tr><td>CVRP</td><td>√</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>OCVRP</td><td>√</td><td>√</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>CVRPB</td><td>√</td><td></td><td>√</td><td></td><td></td><td></td><td></td></tr><tr><td>CVRPL</td><td>√</td><td></td><td></td><td></td><td>√</td><td></td><td></td></tr><tr><td>CVRPTW</td><td>√</td><td></td><td></td><td></td><td></td><td>√</td><td></td></tr><tr><td>OCVRPTW</td><td>√</td><td>√</td><td></td><td></td><td></td><td>√</td><td></td></tr><tr><td>OCVRPB</td><td>√</td><td>√</td><td>√</td><td></td><td></td><td></td><td></td></tr><tr><td>OCVRPL</td><td>√</td><td>√</td><td></td><td></td><td>√</td><td></td><td></td></tr><tr><td>CVRPBL</td><td>√</td><td></td><td>√</td><td></td><td>√</td><td></td><td></td></tr><tr><td>CVRPBTW</td><td>√</td><td></td><td>√</td><td></td><td></td><td>√</td><td></td></tr><tr><td>CVRPLTW</td><td>√</td><td></td><td></td><td></td><td>√</td><td>√</td><td></td></tr><tr><td>OCVRPBL</td><td>√</td><td>√</td><td>√</td><td></td><td>√</td><td></td><td></td></tr><tr><td>OCVRPBTW</td><td>√</td><td>√</td><td>√</td><td></td><td></td><td>√</td><td></td></tr><tr><td>OCVRPLTW</td><td>√</td><td>√</td><td></td><td></td><td>√</td><td>√</td><td></td></tr><tr><td>CVRPBLTW</td><td>√</td><td></td><td>√</td><td></td><td>√</td><td>√</td><td></td></tr><tr><td>OCVRPBLTW</td><td>√</td><td>V</td><td>√</td><td></td><td>√</td><td>√</td><td></td></tr><tr><td>CVRPMB</td><td>√</td><td></td><td>√</td><td>√</td><td></td><td></td><td></td></tr><tr><td>OCVRPMB</td><td>√</td><td>√</td><td>√</td><td>√</td><td></td><td></td><td></td></tr><tr><td>CVRPMBL</td><td>√</td><td></td><td>√</td><td>√</td><td>√</td><td></td><td></td></tr><tr><td>CVRPMBTW</td><td>√</td><td></td><td>√</td><td>√</td><td></td><td>√</td><td></td></tr><tr><td>OCVRPMBL</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td></td><td></td></tr><tr><td>OCVRPMBTW</td><td>√</td><td>√</td><td>√</td><td>√</td><td></td><td>√</td><td></td></tr><tr><td>CVRPMBLTW</td><td>√</td><td></td><td>√</td><td>√</td><td>√</td><td>√</td><td></td></tr><tr><td>OCVRPMBLTW</td><td>V</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td></td></tr><tr><td>MDCVRP</td><td>√</td><td></td><td></td><td></td><td></td><td></td><td>√</td></tr><tr><td>MDOCVRP</td><td>√</td><td>√</td><td></td><td></td><td></td><td></td><td>√</td></tr><tr><td>MDCVRPB</td><td>√</td><td></td><td>√</td><td></td><td></td><td></td><td>√</td></tr><tr><td>MDCVRPL</td><td>√</td><td></td><td></td><td></td><td>√</td><td></td><td>√</td></tr><tr><td>MDCVRPTW</td><td>√</td><td></td><td></td><td></td><td></td><td>√</td><td>√</td></tr><tr><td>MDOCVRPTW</td><td>√</td><td>√</td><td></td><td></td><td></td><td>√</td><td>√</td></tr><tr><td>MDOCVRPB</td><td>√</td><td>√</td><td>√</td><td></td><td></td><td></td><td>√</td></tr><tr><td>MDOCVRPL</td><td>√</td><td>√</td><td></td><td></td><td>√</td><td></td><td>√</td></tr><tr><td>MDCVRPBL</td><td>√</td><td></td><td>√</td><td></td><td>√</td><td></td><td>√</td></tr><tr><td>MDCVRPBTW</td><td>√</td><td></td><td>√</td><td></td><td></td><td>√</td><td>√</td></tr><tr><td>MDCVRPLTW</td><td>√</td><td></td><td></td><td></td><td>√</td><td>√</td><td>√</td></tr><tr><td>MDOCVRPBL</td><td>√</td><td>√</td><td>√</td><td></td><td>√</td><td></td><td>√</td></tr><tr><td>MDOCVRPBTW</td><td>√</td><td>√</td><td>√</td><td></td><td></td><td>√</td><td>√</td></tr><tr><td>MDOCVRPLTW</td><td>√</td><td>√</td><td></td><td></td><td>√</td><td>√</td><td>√</td></tr><tr><td>MDCVRPBLTW</td><td>√</td><td></td><td>√</td><td></td><td>√</td><td>√</td><td>√</td></tr><tr><td>MDOCVRPBLTW</td><td>√</td><td>√</td><td>√</td><td></td><td>√</td><td>√</td><td>√</td></tr><tr><td>MDCVRPMB</td><td>√</td><td></td><td>√</td><td>√</td><td></td><td></td><td>√</td></tr><tr><td>MDOCVRPMB</td><td>√</td><td>√</td><td>√</td><td>√</td><td></td><td></td><td>√</td></tr><tr><td>MDCVRPMBL</td><td>√</td><td></td><td>√</td><td>√</td><td>√</td><td></td><td>√</td></tr><tr><td>MDCVRPMBTW</td><td>√</td><td></td><td>√</td><td>√</td><td></td><td>√</td><td>√</td></tr><tr><td>MDOCVRPMBL</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td></td><td>√</td></tr><tr><td>MDOCVRPMBTW</td><td>√</td><td>√</td><td>√</td><td>√</td><td></td><td>√</td><td>√</td></tr><tr><td>MDCVRPMBLTW</td><td>√</td><td></td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td></tr><tr><td>MDOCVRPMBLTW</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td></tr></table>

(iv) Open Route (O). Under the open route constraint, vehicles are not required to return to the depot after completing their routes. Accordingly, the error injection for the distance limit constraint is modified to compute the maximum one-way distance from the depot to all customers and subtract a small margin (0.1) to define the new distance limit value and record violations.

(v) Time Window (TW). For the time window constraint, we randomly select a node that has not been assigned any error and compute its travel distance from the depot. The left bound of the time window is set to 0.7 times this distance, and the right bound is set to 0.9 times the same distance, which is designed to make the time window infeasible for that node.

(vi) Multi-Depot (MD). Under the multi-depot setting, distance-related constraints are evaluated with respect to the nearest depot. Specifically, the distance limit is computed based on the maximum round-trip distance between each node and its closest depot, rather than a single fixed depot. Similarly, when injecting time window errors, the travel distance used to construct the time window bounds is calculated with respect to the nearest depot to the selected node.

## C More Experiment Details

## C.1 Hyperparameters and Device.

For error detection evaluation, we adopt benchmark instances from the RouteFinder dataset (Berto et al. 2025). For each dataset consisting of 20 instances with 50 nodes, we inject up to 200 errors to construct faulty datasets. For certain datasets (e.g., CVRP), the number of injected errors is correspondingly lower due to structural constraints. All experiments are conducted using GPT-4.1 (OpenAI 2023) as the underlying LLM for error detection on a system equipped with an AMD EPYC 7702P CPU. We will make our code and data publicly available after publication. The reported data are averaged.

## C.2 False Positive Analysis on Valid Instances

Table 10: Number of faults detected by LLM on correct VRP instances.
<table><tr><td></td><td>CVRP</td><td>OCVRP</td><td>MDCVRP</td><td>CVRPB</td><td>CVRPMB</td><td>CVRPTW</td><td>CVRPL</td></tr><tr><td>LLM</td><td>2</td><td>1</td><td>0</td><td>11</td><td>13</td><td>4</td><td>0</td></tr><tr><td>Self-Guided</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr></table>

We further evaluate the error detection behavior on correct VRP instances. As shown in Table 10, direct LLM inference may incorrectly report non-existent errors even when the input instance is valid, which results in false positives across several VRP variants. In contrast, the proposed self-guided framework does not exhibit this issue: when no errors are present in the instance, it consistently reports the instance as correct. This demonstrates that self-guided reasoning effectively reduces false alarms and improves the reliability of error detection.

## C.3 Comparison of Instance-Level Error Detection Accuracy

Table 11: Percentage of instances where all faults are correctly detected.
<table><tr><td>(%)</td><td>CVRP</td><td>OCVRP</td><td>MDCVRP</td><td>CVRPB</td><td>CVRPMB</td><td>CVRPTW</td><td>CVRPL</td></tr><tr><td>LLM</td><td>65</td><td>85</td><td>85</td><td>55</td><td>75</td><td>10</td><td>5</td></tr><tr><td>Self-Guided</td><td>95</td><td>95</td><td>95</td><td>85</td><td>90</td><td>90</td><td>100</td></tr></table>

Table 11 reports the instance-level error detection accuracy, measured as the percentage of instances for which all errors are correctly identified. The results show that direct LLM inference struggles to achieve reliable instance-level correctness, especially for variants with complex constraints such as CVRPTW and CVRPL, where the accuracy drops to 10% and 5%, respectively. In contrast, the proposed selfguided framework consistently achieves higher instance-level accuracy across all VRP variants, with performance remaining above 85% and reaching 100% on CVRPL. These results demonstrate that self-guided reasoning substantially improves the reliability of error detection at the instance level, particularly in settings with complex constraints.

## C.4 Downstream Solver Validation on Repaired Instances

We apply the instances revised by our framework to AFL (Zhang et al. 2025), an LLM-based solver that takes problem instances as input and is capable of handling a wide range of VRP constraints. As shown in Table 12, these revised instances can be directly used by AFL without any additional modification.

## C.5 Impact of Constraint Type Counts.

Table 13: EDR under diferent counts of constraint types.
<table><tr><td>EDR(%)</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td></tr><tr><td>LLM</td><td>87.27</td><td>79.42</td><td>72.42</td><td>57.42</td><td>53.97</td><td>22.30</td></tr><tr><td>Code</td><td>99.49</td><td>96.90</td><td>96.33</td><td>95.12</td><td>97.96</td><td>98.78</td></tr></table>

Table 13 shows that as the number of constraint types increases, the error detection rate of direct LLM inference degrades rapidly. In contrast, the self-guided framework remains highly robust to increasing constraint combination complexity, and consistently achieves over 95% error detection across all settings. These results demonstrate that the proposed framework efectively mitigates the negative impact of complex constraint combination on error detection.

## C.6 Evaluation on the Hard-Mode Dataset.

The controlled error-injection setting used in our main experiments (e.g., at most one error per node with fixed injection parameters) enables attribution-clean evaluation and avoids ambiguities caused by cascading errors. In real-world operational data, however, multiple faults may interact and produce more complex failure patterns. To better reflect such scenarios, we additionally construct a more challenging hard-mode dataset, where multiple errors are allowed to occur within the same node. As shown in Table 14, the self-guided framework remains consistently more robust than direct LLM diagnosis across all evaluated VRP variants, with particularly large improvements on variants involving more complex constraints.

Table 12: Downstream solver performance on repaired VRP instances.
<table><tr><td>AFL</td><td>CVRP</td><td>CVRPTW</td><td>CVRPB</td><td>OCVRPB</td><td>CVRPMBL</td><td>MDCVRPMBTW</td><td>MDOCVRPBTW</td><td>MDOCVRPMBLTW</td></tr><tr><td>Obj.</td><td>14.19</td><td>14.59</td><td>14.87</td><td>9.30</td><td>10.14</td><td>12.83</td><td>8.71</td><td>7.91</td></tr><tr><td>Time (s)</td><td>10.51</td><td>11.01</td><td>10.94</td><td>11.42</td><td>12.63</td><td>11.91</td><td>12.14</td><td>12.72</td></tr></table>

Table 14: Error Detection Rate (EDR, %) on the more challenging hard-mode dataset, where multiple errors may occur within the same node.
<table><tr><td>Method</td><td>CVRP</td><td>OCVRP</td><td>MDCVRP</td><td>CVRPB</td><td>CVRPMB</td><td>CVRPTW</td><td>CVRPL</td></tr><tr><td>LLM</td><td>80.50</td><td>83.00</td><td>90.00</td><td>71.50</td><td>69.00</td><td>60.50</td><td>58.00</td></tr><tr><td>Self-guided</td><td>97.50</td><td>95.50</td><td>96.00</td><td>89.50</td><td>91.00</td><td>89.00</td><td>87.00</td></tr></table>

## C.7 Error-Type-Specific False Positive Analysis.

While EDR summarizes the overall error-detection performance and incorporates localization accuracy, it does not fully characterize diagnostic reliability. In particular, a practical diagnostic system should avoid incorrectly flagging valid instances as erroneous. We therefore further evaluate errortype-specific false positives on valid instances. Table 15 reports the number of false-positive detections for each constraint type. The self-guided framework substantially suppresses spurious diagnoses, eliminating all observed false positives in these experiments. Together with the downstream evaluation of repaired instances in Appendix C.4, these results provide a more complete assessment of the reliability of the proposed framework.

## D Example Prompts

## D.1 Instance Analysis

## Prompt 1.1 Instance Analysis

We need to solve a VRP instance. I will provide you with the instance. Please analyze it carefully.

Step 1. Identify the constraints of the instance and list them clearly in numbered format 1), 2), 3), . . . within [ ]. For each constraint, write both the abbreviation (if any) and a short explanation (e.g., Capacity (C): vehicles have limited capacity). Do not include instance-specific details such as the exact number of nodes, vehicles, or capacity values.

Step 2. Write the standard problem type abbreviation (e.g., TSP, CVRP, CVRPL, VRPTW, OVRP, MDVRP), enclosed in quotation marks “ ”.

Step 3. List the sections that a correct and feasible instance should contain (e.g., DIMENSION, CAPACITY, if present).

## Output format:

1. [constraints]

2. "specific name"

3. [sections]

## Output 1.1 Instance Analysis

1. [

1) Capacity (C): each vehicle has a limited carrying capacity that cannot be exceeded.

2) Visit constraint (V): each customer must be visited exactly once.

3) Depot constraint (D): all routes must start and end at the depot.

2. "CVRP"

3. [DIMENSION, CAPACITY, NODE\_COORD\_SECTION, DEMAND\_SECTION, DEPOT\_SECTION]

## Prompt 1.2 Self-Validation

You are a VRP expert. I will give you:

1) The original .vrp file content

2) LLM’s answer (constraints, specific name, and required sections)

Your task is to judge whether the answer is correct. Specifically, check consistency among:

• Constraints implicitly or explicitly specified in the .vrp file

• The specific problem name vs. the .vrp file

• The listed constraints vs. the specific problem name

Do not include any extra elements not present in the instance.

If any contradiction exists, treat the .vrp file as the ground truth and mark the answer as incorrect. If everything is consistent, mark it as correct.

Table 15: False-positive analysis on valid instances. Each entry denotes the number of valid instances incorrectly identified as containing the corresponding error type. “–” indicates that the error type is not applicable to the corresponding VRP variant.
<table><tr><td>Error Type / Method</td><td>CVRP</td><td>OCVRP</td><td>MDCVRP</td><td>CVRPB</td><td>CVRPMB</td><td>CVRPTW</td><td>CVRPL</td></tr><tr><td>Format (LLM)</td><td>0</td><td>0</td><td>一</td><td>一</td><td>1</td><td>0</td><td>0</td></tr><tr><td>Format (Self-guided)</td><td>0</td><td>0</td><td>一</td><td>一</td><td>0</td><td>0</td><td>0</td></tr><tr><td>Capacity (LLM)</td><td>2</td><td>1</td><td>一</td><td>一</td><td>0</td><td>1</td><td>0</td></tr><tr><td>Capacity (Self-guided)</td><td>0</td><td>0</td><td>一</td><td>一</td><td>0</td><td>0</td><td>0</td></tr><tr><td>Backhaul (LLM)</td><td>一</td><td>一</td><td>11</td><td>12</td><td>一</td><td>一</td><td>一</td></tr><tr><td>Backhaul (Self-guided)</td><td>一</td><td>一</td><td>0</td><td>0</td><td>一</td><td>一</td><td>一</td></tr><tr><td>Distance Limit (LLM)</td><td>一</td><td></td><td>一</td><td>一</td><td>一</td><td>一</td><td>0</td></tr><tr><td>Distance Limit (Self-guided)</td><td>一</td><td></td><td>一</td><td>一</td><td>一</td><td>一</td><td>0</td></tr><tr><td>Time Window (LLM)</td><td>一</td><td></td><td></td><td>一</td><td>一</td><td>3</td><td>一</td></tr><tr><td>Time Window (Self-guided)</td><td>一</td><td></td><td></td><td></td><td>一</td><td>0</td><td></td></tr></table>

## Output format (exactly four lines):

1. [right: True/False]

2. [jud: explanation]

Here is the VRP file:

{vrp\_text}

Here is LLM’s answer:

{problem\_detail}

## Output 1.2 Self-Validation

1. [right: True]

2. [jud: The specific problem name is consistent with the VRP file. All listed constraints are explicitly or implicitly supported by the instance content, and the required sections correspond exactly to those defined in the file. No contradictions are found, and the instance is correctly identified.]

## Prompt 1.3 Revise

We need to solve a VRP instance. I will provide you with the instance. Please analyze it carefully.

Step 1. Identify the constraints of the instance and list them clearly in numbered format 1), 2), 3), . . . within [ ]. For each constraint, write both the abbreviation (if any) and a short explanation (e.g., Capacity (C): vehicles have limited capacity). Do not include instance-specific details such as the exact number of nodes, vehicles, or capacity values.

Step 2. Write the standard problem type abbreviation (e.g., TSP, CVRP, CVRPL, VRPTW, OVRP, MDVRP), enclosed in quotation marks “ ”.

Step 3. List the sections that a correct and feasible instance should contain (e.g., DIMENSION, CAPACITY, if present).

Revision instruction. Below is your previous answer, followed by identified issues. Please revise your answer strictly according to the rules above and correct all inconsistencies.

Here is your previous answer:

## {problem\_detail}

However, there were some issues identified: {judgment}

Output format:

1. [constraints]

2. "specific name"

3. [sections]

## Output 1.3 Revise

1. [

1) Capacity (C): each vehicle has a limited carrying capacity that cannot be exceeded.

2) Visit constraint (V): each customer must be visited exactly once.

3) Depot constraint (D): all routes must start and end at the depot.

]

2. "CVRP"

3. [DIMENSION, CAPACITY, NODE\_COORD\_SECTION, DEMAND\_SECTION, DEPOT\_SECTION]

## D.2 Insight Generation

Prompt 2.1 Reasoning Path and Function Prompt Gen eration

You are an expert software engineer and operations research (OR) researcher. Your task is to design a validator program (program.py) in clean, production-ready Python that checks an instance for: (A) format validity and (B) feasibility under the given problem type and constraints, and produces a structured report.

Context

• Problem type: {specific\_name}

• Constraints: {constraints} • The checks to perform (format validity and feasibility under constraints).

• The functions to implement, with exactly one function per check.

For each function, provide:

• its purpose and role,

• the enforced constraints (if applicable),

• precise input and output definitions,

• its contribution to the overall validation workflow.

Include a main function that sequentially executes all validation functions.

## Output 2.1 Reasoning Path and Function Prompt Generation (Partly)

(1) function\_name: def parse\_vrplib\_instance(vrp\_text)   
description and purpose: Reads and parses the raw VRPLIB   
text, extracting metadata (NAME, TYPE, DIMENSION,   
CAPACITY, EDGE\_WEIGHT\_TYPE) and required sections   
(NODE\_COORD\_SECTION, DEMAND\_SECTION,   
DEPOT\_SECTION), and returns a structured dictionary for   
validation.   
enforced\_constraints: None (data extraction only).   
input: vrp\_text (string containing the raw VRPLIB instance).   
(2) function\_name: def   
check\_required\_sections(parsed\_instance)   
description and purpose: Verifies that each required header   
and section appears exactly once and reports missing or   
duplicated entries.   
enforced\_constraints: VRPLIB format specification.   
input: parsed\_instance (dictionary returned by the parser).   
(3) function\_name: def   
check\_capacity\_constraint(parsed\_instance)   
description and purpose: Checks whether any customer   
demand exceeds the vehicle capacity and reports violating   
nodes.   
enforced\_constraints: Capacity constraint (C).   
input: parsed\_instance (dictionary).

## Prompt 2.2 Code Generation

• The raw .vrp file content is provided only for context.

• Do not hardcode any instance-specific details such as node count, vehicle number, or coordinates.

• All generated functions must be general-purpose and reusable.

If the generated function is a mainfunction, it must:

• Accept a command-line argument –path using the argparse library.

• Sequentially execute all validation functions.

• Print only the summary Total faults detected: [number].

If the generated function validates route feasibility, it must:

• Check feasibility from each customer to the depot (and back if required).

• Respect all declared constraints.

• Report each infeasible case as FAULT: [description].

Output Requirement.   
Return only valid Python code enclosed in a single python code block.

## Output 2.2 Code Generation (Partly)

```python
import argparse
2 import re
3 import sys
4 from collections import OrderedDict ,
defaultdict
5
6 REQUIRED_FIELDS = [" NAME ", " TYPE ",
DIMENSION ", " CAPACITY ",
EDGE_WEIGHT_TYPE "]
7 REQUIRED_SECTIONS = [" NODE_COORD_SECTION ",
DEMAND_SECTION ", " DEPOT_SECTION "]
8
9
10 def check_required_sections ( parsed_instance )
:
11 faults = []
12 missing_fields = []
13 duplicated_fields = []
14 missing_sections = []
15 duplicated_sections = []
16
17 meta = parsed_instance . get (’_meta ’, {})
18 header_counts = meta . get(’ _header_counts
```

Instance Content (Context Only).   
{vrp\_text}

’, {})   
19 section\_counts = meta . get (’   
\_section\_counts ’, {})   
20 for f in REQUIRED\_FIELDS :   
21 c = header\_counts .get (f, 0)   
22 if c == 0:   
23 msg = f" FAULT : Missing required   
field ’{f}’"   
24 faults . append ( msg )   
25 missing\_fields . append (f)   
26 elif c > 1:   
27 msg = f" FAULT : Duplicated field   
’{f}’: appears {c} times "   
28 faults . append ( msg )   
29 duplicated\_fields . append ((f, c))   
30 section\_keys = set ( parsed\_instance . keys   
())   
31 for s in REQUIRED\_SECTIONS :   
32 c = section\_counts . get (s , 0)   
33 section\_exists\_with\_content = s in   
parsed\_instance and bool (   
parsed\_instance [s])   
34 if c == 0 or not   
section\_exists\_with\_content :   
35 msg = f" FAULT : Missing required   
section ’{s}’"   
36 faults . append ( msg )   
37 missing\_sections . append (s)   
38 elif c > 1:   
39 msg = f" FAULT : Duplicated   
section ’{s} ’: appears {c}   
times "   
40 faults . append ( msg )   
41 duplicated\_sections . append ((s, c   
) )   
42 return {   
43 ’ missing\_fields ’: missing\_fields ,   
44 ’ duplicated\_fields ’:   
duplicated\_fields ,   
45 ’ missing\_sections ’: missing\_sections   
46 ’ duplicated\_sections ’:   
duplicated\_sections ,   
47 ’faults ’: faults ,   
48 ’count ’: len ( faults )   
49 }

Listing 1: Example of route feasibility check

## Prompt 2.3 Code Revision

We are working on a VRP problem instance to generate code that judges whether the instance is correct and feasible. Context. If previously generated code is provided, it is shown below and should be used as reference for correction.

## Problem Information.

• Constraints: {constraints}

• Specific problem name: {specific\_name}

• Sections that a correct and feasible instance should contain: {sections}

## Important Rules.

• The raw .vrp file content is provided only for context.

• Do not hardcode any instance-specific details (e.g., number of nodes, vehicle count, node coordinates).

• All generated functions must be general-purpose and reusable for any VRP instance.

Detected Issues and Revision Guidance.

• Issues identified in the previous code: {error\_msg}

• Revision suggestions: {jud}

Task. Please correct the previously generated code according to the issues above, without changing the number of functions, their names, or their function signatures.

Output Requirement. Return only valid Python code enclosed in a single python code block:

## D.3 Report Generation

## Prompt 3.1 Report Generation

You are an expert in VRP instance validation and repair. You are given the following information:

• A correct .vrp file used only as a format reference. Precise numeric values (e.g., distance limits) and optional sections are not treated as ground truth.

• A raw .vrp file that may contain faults.

• An initial list of detected faults, which may include duplicates or omissions.

Your task is to summarize all distinct faults and provide a complete repair.

Problem Context. This instance corresponds to {specific\_name}. You must verify whether all values and structures are valid under the expected specification.

Summary Tasks. Provide the following four parts:

1. Detailed fault descriptions, including where the fault occurs (section, node ID, or line).

2. Suggested corrections, one per fault.

3. Corrected raw .vrp instance text with minimal necessary changes.

## Critical Rules.

• Reason strictly from the raw .vrp content.

• Assume capacity and distance limit values themselves are not faulty.

• Do not remove or rewrite large portions unnecessarily.

• Keep section names, structure, and indexing intact unless invalid.

Input Information.

Format Reference Instance:

{right\_text}

Initial Faults Detected:

51 20   
DEPOT\_SECTION   
1   
-1   
EOF

{all\_fault}   
Raw Instance with Faults:   
{vrp\_text}

Output Format. Return only the following three parts, each enclosed in a separate code block:

• Detailed fault descriptions.

• Suggested corrections.

• Corrected raw .vrp instance text.

No extra commentary is allowed outside these blocks.

## Output 3.1 Report Generation (Partly) Detailed Fault Descriptions.

• Detailed fault descriptions. The node coordinate list is not explicitly preceded by the required NODE\_COORD\_SECTION header, leading to format ambiguity and non-compliance with the VRPLIB specification.

• The depot ID 1009 specified in DEPOT\_SECTION is outside the valid node index range 1..51, does not appear in NODE\_COORD\_SECTION, and is missing from DEMAND\_SECTION.

• The demands of nodes 27, 37, and 47 exceed the declared vehicle capacity, violating the capacity constraint of CVRP.

## Suggested corrections.

• Insert the line NODE\_COORD\_SECTION immediately before the node coordinate list to ensure unambiguous parsing.

• Replace the invalid depot ID 1009 with a valid node index within 1..DIMENSION (e.g., node 1), and ensure that the depot appears in both NODE\_COORD\_SECTION and DEMAND\_SECTION with zero demand.

• Modify the demands of nodes 27, 37, and 47 so that each individual demand does not exceed the vehicle capacity.

Corrected raw .vrp instance text.

NAME : corrected\_instance   
TYPE : CVRP   
DIMENSION : 51   
EDGE\_WEIGHT\_TYPE : EUC\_2D   
CAPACITY : 100   
NODE\_COORD\_SECTION   
1 12.34 56.78   
2 23.45 67.89   
51 98.76 54.32   
DEMAND\_SECTION   
1 0   
2 12   
27 40   
37 35   
47 30