![](images/5a6f2a336bafb71a828f77b97e9d809186bd4eff43738c6b00a2f62bdf659fae.jpg)  
Figure 1: Performance comparison of ZwZ2 and other advanced models.

# ZOOMING WITHOUT ZOOMING V2: FROM SEEING DETAILS TO CONNECTING EVIDENCE FOR GENERALIZABLE MULTIMODAL PERCEPTION

Lai Wei<sup>1,3,∗</sup>, Yatai Li<sup>1,2,∗</sup>, Liangbo He<sup>2,∗</sup>, Jun Lan<sup>2,‡</sup>, Renjie Gu<sup>5</sup>, Linhan Cao<sup>1,2</sup>, Junkai Lin<sup>1</sup>, Huijia Zhu<sup>2</sup>, Weiqiang Wang<sup>2</sup>, Linghe Kong<sup>1</sup>, Yue Wang<sup>3</sup>, Zhuosheng Zhang<sup>1</sup>, Weiran Huang<sup>1,4,†</sup>

<sup>1</sup>School of Computer Science, Shanghai Jiao Tong University <sup>2</sup>Ant Group <sup>3</sup>Zhongguancun Academy <sup>4</sup>Shanghai Innovation Institute <sup>5</sup>Fudan University

§ Code: inclusionAI/Zooming-without-Zooming-v2

Hugging Face: collections/inclusionAI/zooming-without-zooming-v2

## ABSTRACT

Multimodal perception forms the foundation of multimodal intelligence. In particular, spatial perception and multimodal long-context understanding are two pivotal tasks that underpin embodied AI and multimodal agents. However, synthesizing targeted, high-quality training data for these tasks remains difficult and costly. Our key insight is that these tasks share an underlying capability, which we call connecting evidence: identifying relevant visual facts, binding them to the correct entities, recovering their relations, and composing them into an answer. Crucially, targeted training data are far easier to synthesize for this capability than for these tasks themselves. To strengthen this capability, we propose EVIDENCECONNECT, an easy-to-implement data synthesis framework that uses zoomed views of related regions within a single image to generate questions requiring comparison, aggregation, or chaining of visual evidence. To ensure quality, a reference answer is retained only when independent VLMs agree on $\mathbf { i t } ,$ and region-ablation tests discard questions whose answer survives the removal of selected regions. We then

train base VLMs on the resulting data with RLVR, obtaining Zooming without Zooming v2 (ZwZ2). Although the synthesized data are not tailored to any downstream benchmark, ZwZ2 improves fine-grained and spatial perception, with further gains on embodied spatial reasoning, multimodal long-context understanding, multimodal agent tasks, and general visual reasoning (shown in Figure 1). These results establish EVIDENCECONNECT as an effective framework for supervising evidence connection and suggest that strengthening this shared capability leads to more generalizable multimodal perception.

## 1 INTRODUCTION

Multimodal perception is the foundation of multimodal intelligence: before a multimodal model can reason or act, it must determine what is present in a scene. Recent progress on this front, such as multimodal Chain-of-Thought (Zhang et al., 2023; Xu et al., 2024; Li & Ma, 2026) and Thinkingwith-Images (Zheng et al., 2025; Zhang et al., 2025c), has improved the ability of open-source vision-language models (VLMs) to perceive fine-grained visual content and use it in reasoning (Wei et al., 2026; 2025b; Li et al., 2026b).

However, these advances do not reliably extend to tasks in which critical visual cues are dispersed across objects, relationships, frames, or dialogue turns. Two such tasks, spatial perception and multimodal long-context understanding, are particularly important because they underpin embodied AI and multimodal agents, respectively. Spatial perception requires a model to recover how entities are arranged relative to one another and how several spatial relations compose (Gemini Robotics Team, 2025; Li et al., 2025a). Embodied AI relies on it to ground navigation, manipulation, and viewpoint-dependent decisions in object locations and spatial relations (Brohan et al., 2023; Kim et al., 2025; Yang et al., 2026). Multimodal long-context understanding requires a model to locate evidence across images and dialogue history, preserve entity identity, and bring that evidence to bear on the current question (Wu et al., 2025; Liu et al., 2024; Wang et al., 2025d; Song et al., 2024). Multimodal agents rely on it to retrieve relevant past observations, resolve references across interactions, and integrate earlier evidence into current decisions (He et al., 2024; Su et al., 2026; Li et al., 2025b).

Strengthening these capabilities, however, is constrained by the difficulty of building high-quality training data. For spatial perception, existing efforts mainly scale up region-grounded relational data (Chen et al., 2024a; Zhang et al., 2026a), but reliable labels require precise localization and relational grounding across multiple objects. Such labels are expensive to annotate by hand, while automated generators often hallucinate relations. For long-context understanding, existing multiimage and multi-turn instruction data (Jiang et al., 2024; Liu et al., 2024) concentrate on a few formats and task types, because diverse questions that remain answerable over long sequences are hard to construct automatically. Data for embodied spatial reasoning and multimodal agents are even harder to synthesize, since they additionally involve physical or interactive environments and multi-step trajectories. Consequently, neither these capabilities nor the applications built on them have a scalable and high-quality source of clean, verifiable supervision.

Our key insight is that these tasks build on a shared atomic capability of VLMs, which we call connecting evidence: identifying task-relevant visual facts, binding each fact to the correct entity, recovering the relations among them, and integrating the evidence into an answer. These tasks instantiate this capability differently: spatial perception connects objects within a scene through spatial relations, whereas long-context understanding, on which multimodal agents rely, connects facts across images and dialogue turns through entity identity and reference. Despite these differences in where the evidence resides and how it is linked, both tasks rely on the same underlying steps, so strengthening these steps can benefit both. Moreover, since the first step, identifying visual facts, is itself fine-grained perception, fine-grained perception benefits as well.

Crucially, training data that target this capability are far easier to construct and verify than data for the tasks themselves. To turn this insight into trainable supervision, we propose EVIDENCECON-NECT, an easy-to-implement, fully automated framework that synthesizes visual questions requiring the model to connect evidence across regions. Specifically, we select a group of related regions within a single image and synthesize a compositional question over them from privileged zoomed views, which make small details visible but are used only during data construction. Depending on the scene, we synthesize questions to compare, aggregate, or chain evidence across the regions. Answering such questions thus exercises the same steps as the target tasks: relating regions spatially, including from a given viewpoint, parallels spatial perception and embodied spatial reasoning, whereas locating scattered regions, binding their facts to the right references, and integrating them parallels long-context understanding and multimodal agents. To ensure data quality, we keep a question only if VLMs from different families agree on its answer, and we use region-ablation tests to discard questions whose answer survives the removal of selected regions. Finally, each verified question is converted into multiple-choice form for rule-based rewards and paired with the full image, where colored outlines typically mark the selected regions to avoid ambiguous references, so the model must still perceive and connect the evidence by itself.

To evaluate whether this supervision generalizes beyond the synthesized questions, we train Qwen3.5-4B, Qwen3.5-9B, and Qwen3.6-27B (Qwen Team, 2026a;b) with reinforcement learning with verifiable rewards (RLVR) (Lambert et al., 2024; DeepSeek-AI, 2025) using only our synthesized data, without supervised fine-tuning or benchmark-specific question formats. The resulting models, ZwZ2-4B/9B/27B, achieve substantial improvements in fine-grained and spatial perception as well as multimodal long-context understanding, with gains that further extend to embodied spatia reasoning, multimodal agent tasks, and general visual reasoning. For example, ZwZ2-4B improves over its base model from 37.43 to 45.26 (+7.83) on ViewSpatial-Bench (spatial reasoning) and from 60.08 to 70.08 (+10.00) on MileBench (long-context understanding). More broadly, its average over all 25 benchmarks increases from 49.41 to 54.79, and the corresponding averages improve from 51.63 to 56.59 for ZwZ2-9B and from 55.52 to 60.51 for ZwZ2-27B. Notably, although ZwZ2 is trained only on single images, it also improves on multi-image long-context benchmarks and multimodal agent tasks. These results demonstrate that connecting evidence is indeed a generalizable training target for multimodal perception.

To summarize, our key contributions are the following:

• We identify connecting evidence as an important atomic capability of VLMs, underlying finegrained perception, spatial perception, and multimodal long-context understanding.

• We develop EVIDENCECONNECT, a structured and audited synthesis pipeline for single-image, multi-region compositional questions, with privileged region views and independent checks for answer correctness and evidence dependence.

• Experiments demonstrate that RLVR on the resulting synthetic data improves a broad range of perception benchmarks and further generalizes to many real-world tasks, including embodied spatial reasoning, multimodal agent tasks, and general visual reasoning.

## 2 RELATED WORK

Spatial perception. Spatial perception underpins embodied tasks such as robotic manipulation and embodied question answering (Brohan et al., 2023; Kim et al., 2025; Gemini Robotics Team, 2025). Existing approaches use large-scale spatial supervision (Chen et al., 2024a), intermediate visual grounding (Zhang et al., 2026a), and progressive training (Li et al., 2026a). Benchmarks and associated training resources examine perspective-taking, multi-image inference, and multi-hop composition (Li et al., 2025a; Yang et al., 2026). MultihopSpatial additionally studies grounded spatial training and its transfer to embodied tasks (Lee et al., 2026). Euclid’s Gift (Lian et al., 2026) demonstrates transfer from geometric surrogate tasks. Similarly, we train on a surrogate task and evaluate its transfer, but our task is built from natural images and targets the evidence connection shared by spatial perception and long-context understanding, rather than spatial skills alone.

Long-context multimodal understanding. Multimodal long-context understanding requires locating, tracking, and integrating evidence across long sequences of images and dialogue turns. Existing efforts support these abilities through multi-image and multi-turn instruction tuning (Jiang et al., 2024; Huang et al., 2023; Liu et al., 2024) and dialogue-fragment retrieval (Bi et al., 2026), while benchmarks assess long-context integration and memory (Song et al., 2024; Wang et al., 2025d; Ren et al., 2026). Controlled studies nevertheless identify persistent difficulties in aggregating distributed visual information (Das et al., 2026). We target evidence connection as a shared component of these tasks, constructing single-image questions that require identifying, binding, and composing facts across regions. This setting makes the selected evidence easier to localize and audit, allowing us to investigate whether such supervision transfers to multimodal long-context understanding without training on long image sequences or dialogue histories.

![](images/98f4c6a918b8d9e73b7aa6b7afa9df928fccf62b1a37ff778e7147122ceae119.jpg)  
Figure 2: Overview of our multi-region question synthesis pipeline. We select related regions from a single image and construct a privileged crop view to generate compositional questions. The illustrative example connects a flight number on a boarding pass to its departure gate and then to a service icon (T218 → C6 → plug). Cross-model answer anchoring and drop-one region tests assess answer reliability and evidence dependence, followed by multiple-choice conversion and difficulty filtering. During training, the policy receives the full image with region outlines, while privileged crops are used only for data construction.

Multimodal synthetic data. Early resources provide annotated scene graphs and programmatic questions (Krishna et al., 2017; Hudson & Manning, 2019). Later methods automate instruction generation (Zhao et al., 2024; Zhang et al., 2025b), evolution (Luo et al., 2025), and synthesis for unsupervised post-training (Wei et al., 2025a). Zooming without Zooming (Wei et al., 2026) transfers single-region supervision to full images with cross-model answer checks. CRIT (Sung et al., 2026) and FM2DS (Abaskohi et al., 2024) synthesize cross-modal questions and filter unimodal shortcuts, while MM-CondChain (Shen et al., 2026) verifies conditional programs and ReasonVQA (Tran et al., 2025) composes external knowledge paths. HopChain (Wang et al., 2026) targets long-CoT robustness through instance-dependent chains with human-verified answers. Unlike these pipelines, EVIDENCECONNECT builds each question on a coherent group of regions rather than a single region or a sequential chain, so its questions cover aggregation and closed-set comparison in addition to dependency chains. Moreover, it automatically checks both the answer and its dependence on the selected regions, through cross-model agreement and region ablation, without human annotation.

## 3 BOOSTING MULTIMODAL PERCEPTION BY LEARNING TO CONNECT VISUAL EVIDENCE

As discussed in Section 1, fine-grained perception, spatial perception, and multimodal long-context understanding share a core capability that we call connecting evidence. This motivates training data whose questions structurally require this capability and end in concise answers suitable for verifiable rewards. To construct this supervision, we propose EVIDENCECONNECT, a data synthesis framework centered on coherent groups of image regions. Within this framework, high-resolution privileged crops render local details visible for question synthesis, while cross-model answer anchoring and region ablation check label reliability and evidence dependence. The overview of the data synthesis pipeline is shown in Figure 2.

## 3.1 MULTI-REGION QUESTION SYNTHESIS

Target question structure. We first define the target question, whose answer depends on connecting one coherent group of regions in an image. Formally, from a source image I, we obtain localized region proposals $R _ { I }$ and select one evidence group $G = \{ r _ { i _ { 1 } } , \ldots , r _ { i _ { k } } \} \subseteq R _ { I } , 3 \leq k \leq 7 .$ Restricting the evidence to a single image keeps training inexpensive, while several related regions of one image already require a model to locate, bind, and compose dispersed evidence. Importantly, the regions are selected to jointly support one question q, which assigns each of them a role, i.e., the fact it contributes: a value to be combined, a member of a closed candidate set, or a clue on which a later step depends.

Corresponding to these three kinds of roles, q connects the regions through one of three forms. Aggregation combines counts or other numeric facts from the regions, possibly through a short calculation. Comparison and matching treats the regions as the complete candidate set, so that an answer such as an outlier, an extremum, or a pairing can be identified only by considering all of them. Dependency chains and compositions build the answer in several steps, e.g., by using a clue to decide what to read next, fixing a viewpoint for interpreting the rest, or combining predicates from different regions. In all three forms, the answer is determined only when the assigned roles are taken together. These forms also tie our questions to the target tasks: spatial relations, including those in an agent’s or object’s own frame, parallel spatial perception and embodied spatial reasoning, while binding facts to regions referenced by outline colors and integrating them parallels how long-context understanding and multimodal agents handle facts tied to image or turn references. To synthesize such questions, we first locate candidate regions, then group those that support one of the three forms, and finally compose questions over each selected group.

Grounded evidence discovery. Since multi-region questions need detail-rich scenes, we start from high-resolution, visually complex images. For each image, a strong VLM judges whether it can support such questions and, if so, is prompted to propose concrete, clearly localizable regions, each with a bounding box and a short description, prioritizing regions that can be related to one another over isolated salient ones. We then discard proposals that are too small, too large, or heavily overlapping. As these model-generated descriptions may be wrong, they only guide grouping and question writing, whereas reference answers are later derived from the image itself.

Task-conditioned evidence grouping. An arbitrary collection of these proposals, however, does not necessarily support a valid compositional question. We therefore prompt a strong model with the image caption and region descriptions to propose several groups per image, each labeled with a task family (a specific question type such as anomaly detection) and a necessity mode (how removing a region should affect the answer). We then discard invalid or duplicated groups and closed sets of incomparable objects, and keep groups with distinct task families first. The task family only guides question writing, whereas each region’s concrete role is fixed when a question is written.

Privileged evidence layout. Since fine-grained details are difficult to inspect at full-image scale, we construct for each group a privileged evidence layout $V _ { G } ,$ which is accessible only during data construction. Following the layout-preserving design of Liu et al. (2025), $V _ { G }$ arranges nativeresolution crops of the selected regions, with small context margins, on a compact canvas that removes irrelevant background while preserving their relative horizontal and vertical order. In addition, each region is outlined in a distinct color that also marks it in the full image, allowing questions to refer to regions by color unambiguously. Finally, $V _ { G }$ serves as the sole visual input to the question generator, which confines its questions to evidence within the group, and it is later reused for verification (Section 3.2).

Question realization. The question generator, a strong VLM, then receives $V _ { G } ,$ , the region descriptions, and a few reasoning blueprints, i.e., templates of reasoning programs, and composes several candidate questions, each with a concise answer: a single value (e.g., a number, a short phrase, or the outline color of a selected region), a set, or an ordered pair. To make its use of evidence checkable, the generator also states each region’s contribution and the answer expected after removing it, together with a reasoning certificate, i.e., a short sequence of steps that links the regions to the answer. Deterministic rules then reject candidates whose certificate leaves a region unused or contains steps that do not lead to the answer, whose own predicted answer survives the removal of a region, or that use subjective wording, leak the answer through an outline color, or contain inconsistent arithmetic. Implementation details and prompt templates appear in Appendices A and B.

## 3.2 TWO-STAGE VERIFICATION

The checks above concern only the form of a candidate. A compositional question, however, is useful supervision only if its reference answer is reliable and the intended evidence matters. We assess these properties separately, since a plausible reasoning structure establishes neither.

Cross-model answer anchoring. To obtain a more reliable reference answer, two VLMs from different model families answer q from $V _ { G }$ several times, without seeing the region descriptions or the proposed answer. A candidate is kept only if a clear majority of each model’s responses supports the same answer and this answer becomes the reference $a ^ { \star }$

Adaptive drop-one testing. Agreement on $a ^ { \star }$ , however, does not show that the answer actually depends on the selected evidence. Since every question is constructed to require all of its regions (Section 3.1), an answer that survives the removal of a region indicates a defect, such as a decorative region or a shortcut that bypasses the intended reasoning. To detect such defects, we mask each $r _ { i } \in G$ in the privileged layout, $V _ { G } ^ { - i } = \mathcal { M } _ { i } ( V _ { G } )$ , where $\mathcal { M } _ { i }$ replaces the crop of $r _ { i }$ with a gray placeholder. A verifier from one anchoring family then answers $q$ from this view, reporting insufficient information if the masked evidence is needed, and either such a report or a changed answer confirms that the question depends on $r _ { i }$ . If instead $a ^ { \star }$ is still returned, or the response is unusable, we query the verifier again and mark $r _ { i }$ as empirically redundant only if most queries recover $a ^ { \star }$ Note that a single redundant region, however, may reflect noise rather than a defect, because part of a masked region can remain visible in an adjacent crop through overlapping context margins, and individual responses also vary. We therefore reject a question only when more than one region is redundant, which also improves synthesis efficiency, since a stricter rule would discard many potentially valid questions merely because one region was judged redundant. The implementation details are shown in Appendix A.

## 3.3 DATA FINALIZATION

From privileged crops to the policy view. Since the policy, i.e., the model under training, never sees $V _ { G } ,$ , each verified candidate is paired with a policy view $V _ { I }$ , by default the source image with each selected region outlined in its assigned color. To avoid ambiguous references in the full image, these outlines mark which instances a question refers to, without revealing the answer-bearing content that the policy must still perceive and compose. For aggregate questions, whose answers do not depend on the outlines, we also try to replace marker references with natural descriptions, so that the policy must locate the evidence itself. This unmarked version replaces the marked one only if independent audits by both anchoring families, without access to $a ^ { \star }$ , confirm unique references, an unchanged multi-region task grounded in the intended regions, and the answer $a ^ { \star }$

Multiple-choice conversion and difficulty filtering. To support rule-based rewards in RLVR, we then turn each question into a four-option multiple-choice question, adding three distractors in the answer format of $a ^ { \star }$ so that the format does not reveal the answer. Each released example is thus $x = ( V _ { I } , q _ { \mathrm { r e l } } , O )$ with label $y \in \{ \mathrm { A , B , C , D } \}$ , where $q _ { \mathrm { r e l } }$ is the verified question stem, rewritten without marker references when the unmarked view is used, O the shuffled option set, and $y$ the option of $a ^ { \star }$ . Finally, since overly easy examples provide little learning signal, a lightweight VLM answers each example several times, and we discard examples that it always answers correctly.

## 4 EXPERIMENTS

This section evaluates whether learning to connect evidence within a single image yields improvements that generalize across multimodal perception tasks.

## 4.1 EXPERIMENTAL SETTINGS

Benchmarks. We evaluate on 25 benchmarks in six domains (Table 1). (i) Fine-grained perception: VStar (Wu & Xie, 2024), HR-Bench 4K/8K (Wang et al., 2025c), MME-RealWorld (EN/CN) (Zhang et al., 2025d), ZoomBench (Wei et al., 2026), Chartography (Garre et al., 2026), and PerceptionBench (Lin et al., 2026). (ii) Spatial perception: TreeBench (Wang et al., 2025a), O3-Bench (Li et al., 2026c), CountQA (Tamarapalli et al., 2025), MIMIC (Das et al., 2026), MMSI-Bench (Yang et al., 2026), and SPAR-Bench (Zhang et al., 2025a). (iii) Multimodal long-context understanding: MMLongBench (Wang et al., 2025d), MileBench (Song et al., 2024), and Mem-Lens (Ren et al., 2026). (iv) Embodied spatial reasoning: ERQA (Gemini Robotics Team, 2025), ViewSpatial-Bench (Li et al., 2025a), and MindCube-Tiny (Wang et al., 2025b). (v) Multimodal agents: AgentVista (Su et al., 2026), together with ScreenSpot-Pro (Li et al., 2025b) and UI-Vision (Nayak et al., 2025) for GUI grounding. (vi) General visual reasoning: MMStar (Chen et al., 2024b) and BabyVision (Chen et al., 2026). We report accuracy (%) on all benchmarks.

Table 1: Main results on 25 benchmarks (accuracy, %). Domain averages and Overall are unweighted means within each domain and over all 25 benchmarks, respectively.
<table><tr><td rowspan="2">Domains</td><td rowspan="2">Benchmarks</td><td colspan="3">Base models</td><td colspan="3">ZwZ2</td><td colspan="3">Advanced models</td></tr><tr><td>4B</td><td>9B</td><td>27B</td><td>4B</td><td>9B</td><td>27B</td><td>GPT</td><td>GLM</td><td>DeepSeek</td></tr><tr><td rowspan="9">Fine-grained perception</td><td>VStar</td><td>82.20</td><td>83.77</td><td>87.96</td><td>89.00</td><td>90.58</td><td>91.62</td><td>75.92</td><td>83.77</td><td>78.53</td></tr><tr><td>HR-4K</td><td>82.37</td><td>84.88</td><td>84.25</td><td>88.62</td><td>89.75</td><td>89.88</td><td>83.87</td><td>84.25</td><td>79.13</td></tr><tr><td>HR-8K</td><td>78.37</td><td>82.50</td><td>80.00</td><td>83.63</td><td>87.62</td><td>88.00</td><td>79.13</td><td>80.63</td><td>75.00</td></tr><tr><td>MME-en</td><td>70.60</td><td>71.00</td><td>72.62</td><td>75.48</td><td>75.33</td><td>77.41</td><td>58.50</td><td>67.88</td><td>50.75</td></tr><tr><td>MME-cn</td><td>69.33</td><td>69.53</td><td>68.68</td><td>73.89</td><td>73.45</td><td>74.97</td><td>56.90</td><td>68.31</td><td>50.04</td></tr><tr><td>ZoomBench</td><td>50.41</td><td>49.57</td><td>53.61</td><td>58.22</td><td>59.53</td><td>61.78</td><td>54.44</td><td>56.09</td><td>49.94</td></tr><tr><td>Chartography</td><td>5.00</td><td>7.00</td><td>8.00</td><td>14.00</td><td>13.00</td><td>12.00</td><td>17.00</td><td>9.00</td><td>10.00</td></tr><tr><td>PerceptionBench</td><td>35.53</td><td>38.70</td><td>41.57</td><td>37.80</td><td>38.89</td><td>44.93</td><td>48.77</td><td>44.78</td><td>44.00</td></tr><tr><td>Avg.</td><td>59.23</td><td>60.87</td><td>62.09</td><td>65.08</td><td>66.02</td><td>67.57</td><td>59.32</td><td>61.84</td><td>54.67</td></tr><tr><td rowspan="7">Spatial perception</td><td>TreeBench</td><td>50.37</td><td>46.67</td><td>49.63</td><td>51.36</td><td>51.60</td><td>56.05</td><td>52.35</td><td>57.53</td><td>61.98</td></tr><tr><td>O3-Bench</td><td>53.91</td><td>59.13</td><td>65.80</td><td>63.77</td><td>67.83</td><td>72.46</td><td>66.38</td><td>62.32</td><td>73.62</td></tr><tr><td>CountQA</td><td>36.91</td><td>37.63</td><td>46.73</td><td>47.19</td><td>48.43</td><td>54.32</td><td>51.83</td><td>49.35</td><td>64.07</td></tr><tr><td>MIMIC</td><td>55.01</td><td>59.39</td><td>55.95</td><td>66.10</td><td>64.98</td><td>65.16</td><td>57.43</td><td>73.46</td><td>68.55</td></tr><tr><td>MMSI-Bench</td><td>30.70</td><td>32.70</td><td>33.50</td><td>32.30</td><td>33.90</td><td>37.10</td><td>32.80</td><td>37.20</td><td>34.60</td></tr><tr><td>SPAR-Bench</td><td>41.47</td><td>43.53</td><td>46.68</td><td>45.69</td><td>47.23</td><td>50.78</td><td>43.92</td><td>52.13</td><td>40.66</td></tr><tr><td>Avg.</td><td>44.73</td><td>46.51</td><td>49.72</td><td>51.07</td><td>52.33</td><td>55.98</td><td>50.79</td><td>55.33</td><td>57.25</td></tr><tr><td rowspan="4">Long-context understanding</td><td>MMLongBench*</td><td>63.50</td><td>69.89</td><td>73.41</td><td>68.34</td><td>73.45</td><td>76.73</td><td>67.79</td><td>83.15</td><td>82.20</td></tr><tr><td>MileBench</td><td>60.08</td><td>68.96</td><td>71.97</td><td>70.08</td><td>71.46</td><td>73.94</td><td>63.34</td><td>71.33</td><td>65.79</td></tr><tr><td>MemLens</td><td>40.62</td><td>45.82</td><td>52.22</td><td>42.59</td><td>47.34</td><td>55.64</td><td>37.20</td><td>58.40</td><td>66.57</td></tr><tr><td>Avg.</td><td>54.73</td><td>61.56</td><td>65.87</td><td>60.34</td><td>64.08</td><td>68.77</td><td>56.11</td><td>70.96</td><td>71.52</td></tr><tr><td rowspan="4">Embodied spatial reasoning</td><td>ERQA</td><td>45.75</td><td>45.75</td><td>51.25</td><td>52.00</td><td>55.50</td><td>61.75</td><td>54.00</td><td>54.50</td><td>59.50</td></tr><tr><td>ViewSpatial-Bench</td><td>37.43</td><td>42.65</td><td>48.49</td><td>45.26</td><td>45.27</td><td>50.63</td><td>40.18</td><td>46.57</td><td>56.29</td></tr><tr><td>MindCube-Tiny</td><td>47.62</td><td>40.76</td><td>50.76</td><td>53.05</td><td>52.57</td><td>54.76</td><td>57.62</td><td>72.00</td><td>42.00</td></tr><tr><td>Avg.</td><td>43.60</td><td>43.05</td><td>50.17</td><td>50.10</td><td>51.11</td><td>55.71</td><td>50.60</td><td>57.69</td><td>52.60</td></tr><tr><td rowspan="4">Multimodal agent</td><td>AgentVista</td><td>12.44</td><td>8.61</td><td>12.92</td><td>17.22</td><td>14.83</td><td>18.18</td><td>21.53</td><td>22.01</td><td>9.09</td></tr><tr><td>ScreenSpot-Pro</td><td>60.09</td><td>63.88</td><td>70.21</td><td>61.73</td><td>66.98</td><td>70.59</td><td>63.25</td><td>40.48</td><td>54.40</td></tr><tr><td>UI-Vision</td><td>35.83</td><td>42.48</td><td>66.05</td><td>37.67</td><td>45.48</td><td>66.66</td><td>59.43</td><td>38.48</td><td>43.34</td></tr><tr><td>Avg.</td><td>36.12</td><td>38.32</td><td>49.73</td><td>38.87</td><td>42.43</td><td>51.81</td><td>48.07</td><td>33.66</td><td>35.61</td></tr><tr><td rowspan="3">General visual reasoning</td><td>MMStar</td><td>74.13</td><td>77.27</td><td>75.93</td><td>76.27</td><td>78.27</td><td>79.07</td><td>74.73</td><td>80.20</td><td>80.73</td></tr><tr><td>BabyVision</td><td>15.46</td><td>18.56</td><td>19.85</td><td>18.56</td><td>21.39</td><td>28.35</td><td>32.22</td><td>37.11</td><td>45.88</td></tr><tr><td>Avg.</td><td>44.80</td><td>47.92</td><td>47.89</td><td>47.42</td><td>49.83</td><td>53.71</td><td>53.48</td><td>58.66</td><td>63.31</td></tr><tr><td>Overall</td><td></td><td>49.41</td><td>51.63</td><td>55.52</td><td>54.79</td><td>56.59</td><td>60.51</td><td>54.02</td><td>57.24</td><td>55.47</td></tr></table>

Notes. Base models are Qwen3.5-4B/9B and Qwen3.6-27B. GPT: GPT-5.6-Luna; GLM: GLM-5.3-Flash. DeepSeek: DeepSeek-V4.1-Flash. All models are evaluated in low-thinking or non-thinking reasoning modes. The input resolution of the three advanced models is constrained by platform limitations. <sup>∗</sup>: extracted MMLongBench visual-core subset, including V-RAG, Image MM-NIAH, and VH-Multi (25 non-Counting tasks in total).

Training. We train Qwen3.5-4B/9B (Qwen Team, 2026a) and Qwen3.6-27B (Qwen Team, 2026b) in non-thinking mode with GRPO (Shao et al., 2024) on the 20.4K examples synthesized by EVI-DENCECONNECT (Appendix C), without supervised fine-tuning. All three scales share the same data and training configuration (Appendix D), and we refer to the resulting models as ZwZ2- 4B/9B/27B.

Baselines. We compare with two groups of baselines. As references for absolute performance, Table 1 includes three advanced models: GPT-5.6-Luna, GLM-5.3-Flash, and DeepSeek-V4.1-Flash. To compare EVIDENCECONNECT with existing synthetic training data, we further train Qwen3.5- 4B with the same GRPO recipe on the datasets of ZwZ (Wei et al., 2026), TVG (Zhang et al., 2026a), F<sup>2</sup>RVLM (Bi et al., 2026), CRIT (Sung et al., 2026), and MMHops (Zhang et al., 2026b) (Table 2). For CRIT, whose 84,199 training samples contain 269,467 QA pairs from GQA, ActivityNet Captions, and SPIQA in multi-image, interleaved image–text, and multi-turn formats, we uniformly sample about 30K QA pairs across the three domains and train on them without CoT supervision or additional annotations.

Table 2: Training Qwen3.5-4B on different synthetic datasets (accuracy, %). Parentheses give the number of training examples. HR denotes HR-Bench; Chart and O3 denote Chartography and O3-Bench. MMSI, MindCube, and ViewSpatial denote MMSI-Bench, MindCube-Tiny, and ViewSpatial-Bench, respectively.
<table><tr><td rowspan="2">Models</td><td colspan="3">Fine-grained perception</td><td colspan="3">Spatial perception</td><td colspan="3">Embodied spatial reasoning</td><td colspan="2">Long-context understanding</td><td rowspan="2">Avg.</td></tr><tr><td>HR-4K HR-8K</td><td></td><td>Chart</td><td>O3 CountQA MMSI MindCube ERQA ViewSpatial MileBench</td><td></td><td></td><td></td><td></td><td></td><td></td><td>MemLens</td></tr><tr><td>Base</td><td>82.37</td><td>78.37</td><td>5.00</td><td>53.91</td><td>36.91</td><td>30.70</td><td>47.62</td><td>45.75</td><td>37.43</td><td>60.08</td><td>40.62</td><td>47.16</td></tr><tr><td>MMHops (31K)</td><td>77.75</td><td>73.12</td><td>3.00</td><td>45.80</td><td>30.82</td><td>30.50</td><td>42.10</td><td>43.75</td><td>40.90</td><td>66.29</td><td>15.97</td><td>42.73</td></tr><tr><td>TVG (19.4K)</td><td>78.25</td><td>73.62</td><td>13.00</td><td>53.04</td><td>33.97</td><td>29.50</td><td>28.86</td><td>46.00</td><td>42.16</td><td>64.56</td><td>37.29</td><td>45.48</td></tr><tr><td>F2RVLM (37K)</td><td>80.13</td><td>77.62</td><td>10.00</td><td>54.20</td><td>32.72</td><td>31.90</td><td>34.38</td><td>44.75</td><td>40.56</td><td>64.58</td><td>39.80</td><td>46.42</td></tr><tr><td>CRIT (30K)</td><td>82.25</td><td>77.88</td><td>6.00</td><td>45.22</td><td>30.56</td><td>29.90</td><td>43.43</td><td>45.00</td><td>43.98</td><td>68.13</td><td>40.40</td><td>46.61</td></tr><tr><td>ZwZ (73K)</td><td>87.00</td><td>82.75</td><td>7.00</td><td>60.87</td><td>43.32</td><td>31.33</td><td>33.81</td><td>46.00</td><td>43.75</td><td>70.21</td><td>42.46</td><td>49.86</td></tr><tr><td>Ours (20.4K)</td><td>88.62</td><td>83.63</td><td>14.00</td><td>63.77</td><td>47.19</td><td>32.30</td><td>53.05</td><td>52.00</td><td>45.26</td><td>70.08</td><td>42.59</td><td>53.86</td></tr></table>

## 4.2 MAIN RESULTS

Table 1 shows that ZwZ2 improves over its base model on all 25 benchmarks at every scale, raising the overall average by 5.39, 4.96, and 4.99 points at 4B, 9B, and 27B, respectively. In particular, fine-grained and spatial perception improve by 5.2–6.3 points on average, with gains of up to 11.1 points on MIMIC and 10.8 points on CountQA, and embodied spatial reasoning improves by 5.5–8.1 points, including 10.5 points on ERQA at 27B. Qualitative examples are provided in Appendix F.

Notably, although ZwZ2 is trained only on single images, its gains carry over to multi-image and interactive settings. Long-context understanding improves at every scale, most markedly at 4B, where the domain average rises by 5.6 points, including a gain of 10.0 points on MileBench. Likewise, the multimodal agent benchmark AgentVista improves by 4.8–6.2 points, and GUI grounding also improves consistently. These results support our premise that connecting evidence within an image transfers to tasks whose evidence spans images, dialogue turns, or interaction steps.

ZwZ2 remains competitive with frontier models. ZwZ2-27B achieves the highest overall accuracy (60.51%), compared with 54.02%, 57.24%, and 55.47% for GPT-5.6-Luna, GLM-5.3-Flash, and DeepSeek-V4.1-Flash, respectively, and leads the fine-grained perception domain.

## 4.3 COMPARISON WITH EXISTING TRAINING DATA

Table 2 compares EVIDENCECONNECT with existing synthetic datasets on the same 4B base model. These datasets target single-region fine-grained perception (ZwZ), grounded counting and spatial reasoning (TVG), dialogue-fragment retrieval (F<sup>2</sup>RVLM), cross-modal multi-hop QA over interleaved contexts (CRIT), and knowledge-based multi-hop QA (MMHops). Training on TVG, F<sup>2</sup>RVLM, CRIT, or MMHops improves MileBench but degrades HR-Bench, CountQA, MindCube-Tiny, and MemLens, and all four datasets lower the average accuracy below that of the base model. In contrast, EVIDENCECONNECT improves every benchmark over the base model and outperforms these four datasets on all 11 benchmarks, with a comparable amount of data (20.4K examples versus 19.4K–37K).

ZwZ-73K is the strongest baseline. Since it also synthesizes questions from zoomed-in views but grounds each question in a single region, the comparison with ZwZ serves as an approximate ablation of connecting multiple regions. Although ZwZ uses 3.6× as many examples (73K versus 20.4K), EVIDENCECONNECT raises the average accuracy from 49.86 to 53.86 and surpasses ZwZ on 10 of the 11 benchmarks, trailing it only on MileBench by 0.13 points. The largest margin appears on MindCube-Tiny, where ZwZ reduces accuracy by 13.8 points whereas EVIDENCECON-NECT improves it by 5.4 points. This contrast is consistent with our premise that embodied spatial reasoning relies on connecting evidence across regions rather than on inspecting a single region.

Table 3: Data-synthesis ablations on Qwen3.5-4B (accuracy, %). Each trained variant uses 7K examples. The 7K reference model is shared by both comparisons and is reported only once; it uses privileged views and drop-one verification. Avg. is the unweighted mean over 11 benchmarks.
<table><tr><td rowspan="2">Settings</td><td colspan="3">Fine-grained perception</td><td colspan="3">Spatial perception</td><td colspan="3">Embodied spatial reasoning</td><td colspan="2">Long-context understanding</td><td rowspan="2"> $\mathbf { A v } \mathbf { g } .$ </td></tr><tr><td>HR-4K HR-8K</td><td></td><td>Chart</td><td></td><td></td><td></td><td></td><td></td><td>O3 CountQA MMSI MindCube ERQA ViewSpatial MileBench</td><td></td><td>MemLens</td></tr><tr><td>Base</td><td>82.37</td><td>78.37</td><td>5.00</td><td>53.91</td><td>36.91</td><td>30.70</td><td>47.62</td><td>45.75</td><td>37.43</td><td>60.08</td><td>40.62</td><td>47.16</td></tr><tr><td>Without privileged views 85.12</td><td></td><td>82.50</td><td>13.00</td><td>53.62</td><td>40.51</td><td>33.00</td><td>50.19</td><td>52.25</td><td>43.87</td><td>70.22</td><td>40.87</td><td>51.38</td></tr><tr><td>Without drop-one</td><td>85.75</td><td>79.75</td><td>8.00</td><td>58.26</td><td>41.30</td><td>34.10</td><td>47.14</td><td>51.50</td><td>43.84</td><td>70.16</td><td>41.48</td><td>51.03</td></tr><tr><td>Ours (7K)</td><td>86.62</td><td>82.87</td><td>9.00</td><td>62.90</td><td>45.09</td><td>34.60</td><td>52.29</td><td>47.50</td><td>45.68</td><td>70.47</td><td>42.14</td><td>52.65</td></tr></table>

## 4.4 ABLATION STUDIES

Ablation study on drop-one verification. To evaluate adaptive drop-one verification during data synthesis, we compare Qwen3.5-4B trained on 7K examples constructed with or without this veri fication step (Table 3). As described in Section 3.2, this step masks one selected region at a time in the privileged layout and tests whether the reference answer remains recoverable. When the verifier returns the reference answer or an unusable response, additional queries are used; a region is marked as empirically redundant only if most queries recover the reference answer. A question is rejected only when more than one region is empirically redundant, allowing for noise from overlapping crop margins and variable verifier responses. With drop-one verification, average accuracy rises from 51.03 to 52.65 (+1.63 points), with improvements on 10 of the 11 benchmarks. The largest gains are on MindCube-Tiny (+5.15), O3-Bench (+4.64), and CountQA (+3.79), while ERQA decreases by 4.00 points. These results support the usefulness of this synthesis-time verification step for downstream performance, although its gains are not uniform across tasks.

Ablation study on privileged views. To examine the role of privileged views, we train Qwen3.5- 4B on 7K examples synthesized from privileged views or original images, with all other settings fixed (Table 3). Synthesis from privileged views achieves a higher average accuracy than synthesis from original images (52.65 versus 51.38), with higher scores on 9 of the 11 benchmarks. Originalimage synthesis performs better on Chartography and ERQA, by 4.00 and 4.75 points, respectively. Thus, privileged-view synthesis improves average accuracy by 1.27 points in this 7K comparison, although its gains are not uniform across benchmarks.

## 4.5 ANALYSIS

![](images/f807be1e6b2510fcc66715fef4ef346bef7d266c669c031f0642bda5956a9cee.jpg)  
Figure 3: Qwen3-VL generalization.

![](images/009dcafc3e24e8a562c5653af1d76d4f54d4ae3c278917e614c4575373a78a38.jpg)  
Figure 4: Effect of training data scale.

Generalization to Qwen3-VL. To examine whether the gains extend beyond the base models used in the experiments, we evaluate Qwen3- VL-2B and Qwen3-VL-4B before and after training on the same 11 benchmarks in Table 2. Figure 3 reports average scores of 36.82 and 41.63 for Qwen3-VL-2B, and 41.70 and 44.61 for Qwen3-VL-4B, corresponding to gains of 4.81 and 2.91 points, respectively. These results suggest that the benefits of supervision extend to additional base models at both scales.

Effect of training data scale. Figure 4 compares Qwen3.5-4B at different training data scales on the same 11 benchmarks. Average accuracy over the 11 benchmarks in Table 2 increases from 47.16 for the base model to 52.65 with 7K training examples and 53.86 with the full 20.4K training set. The full set yields a further improvement of 1.21 points over 7K examples, while the 7K setting already improves over the base model by 5.49 points.

Context length. Figure 5 breaks down MemLens and the MMLongBench visual core by context length. ZwZ2 outperforms or matches its base model at every length and scale, and on MemLens its gains persist up to 256K tokens. The gains, however, narrow at the longest contexts. At 128K on the MMLongBench visual core, for example, ZwZ2-4B and ZwZ2-9B merely match their base models.

This trend is consistent with our training data, in which each example contains a single image and thus a short context. Training therefore strengthens evidence connection but not retrieval from very long inputs, which likely becomes the main difficulty at the longest lengths.

Attention to evidence. To examine whether ZwZ2 focuses more on task-relevant visual evidence, we follow Wei et al. (2026) and evaluate ZwZ2-4B/9B against their base models on 405 TreeBench (Wang et al., 2025a) examples equipped with annotated evidence bounding boxes. For each full-attention layer, we compute the Intersection over Union (IoU) between the ground-truth evidence tokens E and the |E| visual tokens with the highest attention weights (Appendix E). As shown in Figure 6, ZwZ2 achieves a higher mean IoU across 7 out of 8 layers at the 4B scale and 6 out of 8 layers at the 9B scale. This indicates that the high-attention tokens of ZwZ2 align more closely with the ground-truth evidence in most layers at both scales.

![](images/8cb8f58ec7a9c4452c0cb45ef2ff1dc6e8600bb8463fcfc769a6735e3b3aadca.jpg)

![](images/66f4f48f0cc2f7da7b09f46367589dd8d6326ad0e58e673428d8421f8379e708.jpg)  
Figure 5: Context-length performance on MemLens (left) and the MMLongBench visual core (right). Blue, orange, and green denote 4B, 9B, and 27B. Dashed and solid lines show the base models and ZwZ2, respectively.

![](images/552f2c60bd00178938af76a4f9a178411ff5a9a4c9365023a0bc447dc4c08b32.jpg)  
Figure 6: Attention alignment on TreeBench. Layer-wise change in mean top-k IoU (ZwZ2 minus Qwen3.5, percentage points).

## 5 CONCLUSION

We presented EVIDENCECONNECT, a fully automated framework that synthesizes multi-region visual questions and answers, requiring a model to connect evidence within an image. Trained with RLVR on these data alone, ZwZ2 improves over its base models on all 25 benchmarks at 4B, 9B, and 27B. Notably, although training uses only single images, the gains extend beyond fine-grained and spatial perception to embodied spatial reasoning, multimodal long-context understanding, and multimodal agent tasks. These results suggest that connecting evidence is a transferable training target, and thus tasks whose training data are costly to build can benefit from data that are far easie to construct and verify.

## ACKNOWLEDGEMENT

Weiran Huang is supported by National Natural Science Foundation of China (No. 62406192), Shanghai Municipal Special Program for Basic Research on General AI Foundation Models (Grant No. 2025SHZDZX025G03), Tencent WeChat Rhino-Bird Focused Research Program, and Kuaishou Technology. This work is supported by Ant Group Research Intern Program. This work is also supported by the Zhongguancun Academy (Grant No.s C20250203). We also sincerely thank Xiangyu Zou (Z.ai) for assistance with a portion of the model evaluations.

## REFERENCES

Amirhossein Abaskohi, Spandana Gella, Giuseppe Carenini, and Issam H Laradji. Fm2ds: Few-shot multimodal multihop data synthesis with knowledge distillation for question answering. arXiv preprint arXiv:2412.07030, 2024.

Hanbo Bi, Zhiqiang Yuan, Zexi Jia, Jiapei Zhang, Chongyang Li, Peixiang Luo, Ying Deng, Xiaoyue Duan, and Jinchao Zhang. F2RVLM: Boosting fine-grained fragment retrieval for multi-modal

long-form dialogue with vision language model. In Proceedings of the AAAI Conference on Artificial Intelligence, 2026.

Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, et al. Rt-2: Vision-language-action models transfer web knowledge to robotic control. arXiv preprint arXiv:2307.15818, 2023.

Boyuan Chen, Zhuo Xu, Sean Kirmani, Brain Ichter, Dorsa Sadigh, Leonidas Guibas, and Fei Xia. Spatialvlm: Endowing vision-language models with spatial reasoning capabilities. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pp. 14455–14465, 2024a.

Liang Chen, Weichu Xie, Yiyan Liang, Hongfeng He, Hans Zhao, Zhibo Yang, Zhiqi Huang, Haoning Wu, Haoyu Lu, Yiping Bao, et al. Babyvision: Visual reasoning beyond language. arXiv preprint arXiv:2601.06521, 2026.

Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, et al. Are we on the right way for evaluating large vision-language models? Advances in Neural Information Processing Systems, 37:27056–27087, 2024b.

Anurag Das, Adrian Bulat, Alberto Baldrati, Ioannis Maniadis Metaxas, Bernt Schiele, Georgios Tzimiropoulos, and Brais Martinez. More images, more problems? a controlled analysis of vlm failure modes. In Findings of the Association for Computational Linguistics: ACL 2026, pp. 7423–7442, 2026.

DeepSeek-AI. Deepseek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Suhaas Garre, Chris Mutty, Sushant Mehta, and Edwin Chen. Chartography: A benchmark for professional chart understanding. arXiv preprint arXiv:2608.10677, 2026.

Gemini Robotics Team. Gemini robotics: Bringing ai into the physical world. arXiv preprint arXiv:2503.20020, 2025.

Hongliang He, Wenlin Yao, Kaixin Ma, Wenhao Yu, Yong Dai, Hongming Zhang, Zhenzhong Lan, and Dong Yu. Webvoyager: Building an end-to-end web agent with large multimodal models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 6864–6890, 2024.

Yupan Huang, Zaiqiao Meng, Fangyu Liu, Yixuan Su, Nigel Collier, and Yutong Lu. Sparkles: Unlocking chats across multiple images for multimodal instruction-following models. arXiv preprint arXiv:2308.16463, 2023.

Drew A. Hudson and Christopher D. Manning. GQA: A new dataset for real-world visual reasoning and compositional question answering. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 6700–6709, 2019.

Dongfu Jiang, Xuan He, Huaye Zeng, Cong Wei, Max Ku, Qian Liu, and Wenhu Chen. MANTIS: Interleaved multi-image instruction tuning. arXiv preprint arXiv:2405.01483, 2024.

Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan P. Foster, Pannag R. Sanketi, Quan Vuong, Thomas Kollar, Benjamin Burchfiel, Russ Tedrake, Dorsa Sadigh, Sergey Levine, Percy Liang, and Chelsea Finn. Open-VLA: An open-source vision-language-action model. In Proceedings of the 8th Conference on Robot Learning, volume 270 of Proceedings of Machine Learning Research, pp. 2679–2713. PMLR, 2025.

Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollár, and Ross Girshick. Segment anything. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 4015–4026, 2023.

Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalantidis, Li-Jia Li, David A. Shamma, Michael S. Bernstein, and Li Fei-Fei. Visual Genome: Connecting language and vision using crowdsourced dense image annotations. International Journal ofComputer Vision, 123:32–73, 2017. doi: 10.1007/s11263-016-0981-7.

Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman, et al. Tülu 3: Pushing frontiers in open language model post-training. arXiv preprint arXiv:2411.15124, 2024.

Youngwan Lee, Soojin Jang, Yoorhim Cho, Seunghwan Lee, Yong-Ju Lee, and Sung Ju Hwang. Multihopspatial: Multi-hop compositional spatial reasoning benchmark for vision-language model. In European Conference on Computer Vision, pp. 580–598. Springer, 2026.

Dingming Li, Hongxing Li, Zixuan Wang, Yuchen Yan, Hang Zhang, Siqi Chen, Guiyang Hou, Shengpei Jiang, Wenqi Zhang, Yongliang Shen, Weiming Lu, and Yueting Zhuang. ViewSpatial-Bench: Evaluating multi-perspective spatial localization in vision-language models. arXiv preprint arXiv:2505.21500, 2025a.

Hongxing Li, Dingming Li, Zixuan Wang, Yuchen Yan, Hang Wu, Wenqi Zhang, Yongliang Shen, Weiming Lu, Jun Xiao, and Yueting Zhuang. SpatialLadder: Progressive training for spatial reasoning in vision-language models. In International Conference on Learning Representations (ICLR), 2026a.

Junxian Li, Ruixuan Yang, Tianao Zhang, Tiange Xu, Weisheng Dong, and Yulun Zhang. Fewer tokens, more self-teaching: On-policy self-distillation for extreme visual token reduction. arXiv preprint arXiv:2609.32353, 2026b.

Kaican Li, Lewei Yao, Jiannan Wu, Tiezheng Yu, Jierun Chen, Haoli Bai, Lu Hou, Lanqing Hong, Wei Zhang, and Nevin L Zhang. Insight-o3: Empowering multimodal foundation models with generalized visual search. In The Fourteenth International Conference on Learning Representations, 2026c.

Kaixin Li, Ziyang Meng, Hongzhan Lin, Ziyang Luo, Yuchen Tian, Jing Ma, Zhiyong Huang, and Tat-Seng Chua. Screenspot-pro: Gui grounding for professional high-resolution computer use. In Proceedings of the 33rd ACM International Conference on Multimedia, pp. 8778–8786, 2025b.

Xiping Li and Jianghong Ma. AIM-CoT: Active information-driven multimodal chain-of-thought for vision-language reasoning. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 26656–26681, San Diego, California, United States, 2026. Association for Computational Linguistics. doi: 10.18653/v1/2026.acl-long. 1227.

Shijie Lian, Changti Wu, Laurence Tianruo Yang, Hang Yuan, Bin Yu, Lei Zhang, and Kai Chen. Euclid’s gift: Enhancing spatial perception and reasoning in vision-language models via geometric surrogate tasks. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Findings, pp. 9824–9835, 2026.

Zichao Lin, Yifeng Xie, Bowen Qu, Haiming Wang, Jia Li, Haoning Wu, Yuhao Dong, Zuhao Yang, Jinguo Zhu, Haoyu Lu, et al. Perceptionbench: Evaluating atomic visual perception in multimodal large language models. arXiv preprint arXiv:2607.24957, 2026.

Xianjie Liu, Yiman Hu, Yixiong Zou, Liang Wu, Jian Xu, and Bo Zheng. Hide: Rethinking the zoom-in method in high resolution mllms via hierarchical decoupling. arXiv preprint arXiv:2510.00054, 2025.

Ziyu Liu, Tao Chu, Yuhang Zang, Xilin Wei, Xiaoyi Dong, Pan Zhang, Zijian Liang, Yuanjun Xiong, Yu Qiao, Dahua Lin, and Jiaqi Wang. MMDU: A multi-turn multi-image dialog understanding benchmark and instruction-tuning dataset for LVLMs. arXiv preprint arXiv:2406.11833, 2024.

Run Luo, Haonan Zhang, Longze Chen, Ting-En Lin, Xiong Liu, Yuchuan Wu, Min Yang, Yongbin Li, Minzheng Wang, Pengpeng Zeng, Lianli Gao, Heng Tao Shen, Yunshui Li, Hamid Alinejad Rokny, Xiaobo Xia, Jingkuan Song, and Fei Huang. MMEvol: Empowering multimodal large language models with evol-instruct. In Findings ofthe Associationfor Computational Linguistics:

ACL 2025, pp. 19655–19682. Association for Computational Linguistics, 2025. doi: 10.18653/ v1/2025.findings-acl.1009.

Shravan Nayak, Xiangru Jian, Kevin Qinghong Lin, Juan A. Rodriguez, Montek Kalsi, Rabiul Awal, Nicolas Chapados, M. Tamer Özsu, Aishwarya Agrawal, David Vazquez, Christopher Pal, Perouz Taslakian, Spandana Gella, and Sai Rajeswar. Ui-vision: A desktop-centric gui benchmark for visual perception and interaction. arXiv preprint arXiv:2503.15661, 2025.

Qwen Team. Qwen3.5: Towards native multimodal agents, February 2026a. URL https:// qwen.ai/blog?id=qwen3.5.

Qwen Team. Qwen3.6-27B: Flagship-level coding in a 27B dense model, April 2026b. URL https://qwen.ai/blog?id=qwen3.6-27b.

Xiyu Ren, Zhaowei Wang, Yiming Du, Zhongwei Xie, Chi Liu, Xinlin Yang, Haoyue Feng, Wenjun Pan, Tianshi Zheng, Baixuan Xu, et al. Memlens: Benchmarking multimodal long-term memory in large vision-language models. arXiv preprint arXiv:2605.14906, 2026.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. DeepSeekMath: Pushing the limits of math ematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Haozhan Shen, Shilin Yan, Hongwei Xue, Shuaiqi Lu, Xiaojun Tang, Guannan Zhang, Tiancheng Zhao, and Jianwei Yin. Mm-condchain: A programmatically verified benchmark for visually grounded deep compositional reasoning. arXiv preprint arXiv:2603.12266, 2026.

Dingjie Song, Shunian Chen, Guiming Hardy Chen, Fei Yu, Xiang Wan, and Benyou Wang. Milebench: Benchmarking mllms in long context. arXiv preprint arXiv:2404.18532, 2024.

Zhaochen Su, Jincheng Gao, Hangyu Guo, Zhenhua Liu, Lueyang Zhang, Xinyu Geng, Shijue Huang, Peng Xia, Guanyu Jiang, Cheng Wang, et al. Agentvista: Evaluating multimodal agents in ultra-challenging realistic visual scenarios. arXiv preprint arXiv:2602.23166, 2026.

Junyoung Sung, Seungwoo Lyu, Minjun Kim, Sumin An, Arsha Nagrani, and Paul Hongsuck Seo. CRIT: Graph-based automatic data synthesis to enhance cross-modal multi-hop reasoning. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026.

Jayant Sravan Tamarapalli, Rynaa Grover, Nilay Pande, and Sahiti Yerramilli. CountQA: How well do MLLMs count in the wild? arXiv preprint arXiv:2508.06585, 2025.

Duong T Tran, Trung-Kien Tran, Manfred Hauswirth, and Danh Le Phuoc. Reasonvqa: A multihop reasoning benchmark with structural knowledge for visual question answering. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 18793–18803. IEEE, 2025.

Haochen Wang, Xiangtai Li, Zilong Huang, Anran Wang, Jiacong Wang, Tao Zhang, Jiani Zheng, Sule Bai, Zijian Kang, Jiashi Feng, Zhuochen Wang, and Zhaoxiang Zhang. Traceable evidence enhanced visual grounded reasoning: Evaluation and methodology. arXiv preprint arXiv:2507.07999, 2025a.

Qineng Wang, Baiqiao Yin, Pingyue Zhang, Jianshu Zhang, Kangrui Wang, Zihan Wang, Jieyu Zhang, Keshigeyan Chandrasegaran, Han Liu, Ranjay Krishna, et al. MindCube: Spatial mental modeling from limited views. arXiv preprint arXiv:2506.21458, 2025b.

Shenzhi Wang, Shixuan Liu, Jing Zhou, Chang Gao, Xiong-Hui Chen, Binghai Wang, An Yang, Shiji Song, Bowen Yu, Gao Huang, et al. Hopchain: Multi-hop data synthesis for generalizable vision-language reasoning. arXiv preprint arXiv:2603.17024, 2026.

Wenbin Wang, Liang Ding, Minyan Zeng, Xiabin Zhou, Li Shen, Yong Luo, and Dacheng Tao. Divide, conquer and combine: A training-free framework for high-resolution image perception in multimodal large language models. In Proceedings of the AAAI Conference on Artificial Intelligence, 2025c.

Zhaowei Wang, Wenhao Yu, Xiyu Ren, Jipeng Zhang, Yu Zhao, Rohit Saxena, Liang Cheng, Ginny Wong, Simon See, Pasquale Minervini, Yangqiu Song, and Mark Steedman. MMLongBench: Benchmarking long-context vision-language models effectively and thoroughly. In Advances in Neural Information Processing Systems, 2025d.

Lai Wei, Yuting Li, Chen Wang, Yue Wang, Linghe Kong, Weiran Huang, and Lichao Sun. First sft, second rl, third upt: Continual improving multi-modal llm reasoning via unsupervised posttraining. Advances in Neural Information Processing Systems, 38:62293–62318, 2025a.

Lai Wei, Yuting Li, Kaipeng Zheng, Chen Wang, Yue Wang, Linghe Kong, Lichao Sun, and Weiran Huang. Advancing multimodal reasoning via reinforcement learning with cold start. arXiv preprint arXiv:2505.22334, 2025b.

Lai Wei, Liangbo He, Jun Lan, Lingzhong Dong, Yutong Cai, Siyuan Li, Huijia Zhu, Weiqiang Wang, Linghe Kong, Yue Wang, Zhuosheng Zhang, and Weiran Huang. Zooming without zooming: Region-to-image distillation for fine-grained multimodal perception. In Proceedings of the 43rd International Conference on Machine Learning, 2026.

Penghao Wu and Saining Xie. V\*: Guided visual search as a core mechanism in multimodal LLMs. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 13084–13094, June 2024.

Tsung-Han Wu, Giscard Biamby, Jerome Quenum, Ritwik Gupta, Joseph E. Gonzalez, Trevor Darrell, and David M. Chan. Visual haystacks: A vision-centric needle-in-a-haystack benchmark. In International Conference on Learning Representations (ICLR), 2025.

Guowei Xu, Peng Jin, Ziang Wu, Hao Li, Yibing Song, Lichao Sun, and Li Yuan. Llava-cot: Let vision language models reason step-by-step. arXiv preprint arXiv:2411.10440, 2024.

Sihan Yang, Runsen Xu, Yiman Xie, Sizhe Yang, Mo Li, Jingli Lin, Chenming Zhu, Xiaochen Chen, Haodong Duan, Xiangyu Yue, et al. Mmsi-bench: A benchmark for multi-image spatial intelligence. In International Conference on Learning Representations, volume 2026, pp. 157051–157088, 2026.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Juncai Liu, LingJun Liu, Xin Liu, Haibin Lin, Zhiqi Lin, Bole Ma, Guangming Sheng, Yuxuan Tong, Chi Zhang, Mofan Zhang, Ru Zhang, et al. DAPO: An open-source LLM reinforcement learning system at scale. In Advances in Neural Information Processing Systems, 2025.

Jiahui Zhang, Yurui Chen, Yueming Xu, Ze Huang, Jilin Mei, Chunhui Chen, Yanpeng Zhou, Yu-Jie Yuan, Xinyue Cai, Guowei Huang, et al. From flatland to space: Teaching vision-language models to perceive and reason in 3d. Advances in Neural Information Processing Systems, 38, 2025a.

Junkai Zhang, Yihe Deng, Kai-Wei Chang, and Wei Wang. Thinking with visual grounding. arXiv preprint arXiv:2606.16122, 2026a.

Letian Zhang, Quan Cui, Bingchen Zhao, and Cheng Yang. Oasis: One image is all you need for multimodal instruction data synthesis. arXiv preprint arXiv:2503.08741, 2025b.

Tao Zhang, Ziqi Zhang, Zongyang Ma, Yuxin Chen, Bing Li, Chunfeng Yuan, Guangting Wang, Fengyun Rao, Ying Shan, and Weiming Hu. MMhops-R1: Multimodal multi-hop reasoning. In Proceedings ofthe AAAI Conference on Artificial Intelligence, 2026b.

Yi-Fan Zhang, Xingyu Lu, Shukang Yin, Chaoyou Fu, Wei Chen, Xiao Hu, Bin Wen, Kaiyu Jiang, Changyi Liu, Tianke Zhang, Haonan Fan, Kaibing Chen, Jiankang Chen, Haojie Ding, Kaiyu Tang, Zhang Zhang, Liang Wang, Fan Yang, Tingting Gao, and Guorui Zhou. Thyme: Think beyond images. arXiv preprint arXiv:2508.11630, 2025c.

Yi-Fan Zhang, Huanyu Zhang, Haochen Tian, Chaoyou Fu, Shuangqing Zhang, Junfei Wu, Feng Li, Kun Wang, Qingsong Wen, Zhang Zhang, Liang Wang, Rong Jin, and Tieniu Tan. MME-RealWorld: Could your multimodal LLM challenge high-resolution real-world scenarios that are difficult for humans? In International Conference on Learning Representations (ICLR), 2025d.

Zhuosheng Zhang, Aston Zhang, Mu Li, Hai Zhao, George Karypis, and Alex Smola. Multimodal chain-of-thought reasoning in language models. arXiv preprint arXiv:2302.00923, 2023.

Henry Hengyuan Zhao, Pan Zhou, and Mike Zheng Shou. GENIXER: Empowering multimodal large language models as a powerful data generator. In Computer Vision – ECCV 2024, volume 15081 of Lecture Notes in Computer Science, pp. 129–147. Springer, 2024.

Ziwei Zheng, Michael Yang, Jack Hong, Chenxiao Zhao, Guohai Xu, Le Yang, Chao Shen, and Xing Yu. DeepEyes: Incentivizing “thinking with images” via reinforcement learning. arXiv preprint arXiv:2505.14362, 2025.

## Appendix

## A IMPLEMENTATION DETAILS

This section complements Section 3 with the concrete settings used to construct our training data, following the order of that section. The pipeline mainly uses Qwen3.5-397B-A17B (Qwen Team, 2026a), and Table 4 summarizes all model calls, each with the thinking mode disabled.

Table 4: Model calls in the EVIDENCECONNECT pipeline. $V _ { G }$ denotes the privileged layout, $V _ { G } ^ { - i }$ the layout with region $r _ { i }$ masked, and $V _ { I }$ the policy view.
<table><tr><td>Stage</td><td>Model</td><td>Visual input</td><td>Calls</td><td>Temp.</td></tr><tr><td>Evidence discovery</td><td>Qwen3.5-397B-A17B</td><td>source image</td><td>1 per image</td><td>0.4</td></tr><tr><td>Evidence grouping</td><td>Qwen3.5-397B-A17B</td><td>none (text only)</td><td>1 per image</td><td>0.3</td></tr><tr><td>Question realization</td><td>Qwen3.5-397B-A17B</td><td> $V _ { G }$ </td><td>1 per group</td><td>0.5</td></tr><tr><td>Answer anchoring</td><td>Qwen3.5-397B-A17B, Kimi-K2.6</td><td> $V _ { G }$ </td><td>up to 4 + 4 per candidate</td><td>0.4</td></tr><tr><td>Equivalence judging</td><td>GLM-5.2</td><td>none (text only)</td><td>when needed</td><td>0.0</td></tr><tr><td>Drop-one testing</td><td>Qwen3.5-397B-A17B</td><td> $V _ { G } ^ { - i }$ </td><td>1–3 per region</td><td>0.4</td></tr><tr><td>Unmarked rewrite</td><td>Qwen3.5-397B-A17B</td><td>unmarked source image</td><td>1 per aggregate question</td><td>0.2</td></tr><tr><td>Unmarked audit</td><td>Qwen3.5-397B-A17B, Kimi-K2.6</td><td>source image</td><td>2 + 2 per rewrite</td><td>0.0</td></tr><tr><td>Distractor generation</td><td>Qwen3.5-397B-A17B</td><td>marked image</td><td>1 per marked candidate</td><td>0.8</td></tr><tr><td>Difficulty filtering</td><td>Qwen3.6-35B-A3B</td><td> $V _ { I }$ </td><td>4 per example</td><td>0.7</td></tr></table>

Image source and pre-filtering. We use SA-1B (Kirillov et al., 2023) as the sole image source. Before any model call, we keep only images whose longer side has at least 1280 pixels, whose mean per-channel intensity entropy is at least 6.5 bits, whose Canny edge density is at least 0.04, and whose luminance standard deviation is at least 60.

Grounded evidence discovery. For each image, a single call with the image and its caption (Prompt B.1) judges whether the image can support multi-region questions and returns up to 18 region proposals, whose box coordinates are converted to pixels and clamped to the image. We then discard boxes that cover less than 0.04% or more than 18% of the image, as well as boxes whose IoU with an earlier kept box is at least 0.65. Images judged unsuitable, or with fewer than four remaining regions, are skipped.

Task-conditioned evidence grouping. A text-only call receives the caption together with the label and local description of each region, ranked by grounding confidence (Prompt B.2), and proposes up to six groups of 3–7 regions. Each group is annotated with one of 22 task families, which map to four necessity modes that refine the three question forms of Section 3.1. Specifically, answer\_sensitive covers aggregation (counting, numeric aggregation, and arithmetic), set\_complete (filtering, anomaly, extremum, matching, and geometric, topological, or depth relations) and pair\_complete (cross-region pairing) cover comparison and matching, and chain\_dependency (OCR, spatial, viewpoint, conditional, Boolean, relational, and multi-hop chains) covers dependency chains and compositions. We keep groups whose region identifiers are valid, whose size lies in the allowed range, and that do not repeat an earlier group. In addition, the candidates of a filtering, anomaly, or extremum group must not mix coarse semantic classes, and groups with distinct task families are kept before groups that repeat a family.

Privileged evidence layout. Each region is cropped at native resolution with a margin of 8% of its box size on every side. The crops are placed on a white canvas through an order-preserving remapping of each image axis, which removes the rows and columns not covered by any crop, and the canvas is uniformly downscaled only when its longer side exceeds 4096 pixels. Each region receives a distinct color from a seven-color palette, which bounds the group size at seven. The same colors are used in the question text, in the privileged and ablated layouts, and for the outlines on the marked policy view.

Question realization. For each group, the generator receives the privileged layout, the group metadata (labels, local descriptions, natural referents, box coordinates, and outline colors), four blueprints sampled from a library of 104 reasoning templates, and up to four additional free-form slots, and it returns up to eight candidates (Prompt B.3). The first blueprint follows the proposed task family when possible, and the remaining ones diversify answer forms and question types. Each candidate declares its question type, necessity mode, answer semantics (scalar, unordered set, or ordered pair), answer kind (outline color, number, Boolean pattern, semantic phrase, text, semantic set, or semantic pair), the contribution of every region, the predicted answer after removing each region, and a reasoning certificate. Open-vocabulary answers must contain fewer than ten words.

Rule-based checks and candidate selection. A certificate is valid only if every step reads from earlier steps alone, every selected region lies on a path to the output step, and no step is disconnected from it. Aggregation, arithmetic, and Boolean questions must further use an executable certificate with a restricted operator set. For aggregation questions, every region must receive a positive value, and the values must sum to the answer without all being one, since the answer would otherwise equal the number of outlined regions. Similarly, arithmetic expressions must reference every region and evaluate to the answer. We further reject candidates that expose region identifiers, do not refer to the marked regions, use conditional wording without explicit branches or outside conditional types, count vague background targets, request more than one output for a scalar answer, omit an explicit coordinate frame for spatial types, leak the answer through an outline color, use subjective phrasing, or declare an answer form that the answer does not match. Candidates whose own predicted answer after removing some region equals their proposed answer are rejected as well. The remaining candidates are ranked by the generator’s self-assessed naturalness, visual confidence, and multi-region scores, with bonuses for compositional types and non-numeric answers, and this ranking only determines which candidates proceed to verification. Per group, we keep up to four top-ranked candidates, preferring distinct question types and allowing at most one numeric and two outline-color answers. Per image, we keep up to eight candidates but always retain the best candidate of each question type.

Cross-model answer anchoring. Each candidate is answered on the privileged layout by both anchoring families, with up to four samples per family and without the region descriptions or the generator’s answer (Prompt B.4). For closed-form questions, the prompt also lists the allowed answer labels, and we first draw two samples per family. Such a candidate is rejected early unless at least three of these four responses agree and each family contributes at least one of them. Responses are clustered by normalized string matching with numeric equivalence, and if the largest cluster has fewer than six members, the text-only judge re-clusters all responses by semantic equivalence (Prompt B.5). A candidate is accepted if the largest cluster is not an insufficiency response and contains at least three of the four responses from each family, and its answer then becomes a<sup>⋆</sup>. Finally, $a ^ { \star }$ is re-checked against the answer-form constraints above.

Adaptive drop-one testing. For each selected region, its crop in the privileged layout is replaced by a gray block with a red cross, while all other crops stay in place. The first anchoring family answers each ablated view once, with a mode-specific addendum to the answering prompt (Prompt B.6). If the response matches a<sup>⋆</sup> or cannot be parsed, the same view is queried twice more, and the region is redundant if at least two of its three responses match a<sup>⋆</sup>. A candidate is rejected if two or more of its regions are redundant.

Policy view. Unmarked rewriting is attempted only for aggregate question types (object counts, attribute counts, and sums of visible numbers), whose answers do not depend on the outlines, whereas the answers of selection and pairing questions are often outline colors themselves. The first anchoring family rewrites the question for the unmarked image and proposes distractors (Prompt B.7), and rewrites that still mention boxes or outlines are discarded. Two audits from each family (Prompt B.8) must then all report unique references, task preservation, identifiable targets, multi-region dependence, and necessity of every evidence area. Each audit must also recover a<sup>⋆</sup> and ground exactly one box per intended region, and the grounded boxes must match the intended regions one-to-one, where a grounded box matches a region if its center lies inside the region’s box enlarged by 15%. Otherwise, the example keeps the marked policy view, on which the box of each selected region is outlined in its assigned color.

Multiple-choice conversion. For marked examples, the first anchoring family proposes three distractors given the marked image, the question, and a<sup>⋆</sup> (Prompt B.9), whereas unmarked examples reuse the distractors from the rewrite step. We remove empty distractors, distractors that match a<sup>⋆</sup> after normalization, duplicates, and, for closed-form answers, options outside the label vocabulary. If fewer than three distractors remain, we fill the missing options deterministically, for example with alternative subsets of set-valued answers, unused vocabulary labels, or neighboring integers for count answers. The four options are then shuffled, and the prompt asks for the option letter.

Difficulty filtering. The screening model answers each example four times on its policy view (Prompt B.10). Examples with four correct responses or any unparseable response are discarded, and the remaining ones are labeled easy, medium, hard, or very hard when three, two, one, or none of the responses are correct. Finally, we keep at most four examples per source image, and each training example consists of the policy view and the multiple-choice question.

## B PROMPT TEMPLATES FOR EVIDENCECONNECT

This section lists the prompt templates used by EVIDENCECONNECT. We replace instance-specific values with angle-bracketed placeholders and condense long templates for readability. Specifically, we keep their task definitions, key constraints, and main output fields, and omit examples, formatting reminders, repeated instructions, and fields unused in our setting.

## B.1 EVIDENCE DISCOVERY AND QUESTION CONSTRUCTION

## Prompt B.1: Grounded region inventory

[IMAGE]   
You are a precise visual grounding annotator and data-quality gate.   
Task: Decide whether this image can support natural questions whose   
answer genuinely depends on 3--7 distinct regions, then propose up to   
<MAX\_REGIONS> useful evidence regions.   
Image caption, if available: <IMAGE\_CAPTION>   
Return regions that are:   
- visually concrete and clearly localizable;   
- useful for multi-region reasoning, including people, animals, objects,   
signs, screens, text regions, fine details, and distinctive parts;   
- preferably small or medium-sized, spatially diverse, and not overly   
overlapping;   
- suitable for coherent tasks such as positive-contribution counting,   
closed-set filtering, anomaly/extremum detection, candidate matching,   
pairing, OCR--visual verification, or a short dependent spatial chain.   
Prioritize groups of related regions over isolated interesting objects.   
The image is suitable only if at least one coherent all-region task is   
possible. Avoid vague regions such as "background," "scene," "sky,"   
"room," or the whole image.   
For each region, return:   
- label: short name;   
- bbox: [x1,y1,x2,y2] in pixels, or normalized 0--1000 coordinates;   
- caption: one concise visible description;   
- natural\_referent: an identifying phrase without coordinates or IDs;   
- role: candidate | reference | comparator | text\_region |   
count\_target | fine\_part | distractor;   
- confidence: grounding confidence.

You are an expert VQA data curator. Below are <N> candidate evidence   
regions detected in one image.   
Image caption: <IMAGE\_CAPTION>   
Regions:   
- <REGION\_ID>: label=<LABEL>; caption=<LOCAL\_CAPTION>   
Propose up to <MAX\_GROUPS> combinations that are good for natural   
questions requiring many regions. A good combination is a coherent   
candidate/evidence set, not a bag of unrelated boxes. Prefer the   
largest valid set, but every selected region must have a real role.   
Allowed task families and necessity modes:   
- answer\_sensitive: aggregate\_count, aggregate\_attribute\_count,   
aggregate\_numeric, arithmetic\_composition;   
- set\_complete: set\_filtering, anomaly\_detection, extremum,   
candidate\_matching, geometric\_line\_relation, topological\_relation,   
depth\_ordering;   
- pair\_complete: cross\_region\_pairing;   
- chain\_dependency: ocr\_visual\_verification, spatial\_chain,   
dependent\_chain, viewpoint\_transform, agent\_relative\_direction,   
spatial\_transitivity, conditional\_composition, boolean\_composition,   
relational\_composition, multi\_hop\_reasoning.   
Selection rules:   
1. Select 3--7 regions and prefer larger groups only when coherent.   
2. Every region must change an aggregate, belong to the closed   
candidate/pair set, or supply an indispensable chain constraint.   
3. Prefer objective text, numbers, countable objects, repeated classes   
comparable parts, and unambiguous attributes.   
4. Filtering, anomaly, and extremum candidates must be comparable.   
5. Every aggregate-count region must contribute a positive count.   
6. Spatial spread is preferred, but coherence has priority.   
7. Diversify task families and do not default to counting.   
8. Skip the image if no valid combination exists.   
Return JSON only:   
[   
{   
"region\_ids": ["r000", "r001", "r003"],   
"task\_family": "anomaly\_detection",

```jsonl
Return JSON only:
{
"suitable_for_multi_region_vqa": true,
"richness_reason": "...",
"coherent_sets": [
{"labels": [...], "task_family": "..."}
],
"regions": [
{
"label": "...", "bbox": [...], "caption": "...",
"natural_referent": "...", "role": "...", "confidence": 0.0
}
]
}
```

## Prompt B.2: Task-conditioned region-group proposal

"necessity\_mode": "set\_complete",   
"reason": "short objective rationale"   
}   
]   
Prompt B.3: Privileged question and certificate realization   
[PRIVILEGED LAYOUT WITH K COLOR-OUTLINED CROPS]   
Create <NUM\_QUESTIONS> diverse multi-region VQA questions.   
Scope contract:   
- The selected set is exactly <REGION\_IDS>.   
- Every question must genuinely use all K regions.   
- evidence\_region\_ids, region\_contributions, answer\_if\_dropped, and   
the reasoning certificate must cover exactly this set.   
- Never add decorative observations for otherwise unused regions.   
- Questions must be answerable from visible pixels in this layout,   
not from region captions.   
Visible crop anchors (the same outline colors appear on the marked image)   
- <REGION\_ID>: label=<LABEL>; outline\_color=<COLOR>; bbox=<BBOX>;   
caption=<LOCAL\_CAPTION>; natural\_referent=<REFERENT>   
...   
Task guidance:   
- Preferred task family: <TASK\_FAMILY>.   
- Preferred necessity mode: <NECESSITY\_MODE>.   
- Selected reasoning blueprints: <TEMPLATE\_SPECS>.   
- Instantiate a blueprint only when its facts are objectively visible.   
- Include diverse closed-set, comparison/matching, OCR/spatial/   
relational, and compositional structures.   
- Prefer select-then-report questions in which several regions identify   
a target and a second visible property supplies the final answer.   
- Keep the answer short, objective, unique, and absent from the stem.   
Necessity contract:   
- answer\_sensitive: every region contributes a positive value, so   
masking it changes the result or makes the aggregate unknown.   
- set\_complete / pair\_complete: all regions form the explicitly closed   
candidate or pairing set; masking one prevents proving completeness.   
- chain\_dependency: every region supplies an indispensable clue or hop.   
- For every region, answer\_if\_dropped must differ from the reference   
answer or be "insufficient information."   
Certificate contract:   
- First design a natural question from pixels; then provide its   
certificate. Do not verbalize the certificate mechanically.   
- Use mode="executable" for deterministic reading, counting, filtering,   
matching, comparison, arithmetic, Boolean, and lookup programs.   
- Use mode="evidence\_graph" for open-vocabulary semantic or unusual   
multi-hop reasoning.   
- Every step has id, op, regions, inputs, and query.   
- Inputs may reference only earlier steps.   
- Every selected region must lie on a path to output\_step.   
- No disconnected or decorative step is allowed.   
Wording contract:   
- Ask for one final answer.   
- Use one explicit coordinate frame for spatial questions.

- Use outline colors only as reference anchors, never as leaked object   
attributes.   
- Open answers must be objective and fewer than 10 words.   
- Do not use subjective comparisons, vague counts, or invisible facts.   
Return JSON only, one object per candidate:   
[   
{   
"template\_id": "...",   
"question": "...",   
"question\_type": "...",   
"necessity\_mode": "answer\_sensitive | set\_complete |   
pair\_complete | chain\_dependency",   
"answer\_semantics": "scalar | unordered\_set | ordered\_pair",   
"final\_answer\_kind": "anchor | numeric | boolean | semantic |   
semantic\_set | semantic\_pair | text",   
"evidence\_region\_ids": [...],   
"region\_contributions": {"r000": "...", "...": "..."},   
"region\_values": {"r000": 0},   
"aggregation": "...",   
"operation\_expression": "...",   
"candidate\_region\_ids": [...],   
"closed\_candidate\_set": true,   
"pair\_region\_ids": [[...], ...],   
"chain\_region\_ids": [...],   
"coordinate\_frame": "image\_plane | camera | agent | object |   
scene\_3d | not\_applicable",   
"answer\_vocabulary": [...],   
"answer\_if\_dropped": {"r000": "...", "...": "..."},   
"hypothetical\_answer": "...",   
"reasoning\_certificate": {   
"mode": "executable | evidence\_graph",   
"steps": [   
{"id": "s1", "op": "...", "regions": ["r000"],   
"inputs": [], "query": "..."}   
],   
"output\_step": "..."   
},   
"naturalness\_score": 1,   
"visual\_confidence": 1,   
"multi\_region\_score": 1   
}   
]

## B.2 ANSWER ANCHORING AND DEPENDENCY CHECKS

## Prompt B.4: Privileged-view answer anchoring

[PRIVILEGED LAYOUT]   
Answer the question using only the provided image.   
Question: <QUESTION>   
Allowed canonical answer labels: <OPTIONAL\_NEUTRAL\_VOCABULARY>   
You may reason briefly before answering. Put only the final short answer   
inside <answer>...</answer>.   
Rules:   
- Use an objectively supported expression of fewer than 10 words.

Every answer must occur in exactly one group. Break size ties using the   
lexicographically smallest canonical form.

- Use Arabic digits for numbers, with no units or separators.   
- Do not emit coordinates, grounding tokens, or box markers.   
- Use the same language as the question.   
- If evidence is insufficient, output   
<answer>insufficient information</answer>.   
- The answer tag is required; responses without it are discarded.

## Prompt B.5: Semantic-equivalence clustering

You are an answer-equivalence judge for a VQA task.

Question: <QUESTION>   
Answers:   
A1: <ANSWER\_1>   
AN: <ANSWER\_N>

```json
Return strict JSON only:
{
"groups": [["<CANONICAL_FORM>", [<ANSWER_INDICES>]], ...],
"largest_group_size": <INTEGER>,
"canonical": "<CANONICAL_FORM_OF_LARGEST_GROUP>"
}
```

Prompt B.6: Mode-specific drop-one verification The drop-one verifier inserts one of the following addenda into Prompt B.4, depending on the necessity mode.

[answer\_sensitive]   
ABLATION VIEW: one required aggregate region has been replaced by a   
gray masked box. If the full aggregate cannot be computed, answer   
"insufficient information"; do not count only visible boxes.   
[set\_complete or pair\_complete]   
ABLATION VIEW: one colored candidate box has been replaced by a gray   
masked box. The masked box remains a member of the explicitly closed   
candidate/pairing set. If its hidden content prevents proving the   
complete subset, unique anomaly, global extremum, unique match, or   
complete pairing, answer "insufficient information."   
[chain\_dependency]   
ABLATION VIEW: one required clue region has been replaced by a gray   
masked box. If the missing clue breaks the reasoning chain, answer   
"insufficient information"; do not guess.

## B.3 POLICY-VIEW AND MULTIPLE-CHOICE CONSTRUCTION

## Prompt B.7: Unmarked-question rewrite

[UNMARKED ORIGINAL IMAGE]   
You are given an anchored visual question and its verified answer.   
Anchored question: <QUESTION>

Verified answer: <ANSWER>

Create a minimally changed question that works on the unmarked image. Replace box/outline references with visually unique natural descriptions or a clearly exhaustive image-level scope. Do not add facts, change the operation, or change the answer. If unique reference cannot be guaranteed, set safe\_without\_marks=false. Also generate three plausible incorrect distractors in the same answer format.

Return JSON only:   
{   
"safe\_without\_marks": true,   
"natural\_question": "...",   
"distractors": ["...", "...", "..."]   
}

## Prompt B.8: Independent audit of the unmarked rewrite

[UNMARKED ORIGINAL IMAGE]   
Audit this visual question using only the unmarked original image.

Anchored source question: <ANCHORED\_QUESTION>   
Unmarked natural question: <NATURAL\_QUESTION>

Independently answer the question and judge whether every reference or exhaustive target set is uniquely determined. The question must require multiple spatially separated regions. Ground each evidence area with a box in normalized 0--1000 coordinates, locating it from pixels rather than inferring it from wording. Also judge whether the rewrite preserves the source task and whether every grounded area is required.

```json
Return JSON only:
{
"answer": "...",
"reference_status": "unique | ambiguous",
"same_task_as_source": true,
"all_targets_identifiable": true,
"requires_multiple_regions": true,
"every_evidence_area_required": true,
"evidence_boxes": [
{"description": "...", "bbox_2d": [x1,y1,x2,y2]}
],
"reason": "..."
}
```

## Prompt B.9: Multiple-choice distractor generation

[MARKED ORIGINAL IMAGE]   
You are given a visual question and its correct answer.

Original question: <QUESTION>   
Correct answer: <ANSWER>

Generate three plausible but incorrect multiple-choice distractors. Do not rewrite, paraphrase, simplify, or neutralize the question. Keep the stem exactly unchanged. Distractors must be visually relevant, distinct from the answer, and matched in answer format and approximate length. For open-vocabulary answers, use concrete alternatives with the same grammatical form and specificity. Do not use "none of the above,"

"all of the above," or "cannot be determined."   
Return JSON only:   
[{"distractors": ["d1", "d2", "d3"]}]  
Difficulty distribution

![](images/ac9bcf372205bf3eecab00bdb1d2613dd5cd73855c9a7e29e90e55cd752e3a84.jpg)  
Region-count distribution

![](images/08b38a873cf50cf3d4a4caa24ad8c28e9189669ff3604ed9fc2f9ae12b794229.jpg)

Figure 7: Composition of the 20.4K training examples by the number of selected regions per exam ple and by difficulty level.

Prompt B.10: Difficulty sampling on the policy view

[POLICY VIEW: MARKED OR UNMARKED IMAGE] Answer this multiple-choice visual question using only the image.

<QUESTION\_AND\_OPTIONS>

You may reason briefly before answering. The final answer must be: <answer>X</answer>

Rules: - X must contain only A, B, C, or D.   
- Do not output option text, coordinates, grounding tokens, or markers.   
- Keep any reasoning to a few short sentences.   
- The answer tag is required; otherwise the response is discarded.

## C TRAINING DATA STATISTICS

Our training set contains 20.4K examples synthesized from SA-1B images by the pipeline in Section 3 and Appendix A, and Figure 7 summarizes its composition. Each example involves three to seven regions, most often three (65.74%) or four (24.34%), followed by five (7.93%) and six or seven (1.99%). The difficulty levels, defined by how many of the four screening responses are correct, are broadly distributed. Specifically, easy, medium, hard, and very hard examples account for 33.75%, 22.04%, 19.07%, and 25.14% of the data, so more than 44% of the examples are hard or very hard. Very hard examples, which the screening model never answers correctly, are kept, since their reference answers have already been anchored by two stronger VLMs.

## D TRAINING DETAILS

All models are trained with the same GRPO (Shao et al., 2024) configuration for two epochs, with a batch size of 128 and a learning rate of $1 0 ^ { - 6 }$ . For each prompt, we sample $G { = } 8$ rollouts at temperature 0.7 with a maximum response length of 32K tokens, and the thinking mode is disabled during rollouts to keep responses concise. We use decoupled clipping ratios $\epsilon _ { \mathrm { l o w } } { = } 0 . 2$ and $\epsilon _ { \mathrm { h i g h } } \mathrm { = } 0 . 3$ as proposed in DAPO (Yu et al., 2025), and remove the KL penalty $( \beta _ { \mathrm { K L } } { = } 0 )$ , so that the policy is not anchored to the base model. Finally, the reward checks whether the predicted option letter matches the ground-truth option.

## E ATTENTION MAP COVERAGE ANALYSIS

Evaluation setting. We compare ZwZ2-4B/9B with their Qwen3.5 base models on the same 405 TreeBench examples. Within each scale, both checkpoints use the base-model processor and identical image–question pairs, visual-token grids, and evidence annotations. Since Qwen3.5 interleaves linear-attention and full-attention layers, we evaluate all eight full-attention layers, L = {4, 8, 12, 16, 20, 24, 28, 32} (1-based).

Attention extraction. For each example, we apply the chat template with the generation prompt and perform a single forward pass in evaluation mode using eager attention, without generating an answer. We extract attention from the final token of this complete input sequence to the image tokens. This position predicts the first answer token, so its attention indicates where the model looks as it begins to answer.

Let $V _ { i }$ denote the visual-token indices for example $i ,$ and let $q _ { i }$ be its final input-token position. For model m at layer $\ell ,$ let $T _ { i , \ell , h } ^ { m } ( q _ { i } , v )$ denote the attention weight returned by head h from $q _ { i }$ to visual token v. We convert the extracted weights to float32, average equally over all $n _ { \mathrm { h e a d s } } ^ { m , \ell }$ heads, and then renormalize within the image tokens:

$$
\begin{array} { r l } & { u _ { i , \ell } ^ { m } ( v ) = \displaystyle \frac { 1 } { n _ { \mathrm { h e a d s } } ^ { m , \ell } } \sum _ { h = 1 } ^ { n _ { \mathrm { h e a d s } } ^ { m , \ell } } T _ { i , \ell , h } ^ { m } ( q _ { i } , v ) , } \\ & { a _ { i , \ell } ^ { m } ( v ) = \displaystyle \frac { u _ { i , \ell } ^ { m } ( v ) } { \sum _ { v ^ { \prime } \in V _ { i } } u _ { i , \ell } ^ { m } ( v ^ { \prime } ) } . } \end{array}\tag{1}
$$

This renormalization over image tokens is applied once after head averaging, without renormalizing each head separately or applying an additional softmax.

Evidence mask. The visual tokens form an $H _ { i } \times W _ { i }$ grid for an image of size $H _ { i } ^ { \mathrm { i m g } } \times W _ { i } ^ { \mathrm { i m g } }$ . An annotated box $( x _ { 0 } , y _ { 0 } , x _ { 1 } , y _ { 1 } )$ covers columns $[ \lfloor x _ { 0 } W _ { i } / W _ { i } ^ { \mathrm { i m g } } \rfloor , \lceil x _ { 1 } W _ { i } / W _ { i } ^ { \mathrm { i m g } } \rceil )$ and the analogous row range, clipped to the grid boundaries. This range includes every visual-token cell that overlaps the box. The evidence set $E _ { i } \subseteq V _ { i }$ is the union of the tokens covered by all annotated boxes, counting overlapping tokens once. All evaluated examples have $| E _ { i } | > 0$

Top-k evidence IoU. For each example, we set $k _ { i } = | E _ { i } |$ and select the $k _ { i }$ visual tokens with the largest head-averaged weights $u _ { i , \ell } ^ { m } ( v )$ to form $A _ { i . \ell } ^ { m }$ . Ranking by the normalized weights $a _ { i , \ell } ^ { m } ( v )$ gives the same ordering because the normalization divides all weights by a common positive value. Ties are broken by visual-token index in row-major order, so exactly $k _ { i }$ tokens are selected even when ties occur at the boundary. Because $k _ { i }$ is fixed by the annotation, the base and trained model receive the same token budget.

With $h _ { i , \ell } ^ { m } = | A _ { i , \ell } ^ { m } \cap E _ { i } |$ , we compute

$$
\mathrm { I o U } _ { i , \ell } ^ { m } = \frac { | A _ { i , \ell } ^ { m } \cap E _ { i } | } { | A _ { i , \ell } ^ { m } \cup E _ { i } | } = \frac { h _ { i , \ell } ^ { m } } { 2 k _ { i } - h _ { i , \ell } ^ { m } } .\tag{2}
$$

The score lies in $\lbrack 0 , 1 \rbrack$ , with larger values indicating better spatial alignment with the annotated evidence. Since $| \dot { A _ { i , \ell } ^ { m } } | \overset { } { = } | E _ { i }$ |, precision equals recall, $p = h _ { i , \ell } ^ { m } / k _ { i }$ , and the IoU equals $p / ( 2 - p )$

Table 5: Top-k evidence IoU on TreeBench. Mean IoU is reported as a percentage; $\Delta$ is ZwZ2 minus Qwen3.5 in percentage points. Each entry averages the same 405 paired examples. Differences are computed before rounding.
<table><tr><td></td><td colspan="3">4B</td><td colspan="3">9B</td></tr><tr><td>Layer</td><td>Qwen3.5</td><td>ZwZ2</td><td>∆</td><td>Qwen3.5</td><td>ZwZ2</td><td>∆</td></tr><tr><td>4</td><td>11.625</td><td>11.877</td><td>+0.252</td><td>10.622</td><td>10.699</td><td>+0.077</td></tr><tr><td>8</td><td>10.355</td><td>10.402</td><td>+0.047</td><td>10.069</td><td>10.051</td><td>-0.018</td></tr><tr><td>12</td><td>11.568</td><td>11.692</td><td>+0.124</td><td>11.073</td><td>11.050</td><td>-0.023</td></tr><tr><td>16</td><td>10.400</td><td>10.650</td><td>+0.251</td><td>11.080</td><td>11.256</td><td>+0.175</td></tr><tr><td>20</td><td>11.132</td><td>11.386</td><td>+0.255</td><td>11.493</td><td>11.777</td><td>+0.284</td></tr><tr><td>24</td><td>10.924</td><td>11.162</td><td>+0.238</td><td>10.890</td><td>11.242</td><td>+0.352</td></tr><tr><td>28</td><td>11.181</td><td>11.152</td><td>-0.029</td><td>11.147</td><td>11.680</td><td>+0.533</td></tr><tr><td>32</td><td>11.244</td><td>11.424</td><td>+0.179</td><td>10.471</td><td>10.860</td><td>+0.390</td></tr></table>

Aggregation and layer-wise results. We take the unweighted mean of the per-example IoU over the N = 405 paired examples, rather than pooling intersections and unions, and report differences in percentage points:

$$
\begin{array} { r c l } { \overline { { \mathrm { I o U } } } _ { \ell } ^ { m } = \displaystyle \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathrm { I o U } _ { i , \ell } ^ { m } , } \\ { \Delta _ { \ell } = 1 0 0 \left( \overline { { \mathrm { I o U } } } _ { \ell } ^ { \mathrm { Z w Z 2 } } - \overline { { \mathrm { I o U } } } _ { \ell } ^ { \mathrm { Q w e n 3 . 5 } } \right) . } \end{array}\tag{3}
$$

Table 5 shows gains in 7 of 8 layers at 4B and 6 of 8 layers at 9B. The exceptions, layer 28 at 4B and layers 8 and 12 at 9B, each decrease by less than 0.03 percentage points. These comparisons describe changes in sample means. The metric measures spatial alignment at the final prompt position, rather than total attention mass or answer correctness.

## F CASE STUDY

To complement the quantitative evaluation, we examine three selected examples in which ${ \bf Z } { \bf w } { \bf Z } 2 \cdot$ 4B answers correctly while its base model, Qwen3.5-4B, fails. Figure 8 presents examples from MIMIC, ZoomBench, and CountQA, covering cross-image category identification, spatially specified text recognition, and object counting. We analyze the models’ visible responses to identify how their answers differ in the use of visual evidence and the satisfaction of question constraints.

Cross-image consistency. In Figure 8(a), the MIMIC question asks for a category that appears in all five images. The base model selects wine glass, even though its response explicitly acknowledges that this object is absent from some images. It justifies the answer by referring to the most prominent shared item across the majority of the set, effectively replacing the requirement of universal presence with majority overlap. ZwZ2 checks candidate categories against the full set of images. Its response rules out wine glass because it is absent from Images 4 and 5, and identifies person as the category shared by all five images. This example illustrates the importance of preserving the question’s scope when integrating evidence across images: an object shared by several images does not necessarily satisfy a constraint over the entire set.

Question-conditioned text localization. Figure 8(b) asks for the black text above the lower archway. The base model recognizes the word SEGATORI on the adjacent building, but associates the question with inscriptions on the fountain and ultimately selects FONTE. Its response therefore reveals a mismatch between the recognized text and the region specified by the question. ZwZ2 identifies the entrance of the yellow building as the relevant lower archway and reads the sign immediately above it, yielding the correct answer, SEGATORI. The case highlights the role of spatial reference resolution in visual text understanding: recognizing a visible word must be accompanied by establishing whether it belongs to the queried location.

Structured visual counting. In Figure $8 ( \mathrm { c } ) ,$ the CountQA question asks for the number of tablets in a blister pack. The base model divides the image into regions above and below the yellow rubber

(a) MIMIC: Shared category  
![](images/31bb5caada591f47a7d547859c73aba28449a68f39eb0bfa07f1aa05935c6372.jpg)

## (b) ZoomBench: Target text

![](images/0b52c9f2731bc47d8da79967055f8ab8c41097c16513313ca67b78bdda33d6ad.jpg)  
(c) CountQA: Tablet counting

![](images/c170cf4867c01d152f813428ec66f4b9f8e0d38e0189c0f90c1a8e58a7595971.jpg)

Base: [...] there is a building with a sign that reads “SEGATORI” [...] not part of the fountain’s inscription.

Ours: [...] On that building, just above the entrance (which is a lower archway), there is a sign with black text. That text reads: "SEGATORI". Answer: B. SEGATORI

How many tablets are there?   
Base: [...] The top section (above the yellow rubber band) has 2 tablets. The bottom section (below the yellow rubber band) has 4 tablets. So, total tablets = 2 + 4 = 6   
Answer: 6   
Ours: [...] The blister pack is arranged in 2 columns and 5 rows. [...] Total = 5 + 5 = 10 tablets   
Answer: 10   
Error analysis: Visual undercounting. The pack has two columns of five tablets.

Base: Qwen3.5-4B. Ours: ZwZ2-4B.

Excerpts from visible responses. [...] marks omissions. Answer fields are reformatted.

Figure 8: Qualitative comparisons between Qwen3.5-4B and ZwZ2-4B. Three selected examples show correct predictions by ZwZ2-4B where the base model fails: (a) identifying a category present in every image; (b) associating recognized text with the queried location; and (c) counting all instances in a structured arrangement. Responses are excerpted from the original outputs, with omissions marked by [...] and answer fields reformatted. The enlarged image region in (b) is provided for visualization only.

band, reports two and four tablets, respectively, and returns 2 + 4 = 6. Its arithmetic is consistent with these counts, so the error lies in enumerating the tablets. ZwZ2 describes the pack as two columns and five rows, counts five tablets in each column, and obtains the correct total of 5+5 = 10. Here, the response uses the regular spatial arrangement to account for the complete set of tablets.

Together, these cases reflect the steps of connecting evidence: integrating facts over the full set of images in (a), binding recognized text to the queried location in (b), and accounting for every instance before aggregating in (c).

## G LIMITATIONS AND FUTURE WORK

Our work has two main limitations. First, although our single-image supervision transfers to multiimage and agent tasks, it does not directly model temporal or cross-frame dependencies, so extending the synthesis to evidence spread across video frames is a natural next step. Second, we apply RLVR to models of up to 27B parameters, and whether the same supervision benefits larger models or earlier training stages, such as pre-training, remains to be studied.