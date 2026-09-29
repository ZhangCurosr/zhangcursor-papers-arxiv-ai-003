# WHEN VLMS TRUST CONTEXT: EVALUATING SCENETEXT RECOGNITION UNDER MISLEADING CONTEXT

Yuxing Cheng<sup>1</sup> Yuan Wu<sup>1\*</sup> Yi Chang<sup>1,2,3\*</sup>

<sup>1</sup>School of Artificial Intelligence, Jilin University

<sup>2</sup>Engineering Research Center of Knowledge-Driven Human-Machine Intelligence, MOE, China <sup>3</sup>International Center of Future Science, Jilin University

chengyx26@mails.jlu.edu.cn, yichang@jlu.edu.cn, yuanwu@jlu.edu.cn

## ABSTRACT

Vision-language models (VLMs) can read text in natural scenes, but their predictions may be influenced by the surrounding context. When the printed text conflicts with what the scene suggests, a model may return a more plausible word instead of the shown text. We introduce SceneFaith, a benchmark of 781 generated scene images for studying this behavior. Each output is classified as Literal, Canonical, or Other, separating faithful transcription from context-consistent rewriting and ordinary recognition errors. Across 15 models from seven families, all models show rewriting on clear images, with rates ranging from 8.45% to 58.51%. Controlled experiments further show that surrounding context matters: removing surrounding scene information reduces rewriting and improves literal accuracy, while changing the scene around the same text patch can also change model outputs. Moreover, weakening the target text with blur increases rewriting. These results show that reliable scene-text recognition requires VLMs to balance visual character evidence with contextual information, preserving clear text while using context mainly when the visual evidence is uncertain.

## 1 INTRODUCTION

Vision-language models (VLMs) have become powerful general-purpose visual readers, achieving strong performance on text recognition, localization, document understanding, and text-rich reasoning (Liu et al., 2024; Fu et al., 2026; Huang et al., 2026). Scene Text Recognition (STR), which extracts text from natural images under complex backgrounds and imaging conditions, is increasingly important for both OCR applications and large-scale corpus construction. However, strong performance on standard OCR benchmarks does not necessarily imply faithful transcription. Unlike conventional OCR systems (Cui et al., 2025; Li et al., 2026), which are primarily designed to recover the visible character sequence, VLMs interpret text together with linguistic and visual semantics. These semantic priors are useful when visual evidence is incomplete, but they can also override what is actually printed. Recent studies show that VLMs may correct or rewrite unusual text into more plausible forms, while conventional OCR systems remain more faithful under controlled text perturbations (Zhang et al., 2026a; Lee et al., 2026). As a result, for misspellings, names, identifiers, and other atypical strings, stronger semantic understanding can sometimes become a source of recognition error rather than an advantage.

This conflict is particularly important for scene text recognition, a sub-task of OCR that recognizes text embedded in natural scenes. Conventional OCR models or STR models are highly specialized: they are typically designed to map visual text regions to character sequences, and many use linguistic information to resolve ambiguous characters (Fang et al., 2021; Wang et al., 2021; Na et al., 2022). However, they are not designed for general instruction following. VLMs can follow instructions such as “transcribe exactly what is written” while also understanding the surrounding scene. This flexibility introduces a different risk: even when literal transcription is explicitly requested, scene semantics may pull the prediction toward a more plausible word. For example, when a sign beside a dolphin reads ‘dolphiin,” a VLM may instead output ‘dolphin,” correcting the visible text to match the scene. This raises a fundamental question: can VLMs correctly transcribe text with misleading scene context?

To investigate this question, we construct a controlled benchmark for context-induced rewriting in scene text recognition. Our benchmark uses generated scenes in which the target text deliberately contains a typo while the surrounding visual scene supports its conventional form. This creates a direct conflict between local character evidence and global scene semantics. Importantly, we place the same target text in different contextual conditions, allowing changes in model predictions to be attributed to the surrounding scene rather than the text itself. Across the evaluated VLMs, we find a clear rewriting effect: models frequently replace the visible typo with the contextually expected word, and becomes more pronounced with matching scene context. We further examine how rewriting varies across visual conditions, contextual cues, and task settings. We also analyze how target-text blur and the strength of lexical priors affect model prediction. Our results show that successful scene text recognition requires not only the use of context, but also the ability to prioritize visual evidence when the two conflict. The contributions of this paper are summarized as follows:

• SceneFaith benchmark. We introduce SceneFaith<sup>1</sup>,, comprising 781 generated scene images across 17 categories. Each sample is constructed by perturbing a conventional word, embedding the resulting unusual string into a semantically matching scene, and applying quality control and blind crop reading before acceptance. With the printed string as gold, our Literal/Canonical/Other (L/C/O) evaluation separates faithful transcription, canonical rewriting, and other recognition errors (Sections 3 and 4.1).

• Analysis of the rewriting effect. Across 15 model variants from seven families, all evaluated models exhibit canonical rewriting on clear images, with rates ranging from 8.45% to 58.51%. We further isolate the effect of surrounding context through scene removal and fixed-target-patch comparisons. Removing peripheral scene information reduces rewriting and improves literal accuracy, while changing the surrounding scene around the same target patch can change model predictions (Sections 4.2).

• Analysis of factors affecting rewriting. We examine how rewriting changes with targettext degradation, lexical preference, and the use of full images versus target crops. Blur consistently increases rewriting, while stronger lexical preference for the conventional form is associated with higher rewriting rates. Cropped views further show that rewriting is sensitive to input presentation.

## 2 RELATED WORK

## 2.1 TEXT RECOGNITION SYSTEMS AND BENCHMARKS

Modern text recognition systems increasingly integrate linguistic information into the recognition process. ABINet (Fang et al., 2021), VisionLAN (Wang et al., 2021) and MATRN (Na et al., 2022) model visual and linguistic dependencies within a text sequence. CLIP4STR adapts pre-trained CLIP image and text encoders to refine crop predictions (Zhao et al., 2024), CLIPTER extends the evidence beyond the sequence by conditioning crop recognition on the whole image (Aberdam et al., 2023). Document VLMs achieve strong performance through post-training (PaddleOCR-VL-1.6), unified OCR task formats (HunyuanOCR-1.5), visual-token reordering (DeepSeek-OCR 2), and separation of global layout from local recognition (MinerU2.5) (Zhang et al., 2026b; Li et al., 2026; Wei et al., 2026; Niu et al., 2025). Evaluation has followed the same direction: earlier, Union14M isolates nonlexical and incomplete strings as a distinct difficulty (Jiang et al., 2023), while OCRBench and OCRBench v2 measure recognition, localization, parsing, and text-centric reasoning (Liu et al., 2024; Fu et al., 2026), OCR-Reasoning extend reasoning ability to text-rich reasoning (Huang et al., 2026). These benchmarks evaluate transcription accuracy and reasoning, but do not distinguish semantically plausible substitutions from ordinary recognition errors.

## 2.2 FAITHFULNESS UNDER VISUAL–SEMANTIC CONFLICT

This section reviews studies on how models respond when visual text conflicts with linguistic expectations. TextHalu-Bench evaluates semantic hallucination in scene-text spotting and understanding, and illustrates the behavior with a sign reading MMOTEL returned as MOTEL (Shu et al., 2026). FaithC4 evaluates whether models faithfully transcribe altered text in rendered multilingual documents (Lee et al., 2026), and CHAOS-Bench measures page-averaged recall of meaningless words created by modifying characters on academic paper pages (Li et al., 2026). The failure is not confined to scene text: it appears as over-correction in handwritten mathematics (Seong et al., 2026) and as model- and script-dependent grounding failures in ancient Greek and Arabic editions (Karamolegkou et al., 2026), while HallusionBench and CDH-Bench examine the broader evidence– expectation conflict through visual question answering (Guan et al., 2024; Chen et al., 2026). Architectural evidence points the same way, with stronger visual token compression in DeepSeek-OCR associated with greater reliance on language priors (Liang et al., 2026). Across this group the measurement is a transcript scored against a gold string within one fixed image, which leaves the substitution indistinguishable from other errors and the surrounding scene’s contribution unmeasured.

## 2.3 IMPROVING OCR FAITHFULNESS

Mitigation work has begun to target the same failure. ConCLR contrasts character representations across text contexts to reduce vocabulary reliance (Zhang et al., 2022). TextHalu-Bench pairs attention-based region estimation (ZoomText) with decoding guided by a training-free visually grounded intermediate layer (Grounded Layer Correction) (Shu et al., 2026). These methods adjust the model’s attention and decoding. Our experiments complement this work by testing how transcription changes when the surrounding scene is removed or the target text is blurred. The results help identify what mitigation methods need to address.

## 3 BENCHMARK DESIGN AND DIAGNOSTIC FRAMEWORK

## 3.1 BENCHMARK OVERVIEW

The benchmark is built to answer one question: When printed text conflicts with scene context, does the model transcribe it faithfully? We evaluate transcription under clear and blurred targettext conditions. Clear images test whether models preserve printed text despite misleading context, while target-local blur tests how rewriting changes as character evidence weakens. In SceneFaith, the scene and lexical preference both support the conventional spelling. The printed text uses a different perturbed spelling. Context refers to information outside the target-text region, and lexical preference is analyzed separately.

Given an image $x _ { i }$ with one red outline marking the target, the model returns a string yˆ . The printed gold $y _ { i }$ is stored as observed text; the conventional alternative $c _ { i } \neq y _ { i }$ is stored as canonical text for analysis only. SceneFaith contains 781 PNG images and annotation records across 17 categories, varying in typography, viewpoint, texture, and target size. It is designed as a stress test: images are deliberately selected to create a conflict between unusual printed text and familiar scene content. The resulting rates therefore characterize model behavior under this conflict rather than estimate general OCR accuracy.

Controlled Fixed-Target-Patch Set. Each pair places the same opaque text patch in one of two contexts: a scene that supports the conventional word (A), or a scene that encourages copying the observed text exactly (B). The target pixels, font, size, red outline, local background, and coordinates are identical on 1024 × 1024 canvases; only pixels outside the target patch differ. These pairs are constructed from the same pipelines as the main benchmark and use the observed string as the gold answer. Because this panel serves as a controlled paired analysis, we report it separately rather than merging it with the SceneFaith benchmark results. Section 4.3 presents the 11-alias comparison, while Appendix H details the control variables and inference settings.

## 3.2 CONSTRUCTION PIPELINE

Generation. We select a conventional word and a semantically relevant scene, then create an unusual string through insertion, duplication, substitution, deletion, alphanumeric confusion, or spacing changes. The generative model gpt-image-2 integrates this string into a sign, label, board, or product surface with plausible typography, perspective, and materials. A single red rectangle marks the target without overlapping any characters. The surrounding scene suggests the conventional word, creating a conflict with the printed text.

![](images/f837338c009ecb8754b9445861d69598dde751b22419b75afe0db40e1b4754fb.jpg)  
Figure 1: SceneFaith construction pipeline. Spelling perturbation and scene generation are followed by quality checks, blind crop reading, and dataset completion.

Quality control. Review checks target spelling, red-box validity, scene integration, layout, clarity, and intended scene meaning. It excludes answer leakage, meaning that the conventional spelling must not appear elsewhere in the image and the target text must not be repeated. A blind targetcrop transcription then checks whether the unusual string can be recovered without the scene, which distinguishes a legible conflicting target from an illegible one. We use GPT-5.5 to assist with quality checks.

Accepted records store the image ID, category, observed and canonical strings, dimensions, red-box coordinates, image hashes, and review evidence. All gold fields agree with observed text. Appendix A gives the domain inventory and artifact contract.

## 4 EVALUATION AND DISCUSSION

## 4.1 EVALUATION PROTOCOL

Evaluated Models. We evaluate 15 model aliases from seven families, yielding 11,715 accepted outputs on 781 full images. We group the evaluated models into two categories according to model availability. (a) Open-source MLLMs. We evaluate Qwen3-VL-8B Instruct, Qwen3-VL-32B Instruct, Qwen3-VL-235B-A22B Instruct, Qwen3-VL-8B Thinking, Qwen3-VL-32B Thinking, Qwen3-VL-235B-A22B Thinking, GLM-4.6V, InternVL3-38B, and Kimi K2.5. These models cover both instruction-tuned and reasoning-oriented configurations across a range of model scales. (b) Closed-source MLLMs. We evaluate Gemini 3.1 Flash, Gemini 3.5 Flash, Gemini 3 Flash, Claude Sonnet 4.6, GPT-5.5, and GPT-5.2. These systems are accessed through provider APIs.

Inference Setup. All requests use a temperature of 0 and a maximum output length of 2,048 tokens. Image-detail and reasoning settings remain at the provider defaults. We adopt a zeroshot evaluation protocol without fine-tuning or few-shot prompting. The prompt is: “Read the text enclosed by the single red rectangular outline in the image. Return only that text and nothing else.”.

Metrics. We classify each output as Literal (L), Canonical (C), or Other (O) to distinguish faithful transcription from correction to the conventional spelling and other errors. Let ν denote a normalization function that strips leading and trailing whitespace, collapses internal whitespace, and deletes unrelated text. For printed gold $y _ { i }$ and conventional spelling c<sub>i</sub>, we assign

![](images/77c1d5d1b65570595dc4a536f92cf05c91a6d15dc501e248fcc397fd7c4d9ea4.jpg)  
Figure 2: Qualitative example of canonical rewriting. The printed gold is ieopard; leopard is the canonical alternative.

$$
g _ { i } = \left\{ \begin{array} { l l } { L \mathrm { ( L i t e r a l ) } , } & { \nu ( z _ { i } ) = \nu ( y _ { i } ) , } \\ { C \mathrm { ( C a n o n i c a l ) } , } & { \nu ( z _ { i } ) = \nu ( c _ { i } ) , } \\ { O \mathrm { ( O t h e r ) } , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.
$$

Since $\nu ( y _ { i } ) \neq \nu ( c _ { i } )$ for every target, the categories are mutually exclusive. Their rates share the same denominator:

$$
r _ { k } = \sum _ { i = 1 } ^ { N } { \bf 1 } [ g _ { i } = k ] , \qquad r _ { L } + r _ { C } + r _ { O } = 1 0 0 .
$$

Literal accuracy is $r _ { L }$ and rewriting rate is $\mathrm { R R } = r _ { C }$ . Because $1 0 0 - \mathrm { R R } = r _ { L } + r _ { O }$ , a lower RR does not necessarily mean higher literal accuracy, we report both metrics in all comparisons. “Rewriting” refers only to the final output category and does not imply that the model first recognized the printed text and then corrected it. Complete benchmark runs use $N = 7 8 1$ ; the controlled fixed-target-patch Set. uses $N = 1 5 0$ . We retain all raw responses for auditing. All conditions are scored using the same rules, without an LLM judge.

## 4.2 SCENEFAITH EVALUATION RESULTS ANALYSIS

Every alias returns the conventional word on clear print. Table 1 reports the full panel on clear images. RR runs from 8.45% for Gemini 3.1 Flash Lite to 58.51% for InternVL3-38B, with a 15- alias mean of 26.35%. Giving each model family equal weight yields a similar clear-image RR of 27.16%. These rates describe the constructed stress set under the recorded provider configurations, not general OCR accuracy.

Closed-source models show lower rewriting overall. On clear images, the six closed-source models have a mean RR of 14.94%, compared with 33.96% for the nine open-source models. Their mean literal accuracy is also higher, at 83.12% versus 57.19%. Under moderate blur, the same pattern remains: mean RR is 27.89% for closed-source models and 42.96% for open-source models. These differences are descriptive, since the two groups differ in model family, scale, and training setup.

<table><tr><td rowspan="2">Model alias</td><td colspan="3">Clear</td><td colspan="3">Blur</td></tr><tr><td>L (Acc.) ↑ C (RR) ↓</td><td></td><td>O↓</td><td>L (Acc.) ↑</td><td>C (RR) ↓</td><td>0↓</td></tr><tr><td colspan="7">Closed-Source MLLMs</td></tr><tr><td>Gemini 3.1 Flash</td><td>90.14</td><td>8.45</td><td>1.41</td><td>78.36</td><td>20.74</td><td>0.90</td></tr><tr><td>Gemini 3.5 Flash</td><td>89.76</td><td>8.96</td><td>1.28</td><td>80.41</td><td>18.44</td><td>1.15</td></tr><tr><td>Gemini 3 Flash</td><td>88.35</td><td>11.01</td><td>0.64</td><td>79.39</td><td>19.97</td><td>0.64</td></tr><tr><td>Claude Sonnet 4.6</td><td>85.02</td><td>12.16</td><td>2.82</td><td>71.06</td><td>22.66</td><td>6.27</td></tr><tr><td>GPT-5.5</td><td>78.10</td><td>19.46</td><td>2.43</td><td>56.34</td><td>40.20</td><td>3.46</td></tr><tr><td>GPT-5.2</td><td>67.35</td><td>29.58</td><td>3.07</td><td>50.19</td><td>45.33</td><td>4.48</td></tr><tr><td colspan="7">Open-Source MLLMs</td></tr><tr><td>Qwen3-VL-8B Instruct</td><td>62.23</td><td>19.46</td><td>18.31</td><td>53.01</td><td></td><td>26.76 20.23</td></tr><tr><td>GLM-4.6V</td><td>74.26</td><td>21.25</td><td>4.48</td><td>62.61</td><td>32.01</td><td>5.38</td></tr><tr><td>Qwen3-VL-32B Instruct</td><td>60.95</td><td>27.53 11.52</td><td></td><td>51.98</td><td></td><td>33.80 14.21</td></tr><tr><td>Kimi K2.5</td><td>65.43</td><td>31.88</td><td>2.69</td><td>44.17</td><td>49.68</td><td>6.15</td></tr><tr><td>Qwen3-VL-235B-A22B Thinking</td><td>62.36</td><td>32.27</td><td>5.38</td><td>49.42</td><td>43.28</td><td>7.30</td></tr><tr><td>Qwen3-VL-235B-A22B Instruct</td><td>61.20</td><td>32.39</td><td>6.40</td><td>51.09</td><td>41.74</td><td>7.17</td></tr><tr><td>Qwen3-VL-8B Thinking</td><td>48.53</td><td>37.9013.57</td><td></td><td>34.70</td><td></td><td>47.38 17.93</td></tr><tr><td>Qwen3-VL-32B Thinking</td><td>47.50</td><td>44.43</td><td>8.07</td><td>35.60</td><td>51.47</td><td>12.93</td></tr><tr><td>InternVL3-38B</td><td>32.27</td><td>58.51</td><td>9.22</td><td>26.15</td><td>60.51</td><td>13.33</td></tr></table>

Table 1: Transcription outcomes on clear and moderately blurred text. All rates are percentages. The 15 aliases use full images and the neutral prompt and are ordered by clear RR within each group.

Rewriting rate does not restate literal accuracy. Models that rewrite equally often can read very different amounts of text correctly. GPT-5.5 and Qwen3-VL-8B Instruct both rewrite 19.46% of targets, yet their literal accuracies are 78.10% and 62.23%, with O rates of 2.43% and 18.31%. The inversion also runs the other way. GPT-5.2 rewrites more than Qwen3-VL-32B Instruct (29.58% versus 27.53%) while reading more targets correctly (67.35% versus 60.95%). GLM-4.6V rewrites more than Qwen3-VL-8B Instruct (21.25% versus 19.46%) at 12 points higher literal accuracy. Ranking on RR alone would therefore order these pairs wrongly for either purpose. Table 1 reports all three outcomes for both conditions, and Appendices C and G give uncertainty estimates and supporting plots.

Thinking aliases are associated with higher rewriting. Within Qwen3-VL, same image comparisons under identical prompts and recorded decoding settings give higher RR for Thinking than Instruct at the two smaller sizes. The differences are 18.44 and 16.90 percentage points, and both survive Holm correction across three sizes. Literal accuracy falls by a similar amount, from 62.23% to 48.53% at 8B and from 60.95% to 47.50% at 32B. At 235B-A22B the RR difference is −0.13 points (95% CI [−2.94, 2.69]), where 49 literal-to-canonical changes are offset by 54 in the reverse direction. Mixture-of-experts architecture might be a reason why the results were not noticeable. Archived reasoning text shows what the gap looks like in individual cases. For “uher”, the 8B Thinking trace repeats the printed candidate, questions it, and answers “usher”, while Instruct returns “uher”. All 2,343 Thinking outputs retain reasoning text, and seven canonical-answer traces contain both exact candidates. These traces show behavioral differences but do not explain their internal cause.

What does Other contain? We reviewed 100 of the 713 Other outputs, sampled proportionally across 15 models. Most were near-target spelling errors (66%); 30% included extra text and 4% named a different concept.

## 4.3 CONTEXT EFFECT ANALYSIS

The preceding results show that rewriting occurs, but they do not isolate the effect of the surrounding scene because the target text and context appear together. We analyze the effect of scene context through the scene-removal experiment on SceneFaith and the evaluation on the Controlled Fixed-Target-Patch Set.

<table><tr><td rowspan="2">Model alias</td><td colspan="4">Full image</td><td colspan="4">Gray mask</td><td rowspan="2">95% CI</td></tr><tr><td>L</td><td></td><td>C</td><td>0</td><td>L</td><td>C</td><td>0</td><td>∆C</td></tr><tr><td>GPT-5.2</td><td>67.35</td><td>29.58</td><td>3.07</td><td></td><td>84.25</td><td>11.01</td><td>4.74</td><td>-18.57</td><td>[-21.64, -15.49]</td></tr><tr><td>Gemini 3.5 Flash</td><td>89.76</td><td>8.96</td><td>1.28</td><td></td><td>96.93</td><td>2.43</td><td>0.64</td><td>-6.53</td><td>[-8.45, -4.74]</td></tr><tr><td>Claude Sonnet 4.6</td><td>85.02</td><td>12.16</td><td>2.82</td><td></td><td>96.80</td><td>2.56</td><td>0.64</td><td>-9.60</td><td>[-11.91, -7.43]</td></tr><tr><td>Qwen3-VL-32B T.</td><td>47.50</td><td>44.43</td><td>8.07</td><td></td><td>67.86</td><td>24.33</td><td>7.81</td><td>-20.10</td><td>[-23.30, -17.03]</td></tr></table>

Table 2: Scene removal with fixed target pixels. Intervals use 20,000 paired-image bootstrap resamples.

<table><tr><td rowspan="2">Model alias</td><td colspan="3">Full image</td><td colspan="3">Crop only</td></tr><tr><td>L (Acc.) ↑</td><td>C (RR) ↓</td><td>0↓</td><td>L (Acc.) ↑</td><td>C (RR)↓</td><td>0↓</td></tr><tr><td>GPT-5.2</td><td>67.35</td><td>29.58</td><td>3.07</td><td>89.88</td><td>5.63</td><td>4.48</td></tr><tr><td>Gemini 3.5 Flash</td><td>89.76</td><td>8.96</td><td>1.28</td><td>96.93</td><td>2.18</td><td>0.90</td></tr><tr><td>Claude Sonnet 4.6</td><td>85.02</td><td>12.16</td><td>2.82</td><td>93.60</td><td>3.59</td><td>2.82</td></tr><tr><td>Qwen3-VL-32B Thinking</td><td>47.50</td><td>44.43</td><td>8.07</td><td>86.94</td><td>7.55</td><td>5.51</td></tr></table>

Table 3: Transcription outcomes for full-image and crop-only recognition. All rates are percentages, with 781 images per entry.

The effect of removing the surrounding scene. Gray masking replaces all pixel outside the redbox rectangle with RGB (128, 128, 128) while keeping the target region, size, and position unchanged. Masking removes all surrounding information at once, including semantic cues, nearby text, and clutter. Pixel replay verifies all 781 inputs, and the four models provide 3,124 accepted responses under fixed decoding settings. RR falls by 18.57, 6.53, 9.60, and 20.10 percentage points for GPT, Gemini, Claude, and Qwen, and all paired 95% intervals exclude zero (Table 2). Literal accuracy rises in parallel, by 16.90, 7.17, 11.78, and 20.36 points. This increase reflects recovery rather than a shift to other errors: many canonical outputs become literal after masking, few become other errors. Correct transcription after changing only the background shows that the model’s output is influenced by the surrounding scene.

Cropping reduces rewriting and recovers many difficult cases. Crop-only recognition reduces RR to 2.18–7.55% and improves literal accuracy for all four models (Table 3). Each crop keeps the red outline and a small margin around the target, and is upsampled when needed. This produces 3,124 valid responses under the same request settings. Cropping changes not only the surrounding scene, but also the framing and target scale. Therefore, the lower RR reflects sensitivity to the input view rather than to scene content alone.

Controlled Fixed-Target-Patch Set Analysis. In this set, we test sensitivity to surrounding context in two ways while keeping the target pixels unchanged: retaining the original background or replacing it with a scene semantically unrelated to the target word. The 150-pair evaluation varies the surrounding without removing it, holding the target patch and its scale identical across the two scenes. Mean RR rises from 6.85% in identifier scenes to 11.15% in semantic scenes, while literal accuracy falls from 89.70% to 85.39% and mean O stays at 3.45%. Nine of eleven aliases show positive differences, and five have 95% paired-bootstrap intervals above zero before multiplicity correction. GPT-5.2 shows the largest effect, at +10.00 points. The result is consistent with the masking experiment, even though both conditions contain surrounding content. All evaluated aliases show increased rewriting in matching scenes compared with shuffled scenes, suggesting a consistent effect of the surrounding scene. However, the semantic and visual sources of this effect remain entangled.

## 4.4 TARGET BLUR INCREASES REWRITING

Section 4.3 keep the target text unchanged. In contrast, target-local blur weakens the target text while keeping the surrounding scene and red outline unchanged. For box height h, Gaussian standard deviations in clear image pixels are

$$
\sigma _ { \mathrm { m i l d } } = \operatorname* { m a x } ( 0 . 6 , 0 . 0 1 5 h ) , \quad \sigma _ { \mathrm { m o d e r a t e } } = \operatorname* { m a x } ( 1 . 2 , 0 . 0 3 0 h ) , \quad \sigma _ { \mathrm { s t r o n g } } = \operatorname* { m a x } ( 2 . 4 , 0 . 0 6 0 h ) .
$$

We conduct a moderate-blur evaluation on all 15 models and a three-level blur evaluation on four selected models. All 15 aliases rewrite more under moderate blur, and rewriting rises monotonically with severity in the four-model experiment. Every alias in Table 1 has a higher RR under moderate blur than on clear images. The RR increase remains similar with equal family weighting (+11.15 points) and after removing Qwen (+11.60 points). Literal accuracy also drops by 13.76 points (Appendix D). This suggests that the pattern is not driven by the model panel composition. In the four-model experiment, RR increases from moderate to strong blur for all models, reaching 43.02– 60.69% under strong blur (Table 14).

## 4.5 ANALYZING LEXICAL PRIORS IN REWRITING

The interventions above vary the input for a fixed set of targets. Rewriting also varies across targets under identical conditions, and the following analyses relate that variation to external lexical preference, string length, and the specific character edit. All three use clear image outputs and are exploratory associations on a constructed set, not causal claims.

Stronger lexical preference is associated with more rewriting. We use DistilGPT2 (Sanh et al., 2020) to estimate semantic-prior strength $A _ { i } = \ell ( c _ { i } )$ and the preference gap $D _ { i } = \ell ( c _ { i } ) - \ell ( y _ { i } )$

$$
\ell ( w ) = \frac { 1 } { K } \sum _ { t = 1 } ^ { K } \log P _ { \mathrm { L M } } \big ( b ( w ) \mid T _ { t } \big ) , \qquad A _ { i } = \ell ( c _ { i } ) , \quad D _ { i } = \ell ( c _ { i } ) - \ell ( y _ { i } ) .\tag{1}
$$

using average log-likelihood over four neutral prefixes without access to images or OCR outputs. Across 781 images and 15 aliases, D is positively correlated with mean RR $( \rho = 0 . 2 5 5 )$ . When the samples are divided into five groups by D, RR is 20.47% in the lowest group and 38.03% in the highest group. The association remains after controlling for other factors, including model, domain, string properties, and box geometry. The preference gap D matters more than the absolute score A of the conventional string, whose association is unclear. These use external lexical proxies rather than any VLM’s internal prior; Appendix B reports detailed results and additional sensitivity checks.

Longer conventional strings tend to be rewritten. RR rises from 14.27% for conventional strings of 4–5 characters to 45.02% for strings of at least 10, while literal accuracy falls from 79.80% to 49.79%. The positive association appears in all 15 models. Each additional character is associated with a 23% increase in the odds of rewriting. The association remains after controlling for lexical scores or perturbation type, suggesting that the length effect is not explained by either factor. Appendix B.3 specifies controls and full L/C/O results.

Lowercase l/i substitutions are rewritten at 59.78%, far above other letter-shape substitutions. Among letter-shape substitutions, RR is 59.78% for the 31 lowercase l↔i images versus 37.17% for the other 120. These two characters are visually similar and lexically substitutable, so the result is compatible with both perceptual misreading and lexical correction. (Appendix F).

## 4.6 POST-TRAINING MITIGATION PILOT

We compare SFT and SFT+GRPO using Qwen3-VL-4B-Instruct trained on 300 clear–blurred image pairs (600 images). Both methods use task specific prompts and are evaluated on 781 test pairs (1,562 images) with greedy decoding. The reward function is defined below:

$$
z ( \hat { y } ) = \bigg [ \lambda [ \hat { y } = y ^ { \star } ] + ( 1 - \lambda ) \operatorname* { m a x } \bigg ( 0 , 1 - \frac { d _ { \mathrm { e d i t } } ( \hat { y } , y ^ { \star } ) } { \vert y ^ { \star } \vert } \bigg ) \bigg ]\tag{2}
$$

where $\hat { y }$ is the prediction, $y ^ { \star }$ is the target, and $d _ { \mathrm { e d i t } }$ denotes Levenshtein distance (Levenshtein, 1966). We remove only leading and trailing whitespace before comparison and set $\lambda = 0 . 8$ for the exact-match term. The reward function $z ( \hat { y } )$ allows incorrect responses to receive partial credit based on their similarity to the target. The exact-match term rewards correct outputs, while the similarity term provides partial credit for incorrect outputs. GRPO uses the frozen SFT policy as its reference, with a KL coefficient of 0.04. As shown in Table 4, SFT+GRPO improves overall exact match (EM) by 0.53 percentage points and pair accuracy by 1.07 percentage points, based on unrounded results. Character error rate (CER) is the total character edit distance divided by the total target length; lower values indicate better performance. CER decreases from 5.80% with SFT to 5.64% with SFT+GRPO. Overall EM improves across all three seeds, showing a modest but consistent gain over SFT.

<table><tr><td>Method</td><td>Overall EM ↑</td><td>Clear EM ↑</td><td>Blurred EM ↑</td><td>Pair accuracy ↑</td><td>CER↓</td></tr><tr><td>Base Model</td><td>67.93</td><td>66.07</td><td>69.78</td><td>42.25</td><td> $7 . 7 7$ </td></tr><tr><td>SFT</td><td> $8 1 . 9 0 \pm 0 . 5 0$ </td><td> $8 3 . 3 5 \pm 0 . 6 8$ </td><td> $8 0 . 4 5 \pm 1 . 6 0$ </td><td> $6 6 . 5 8 \pm 0 . 5 9$ </td><td> $5 . 8 0 \pm 0 . 5 3$ </td></tr><tr><td>SFT+GRPO</td><td> ${ \mathbf { 8 2 . 4 4 \pm 0 . 3 3 } }$ </td><td> ${ \pm } 4 . 0 8 \pm 0 . 6 4$ </td><td> ${ \bf 8 0 . 7 9 \pm 1 . 2 2 }$ </td><td> ${ \bf 6 7 . 6 5 \pm 0 . 7 7 }$ </td><td> ${ \bf 5 . 6 4 \pm 0 . 3 4 }$ </td></tr></table>

Table 4: Results with 300 training pairs. Values are percentages, reported as mean ± sample standard deviation over three training seeds. Pair accuracy requires both predictions in a pair to be correct.

## 5 DISCUSSION

Surrounding context affects transcription fidelity. SceneFaith shows that transcription depends on more than the target characters. Gray masking improves literal accuracy while keeping the target pixels unchanged, and the fixed-patch experiment shows that changing the surrounding scene can also change the output. Cropping and lexical analyses further show that transcription is sensitive to image presentation, and lexical preference. The different behaviors of the Qwen Thinking variants also suggest that character fidelity should be studied within the same model family.

Models could balance character evidence and context. A reliable recognizer should preserve clear characters while using context only when the text is ambiguous. Crop recovery shows that some clear-image errors remain recoverable under a different view. Under blur, rewriting still increases after the surrounding scene is removed in three of four models, suggesting that both lexica completion and visual ambiguity may contribute. Future evaluations should therefore compare help ful, neutral, and misleading contexts around the same degraded target.

Rewriting and recovery should be measured together. A lower RR does not always mean better transcription: canonical outputs may become either correct literal outputs $( C \to L )$ or other errors $( C \to O )$ . Reporting L/C/O together separates these cases and keeps literal accuracy as the main measure of recovery. Section 4.6 reports a pilot study on mitigating rewriting and improving transcription faithfulness.

## 6 LIMITATIONS

SceneFaith focuses on generated scenes with deliberately perturbed text; generalization to natural images and broader OCR tasks remains to be tested. Scene interventions combine semantic and visual changes, while external lexical scores provide indirect evidence about internal mechanisms. Blurred targets lack complete readability validation, making information loss difficult to separate from lexical completion. Separate API runs may also introduce provider variation. The post-training pilot uses one model; further evaluation requires matched compute, independent test data, and additional models.

## 7 CONCLUSION

SceneFaith makes canonical substitution auditable through 781 images and Literal/Canonical/Other outcomes across 15 model aliases. Fixed-target scene removal reduces rewriting and improves literal accuracy in four models, while complementary diagnostics characterize sensitivity to scene substitutions, image views, degradation, and lexical preference. These findings establish peripheral-input sensitivity as a practical OCR reliability concern in the evaluated conditions. Evidence-calibrated recognition requires preserving clear text and recovering the printed string from ambiguous inputs with reliable context. SceneFaith provides a basis for testing transcription fidelity. Future work can extend it to test whether reliable context helps recover degraded text.

## AI USE STATEMENT

Generative models assisted with benchmark construction, experiment design, code and paper polishing. We checked the AI work and take responsibility for the final content.

## ETHICS STATEMENT

The study uses generated scenes and model outputs, with no personal data or human-subject annotations. Replacing visible text with familiar spellings may cause exact-match errors; retained raw responses make them traceable.

## REPRODUCIBILITY STATEMENT

Retained manifests, image hashes, prompts, parser, raw responses, and deterministic audits support the reported results. Subsequent appendices document the external scorer, paired-image statistics, fixed-patch, consensus, and blur/mask audits. The 781-image analyses use 15 aliases; the separate 150-pair diagnostic uses 11.

## REFERENCES

Aviad Aberdam, David Bensa¨ıd, Alona Golts, Roy Ganz, Oren Nuriel, Royee Tichauer, Shai Mazor, and Ron Litman. Clipter: Looking at the bigger picture in scene text recognition. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 21649–21660. IEEE, 2023.

Kesheng Chen, Yamin Hu, Qi Zhou, Zhenqian Zhu, and Wenjian Luo. Cdh-bench: A commonsensedriven hallucination benchmark for evaluating visual fidelity in vision-language models. In International Conference on Intelligent Computing, pp. 205–217. Springer, 2026.

Cheng Cui, Ting Sun, Manhui Lin, Tingquan Gao, Yubo Zhang, Jiaxuan Liu, Xueqing Wang, Zelun Zhang, Changda Zhou, Hongen Liu, Yue Zhang, Wenyu Lv, Kui Huang, Yichao Zhang, Jing Zhang, Jun Zhang, Yi Liu, Dianhai Yu, and Yanjun Ma. Paddleocr 3.0 technical report, 2025. URL https://arxiv.org/abs/2507.05595.

Shancheng Fang, Hongtao Xie, Yuxin Wang, Zhendong Mao, and Yongdong Zhang. Read like humans: Autonomous, bidirectional and iterative language modeling for scene text recognition. In 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 7094– 7103. IEEE, 2021.

Ling Fu, Zhebin Kuang, Jiajun Song, Mingxin Huang, Biao Yang, Yuzhe Li, Linghao Zhu, Qidi Luo, Xinyu Wang, Hao Lu, et al. Ocrbench v2: An improved benchmark for evaluating large multimodal models on visual text localization and reasoning. Advances in Neural Information Processing Systems, 38, 2026.

Tianrui Guan, Fuxiao Liu, Xiyang Wu, Ruiqi Xian, Zongxia Li, Xiaoyu Liu, Xijun Wang, Lichang Chen, Furong Huang, Yaser Yacoob, et al. Hallusionbench: an advanced diagnostic suite for entangled language hallucination and visual illusion in large vision-language models. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 14375–14385. IEEE, 2024.

Mingxin Huang, Yongxin Shi, Dezhi Peng, Songxuan Lai, Zecheng Xie, and Lianwen Jin. Ocrreasoning benchmark: Unveiling the true capabilities of mllms in complex text-rich image reasoning. In International Conference on Learning Representations, volume 2026, pp. 153760– 153779, 2026.

Qing Jiang, Jiapeng Wang, Dezhi Peng, Chongyu Liu, and Lianwen Jin. Revisiting scene text recognition: A data perspective. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 20486–20497. IEEE, 2023.

Antonia Karamolegkou, Nicolas Angleraud, Benoˆıt Sagot, and Thibault Clerice. Reading or guess-´ ing? visual grounding failures of vision-language models for ocr in ancient greek editions. arXiv preprint arXiv:2605.27750, 2026.

Gwang Gook Lee, Kenan Emir Ak, Jay Mohta, Yan Xu, and Dimitrios Dimitriadis. Do vlms read or rewrite? on transcription faithfulness in vision-language models. arXiv preprint arXiv:2607.21617, 2026.

VI Levenshtein. Binary coors capable or ‘correcting deletions, insertions, and reversals. In Soviet physics-doklady, volume 10, pp. 707–710, 1966.

Gengluo Li, Xingyu Wan, Shangpin Peng, Weinong Wang, Hao Feng, Yongkun Du, Binghong Wu, Zheng Ruan, Zhiqiong Lu, Liang Wu, Pengyuan Lyu, Huawen Shen, Zibin Lin, Shijing Hu, Jieneng Yang, Hongbing Wen, Guanghua Yu, Hong Liu, Bochao Wang, Can Ma, Han Hu, Chengquan Zhang, and Yu Zhou. HunyuanOCR-1.5: Making lightweight OCR VLMs faster and better. arXiv preprint arXiv:2607.04884, 2026.

Yunhao Liang, Ruixuan Ying, Bo Li, Hong Li, Kai Yan, Qingwen Li, Min Yang, Okamoto Satoshi, Zhe Cui, and Shiwen Ni. Visual merit or linguistic crutch? a close look at deepseek-ocr. arXiv preprint arXiv:2601.03714, 2026.

Yuliang Liu, Zhang Li, Mingxin Huang, Biao Yang, Wenwen Yu, Chunyuan Li, Xu-Cheng Yin, Cheng-Lin Liu, Lianwen Jin, and Xiang Bai. Ocrbench: on the hidden mystery of ocr in large multimodal models. Science China Information Sciences, 67(12):220102, 2024.

Byeonghu Na, Yoonsik Kim, and Sungrae Park. Multi-modal text recognition networks: Interactive enhancements between visual and semantic features. In European conference on computer vision, pp. 446–463. Springer, 2022.

Junbo Niu, Zheng Liu, Zhuangcheng Gu, Bin Wang, Linke Ouyang, Zhiyuan Zhao, Tao Chu, Tianyao He, Fan Wu, Qintong Zhang, Zhenjiang Jin, Guang Liang, Rui Zhang, Wenzheng Zhang, Yuan Qu, Zhifei Ren, Yuefeng Sun, Yuanhong Zheng, Dongsheng Ma, Zirui Tang, Boyu Niu, Ziyang Miao, Hejun Dong, Siyi Qian, Junyuan Zhang, Jingzhou Chen, Fangdong Wang, Xiaomeng Zhao, Liqun Wei, Wei Li, Shasha Wang, Ruiliang Xu, Yuanyuan Cao, Lu Chen, Qianqian Wu, Huaiyu Gu, Lindong Lu, Keming Wang, Dechen Lin, Guanlin Shen, Xuanhe Zhou, Linfeng Zhang, Yuhang Zang, Xiaoyi Dong, Jiaqi Wang, Bo Zhang, Lei Bai, Pei Chu, Weijia Li, Jiang Wu, Lijun Wu, Zhenxiang Li, Guangyu Wang, Zhongying Tu, Chao Xu, Kai Chen, Yu Qiao, Bowen Zhou, Dahua Lin, Wentao Zhang, and Conghui He. Mineru2.5: A decoupled vision-language model for efficient high-resolution document parsing, 2025. URL https://arxiv.org/abs/2509.22186.

Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf. Distilbert, a distilled version of bert: smaller, faster, cheaper and lighter, 2020. URL https://arxiv.org/abs/1910. 01108.

Jin Seong, Wencke Liermann, Minho Kim, Jong-hun Shin, and Soojong Lim. When vlms’ fix’students: Identifying and penalizing over-correction in the evaluation of multi-line handwritten math ocr. arXiv preprint arXiv:2604.22774, 2026.

Yan Shu, Hangui Lin, Yexin Liu, Yan Zhang, Gangyan Zeng, Yan Li, Yu Zhou, Ser Nam Lim, Harry Yang, and Nicu Sebe. When semantics mislead vision: Mitigating large multimodal models hal lucinations in scene text spotting and understanding. Advances in Neural Information Processing Systems, 38:68952–68980, 2026.

Yuxin Wang, Hongtao Xie, Shancheng Fang, Jing Wang, Shenggao Zhu, and Yongdong Zhang. From two to one: A new scene text recognizer with visual language modeling network. In 2021 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 14174–14183. IEEE, 2021.

Haoran Wei, Yaofeng Sun, and Yukun Li. Deepseek-ocr 2: Visual causal flow, 2026. URL https: //arxiv.org/abs/2601.20552.

Can Zhang, Ziheng Wu, Zhenghao Chen, Yufei Zhan, Yifan Li, Zhao Zhang, Xian Wang, Minghui Qiu, et al. Seeing is believing? mitigating ocr hallucinations in multimodal large language models. Advances in Neural Information Processing Systems, 38:74230–74248, 2026a.

Xinyun Zhang, Binwu Zhu, Xufeng Yao, Qi Sun, Ruiyu Li, and Bei Yu. Context-based contrastive learning for scene text recognition. In Proceedings of the AAAI conference on artificial intelligence, volume 36, pp. 3353–3361, 2022.

Zelun Zhang, Hongen Liu, Suyin Liang, Yubo Zhang, Yiqing Xiang, Jiaxuan Liu, Ting Sun, Manhui Lin, Yue Zhang, Changda Zhou, Tingquan Gao, Cheng Cui, Yi Liu, Dianhai Yu, and Yanjun Ma. Paddleocr-vl-1.6: Expanding the frontier of document parsing with under-optimized region refinement and progressive post-training, 2026b. URL https://arxiv.org/abs/2606. 03264.

Shuai Zhao, Ruijie Quan, Linchao Zhu, and Yi Yang. Clip4str: A simple baseline for scene text recognition with pre-trained vision-language model. IEEE transactions on Image Processing, 33: 6893–6904, 2024.

## A CONSTRUCTION INVENTORY AND AUDIT

The 781 PNGs retain their native resolution, unique IDs, printed and canonical strings, and file and RGB hashes. The printed string is the transcription gold; the canonical string is used only for analysis. Model-assisted screening, blind target readings, and visual review excluded unclear or missing targets, unsuitable scenes, and answer leakage. Hash audits verified all 781 crops and 2,343 blur transforms. These checks confirm pixel integrity, not readability.

<table><tr><td>Category</td><td>Images</td><td>Category</td><td>Images</td><td>Category</td><td>Images</td></tr><tr><td>Animal</td><td>109</td><td>Electronics</td><td>14</td><td>Profession</td><td>79</td></tr><tr><td>Building</td><td>20</td><td>Home</td><td>27</td><td>Recipe</td><td>91</td></tr><tr><td>City</td><td>64</td><td>Landform</td><td>5</td><td>Sport</td><td>44</td></tr><tr><td>Clothing</td><td>26</td><td>Medicine</td><td>37</td><td>Tool</td><td>14</td></tr><tr><td>Command</td><td>32</td><td>Music</td><td>42</td><td>Vehicle</td><td>34</td></tr><tr><td>Country</td><td>56</td><td>Plant</td><td>87</td><td>Total</td><td>781</td></tr></table>

Table 5: Category inventory of the complete 781-image benchmark. All reported experiments use this aggregate collection.

## B LEXICAL PRIOR ANALYSIS

We examine associations between external language-model scores and 11,715 clear-image outputs from 15 models and 781 images. The score definition was fixed before linking scores to OCR outcomes; the analyses are exploratory.

## B.1 SCORING AND ESTIMATION

Let $y _ { i }$ be the printed string and $c _ { i }$ its conventional spelling. We use DistilGPT2 to score each string after four fixed neutral prefixes:

$$
\ell ( w ) = \frac { 1 } { K } \sum _ { t = 1 } ^ { K } \log P _ { \mathrm { L M } } \big ( b ( w ) \mid T _ { t } \big ) , \qquad A _ { i } = \ell ( c _ { i } ) , \quad D _ { i } = \ell ( c _ { i } ) - \ell ( y _ { i } ) .\tag{3}
$$

where $b ( w )$ adds a leading space and a final period. Thus, $A _ { i }$ measures the conventional string’s likelihood, while $D _ { i }$ measures its advantage over the printed string.

Correlations and quintiles use each image’s mean rewriting rate across the 15 models. Regressions use individual outputs and adjust for model, domain, string length, edit distance, nonalphabetic characters, and target-box geometry. Intervals account for repeated outputs from the same image.

## B.2 ASSOCIATIONS AND SENSITIVITY

The preference gap D correlates with rewriting $( \rho = 0 . 2 5 5 , 9 5 \% \mathrm { C I } \left[ 0 . 1 8 9 , 0 . 3 2 1 \right] )$ . Rewriting rises from 20.47% in the lowest quintile to 38.03% in the highest, a difference of 17.57 points ([11.86, 23.23]). In the joint adjusted model, D remains associated with rewriting, whereas A does not (Tables 6 and 7). The association for $D$ persists across the reported scoring and string-subset checks. These associations involve external scores and do not identify a VLM-internal lexical mech anism.

## B.3 STRING LENGTH

Rewriting rises from 14.27% for conventional strings of 4–5 characters to 45.02% for strings of at least 10 characters (Table 8). The adjusted odds ratio is 1.230 per additional character ([1.157, 1.309]), or 1.127 ([1.039, 1.223]) after adding A and D.

<table><tr><td>Analysis</td><td>Words</td><td>Adjusted OR</td><td>95% CI</td></tr><tr><td>Primary: period-terminated gap</td><td>781</td><td>1.352</td><td>[1.197, 1.527]</td></tr><tr><td>No-period gap</td><td>781</td><td>1.377</td><td>[1.214, 1.562]</td></tr><tr><td>No-period gap, per character</td><td>781</td><td>1.341</td><td>[1.195, 1.506]</td></tr><tr><td>Period gap, per character</td><td>781</td><td>1.312</td><td>[1.169, 1.473]</td></tr><tr><td>Equal-length candidates</td><td>398</td><td>1.387</td><td>[1.190, 1.617]</td></tr><tr><td>Alphabetic printed strings</td><td>704</td><td>1.340</td><td>[1.183, 1.519]</td></tr><tr><td>Template 1 only</td><td>781</td><td>1.341</td><td>[1.192, 1.509]</td></tr><tr><td>Template 2 only</td><td>781</td><td>1.321</td><td>[1.171, 1.490]</td></tr><tr><td>Template 3 only</td><td>781</td><td>1.303</td><td>[1.159, 1.465]</td></tr><tr><td>Template 4 only</td><td>781</td><td>1.348</td><td>[1.195, 1.520]</td></tr></table>

Table 6: Exploratory lexical-preference analysis and 9 sensitivity checks.
<table><tr><td>Analysis</td><td>Words</td><td>Strength A: OR [95% CI]</td><td>Gap D: OR [95% CI]</td></tr><tr><td>All: separate predictors</td><td>781</td><td>1.202 [1.081, 1.337]</td><td>1.352 [1.197, 1.527]</td></tr><tr><td>All: joint predictors</td><td>781</td><td>1.017 [0.895, 1.157]</td><td>1.339 [1.155, 1.551]</td></tr><tr><td>Alphabetic only</td><td>704</td><td>1.225 [1.099, 1.364]</td><td>1.340 [1.183, 1.519]</td></tr><tr><td>Equal length</td><td>398</td><td>1.179 [1.008, 1.379]</td><td>1.387 [1.190, 1.617]</td></tr></table>

Table 7: DistilGPT2 strength A and preference gap D on the 781-image benchmark.

## C SUPPLEMENTARY STATISTICS FOR THE CLEAR-IMAGE EVALUATION

Table 9 reports rewriting rates (RR) for 15 aliases on 781 clear images. Domain-macro RR weights 17 categories equally; the model mean weights aliases equally. The 95% confidence interval reflects uncertainty due to the limited number of images. Appendix D tests family weighting.

## D SENSITIVITY TO MODEL-FAMILY COMPOSITION

We average aliases within each of seven model families, then weight families equally. Qwen has six aliases, Gemini three, GPT two, and Claude, GLM, InternVL, and Kimi one each. All comparisons use the same 781 images with accepted clear and moderate-blur outputs for every alias. Pointwise 95% intervals use 10,000 paired-image bootstrap resamples; aliases and families stay fixed.

In all seven leave-one-family-out analyses, moderate blur raises C by 9.97–12.66 points and lowers L by 12.07–14.59.

## E QWEN INSTRUCT–THINKING COMPARISONS ANALYSIS

Paired comparisons. Each Instruct–Thinking pair covers the same 781 images at one of three Qwen sizes. Pointwise 95% intervals use 20,000 paired-image bootstrap resamples; exact McNemar tests compare aliases within each size, with Holm correction across sizes. Thinking has lower literal accuracy at 8B and 32B, but no clear change at 235B-A22B. The 235B-A22B model uses a mixtureof-experts (MoE) architecture. The respective accuracy changes are −13.70 ([−17.67, −9.86]), −13.44 ([−16.90, −9.86]), and +1.15 ([−1.79, 4.10]) points. Scoring whole answers gives similar RR gaps: +18.95, +17.67, and +0.38 points. These are comparisons between distinct served aliases, not a thinking toggle within one model.

<table><tr><td>Characters in ci</td><td>Images</td><td>L (%)</td><td>C / RR (%)</td><td>O (%)</td><td>RR 95% CI</td></tr><tr><td>4-5</td><td>199</td><td>79.80</td><td>14.27</td><td>5.93</td><td>[11.99, 16.68]</td></tr><tr><td>6-7</td><td>302</td><td>69.16</td><td>24.48</td><td>6.36</td><td>[21.88, 27.17]</td></tr><tr><td>8-9</td><td>199</td><td>60.13</td><td>33.67</td><td>6.20</td><td>[30.05, 37.39]</td></tr><tr><td>≥ 10</td><td>81</td><td>49.79</td><td>45.02</td><td>5.19</td><td>[39.26, 50.95]</td></tr></table>

Table 8: Longer conventional strings accompany more canonical substitutions.

<table><tr><td rowspan="2">Model alias</td><td colspan="3">Rewriting rate (%)</td></tr><tr><td>Overall ↓</td><td>95% Wilson CI</td><td>Category macro ↓</td></tr><tr><td>Gemini 3.1 Flash</td><td>8.45</td><td>[6.70, 10.61]</td><td>8.33</td></tr><tr><td>Gemini 3.5 Flash</td><td>8.96</td><td>[7.16, 11.17]</td><td>7.43</td></tr><tr><td>Gemini 3 Flash</td><td>11.01</td><td>[9.00, 13.40]</td><td>9.34</td></tr><tr><td>Claude Sonnet 4.6</td><td>12.16</td><td>[10.05, 14.64]</td><td>11.27</td></tr><tr><td>GPT-5.5</td><td>19.46</td><td>[16.84, 22.39]</td><td>18.33</td></tr><tr><td>Qwen3-VL-8B Instruct</td><td>19.46</td><td>[16.84, 22.39]</td><td>19.06</td></tr><tr><td>GLM-4.6V</td><td>21.25</td><td>[18.53, 24.26]</td><td>19.64</td></tr><tr><td>Qwen3-VL-32B Instruct</td><td>27.53</td><td>[24.51, 30.77]</td><td>26.90</td></tr><tr><td>GPT-5.2</td><td>29.58</td><td>[26.48, 32.87]</td><td>27.80</td></tr><tr><td>Kimi K2.5</td><td>31.88</td><td>[28.71, 35.23]</td><td>29.51</td></tr><tr><td>Qwen3-VL-235B-A22B Thinking</td><td>32.27</td><td>[29.08, 35.62]</td><td>31.40</td></tr><tr><td>Qwen3-VL-235B-A22B Instruct</td><td>32.39</td><td>[29.21, 35.76]</td><td>34.16</td></tr><tr><td>Qwen3-VL-8B Thinking</td><td>37.90</td><td>[34.56, 41.35]</td><td>38.82</td></tr><tr><td>Qwen3-VL-32B Thinking</td><td>44.43</td><td>[40.98, 47.93]</td><td>42.17</td></tr><tr><td>InternVL3-38B</td><td>58.51</td><td>[55.03, 61.92]</td><td>58.36</td></tr><tr><td>Model mean</td><td>26.35</td><td>一</td><td>25.50</td></tr></table>

Table 9: Supplementary statistics for clear-image rewriting.

<table><tr><td rowspan="2">Weighting</td><td colspan="3">Clear (%)</td><td colspan="3">Moderate (%)</td></tr><tr><td>L</td><td>C</td><td>0</td><td>L</td><td>C</td><td>0</td></tr><tr><td>Equal alias weighting (15 aliases)</td><td>67.55</td><td>26.36</td><td>6.09</td><td>54.94</td><td>36.95</td><td>8.11</td></tr><tr><td>Equal family weighting (7 families)</td><td>68.03</td><td>27.16</td><td>4.81</td><td>54.65</td><td>38.31</td><td>7.05</td></tr><tr><td>Excluding Qwen: equal alias weighting (9 aliases)</td><td>74.52</td><td>22.36</td><td>3.12</td><td>60.95</td><td>34.40</td><td>4.64</td></tr><tr><td>Excluding Qwen: equal family weighting (6 families)</td><td>69.86</td><td>26.29</td><td>3.85</td><td>56.10</td><td>37.90</td><td>6.00</td></tr><tr><td>Paired change (pp)</td><td colspan="2">∆L</td><td colspan="2">∆C</td><td colspan="2">∆0</td></tr><tr><td>Equal alias weighting (15 aliases)</td><td colspan="2">-12.61 [-13.82, -11.44] -13.39</td><td colspan="2">+10.59 [+9.44, +11.74]</td><td colspan="2">+2.02 [+1.14, +2.94]</td></tr><tr><td>Equal family weighting (7 families)</td><td colspan="2">[-14.72, -12.08] -13.56</td><td colspan="2">+11.15 [+9.87, +12.43]</td><td colspan="2">+2.24 [+1.32, +3.22]</td></tr><tr><td>Excluding Qwen: equal alias weighting (9 aliases)</td><td colspan="2">[-14.89, -12.25][+10.81, +13.29]</td><td colspan="2">+12.04</td><td colspan="2">+1.52 [+0.81, +2.28]</td></tr><tr><td>Excluding Qwen: equal family weighting (6 families)</td><td colspan="2">-13.76</td><td colspan="2">+11.60</td><td colspan="2">+2.15 [-15.16,-12.36] [+10.25, +12.96] [+1.24,+3.13]</td></tr></table>

Table 10: Sensitivity to model-family weighting. Brackets show 95% paired-image bootstrap confidence intervals for changes from clear to moderate blur.

<table><tr><td>Size</td><td>Variant</td><td>Literal</td><td>Canonical</td><td>Other</td></tr><tr><td>8B</td><td>Instruct</td><td>486 (62.23)</td><td>152 (19.46)</td><td>143 (18.31)</td></tr><tr><td>8B</td><td>Thinking</td><td>379 (48.53)</td><td>296 (37.90)</td><td>106 (13.57)</td></tr><tr><td>32B</td><td>Instruct</td><td>476 (60.95)</td><td>215 (27.53)</td><td>90 (11.52)</td></tr><tr><td>32B</td><td>Thinking</td><td>371 (47.50)</td><td>347 (44.43)</td><td>63 (8.07)</td></tr><tr><td>235B-A22B</td><td>Instruct</td><td>478 (61.20)</td><td>253 (32.39)</td><td>50 (6.40)</td></tr><tr><td>235B-A22B</td><td>Thinking</td><td>487 (62.36)</td><td>252 (32.27)</td><td>42 (5.38)</td></tr></table>

Table 11: Qwen clear-image outcome counts (percentages), with $N = 7 8 1$ for every row.

<table><tr><td>Size</td><td>Instruct outcome</td><td>Thinking L</td><td>Thinking C</td><td>Thinking O</td></tr><tr><td>8B</td><td>L</td><td>304</td><td>136</td><td>46</td></tr><tr><td>8B</td><td>C</td><td>16</td><td>128</td><td>8</td></tr><tr><td>8B</td><td>0</td><td>59</td><td>32</td><td>52</td></tr><tr><td>32B</td><td>L</td><td>317</td><td>134</td><td>25</td></tr><tr><td>32B</td><td>C</td><td>23</td><td>183</td><td>9</td></tr><tr><td>32B</td><td>0</td><td>31</td><td>30</td><td>29</td></tr><tr><td>235B-A22B</td><td>L</td><td>412</td><td>49</td><td>17</td></tr><tr><td>235B-A22B</td><td>C</td><td>54</td><td>190</td><td>9</td></tr><tr><td>235B-A22B</td><td>0</td><td>21</td><td>13</td><td>16</td></tr></table>

Table 12: Matched-image outcome counts from Instruct (row) to Thinking (column).

Reasoning text. All 2,343 Thinking outputs retain reasoning text; Instruct outputs do not. Exact searches find both the printed and canonical strings in seven canonical-answer traces (1, 1, and 5 by size). An 8B trace changes “uher” to “usher”; a 32B trace identifies “mont-evideo” as a likely typo but copies it. These traces suggest that the model may sometimes recognize the printed text but still produce an incorrect final answer.

Exploratory checks. Moderate blur changes the Thinking–Instruct RR gap by +2.18 ([−1.54, 6.02]), +0.77 ([−2.94, 4.48]), and +1.66 ([−1.79, 5.12]) points by size. The intervals leave blur amplification unresolved. Regressions against external lexical gap D show no positive association at 8B or 32B; the 235B slope is −3.56 points per SD $( [ - 6 . 2 9 , - 0 . 9 4 ] )$ ). These exploratory intervals are unadjusted for multiple checks, and the associations do not identify a lexical mechanism.

## F CHARACTER PERTURBATIONS AND CANONICAL REWRITING

We classified the printed-to-conventional edits in all 781 images into eight groups, independently of model outputs (Table 13). Fixed mappings define letter-shape substitutions (l/i, $\mathrm { e / c , u / v , n / m , h / b , a / o }$ , in both directions), digit/symbol substitutions $( \circ , \mathrm { i } , \mathsf { \ i } , \mathsf { \ i } , \mathsf { s } , \mathsf { a } , \mathsf { g } ,$ ,b to $0 , 1 , 1 , 5 , \Theta , 9 , 6 )$ , and split/merge edits $\left( \mathrm { m } / \mathrm { r n } , \mathrm { w } / \mathrm { v v } , \mathrm { c } 1 / \mathrm { d } \right)$ . The other groups are duplication, deletion, insertion, transposition, and other substitution; inserting an adjacent identical character counts as duplication.

For 31 lowercase l↔i images, RR exceeds that of the other 120 letter-shape images by 22.62 points ([12.59, 32.90]). The post hoc logistic model adjusts for model, domain, conventional length, box geometry, and external lexical scores A and D (Appendix B). Its image-clustered interval gives OR 3.39 ([2.16, 5.30]).

## G BLUR STATISTICS

Tables 14 and table 15 report rewriting rates across blur levels and paired changes under moderate blur. Pointwise 95% intervals use 10,000 paired-image bootstrap resamples and describe image variation in the saved calls.

<table><tr><td>Perturbation</td><td>n</td><td>Acc.</td><td>RR Other</td><td></td><td>RR 95% CI</td></tr><tr><td>Letter-shape substitution</td><td>151</td><td>53.02</td><td>41.81</td><td>5.17</td><td>[37.31, 46.23]</td></tr><tr><td>Character split / merge</td><td>26</td><td>50.26</td><td>38.72</td><td>11.03</td><td>[28.72, 48.72]</td></tr><tr><td>Character duplication</td><td>149</td><td>61.21</td><td>33.47</td><td>5.32</td><td>[29.71, 37.23]</td></tr><tr><td>Digit / symbol substitution</td><td>61</td><td>66.67</td><td>27.98</td><td>5.36</td><td>[20.98, 35.30]</td></tr><tr><td>Character deletion</td><td>172</td><td>77.09</td><td>18.18</td><td>4.73</td><td>[15.23, 21.28]</td></tr><tr><td>Character insertion</td><td>36</td><td>77.04</td><td>17.59</td><td>5.37</td><td>[10.56, 25.56]</td></tr><tr><td>Adjacent transposition</td><td>141</td><td>75.37</td><td>15.93</td><td>8.70</td><td>[13.14, 18.82]</td></tr><tr><td>Other letter substitution</td><td>45</td><td>80.15</td><td>12.44</td><td>7.41</td><td>[8.15, 17.19]</td></tr><tr><td>Selected character substitutions: conventional → printed</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>1→i</td><td>10</td><td>37.33</td><td>60.00</td><td>2.67</td><td>[40.00, 79.33]</td></tr><tr><td>i→1</td><td>21</td><td>36.51</td><td>59.68</td><td>3.81</td><td>[50.48, 68.58]</td></tr><tr><td>1→1</td><td>7</td><td>41.90</td><td>56.19</td><td>1.90</td><td>[36.19, 76.21]</td></tr><tr><td>i→1</td><td>16</td><td>61.67</td><td>27.50</td><td>10.83</td><td>[15.42, 41.67]</td></tr></table>

Table 13: Clear-image outcomes by recorded character perturbation.
<table><tr><td>Model alias</td><td>Clear</td><td>Mild</td><td>Moderate</td><td>Strong</td></tr><tr><td>GPT-5.2</td><td>29.58</td><td>30.09</td><td>45.33</td><td>60.69</td></tr><tr><td>Gemini 3.5 Flash</td><td>8.96</td><td>10.50</td><td>18.44</td><td>54.67</td></tr><tr><td>Claude Sonnet 4.6</td><td>12.16</td><td>10.37</td><td>22.66</td><td>43.02</td></tr><tr><td>Qwen3-VL-32B Thinking</td><td>44.43</td><td>47.25</td><td>51.47</td><td>59.72</td></tr></table>

Table 14: Rewriting rates (%) across blur severity.

<table><tr><td>Model alias</td><td>Clear</td><td>Moderate</td><td>∆RR</td><td>95% CI</td></tr><tr><td>Gemini 3.1 Flash Lite</td><td>8.45</td><td>20.74</td><td>+12.29</td><td>[+9.99, +14.72]</td></tr><tr><td>Gemini 3.5 Flash</td><td>8.96</td><td>18.44</td><td>+9.48</td><td>[+7.04, +11.91]</td></tr><tr><td>Gemini 3 Flash Preview</td><td>11.01</td><td>19.97</td><td>+8.96</td><td>[+6.53, +11.40]</td></tr><tr><td>Claude Sonnet 4.6</td><td>12.16</td><td>22.66</td><td>+10.50</td><td>[+7.43, +13.57]</td></tr><tr><td>GPT-5.5</td><td>19.46</td><td>40.20</td><td>+20.74</td><td>[+17.41, +24.07]</td></tr><tr><td>Qwen3-VL-8B I.</td><td>19.46</td><td>26.76</td><td>+7.30</td><td>[+4.87, +9.86]</td></tr><tr><td>GLM-4.6V</td><td>21.25</td><td>32.01</td><td>+10.76</td><td>[+8.32, +13.32]</td></tr><tr><td>Qwen3-VL-32B I.</td><td>27.53</td><td>33.80</td><td>+6.27</td><td>[+3.59, +9.09]</td></tr><tr><td>GPT-5.2</td><td>29.58</td><td>45.33</td><td>+15.75</td><td>[+12.80, +18.82]</td></tr><tr><td>Kimi K2.5</td><td>31.88</td><td>49.68</td><td>+17.80</td><td>[+14.60, +21.13]</td></tr><tr><td>Qwen3-VL-235B T.</td><td>32.27</td><td>43.28</td><td>+11.01</td><td>[+8.07, +14.08]</td></tr><tr><td>Qwen3-VL-235B I.</td><td>32.39</td><td>41.74</td><td>+9.35</td><td>[+6.79, +12.04]</td></tr><tr><td>Qwen3-VL-8B T.</td><td>37.90</td><td>47.38</td><td>+9.48</td><td>[+6.27, +12.68]</td></tr><tr><td>Qwen3-VL-32B T.</td><td>44.43</td><td>51.47</td><td>+7.04</td><td>[+3.97, +10.24]</td></tr><tr><td>InternVL3-38B</td><td>58.46</td><td>60.51</td><td>+2.05</td><td>[-0.38, +4.49]</td></tr></table>

Table 15: Matched moderate-blur changes on the SceneFaith benchmark.

## H FIXED-TARGET-PATCH DIAGNOSTIC ON 150 PAIRS

We constructed 150 image pairs to test how surrounding scenes affect rewriting while keeping target pixels fixed. Each pair places an identical target patch, including its local background, at the same position and scale on a 1024 × 1024 canvas. Scene A supports the conventional word, while scene B presents the printed string as an identifier in a context that does not encourage rewriting. The printed gold and prompts are identical within each pair. All 11 models have complete responses for all 150 pairs. RR differences are computed within matched pairs, with pointwise 95% paired-image bootstrap intervals.

Across the 11 models, mean RR increases from 6.85% in B to 11.15% in A, a rise of 4.30 percentage points. Nine models show positive differences, and five have intervals entirely above zero before adjustment for multiple comparisons (Table 16). These results show that rewriting responds to the surrounding scene even when the target pixels remain unchanged.
<table><tr><td>Model alias</td><td>A: L</td><td>A: C</td><td>B: L</td><td>B: C</td><td>∆</td><td>95% CI</td></tr><tr><td>GPT-5.2</td><td>80.67</td><td>17.33</td><td>89.33</td><td>7.33</td><td>+10.00</td><td>[4.00, 16.00]</td></tr><tr><td>GPT-5.5</td><td>80.67</td><td>16.00</td><td>90.00</td><td>8.00</td><td>+8.00</td><td>[2.00, 14.67]</td></tr><tr><td>Claude Sonnet 4.6</td><td>99.33</td><td>0.67</td><td>98.00</td><td>0.67</td><td>+0.00</td><td>[−2.00, 2.00]</td></tr><tr><td>Gemini 3.1 Flash Lite</td><td>98.67</td><td>1.33</td><td>98.67</td><td>1.33</td><td>+0.00</td><td>[−2.00, 2.00]</td></tr><tr><td>Gemini 3.5 Flash</td><td>96.67</td><td>2.67</td><td>96.00</td><td>2.00</td><td>+0.67</td><td>[−1.33, 3.33]</td></tr><tr><td>Qwen3-VL-8B-I</td><td>88.00</td><td>8.00</td><td>92.67</td><td>4.00</td><td>+4.00</td><td>[0.67, 8.00]</td></tr><tr><td>Qwen3-VL-8B-T</td><td>76.67</td><td>15.33</td><td>79.33</td><td>12.00</td><td>+3.33</td><td>[−1.33, 8.00]</td></tr><tr><td>Qwen3-VL-32B-I</td><td>82.00</td><td>12.67</td><td>86.00</td><td>10.67</td><td>+2.00</td><td>[−2.00, 6.02]</td></tr><tr><td>Qwen3-VL-32B-T</td><td>71.33</td><td>22.00</td><td>77.33</td><td>15.33</td><td>+6.67</td><td>[2.00, 12.00]</td></tr><tr><td>Qwen3-VL-235B-A22B-I</td><td>79.33</td><td>16.67</td><td>90.00</td><td>7.33</td><td>+9.33</td><td>[4.67, 14.67]</td></tr><tr><td>Qwen3-VL-235B-A22B-T</td><td>86.00</td><td>10.00</td><td>89.33</td><td>6.67</td><td>+3.33</td><td>[−0.67, 7.33]</td></tr></table>

Table 16: Fixed-target-patch comparison on 150 matched pairs per alias.

We further examine background sensitivity using a gray-background control M for all 11 models and a same-domain shuffled scene S for eight models (Table 17). Since A and B produce identical gray inputs, each target and model shares a single M response. All eight models evaluated on S have higher RR in A than in S, with intervals entirely above zero for both 235B-A22B variants. The gray control also reveals that removing the scene can increase rewriting: RR reaches 25.33% for 8B Instruct and 36.67% for 235B-A22B Instruct, exceeding both corresponding full-scene rates.
<table><tr><td>Model alias</td><td>A: 0</td><td>B:0</td><td>M:L</td><td>M: C</td><td>S: C</td><td>∆A-S</td><td>95% CI</td></tr><tr><td>GPT-5.2</td><td>2.00</td><td>3.33</td><td>90.00</td><td>9.33</td><td>12.00</td><td>+5.33</td><td>[−1.33, 12.00]</td></tr><tr><td>GPT-5.5</td><td>3.33</td><td>2.00</td><td>88.67</td><td>10.00</td><td>10.00</td><td>+6.00</td><td>[−0.67, 12.67]</td></tr><tr><td>Claude Sonnet 4.6</td><td>0.00</td><td>1.33</td><td>99.33</td><td>0.67</td><td>0.00</td><td>+0.67</td><td>[0.00, 2.00]</td></tr><tr><td>Gemini 3.1 Flash Lite</td><td>0.00</td><td>0.00</td><td>98.00</td><td>1.33</td><td></td><td></td><td></td></tr><tr><td>Gemini 3.5 Flash</td><td>0.67</td><td>2.00</td><td>95.33</td><td>1.33</td><td>2.00</td><td>+0.67</td><td>[0.00, 2.00]</td></tr><tr><td>Qwen3-VL-8B-I</td><td>4.00</td><td>3.33</td><td>66.67</td><td>25.33</td><td>7.33</td><td>+0.67</td><td>[-3.35, 4.67]</td></tr><tr><td>Qwen3-VL-8B-T</td><td>8.00</td><td>8.67</td><td>81.33</td><td>12.00</td><td></td><td></td><td></td></tr><tr><td>Qwen3-VL-32B-I</td><td>5.33</td><td>3.33</td><td>92.00</td><td>6.00</td><td>10.00</td><td>+2.67</td><td>[-1.33, 6.67]</td></tr><tr><td>Qwen3-VL-32B-T</td><td>6.67</td><td>7.33</td><td>88.67</td><td>8.00</td><td></td><td></td><td></td></tr><tr><td>Qwen3-VL-235B-A22B-I</td><td>4.00</td><td>2.67</td><td>58.67</td><td>36.67</td><td>10.00</td><td>+6.67</td><td>[1.33, 12.00]</td></tr><tr><td>Qwen3-VL-235B-A22B-T</td><td>4.00</td><td>4.00</td><td>88.00</td><td>6.00</td><td>6.00</td><td>+4.00</td><td>[1.33, 7.33]</td></tr></table>

Table 17: Additional outcomes on the same 150 targets.

Together, these comparisons establish sensitivity to surrounding visual input under fixed target pixels. The A–B comparison captures the combined influence of scene meaning, identifier role, and visual appearance, while the gray control tests the effect of removing the surrounding scene.

## I CONSENSUS FILTERING AND RECOVERABLE SHARED ERRORS

The archived three-model panel used GPT-5.2, Claude Sonnet 4.6, and gemini-3-flash. Exact agreement accepted 474 of 781 images, including 18 nonliteral answers (16 C, two O; Table 18). Adding Qwen3-VL-32B Thinking kept all 18 errors but removed 165 literal answers.

## J CROSSING TARGET BLUR WITH PERIPHERAL SCENE REMOVAL

Four models have clear/moderate × full/gray responses for the same 781 images (12,496 outputs). Gray replaces pixels outside the red box with RGB (128, 128, 128); moderate blur uses the fixed recipe in Section 4.4. At each clarity level, full and gray images retain identical target pixels, position, and scale. Table 19 reports all L/C/O rates.

<table><tr><td>Agreement rule</td><td>Accepted n</td><td>Coverage (%)</td><td>L/C/O</td><td>Error (%)</td></tr><tr><td>Three models, exact</td><td>474</td><td>60.69</td><td>456/16/2</td><td>3.80</td></tr><tr><td>Four models, exact</td><td>309</td><td>39.56</td><td>291/16/2</td><td>5.83</td></tr><tr><td>Three models, normalized</td><td>493</td><td>63.12</td><td>465/26/2</td><td>5.68</td></tr><tr><td>Four models, normalized</td><td>327</td><td>41.87</td><td>299/26/2</td><td>8.56</td></tr></table>

Table 18: Consensus filtering on the archived 781-image evaluation.

<table><tr><td rowspan="2">Model alias</td><td colspan="3">Full clear</td><td colspan="3">Full moderate</td><td colspan="3">Gray clear</td><td colspan="3">Gray moderate</td></tr><tr><td>L</td><td>C</td><td>0</td><td>L</td><td>C</td><td>0</td><td>L</td><td>C</td><td>0</td><td>L</td><td>C</td><td>0</td></tr><tr><td>GPT-5.2</td><td>67.35</td><td>29.58</td><td>3.07</td><td>50.19</td><td>45.33</td><td>4.48</td><td>84.25</td><td>11.01</td><td>4.74</td><td>68.25</td><td>20.87</td><td>10.88</td></tr><tr><td>Gemini 3.5 Flash</td><td>89.76</td><td>8.96</td><td>1.28</td><td>80.41</td><td>18.44</td><td>1.15</td><td>96.93</td><td>2.43</td><td>0.64</td><td>87.32</td><td>10.76</td><td>1.92</td></tr><tr><td>Claude Sonnet 4.6</td><td>85.02</td><td>12.16</td><td>2.82</td><td>71.06</td><td>22.66</td><td>6.27</td><td>96.80</td><td>2.56</td><td>0.64</td><td>82.71</td><td>11.40</td><td>5.89</td></tr><tr><td>Qwen3-VL-32B T.</td><td>47.50</td><td>44.43</td><td>8.07</td><td>35.60</td><td>51.47</td><td>12.93</td><td>67.86</td><td>24.33</td><td>7.81</td><td>58.64</td><td>25.61</td><td>15.75</td></tr></table>

Table 19: Transcription outcomes across blur and scene-removal conditions. All rates are percentages. T. denotes Thinking.

For each outcome, the blur effect is the moderate-minus-clear rate; the interaction is the full blur effect minus the gray blur effect. Pointwise intervals use 20,000 paired image-bootstrap resamples. All literal-accuracy interaction intervals include zero, while canonical-rewriting interactions are positive for GPT and Qwen. Together, these results suggest that surrounding cues can reinforce canonical rewriting when visual evidence from the target becomes weaker.