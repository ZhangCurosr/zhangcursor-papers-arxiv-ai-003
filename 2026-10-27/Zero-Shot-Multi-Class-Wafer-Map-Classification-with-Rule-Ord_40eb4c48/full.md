# Zero-Shot Multi-Class Wafer Map Classification with Rule-Ordered Prompts for Large Multimodal Models

Tianjian Cao

Fudan University

Shanghai, China

tjcao24@m.fudan.edu.cn

Jun Tao   
Fudan University   
Shanghai, China   
taojun@fudan.edu.cn

## Abstract

Wafer-map defect classification often requires labeled data and retraining when production conditions change. We present a trainingfree six-class framework that combines connected-component high lighting with morphology-, location-, and exclusion-aware prompts for a frozen large multimodal model. On Mixed-WM38 (6,866 maps), multi-region prompting achieves 92.9% macro and 93.8% overall accuracy. On WM811K (9,000 maps), it achieves 79.1% and 81.3%, respectively. Eight alternative rule orders achieve 90.1–91.9% overall accuracy, compared with 93.8% for the selected order.

## Keywords

wafer map classification, zero-shot learning, large multimodal model, prompt design, defect pattern recognition, semiconductor manufacturing

## 1 Introduction

Wafer maps use a two-dimensional grid to record the spatial distribution of tested dies on a semiconductor wafer, and they are one of the most common quality representations in the manufacturing test stage. Each die is typically encoded as pass or fail and arranged according to its physical location on the wafer [1, 2]. Recognizing spatial defect patterns in wafer maps is important for yield analysis, process diagnosis, and production control. From a process perspective, diferent manufacturing deviations can create diferent spatial failure patterns on the wafer plane. For example, chemical-mechanical planarization (CMP) non-uniformity may produce center-region failure aggregation. Layer-to-layer misalignment in the storage-node process may generate edge patterns along the wafer boundary, and localized non-uniformity or uneven cleaning can lead to local defect clusters [8]. Therefore, recognizing these specific patterns is practically valuable for translating wafer test results into actionable process feedback.

Existing automatic wafer map classification methods can be broadly grouped into unsupervised and supervised methods. Unsupervised methods extract spatial structures from unlabeled wafer maps using principal component analysis [3], clustering, and spatial filtering [4]. They are useful for discovering latent failure modes, but are often sensitive to preprocessing and handcrafted feature design. Supervised methods learn class decision boundaries from labeled wafer maps. Earlier studies apply linear discriminant analysis [5] and support vector machine [6] to map handcrafted or statistical features to defect classes. CNN-based methods and deep models for mixed-type patterns later improve spatial feature learning [7, 8]. Recent studies explore few-shot/self-supervised learning [9], and state-space models for low-data settings or complex pattern model ing [10]. Although these methods can achieve strong performance on a given dataset, they assume that feature representations and decision boundaries are obtained through training samples, task adaptation, or model learning.

In real manufacturing environments, supervised methods face three major limitations. First, high-quality wafer maps require tapeout, testing, and expert annotation, while new production lines or process nodes may have few historical samples [11]. Second, changes in production line, process node, wafer size, or product design can shift defect distributions and require retraining or calibration [12]. Third, human labels may contain noise and inconsistency, degrading model reliability [13].

Large multimodal models provide an alternative route to targetdomain training. CLIP demonstrates zero-shot image classification through natural-language supervision [14], while instructionfollowing LMMs such as GPT-4o support direct image-text reasoning [19]. Unlike prompt learning [20] and visual prompt tuning [21], which optimize task-specific parameters, visual prompting uses visible marks or annotations to guide a frozen model toward relevant image regions [15, 16]. Recent wafer-map research further shows that connected-component highlighting can support training-free binary normal/abnormal recognition with a frozen LMM [24].

Building on the connected-component visual interface of [24], which is developed for binary normal/abnormal recognition, this work studies fine-grained six-class classification. The main challenge is no longer only to locate salient defect regions, but to resolve overlapping class boundaries using morphology-, location-, and exclusion-aware textual rules. We formulate the task as a trainingfree semantic decision process in which the wafer map provides visual input and a fixed rule-presentation sequence specifies defect morphology, location, and class boundaries. This setting complements supervised classification by targeting the stage before representative target-domain data become available, and it supports inspection and revision of the rules without parameter retraining.

We consider zero-shot classification of six wafer defect patterns: Center, Donut, Edge, Local, Random, and Scratch, as illustrated in Figure 1. Their recognition requires interpretation of both the morphology of dominant defect regions and their spatial organization across the wafer.

The proposed framework couples a connected-component-based visual representation with an ordered set of textual decision rules. The semantic rules map observed regions and distributions to defect classes, and the presentation order encodes an intended priority when multiple class descriptions are partially compatible with the same wafer map. Together, these components provide an inspectable, training-free decision process. The main contributions are:

![](images/2f52f91689e70ae57e85acc3979cdd3452c9783bc42db7200bc42c3263bfa270.jpg)  
Figure 1: Representative raw wafer maps of the six target defect patterns from Mixed-WM38 [18]. White pixels denote failed dies, gray pixels denote non-defect wafer areas, and black regions denote the background.

• We formulate six-class wafer-map recognition as a trainingfree image-text semantic decision problem and evaluate a frozen LMM on two industrial datasets.

• We introduce boundary-aware semantic rules that combine positive morphology, spatial anchoring, and exclusion conditions for neighboring classes.

• We define a fixed rule-presentation order that encodes an intended priority for overlapping evidence and evaluate controlled changes in regional marking, semantic detail, and rule order.

## 2 Proposed Method

## 2.1 Overview

We formulate multi-class wafer map classification as a training-free semantic decision problem. We consider six wafer defect patterns defined by their spatial morphology and location: Center, Donut, Edge, Local, Random, and Scratch, as illustrated in Fig. 1. Center denotes a defect cluster near the wafer center; Donut denotes a ring-like or hollow structure around the center; Edge denotes defect aggregation along the boundary; Local denotes compact local clusters; Scratch denotes an elongated defect with long-axis continuity; and Random denotes large block-like failures distributed across much of the wafer interior, leaving little continuous gray area. Because several classes share spatial cues—for example, Center and Donut are both center-related, while Edge and Local may both occur near the wafer boundary—we formulate recognition as a boundary-aware semantic decision problem. Let � denote a raw wafer map and let

$$
\pmb { \mathcal { Y } } = \{ \mathrm { C e n t e r , D o n u t , E d g e , L o c a l , R a n d o m , S c r a t c h } \} .\tag{1}
$$

The visual prompting interface transforms � into a visual representation $Z = V ( X )$ , while the textual prompt encodes a set of class-semantic rules $\mathcal { R } = \{ r _ { c } \ | \ c \in \mathcal { Y } \}$ and their presentation order �. A frozen LMM then predicts

$$
\hat { y } = F _ { \theta } ( Z , P ( Z ; \mathcal { R } , \pi ) ) , \qquad \hat { y } \in \mathcal { Y } ,\tag{2}
$$

where � remains fixed throughout inference. This formulation separates the visual evidence � from the semantic decision specification $P ,$ while allowing the textual legend to remain consistent with the visible form of �. Figure 2 summarizes the resulting image-text classification framework.

Failed dies naturally form connected spatial structures on the wafer plane. Following [24], � uses these connected components as region-level visual references, allowing textual concepts such as “dominant region,” “rim-following arc,” and “global distribution” to be linked to visible regions in the image. Black pixels denote the background, gray pixels denote non-defect wafer areas, and failed dies constitute the defect mask. The mask is partitioned using 8-connectivity [17], and the top-� components ranked by size are highlighted with distinct colors. The largest selected component is shown in red, while subsequent components receive diferent colors. These colors serve only as region identifiers rather than defect labels, and unselected failed dies remain visible in white.

![](images/d44237613ab4069de3377e9f286aed44439e64d70136af96422a037df2f5f446.jpg)  
Figure 2: Overview of the training-free image-text classification framework. The visual input is paired with a textual prompt, and a frozen LMM processes the image-text pair once and returns exactly one label.

We consider two forms of �: Single-region highlights the largest component (� = 1), whereas Multi-region highlights the five largest components $\left( K = 5 \right)$ . These alternatives control the scope of explicitly marked regional evidence without changing the underlying wafer map. The construction is fixed across diferent wafers.

Given �, the textual prompt connects the highlighted visual evidence to the six class definitions. For each class � ∈ Y, we define a structured semantic rule $r _ { c } = \left( M _ { c } , A _ { c } , E _ { c } \right)$ , where $M _ { c }$ describes the characteristic morphology, $A _ { c }$ specifies its spatial anchor on the wafer, and $E _ { c }$ distinguishes the class from neighboring interpretations. Section 2.2 develops these three rule components. Let $\pi = ( \pi _ { 1 } , \ldots , \pi _ { | { \cal y } | } )$ denote a permutation of the class labels that determines the presentation order of the rules in the textual prompt. This order encodes an intended priority for cases with overlapping evidence. The domain rationale for the selected order is developed in Section 2.3.

The complete textual prompt is constructed as

$$
P ( Z ; \mathcal { R } , \pi ) = L ( Z ) \oplus \left[ \bigoplus _ { j = 1 } ^ { | y | } T \Big ( r _ { \pi _ { j } } \Big ) \right] \oplus C , \qquad r _ { c } = ( M _ { c } , A _ { c } , E _ { c } ) .\tag{3}
$$

where $L ( Z )$ is the legend associated with the current visual representation, � verbalizes a structured class rule in natural language, � specifies the allowed labels and response format, and ⊕ denotes textual concatenation. Thus, the middle term preserves the semantic content of every class rule while arranging the rules according to �. The formulation makes explicit the two complementary design choices developed below: the internal construction of each rule $r _ { c }$ and the presentation order assigned to the rule sequence. The prompt comprises a visual legend, the six rules summarized in Table 1 and arranged in the fixed order Donut > Scratch > Edge > Random > Center > Local, and an instruction to return exactly one label. For Local and Scratch, it emphasizes the largest component; for Edge and Random, the global defect distribution. The same wording is used for all inputs within each visual condition. For every input, the same rule set R and selected presentation order � are used without access to ground-truth labels or dataset-specific adaptation. Sections 2.2 and 2.3 develop the rule semantics and presentation-order design represented in Eq. (3), respectively.

## 2.2 Class-Semantic Rule Construction

Wafer defect classes are usually named by process engineers according to the spatial shape and location of defects, but the names alone are insuficient to constrain zero-shot LMM decisions. Center, Donut, Edge, Local, Random, and Scratch are not six fully independent low-level visual labels. They share part of their spatial evidence. For example, a near-center defect may be interpreted as Center because of its location, but as Donut because of its hollow structure; a local defect cluster close to the edge may be interpreted as Edge or Local. If the prompt only lists class names, the model may treat such shared evidence as final class evidence. We therefore rewrite each class as a semantic rule, so that the prompt states what should be observed, where it should be observed, and why neighboring classes should be rejected. Accordingly, this section defines the internal structure of each rule $r _ { c }$ in Eq. (3), specifying what visual evidence supports class � and what evidence separates it from its closest alternatives.

As introduced in $\operatorname { E q } .$ (3), each class rule is represented as $r _ { c } = { }$ $( M _ { c } , A _ { c } , E _ { c } )$ , comprising morphology, spatial anchoring, and an exclusion boundary. Morphology $M _ { c }$ characterizes the dominant geometric structure of a defect, such as a solid cluster, hollow ring, elongated trace, edge-following arc, or broadly distributed pattern. Spatial anchoring $A _ { c }$ specifies the location of that structure relative to key wafer regions, including the center, interior, and circular boundary. The exclusion boundary $E _ { c }$ distinguishes the target class from visually similar neighboring classes by identifying evidence that supports or rules out alternative interpretations. Table 1 summarizes these components.

The rules use qualitative morphology and spatial relations rather than dataset-specific thresholds or sample-level information. Figure 3 grounds two representative pairwise rules in the Multi-region visual representation used by the LMM. In the Center-Donut pair, both maps contain defect evidence near the wafer center. Their class assignments therefore cannot be determined from central location alone. The Center example contains a predominantly solid central mass, whereas the Donut example contains a meaningful hollow or annular structure in the dominant central pattern. The exclusion condition of the Center rule consequently prevents secondary surrounding regions from being interpreted as a Donut when the main central structure remains solid.

The Edge-Local pair illustrates a second shared-evidence bound ary. Both examples contain defect regions near the wafer boundary, so boundary contact alone is not suficient to support an Edge prediction. In the Edge example, the dominant defect structure follows the circular wafer rim as an arc-like segment. In the Local example, the dominant structure forms a compact clump that extends inward from the boundary rather than following its circumference. This distinction corresponds directly to the exclusion conditions in Table 1: Edge rejects inward compact clusters, whereas Local rejects rim-following arcs and peripheral bands.

![](images/6a6e576b4c9e97402f29696ca9feeb483000f2d2cce04130da515fe092a9da7c.jpg)  
Figure 3: Visual grounding of two pairwise exclusion boundaries in the class-semantic rules. Center and Donut difer in solid versus hollow morphology; Edge and Local difer in rim alignment versus inward compactness.

Together, the positive morphology, spatial anchor, and exclusion condition link each class decision to explicit visual evidence and distinguish it from the closest competing interpretations.

On the image side, connected-component marks identify the regions to which the rules refer. A single marked region presents a focused view of the largest component, whereas multiple marked regions present broader coverage when several components may be relevant. Their relative usefulness can vary across defect patterns according to the number, spatial arrangement, and diagnostic importance of the connected regions. We therefore compare diferent marking configurations to examine how the amount of highlighted visual evidence afects pattern-specific and overall performance.

## 2.3 Rule-Ordered LMM Classification

Section 2.2 defines the semantic content of each rule $r _ { c } ,$ but some wafer maps may remain partially compatible with multiple class descriptions. For example, a central defect may exhibit both Centerlike location and Donut-like topology, while a thick boundary cluster may retain cues of both Local and Edge. We encode an intended priority for such cases through the permutation � in Eq. (3). Specifically, � determines the presentation sequence $T ( r _ { \pi _ { 1 } } ) , \ldots , T ( r _ { \pi _ { | . 9 | } } )$ in the textual prompt while leaving the semantic content of every $r _ { c }$ unchanged. The prompt instructs the frozen LMM to consider earlier rules first when multiple descriptions appear compatible.

Because the LMM processes the complete image-text input jointly, this order provides an instruction-level decision bias rather than a deterministic symbolic decision tree.

The ordering of class-specific rules is treated as a controlled component of the prompt. This design is motivated by documented order sensitivity of LLMs: even when candidate content remains unchanged, reordering answer options or other input elements can alter model predictions, particularly when the model is uncertain among competing alternatives [22]. Similar efects have been observed in closed-source and API-based LLMs across paraphrasing, relevance judgment, and multiple-choice question answering [23]. A fixed rule-presentation order therefore controls positional variation as an explicit input condition in frozen-LMM classification.

This control is especially important in the zero-shot setting, where no target-domain examples or parameter updates are available to help the model resolve ambiguity among visually similar classes. We use the fixed order Donut > Scratch > Edge > Random > Center > Local as a domain-informed presentation sequence. The order is motivated by pairwise boundary relations among visually overlapping classes. Donut precedes Center because annular structure is more diagnostic than central location alone, while Scratch precedes Edge because non-circumferential elongation should be considered before boundary contact. Edge precedes Random so that rim-following arcs or rings are distinguished before broad interior coverage, and Random precedes Center and Local so that wafer-wide coverage is considered before compact aggregation. Finally, Center precedes Local because central anchoring is more specific than a general non-central compact region. These relations motivate the selected sequence but do not uniquely determine the relative positions of every class pair. The remaining placements are fixed as a consistent presentation convention and are retained across all inputs and datasets. The selected sequence is evaluated as a fixed prompt condition rather than as a globally optimal ordering.

Table 1: Class-semantic rules used in the textual prompt.
<table><tr><td>Class</td><td>Positive morphology</td><td>Spatial anchor</td><td>Exclusion condition</td></tr><tr><td>Center</td><td>Dense, compact, predominantly solid cluster</td><td>Centered or near wafer center</td><td>Exclude hollow, annular, rim-aligned, elongated, and wafer-wide patterns.</td></tr><tr><td>Donut</td><td>ral, or partial annulus</td><td>Ring, hollow center, arc, crescent, C-shape, spi- Around or partly surrounding wafer center</td><td>Exclude solid central clusters without a clear opening.</td></tr><tr><td>Edge</td><td>rim-following segment, or edge ring</td><td>boundary</td><td>Continuous or fragmented arc, peripheral band, Attached to and aligned with the circular Boundary contact alone is insufficient; Exclude inward compact clusters and non-circumferential elongated traces.</td></tr><tr><td>Local</td><td>gregation</td><td>extending inward</td><td>Thick, compact, solid clump or polygonal ag- Non-central interior, or boundary-near but Exclude rim-following arcs, wafer-wide patterns, central clusters, and elongated traces.</td></tr><tr><td>Random</td><td>coverage and little continuous gray area</td><td>finement</td><td>Large block-like regions with broad interior Across much of the interior without rim con- Exclude local clusters, annular structures, elongated traces, and rim-aligned arcs or rings.</td></tr><tr><td>Scratch</td><td>Long, narrow, line-like, stripe-like, diagonal, or Any wafer location aligned broken chain</td><td></td><td>Exclude compact blobs with short tails and arcs whose principal direction follows the rim.</td></tr></table>

Algorithm 1 Training-Free Wafer Map Classification   
Input: Wafer maps $\Phi ,$ frozen LMM $F _ { \theta } ,$ , and number of highlighted   
components $K$   
Output: Predicted labels $\{ \hat { y } _ { i } \} _ { \phi _ { i } \in \Phi }$   
1: Construct the six semantic rules and fix their presentation   
order.   
2: for all wafer map $\phi _ { i } \in \Phi$ do   
3: Highlight the � largest connected defect components.   
4: Combine the prompted map, visual legend, and ordered   
rules.   
5: $\hat { y } _ { i } \gets F _ { \theta } ($ image-text pair).   
6: end for   
7: return $\{ \hat { y } _ { i } \} _ { \phi _ { i } \in \Phi } .$

The selected order leaves the definitions of Center, Donut, Edge, Local, Random, and Scratch unchanged and carries no assumption about their relative frequency or importance. Instead, its presentation sequence encodes an intended interpretive priority for cases in which a wafer map exhibits partially overlapping cues from multiple classes. Accordingly, the order is evaluated as a controlled promptpresentation condition rather than as a deterministic decision tree. Algorithm 1 summarizes how the visual prompting interface and the ordered semantic rules are combined into an image-text input for the frozen LMM.

## 3 Experiments

We evaluate controlled zero-shot prompt variants under a fixed frozen-LMM setting. Prompt templates were fixed before evaluation and reused across all samples and both datasets without datasetspecific modification. The exact same semantic rules and rule order were used for both datasets; legends and output constraints were reused for every map within each visual condition. Following the visual-prompt configuration in [24], Raw preserves the original de fect distribution without component overlays, Single-region marks one major connected defect region (� = 1), whereas Multi-region marks five major connected defect regions (� = 5). Neither the im ages nor the labels in Mixed-WM38 [18] and WM811K [2] are used for training, fine-tuning, or few-shot prompting before inference. When visual prompting is applied, it only marks connected defect regions according to Section 2.1, with a legend that explains the visible marks.

All experiments use the frozen gpt-4o-2024-08-06 model through the OpenAI API [19], with the temperature fixed at 0.1. Each wafer map is submitted once as an image-text pair; thus, the reported results reflect single-query inference and do not characterize queryto-query variation. We report per-class accuracy, macro accuracy, defined as the unweighted mean of the six per-class accuracies, and overall accuracy, defined as the proportion of correctly classified maps in the complete evaluation set. Per-class accuracy is computed within each ground-truth class. Center, Donut, Edge, Local, Random, and Scratch will be abbreviated as C, D, E, L, R, and S, respectively.

## 3.1 Mixed-WM38

Mixed-WM38 [18] contains approximately 38,000 wafer maps covering normal wafers, eight single-pattern defect categories, and mixed-type defects. For the six-class evaluation considered in this study, we retain all single-pattern wafer maps whose source labels correspond to Center, Donut, Edge, Local, Random, or Scratch. The resulting evaluation set comprises 6,866 maps, including 1,000 maps for each of Center, Donut, Local, and Scratch, 2,000 maps for Edge by combining Edge\_Loc and Edge\_Ring, and 866 maps for Random. The original dataset labels are used exclusively to compute the per-class, macro, and overall accuracies reported in Tables 2 and 3. They are not used to construct the prompts, adapt the model, or select predictions.

We first examine how the presentation and scope of regional evidence afect multi-class classification. Table 2 compares four prompting conditions on Mixed-WM38: the unmodified wafer map, a single-region representation, a multi-region representation, and a names-only prompt. Regional visual marking produces a substantial improvement over the raw image representation. Single-region prompting increases the overall accuracy from 37.5% to 93.3% and the macro accuracy from 44.6% to 92.4%. Multi-region prompting further improves these values to 93.8% and 92.9%, respectively, and therefore achieves the best overall and macro accuracy in this comparison. Single-region performs better for Center and Local, whereas Multi-region provides higher accuracy for Donut, Edge, and Random by preserving additional spatial evidence from secondary components. When the semantic descriptions are removed and only class names are provided, the overall accuracy decreases to 63.6% and the macro accuracy decreases to 58.2%. This reduction indicates that class names alone do not suficiently specify the morphology, spatial location, and exclusion boundaries required for fine-grained classification.

Table 2: Accuracy (%) on Mixed-WM38.
<table><tr><td>Method</td><td>C</td><td>D</td><td>E</td><td>L</td><td>R</td><td>S</td><td>Macro</td><td>Overall</td></tr><tr><td>Raw</td><td>9.5</td><td>97.1</td><td>2.8</td><td>11.7</td><td>97.9</td><td>46.8</td><td>44.6</td><td>37.5</td></tr><tr><td>Single</td><td>99.6</td><td>97.3</td><td>95.6</td><td>98.0</td><td>70.1</td><td>94.0</td><td>92.4</td><td>93.3</td></tr><tr><td>Multi</td><td>97.6</td><td>98.7</td><td>96.8</td><td>92.1</td><td>78.5</td><td>93.9</td><td>92.9</td><td>93.8</td></tr><tr><td>Names-only</td><td>69.7</td><td>39.4</td><td>90.9</td><td>64.2</td><td>24.7</td><td>60.5</td><td>58.2</td><td>63.6</td></tr></table>

Table 3: Accuracy (%) of rule-presentation orders on Mixed-WM38.
<table><tr><td>Variant</td><td>C</td><td>D</td><td>E</td><td>L</td><td>R</td><td></td><td>S Macro</td><td>Overall</td></tr><tr><td>Default</td><td>97.6</td><td>98.7</td><td>96.8</td><td>92.1</td><td>78.5</td><td>93.9</td><td>92.9</td><td>93.8</td></tr><tr><td>Center-first</td><td>92.1</td><td>96.3</td><td>94.0</td><td>93.0</td><td>76.5</td><td>91.1</td><td>90.5</td><td>91.3</td></tr><tr><td>Edge-before-Scratch</td><td>94.5</td><td>97.7</td><td>92.3</td><td>93.7</td><td>77.5</td><td>87.0</td><td>90.4</td><td>91.0</td></tr><tr><td>Local-before-Edge</td><td>96.1</td><td>97.5</td><td>89.0</td><td>94.0</td><td>71.7</td><td>91.0</td><td>89.9</td><td>90.1</td></tr><tr><td>Local-before-Scratch</td><td>95.5</td><td>98.1</td><td>92.0</td><td>96.0</td><td>80.1</td><td>80.5</td><td>90.4</td><td>90.8</td></tr><tr><td>Random-before-Edge</td><td>94.0</td><td>96.9</td><td>91.5</td><td>95.5</td><td>80.3</td><td>92.3</td><td>91.8</td><td>91.9</td></tr><tr><td>Random-last</td><td>97.5</td><td>97.7</td><td>85.5</td><td>95.0</td><td>86.5</td><td>94.5</td><td>92.8</td><td>91.9</td></tr><tr><td>Edge-Local-Scratch</td><td>97.0</td><td>98.0</td><td>91.8</td><td>94.5</td><td>84.5</td><td>77.5</td><td>90.5</td><td>90.8</td></tr><tr><td>Center-Local swap</td><td>94.3</td><td>98.1</td><td>86.8</td><td>96.0</td><td>80.5</td><td>94.0</td><td>91.6</td><td>91.1</td></tr></table>

## 3.2 Rule-presentation order

We next isolate the efect of rule presentation by keeping the Multiregion images, visual legend, class-rule wording, and output format unchanged while varying only the rule-presentation order of the six rule blocks. The default sequence is D>S>E>R>C>L. The eight targeted variants modify this sequence to examine boundaries between visually related classes: Center-first uses C>D>S>E>R>L; Edge-before-Scratch uses D>E>S>R>C>L; Local-before-Edge uses D>S>L>E>R>C; Local-before-Scratch uses D>L>S>E>R>C; Random-before-Edge uses D>S>R>E>C>L; Random-last uses D>S>E>C>L>R; Edge-Local-Scratch uses D>E>L>S>R>C; and Center-Local swap uses D>S>E>R>L>C. Because the semantic content remains identical across these variants, the comparison directly characterizes the sensitivity of the frozen LMM to rulepresentation order.

The default order achieves the highest overall accuracy, reaching 93.8%, with a macro accuracy of 92.9%. All eight alternative orders reduce the overall accuracy to between 90.1% and 91.9%, although the efects are not uniform across classes. Placing Local before Edge increases Local accuracy from 92.1% to 94.0%, but decreases Edge ac curacy from 96.8% to 89.0%. Placing Random last increases Random accuracy from 78.5% to 86.5%, while reducing Edge accuracy from 96.8% to 85.5%. Similarly, placing Local before Scratch improves Local accuracy to 96.0%, but reduces Scratch accuracy from 93.9% to 80.5%. These non-uniform changes indicate that rule order acts as an instruction-level bias for resolving partially compatible class descriptions, rather than as a deterministic priority mechanism [22].

## 3.3 WM811K

WM811K [2] contains 172,950 pre-labeled wafer maps and is a large dataset collected from a real manufacturing environment. For testing, we randomly selected 9,000 maps for the six target patterns: 1,747 Center, 502 Donut, 3,430 Edge, 1,830 Local, 555 Random, and 936 Scratch maps. Edge again combines the two edge-pattern classes Edge\_Loc and Edge\_Ring.

Table 4: Accuracy (%) on WM811K.
<table><tr><td>Method</td><td>C</td><td>D</td><td>E</td><td>L</td><td>R</td><td>S</td><td>Macro</td><td>Overall</td></tr><tr><td>Raw</td><td>52.4</td><td>72.1</td><td>26.5</td><td>31.0</td><td>99.6</td><td>70.8</td><td>58.7</td><td>43.7</td></tr><tr><td>Single</td><td>94.7</td><td>61.6</td><td>79.1</td><td>78.4</td><td>69.8</td><td>76.4</td><td>76.7</td><td>80.1</td></tr><tr><td>Multi</td><td>91.3</td><td>73.1</td><td>81.0</td><td>76.4</td><td>74.8</td><td>80.9</td><td>79.1</td><td>81.3</td></tr></table>

![](images/de2f99e6a2cdde3682e3a0ca41b6fad770f5e086b7c8dd6837e11ac0e7962f45.jpg)  
Figure 4: Representative WM811K samples with weak or ambiguous class-specific morphologies; column headings denote source labels.

Table 4 shows that regional prompting remains efective on WM811K. Single-region and Multi-region improve overall accuracy from 43.7% for Raw to 80.1% and 81.3%, respectively, while increasing macro accuracy from 58.7% to 76.7% and 79.1%. The improvement in macro accuracy indicates that the overall gain is not attributable only to the relatively large Edge class. Multi-region achieves the best overall and macro accuracies, although Singleregion remains more efective for Center and Local. This result is consistent with the evidence-scope trade-of observed on Mixed-WM38: a single highlighted component emphasizes a dominant compact region, whereas multiple highlighted components better preserve distributed or secondary evidence.

Nevertheless, Multi-region accuracy is lower on WM811K than on Mixed-WM38 by 12.5 percentage points overall and 13.8 percentage points in macro accuracy. The largest class-wise decrease occurs for Donut, followed by Edge, Local, and Scratch. As discussed in [24], Mixed-WM38 contains uniformly sized wafer maps with visually dominant abnormal patterns, whereas WM811K contains wafer maps of varying sizes and many ambiguous or fragmented defect patterns. These can weaken the morphology required to distinguish neighboring classes, such as Donut from Center or Local, and Edge from Scratch or Local.

Figure 4 illustrates these sources of error. Some source labels are dificult to reconcile with the dominant visual morphology: the Donut sample contains only an incomplete curved structure without a clear hollow center or annular topology, making the source label dificult to distinguish from Local. The Edge sample combines boundary contact with elongated structures and therefore provides evidence compatible with Edge and Scratch. The dominant region in the Local sample is long and narrow rather than compact, which makes it visually closer to Scratch. Conversely, the highlighted region in the Scratch sample is short and attached to the wafer boundary, with limited long-axis continuity, making Edge or Local plausible alternatives. These examples show that some WM811K wafer maps only weakly express the morphology associated with their source labels.

The results therefore suggest that the remaining performance gap arises from weaker morphological salience, greater within-class variation, and overlapping class boundaries in WM811K.

API inference required approximately 0.5–1 s per image in our environment. Although it is slower and more costly than local inference with a trained classifier, it removes the need for targetdomain training before deployment.

## 4 Conclusion

This paper presents a training-free framework for zero-shot multiclass wafer map classification with a frozen large multimodal model. The framework combines connected-component highlighting with boundary-aware semantic rules to make defect regions, class characteristics, and exclusion boundaries explicit in the image-text input. A fixed rule-presentation order further provides a consistent instruction-level mechanism for resolving partially overlapping class descriptions. Experiments on two industrial datasets show that regional highlighting and complete semantic rules substantially improve classification over raw inputs and class names alone, while controlled order variants confirm that the placement of competing rules afects the resolution of ambiguous evidence. These findings show that prompt construction can serve as an explicit semantic decision specification for fine-grained wafer map recognition. The framework requires no labeled training examples, fine-tuning, or parameter updates, and its visual and textual specifications can be inspected and revised for new classification requirements. Future work will evaluate robustness across multimodal models and mixed-pattern wafer maps.

## References

[1] T. Yuan, W. Kuo, and S. J. Bae, “Detection of spatial defect patterns generated in semiconductor fabrication process,” IEEE Transactions on Semiconductor Manufacturing, vol. 24, no. 3, pp. 392–403, 2011.

[2] M. Wu, J. Jang, and J. Chen, “Wafer map failure pattern recognition and similarity ranking for large-scale data sets,” IEEE Transactions on Semiconductor Manufacturing, vol. 28, no. 1, 2015.

[3] Y. Hsieh, G. Tzeng, G. Lin, and H. Yu, “Wafer sort bitmap data analysis using the PCA-based approach for yield analysis and optimization,” IEEE Transactions on Semiconductor Manufacturing, vol. 23, no. 4, pp. 493–502, 2010.

[4] A. Ezzat, S. Liu, D. Hochbaum, and Y. Ding, “A graph-theoretic approach for spatial filtering and its impact on mixed-type spatial pattern recognition in wafer bin maps,” IEEE Transactions on Semiconductor Manufacturing, vol. 34, no. 2, pp. 194–206, 2021.

[5] J. Yu and X. Lu, “Wafer map defect detection and recognition using joint local and nonlocal linear discriminant analysis,” IEEE Transactions on Semiconductor Manufacturing, vol. 29, no. 1, pp. 33–43, 2015.

[6] R. Baly and H. Hajj, “Wafer classification using support vector machines,” IEEE Transactions on Semiconductor Manufacturing, vol. 25, no. 3, pp. 373–383, 2012.

[7] T. Nakazawa and D. Kulkarni, “Wafer map defect pattern classification and image retrieval using convolutional neural network,” IEEE Transactions on Semiconducto Manufacturing, vol. 31, no. 2, pp. 309–314, 2018.

[8] K. Kyeong and H. Kim, “Classification of mixed-type defect patterns in wafer bin maps using convolutional neural networks,” IEEE Transactions on Semiconductor Manufacturing, vol. 31, no. 3, pp. 395–402, 2018.

[9] H. Geng, F. Yang, X. Zeng, and B. Yu, “When wafer failure pattern classification meets few-shot learning and self-supervised learning,” International Conference on Computer-Aided Design, 2021

[10] M. Nie, S. Zhu, A. Yan, C. Zhuo, X. Wen, and T. Ni, “Eficient modulated state space model for mixed-type wafer defect pattern recognition,” Design, Automation & Test in Europe Conference, 2025.

[11] C. Chen, W. Wang, and J. Cheng, “Data mining for yield enhancement in semiconductor manufacturing and an empirical study,” Expert Systems with Applications, vol. 33, no. 1, pp. 192–198, 2007.

[12] S. Zhao, Z. Zhu, X. Li, and Y. Chen, “Robust wafer classification with imperfectly labeled data based on self-boosting co-teaching,” IEEE Transactions on Computer-Aided Design ofIntegrated Circuits and Systems, vol. 42, no. 7, pp. 2214–2226,

2022.

[13] B. Frenay and M. Verleysen, “Classification in the presence of label noise: A survey,” IEEE Transactions on Neural Networks and Learning Systems, vol. 25, no. 5, pp. 845–869, 2014.

[14] A. Radford, J. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, and G. Krueger, “Learning transferable visual models from natural language supervision,” International Conference on Machine Learning, pp. 8748–8763, 2021.

[15] L. Yang, Y. Wang, X. Li, X. Wang, and J. Yang, “Fine-grained visual prompting,” Neural Information Processing Systems, vol. 36, pp. 24993–25006, 2023.

[16] J. Yang, H. Zhang, F. Li, X. Zou, C. Li, and J. Gao, “Set-of-mark prompting unleashes extraordinary visual grounding in GPT-4V,” arXiv preprint arXiv:2310.11441, 2023.

[17] K. Wu, E. Otoo, and K. Suzuki, “Optimizing two-pass connected-component labeling algorithms,” Pattern Analysis and Applications, vol. 12, no. 2, pp. 117–135, 2009.

[18] J. Wang, C. Xu, Z. Yang, J. Zhang, and X. Li, “Deformable convolutional networks for eficient mixed-type wafer defect pattern recognition,” IEEE Transactions on Semiconductor Manufacturing, vol. 33, no. 4, pp. 587–596, 2020.

[19] A. Hurst, A. Lerer, A. Goucher, A. Perelman, A. Ramesh, A. Clark, et al., “GPT-4o system card,” arXiv preprint arXiv:2410.21276, 2024.

[20] K. Zhou, J. Yang, C. C. Loy, and Z. Liu, “Learning to prompt for vision-language models,” International Journal ofComputer Vision, vol. 130, pp. 2337–2348, 2022.

[21] M. Jia, L. Tang, B. Chen, C. Cardie, S. Belongie, H. Hariharan, and S. Lim, “Visual prompt tuning,” European Conference on Computer Vision, pp. 709–727, 2022.

[22] P. Pezeshkpour and E. Hruschka, “Large language models sensitivity to the order of options in multiple-choice questions,” Findings ofthe Association for Computational Linguistics: NAACL, pp. 2006–2017, 2024.

[23] B. Guan, T. Roosta, P. Passban, and M. Rezagholizadeh, “The order efect: investigating prompt sensitivity to input order in LLMs,” arXiv preprint arXiv:2502.04134, 2025.

[24] T. Cao, J. Tao, C.-W. Sham, and X. Li, “Eficient zero-shot wafer classification method based on the large multi-modal model,” IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems, Early Access, 2026, doi: 10.1109/TCAD.2026.3699293.