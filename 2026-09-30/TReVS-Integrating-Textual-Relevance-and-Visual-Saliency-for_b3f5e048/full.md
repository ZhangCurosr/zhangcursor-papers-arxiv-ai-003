# TReVS: Integrating Textual Relevance and Visual Saliency for Eficient Vision-Language Model Token Pruning

Jing Wang<sup>1∗</sup>, Zhiping Wu<sup>2∗</sup>, Dongdong Ren<sup>3</sup>, Youfang Han<sup>4</sup>, Wei Zhao<sup>4</sup>, Wenbin Li<sup>1†</sup>

<sup>1</sup>School of Intelligence Science and Technology, Nanjing University

<sup>2</sup>School of Electronic Science and Engineering, Nanjing University

<sup>3</sup>Geely Automobile Research Institute (Ningbo) Co., Ltd., 315000 <sup>4</sup>Alpha Labs, Goertek

Jing Wang: 221900040@smail.nju.edu.cn; Zhiping Wu: zhipingwu@smail.nju.edu.cn; Dongdong Ren: Dongdong.Ren2@geely.com; Youfang Han: fred.hanyf@goertek.com; Wei Zhao: charles.zhaow@goertek.com; Wenbin Li: liwenbin@nju.edu.cn

## Abstract

Vision-Language Models (VLMs) excel at visual understanding and reasoning but often incur substantial inference costs due to the large number of visual tokens. Recent visual token pruning methods increasingly follow a two-stage paradigm: they first remove visually redundant tokens after the vision encoder and then discard tokens irrelevant to the textual query within the Large Language Model (LLM). However, since the first stage typically relies solely on vision-encoder saliency, it may prematurely eliminate query-relevant tokens, depriving the subsequent text-guided stage of critical visual evidence. Our empirical analysis shows that incorporating query guidance into first-stage pruning better preserves task-relevant evidence and consistently improves performance over vision-only saliency-based pruning. We further find that high-variance attention heads are more sensitive to the textual query and yield more discriminative text-to-vision attention signals for second-stage pruning. Motivated by these findings, we propose TReVS, a training-free framework that combines textual relevance with vision-encoder saliency for pre-LLM pruning and leverages high-variance attention heads to remove task-irrelevant tokens at shallow-to-intermediate layers of the LLM. On LLaVA-1.5-7B, TReVS retains 92.8% of the unpruned baseline performance while pruning 94.4% of visual tokens, outperforming prior state-of-the-art methods.

## 1 Introduction

Vision-Language Models (VLMs) extend the reasoning of pretrained Large Language Models (LLMs) (Touvron et al. 2023; Bai et al. 2023; Achiam et al. 2023) to visual inputs, achieving strong performance in visual question answering, multimodal reasoning, high-resolution image understanding, and video understanding (Liu et al. 2023; Zhu et al. 2023; Li et al. 2023a, 2024). Recent VLMs improve fine-grained perception across high-resolution images, multiple images, and videos. This broader coverage, however, produces visualtoken sequences substantially longer than text sequences. LLaVA-1.5 encodes each image into 576 visual tokens (Liu et al. 2024a), LLaVA-NeXT uses up to 2,880 tokens per high-resolution image (Liu et al. 2024b), and Qwen2.5-VL processes up to 16,384 visual tokens for multi-image and video inputs (Bai et al. 2025). These sequences increase computation, latency, and memory use, making visual-token processing a major bottleneck in eficient VLM inference.

![](images/b8c6c2b42775a9188cae6621d0b8806065953edb05c4d17be8c93c15cd291b68.jpg)  
Figure 1: (a–c) A two-stage baseline guided only by [CLS] attention during pre-LLM pruning discards visual tokens representing the queried grass, leaving incomplete evidence for subsequent query-guided pruning. TReVS uses textual relevance in pre-LLM pruning, preserving this evidence for the in-LLM stage. (d) TReVS achieves the best performance across six image-understanding benchmarks.

Visual token pruning addresses this bottleneck by exploiting the substantial redundancy in visual inputs (Bolya et al. 2023; Chen et al. 2024; Zhang et al. 2025b; Yang et al. 2025). Recent methods increasingly adopt a two-stage design that reduces visual redundancy after the vision encoder and then removes textually irrelevant tokens within the LLM (Takezoe et al. 2026; Zhang et al. 2026; Singh et al. 2026). The pre-LLM stage typically uses vision-encoder saliency or diversity to select tokens. The in-LLM stage uses text-to-vision cross-attention to retain query-relevant tokens. Although this division improves the performance, it creates an information bottleneck. Current pre-LLM pruning remains largely queryagnostic and can discard visual evidence that is essential to the text query. Once removed, such evidence is unavailable to subsequent query-aware pruning. The second stage must therefore identify task-relevant tokens within a potentially incomplete visual context.

We first examine the information loss caused by queryagnostic pre-LLM pruning and obtain two findings. First, incorporating textual relevance into pre-LLM pruning consistently improves performance by preserving query-relevant evidence (Figure 2(b)). Second, vision-encoder saliency and textual relevance produce distinct token rankings, with negative Spearman correlations across datasets and token budgets (Figure 2(c)). Together, these results show that textual relevance complements vision-encoder saliency and should be introduced before irreversible token reduction. We then examine the in-LLM stage, where the retained visual tokens are further refined using text-to-vision attention. We find that high-variance attention heads produce more discriminative and query-sensitive attention signals for identifying task-relevant tokens. These findings motivate a coordinated two-stage design: the first stage preserves visually salient, textually relevant, and diverse evidence, while the second uses high-variance heads to remove residual task-irrelevant tokens after cross-modal interaction develops. Section 3 provides detailed evidence for these findings.

Query: Is there a car in this image?GT: Yes.  
![](images/9b23632a3a31c31b2abd5d1c9c03c6aa91c5310722746406d35be21c218139fc.jpg)  
(a) Prune Visualization

(b) Relative Performance under Different Textual Ratios  
![](images/e8d9e1e3739262d70df7d89bf02f18d87b18c1258ddca1f360e80124d97b5546.jpg)

(c) Mean [CLS]-Textual Rank Correlation in Top-K Tokens  
![](images/f21558358852b87f08bd6c06ec2d3f241440086c57bcd6168d85878fe6785099.jpg)  
Figure 2: Efect of textual guidance on pre-LLM pruning. (a) Vision-encoder saliency focuses on the dominant fire hydrant, whereas textual relevance identifies the queried car. Their integration preserves both. (b) Performance relative to dense inference under diferent textual ratios. (c) Mean Spearman correlation between the two signals within the top-K saliency-selected tokens.

As illustrated in Figure 1, we propose TReVS, a trainingfree framework that coordinates pre-LLM and in-LLM visual token pruning. Before the LLM, TReVS scores visual tokens using [CLS] attention from the penultimate ViT layer and cosine similarity between projected visual tokens and embedded text tokens. It fuses these scores and supplements the top-ranked tokens with a small diversity set. During LLM prefilling, TReVS prunes again at a shallow-to-middle layer, using high-variance text-to-vision attention heads to identify query-relevant tokens. The first stage reduces visual redundancy while preserving salient, textually relevant, and diverse evidence. The second removes the remaining task-irrelevant tokens after cross-modal interaction develops.

Extensive experiments across vision-language benchmarks show that TReVS outperforms state-of-the-art training-free methods at multiple reduction ratios. On

LLaVA-1.5-7B, TReVS prunes 94.4% of visual tokens while retaining 92.8% of the original performance. Without additional training, it accelerates prefilling by 2.2× and end-toend inference by 1.4×. These results establish the importance of preserving textual relevance before visual token reduction. Our main contributions are summarized as follows:

• We show that introducing textual relevance before the first irreversible visual-token reduction preserves queryrelevant evidence and consistently improves performance. The rankings induced by textual relevance and visionencoder saliency remain negatively correlated across datasets and token budgets, showing that the two signals favor complementary visual tokens.

• We show that attention variance provides a simple, training-free criterion for identifying query-sensitive attention heads. These heads produce more discriminative text-to-vision attention signals for in-LLM pruning.

• We introduce TReVS, a training-free two-stage framework that combines vision-encoder saliency, textual relevance, and token diversity before the LLM, then uses high-variance attention heads to remove the remaining task-irrelevant visual tokens within the LLM. TReVS retains 92.8% of the baseline performance while pruning 94.4% of visual tokens.

## 2 Related Work

## 2.1 Vision-Language Models

Vision-Language Models (VLMs) connect pretrained LLMs with vision encoders through projection modules such as MLPs and Q-Formers (Liu et al. 2023; Zhu et al. 2023; Li et al. 2023a). Recent VLMs extend visual understanding from standard images to high-resolution images and long videos (Liu et al. 2024a,b; Bai et al. 2025; Clark et al. 2026; Liu et al. 2026). As image resolution and video duration increase, the resulting visual sequences can substantially exceed the corresponding text sequences, making visual token reduction important for eficient VLM inference.

![](images/f4322ac15200e25ffab47d087d8f2ea23ea1a877b294402d0567f70cd32cbe11.jpg)  
(a) Text Sensitivity of Variance-Ranked Head Subsets

![](images/1ff0191b81271e38e2858f6ad76160db903cb35e7a4696b2df3096f99e92beda.jpg)  
(b) Text-to-Vision Attention across Token Positions  
Figure 3: Head sensitivity and positional attention patterns. (a) Mean textual sensitivity of cumulative head subsets ranked by text-to-vision attention variance. (b) Average text-to-vision attention across visual-token positions at selected LLM layers.

## 2.2 Visual Token Reduction for VLMs

Existing methods difer primarily in where token reduction occurs and when the text query becomes available.

Vision-guided pre-LLM reduction. These methods remove redundant visual tokens after the vision encoder using visual-side signals such as saliency, similarity, or diversity (Alvar et al. 2025). VisionZip (Yang et al. 2025) retains dominant tokens according to vision-encoder saliency and merges residual tokens into contextual representations. VisPruner (Zhang et al. 2025a) combines vision-encoder attention with similarity-based deduplication. Although these methods reduce computation before the LLM, their selection remains independent of the text query.

Query-aware in-LLM pruning. These methods prune within the LLM using text-to-vision cross-attention to retain task-relevant visual information. FastV (Chen et al. 2024) performs attention-based pruning in shallow LLM layers, PyramidDrop (Xing et al. 2024) progressively reduces tokens as visual redundancy increases with depth, and SparseVLM (Zhang et al. 2025b) combines text-guided attention with rank-adaptive sparsity. Since all visual tokens first enter the LLM, pruning often occurs in very shallow layers to provide meaningful computational savings. At these layers, text-to-vision cross-attention is susceptible to attention shift and dispersion (Zhang et al. 2025a), which can cause substantial accuracy loss.

Two-stage pruning. Recent methods combine pre-LLM visual reduction with query-guided refinement inside the LLM (Zhang et al. 2026; Singh et al. 2026; Takezoe et al. 2026). DUET-VLM (Singh et al. 2026) first merges redundant tokens through local clustering and then performs layer-wise pruning using cross-modal attention. LearnPruner (Takezoe et al. 2026) replaces vision-encoder attention with a learnable pre-LLM importance predictor and performs textguided pruning in intermediate LLM layers. This design reduces early computation while allowing the second stage to use more mature cross-modal interactions.

Existing two-stage methods typically separate visualonly pre-LLM reduction from query-aware in-LLM pruning. In contrast, TReVS coordinates query guidance across both stages: textual relevance complements vision-encoder saliency before the LLM, while query-sensitive attention further refines the retained tokens within the LLM. This design reduces visual redundancy without depriving the second stage of query-relevant context.

## 3 Preliminary Analysis

## 3.1 Does textual guidance improve performance?

Two-stage methods typically use vision-encoder saliency or diversity signals for pre-LLM pruning and query-aware attention for in-LLM pruning. This mismatch creates an irreversible bottleneck: query-relevant evidence removed before the LLM cannot be recovered later. We therefore ask: Can textual guidance during pre-LLM pruning preserve such evidence and improve performance?

We evaluate LLaVA-1.5-7B on MME (Fu et al. 2023), POPE (Li et al. 2023b), TextVQA (Singh et al. 2019), and GQA (Hudson and Manning 2019). Under a fixed token budget, we vary the allocation between vision-encoder saliency, measured by [CLS] attention in the penultimate ViT layer, and textual relevance, measured by cosine similarity between projected visual tokens and text embeddings.

Figure 2(a) illustrates their complementary behavior. Vision-encoder saliency favors the dominant fire hydrant, while textual relevance recovers the car targeted by the query. Their integration retains both sources of evidence.

As shown in Figure 2(b), allocating even a modest fraction of the budget to textual relevance consistently outperforms visual-only pruning. Figure 2(c) further shows negative Spearman correlations between the two scores among the top-K saliency-selected tokens across all datasets and budgets. Thus, textual relevance reorders even visually salient candidates and complements, rather than replaces, visionencoder saliency. Together, these results show that textual guidance contributes complementary selection information and improves performance under a fixed token budget.

## 3.2 Are High-Variance Attention Heads More Query-Sensitive?

Several in-LLM pruning methods average text-to-vision attention across all heads (Chen et al. 2024; Takezoe et al. 2026), although heads difer in how strongly they respond to the query. We therefore ask: Can query-sensitive heads be identified with a training-free criterion?

A concentrated attention distribution has high variance across visual tokens, but concentration alone does not imply query sensitivity. We therefore test whether attention variance ranks heads whose visual-token attention changes more with the question. We pair 1,000 TextVQA (Singh et al. 2019) images with two distinct questions and process them using LLaVA-1.5-7B. At each layer, we rank the 32 heads by the variance of their text-to-vision attention and evaluate cumulative top-4, top-8, top-12, and all-head subsets. Let $ { \mathbf { s } } _ { h } ^ { ( 1 ) }$ and $\mathbf { s } _ { h } ^ { ( 2 ) }$ be the visual-token attention distributions of head h for the two questions. We define its textual sensitivity as

$$
\mathrm { T S } _ { h } = 1 - \cos \left( \mathbf { s } _ { h } ^ { ( 1 ) } , \mathbf { s } _ { h } ^ { ( 2 ) } \right) .\tag{1}
$$

Figure 3(a) shows that the highest-variance subsets are generally more text-sensitive, with the clearest separation in the middle layers. Since the subsets are cumulative, the decrease from top-4 to all 32 heads shows that adding lowerranked heads reduces the mean per-head sensitivity. Attention variance therefore provides a simple criterion for selecting query-sensitive heads.

Figure 3(b) compares average text-to-vision attention across visual-token positions at diferent layers. In the earliest layers, attention generally increases with token position, favoring visual tokens closer to the subsequent text sequence. Although all layers exhibit peaks near the sequence boundaries, later layers show flatter profiles across the interior positions, suggesting weaker positional dependence. This observation motivates delaying attention-based pruning beyond the earliest layers.

Together, these analyses motivate two design choices: using high-variance heads for query-sensitive token scoring and pruning at a shallow-to-middle layer to avoid strong earlylayer position dependence.

## 4 Method

In this section, we present TReVS, a training-free framework that coordinates pre-LLM and in-LLM visual token pruning (Figure 4). Stage 1 retains $K _ { 1 }$ visually salient, textually relevant, and diverse tokens before the LLM. Stage 2 uses highvariance attention heads to retain $K _ { 2 }$ query-relevant tokens at a shallow-to-middle LLM layer. The following subsections detail the two stages.

## 4.1 Query-Aware Visual Redundancy Reduction

Visual saliency score. To preserve visually salient information, we follow prior works (Yang et al. 2025) and leverage [CLS]-attention from the penultimate ViT layer as the visionencoder saliency signal. Let $x _ { [ \mathrm { C L S } ] } \in \mathbb { R } ^ { d }$ denote the [CLS] token, $X _ { v } ~ \in ~ \mathbb { R } ^ { N \times d }$ the $N$ patch token features output by the ViT, and $W _ { Q } ^ { h } , W _ { K } ^ { h } \in \mathbb { R } ^ { d \times d _ { h } }$ the query and key projection matrices for head $h ,$ where $d _ { h } = d / H$ is the per-head dimension. The [CLS]-to-patch attention for head h is:

$$
\begin{array} { r } { q _ { \mathrm { [ C L S ] } } = x _ { \mathrm { [ C L S ] } } W _ { Q } ^ { h } , \quad K _ { v } = X _ { v } W _ { K } ^ { h } , } \\ { A _ { \mathrm { [ C L S ] } } ^ { h } = \operatorname { S o f t m a x } \Biggl ( \frac { q _ { \mathrm { [ C L S ] } } K _ { v } ^ { \top } } { \sqrt { d _ { h } } } \Biggr ) . } \end{array}\tag{2}
$$

The visual saliency score for the i-th patch token is obtained by averaging the [CLS]-attention across all H heads:

$$
s _ { i } ^ { v } = \frac { 1 } { H } \sum _ { h = 1 } ^ { H } A _ { \mathrm { [ C L S ] } } ^ { h } [ i ] .\tag{3}
$$

Textual-relevance score. As demonstrated in Section 3.1, incorporating a text-guidance signal at the pre-LLM stage is critical for retaining the text-relevant visual tokens that the LLM stage requires. We first project the ViT output through the visual projector to obtain text-aligned representations: $Z = { \mathrm { P r o j e c t o r } } ( X _ { v } ) \in \mathbb { R } ^ { N \times d ^ { \prime } }$ , where $z _ { i } \in \mathbb { R } ^ { d ^ { \prime } }$ is the projected feature of visual token i. Let $t _ { j } \in \mathbb { R } ^ { d ^ { \prime } }$ denote the embedding of the j-th text token, with M tokens in total. We normalize the visual and text features as $\bar { z } _ { i } = z _ { i } / \| z _ { i } \| _ { 2 }$ and $\bar { t } _ { j } = t _ { j } / \| t _ { j } \| _ { 2 }$ , respectively. We then compute the rectified cosine similarity between each text–visual pair:

$$
m _ { j , i } = \operatorname* { m a x } ( \bar { t } _ { j } ^ { \top } \bar { z } _ { i } , 0 ) .\tag{4}
$$

The query-relevance score for visual token i is the Root Mean Square (RMS) aggregation over all text tokens:

$$
s _ { i } ^ { t } = \sqrt { \frac { 1 } { M } \sum _ { j = 1 } ^ { M } m _ { j , i } ^ { 2 } } .\tag{5}
$$

This aggregation suppresses background noise from weakly related text tokens while amplifying tokens that are broadly attended to by the query.

Normalization and temperature scaling. Since $s _ { i } ^ { v }$ and $s _ { i } ^ { t }$ are drawn from distributions with diferent scales and statistics, we independently apply robust normalization to each before fusion. We use the Median Absolute Deviation (MAD) as a spread estimator, which is resistant to the attention outliers commonly observed in [CLS] Attention:

$$
\hat { s } _ { i } ^ { c } = \left[ \frac { s _ { i } ^ { c } - \mathrm { M e d } ( \mathbf { s } ^ { c } ) } { \mathrm { M A D } ( \mathbf { s } ^ { c } ) + \epsilon } \right] _ { + } \bigg / \tau _ { c } , \quad c \in \{ v , t \} .\tag{6}
$$

Here, $[ x ] _ { + } = \operatorname* { m a x } ( x , 0 )$ clamps negative values to zero. $\mathrm { M e d } ( \mathrm { \bf s } ^ { c } ) = \mathrm { m e d i a n } ( s _ { i } ^ { c } )$ denotes the median score across all visual tokens, while $\mathrm { M A D } ( \mathbf { s } ^ { c } ) = \mathrm { m e d i a n } \left( | s _ { i } ^ { c } - \mathrm { M e d } ( \mathbf { s } ^ { c } ) | \right)$ measures the median absolute deviation from this center. The constant ϵ ensures numerical stability, and $\tau _ { c }$ controls the temperature scaling, with $\tau _ { v } ~ = ~ 1 . 4$ and $\tau _ { t } ~ = ~ 1 . 0$ by default.

Pivot token selection. Visually salient tokens do not necessarily exhibit high textual relevance, suggesting that the two signals capture complementary evidence. We therefore define a unified score that preserves tokens favored by either signal and adds a reward when both scores are high:

![](images/72a0b652c05e9b75f70a93750ab63e5cfc3320cd6ed1c60ef5122484cc27ab16.jpg)  
Figure 4: The architecture of TReVS. Stage 1 performs query-aware pre-LLM pruning by preserving visually salient, queryrelevant, and diverse tokens. Stage 2 uses high-variance attention heads to guide in-LLM visual token pruning.

$$
\begin{array} { r } { s _ { i } = \operatorname* { m a x } \ ( \hat { s } _ { i } ^ { v } , \hat { s } _ { i } ^ { t } ) + \lambda \sqrt { \hat { s } _ { i } ^ { v } \cdot \hat { s } _ { i } ^ { t } } , } \end{array}\tag{7}
$$

where $\lambda _ { \mathrm { V } } \sqrt { \hat { s } _ { i } ^ { v } \hat { s } _ { i } ^ { t } }$ is the consistency reward, with $\lambda = 1 . 0$ by default. We retain the top-K<sub>r</sub> tokens as pivot tokens $\begin{array} { r } { S _ { r } = } \end{array}$ $\mathrm { T o p K } ( s _ { i } , K _ { r } )$

Context preservation. To mitigate the loss of background information, we sample $K _ { d }$ tokens from the remaining candidates $\mathcal { C } = \{ 1 , 2 , \ldots , \bar { N } \} \setminus \mathcal { S } _ { \tau }$ using Farthest Point Sampling (FPS). FPS iteratively selects the token farthest from the selected tokens in cosine distance, maximizing feature diversity. The resulting set is $\begin{array} { r } { S _ { d } = \mathrm { F P S } ( \{ z _ { i } \mid i \in \mathcal { C } \} , K _ { d } ) } \end{array}$ . The union $S _ { 1 } = S _ { r } \cup { \mathsf { \bar { S } } } _ { d }$ of $K _ { 1 } = K _ { r } + \dot { K _ { d } }$ tokens is passed into the LLM for subsequent processing.

## 4.2 Query-Driven Visual Token Compression

After Stage 1 pruning, $K _ { 1 }$ visual tokens are concatenated with the text tokens and fed into the LLM. As the forward pass progresses, the text tokens gradually absorb task-relevant visual information through cross-modal attention, causing the visual tokens to become increasingly redundant (Kaduri, Bagon, and Dekel 2025). To further reduce computational overhead, we perform a second pruning step at layer $l ^ { * }$ , retaining $K _ { 2 }$ visual tokens for subsequent layers.

High-variance head selection. As shown in Section 3.2, heads with higher text-to-vision attention variance are more sensitive to changes in the text query. We therefore use attention variance as a training-free criterion for identifying query-sensitive heads at the second-stage pruning layer $l ^ { * }$ Let $\mathbf { \bar { \boldsymbol { A } } } _ { T  V } ^ { h } \in \mathbb { R } ^ { L _ { \mathrm { t e x t } } \times K _ { 1 } }$ denote the text-to-vision attention matrix of head h at this layer, where $q$ indexes text tokens and i indexes visual tokens. The attention variance of head $h$ is:

$$
u _ { h } = \frac { 1 } { L _ { \mathrm { t e x t } } } \sum _ { q = 1 } ^ { L _ { \mathrm { t e x t } } } \mathrm { V a r } _ { i } \big ( A _ { T  V } ^ { h } [ q , i ] \big ) .\tag{8}
$$

Here, $\operatorname { V a r } _ { i }$ computes the variance across the $K _ { 1 }$ visual-token positions for each text token. Averaging over all text tokens yields $u _ { h }$ , which measures the overall visual-token selectivity of head h. We select the top half of the heads ranked by $u _ { h }$ to retain query-sensitive attention signals while maintaining suficient head coverage for stable visual-token scoring. The selected heads form the high-variance head set $\mathcal { H } ^ { \ast }$ used in the subsequent token selection.

Query-irrelevant token reduction. Using $\mathcal { H } ^ { \ast }$ , we score each visual token by its maximum text-to-vision attention across all text positions, averaged over the selected heads:

$$
p _ { i } = \operatorname* { m a x } _ { \boldsymbol { q } } \ \frac { 1 } { | \mathcal { H } ^ { * } | } \sum _ { h \in \mathcal { H } ^ { * } } A _ { T  V } ^ { h } [ \boldsymbol { q } , i ] .\tag{9}
$$

We retain the top-K tokens by this score $\begin{array} { r l } { S _ { 2 } } & { { } = } \end{array}$ $\mathrm { T o p K } ( p _ { i } , \ K _ { 2 } )$ . The resulting $K _ { 2 }$ visual tokens are then forwarded through the remaining LLM layers.

Overall, the two stages form a progressive query-aware pruning process. By introducing textual relevance into pre-LLM pruning, TReVS preserves query-relevant visual evidence before irreversible reduction, ensuring that the subsequent in-LLM stage operates on an informative visual context. Within the LLM, high-variance heads provide more discriminative text-to-vision attention, enabling the in-LLM stage to better retain task-relevant visual tokens.

## 5 Experiments

We evaluate TReVS across image, high-resolution, and video understanding to examine whether its query-aware twostage design remains efective across diferent visual-token lengths. We then analyze its inference eficiency.

## 5.1 Experimental Setup

Models and benchmarks. We evaluate TReVS on LLaVA-1.5-7B (Liu et al. 2024a), LLaVA-NeXT-7B (Liu et al. 2024b), and Video-LLaVA-7B (Lin et al. 2023), which process 576, 2,880 and 2,048 visual tokens, respectively. For LLaVA-1.5-7B, we use GQA (Hudson and Manning 2019), ScienceQA-IMG (Lu et al. 2022), TextVQA (Singh et al. 2019), POPE (Li et al. 2023b), MME (Fu et al. 2023), and the English and Chinese splits of MMBench (Liu et al. 2024c).

We evaluate LLaVA-NeXT-7B on GQA, TextVQA, MME, and MMBench, and Video-LLaVA-7B on TGIF-QA (Jang et al. 2017), MSVD-QA, and MSRVTT-QA (Xu et al. 2017).

Implementation details. TReVS is training-free and leaves all parameters of the underlying VLM unchanged. Unless otherwise specified, we perform the second pruning stage after the eighth LLM layer and retain the top half of the attention heads according to their text-to-vision attention variance. We set the ratio between pivot and diversity tokens to $K _ { r } : K _ { d } = 3 : 1$ and maintain $K _ { 1 } : K _ { 2 } = \dot { 3 } : 1$ between the two pruning stages. The visual and textual temperatures are set to $\tau _ { v } = 1$ .4 and $\tau _ { t } = 1 . 0$ , respectively. The consistency-reward weight is set to λ = 1.0.

## 5.2 Main Results

Tables 1, 2, and 3 compare TReVS with existing visual token pruning methods across image and video VLMs.

Results on LLaVA-1.5-7B. As shown in Table 1, TReVS achieves the highest RelAcc. at all three budgets, retaining 98.8%, 96.5%, and 92.8% of the unpruned performance. Notably, TReVS consistently obtains the best MME scores across these budgets, which indicates stable preservation of visual evidence required for diverse perception and cognition tasks. Moreover, its margin over the second-best RelAcc. increases from 0.6% at 128 tokens to 1.2% at 32 tokens. This widening margin is consistent with the two-stage design of TReVS. Before the LLM, textual relevance prioritizes queryrelevant tokens, reducing critical information loss when only a small number of tokens can be retained. Within the LLM, query-sensitive heads provide more discriminative attention for selecting task-relevant tokens, making more efective use of the limited remaining token budget.

Hallucination robustness. Figure 5 compares POPE performance across token budgets. While the methods perform similarly at moderate budgets, TReVS achieves 82.7% accuracy with 32 tokens, outperforming DivPrune and VScan by 1.2% and 2.8%. This robustness is consistent with our query-aware pre-LLM design, which preserves evidence for queried objects before irreversible pruning and thereby mitigates hallucinations caused by missing visual evidence.

![](images/c6b7b3d189e5bc385b0c39da655b0108685d76a28d54251f542b8072db4add8f.jpg)  
Figure 5: POPE accuracy on LLaVA-1.5-7B across retainedtoken budgets. While all methods perform similarly at moderate budgets, TReVS degrades more slowly under increasingly aggressive compression and achieves the best performance at 32 tokens. The dashed line denotes the unpruned model.

Results on LLaVA-NeXT-7B. Table 2 evaluates TReVS on the longer visual sequences produced by high-resolution inputs. TReVS achieves the highest RelAcc. at both budgets, retaining 96.6% and 93.0% of the unpruned performance with 320 and 160 tokens, respectively. The best MME scores at both budgets and the best MMB score at 160 tokens further show that TReVS preserves key visual evidence under substantial compression. Despite these overall gains, TReVS trails the best GQA results by 0.5% and 0.6% under highresolution inputs, whereas the gap is less pronounced with lower-resolution inputs. GQA requires global reasoning over multiple objects and their relations, and such evidence may be distributed across longer visual sequences. We hypothesize that high-variance heads prioritize sparse query-relevant details and therefore retain slightly less global context, making this trade-of more evident at higher resolutions.

Results on Video-LLaVA-7B. Table 3 evaluates whether TReVS generalizes from images to longer video inputs. With 93.4% of visual tokens removed, TReVS achieves the best result on all three benchmarks and retains 99% of the average unpruned performance, exceeding DUET-VLM by 3.2%. These results demonstrate that query-aware two-stage pruning remains efective for highly redundant video sequences.

## 5.3 Ablation Study

Table 4 isolates the contributions of the two pruning stages under the 32-token preset. Replacing [CLS]-attention-only pre-LLM pruning with the first stage of TReVS improves RelAcc. from 91.8% to 92.7% while retaining all attention heads in the second stage. This gain confirms that combining textual relevance with visual saliency and token diversity better preserves query-relevant and complementary evidence before irreversible token reduction. Selecting high-variance heads further increases RelAcc. to 93.2%, demonstrating that high-variance heads provide more discriminative queryaware signals than uniformly using all heads. Together, the two stages improve RelAcc. by 1.4%, confirming their complementary roles in preventing premature information loss and removing residual task-irrelevant tokens.

## 5.4 Eficiency Analysis

Computational eficiency. Table 5 reports the inference eficiency ofTReVS on POPE with LLaVA-1.5-7B, measured on an NVIDIA RTX 4090 GPU. At 32 tokens, TReVS achieves a 2.2× prefill speedup and a 1.4× end-to-end speedup while reducing KV-cache consumption by 6.5×. Together with the accuracy retained under the same budget in Table 1, these results demonstrate a favorable accuracy–eficiency trade-of under aggressive compression.

<table><tr><td>Method</td><td>Token</td><td>Total Time</td><td>Prefill Time</td><td>KV Cache (MB)</td></tr><tr><td>Vanilla</td><td>576</td><td>126.5 (1.0×)</td><td>72.7 (1.0×)</td><td>321.1 (1.0×)</td></tr><tr><td rowspan="3">Ours</td><td>128</td><td>100.7 (1.3×)</td><td>40.6 (1.8×)</td><td>99.7 (3.2×)</td></tr><tr><td>64</td><td>90.4 (1.4×)</td><td>33.7 (2.2×)</td><td>66.2 (4.9×)</td></tr><tr><td>32</td><td>88.9 (1.4×)</td><td>33.5 (2.2×)</td><td>49.7 (6.5×)</td></tr></table>

Table 5: Inference eficiency on POPE with LLaVA-1.5-7B. Latency is measured per sample in milliseconds, and parenthesized values report speedup or KV-cache reduction over dense inference.

<table><tr><td>Method</td><td>GQA</td><td> $\mathbf { S Q A } ^ { I }$   $\mathbf { V Q A } ^ { T }$ </td><td>POPE</td><td>MME</td><td>MMB</td><td> ${ \bf M M B } ^ { C N }$ </td><td>RelAcc.</td></tr><tr><td colspan="8">Upper Bound, 576 Tokens (100%)</td></tr><tr><td>Vanilla</td><td>61.9 69.5</td><td>58.2</td><td>85.9</td><td>1862</td><td>64.7</td><td>58.3</td><td>100.0%</td></tr><tr><td colspan="8">Retain Averaged 128 Tokens (↓77.8%)</td></tr><tr><td>FastV (ECCV 2024)</td><td>49.6 60.2</td><td>50.6 54.9</td><td>59.6 80.5</td><td>1490 1696</td><td>56.1 60.0</td><td>51.4 51.1</td><td>82.6% 92.4%</td></tr><tr><td>SparseVLM (ICML 2025)</td><td>56.0 59.3</td><td>67.1 69.0</td><td>86.7</td><td>1718</td><td>62.0</td><td>54.8</td><td>96.4%</td></tr><tr><td>DivPrune (CVPR 2025)</td><td></td><td>56.1</td><td>83.2</td><td>1762</td><td>62.0</td><td>56.7</td><td>96.3%</td></tr><tr><td>VisionZip (CVPR 2025)</td><td>57.6 68.9</td><td>56.8</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>VScan (TMLR 2026)</td><td>59.8 68.9</td><td>57.3</td><td>86.1</td><td>1792</td><td>63.0</td><td>58.0</td><td>98.2%</td></tr><tr><td>DUET-VLM (CVPR 2026)</td><td>59.0 70.2</td><td>57.8</td><td>85.9</td><td>1767</td><td>63.3</td><td>56.7</td><td>97.9%</td></tr><tr><td>TReVS</td><td>60.3 68.9</td><td>57.8</td><td>86.5</td><td>1842</td><td>63.6</td><td>57.2</td><td>98.8%</td></tr><tr><td colspan="8">Retain Averaged 64 Tokens (↓88.9%)</td></tr><tr><td>FastV (ECCV 2024)</td><td>46.1 52.7</td><td>51.1 47.8 62.2 51.8</td><td>48.0 75.1</td><td>1256 1505</td><td>48.0 56.2</td><td>42.7 46.1</td><td>71.6%</td></tr><tr><td>SparseVLM (ICML 2025)</td><td>57.8 68.2</td><td>54.7</td><td>85.6</td><td>1674</td><td>59.3</td><td>52.3</td><td>85.4%</td></tr><tr><td>DivPrune (CVPR 2025)</td><td>55.1 69.0</td><td>55.5</td><td>77.0</td><td>1690</td><td></td><td></td><td>93.8%</td></tr><tr><td>VisionZip (CVPR 2025)</td><td></td><td></td><td></td><td></td><td>60.1</td><td>55.4</td><td>93.1%</td></tr><tr><td>VScan (TMLR 2026)</td><td>58.3 69.1</td><td>55.6</td><td>85.0</td><td>1698</td><td>62.1</td><td>55.7</td><td>95.8%</td></tr><tr><td>DUET-VLM (CVPR 2026)</td><td>56.7 68.3</td><td>56.4</td><td>82.5</td><td>1751</td><td>62.0</td><td>56.3</td><td>95.6%</td></tr><tr><td>TReVS</td><td>57.8 68.9</td><td>56.6</td><td>85.2</td><td>1769</td><td>61.9</td><td>56.0</td><td>96.5%</td></tr><tr><td colspan="8">Retain Averaged 32 Tokens (、 (↓94.4%)</td></tr><tr><td>FastV (ECCV 2024)</td><td>41.5</td><td>42.6 57.3</td><td>42.5 32.5</td><td>1090</td><td>37.8</td><td>33.2</td><td>59.0%</td></tr><tr><td>SparseVLM (ICML 2025)</td><td>48.3</td><td>46.1</td><td>67.9</td><td>1290</td><td>51.4</td><td>40.6</td><td>76.7%</td></tr><tr><td>DivPrune (CVPR 2025)</td><td>54.9 68.6</td><td>52.9</td><td>81.5</td><td>1611</td><td>57.6</td><td>49.1</td><td>90.4%</td></tr><tr><td>VisionZip (CVPR 2025)</td><td>51.8 68.8</td><td>53.1</td><td>68.7</td><td>1536</td><td>57.7</td><td>50.3</td><td>87.4%</td></tr><tr><td>VScan (TMLR 2026)</td><td>54.8 69.4</td><td>53.9</td><td>79.9</td><td>1598</td><td>59.5</td><td>51.9</td><td>91.5%</td></tr><tr><td>DUET-VLM (CVPR 2026)</td><td>53.6 69.1</td><td>54.7</td><td>74.8</td><td>1633</td><td>60.0</td><td>54.8</td><td>91.6%</td></tr><tr><td></td><td></td><td>53.9</td><td>82.7</td><td>1651</td><td>61.4</td><td>51.7</td><td></td></tr><tr><td>TReVS</td><td>55.0</td><td>69.0</td><td></td><td></td><td></td><td></td><td>92.8%</td></tr></table>

Table 1: Performance comparison on LLaVA-1.5-7B under matched average token budgets. RelAcc. averages benchmark-wise performance relative to the unpruned model. Bold and underlined entries denote the best and second-best results.

<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=3>GQA  $\mathbf { V Q A } ^ { T }$  MMEMMB RelAcc.</td></tr><tr><td rowspan=1 colspan=4>Upper Bound, 2,880 Tokens (100%)</td></tr><tr><td rowspan=1 colspan=1>Vanilla</td><td rowspan=1 colspan=2>64.2   61.3   1842   67.9</td><td rowspan=1 colspan=1>100.0%</td></tr><tr><td rowspan=1 colspan=1>Retai</td><td rowspan=1 colspan=2>n Averaged 320 Tokens ((↓88.9%)</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>SparseVLM</td><td rowspan=1 colspan=2>57.7   55.9   1694   64.3</td><td rowspan=1 colspan=1>91.9%</td></tr><tr><td rowspan=1 colspan=1>DivPrune</td><td rowspan=1 colspan=2>61.1   56.2   1724   63.9</td><td rowspan=1 colspan=1>93.6%</td></tr><tr><td rowspan=1 colspan=1>VisionZip</td><td rowspan=1 colspan=2>59.3   58.9   1702   63.1</td><td rowspan=1 colspan=1>93.4%</td></tr><tr><td rowspan=1 colspan=1>DUET-VLM</td><td rowspan=1 colspan=2>60.6   59.9   1788   64.3</td><td rowspan=1 colspan=1>96.0%</td></tr><tr><td rowspan=1 colspan=1>VScan</td><td rowspan=1 colspan=2>61.4   59.4   1775   65.5</td><td rowspan=1 colspan=1>96.3%</td></tr><tr><td rowspan=1 colspan=1>TReVS</td><td rowspan=1 colspan=2>60.9   59.3   1826  64.9</td><td rowspan=1 colspan=1>96.6%</td></tr><tr><td rowspan=1 colspan=1>Retai</td><td rowspan=1 colspan=2>n Averaged 160 Tokens (↓94.4%)</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>SparseVLM</td><td rowspan=1 colspan=1>51.2   46.4   1542</td><td rowspan=1 colspan=1>63.1</td><td rowspan=2 colspan=1>83.0%90.6%</td></tr><tr><td rowspan=2 colspan=1>DivPruneVisionZip</td><td rowspan=1 colspan=1>59.3   54.1   1643</td><td rowspan=1 colspan=1>62.9</td></tr><tr><td rowspan=1 colspan=1>55.5   56.2   1630</td><td rowspan=1 colspan=1>60.1</td><td rowspan=1 colspan=1>88.8%</td></tr><tr><td rowspan=1 colspan=1>DUET-VLM</td><td rowspan=1 colspan=2>58.6   57.2   1686  62.9</td><td rowspan=1 colspan=1>92.2%</td></tr><tr><td rowspan=1 colspan=1>VScan</td><td rowspan=1 colspan=2>59.6   57.7   1699   62.0</td><td rowspan=1 colspan=1>92.6%</td></tr><tr><td rowspan=1 colspan=1>TReVS</td><td rowspan=1 colspan=2>59.0   56.9   1717   63.9</td><td rowspan=1 colspan=1>93.0%</td></tr></table>

Table 2: Performance comparison on LLaVA-NeXT-7B under matched average token budgets.

## 6 Conclusion

This work identifies an irreversible information bottleneck in two-stage visual token pruning: query-agnostic pre-LLM reduction can discard task-relevant evidence before queryaware reasoning begins. We introduce TReVS, a training-free framework that coordinates vision-encoder saliency, textual relevance, token diversity, and high-variance attention heads across pre-LLM and in-LLM pruning. Our findings highlight two key principles for efective visual token pruning: preserving query-relevant evidence before the LLM and using query-sensitive attention heads to identify task-relevant tokens within the LLM. Future work may explore adaptive token allocation between the pre-LLM and in-LLM stages based on input complexity and query demands.

<table><tr><td>Method</td><td>TGIF</td><td>MSVD</td><td>MSRVTT</td><td>RelAcc.</td></tr><tr><td colspan="5">Upper Bound, 2,048 Tokens (100%)</td></tr><tr><td>Video-LLaVA</td><td>48.7</td><td>70.1</td><td>57.4</td><td>100.0%</td></tr><tr><td colspan="3">Retain Averaged 136 Tokens (↓93.4%)</td><td></td><td></td></tr><tr><td>FastV</td><td>30.4</td><td>44.3</td><td>39.4</td><td>64.8%</td></tr><tr><td>SparseVLM</td><td>44.7</td><td>68.2</td><td>31.0</td><td>81.0%</td></tr><tr><td>VisionZip</td><td>42.4</td><td>63.5</td><td>52.1</td><td>89.5%</td></tr><tr><td>DUET-VLM</td><td>46.7</td><td>68.0</td><td>54.2</td><td>95.8%</td></tr><tr><td>TReVS</td><td>48.9</td><td>69.2</td><td>56.2</td><td>99.0%</td></tr></table>

Table 3: Performance comparison on Video-LLaVA-7B with 136 average retained tokens.

<table><tr><td>Stage 1</td><td>Stage 2</td><td>RelAcc. (%)</td></tr><tr><td>[CLS] Attn</td><td>All Heads</td><td>91.8</td></tr><tr><td rowspan="2">TReVS</td><td>All Heads</td><td>92.7</td></tr><tr><td>High-Variance Heads</td><td>93.2</td></tr></table>

Table 4: Ablation study of the two-stage token-pruning strategies on LLaVA-1.5-7B under the 32-token preset. RelAcc. denotes the average relative accuracy over TextVQA, MM-Bench, GQA, and POPE.

## References

Achiam, J.; Adler, S.; Agarwal, S.; Ahmad, L.; Akkaya, I.; Aleman, F. L.; Almeida, D.; Altenschmidt, J.; Altman, S.; Anadkat, S.; et al. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774.

Alvar, S. R.; Singh, G.; Akbari, M.; and Zhang, Y. 2025. Divprune: Diversity-based visual token pruning for large multimodal models. In Proceedings ofthe Computer Vision and Pattern Recognition Conference, 9392–9401.

Bai, J.; Bai, S.; Chu, Y.; Cui, Z.; Dang, K.; Deng, X.; Fan, Y.; Ge, W.; Han, Y.; Huang, F.; et al. 2023. Qwen technical report. arXiv preprint arXiv:2309.16609.

Bai, S.; Chen, K.; Liu, X.; Wang, J.; Ge, W.; Song, S.; Dang, K.; Wang, P.; Wang, S.; Tang, J.; Zhong, H.; Zhu, Y.; Yang, M.; Li, Z.; Wan, J.; Wang, P.; Ding, W.; Fu, Z.; Xu, Y.; Ye, J.; Zhang, X.; Xie, T.; Cheng, Z.; Zhang, H.; Yang, Z.; Xu, H.; and Lin, J. 2025. Qwen2.5-VL Technical Report. arXiv preprint arXiv:2502.13923.

Bolya, D.; Fu, C.-Y.; Dai, X.; Zhang, P.; Feichtenhofer, C.; and Hofman, J. 2023. Token Merging: Your ViT But Faster. In International Conference on Learning Representations.

Chen, L.; Zhao, H.; Liu, T.; Bai, S.; Lin, J.; Zhou, C.; and Chang, B. 2024. An image is worth 1/2 tokens after layer 2: Plug-and-play inference acceleration for large vision-language models. In European Conference on Computer Vision, 19–35. Springer.

Clark, C.; Zhang, J.; Ma, Z.; Park, J. S.; Salehi, M.; Tripathi, R.; Lee, S.; Ren, Z.; Kim, C. D.; Yang, Y.; Shao, V.; Yang, Y.; Huang, W.; Gao, Z.; Anderson, T.; Zhang, J.; Jain, J.; Stoica, G.; Han, W.; Farhadi, A.; and Krishna, R. 2026. Molmo2: Open Weights and Data for Vision-Language Models with Video Understanding and Grounding. arXiv:2601.10611.

Fu, C.; Chen, P.; Shen, Y.; Qin, Y.; Zhang, M.; Lin, X.; Yang, J.; Zheng, X.; Li, K.; Sun, X.; et al. 2023. MME: A Comprehensive Evaluation Benchmark for Multimodal Large Language Models. arXiv preprint arXiv:2306.13394.

Hudson, D. A.; and Manning, C. D. 2019. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 6700–6709.

Jang, Y.; Song, Y.; Yu, Y.; Kim, Y.; and Kim, G. 2017. Tgif-qa: Toward spatio-temporal reasoning in visual question answering. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2758–2766.

Kaduri, O.; Bagon, S.; and Dekel, T. 2025. What’s in the Image? A Deep-Dive into the Vision of Vision Language Models. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 14549–14558. IEEE.

Li, B.; Zhang, Y.; Guo, D.; Zhang, R.; Li, F.; Zhang, H.; Zhang, K.; Zhang, P.; Li, Y.; Liu, Z.; et al. 2024. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326.

Li, J.; Li, D.; Savarese, S.; and Hoi, S. 2023a. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In International Conference on Machine Learning, 19730–19742. PMLR.

Li, Y.; Du, Y.; Zhou, K.; Wang, J.; Zhao, W. X.; and Wen, J.-R. 2023b. Evaluating object hallucination in large visionlanguage models. arXiv preprint arXiv:2305.10355.

Lin, B.; Ye, Y.; Zhu, B.; Cui, J.; Ning, M.; Jin, P.; and Yuan, L. 2023. Video-llava: Learning united visual representation by alignment before projection. arXiv preprint arXiv:2311.10122.

Liu, H.; Li, C.; Li, Y.; and Lee, Y. J. 2024a. Improved Baselines with Visual Instruction Tuning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.

Liu, H.; Li, C.; Li, Y.; Li, B.; Zhang, Y.; Shen, S.; and Lee, Y. J. 2024b. Llavanext: Improved reasoning, ocr, and world knowledge.

Liu, H.; Li, C.; Wu, Q.; and Lee, Y. J. 2023. Visual instruction tuning. Advances in neural information processing systems, 36: 34892–34916.

Liu, J.; Wang, Y.; Ma, H.; Wu, X.; Ma, X.; Wei, X.; Jiao, J.; Wu, E.; and Hu, J. 2026. Kangaroo: A Powerful Video-Language Model Supporting Long-context Video Input. International Journal ofComputer Vision, 134(3).

Liu, Y.; Duan, H.; Zhang, Y.; Li, B.; Zhang, S.; Zhao, W.; Yuan, Y.; Wang, J.; He, C.; Liu, Z.; et al. 2024c. Mmbench: Is your multi-modal model an all-around player? In European Conference on Computer Vision, 216–233. Springer.

Lu, P.; Mishra, S.; Xia, T.; Qiu, L.; Chang, K.-W.; Zhu, S.-C.; Tafjord, O.; Clark, P.; and Kalyan, A. 2022. Learn to explain: Multimodal reasoning via thought chains for science question answering. Advances in Neural Information Processing Systems, 35: 2507–2521.

Singh, A.; Natarajan, V.; Shah, M.; Jiang, Y.; Chen, X.; Batra, D.; Parikh, D.; and Rohrbach, M. 2019. Towards vqa models that can read. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 8317–8326.

Singh, A. K.; Kandala, H.; Brahma, P. P.; Liu, Z.; and Barsoum, E. 2026. DUET-VLM: Dual Stage Unified Eficient Token Reduction for VLM Training and Inference. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.

Takezoe, R.; Li, Y.; Bo, Z.; Hou, A.; Guang, M.; and Long, K. 2026. LearnPruner: Rethinking Attention-based Token Pruning in Vision Language Models. In International Conference on Learning Representations.

Touvron, H.; Lavril, T.; Izacard, G.; Martinet, X.; Lachaux, M.-A.; Lacroix, T.; Rozière, B.; Goyal, N.; Hambro, E.; Azhar, F.; Rodriguez, A.; Joulin, A.; Grave, E.; and Lample, G. 2023. LLaMA: Open and Eficient Foundation Language Models. arXiv preprint arXiv:2302.13971.

Xing, L.; Huang, Q.; Dong, X.; Lu, J.; Zhang, P.; Zang, Y.; Cao, Y.; He, C.; Wang, J.; Wu, F.; et al. 2024. Pyramiddrop: Accelerating your large vision-language models via pyramid visual redundancy reduction. arXiv preprint arXiv:2410.17247.

Xu, D.; Zhao, Z.; Xiao, J.; Wu, F.; Zhang, H.; He, X.; and Zhuang, Y. 2017. Video question answering via gradually refined attention over appearance and motion. In Proceedings of the 25th ACM International Conference on Multimedia, 1645–1653.

Yang, S.; Chen, Y.; Tian, Z.; Wang, C.; Li, J.; Yu, B.; and Jia, J. 2025. Visionzip: Longer is better but not necessary in vision language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 19792–19802.

Zhang, C.; Ma, K.; Fang, T.; Yu, W.; Zhang, H.; Zhang, Z.; Mi, H.; and Yu, D. 2026. VScan: Rethinking Visual Token Reduction for Eficient Large Vision-Language Models. Transactions on Machine Learning Research.

Zhang, Q.; Cheng, A.; Lu, M.; Zhang, R.; Zhuo, Z.; Cao, J.; Guo, S.; She, Q.; and Zhang, S. 2025a. Beyond Text-Visual Attention: Exploiting Visual Cues for Efective Token Pruning in VLMs. Proceedings of the IEEE International Conference on Computer Vision.

Zhang, Y.; Fan, C.-K.; Ma, J.; Zheng, W.; Huang, T.; Cheng, K.; Gudovskiy, D.; Okuno, T.; Nakata, Y.; Keutzer, K.; et al. 2025b. SparseVLM: Visual Token Sparsification for Eficient Vision-Language Model Inference. In International Conference on Machine Learning.

Zhu, D.; Chen, J.; Shen, X.; Li, X.; and Elhoseiny, M. 2023. Minigpt-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592.