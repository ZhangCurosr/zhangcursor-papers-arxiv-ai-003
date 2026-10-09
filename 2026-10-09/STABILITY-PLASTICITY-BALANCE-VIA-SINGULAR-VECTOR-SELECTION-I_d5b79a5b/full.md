# STABILITY–PLASTICITY BALANCE VIA SINGULAR-VECTOR SELECTION IN LLM CONTINUAL LEARNING

Lingxiang Wang<sup>1,3∗</sup>, Hainan Zhang<sup>1,3†</sup>, Liang Pang<sup>2</sup>, Hongwei Zheng<sup>3</sup>, Zhiming Zheng<sup>1,3</sup> <sup>1</sup>School of Artificial Intelligence, Beihang University

<sup>2</sup>Institute of Computing Technology, Chinese Academy of Sciences

<sup>3</sup>Beijing Advanced Innovation Center for Future Blockchain and Privacy Computing

## ABSTRACT

Domain-specific continual adaptation of LLMs risks catastrophic forgetting, creating a fundamental tension between acquiring new capabilities and preserving those learned during pretraining. PEFT mitigates this problem by restricting the number of trainable parameters, but existing methods lack a principled unit for deciding where plasticity should be allocated and stability should be preserved. We identify the singular-vector channel as a natural unit for managing this trade-off. Each channel represents an input–output transformation, which can be updated to acquire new knowledge or fixed to preserve pretrained capabilities. Based on this perspective, we introduce SVC, a parameter-efficient continual-learning method that selectively updates Singular-Vector Channels. Before fine-tuning, SVC uses domain-specific data to estimate each channel’s adaptation benefit and a fixed public general-domain corpus only as a history activation proxy for estimating forgetting cost. It then adaptively selects trainable channels based on these scores via knee-based cost screening, Pareto-front filtering, and Otsu thresholding. Experimental results across four LLM families and eight downstream tasks show that SVC better preserves pretrained capabilities while achieving strong downstream performance relative to existing PEFT baselines. Further analysis of channel scoring and selection demonstrates that selective plasticity at the singular-vectorchannel level enables effective continual LLM adaptation.

## 1 INTRODUCTION

Large language models (LLMs) gain broad knowledge and general-purpose reasoning through largescale pretraining (Karanikolas et al., 2023), but often require domain-specific fine-tuning to meet specialized data and task demands (Chen et al., 2025). However, such adaptation can cause catastrophic forgetting, degrading pretrained capabilities and reducing reliability in real-world settings (French, 1999; Liu et al., 2020). Therefore, continual domain adaptation requires a careful balance between plasticity, to acquire new domain knowledge, and stability, to preserve pretrained capabilities.

Traditional continual learning methods (Shi et al., 2025) mitigate forgetting primarily by preserving information from previous learning stages. Regularization-based approaches constrain parameter updates or functional changes to protect previously acquired knowledge (Kirkpatrick et al., 2017; Li & Hoiem, 2017), while replay-based approaches revisit stored or generated samples from earlier tasks (Rebuffi et al., 2017; Rolnick et al., 2019). Applying these strategies to LLMs is challenging because the original pretraining data and training signals are typically unavailable, making it difficult to directly preserve the broad capabilities acquired during pretraining. Moreover, overly strong constraints can impede adaptation to new domains, whereas weak constraints may fail to prevent forgetting. As a result, achieving a favorable stability-plasticity trade-off remains challenging for continual domain adaptation of LLMs.

![](images/48f1c890bd08217ce51fc2a908eabc034b14648fb4f54b68d0f0f1f18e6390c2.jpg)

![](images/5b0e82e2988ef813c52ed342341c2a8a1423f9c15d15d21020b1bb8592efc02e.jpg)  
Figure 1: Stability–plasticity trade-off between LoRA and SVF. (a) GSM8K retention across fine-tuning tasks. (b) Downstream performance. Details in Appendix C

Parameter-efficient fine-tuning (PEFT) offers a promising alternative by freezing most pretrained parameters and adapting only a small subset (Ding et al., 2023; Li et al., 2025). A prominent example is low-rank adaptation (LoRA) (Hu et al., 2022), which learns low-rank weight updates to enable effective downstream adaptation while often reducing degradation of pretrained capabilities (Bider man et al., 2024). However, the extent of such preservation can vary substantially across adaptation tasks. As shown in Figure 1(a), when the same pretrained model is independently adapted to each of the eight TRACE tasks (Wang et al., 2023b) using LoRA, the resulting models exhibit markedly different drops in GSM8K performance (Cobbe et al., 2021). This suggests that restricting the update rank alone does not provide consistent control over forgetting.

Singular value fine-tuning (SVF) (Sun et al., 2022) imposes a stronger structural constraint. Given a weight matrix decomposed by singular value decomposition, SVF updates only the singular values while keeping the corresponding left and right singular vectors fixed. Since the singular vectors determine the input and output directions of each rank-one component, whereas the singular values control their strengths (Muller et al., 2004), SVF modifies the magnitude of existing transformations without changing their directions. As shown in Figure 1(a), this stronger directional constraint yields substantially more consistent retention of GSM8K performance across tasks. However, Figure 1(b) shows that it also limits downstream adaptation performance.

These observations reveal a central tension in parameter-efficient adaptation: allowing transformation directions to change improves plasticity but can increase interference with pretrained capabilities, while fixing all directions improves stability at the cost of adaptation. This suggests that the key question is not whether transformation directions should be updated, but which directions should be updatedfor a given task. To enable such task-dependent control, we treat each paired left and right singular-vector direction associated with a singular value as a singular-vector channel, which serves as the basic unit for balancing adaptation benefit against forgetting cost.

In this paper, we introduce SVC, a parameter-efficient continual adaptation method that selectively updates Singular-Vector Channels. Before fine-tuning, SVC evaluates each channel using two task-dependent signals: its potential benefit for learning the current domain and its potential cost to previously acquired or general-domain capabilities. SVC then identifies candidate channels through knee-based cost screening and Pareto-front filtering in the benefit–cost plane, and applies Otsu thresholding to automatically select a task-specific subset for optimization. During adaptation, all singular values and the singular vectors of the selected channels are updated jointly, while the remaining singular-vector directions are kept fixed. In this way, SVC retains the structural stability of singular value fine-tuning while introducing directional flexibility only where it is beneficial for the current task.

Extensive experiments across four LLM families and eight downstream tasks show that SVC effectively adapts to sequential tasks while consistently preserving pretrained general capabilities. Further analyses validate the effectiveness of channel-level benefit-cost estimation and the multistage selection mechanism, showing that selective channel updates better balance model stability and plasticity. Our contributions are summarized as follows:

• We identify singular-vector channels as a principled unit for controlling the stabilityplasticity trade-off in parameter-efficient continual adaptation.

• We propose a parameter-efficient continual-learning method for LLMs that adaptively bal ances stability and plasticity across tasks.

• SVC requires neither prior-task samples nor task-specific modules, yet consistently performs well across models and task orders, demonstrating strong generality and scalability.

![](images/f7b8d404bab2655fe0119467bd786a3a17e2a5e2d27fab77ee1120e148ea0b77.jpg)  
Figure 2: Overview of SVC. SVC ⃝1 collects reference activation statistics from a publicly available, unlabeled general-domain reference corpus. It then ⃝2 computes layer-wise average gradients from current-task data. From these quantities, SVC ⃝3 estimates each singular-vector channel’s local adaptation benefit and forgetting cost. SVC next ⃝4 selects trainable channels through knee-based cost screening, Pareto-front filtering, and Otsu thresholding. Finally, SVC ⃝5 updates all singular values while updating singular vectors only in selected channels. Post-training, SVC merges the learned increments into the model weights.

## 2 RELATED WORK

Parameter-Efficient Fine-Tuning (PEFT) adapts LLMs through a small number of trainable parameters while keeping most pretrained weights fixed (Ding et al., 2023). Recent studies have shown that this restricted update regime can reduce catastrophic forgetting (Vu et al., 2022; Wang et al., 2025b), with LoRA exhibiting particularly favorable retention behavior (Biderman et al., 2024). O-LoRA (Wang et al., 2023a), a representative LoRA-based continual-learning method, learns successive tasks in orthogonal low-rank subspaces to reduce interference. However, its orthogonality constraint is defined only relative to the low-rank subspaces learned for previous tasks. It does not di rectly estimate whether changing a particular direction benefits the current domain or compromises pretrained general capabilities. Spectral PEFT methods provide another way to constrain adaptation by exploiting the singular-value structure of pretrained weights. SVF (Sun et al., 2022), SVDiff (Han et al., 2023), and SAM-PARSER (Peng et al., 2024) keep the left and right singular-vector bases fixed and optimize only the diagonal SVD coefficients, thereby rescaling the existing matched rank-one components. SVFT (Lingam et al., 2024) instead learns a sparse coefficient matrix over fixed singular-vector bases, including off-diagonal coefficients that combine different left and right singular vectors. Across these methods, the set of trainable spectral coefficients is specified before fine-tuning rather than selected using task-specific estimates of adaptation benefit and forgetting cost. SVC instead treats singular-vector channels as the basic units for weighing domain-adaptation benefits against general-capability forgetting costs and adaptively selects trainable directions for each domain.

## 3 METHOD

This section presents SVC, covering the preliminaries, the estimation of adaptation benefit and forgetting cost for singular-vector channels, Pareto-frontier-based adaptive channel selection, and the overall fine-tuning procedure. As illustrated in Figure 2, singular value decomposition represents each linear-layer weight matrix as a collection of rank-one components, each defined by a singular value and a corresponding pair of left and right singular vectors.

## 3.1 PRELIMINARIES

Continual learning requires a model to learn a sequence of tasks while preserving previously acquired capabilities. Consider a task sequence $\mathcal { T } \overset { \cdot } { = } \{ T _ { 1 } , T _ { 2 } , \ldots , T _ { T } \}$ , where task $T _ { t }$ is associated

with a dataset $\mathcal { D } _ { t } = \{ ( x _ { t , i } , y _ { t , i } ) \} _ { i = 1 } ^ { n _ { t } }$ . Let $\theta$ denote the parameters of the model $f ( \cdot ; \theta )$ . When training on task $T _ { t } .$ , the model is initialized with $\theta _ { t - 1 }$ , obtained after the preceding task, and updates its parameters by minimizing the current-task loss:

$$
\theta _ { t } = \arg \operatorname* { m i n } _ { \theta } \frac { 1 } { n _ { t } } \sum _ { j = 1 } ^ { n _ { t } } L _ { t } ( f ( x _ { t , j } ; \theta ) , y _ { t , j } ) ,\tag{1}
$$

where $L _ { t } ( \cdot , \cdot )$ denotes the loss function for task $T _ { t } , \mathrm { e . g . }$ ., cross-entropy.

We treat pretraining as an inaccessible stage that precedes $T _ { 1 }$ . When learning task $T _ { t }$ , the model has no access to the original pretraining data, pretraining statistics, or earlier-task data. We also keep the model size fixed across tasks to avoid accumulating task-specific modules and increasing storage or deployment overhead.

SVC follows the parameter-efficient fine-tuning paradigm and operates on linear layers. Consider the l-th linear layer before learning task $T _ { t }$ . Its inherited weight matrix is $\mathbf { W } _ { t - 1 } ^ { ( l ) } \in \overline { { \mathbb { R } } } ^ { d _ { \mathrm { o u t } } ^ { ( l ) } \times d _ { \mathrm { i n } } ^ { ( l ) } }$ . Let $\mathbf { X } _ { t , \mathrm { i n } } ^ { ( l ) } \in \mathbb { R } ^ { N \times d _ { \mathrm { i n } } ^ { ( l ) } }$ denote the input activations of this layer. Here, N denotes the number of activation vectors. The layer output is $\mathbf { X } _ { t , \mathrm { o u t } } ^ { ( l ) } = \mathbf { X } _ { t , \mathrm { i n } } ^ { ( l ) } \big ( \mathbf { W } _ { t - 1 } ^ { ( l ) } \big ) ^ { \top }$ . We decompose the weight matrix as $\mathbf { W } _ { t - 1 } ^ { ( l ) } =$ $\begin{array} { r } { \mathbf { U } _ { t - 1 } ^ { ( l ) } \Sigma _ { t - 1 } ^ { ( l ) } \big ( \mathbf { V } _ { t - 1 } ^ { ( l ) } \big ) ^ { \top } = \sum _ { i = 1 } ^ { r _ { l } } \sigma _ { t - 1 , i } ^ { ( l ) } \mathbf { u } _ { t - 1 , i } ^ { ( l ) } \mathbf { v } _ { t - 1 , i } ^ { ( l ) \top } , } \end{array}$ where $r _ { l } = \mathrm { r a n k } ( \mathbf { W } _ { t - 1 } ^ { ( l ) } )$ . The corresponding linear transformation can therefore be written as $\begin{array} { r } { \mathbf { X } _ { t , \mathrm { o u t } } ^ { ( l ) } = \sum _ { i = 1 } ^ { r _ { l } } \left( \mathbf { X } _ { t , \mathrm { i n } } ^ { ( l ) } \mathbf { v } _ { t - 1 , i } ^ { ( l ) } \right) \sigma _ { t - 1 , i } ^ { ( l ) } \mathbf { u } _ { t - 1 , i } ^ { ( l ) \top } } \end{array}$ , where $\sigma _ { t - 1 , i } ^ { ( l ) } , \mathbf { u } _ { t - 1 , i } ^ { ( l ) } .$ , and $\mathbf { v } _ { t - 1 , i } ^ { ( l ) }$ denote the i-th singular value, left singular vector, and right singular vector, respectively. The singular value controls the strength of the corresponding rank-one component, while the right and left singular vectors determine its input and output directions. Motivated by the stable capability retention of singular-value fine-tuning, we treat the paired input and output directions associated with each singular value as a single adaptation unit. We define the corresponding singular-vector channel as $\mathbf { \sigma } ^ { 1 } \mathbf { \mathcal { S } } _ { t - 1 , i } ^ { ( l ) } : = \left( \mathbf { u } _ { t - 1 , i } ^ { ( l ) } , \mathbf { v } _ { t - 1 , i } ^ { ( l ) } \right)$ . SVC then selectively enables updates to these channels while keeping all singular values trainable.

## 3.2 BENEFIT–COST ESTIMATION FOR SINGULAR-VECTOR CHANNELS

We assess each singular-vector channel by its adaptation benefit and forgetting cost. Since task $T _ { t }$ is learned by minimizing the current-task loss in Equation 1, the loss reduction resulting from a channel update provides a natural measure of its adaptation benefit. Forgetting cost, however, cannot be evaluated in the same way, because the original historical data are unavailable and a unified loss is difficult to construct for capturing changes in the model’s broad general capabilities. Inspired by methods such as GPM (Saha et al., 2021) that constrain updates based on activation subspaces, we therefore use linear-layer output drift on historical activations as a local proxy for forgetting cost. Although these activations are unavailable, their effect can be approximated using activation statistics collected from unlabeled general-domain data. Let $\Delta \mathbf { W } _ { t , i } ^ { ( l ) }$ denote the directional weight change induced by updating singular-vector channel $S _ { t - 1 , i } ^ { ( l ) } .$ The corresponding adaptation benefit $B _ { t , i } ^ { ( l ) }$ and forgetting cost $C _ { t , i } ^ { ( l ) }$ are conceptually defined as:

$$
B _ { t , i } ^ { ( l ) } = \mathcal { L } _ { t } \left( \mathbf { W } _ { t - 1 } ^ { ( l ) } \right) - \mathcal { L } _ { t } \left( \mathbf { W } _ { t - 1 } ^ { ( l ) } + \Delta \mathbf { W } _ { t , i } ^ { ( l ) } \right) ,\tag{2}
$$

$$
C _ { t , i } ^ { ( l ) } = \mathbb { E } _ { \boldsymbol { x } \sim \mathcal { D } _ { \mathrm { o l d } } } \left\| \Delta \mathbf { W } _ { t , i } ^ { ( l ) } x ^ { ( l ) } \right\| _ { 2 } ^ { 2 } = \mathrm { t r } \left( \Delta \mathbf { W } _ { t , i } ^ { ( l ) } \boldsymbol { \Sigma } _ { \mathrm { o l d } } ^ { ( l ) } \Delta \mathbf { W } _ { t , i } ^ { ( l ) \top } \right) ,\tag{3}
$$

where $\Sigma _ { \mathrm { o l d } } ^ { ( l ) } = \mathbb { E } _ { \boldsymbol { x } ^ { ( l ) } \sim \mathcal { D } _ { \mathrm { o l d } } } [ \boldsymbol { x } ^ { ( l ) } \boldsymbol { x } ^ { ( l ) \top } ] \in \mathbb { R } ^ { d _ { \mathrm { i n } } ^ { ( l ) } \times d _ { \mathrm { i n } } ^ { ( l ) } }$ denotes the second-moment matrix of historical input activations at layer l, serving here as an ideal reference statistic.

However, separately optimizing each channel to obtain $\Delta \mathbf { W } _ { t , i } ^ { ( l ) }$ would incur prohibitive computational overhead. We therefore use the average current-task gradient at layer l as the basis for estimating each channel’s initial directional update tendency:

$$
\mathbf { G } _ { t } ^ { ( l ) } = \frac { 1 } { n _ { t } } \sum _ { j = 1 } ^ { n _ { t } } \nabla _ { \mathbf { W } _ { t - 1 } ^ { ( l ) } } L _ { t } \left( f ( \mathbf { x } _ { t , j } ; \boldsymbol { \theta } _ { t - 1 } ) , \mathbf { y } _ { t , j } \right) ,\tag{4}
$$

After omitting positive constants shared by all channels within each objective, including the learning-rate terms, the resulting estimates take the following form:

$$
B _ { t , i } ^ { ( l ) } = \| \mathbf { p } _ { t , i } ^ { ( l ) } \| _ { 2 } ^ { 2 } + \| \mathbf { q } _ { t , i } ^ { ( l ) } \| _ { 2 } ^ { 2 } ,\tag{5}
$$

$$
C _ { t , i } ^ { ( l ) } = \left( \sigma _ { t - 1 , i } ^ { ( l ) } \right) ^ { 2 } \left[ \left( \mathbf { v } _ { t - 1 , i } ^ { ( l ) \top } \Sigma _ { \mathrm { o l d } } ^ { ( l ) } \mathbf { v } _ { t - 1 , i } ^ { ( l ) } \right) \| \mathbf { p } _ { t , i } ^ { ( l ) } \| _ { 2 } ^ { 2 } + \mathbf { q } _ { t , i } ^ { ( l ) \top } \Sigma _ { \mathrm { o l d } } ^ { ( l ) } \mathbf { q } _ { t , i } ^ { ( l ) } \right] ,\tag{6}
$$

where $\mathbf { p } _ { t , i } ^ { ( l ) }$ and $\mathbf { q } _ { t , i } ^ { ( l ) }$ denote the effective gradients for updating the left and right singular-vector directions, respectively:

$$
\mathbf { p } _ { t , i } ^ { ( l ) } = \left( \mathbf { I } - \mathbf { u } _ { t - 1 , i } ^ { ( l ) } \mathbf { u } _ { t - 1 , i } ^ { ( l ) \top } \right) \sigma _ { t - 1 , i } ^ { ( l ) } \mathbf { G } _ { t } ^ { ( l ) } \mathbf { v } _ { t - 1 , i } ^ { ( l ) } ,\tag{7}
$$

$$
\mathbf { q } _ { t , i } ^ { ( l ) } = \left( \mathbf { I } - \mathbf { v } _ { t - 1 , i } ^ { ( l ) } \mathbf { v } _ { t - 1 , i } ^ { ( l ) \top } \right) \sigma _ { t - 1 , i } ^ { ( l ) } \mathbf { G } _ { t } ^ { ( l ) \top } \mathbf { u } _ { t - 1 , i } ^ { ( l ) } .\tag{8}
$$

These projected gradients are used only for channel scoring before fine-tuning and do not constrain the subsequent optimization of the selected singular-vector directions.

Intuitively, the adaptation benefit measures the magnitude of the initial directional update tendency. The forgetting cost further weights this tendency by its interaction with the second-order statistics of historical activations.

Equations 3 and 6 show that an ideal evaluation of forgetting cost requires unavailable historical activations. However, historical data enter the resulting score only through the second-moment matrix $\Sigma _ { \mathrm { o l d } } ^ { ( l ) }$ , whose estimation requires no ground-truth labels. This allows us to approximate these statistics using a more accessible open-source, unlabeled general-domain corpus. In practice, we replace $\Sigma _ { \mathrm { o l d } } ^ { ( l ) }$ with the second-moment matrix computed from the shared general-domain reference corpus. Detailed derivations of the adaptation benefit and forgetting cost are provided in Appendix A.

## 3.3 PARETO-FRONTIER-BASED ADAPTIVE CHANNEL SELECTION

Given the scores estimated in Section 3.2, we seek channels with low forgetting cost and sufficient adaptation benefit. A weighted combination of the two objectives would require a trade-off coefficient that may vary across layers and tasks. We therefore use a three-stage procedure: knee-based cost screening excludes high-cost channels, Pareto-front filtering removes dominated candidates, and an adaptive efficiency cutoff removes channels with unfavorable benefit–cost trade-offs.

First, we log-transform the scores to reduce their scale variation: $b _ { t , i } ^ { ( l ) } = \log ( \operatorname* { m a x } \{ B _ { t , i } ^ { ( l ) } , \epsilon \} )$ ) and $c _ { t , i } ^ { ( l ) } = \log ( \operatorname* { m a x } \{ C _ { t , i } ^ { ( l ) } , \epsilon \} )$ , where ϵ ensures numerical stability. We sort the log-cost scores as $c _ { t , ( 1 ) } ^ { ( l ) } \leq \cdots \leq c _ { t , ( r _ { l } ) } ^ { ( l ) }$ and apply Kneedle to the resulting rank–cost curve, using a convex increasing configuration with sensitivity $S = 1 . 0$ (Satopaa et al., 2011). The detected knee $k _ { t } ^ { ( l ) }$ defines $c _ { t , * } ^ { ( \overline { { l } } ) } = c _ { t , ( k _ { t } ^ { ( l ) } ) } ^ { ( \overline { { l } } ) }$ . If no knee is detected, we set $c _ { t , * } ^ { ( l ) } = + \infty$ , retaining all channels at this stage. The resulting candidate set is

$$
\mathcal { F } _ { t } ^ { ( l ) } = \left\{ { \cal S } _ { t - 1 , i } ^ { ( l ) } \ \Big | \ c _ { t , i } ^ { ( l ) } \leq c _ { t , * } ^ { ( l ) } \right\} .\tag{9}
$$

Within $\mathcal { F } _ { t } ^ { ( l ) }$ , we retain the nondominated channels through Pareto-front filtering. A channel $S _ { t - 1 } ^ { ( l ) }$ ,i dominates $S _ { t - 1 , j } ^ { ( l ) }$ if it has no lower benefit and no higher cost, with at least one strict inequality. The Pareto frontier is defined as

$$
\begin{array} { r l } & { \mathcal { P } _ { t } ^ { ( l ) } = \left\{ S _ { t - 1 , j } ^ { ( l ) } \in \mathcal { F } _ { t } ^ { ( l ) } \left| \right. \exists S _ { t - 1 , i } ^ { ( l ) } \in \mathcal { F } _ { t } ^ { ( l ) } : S _ { t - 1 , i } ^ { ( l ) } \succ S _ { t - 1 , j } ^ { ( l ) } \right. \right\} , } \\ & { S _ { t - 1 , i } ^ { ( l ) } \succ S _ { t - 1 , j } ^ { ( l ) } \iff \left( b _ { t , i } ^ { ( l ) } \geq b _ { t , j } ^ { ( l ) } \right) \land \left( c _ { t , i } ^ { ( l ) } \leq c _ { t , j } ^ { ( l ) } \right) } \\ & { \qquad \left. \land \left[ \left( b _ { t , i } ^ { ( l ) } > b _ { t , j } ^ { ( l ) } \right) \lor \left( c _ { t , i } ^ { ( l ) } < c _ { t , j } ^ { ( l ) } \right) \right] . } \end{array}\tag{10}
$$

Pareto-front filtering removes dominated channels, but some remaining channels may still offer insufficient benefit relative to their costs. We therefore define the log-efficiency score $r _ { t , i } ^ { ( l ) } = b _ { t , i } ^ { ( l ) } -$ $c _ { t , i } ^ { ( l ) }$

Let ${ \mathcal { T } _ { P , t } ^ { ( l ) } = \left\{ i \in \left\{ 1 , \dots , r _ { l } \right\} \ : \left| \ : S _ { t - 1 , i } ^ { ( l ) } \in \mathcal { P } _ { t } ^ { ( l ) } \right. \right\} }$ , and $\pi _ { t } ^ { ( l ) }$ be a permutation of $\boldsymbol { \mathcal { T } } _ { P , t } ^ { ( l ) }$ that orders its channels by decreasing log-efficiency: $r _ { t , \pi _ { t } ^ { ( l ) } ( 1 ) } ^ { ( l ) } \geq \cdots \geq r _ { t , \pi _ { t } ^ { ( l ) } ( | \mathcal { T } _ { P , t } ^ { ( l ) } | ) } ^ { ( l ) }$ . We apply Otsu’s method to the efficiency scores of the Pareto-front channels (Otsu, 1979). If it yields a valid two-group partition, channels with scores no smaller than the threshold are selected. Otherwise, all channels in $\bar { \mathcal { P } } _ { t } ^ { ( l ) }$ are retained. The resulting cutoff determines the number of selected channels $K _ { t } ^ { ( l ) }$ . The final selected set is

$$
\mathcal { A } _ { t } ^ { ( l ) } = \left\{ { S } _ { t - 1 , \pi _ { t } ^ { ( l ) } ( k ) } ^ { ( l ) } \ \middle | \ 1 \leq k \leq K _ { t } ^ { ( l ) } \right\} .\tag{11}
$$

## 3.4 SVC

We illustrate the SVC fine-tuning procedure on task $T _ { t }$ . Before fine-tuning, SVC computes layerwise activation second moments from an unlabeled general-domain corpus and average gradients from the current-task data. SVC then applies SVD to each layer and scores its singular-vector channels using Equations 5 and 6. For each layer, SVC obtains $\mathcal { A } _ { t } ^ { ( l ) }$ through knee-based cost screening, Pareto-front filtering, and adaptive thresholding, as defined in Equations 9–11. This selection is completed before fine-tuning and remains fixed throughout training on $T _ { t }$

During fine-tuning, $\mathbf { W } _ { t - 1 } ^ { ( l ) }$ remains frozen, and only structured increments are optimized. Specifically, all singular values are assigned trainable increments, whereas singular-vector increments are registered only for channels in $\mathcal { A } _ { t } ^ { ( l ) }$ . Let $\delta \sigma _ { t , i } ^ { ( l ) } , \delta \mathbf { u } _ { t , i } ^ { ( l ) }$ , and $\delta \mathbf { v } _ { t , i } ^ { ( l ) }$ denote the corresponding increments, and let $N _ { \mathrm { l i n } }$ denote the number of linear layers adapted by SVC. The trainable parameter sets for layer l and task $T _ { t }$ are

$$
\Phi _ { t } ^ { ( l ) } = \left\{ \delta \sigma _ { t , i } ^ { ( l ) } \right\} _ { i = 1 } ^ { r _ { l } } \cup \left\{ \left( \delta \mathbf { u } _ { t , i } ^ { ( l ) } , \delta \mathbf { v } _ { t , i } ^ { ( l ) } \right) \bigg | S _ { t - 1 , i } ^ { ( l ) } \in A _ { t } ^ { ( l ) } \right\} , \qquad \Phi _ { t } = \bigcup _ { l = 1 } ^ { N _ { \operatorname { l i n } } } \Phi _ { t } ^ { ( l ) } .\tag{12}
$$

All incremental parameters are initialized to zero. During fine-tuning, the updated factors and the forward pass of layer l are given by

$$
\begin{array} { r l } & { \widetilde { \sigma } _ { t , i } ^ { ( l ) } = \sigma _ { t - 1 , i } ^ { ( l ) } + \delta \sigma _ { t , i } ^ { ( l ) } , } \\ & { \left( \widetilde { \mathbf { u } } _ { t , i } ^ { ( l ) } , \widetilde { \mathbf { v } } _ { t , i } ^ { ( l ) } \right) = \left\{ \left( \begin{array} { l l } { \mathbf { u } _ { t - 1 , i } ^ { ( l ) } + \delta \mathbf { u } _ { t , i } ^ { ( l ) } , \mathbf { v } _ { t - 1 , i } ^ { ( l ) } + \delta \mathbf { v } _ { t , i } ^ { ( l ) }  , } & { S _ { t - 1 , i } ^ { ( l ) } \in \mathcal { A } _ { t } ^ { ( l ) } , } \\ { \left( \mathbf { u } _ { t - 1 , i } ^ { ( l ) } , \mathbf { v } _ { t - 1 , i } ^ { ( l ) } \right) , } & { S _ { t - 1 , i } ^ { ( l ) } \notin \mathcal { A } _ { t } ^ { ( l ) } , } \end{array} \right. } \\ & \right){ \quad \quad \quad \mathbf { X } _ { t , \mathrm { o u t } } ^ { ( l ) } = \displaystyle \sum _ { i = 1 } ^ { r _ { l } } \left( \mathbf { X } _ { t , \mathrm { i n } } ^ { ( l ) } \widetilde { \mathbf { v } } _ { t , i } ^ { ( l ) } \right) \widetilde { \sigma } _ { t , i } ^ { ( l ) } \widetilde { \mathbf { u } } _ { t , i } ^ { ( l ) \top } . } \end{array}\tag{13}
$$

After fine-tuning, the learned factor changes are merged into the inherited weight: $\mathbf { W } _ { t } ^ { ( l ) } ~ =$ $\begin{array} { r } { \mathbf { W } _ { t - 1 } ^ { ( l ) } + \sum _ { i = 1 } ^ { r _ { l } } \left\lceil \widetilde { \sigma } _ { t , i } ^ { ( l ) , * } \widetilde { \mathbf { u } } _ { t , i } ^ { ( l ) , * } \widetilde { \mathbf { v } } _ { t , i } ^ { ( l ) , * \top } - \sigma _ { t - 1 , i } ^ { ( l ) } \mathbf { \widetilde { u } } _ { t - 1 , i } ^ { ( l ) } \mathbf { v } _ { t - 1 , i } ^ { ( l ) \top } \right\rceil } \end{array}$ , where the superscript ∗ denotes the corresponding values after fine-tuning. The merged weight $\mathbf { \bar { W } } _ { t } ^ { ( l ) }$ is then carried forward to task $T _ { t + 1 }$ Thus, SVC requires no persistent task-specific modules and introduces no task-wise model growth.

## 4 EXPERIMENTS

We evaluate SVC across model families and against continual learning baselines.

## 4.1 EXPERIMENTAL SETUP

Datasets and Metrics. For sequential learning, we use TRACE (Wang et al., 2023b), covering science (ScienceQA), finance (FOMC), summarization (MeetingBank), multilingual tasks (German 20Minuten and Chinese C-STANCE), code (Py150), and mathematics (NumGLUE-cm and NumGLUE-ds). Classification and single-token question-answering tasks are evaluated by accuracy, while other generation tasks use the average of ROUGE-L and BLEU. Let $R _ { t , i }$ denote performance on task i after learning task t. We report Overall Performance (OP) (Chaudhry et al., 2018), $\begin{array} { r } { O P _ { t } = \frac { 1 } { t } \sum _ { i = 1 } ^ { t } R _ { t , i } . } \end{array}$ , and Backward Transfer (BWT) (Lopez-Paz & Ranzato, 2017), $\begin{array} { r } { B W T _ { t } = \frac { 1 } { t - 1 } \sum _ { i = 1 } ^ { t - 1 } ( R _ { t , i } - R _ { i , i } ) } \end{array}$ , which measure average learned-task performance and subsequent changes on previous tasks, respectively. We measure capability preservation using Pass@1 on HumanEval (Chen et al., 2021) for code generation and GSM8K (Cobbe et al., 2021) for mathematical reasoning. To estimate $\mathbf { S V C } \mathbf { \ ' } _ { \mathbf { S } }$ activation second moments, we sample 30,000 unlabeled examples from the open-source C4 (Raffel et al., 2020) and Stack-v2-Python (Lozhkov et al., 2024) datasets at a 2:1 ratio. This corpus is fixed across all models and tasks. Details are provided in Appendix B.

Baselines. We compare SVC with four baseline categories. Regularization methods include EWC (Kirkpatrick et al., 2017), GEM (Lopez-Paz & Ranzato, 2017), and LwF (Li & Hoiem, 2017), which preserve prior knowledge through parameter-importance constraints, gradient projection, and knowledge distillation, respectively. Replay reuses past data after each task (Replay) (Rebuffi et al., 2017) or throughout training (Replay-online) (Rolnick et al., 2019). PEFT includes SeqLoRA, O LoRA (Wang et al., 2023a), PiSSA (Meng et al., 2024), and MiLoRA (Wang et al., 2025a). O-LoRA enforces orthogonality among task-specific LoRA subspaces. PiSSA updates adapter components initialized from the largest singular values, whereas MiLoRA uses the smallest-value components while freezing the dominant ones. PiSSA and MiLoRA thus make opposing fixed selections based on singular-value magnitude. In addition, we include OSFT (Nayak et al., 2026), an SVD-based constrained full-parameter fine-tuning method. OSFT uses previous-task data to identify important parameter subspaces and restricts new-task updates to their orthogonal complements. Sequential full-parameter fine-tuning (SeqFT) is also included. Details are provided in Appendix C.

Models and Implementation. We evaluate SVC on Llama3-8B (Grattafiori et al., 2024), Mistral-7B-v0.3 (Jiang et al., 2023), Gemma2-9B-it (Team et al., 2024), and Qwen3.5-9B-Base (Team, 2026), covering four representative open-source model families. All methods use Adam, a batch size of 64, no warmup, and five epochs per task. Learning rates are $1 0 ^ { - 4 }$ for SVC and all PEFT baselines and $1 0 ^ { - 5 }$ otherwise. Given the substantial impact of task order on continual learning (Yoon et al., 2020), drawing on the experimental setups of He et al. (2024) and Wang et al. (2025b), we evaluate all methods under three task orders. All experiments run on four 80-GB NVIDIA A100 GPUs. Further experimental details are provided in Appendix C.

## 4.2 MAIN RESULTS

Table 1 reports general-capability preservation and sequential task learning across four models and three task orders. We draw the following observations:

(1) SVC effectively preserves general capabilities. Across the 12 model and task-order settings, SVC achieves or ties the best result in 10 HumanEval settings and 11 GSM8K settings. Against the strongest baseline in each setting, SVC achieves mean gains of 1.68 percentage points on HumanEval and 5.18 percentage points on GSM8K. Together, these results show that SVC provides the strongest overall preservation of general capabilities across model families and task orders.

(2) SVC achieves a strong stability–plasticity balance. SVC obtains the best OP in nearly all settings across the four models and three task orders, while its BWT results also indicate substantially reduced forgetting. Despite storing no previous-task samples, SVC consistently surpasses both replay baselines in OP and remains competitive with them in BWT. Together, these results show that SVC acquires new-task knowledge while limiting interference with previously learned tasks.

(3) SVC is robust across task orders. Although task order affects final performance, SVC maintains strong general-capability preservation and sequential task learning across all three orders. It ranks first or second in 46 of the 48 combinations of model, task order, and metric. These results demonstrate that SVC remains robust to task order across different model families.

## 4.3 ANALYSIS AND DISCUSSION

To understand how SVC balances stability and plasticity, we examine three questions about its channel-level design on Llama3-8B under Order 1.

1. Are benefit and cost scores meaningful? Because SVC selects singular-vector channels to balance learning and forgetting, we test whether the scores defined in Section 3.2 reflect actual update outcomes. We first test whether C identifies risky update directions. For each task and linear layer, we match SVC’s channel count and select channels from the highest-, middle-, or lowest-cost region, with all other settings fixed. As shown in Table 2, the retention of pretrained capabilities improves consistently as the cost of the selected channels decreases. High-cost selection also affects sequential learning, suggesting that excessive cost can offset adaptation gains. These results show that C distinguishes risky channel updates and provides a reliable basis for controlling forgetting.

Table 1: Main results across four LLMs and three task orders. HumanEval and GSM8K measure general-capability preservation, while OP and BWT evaluate sequential task learning. Higher is better for all metrics. Best and second-best results are shown in bold and underlined, respectively.
<table><tr><td rowspan=1 colspan=13>Order 1                       Order 2                       Order 3MethodsHumanEval GSM8KOP  BWT HumanEval GSM8K OP  BWT HumanEval GSM8K OP  BWT</td></tr><tr><td rowspan=1 colspan=13>Llama3-8B</td></tr><tr><td rowspan=1 colspan=4>SeqFT        0.0  2.88 39.22</td><td rowspan=1 colspan=1>-10.69</td><td rowspan=1 colspan=3>0.61  4.32 29.78</td><td rowspan=1 colspan=5>-10.25  0.0  3.11 20.10-28.79</td></tr><tr><td rowspan=1 colspan=4>EWC         0.0  3.26 32.95</td><td rowspan=1 colspan=1>-10.05</td><td rowspan=1 colspan=2>0.0  5.46</td><td rowspan=1 colspan=1>34.33</td><td rowspan=1 colspan=4>-11.27 2.44  3.26 36.92</td><td rowspan=1 colspan=1>-6.74</td></tr><tr><td rowspan=1 colspan=4>GEM         0.0  3.34 39.37</td><td rowspan=1 colspan=3>-10.24  0.0  2.58</td><td rowspan=1 colspan=1>32.03</td><td rowspan=1 colspan=4>-12.44  0.0  2.81  39.07</td><td rowspan=1 colspan=1>-7.15</td></tr><tr><td rowspan=1 colspan=4>LwF          0.0  3.03 33.06</td><td rowspan=1 colspan=3>-9.94  1.83  2.73</td><td rowspan=1 colspan=1>27.20</td><td rowspan=1 colspan=4>-8.30  1.22  2.50 26.69</td><td rowspan=1 colspan=1>-13.53</td></tr><tr><td rowspan=1 colspan=3>Replay        0.0  2.05</td><td rowspan=1 colspan=1>38.62</td><td rowspan=1 colspan=1>-2.94</td><td rowspan=1 colspan=2>0.61  3.26</td><td rowspan=1 colspan=1>41.90</td><td rowspan=1 colspan=4>-0.24  0.61  3.34 39.56</td><td rowspan=1 colspan=1>-3.87</td></tr><tr><td rowspan=1 colspan=3>Replay-online  0.0  3.11</td><td rowspan=1 colspan=1>45.47</td><td rowspan=1 colspan=1>-1.99</td><td rowspan=1 colspan=2>0.0  3.03</td><td rowspan=1 colspan=1>41.33</td><td rowspan=1 colspan=4>-1.48   0.0  3.79 40.90</td><td rowspan=1 colspan=1>-1.64</td></tr><tr><td rowspan=1 colspan=4>SeqLoRA     28.05 44.2046.04</td><td rowspan=1 colspan=1>-4.39</td><td rowspan=1 colspan=2>27.44 21.83</td><td rowspan=1 colspan=1>44.36</td><td rowspan=1 colspan=4>-7.62 30.49 36.39 32.14</td><td rowspan=1 colspan=1>-2.80</td></tr><tr><td rowspan=1 colspan=1>O-LoRA</td><td rowspan=1 colspan=3>26.83 42.8439.73</td><td rowspan=1 colspan=1>-7.19</td><td rowspan=1 colspan=2>26.22 39.95</td><td rowspan=1 colspan=1>41.39</td><td rowspan=1 colspan=1>-4.07</td><td rowspan=1 colspan=3>29.27 37.91 43.43</td><td rowspan=1 colspan=1>-2.43</td></tr><tr><td rowspan=1 colspan=1>PiSSA</td><td rowspan=1 colspan=3>28.05 39.4247.81</td><td rowspan=1 colspan=1>-2.47</td><td rowspan=1 colspan=2>30.49 21.30</td><td rowspan=1 colspan=1>47.54</td><td rowspan=1 colspan=1>-2.33</td><td rowspan=1 colspan=3>28.05 16.4547.21</td><td rowspan=1 colspan=1>-2.92</td></tr><tr><td rowspan=1 colspan=1>MiLoRA</td><td rowspan=1 colspan=3>29.27 45.49 47.90</td><td rowspan=1 colspan=1>-1.55</td><td rowspan=1 colspan=2>28.66 41.24</td><td rowspan=1 colspan=1>47.66</td><td rowspan=1 colspan=1>-2.38</td><td rowspan=1 colspan=3>30.4946.17 44.15</td><td rowspan=1 colspan=1>-4.21</td></tr><tr><td rowspan=1 colspan=1>OSFT</td><td rowspan=1 colspan=2>0.0  1.97</td><td rowspan=1 colspan=1>28.00</td><td rowspan=1 colspan=1>-11.82</td><td rowspan=1 colspan=2>0.61  2.65</td><td rowspan=1 colspan=1>31.38</td><td rowspan=1 colspan=1>-7.47</td><td rowspan=1 colspan=3>0.0  2.96 27.37</td><td rowspan=1 colspan=1>-13.55</td></tr><tr><td rowspan=1 colspan=3>SVC (ours)   30.4951.55</td><td rowspan=1 colspan=1>49.62</td><td rowspan=1 colspan=1>-0.94</td><td rowspan=1 colspan=2>31.10 51.02</td><td rowspan=1 colspan=1>50.48</td><td rowspan=1 colspan=1>-0.05</td><td rowspan=1 colspan=3>29.88 47.31 49.21</td><td rowspan=1 colspan=1>-1.43</td></tr><tr><td rowspan=1 colspan=4>Mistral-7B-v0.3</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2>SeqFT        0.0</td><td rowspan=1 colspan=1>1.14</td><td rowspan=1 colspan=1>14.76</td><td rowspan=1 colspan=1>-22.20</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>1.97</td><td rowspan=1 colspan=1>23.79</td><td rowspan=1 colspan=1>-12.85</td><td rowspan=1 colspan=3>0.0  0.99 13.20</td><td rowspan=1 colspan=1>-23.57</td></tr><tr><td rowspan=1 colspan=2>EWC         0.0</td><td rowspan=1 colspan=1>1.06</td><td rowspan=1 colspan=1>17.65</td><td rowspan=1 colspan=1>-18.72</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>2.05</td><td rowspan=1 colspan=1>22.44</td><td rowspan=1 colspan=1>-13.20</td><td rowspan=1 colspan=3>0.0  1.90 15.11</td><td rowspan=1 colspan=1>-22.43</td></tr><tr><td rowspan=1 colspan=2>GEM         0.0</td><td rowspan=1 colspan=1>2.05</td><td rowspan=1 colspan=1>16.80</td><td rowspan=1 colspan=1>-21.40</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>2.27</td><td rowspan=1 colspan=1>26.85</td><td rowspan=1 colspan=1>-8.73</td><td rowspan=1 colspan=2>0.0  0.68</td><td rowspan=1 colspan=1>15.13</td><td rowspan=1 colspan=1>-21.82</td></tr><tr><td rowspan=1 colspan=2>LwF          0.0</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>7.13</td><td rowspan=1 colspan=1>-23.47</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>16.37</td><td rowspan=1 colspan=1>-9.83</td><td rowspan=1 colspan=2>0.0  0.76</td><td rowspan=1 colspan=1>12.40</td><td rowspan=1 colspan=1>-18.06</td></tr><tr><td rowspan=1 colspan=3>Replay        0.0  0.45</td><td rowspan=1 colspan=1>17.11</td><td rowspan=1 colspan=1>-5.10</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>1.97</td><td rowspan=1 colspan=1>30.14</td><td rowspan=1 colspan=1>-1.06</td><td rowspan=1 colspan=3>0.0  2.12 31.30</td><td rowspan=1 colspan=1>-2.37</td></tr><tr><td rowspan=1 colspan=2>Replay-online  0.0</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>17.36</td><td rowspan=1 colspan=1>-6.46</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>1.52</td><td rowspan=1 colspan=1>32.66</td><td rowspan=1 colspan=1>-1.28</td><td rowspan=1 colspan=2>0.0  1.52</td><td rowspan=1 colspan=1>33.17</td><td rowspan=1 colspan=1>-1.02</td></tr><tr><td rowspan=1 colspan=1>SeqLoRA</td><td rowspan=1 colspan=1>16.46</td><td rowspan=1 colspan=1>29.11</td><td rowspan=1 colspan=1>45.67</td><td rowspan=1 colspan=1>-3.91</td><td rowspan=1 colspan=1>15.24</td><td rowspan=1 colspan=1>20.85</td><td rowspan=1 colspan=1>41.63</td><td rowspan=1 colspan=1>-11.31</td><td rowspan=1 colspan=1>19.51</td><td rowspan=1 colspan=1>12.96</td><td rowspan=1 colspan=1>45.17</td><td rowspan=1 colspan=1>-5.48</td></tr><tr><td rowspan=1 colspan=1>O-LoRA</td><td rowspan=1 colspan=1>12.20</td><td rowspan=1 colspan=1>24.64</td><td rowspan=1 colspan=1>40.29</td><td rowspan=1 colspan=1>-5.16</td><td rowspan=1 colspan=1>17.07</td><td rowspan=1 colspan=1>23.35</td><td rowspan=1 colspan=1>35.92</td><td rowspan=1 colspan=1>-11.31</td><td rowspan=1 colspan=1>10.37</td><td rowspan=1 colspan=1>12.74</td><td rowspan=1 colspan=1>39.36</td><td rowspan=1 colspan=1>-5.38</td></tr><tr><td rowspan=1 colspan=1>PiSSA</td><td rowspan=1 colspan=1>21.34</td><td rowspan=1 colspan=1>11.90</td><td rowspan=1 colspan=1>38.98</td><td rowspan=1 colspan=1>-3.70</td><td rowspan=1 colspan=1>16.46</td><td rowspan=1 colspan=1>19.86</td><td rowspan=1 colspan=1>44.67</td><td rowspan=1 colspan=1>-3.50</td><td rowspan=1 colspan=1>11.59</td><td rowspan=1 colspan=1>3.79</td><td rowspan=1 colspan=1>44.26</td><td rowspan=1 colspan=1>-1.73</td></tr><tr><td rowspan=1 colspan=1>MiLoRA</td><td rowspan=1 colspan=1>25.61</td><td rowspan=1 colspan=1>30.78</td><td rowspan=1 colspan=1>46.44</td><td rowspan=1 colspan=1>-2.56</td><td rowspan=1 colspan=1>24.39</td><td rowspan=1 colspan=1>26.76</td><td rowspan=1 colspan=1>42.47</td><td rowspan=1 colspan=1>-5.05</td><td rowspan=1 colspan=1>25.61</td><td rowspan=1 colspan=1>28.05</td><td rowspan=1 colspan=1>43.45</td><td rowspan=1 colspan=1>-2.75</td></tr><tr><td rowspan=1 colspan=1>OSFT</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>2.43</td><td rowspan=1 colspan=1>13.36</td><td rowspan=1 colspan=1>-18.42</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>2.20</td><td rowspan=1 colspan=1>-19.86</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>2.54</td><td rowspan=1 colspan=1>-16.52</td></tr><tr><td rowspan=1 colspan=1>SVC (ours)</td><td rowspan=1 colspan=2>27.44 36.54</td><td rowspan=1 colspan=1>48.37</td><td rowspan=1 colspan=1>-0.37</td><td rowspan=1 colspan=1>29.88</td><td rowspan=1 colspan=1>37.07</td><td rowspan=1 colspan=1>48.70</td><td rowspan=1 colspan=1>-1.75</td><td rowspan=1 colspan=2>25.61 37.68</td><td rowspan=1 colspan=1>44.78</td><td rowspan=1 colspan=1>-1.53</td></tr><tr><td rowspan=1 colspan=1>Gemma2-9B-it</td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>SeqFT</td><td rowspan=1 colspan=2>9.76  5.38</td><td rowspan=1 colspan=1>34.09</td><td rowspan=1 colspan=1>-12.37</td><td rowspan=1 colspan=1>7.32</td><td rowspan=1 colspan=1>5.53</td><td rowspan=1 colspan=1>34.77</td><td rowspan=1 colspan=1>-7.30</td><td rowspan=1 colspan=2>7.32  1.74</td><td rowspan=1 colspan=1>33.97</td><td rowspan=1 colspan=1>-11.17</td></tr><tr><td rowspan=1 colspan=1>EWC</td><td rowspan=1 colspan=1>7.32</td><td rowspan=1 colspan=1>2.73</td><td rowspan=1 colspan=1>38.58</td><td rowspan=1 colspan=1>-6.96</td><td rowspan=1 colspan=1>5.49</td><td rowspan=1 colspan=1>6.75</td><td rowspan=1 colspan=1>36.85</td><td rowspan=1 colspan=1>-8.90</td><td rowspan=1 colspan=2>9.15  2.12</td><td rowspan=1 colspan=1>38.72</td><td rowspan=1 colspan=1>-4.53</td></tr><tr><td rowspan=1 colspan=1>GEM</td><td rowspan=1 colspan=1>10.37</td><td rowspan=1 colspan=1>3.11</td><td rowspan=1 colspan=1>32.55</td><td rowspan=1 colspan=1>-8.98</td><td rowspan=1 colspan=1>10.98</td><td rowspan=1 colspan=1>6.67</td><td rowspan=1 colspan=1>37.22</td><td rowspan=1 colspan=1>-5.70</td><td rowspan=1 colspan=2>5.49  1.97</td><td rowspan=1 colspan=1>33.89</td><td rowspan=1 colspan=1>-13.20</td></tr><tr><td rowspan=1 colspan=1>LwF</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>2.50</td><td rowspan=1 colspan=1>26.51</td><td rowspan=1 colspan=1>-12.52</td><td rowspan=1 colspan=1>1.22</td><td rowspan=1 colspan=1>2.58</td><td rowspan=1 colspan=1>23.76</td><td rowspan=1 colspan=1>-10.12</td><td rowspan=1 colspan=3>1.22  3.87 28.18</td><td rowspan=1 colspan=1>-9.94</td></tr><tr><td rowspan=1 colspan=1>Replay</td><td rowspan=1 colspan=1>6.71</td><td rowspan=1 colspan=1>3.26</td><td rowspan=1 colspan=1>42.30</td><td rowspan=1 colspan=1>-0.66</td><td rowspan=1 colspan=1>0.61</td><td rowspan=1 colspan=1>3.49</td><td rowspan=1 colspan=1>39.44</td><td rowspan=1 colspan=1>0.51</td><td rowspan=1 colspan=3>3.05  2.35 41.78</td><td rowspan=1 colspan=1>-1.32</td></tr><tr><td rowspan=1 colspan=1>Replay-online</td><td rowspan=1 colspan=1>3.66</td><td rowspan=1 colspan=1>2.58</td><td rowspan=1 colspan=1>44.12</td><td rowspan=1 colspan=1>-1.95</td><td rowspan=1 colspan=1>1.22</td><td rowspan=1 colspan=1>2.73</td><td rowspan=1 colspan=1>38.61</td><td rowspan=1 colspan=1>-3.30</td><td rowspan=1 colspan=1>3.05</td><td rowspan=1 colspan=1>1.97</td><td rowspan=1 colspan=1>40.25</td><td rowspan=1 colspan=1>-3.32</td></tr><tr><td rowspan=1 colspan=1>SeqLoRA</td><td rowspan=1 colspan=1>37.20</td><td rowspan=1 colspan=1>73.09</td><td rowspan=1 colspan=1>50.19</td><td rowspan=1 colspan=1>-6.83</td><td rowspan=1 colspan=1>32.32</td><td rowspan=1 colspan=1>56.86</td><td rowspan=1 colspan=1>51.58</td><td rowspan=1 colspan=1>-4.63</td><td rowspan=1 colspan=1>28.66</td><td rowspan=1 colspan=1>3.94</td><td rowspan=1 colspan=1>51.97</td><td rowspan=1 colspan=1>-4.56</td></tr><tr><td rowspan=1 colspan=1>O-LoRA</td><td rowspan=1 colspan=1>41.46</td><td rowspan=1 colspan=1>63.08</td><td rowspan=1 colspan=1>47.91</td><td rowspan=1 colspan=1>-5.88</td><td rowspan=1 colspan=1>37.80</td><td rowspan=1 colspan=1>57.01</td><td rowspan=1 colspan=1>45.43</td><td rowspan=1 colspan=1>-2.60</td><td rowspan=1 colspan=1>15.85</td><td rowspan=1 colspan=1>46.40</td><td rowspan=1 colspan=1>46.00</td><td rowspan=1 colspan=1>-7.03</td></tr><tr><td rowspan=1 colspan=1>PiSSA</td><td rowspan=1 colspan=1>49.39</td><td rowspan=1 colspan=1>67.63</td><td rowspan=1 colspan=1>51.93</td><td rowspan=1 colspan=1>-3.60</td><td rowspan=1 colspan=1>48.78</td><td rowspan=1 colspan=1>66.19</td><td rowspan=1 colspan=1>53.90</td><td rowspan=1 colspan=1>-0.92</td><td rowspan=1 colspan=1>48.78</td><td rowspan=1 colspan=1>55.04</td><td rowspan=1 colspan=1>52.45</td><td rowspan=1 colspan=1>-2.95</td></tr><tr><td rowspan=1 colspan=1>MiLoRA</td><td rowspan=1 colspan=1>50.61</td><td rowspan=1 colspan=1>73.54</td><td rowspan=1 colspan=1>49.91</td><td rowspan=1 colspan=1>-4.21</td><td rowspan=1 colspan=1>47.56</td><td rowspan=1 colspan=1>73.69</td><td rowspan=1 colspan=1>53.20</td><td rowspan=1 colspan=1>-1.54</td><td rowspan=1 colspan=1>44.51</td><td rowspan=1 colspan=1>5.53</td><td rowspan=1 colspan=1>54.31</td><td rowspan=1 colspan=1>-0.91</td></tr><tr><td rowspan=1 colspan=1>OSFT</td><td rowspan=1 colspan=1>7.32</td><td rowspan=1 colspan=1>3.41</td><td rowspan=1 colspan=1>30.35</td><td rowspan=1 colspan=1>-4.29</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>0.99</td><td rowspan=1 colspan=1>19.14</td><td rowspan=1 colspan=1>-12.29</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>0.76</td><td rowspan=1 colspan=1>11.14</td><td rowspan=1 colspan=1>-23.95</td></tr><tr><td rowspan=1 colspan=1>SVC (ours)</td><td rowspan=1 colspan=1>56.10</td><td rowspan=1 colspan=1>75.51</td><td rowspan=1 colspan=1>50.25</td><td rowspan=1 colspan=1>-1.37</td><td rowspan=1 colspan=1>53.66</td><td rowspan=1 colspan=1>74.22</td><td rowspan=1 colspan=1>54.95</td><td rowspan=1 colspan=1>-0.72</td><td rowspan=1 colspan=1>53.66</td><td rowspan=1 colspan=1>74.91</td><td rowspan=1 colspan=1>55.84</td><td rowspan=1 colspan=1>-0.11</td></tr><tr><td rowspan=1 colspan=1>Qwen3.5-9B-Bas</td><td rowspan=1 colspan=3>e</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>SeqFT</td><td rowspan=1 colspan=1>20.73</td><td rowspan=1 colspan=1>11.07</td><td rowspan=1 colspan=1>50.23</td><td rowspan=1 colspan=1>-3.66</td><td rowspan=1 colspan=1>23.78</td><td rowspan=1 colspan=1>49.58</td><td rowspan=1 colspan=1>46.18</td><td rowspan=1 colspan=1>-7.47</td><td rowspan=1 colspan=1>17.68</td><td rowspan=1 colspan=1>4.25</td><td rowspan=1 colspan=1>48.51</td><td rowspan=1 colspan=1>-5.24</td></tr><tr><td rowspan=1 colspan=1>EWC</td><td rowspan=1 colspan=1>21.95</td><td rowspan=1 colspan=1>11.75</td><td rowspan=1 colspan=1>47.88</td><td rowspan=1 colspan=1>-5.57</td><td rowspan=1 colspan=1>20.12</td><td rowspan=1 colspan=1>48.07</td><td rowspan=1 colspan=1>47.38</td><td rowspan=1 colspan=1>-3.60</td><td rowspan=1 colspan=1>16.46</td><td rowspan=1 colspan=1>4.70</td><td rowspan=1 colspan=1>48.01</td><td rowspan=1 colspan=1>-3.59</td></tr><tr><td rowspan=1 colspan=1>GEM</td><td rowspan=1 colspan=1>18.29</td><td rowspan=1 colspan=1>14.40</td><td rowspan=1 colspan=1>49.64</td><td rowspan=1 colspan=1>-3.38</td><td rowspan=1 colspan=1>16.46</td><td rowspan=1 colspan=1>41.62</td><td rowspan=1 colspan=1>45.15</td><td rowspan=1 colspan=1>-7.32</td><td rowspan=1 colspan=1>18.29</td><td rowspan=1 colspan=1>4.25</td><td rowspan=1 colspan=1>47.26</td><td rowspan=1 colspan=1>-3.65</td></tr><tr><td rowspan=1 colspan=1>LwF</td><td rowspan=1 colspan=1>16.46</td><td rowspan=1 colspan=1>28.73</td><td rowspan=1 colspan=1>38.23</td><td rowspan=1 colspan=1>-8.44</td><td rowspan=1 colspan=1>21.34</td><td rowspan=1 colspan=1>23.73</td><td rowspan=1 colspan=1>31.94</td><td rowspan=1 colspan=1>-9.66</td><td rowspan=1 colspan=1>22.56</td><td rowspan=1 colspan=1>30.17</td><td rowspan=1 colspan=1>38.27</td><td rowspan=1 colspan=1>-9.17</td></tr><tr><td rowspan=1 colspan=1>Replay</td><td rowspan=1 colspan=1>29.88</td><td rowspan=1 colspan=1>16.07</td><td rowspan=1 colspan=1>49.36</td><td rowspan=1 colspan=1>-0.99</td><td rowspan=1 colspan=1>17.07</td><td rowspan=1 colspan=1>12.89</td><td rowspan=1 colspan=1>49.97</td><td rowspan=1 colspan=1>-0.94</td><td rowspan=1 colspan=1>12.20</td><td rowspan=1 colspan=1>5.23</td><td rowspan=1 colspan=1>46.70</td><td rowspan=1 colspan=1>-1.04</td></tr><tr><td rowspan=1 colspan=1>Replay-online</td><td rowspan=1 colspan=1>24.39</td><td rowspan=1 colspan=1>12.13</td><td rowspan=1 colspan=1>51.16</td><td rowspan=1 colspan=1>-0.78</td><td rowspan=1 colspan=1>28.66</td><td rowspan=1 colspan=1>36.39</td><td rowspan=1 colspan=1>51.91</td><td rowspan=1 colspan=1>-0.16</td><td rowspan=1 colspan=1>12.20</td><td rowspan=1 colspan=1>12.05</td><td rowspan=1 colspan=1>47.25</td><td rowspan=1 colspan=1>-1.79</td></tr><tr><td rowspan=1 colspan=2>SeqLoRA     50.61</td><td rowspan=1 colspan=1>71.19</td><td rowspan=1 colspan=1>49.86</td><td rowspan=1 colspan=1>-4.58</td><td rowspan=1 colspan=1>50.61</td><td rowspan=1 colspan=1>66.94</td><td rowspan=1 colspan=1>43.09</td><td rowspan=1 colspan=1>-7.46</td><td rowspan=1 colspan=1>49.39</td><td rowspan=1 colspan=2>74.6050.39</td><td rowspan=1 colspan=1>-4.64</td></tr><tr><td rowspan=1 colspan=2>O-LoRA     28.05</td><td rowspan=1 colspan=1>42.99</td><td rowspan=1 colspan=1>39.11</td><td rowspan=1 colspan=1>-3.80</td><td rowspan=1 colspan=1>34.76</td><td rowspan=1 colspan=1>39.12</td><td rowspan=1 colspan=1>42.01</td><td rowspan=1 colspan=1>-6.88</td><td rowspan=1 colspan=1>23.17</td><td rowspan=1 colspan=1>27.82</td><td rowspan=1 colspan=1>45.05</td><td rowspan=1 colspan=1>-6.74</td></tr><tr><td rowspan=1 colspan=1>PiSSA</td><td rowspan=1 colspan=1>54.27</td><td rowspan=1 colspan=1>71.87</td><td rowspan=1 colspan=1>51.46</td><td rowspan=1 colspan=1>-1.31</td><td rowspan=1 colspan=1>48.17</td><td rowspan=1 colspan=1>67.17</td><td rowspan=1 colspan=1>48.09</td><td rowspan=1 colspan=1>-7.19</td><td rowspan=1 colspan=1>51.22</td><td rowspan=1 colspan=1>68.92</td><td rowspan=1 colspan=1>54.56</td><td rowspan=1 colspan=1>0.13</td></tr><tr><td rowspan=1 colspan=1>MiLoRA</td><td rowspan=1 colspan=1>42.07</td><td rowspan=1 colspan=1>68.92</td><td rowspan=1 colspan=1>54.34</td><td rowspan=1 colspan=1>0.59</td><td rowspan=1 colspan=1>49.39</td><td rowspan=1 colspan=1>70.13</td><td rowspan=1 colspan=1>44.71</td><td rowspan=1 colspan=1>-10.09</td><td rowspan=1 colspan=1>47.56</td><td rowspan=1 colspan=1>69.22</td><td rowspan=1 colspan=1>54.28</td><td rowspan=1 colspan=1>2.36</td></tr><tr><td rowspan=1 colspan=1>OSFT</td><td rowspan=1 colspan=1>11.59</td><td rowspan=1 colspan=1>5.31</td><td rowspan=1 colspan=1>47.99</td><td rowspan=1 colspan=1>-2.04</td><td rowspan=1 colspan=1>17.68</td><td rowspan=1 colspan=1>48.14</td><td rowspan=1 colspan=1>49.04</td><td rowspan=1 colspan=1>-1.78</td><td rowspan=1 colspan=1>1.22</td><td rowspan=1 colspan=1>4.55</td><td rowspan=1 colspan=1>45.83</td><td rowspan=1 colspan=1>-5.97</td></tr><tr><td rowspan=1 colspan=1>SVC (ours)</td><td rowspan=1 colspan=3>43.2972.25 57.05</td><td rowspan=1 colspan=1>0.02</td><td rowspan=1 colspan=2>56.1071.27</td><td rowspan=1 colspan=1>56.65</td><td rowspan=1 colspan=1>-0.53</td><td rowspan=1 colspan=4>53.05 70.2057.87 0.76</td></tr></table>

Table 2: Validation of channel-level benefit and cost scores on Llama3-8B under Order 1.
<table><tr><td>Metric</td><td>High C</td><td>Mid C</td><td>Low C</td><td>Low B, Mid C</td><td>High B, Mid C</td><td>Low B, Low C</td><td>High B, Low C</td></tr><tr><td>HumanEval</td><td>0.00</td><td>28.05</td><td>29.27</td><td>27.44</td><td>28.66</td><td>29.88</td><td>29.88</td></tr><tr><td>GSM8K</td><td>2.65</td><td>44.05</td><td>51.40</td><td>45.49</td><td>44.35</td><td>52.08</td><td>51.71</td></tr><tr><td>OP</td><td>29.27</td><td>47.99</td><td>51.12</td><td>47.26</td><td>47.63</td><td>50.66</td><td>50.28</td></tr><tr><td>BWT</td><td>-8.32</td><td>-1.06</td><td>-0.20</td><td>-2.21</td><td>-2.51</td><td>0.05</td><td>-1.19</td></tr><tr><td>AP</td><td>36.55</td><td>48.92</td><td>51.30</td><td>49.19</td><td>49.83</td><td>50.62</td><td>51.32</td></tr></table>

To reduce the influence of C when assessing B, we compare channels under cost-matched conditions. For each task and layer, we rank channels by C and separately take 2K candidates from the lowest- and middle-cost regions, where K is SVC’s channel count. We pair cost-adjacent channels and select the higher- or lower-B channel from each pair. Within each region, this produces two disjoint K-channel sets with closely matched costs. Because B measures current-task adaptation, we report $\begin{array} { r } { A P _ { t } = \frac { 1 } { t } \sum _ { i = 1 } ^ { t } R _ { i , i } } \end{array}$ , which averages performance immediately after each task is learned. As shown in Table 2, the high-B groups achieve higher AP than their low-B counterparts in both cost regions. Although strict cost matching limits the magnitude of the differences between the two groups, the trend remains consistent across both regions. This consistent trend indicates that B captures the contribution of a channel to current-task adaptation. Together, the results for B and C support their use in balancing learning and forgetting at the level of singular-vector channels.

2. How important is each selection stage? Section 3.3 combines knee-based cost screening, Paretofront filtering, and Otsu thresholding of benefit–cost efficiency scores. We remove each stage in turn, keep all other settings fixed, and let the remaining stages determine the channel count. Figure 3 shows that removing Pareto-front filtering causes the largest degradation, nearly eliminating the retention of codegeneration and mathematical-reasoning capabilities and sharply reducing sequential performance. Re-

![](images/a33c412824c33df7fd274a7b13cc5c1a7aeecf3f4c364858dd771f8e87a830b0.jpg)  
Figure 3: Ablation of the channel-selection stages. BWT is reported in absolute value.

moving knee-based cost screening has a smaller effect but still weakens mathematical-reasoning retention, lowers sequential performance, and increases forgetting. These results show that reliable channel selection depends on the complete three-stage procedure. Pareto filtering forms the core, while knee-based screening and the Otsu-based cutoff further restrict high-cost and low-efficiency channels to maintain the overall stability-plasticity balance.

3. Should singular values be selected? By default, SVC selectively updates singular vectors while training all singular values. To test whether singular values should also be selected, we redefine each channel as $( \sigma _ { i } , u _ { i } , v _ { i } )$ , recompute its benefit and cost from the joint perturbation, and apply the same three-stage selection procedure. See Appendix A for the derivation. Compared with default SVC, joint selection decreases HumanEval, GSM8K, OP, and BWT by 1.22, 2.80, 0.65, and 0.56 points, respectively. One possible explanation is that including the singular-value term changes the score distributions, which may reduce the separation between channels and alter both the selected set and its size. Moreover, singular values continue to adjust throughout training, so a local estimate computed before training may not accurately reflect their final contribution. These results support the default SVC design, which selectively updates singular vectors while optimizing all singular values to better balance learning and forgetting.

## 5 CONCLUSION

We present Selective Singular-Vector Channel Adaptation (SVC), which treats singular-vector channels as the basic units for balancing adaptation benefits against forgetting costs in continual learning for large language models. SVC estimates both quantities using current-task gradients and activation statistics from an unlabeled general-domain reference corpus, and adaptively selects trainable channels accordingly. Experiments across four LLM families and three task orders show that SVC improves continual-learning performance while preserving pretrained general capabilities, offering a task-adaptive route to stable and scalable continual domain adaptation for LLMs. In future work, we will investigate reference-corpus construction methods that provide broader capability coverage and greater robustness to distribution shifts, as well as channel-selection strategies that account for interactions among channels and adjust dynamically throughout training. We will further evaluate SVC over longer task sequences and a broader range of general-capability benchmarks to characterize the limits of its robustness and computational efficiency.

## REFERENCES

Dan Biderman, Jacob Portes, Jose Javier Gonzalez Ortiz, Mansheej Paul, Philip Greengard, Connor Jennings, Daniel King, Sam Havens, Vitaliy Chiley, Jonathan Frankle, Cody Blakeney, and John Patrick Cunningham. LoRA learns less and forgets less. Transactions on Machine Learning Research, 2024. ISSN 2835-8856. URL https://openreview.net/forum?id= aloEru2qCG. Featured Certification.

Arslan Chaudhry, Puneet K Dokania, Thalaiyasingam Ajanthan, and Philip HS Torr. Riemannian walk for incremental learning: Understanding forgetting and intransigence. In European conference on computer vision, pp. 556–572. Springer, 2018.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde De Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021.

Yanyuan Chen, Dexuan Xu, Yu Huang, Songkun Zhan, Hanpin Wang, Dongxue Chen, Xueping Wang, Meikang Qiu, and Hang Li. Mimo: A medical vision language model with visual referring multimodal input and pixel grounding multimodal output. In Proceedings ofthe Computer Vision and Pattern Recognition Conference, pp. 24732–24741, 2025.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

Ning Ding, Yujia Qin, Guang Yang, Fuchao Wei, Zonghan Yang, Yusheng Su, Shengding Hu, Yulin Chen, Chi-Min Chan, Weize Chen, et al. Parameter-efficient fine-tuning of large-scale pre-trained language models. Nature machine intelligence, 5(3):220–235, 2023.

Robert M French. Catastrophic forgetting in connectionist networks. Trends in cognitive sciences, 3(4):128–135, 1999.

Annette Rios Gonzales, Nicolas Spring, Tannon Kew, Marek Kostrzewa, Andreas Sauberli, Mathias¨ Muller, and Sarah Ebling. A new dataset and efficient baselines for document-level text simplifi-¨ cation in german. In Proceedings of the Third Workshop on New Frontiers in Summarization, pp. 152–161, 2021.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

Ligong Han, Yinxiao Li, Han Zhang, Peyman Milanfar, Dimitris Metaxas, and Feng Yang. Svdiff: Compact parameter space for diffusion fine-tuning. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 7289–7300. IEEE, 2023.

Jinghan He, Haiyun Guo, Kuan Zhu, Zihan Zhao, Ming Tang, and Jinqiao Wang. Seekr: Selective attention-guided knowledge retention for continual learning of large language models. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 3254–3266, 2024.

Edward J Hu, yelong shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum? id=nZeVKeeFYf9.

Yebowen Hu, Timothy Ganter, Hanieh Deilamsalehy, Franck Dernoncourt, Hassan Foroosh, and Fei Liu. Meetingbank: A benchmark dataset for meeting summarization. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 16409–16423, 2023.

Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lelio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril,´ Thomas Wang, Timothee Lacroix, and William El Sayed. Mistral 7b, 2023. URL´ https: //arxiv.org/abs/2310.06825.

Nikitas Karanikolas, Eirini Manga, Nikoletta Samaridi, Eleni Tousidou, and Michael Vassilakopoulos. Large language models versus natural language understanding and generation. In Proceedings of the 27th Pan-Hellenic Conference on Progress in Computing and Informatics, pp. 278–290, 2023.

James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, et al. Overcoming catastrophic forgetting in neural networks. Proceedings of the national academy of sciences, 114(13):3521–3526, 2017.

Xinlong Li, Weijieying Ren, Wei Qin, Lei Wang, Tianxiang Zhao, and Richang Hong. Analyzing and reducing catastrophic forgetting in parameter efficient tuning. In ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 1–5, 2025. doi: 10.1109/ICASSP49660.2025.10889361.

Zhizhong Li and Derek Hoiem. Learning without forgetting. IEEE transactions on pattern analysis and machine intelligence, 40(12):2935–2947, 2017.

Vijay Lingam, Atula Tejaswi, Aditya Vavre, Aneesh Shetty, Gautham K Gudur, Joydeep Ghosh, Alex Dimakis, Eunsol Choi, Aleksandar Bojchevski, and Sujay Sanghavi. Svft: Parameterefficient fine-tuning with singular vectors. Advances in neural information processing systems, 37:41425–41446, 2024.

Hong Liu, Mingsheng Long, Jianmin Wang, and Yu Wang. Learning to adapt to evolving domains. Advances in neural information processing systems, 33:22338–22348, 2020.

David Lopez-Paz and Marc’Aurelio Ranzato. Gradient episodic memory for continual learning. Advances in neural information processing systems, 30, 2017.

Anton Lozhkov, Raymond Li, Loubna Ben Allal, Federico Cassano, Joel Lamy-Poirier, Nouamane Tazi, Ao Tang, Dmytro Pykhtar, Jiawei Liu, Yuxiang Wei, et al. Starcoder 2 and the stack v2: The next generation. arXiv preprint arXiv:2402.19173, 2024.

Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. Advances in Neural Information Processing Systems, 35:2507–2521, 2022.

Shuai Lu, Daya Guo, Shuo Ren, Junjie Huang, Alexey Svyatkovskiy, Ambrosio Blanco, Colin Clement, Dawn Drain, Daxin Jiang, Duyu Tang, et al. Codexglue: A machine learning benchmark dataset for code understanding and generation. arXiv preprint arXiv:2102.04664, 2021.

Fanxu Meng, Zhaohui Wang, and Muhan Zhang. Pissa: Principal singular values and singular vectors adaptation of large language models. Advances in Neural Information Processing Systems, 37:121038–121072, 2024.

Swaroop Mishra, Arindam Mitra, Neeraj Varshney, Bhavdeep Sachdeva, Peter Clark, Chitta Baral, and Ashwin Kalyan. Numglue: A suite of fundamental yet challenging mathematical reasoning tasks. In Proceedings ofthe 60th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pp. 3505–3523. Association for Computational Linguistics, 2022.

Neil Muller, Lourenc¸o Magaia, and Ben M Herbst. Singular value decomposition, eigenfaces, and 3d reconstructions. SIAM review, 46(3):518–545, 2004.

Nikhil Shivakumar Nayak, Krishnateja Killamsetty, Ligong Han, Abhishek Bhandwaldar, Prateek Chanda, Kai Xu, Oleg Silkin, Mustafa Eyceoz, Hao Wang, Aldo Pareja, and Akash Srivastava. Sculpting subspaces: Constrained full fine-tuning in LLMs for continual learning. In The Fourteenth International Conference on Learning Representations, 2026. URL https: //openreview.net/forum?id=vQcyqsGJDw.

Nobuyuki Otsu. A threshold selection method from gray-level histograms. IEEE transactions on systems, man, and cybernetics, 9(1):62–66, 1979.

Zelin Peng, Zhengqin Xu, Zhilin Zeng, Xiaokang Yang, and Wei Shen. Sam-parser: Fine-tuning sam efficiently by parameter space reconstruction. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pp. 4515–4523, 2024.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal ofmachine learning research, 21(140):1–67, 2020.

Sylvestre-Alvise Rebuffi, Alexander Kolesnikov, Georg Sperl, and Christoph H Lampert. icarl: Incremental classifier and representation learning. In Proceedings of the IEEE conference on Computer Vision and Pattern Recognition, pp. 2001–2010, 2017.

David Rolnick, Arun Ahuja, Jonathan Schwarz, Timothy Lillicrap, and Gregory Wayne. Experience replay for continual learning. Advances in neural information processing systems, 32, 2019.

Gobinda Saha, Isha Garg, and Kaushik Roy. Gradient projection memory for continual learning. In International Conference on Learning Representations, 2021. URL https://openreview. net/forum?id=3AOj0RCNC2.

Ville Satopaa, Jeannie Albrecht, David Irwin, and Barath Raghavan. Finding a” kneedle” in a haystack: Detecting knee points in system behavior. In 2011 31st international conference on distributed computing systems workshops, pp. 166–171. IEEE, 2011.

Agam Shah, Suvan Paturi, and Sudheer Chava. Trillion dollar words: A new financial dataset, task & market analysis. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 6664–6679, 2023.

Haizhou Shi, Zihao Xu, Hengyi Wang, Weiyi Qin, Wenyuan Wang, Yibin Wang, Zifeng Wang, Sayna Ebrahimi, and Hao Wang. Continual learning of large language models: A comprehensive survey. ACM Computing Surveys, 58(5):1–42, 2025.

Yanpeng Sun, Qiang Chen, Xiangyu He, Jian Wang, Haocheng Feng, Junyu Han, Errui Ding, Jian Cheng, Zechao Li, and Jingdong Wang. Singular value fine-tuning: Few-shot segmentation requires few-parameters fine-tuning. Advances in neural information processing systems, 35: 37484–37496, 2022.

Gemma Team, Morgane Riviere, Shreya Pathak, Pier Giuseppe Sessa, Cassidy Hardin, Surya Bhupatiraju, Leonard Hussenot, Thomas Mesnard, Bobak Shahriari, Alexandre Ram´ e, et al. Gemma´ 2: Improving open language models at a practical size. arXiv preprint arXiv:2408.00118, 2024.

Qwen Team. Qwen3.5: Towards native multimodal agents, February 2026. URL https://qwen. ai/blog?id=qwen3.5.

Tu Vu, Aditya Barua, Brian Lester, Daniel Cer, Mohit Iyyer, and Noah Constant. Overcoming catastrophic forgetting in zero-shot cross-lingual generation. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pp. 9279–9300, 2022.

Hanqing Wang, Yixia Li, Shuo Wang, Guanhua Chen, and Yun Chen. Milora: Harnessing minor singular components for parameter-efficient llm finetuning. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pp. 4823–4836, 2025a.

Lingxiang Wang, Hainan Zhang, and Zhiming Zheng. Parameter importance-driven continual learning for foundation models. arXiv preprint arXiv:2511.15375, 2025b.

Xiao Wang, Tianze Chen, Qiming Ge, Han Xia, Rong Bao, Rui Zheng, Qi Zhang, Tao Gui, and Xuan-Jing Huang. Orthogonal subspace learning for language model continual learning. In Findings of the Association for Computational Linguistics: EMNLP 2023, pp. 10658–10671, 2023a.

Xiao Wang, Yuansen Zhang, Tianze Chen, Songyang Gao, Senjie Jin, Xianjun Yang, Zhiheng Xi, Rui Zheng, Yicheng Zou, Tao Gui, et al. Trace: A comprehensive benchmark for continual learning in large language models. arXiv preprint arXiv:2310.06762, 2023b.

Jaehong Yoon, Saehoon Kim, Eunho Yang, and Sung Ju Hwang. Scalable and order-robust continual learning with additive parameter decomposition. In International Conference on Learning Representations, 2020. URL https://openreview.net/forum?id=r1gdj2EKPB.

Chenye Zhao, Yingjie Li, and Cornelia Caragea. C-stance: A large dataset for chinese zero-shot stance detection. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 13369–13385, 2023.

## APPENDIX

## A DERIVATION OF ADAPTATION BENEFIT AND FORGETTING COST

For notational simplicity, we suppress the task index t and layer index l throughout this subsection, while retaining the channel index i. Accordingly, $\mathcal { N } , G _ { \mathrm { n e w } } , \mathcal { L } _ { \mathrm { n e w } } .$ , and $\Sigma _ { \mathrm { o l d } }$ correspond to $\mathbf { W } _ { t - 1 } ^ { ( l ) }$ $\mathbf { G } _ { t } ^ { ( l ) } , \mathcal { L } _ { t }$ , and $\Sigma _ { \mathrm { o l d } } ^ { ( l ) }$ , respectively. The task and layer indices are restored directly in the main-text expressions. Consider a linear layer $y = W x$ , where $W \in \mathbb { R } ^ { d _ { \mathrm { o u t } } \times d _ { \mathrm { i n } } }$ is the weight matrix before fine-tuning and $x \in \mathbb { R } ^ { d _ { \mathrm { i n } } }$ is the input activation. The SVD of W is

$$
W = U \Sigma V ^ { \top } = \sum _ { i = 1 } ^ { r } \sigma _ { i } \mathbf { u } _ { i } \mathbf { v } _ { i } ^ { \top } ,\tag{14}
$$

where $r = { \mathrm { r a n k } } ( W )$ . We refer to ${ \cal S } _ { i } = ( { \bf u } _ { i } , { \bf v } _ { i } )$ as the i-th singular-vector channel and to $\sigma _ { i }$ as its associated singular value. Together, they define the rank-one component $\sigma _ { i } \mathbf { u } _ { i } \mathbf { v } _ { i } ^ { \top }$ . The following benefit and cost analysis is carried out at the granularity of individual singular-vector channels.

The adaptation benefit is naturally characterized by the reduction in domain loss. Forgetting, however, cannot be fully characterized by a single loss, so following methods such as GPM, we instead measure it by the output drift of the linear layer on old-task input activations. Let ${ \mathcal { L } } _ { \mathrm { n e w } } ( W )$ denote the new-task loss, and let $\Sigma _ { \mathrm { o l d } } = \mathbb { E } _ { { x } \sim \mathcal { D } _ { \mathrm { o l d } } } [ \bar { { x } } { x } ^ { \top } ] \in \mathbb { R } ^ { d _ { \mathrm { i n } } \times d _ { \mathrm { i n } } }$ denote the second-moment matrix of old-task input activations. For the local update $\Delta W _ { i }$ associated with the i-th singular-vector channel, we define its adaptation benefit $B _ { i }$ and forgetting cost $C _ { i }$ as

$$
B _ { i } = { \mathcal { L } } _ { \mathrm { n e w } } ( W ) - { \mathcal { L } } _ { \mathrm { n e w } } ( W + \Delta W _ { i } )\tag{15}
$$

$$
C _ { i } = \mathbb { E } _ { \boldsymbol { x } \sim \mathcal { D } _ { \mathrm { o l d } } } [ \| \Delta W _ { i } \boldsymbol { x } \| _ { 2 } ^ { 2 } ] = \mathrm { t r } \left( \Delta W _ { i } \Sigma _ { \mathrm { o l d } } \Delta W _ { i } ^ { \top } \right)\tag{16}
$$

In the default SVC formulation, channel selection determines whether the two singular-vector directions $( \mathbf { u } _ { i } , \mathbf { v } _ { i } )$ are trainable, whereas all singular values are optimized regardless of the selection result. We therefore first derive the default benefit and cost scores from the singular-vector perturbations. We then include the singular-value perturbation separately to obtain the alternative scores used in the corresponding ablation.

For completeness, consider small perturbations of the associated singular value and the two singular vectors:

$$
\sigma _ { i }  \sigma _ { i } + \delta \sigma _ { i } \qquad { \bf u } _ { i }  { \bf u } _ { i } + \delta { \bf u } _ { i } , \qquad { \bf v } _ { i }  { \bf v } _ { i } + \delta { \bf v } _ { i }
$$

The local weight update is

$$
\begin{array} { r l } & { \Delta W _ { i } = \delta \sigma _ { i } \mathbf { u } _ { i } \mathbf { v } _ { i } ^ { \top } + \sigma _ { i } \delta \mathbf { u } _ { i } \mathbf { v } _ { i } ^ { \top } + \sigma _ { i } \mathbf { u } _ { i } \delta \mathbf { v } _ { i } ^ { \top } + O ( \delta ^ { 2 } ) } \\ & { \qquad = ( \delta \sigma _ { i } + \sigma _ { i } \mathbf { u } _ { i } ^ { \top } \delta \mathbf { u } _ { i } + \sigma _ { i } \delta \mathbf { v } _ { i } ^ { \top } \mathbf { v } _ { i } ) \mathbf { u } _ { i } \mathbf { v } _ { i } ^ { \top } + \sigma _ { i } \delta \mathbf { u } _ { i , \perp } \mathbf { v } _ { i } ^ { \top } + \sigma _ { i } \mathbf { u } _ { i } \delta \mathbf { v } _ { i , \perp } ^ { \top } + O ( \delta ^ { 2 } ) } \\ & { \qquad = \Delta \sigma _ { i } \mathbf { u } _ { i } \mathbf { v } _ { i } ^ { \top } + \sigma _ { i } \delta \mathbf { u } _ { i , \perp } \mathbf { v } _ { i } ^ { \top } + \sigma _ { i } \mathbf { u } _ { i } \delta \mathbf { v } _ { i , \perp } ^ { \top } + O ( \delta ^ { 2 } ) } \end{array}\tag{17}
$$

where $\Delta \sigma _ { i } = \delta \sigma _ { i } + \sigma _ { i } \mathbf { u } _ { i } ^ { \top } \delta \mathbf { u } _ { i } + \sigma _ { i } \delta \mathbf { v } _ { i } ^ { \top } \mathbf { v } _ { i } , \mathbf { u } _ { i } ^ { \top } \delta \mathbf { u } _ { i , \perp } = 0$ , and $\mathbf { v } _ { i } ^ { \top } \delta \mathbf { v } _ { i , \perp } = 0$ . Ignoring second-order perturbation terms, the default channel score isolates the orthogonal singular-vector perturbations. The singular-value perturbation is considered separately in the ablation derived below.

$$
\begin{array} { r } { \Delta W _ { i } ^ { \perp } = \sigma _ { i } \delta \mathbf { u } _ { i , \perp } \mathbf { v } _ { i } ^ { \top } + \sigma _ { i } \mathbf { u } _ { i } \delta \mathbf { v } _ { i , \perp } ^ { \top } . } \end{array}\tag{18}
$$

Let $G _ { \mathrm { n e w } } = \nabla _ { W } \mathcal { L } _ { \mathrm { n e w } } ( W )$ denote the gradient of the new-task loss with respect to the weight matrix. A first-order Taylor expansion of ${ \mathcal { L } } _ { \mathrm { n e w } }$ at W gives

$$
\mathcal { L } _ { \mathrm { n e w } } ( W + \Delta W _ { i } ^ { \perp } ) - \mathcal { L } _ { \mathrm { n e w } } ( W ) \approx \left. G _ { \mathrm { n e w } } , \Delta W _ { i } ^ { \perp } \right. .\tag{19}
$$

Expanding the inner product with equation 18 substituted in,

$$
\begin{array} { r l } & { \left. { G _ { \mathrm { n e w } } , \Delta W _ { i } ^ { \perp } } \right. = \left. { G _ { \mathrm { n e w } } , \sigma _ { i } \delta \mathbf { u } _ { i , \perp } \mathbf { v } _ { i } ^ { \top } + \sigma _ { i } \mathbf { u } _ { i } \delta \mathbf { v } _ { i , \perp } ^ { \top } } \right. } \\ & { \quad \quad \quad \quad = \sigma _ { i } \mathrm { t r } \left( G _ { \mathrm { n e w } } ^ { \top } \delta \mathbf { u } _ { i , \perp } \mathbf { v } _ { i } ^ { \top } \right) + \sigma _ { i } \mathrm { t r } \left( G _ { \mathrm { n e w } } ^ { \top } \mathbf { u } _ { i } \delta \mathbf { v } _ { i , \perp } ^ { \top } \right) } \\ & { \quad \quad \quad = \sigma _ { i } \mathrm { t r } \left( \mathbf { v } _ { i } ^ { \top } G _ { \mathrm { n e w } } ^ { \top } \delta \mathbf { u } _ { i , \perp } \right) + \sigma _ { i } \mathrm { t r } \left( \delta \mathbf { v } _ { i , \perp } ^ { \top } G _ { \mathrm { n e w } } ^ { \top } \mathbf { u } _ { i } \right) } \\ & { \quad \quad \quad = \left. { \sigma _ { i } G _ { \mathrm { n e w } } \mathbf { v } _ { i } , \delta \mathbf { u } _ { i , \perp } } \right. + \left. { \sigma _ { i } G _ { \mathrm { n e w } } ^ { \top } \mathbf { u } _ { i } , \delta \mathbf { v } _ { i , \perp } } \right. . } \end{array}\tag{20}
$$

This identifies the gradients associated with $\delta \mathbf { u } _ { i , \perp }$ and $\delta \mathbf { v } _ { i , \perp } \mathbf { \hat { \mathbf { \rho } } }$

$$
\mathbf { g } _ { u _ { i } } = \sigma _ { i } G _ { \mathrm { n e w } } \mathbf { v } _ { i } , \qquad \mathbf { g } _ { v _ { i } } = \sigma _ { i } G _ { \mathrm { n e w } } ^ { \top } \mathbf { u } _ { i } .\tag{21}
$$

Since $\delta \mathbf { u } _ { i , \perp }$ and $\delta \mathbf { v } _ { i , \perp }$ must satisfy $\mathbf { u } _ { i } ^ { \top } \delta \mathbf { u } _ { i , \perp } = 0$ and $\mathbf { v } _ { i } ^ { \top } \delta \mathbf { v } _ { i , \perp } = 0$ , we project $\mathbf { g } _ { u _ { i } } , \mathbf { g } _ { v _ { i } }$ onto the orthogonal complements of $\mathbf { u } _ { i } , \mathbf { v } _ { i }$ to obtain the feasible effective gradients:

$$
\mathbf { p } _ { i } = ( I - \mathbf { u } _ { i } \mathbf { u } _ { i } ^ { \top } ) \mathbf { g } _ { u _ { i } } , \qquad \mathbf { q } _ { i } = ( I - \mathbf { v } _ { i } \mathbf { v } _ { i } ^ { \top } ) \mathbf { g } _ { v _ { i } } .\tag{22}
$$

Updating along $- \mathbf { p } _ { i } , - \mathbf { q } _ { i }$ with step size η and substituting into equation 18 yields the local weight update along the orthogonal perturbation directions:

$$
\Delta W _ { i } ^ { \perp } = - \eta \sigma _ { i } \mathbf { p } _ { i } \mathbf { v } _ { i } ^ { \top } - \eta \sigma _ { i } \mathbf { u } _ { i } \mathbf { q } _ { i } ^ { \top }\tag{23}
$$

Substituting equation 23 into the benefit definition, equation 15, and using $\mathbf { u } _ { i } ^ { \top } \mathbf { p } _ { i } = 0 , \mathbf { v } _ { i } ^ { \top } \mathbf { q } _ { i } = 0$ to eliminate the cross terms

$$
\begin{array} { r l } & { B _ { i } = \mathcal { L } _ { \mathrm { n e w } } ( W ) - \mathcal { L } _ { \mathrm { n e w } } ( W + \Delta W _ { i } ^ { \perp } ) } \\ & { \quad \approx - \left. G _ { \mathrm { n e w } } , \Delta W _ { i } ^ { \perp } \right. } \\ & { \quad = \eta \sigma _ { i } \left. G _ { \mathrm { n e w } } , \mathbf { p } _ { i } \mathbf { v } _ { i } ^ { \top } + \mathbf { u } _ { i } \mathbf { q } _ { i } ^ { \top } \right. } \\ & { \quad = \eta \sigma _ { i } \operatorname { t r } ( G _ { \mathrm { n e w } } ^ { \top } \mathbf { p } _ { i } \mathbf { v } _ { i } ^ { \top } ) + \eta \sigma _ { i } \operatorname { t r } ( G _ { \mathrm { n e w } } ^ { \top } \mathbf { u } _ { i } \mathbf { q } _ { i } ^ { \top } ) } \\ & { \quad = \eta \operatorname { t r } ( ( \sigma _ { i } G _ { \mathrm { n e w } } \mathbf { v } _ { i } ) ^ { \top } \mathbf { p } _ { i } ) + \eta \operatorname { t r } ( \mathbf { q } _ { i } ^ { \top } ( \sigma _ { i } G _ { \mathrm { n e w } } ^ { \top } \mathbf { u } _ { i } ) ) } \\ & { \quad = \eta \operatorname { t r } ( ( \mathbf { p } _ { i } + \sigma _ { i } ( \mathbf { u } _ { i } ^ { \top } G _ { \mathrm { n e w } } \mathbf { v } _ { i } ) \mathbf { u } _ { i } ) ^ { \top } \mathbf { p } _ { i } ) + \eta \operatorname { t r } ( \mathbf { q } _ { i } ^ { \top } ( \mathbf { q } _ { i } + \sigma _ { i } ( \mathbf { v } _ { i } ^ { \top } G _ { \mathrm { n e w } } ^ { \top } \mathbf { u } _ { i } ) \mathbf { v } _ { i } ) ) } \\ & { \quad = \eta \left( \lVert \mathbf { p } _ { i } \rVert _ { 2 } ^ { 2 } + \lVert \mathbf { q } _ { i } \rVert _ { 2 } ^ { 2 } \right) . } \end{array}\tag{24}
$$

Similarly, substituting equation 23 into the cost definition, equation 16,

$$
\begin{array} { r l } & { C _ { i } = \mathrm { t r } ( \Delta W _ { i } ^ { \perp } \Sigma _ { \mathrm { o l d } } ( \Delta W _ { i } ^ { \perp } ) ^ { \top } ) } \\ & { \quad = \eta ^ { 2 } \sigma _ { i } ^ { 2 } \mathrm { t r } [ ( \mathbf { p } _ { i } \mathbf { v } _ { i } ^ { \top } + \mathbf { u } _ { i } \mathbf { q } _ { i } ^ { \top } ) \Sigma _ { \mathrm { o l d } } ( \mathbf { v } _ { i } \mathbf { p } _ { i } ^ { \top } + \mathbf { q } _ { i } \mathbf { u } _ { i } ^ { \top } ) ] } \\ & { \quad = \eta ^ { 2 } \sigma _ { i } ^ { 2 } \mathrm { t r } ( \mathbf { p } _ { i } \mathbf { v } _ { i } ^ { \top } \Sigma _ { \mathrm { o l d } } \mathbf { v } _ { i } \mathbf { p } _ { i } ^ { \top } ) + \eta ^ { 2 } \sigma _ { i } ^ { 2 } \mathrm { t r } ( \mathbf { p } _ { i } \mathbf { v } _ { i } ^ { \top } \Sigma _ { \mathrm { o l d } } \mathbf { q } _ { i } \mathbf { u } _ { i } ^ { \top } ) } \\ & { \quad \quad + \eta ^ { 2 } \sigma _ { i } ^ { 2 } \mathrm { t r } ( \mathbf { u } _ { i } \mathbf { q } _ { i } ^ { \top } \Sigma _ { \mathrm { o l d } } \mathbf { v } _ { i } \mathbf { p } _ { i } ^ { \top } ) + \eta ^ { 2 } \sigma _ { i } ^ { 2 } \mathrm { t r } ( \mathbf { u } _ { i } \mathbf { q } _ { i } ^ { \top } \Sigma _ { \mathrm { o l d } } \mathbf { q } _ { i } \mathbf { u } _ { i } ^ { \top } ) } \\ &  \quad = \eta ^ { 2 } \sigma _ { i } ^ { 2 } [ ( \mathbf { v } _ { i } ^ { \top } \Sigma _ { \mathrm { o l d } } \mathbf { v } _ { i } ) \mathrm { t r } ( \mathbf { p } _ { i } \mathbf { p } _ { i } ^ { \top } ) + ( \mathbf { v } \end{array}\tag{25}
$$

After omitting the positive learning-rate factors shared by all channels, equation 24 and equation 25 yield the default benefit and forgetting-cost scores reported in the main text.

Singular-value-inclusive ablation. For the ablation in which the channel is instead defined as $\widetilde { \cal S } _ { i } = ( \sigma _ { i } , { \bf u } _ { i } , { \bf v } _ { i } )$ , we include the singular-value perturbation in the same local analysis.

Incorporating the singular-value perturbation, gradient descent gives

$$
\Delta \sigma _ { i } = - \eta g _ { i } , \qquad g _ { i } \triangleq  { \mathbf { u } } _ { i } ^ { \top } G _ { \mathrm { n e w } }  { \mathbf { v } } _ { i } .\tag{26}
$$

The full update of channel i is

$$
\Delta W _ { i } = - \eta \left( g _ { i } \mathbf { u } _ { i } \mathbf { v } _ { i } ^ { \top } + \sigma _ { i } \mathbf { p } _ { i } \mathbf { v } _ { i } ^ { \top } + \sigma _ { i } \mathbf { u } _ { i } \mathbf { q } _ { i } ^ { \top } \right) .\tag{27}
$$

Substituting into equation 15,

$$
\begin{array} { r l } & { \tilde { B } _ { i } = \mathcal { L } _ { \mathrm { n e w } } ( W ) - \mathcal { L } _ { \mathrm { n e w } } ( W + \Delta W _ { i } ) } \\ & { \quad \approx - \left. G _ { \mathrm { n e w } } , \Delta W _ { i } \right. } \\ & { \quad = \eta \left. G _ { \mathrm { n e w } } , g _ { i } \mathbf { u } _ { i } \mathbf { v } _ { i } ^ { \top } + \sigma _ { i } \mathbf { p } _ { i } \mathbf { v } _ { i } ^ { \top } + \sigma _ { i } \mathbf { u } _ { i } \mathbf { q } _ { i } ^ { \top } \right. } \\ & { \quad = \eta g _ { i } \mathbf { t r } ( G _ { \mathrm { n e w } } ^ { \top } \mathbf { u } _ { i } \mathbf { v } _ { i } ^ { \top } ) + \eta \sigma _ { i } \operatorname { t r } ( G _ { \mathrm { n e w } } ^ { \top } \mathbf { p } _ { i } \mathbf { v } _ { i } ^ { \top } ) + \eta \sigma _ { i } \operatorname { t r } ( G _ { \mathrm { n e w } } ^ { \top } \mathbf { u } _ { i } \mathbf { q } _ { i } ^ { \top } ) } \\ & { \quad = \eta g _ { i } \left( \mathbf { v } _ { i } ^ { \top } G _ { \mathrm { n e w } } ^ { \top } \mathbf { u } _ { i } \right) + \eta \operatorname { t r } ( ( \sigma _ { i } G _ { \mathrm { n e w } } \mathbf { v } _ { i } ) ^ { \top } \mathbf { p } _ { i } ) + \eta \operatorname { t r } ( \mathbf { q } _ { i } ^ { \top } ( \sigma _ { i } G _ { \mathrm { n e w } } ^ { \top } \mathbf { u } _ { i } ) ) } \\ &  \quad = \eta g _ { i } ^ { 2 } + \eta \operatorname { t r } ( ( \mathbf { p } _ { i } + \sigma _ { i } ( \mathbf { u } _ { i } ^ { \top } G _ { \mathrm { n e w } } \mathbf { v } _ { i } ) \mathbf { u } _ { i } ) ^ { \top } \mathbf { p } _ { i } ) + \eta \operatorname { t r } ( \mathbf { q } _ { i } ^ { \top } ( \mathbf  \end{array}\tag{28}
$$

Substituting into equation 16,

$$
\begin{array} { r l } & { \widetilde { C } _ { i } = \mathrm { t r } ( \Delta W _ { i } \Sigma _ { \mathrm { o l d } } \Delta W _ { i } ^ { \top } ) } \\ & { \quad = \eta ^ { 2 } \mathrm { t r } [ ( g _ { i } \mathbf { u } _ { i } \mathbf { v } _ { i } ^ { \top } + \sigma _ { i } \mathbf { p } _ { i } \mathbf { v } _ { i } ^ { \top } + \sigma _ { i } \mathbf { u } _ { i } \mathbf { q } _ { i } ^ { \top } ) \Sigma _ { \mathrm { o l d } } ( g _ { i } \mathbf { v } _ { i } \mathbf { u } _ { i } ^ { \top } + \sigma _ { i } \mathbf { v } _ { i } \mathbf { p } _ { i } ^ { \top } + \sigma _ { i } \mathbf { q } _ { i } \mathbf { u } _ { i } ^ { \top } ) ] } \\ & { \quad = \eta ^ { 2 } g _ { i } ^ { 2 } \mathrm { t r } ( \mathbf { u } _ { i } \mathbf { v } _ { i } ^ { \top } \Sigma _ { \mathrm { o l d } } \mathbf { v } _ { i } \mathbf { u } _ { i } ^ { \top } ) + \eta ^ { 2 } g _ { i } \sigma _ { i } \mathrm { t r } ( \mathbf { u } _ { i } \mathbf { v } _ { i } ^ { \top } \Sigma _ { \mathrm { o l d } } \mathbf { v } _ { i } \mathbf { p } _ { i } ^ { \top } ) + \eta ^ { 2 } g _ { i } \sigma _ { i } \mathrm { t r } ( \mathbf { u } _ { i } \mathbf { v } _ { i } ^ { \top } \Sigma _ { \mathrm { o l d } } \mathbf { q } _ { i } \mathbf { u } _ { i } ^ { \top } ) } \\ &  \quad \quad + \eta ^ { 2 } \sigma _ { i } g _ { i } \mathrm { t r } ( \mathbf { p } _ { i } \mathbf { v } _ { i } ^ { \top } \Sigma _ { \mathrm { o l d } } \mathbf { v } _ { i } \mathbf { u } _ { i } ^ { \top } ) + \eta ^ { 2 } \sigma _ { i } ^ { 2 } \mathrm { t r } ( \mathbf { p } _ { i } \mathbf { v } _ { i }  \end{array}\tag{29}
$$

Reducing each term via $\operatorname { t r } ( \mathbf { a } \mathbf { b } ^ { \top } M ) = \mathbf { b } ^ { \top }$ Ma, and eliminating cross terms with $\mathbf { u } _ { i } ^ { \top } \mathbf { p } _ { i } = 0 , \mathbf { v } _ { i } ^ { \top } \mathbf { q } _ { i } =$ 0,

$$
\begin{array} { r l } & { \mathrm { t r } ( \mathbf { u } _ { i } \mathbf { v } _ { i } ^ { \top } \Sigma _ { \mathrm { o d d } } \mathbf { v } _ { i } \mathbf { u } _ { i } ^ { \top } ) = \big ( \mathbf { v } _ { i } ^ { \top } \Sigma _ { \mathrm { o d d } } \mathbf { v } _ { i } \big ) \big ( \mathbf { u } _ { i } ^ { \top } \mathbf { u } _ { i } \big ) = \mathbf { v } _ { i } ^ { \top } \Sigma _ { \mathrm { o d d } } \mathbf { v } _ { i } , } \\ & { \mathrm { t r } ( \mathbf { u } _ { i } \mathbf { v } _ { i } ^ { \top } \Sigma _ { \mathrm { o d d } } \mathbf { v } _ { i } \mathbf { u } _ { i } ^ { \top } ) = \big ( \mathbf { v } _ { i } ^ { \top } \Sigma _ { \mathrm { o d d } } \mathbf { v } _ { i } \big ) \big ( \mathbf { p } _ { i } ^ { \top } \mathbf { u } _ { i } \big ) = 0 , } \\ & { \mathrm { t r } ( \mathbf { u } _ { i } \mathbf { v } _ { i } ^ { \top } \Sigma _ { \mathrm { o d d } } \mathbf { q } _ { i } \mathbf { u } _ { i } ^ { \top } ) = \big ( \mathbf { v } _ { i } ^ { \top } \Sigma _ { \mathrm { o d d } } \mathbf { q } _ { i } \big ) \big ( \mathbf { u } _ { i } ^ { \top } \mathbf { u } _ { i } \big ) = \mathbf { v } _ { i } ^ { \top } \Sigma _ { \mathrm { o d d } } \mathbf { z } _ { i } , } \\ & { \mathrm { t r } ( \mathbf { p } _ { i } \mathbf { v } _ { i } ^ { \top } \Sigma _ { \mathrm { o d d } } \mathbf { v } _ { i } \mathbf { u } _ { i } ^ { \top } ) = \big ( \mathbf { v } _ { i } ^ { \top } \Sigma _ { \mathrm { o d d } } \mathbf { v } _ { i } \big ) \big ( \mathbf { u } _ { i } ^ { \top } \mathbf { p } _ { i } \big ) = 0 , } \\ &  \mathrm { t r } ( \mathbf { p } _ { i } \mathbf { v } \end{array}\tag{30}
$$

Substituting back into equation 29 and keeping only the nonzero terms,

$$
\begin{array} { r l r } {  { \widetilde { C } _ { i } = \eta ^ { 2 } g _ { i } ^ { 2 } ( \mathbf { v } _ { i } ^ { \top } \Sigma _ { \mathrm { o l d } } \mathbf { v } _ { i } ) + 2 \eta ^ { 2 } g _ { i } \sigma _ { i } ( \mathbf { v } _ { i } ^ { \top } \Sigma _ { \mathrm { o l d } } \mathbf { q } _ { i } ) + \eta ^ { 2 } \sigma _ { i } ^ { 2 } ( \mathbf { v } _ { i } ^ { \top } \Sigma _ { \mathrm { o l d } } \mathbf { v } _ { i } ) \| \mathbf { p } _ { i } \| _ { 2 } ^ { 2 } + \eta ^ { 2 } \sigma _ { i } ^ { 2 } ( \mathbf { q } _ { i } ^ { \top } \Sigma _ { \mathrm { o l d } } \mathbf { q } _ { i } ) } } \\ & { } & { ( 3 ! ) ^ { 2 } [ \sigma _ { i } ^ { 2 } \| \mathbf { p } _ { i } \| _ { 2 } ^ { 2 } ( \mathbf { v } _ { i } ^ { \top } \Sigma _ { \mathrm { o l d } } \mathbf { v } _ { i } ) + ( g _ { i } \mathbf { v } _ { i } + \sigma _ { i } \mathbf { q } _ { i } ) ^ { \top } \Sigma _ { \mathrm { o l d } } ( g _ { i } \mathbf { v } _ { i } + \sigma _ { i } \mathbf { q } _ { i } ) ] . } \end{array}
$$

The default scores in equation 24-equation 25 therefore quantify the marginal effect of enabling the singular-vector directions, whereas equation 28-equation 31 quantify the joint perturbation of the singular value and its associated singular-vector pair for the corresponding ablation.

Table 3: Statistics of the continual-learning tasks and capability-preservation benchmarks. A dash indicates that a benchmark is used only for evaluation and is not included in continual training.
<table><tr><td rowspan="2">Split</td><td colspan="8">Continual-learning tasks</td><td colspan="2">Capability-preservation benchmarks</td></tr><tr><td>Sin.</td><td>Fom.</td><td>Mee.</td><td>Cst.</td><td>20M.</td><td>Py150</td><td>Ngc.</td><td>Ngd.</td><td>HumanEval</td><td>GSM8K</td></tr><tr><td>Train</td><td></td><td></td><td>1,000 examples per task</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Test</td><td>2,000</td><td>496</td><td>692 </td><td>2,000</td><td>200</td><td>2,000</td><td>81</td><td>325</td><td>164</td><td>1,319</td></tr></table>

## B DATASETS

## B.1 CONTINUAL-LEARNING TASKS

We construct the continual-learning sequences using the eight language tasks from TRACE (Wang et al., 2023b). TRACE combines challenging and heterogeneous tasks for which pretrained language models generally have substantial room for adaptation. The selected tasks cover specialized domains, multiple languages, code completion, and mathematical reasoning. Following the TRACE protocol, we use 1,000 training examples for each task and retain the corresponding test sets for evaluation. Dataset statistics are summarized in Table 3.

Domain-specific tasks. This group evaluates adaptation to scientific, financial, and meetingrelated domains.

• ScienceQA (Sin.) (Lu et al., 2022) contains questions derived from elementary- and highschool curricula across natural, social, and language science. We retain only the text-based examples, thereby excluding samples that require visual inputs.

• FOMC (Fom.) (Shah et al., 2023) is a financial stance classification task in which the model determines whether a Federal Reserve policy statement is hawkish or dovish. We use the combined version containing meeting minutes, press-conference transcripts, and speeches.

• MeetingBank (Mee.) (Hu et al., 2023) is a meeting summarization dataset built from citycouncil proceedings. It requires the model to identify salient information from long transcripts and generate concise summaries.

Multilingual tasks. The multilingual tasks evaluate whether a model can acquire new information and follow task-specific output formats in languages other than English.

• C-STANCE (Cst.) (Zhao et al., 2023) is a Chinese stance detection dataset collected from Sina Weibo. We use its target-based setting, in which the targets appearing in the test set are not observed during training.

• 20Minuten (20M.) (Gonzales et al., 2021) is a German text simplification dataset derived from the Swiss news outlet 20 Minuten. Each example pairs a news article with a simplified version, making the task a conditional German text-generation problem.

Code-completion task. Py150 (Lu et al., 2021) is constructed from approximately 150,000 Python source files collected from public GitHub repositories. Given a code context, the model must predict the subsequent code content. The task evaluates adaptation under structured inputs and output constraints that differ from ordinary natural-language generation.

Mathematical-reasoning tasks. TRACE includes two tasks derived from NumGLUE (Mishra et al., 2022): NumGLUE-cm (Ngc.) and NumGLUE-ds (Ngd.). Both tasks require numerical computation and multi-step reasoning over natural-language problem statements, providing complementary continual-learning tasks for mathematical reasoning.

## B.2 CAPABILITY-PRESERVATION BENCHMARKS

We evaluate the preservation of pretrained code-generation and mathematical-reasoning capabilities using HumanEval and GSM8K, respectively. Neither benchmark is included in the continualtraining sequences.

HumanEval. HumanEval (Chen et al., 2021) contains 164 hand-written Python programming problems. Each problem specifies a function signature and a natural-language docstring, from which the model must generate a complete function implementation. Correctness is determined by executing the generated code against hidden unit tests. We use the full benchmark to measure whether sequential domain adaptation degrades the model’s pretrained code-generation capability.

GSM8K. GSM8K (Cobbe et al., 2021) is a collection of linguistically diverse grade-school mathematics word problems written by human annotators. Its problems typically require multiple steps of elementary arithmetic together with natural-language reasoning. We evaluate models on the complete official test split of 1,319 problems and report the proportion of problems for which the model produces the correct final answer. GSM8K examples are used only for evaluation, allowing changes in performance to reflect the retention of pretrained mathematical-reasoning capability rather than additional training on the benchmark.

## B.3 GENERAL-DOMAIN REFERENCE CORPUS

SVC estimates the potential forgetting cost of each singular-vector channel using activation statistics collected from an unlabeled general-domain reference corpus. We construct this corpus by sampling 30,000 examples from C4 (Raffel et al., 2020) and the Python subset of The Stack v2 (Lozhkov et al., 2024) at a fixed ratio of 2:1. This corresponds to 20,000 C4 examples and 10,000 Python examples. The sampled corpus is fixed and shared across all models, tasks, and task orders.

The reference corpus is not used to optimize model parameters and does not provide supervision for any continual-learning task. We use only its input texts to collect layer-wise activations and estimate the corresponding second-moment matrices required by the forgetting-cost calculation. Thus, the same reference data can be reused throughout the task sequence without storing examples from previous tasks.

## C BASELINES AND IMPLEMENTATION DETAILS

## C.1 BASELINES

We compare SVC with four groups of continual-learning baselines: regularization and gradientconstraint methods, replay-based methods, parameter-efficient fine-tuning methods, and full or constrained full-parameter fine-tuning methods. Unless otherwise specified, all methods use the same task data, task orders, optimization schedule, and evaluation protocols.

## C.1.1 REGULARIZATION AND GRADIENT-CONSTRAINT METHODS

Elastic Weight Consolidation. Elastic Weight Consolidation (EWC) (Kirkpatrick et al., 2017) penalizes changes to parameters that are estimated to be important for previous tasks. When learning task $t ,$ its objective is

$$
\mathcal { L } _ { \mathrm { E W C } } = \mathcal { L } _ { t } ( \theta ) + \lambda \sum _ { i } F _ { i } \left( \theta _ { i } - \theta _ { i } ^ { * } \right) ^ { 2 } ,\tag{32}
$$

where $\theta ^ { * }$ denotes the parameters obtained after the preceding task and $F _ { i }$ is the estimated Fisher information for parameter i. We set the regularization coefficient to $\lambda = 0 . 5$

Gradient Episodic Memory. Gradient Episodic Memory (GEM) (Lopez-Paz & Ranzato, 2017) maintains an episodic memory $\mathcal { M } _ { k }$ for each previous task k. When learning task $t ,$ it computes the current gradient $g _ { t }$ and a reference gradient $g _ { k }$ from each $\mathcal { M } _ { k }$ . If $\langle g _ { t } , g _ { k } \rangle < 0$ for any $k < t ,$ GEM

replaces $g _ { t }$ with the solution to

$$
\begin{array} { r l r } { \underset { \boldsymbol { \widetilde { g } } } { \operatorname* { m i n } } } & { \frac { 1 } { 2 } \| \boldsymbol { \widetilde { g } } - \boldsymbol { g } _ { t } \| _ { 2 } ^ { 2 } , } \\ { \mathrm { s . t . } } & { \langle \boldsymbol { \widetilde { g } } , \boldsymbol { g } _ { k } \rangle \ge 0 , } & { \forall k < t . } \end{array}\tag{33}
$$

The resulting gradient is used for parameter updates, limiting interference with the stored examples from earlier tasks.

Learning without Forgetting. Learning without Forgetting (LwF) (Li & Hoiem, 2017) retains a frozen copy of the preceding model and uses its predictions as soft targets. The training objective combines the current-task loss with a knowledge-distillation term:

$$
\mathcal { L } _ { \mathrm { L w F } } = \mathcal { L } _ { t } ^ { \mathrm { C E } } + \alpha \mathcal { L } _ { \mathrm { K D } } .\tag{34}
$$

We use a distillation weight of $\alpha = 0 . 5$ and a softmax temperature of $T = 2$

## C.1.2 REPLAY-BASED METHODS

Both replay baselines maintain a unified memory buffer containing real examples from previously learned tasks. Samples are drawn uniformly across earlier tasks, and no synthetic data or auxiliary generative model is used.

Replay. After training on the current task, Replay performs an additional rehearsal phase using samples drawn from the memory buffer. For each previous task, the buffer stores and replays 1% of its training examples.

Replay-online. Replay-online interleaves historical examples with current-task examples during training. At each stage, approximately 1% of the stored data are sampled from the same memory buffer and mixed with the current-task mini-batches.

## C.1.3 PARAMETER-EFFICIENT FINE-TUNING METHODS

For SeqLoRA, O-LoRA, PiSSA, and MiLoRA, we use rank $r = 8 ,$ , scaling factor $\alpha = 3 2$ , and apply the adapters to the q proj and v proj modules. SeqLoRA, O-LoRA, and MiLoRA use an adapter dropout of 0.1. For PiSSA, we disable adapter dropout following its original implementation. PiSSA and MiLoRA otherwise use the same LoRA configuration and differ primarily in whether their lowrank factors are initialized from the principal or minor singular components.

Sequential LoRA. Low-Rank Adaptation (LoRA) (Hu et al., 2022) parameterizes the update to a frozen weight matrix W as

$$
W ^ { \prime } = W + { \frac { \alpha } { r } } B A ,\tag{35}
$$

where A and B are trainable low-rank factors. Sequential LoRA (SeqLoRA) continually updates the same adapter parameters across the complete task sequence.

Orthogonal LoRA. Orthogonal LoRA (O-LoRA) (Wang et al., 2023a) assigns successive tasks to task-specific low-rank subspaces and regularizes these subspaces toward orthogonality. It uses the same LoRA configuration as SeqLoRA, with the orthogonality coefficient set to $\gamma = 0 . 5$

PiSSA. Principal Singular Values and Singular Vectors Adaptation (PiSSA) (Meng et al., 2024) decomposes each pretrained weight matrix through singular value decomposition. The r largest singular values and their corresponding left and right singular vectors form a rank-r principal com ponent, which is used to initialize the trainable low-rank factors. The remaining singular components are absorbed into a frozen residual matrix, such that the trainable component and the residual together reconstruct the original weight at initialization. PiSSA uses the same rank, scaling factor, and target modules as the other LoRA-based baselines, but disables adapter dropout.

MiLoRA. Minor Singular Component Low-Rank Adaptation (MiLoRA) (Wang et al., 2025a) also decomposes each pretrained weight matrix through singular value decomposition. In contrast to PiSSA, MiLoRA uses the r smallest singular values and their corresponding left and right singular vectors to initialize the trainable low-rank factors. The remaining principal singular components are kept frozen, allowing adaptation to occur within the less dominant singular subspace. MiLoRA uses the same rank, scaling factor, and target modules as SeqLoRA and PiSSA, and uses the same adapter dropout of 0.1 as SeqLoRA.

## C.1.4 FULL AND CONSTRAINED FINE-TUNING METHODS

Sequential Full-Parameter Fine-Tuning. Sequential full-parameter fine-tuning (SeqFT) updates all model parameters on each task in sequence without explicit regularization, replay, or subspace constraint.

Orthogonal Subspace Fine-Tuning. Orthogonal Subspace Fine-Tuning (OSFT) (Nayak et al., 2026) constrains full-parameter updates through layer-wise singular subspaces. Before learning each task, OSFT decomposes every adapted weight matrix as $W ^ { ( l ) } = U ^ { ( l ) } \dot { \Sigma ^ { ( l ) } } V ^ { ( l ) \top }$ . It then estimates the importance $I ^ { ( l ) }$ of layer l from the average cosine similarity between that layer’s input and output activations on a probing corpus. Layers with higher importance scores are assigned larger protected subspaces. The fraction of singular directions retained at layer l is determined by

$$
r ^ { ( l ) } = \mathrm { m r r } + I ^ { ( l ) } \left( \mathrm { t r r - m r } \right) ,\tag{36}
$$

where the normalized importance score $I ^ { ( l ) }$ controls the allocation between the minimum retention ratio mrr and the target retention ratio trr. We follow the original configuration and set mrr = 0.1 and trr = 0.8.

Based on $r ^ { ( l ) }$ , OSFT selects the singular vectors associated with the largest singular values to form the protected left and right subspaces, denoted by $U _ { \mathrm { h i g h } } ^ { ( l ) }$ and $V _ { \mathrm { h i g h } } ^ { ( l ) }$ , respectively. Given the current gradient $G ^ { ( l ) }$ , OSFT removes the component aligned with both protected subspaces:

$$
G _ { \mathrm { p r o j } } ^ { ( l ) } = G ^ { ( l ) } - U _ { \mathrm { h i g h } } ^ { ( l ) } U _ { \mathrm { h i g h } } ^ { ( l ) \top } G ^ { ( l ) } V _ { \mathrm { h i g h } } ^ { ( l ) } V _ { \mathrm { h i g h } } ^ { ( l ) \top } .\tag{37}
$$

The projected gradient $G _ { \mathrm { p r o j } } ^ { ( l ) }$ is then used to update the full weight matrix.

The original OSFT protocol estimates layer importance using samples from the preceding task. This is incompatible with our setting, in which historical task data cannot be revisited and the origi nal pretraining data are unavailable before the first continual-learning task. We therefore use the same fixed, unlabeled general-domain reference corpus employed by SVC as the probing corpus for OSFT. All other OSFT configurations follow the original method.

Additional implementation details. Unless otherwise specified above, all methods follow the common optimization and training settings described in Section 4.1. SVC is applied to all linear modules in the model. Its trainable singular-vector channels are selected adaptively for each layer and task, without a manually specified global selection ratio.

## C.2 TASK ORDERS

We evaluate every method under three task orders to account for the sensitivity of continual-learning performance to task ordering. Each order contains all eight TRACE tasks, and the same orders are used for every model and baseline.

Order 1: C-STANCE → FOMC → MeetingBank → ScienceQA → NumGLUE-cm → NumGLUEds → 20Minuten → Py150.

Order 2: NumGLUE-cm → NumGLUE-ds → FOMC → 20Minuten → C-STANCE → Meeting-Bank → ScienceQA → Py150.

Order 3: MeetingBank → FOMC → 20Minuten → C-STANCE → ScienceQA → NumGLUE-cm → NumGLUE-ds → Py150.

Table 4: Effect of target-module coverage on Llama3-8B under Order 1. The default PiSSA and MiLoRA results and the SVC result are reproduced from Table 1. Higher is better for all metrics.
<table><tr><td>Method</td><td>Target modules</td><td>HumanEval</td><td>GSM8K</td><td>OP</td><td>BWT</td></tr><tr><td>PiSSA</td><td>q-proj, v-proj</td><td>28.05</td><td>39.42</td><td>47.81</td><td>-2.47</td></tr><tr><td>PiSSA</td><td>All linear</td><td>10.37</td><td>3.56</td><td>39.80</td><td>-4.19</td></tr><tr><td>MiLoRA</td><td>q-proj, v-proj</td><td>29.27</td><td>45.49</td><td>47.90</td><td>-1.55</td></tr><tr><td>MiLoRA</td><td>All linear</td><td>24.39</td><td>41.55</td><td>48.46</td><td>-1.65</td></tr><tr><td>SVC</td><td>All linear</td><td>30.49</td><td>51.55</td><td>49.62</td><td>-0.94</td></tr></table>

## C.3 MOTIVATION EXPERIMENT DETAILS

Figure 1 compares LoRA and SVF under independent adaptation to the eight TRACE tasks. Each run starts from the same Llama3-8B pretrained checkpoint and uses the task data and splits described in Section 4.1 and Appendix B.1. LoRA follows the configuration in Appendix C.1.3, with rank $r = 8$ , scaling factor $\alpha = 3 2$ , dropout 0.1, and adapters on q proj and v proj. SVF updates all singular values in every linear layer while keeping the corresponding left and right singular vectors fixed. Both methods use the training settings in Section 4.1: Adam, a batch size of 64, five epochs, no warmup, and a learning rate of $1 \overline { { 0 } } ^ { - 4 }$ . Figure 1(a) reports GSM8K Pass@1 before and after each independent adaptation. Figure 1(b) reports performance on each task’s test set using the metrics defined in Section 4.1.

## D EFFECT OF EXPANDING PEFT TARGET MODULES

To examine whether SVC’s advantage can be explained solely by its broader module coverage, we expanded the target modules of PiSSA and MiLoRA from q proj and v proj to all linear modules. We evaluated both methods on Llama3-8B under Order 1, keeping their other experimental settings unchanged. Table 4 compares these runs with the corresponding main results. Expanding the target modules reduced HumanEval and GSM8K performance for both methods. PiSSA also showed lower OP and BWT. MiLoRA’s OP increased slightly, from 47.90 to 48.46, while its BWT decreased from −1.55 to −1.65. SVC outperformed both expanded baselines on all four metrics. These results indicate that broader module coverage alone does not account for SVC’s advantage in this setting.