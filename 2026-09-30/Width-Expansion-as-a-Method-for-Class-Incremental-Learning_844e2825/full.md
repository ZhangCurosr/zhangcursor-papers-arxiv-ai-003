# Width Expansion as a Method for Class Incremental Learning

André L. S. Conde<sup>a,∗</sup>, Yehia Elkhatib<sup>b</sup> and Cateano M. Ranieri<sup>a</sup>

<sup>a</sup>Institute ofGeosciences and Exact Sciences, São Paulo State University (UNESP), Av. 24-A, 1515, Rio Claro, 13506-692, SP, Brazil <sup>b</sup>School of Computing Science, University of Glasgow, Glasgow, G12 8QQ, Scotland, United Kingdom

## A R T I C L E I N F O

Keywords:   
Class-Incremental Learning   
Continual Learning   
Catastrophic Forgetting   
Network Expansion   
Width Expansion

## A BS T RA C T

Class Incremental Learning (Class-IL) is a critical frontier in Continual Learning, which requires that models learn new classes over time while preserving previously acquired knowledge without access to past data or task identity. This setting acutely intensifies the stability-plasticity dilemma, making catastrophic forgetting a central challenge. Existing approaches to mitigate forgetting can be broadly categorized into (i) regularization-based methods, which constrain parameter updates; (ii) functional methods, which rely on knowledge distillation; (iii) replay-based strategies, which revisit stored samples; and (iv) architectural approaches, which expand model capacity over time. Regarding architectural expansion, most methods rely on an explicit task identifier or predefined growth strategies, which limit their applicability in Class-IL settings where task boundaries are not available during inference. To address this challenge, this work proposes a dynamic width expansion method that increases the number of neurons within existing layers based on a normalized loss criterion, eliminating the need for task-specific information. Additionally, an attention mechanism with persistent key-value memory is incorporated to stabilize feature representations and mitigate interference between previously learned and newly introduced classes. The proposed approach is evaluated on Split MNIST and Split CIFAR-100 benchmarks under the standard Class-IL protocol. Experiments compare fixed-capacity models with dynamically expanding architectures, both in isolation and in combination with the attention mechanism, using established continual learning strategies such as EWC, LwF, and A-GEM. The results suggest that progressively increasing the model capacity leads to consistent improvements over fixed architectures, particularly when combined with functional methods and A-GEM. The combination of width expansion and attention mechanisms yields the most consistent gains. In conclusion, a dynamically expanding network width based on representational demand provides an efective and flexible strategy for Class-IL. However, caution is required, as uncontrolled growth may lead to overfitting and increased computational cost.

## 1. Introduction

Deep neural networks have achieved remarkable performance across a wide range of tasks when trained under the traditional assumption of independent and identically distributed (i.i.d) data (Cao, 2022). However, in many realworld applications, data are not available all at once but arrive sequentially over time, a problem with non-stationary data that remains unsolved (Hadsell, Rao, Rusu and Pascanu, 2020). In such scenarios, models are required to continuously incorporate new information while retaining previously acquired knowledge. This setting is commonly referred to as continual or incremental learning. A central challenge in this paradigm is catastrophicforgetting, in which learning new information leads to significant performance degradation on previously learned tasks or classes (Liu, Zhou, Liu, Zhao, Yao and Shao, 2023).

Among the diferent continual learning scenarios, classincremental learning (Class-IL) stands out as one of the most challenging. In this setting, the model must learn to discriminate among an ever-growing set of classes using a single shared classifier, without access to task identity during inference. As new classes are introduced incrementally, the model must balance preserving prior knowledge with acquiring new information, thereby intensifying the well-known stability-plasticity dilemma (Kim and Han, 2023).

A broad range of methods has been proposed to address catastrophic forgetting in Class-IL (Van de Ven, Tuytelaars and Tolias, 2022). Regularization-based approaches constrain parameter updates to protect knowledge deemed important for previously learned classes. Functional methods, by contrast, aim to preserve the model’s input-output mapping via knowledge distillation. Replay-based strategies mitigate forgetting by revisiting past data, either through stored samples or generative models. More recently, architectural approaches based on network expansion have been explored, increasing model capacity as new tasks or classes are introduced. Template-Based methods, in turn, retain compact representations of past classes in the form of prototypes or exemplars to guide future predictions. Finally, model rectification approaches aim to correct the bias introduced during incremental updates, typically by re-balancing decision boundaries or calibrating classifier outputs (Zhou, Wang, Qi, Ye, Zhan and Liu, 2024b).

While these approaches have shown promising results, they also present limitations. Regularization and functional methods may excessively restrict model plasticity, replaybased methods introduce additional memory or computational overhead, and expansion-based approaches often rely on task-specific components or predefined growth strategies.

Template-based methods depend on the quality and representativeness of stored prototypes, and model rectification techniques may require additional calibration steps or assumptions about data distribution shifts. (Zhou et al., 2024b)

Recent work highlights the growing momentum ofexpans based methods in the CIL literature. Approaches such as RNE (Jiang, Bai and Zhou, 2026) and Orth-DER (Dong, Zhang, Tan, Qiu and Xie, 2026) extend this paradigm through recurrent inter-expert connections and orthogonality, respectively, while applications to open-world streaming (Li, Dou, Li, Gao and Zhou, 2026) and real-time surveillance (Hussain, Ullah, Khan, Khan, Yar and Baik, 2026) demonstrate its reach across diverse settings. Nevertheless, most of these methods increase model capacity by adding new task-specific modules, which introduces dependence on task identifiers at inference or leads to accelerated parameter growth. In this work, we propose a complementary strategy that expands representational capacity within existing layers, preserving the single shared classifier required by the standard Class-IL protocol.

To further reduce interference between previously learned and newly introduced classes, we incorporate a linear attention mechanism augmented with a persistent key-value memory. This mechanism provides a stable reference across incremental steps, helping to mitigate catastrophic forgetting and representational drift, while enabling the model to adaptively maintain its representations to new data.

The main contributions of this work are:

• A dynamic width expansion mechanism for Class-IL (Section 3.2): a task-agnostic method that increases the number of neurons within existing layers based on a normalized loss criterion, eliminating the need for explicit task identifiers during inference.

• A linear attention mechanism with persistent keyvalue memory (Section 3.3): a complementary module that stabilizes feature representations across incremental steps and mitigates interference between previously learned and newly introduced classes.

• A systematic empirical evaluation (Sections 4 and 5): experiments on Split MNIST and Split CIFAR-100 benchmarks compare fixed-capacity and dynamically expanding architectures, in isolation and in combination with the attention mechanism, across established continual learning strategies including EWC, LwF, and A-GEM.

## 2. Related Work

Catastrophic forgetting in Class-IL has motivated a broad range of approaches, which can be grouped into six paradigms (Zhou et al., 2024b): regularization-based, functional, replay-based, architectural, template-based, and model rectification.

Regularization-based methods such as EWC (Kirkpatrick, Pascanu, Rabinowitz, Veness, Desjardins, Rusu, Milan, Quan, Ramalho, Grabska-Barwinska, Hassabis, Clopath,

Kumaran and Hadsell, 2017) and SI (Zenke, Poole and Ganguli, 2017) constrain parameter updates by penalizing deviations from weights deemed important for prior tasks. While efective at reducing interference, these methods tend n-to over-restrict plasticity in Class-IL, where all classes share a common output space.

Functional methods such as LwF (Li and Hoiem, 2017) and LwM (Dhar, Singh, Peng, Wu and Chellappa, 2019) preserve the model’s input-output behavior via knowledge distillation, rather than constraining individual parameters. LwF encourages consistency with previous model outputs through a temperature-scaled KL divergence, while LwM extends this by additionally penalizing divergence in intermediate attention maps. Both approaches ofer more flexibility than parameter constraints, but may become insuficient when substantial representational adaptation is required.

Replay-based methods such as ER (Rolnick, Ahuja, Schwarz, Lillicrap and Wayne, 2019) and A-GEM (Chaudhry, Ranzato, Rohrbach and Elhoseiny, 2019) mitigate forgetting by revisiting stored samples from prior classes. A-GEM constrains gradient updates so that new learning does not increase loss on bufered samples. These methods often achieve strong performance, but introduce memory overhead and raise scalability concerns as the number of observed classes grows.

Architectural expansion methods increase model capacity as new classes are introduced. DER (Yan, Xie and He, 2021) appends a new extractor while freezing the previous extractor at each step; DNE (Hu, Li, Lyu, Gao and Vasconcelos, 2023) extends this with dense cross-step connections; EASE (Zhou, Sun, Ye and Zhan, 2024a) employs lightweight adapters on a frozen pre-trained backbone; and KANets (Fu, Wang, Xu, Li and Yang, 2023) separates old and new knowledge into parallel branches before compressing them. Recent approaches such as RNE (Jiang et al., 2026) and Orth-DER (Dong et al., 2026) further extend this paradigm via recurrent inter-expert connections and orthogonal constraints, respectively. A common limitation across these methods is reliance on task-specific modules or explicit task identifiers, which are unavailable under the standard Class-IL protocol.

Template-based methods such as iCaRL (Rebufi, Kolesnikov, Sperl and Lampert, 2017), CoPE (De Lange and Tuytelaars, 2021), and SDC (Yu, Twardowski, Liu, Herranz, Wang, Cheng, Jui and Van De Weijer, 2020) retain compact class prototypes to guide classification and representation learning. Their efectiveness is bounded by exemplar quality and the per-class memory budget, both of which degrade as the class space grows. Model rectification methods such as WA (Zhao, Xiao, Gan, Zhang and Xia, 2020) and FACT (Zhou, Wang, Ye, Ma, Pu and Zhan, 2022) address classifier bias introduced during incremental updates, but operate primarily at the output layer and may be insuficient when feature-level drift is substantial.

The method proposed in this work addresses a gap in the architectural expansion paradigm: rather than introducing per-task modules, it expands representational capacity within existing layers, guided by a normalized loss criterion, making it directly applicable to the standard Class-IL setting without access to task identity at inference.

## 3. Proposed Approach

The proposed method operates under the standard Class-IL setting, where a model is trained over a sequence of incremental steps, each introducing a disjoint set of new classes. At each step, the model must learn to recognize newly introduced classes while preserving knowledge of previously learned ones, using a single shared classifier and without access to explicit task identifiers during inference.

The central motivation for the proposed approach stems from the stability-plasticity dilemma inherent in this setting Van de Ven et al. (2022). As new classes are introduced, a fixed-capacity model must reallocate resources, often at the expense of previously acquired knowledge. To address this, we propose incrementally expanding the number of neurons within existing layers, allowing the model to accommodate new classes with reduced interference to older ones. However, increasing capacity alone is insuficient to prevent representational drift, as newly added parameters may disrupt the feature space learned for prior classes. To mitigate this, we further incorporate an attention mechanism with persistent key-value memory, which stabilizes feature representations across incremental steps and reduces interference between previously learned and newly introduced classes.

In contrast to network expansion approaches such as DER and DNE, the proposed method increases representational capacity within existing layers rather than introducing new task-specific modules. While DER expands the feature space by concatenating independent extractors and DNE promotes feature reuse via cross-task attention, both approaches rely on the progressive addition of new structures tied to specific learning steps. The proposed approach, called Width Expansion (WE) and its Width Dynamic layers (WD), departs from this paradigm by enabling continuous adaptation of learned representations, thereby allowing a more flexible balance between stability and plasticity: the model can both retain prior knowledge and refine its feature space as capacity expands.

## 3.1. Preliminaries and Loss Function

The proposed method builds upon a multiclass classification objective based on the cross-entropy loss, defined as:

$$
\mathcal { L } _ { C E } = - \sum _ { k = 1 } ^ { K } y _ { k } l o g ( p _ { k } )\tag{1}
$$

where $y _ { k }$ denotes the ground-truth label and $p _ { k }$ is the predicted probability for class �, obtained through the softmax function:

$$
p _ { k } = s o f t m a x ( z _ { k } ) = { \frac { e ^ { z _ { k } } } { \sum _ { j = 1 } ^ { C } e ^ { z j } } }\tag{2}
$$

with $z _ { k }$ representing the logit associated with class �.

An important reference point in the Class-IL setting is the expected loss of a model that has not yet adapted to the classes introduced at the current incremental step. Under the assumption that such a model produces approximately uniform predictions over all observed classes, the expected cross-entropy loss is given by:

$$
\mathcal { L } _ { m a x } = l n ( | s e e n \_ c l a s s e s | )\tag{3}
$$

where $| C _ { s e e n } |$ denotes the total number of classes observed up to and including the current incremental step. This quantity serves as a scale-invariant upper bound on the loss, and is subsequently used to normalize the expansion criterion described in the following section.

## 3.2. Expansion Mechanism

At the beginning of each incremental step, the model’s capacity to represent newly introduced classes is assessed before any adaptation takes place. Specifically, the average cross-entropy loss is computed over the training samples of the current step using the model trained at the previous step, as described in Algorithm 1. This value reflects the degree to which the current architecture can accommodate the new classes without modification.

Algorithm 1 Mean Loss Computation   
1: � ← 0; � ← 0   
2: for mini-batch (�, �) in the dataset do   
3: � ← �����(�)   
4: � ← ���������(�, �)   
5: � ← � + � ⋅ |�|; � ← � + |�|   
6: end for   
7: ���\_���� ← �∕�   
8: return ���\_����

To obtain a scale-invariant measure of representational saturation, this average loss is normalized by $\mathcal { L } _ { m a x }$ as defined in Equation 3, yielding a normalized loss $g$ bounded between 0 and 1, as described in Algorithm 2. When the average loss remains below a predefined threshold, the model is assumed to have suficient capacity to adapt to the new classes without architectural modification, and no expansion is performed. When expansion is required, it is carried out on selected lay-

Algorithm 2 Normalized Global Loss   
Require: ���\_����, $C _ { s e e n } ,$ ����\_�ℎ���ℎ���   
1: $L _ { \operatorname* { m a x } }  \ln ( | C _ { s e e n } | )$   
2: � ← ���\_����∕�<sub>max</sub>   
3: � ← min(�, 1)   
4: if ���\_���� ≤ ����\_�ℎ���ℎ��� then   
5: return   
6: end if   
7: return �

ers using a combination of the global dificulty signal � and local neuron utilization statistics. For each width-dynamic layer (WD) �, a local usage ratio � is computed, reflecting how actively the layer’s neurons have been engaged during previous training steps. The expansion coeficient is then defined as:

$$
c = g w _ { l o s s } + u * w _ { l o c a l }\tag{4}
$$

where $w _ { l o s s }$ and $w _ { l o c a l }$ are weighting factors that control the relative influence of the global loss signal and local utilization estimate, respectively. The number of neurons to be added is proportional to the layer size and is constrained within a bounded range to prevent uncontrolled growth, as detailed in Algorithm 3.

Algorithm 3 Expansion per Width Dynamic Layer   
Require: �, �����, $w _ { l o c a l } , w _ { l o s s } ,$ �����ℎ\_������   
1: for Dynamic Width Layer � in ����� do   
2: � ← �.�����.�����\_�����()   
3: $c \gets w _ { \mathrm { l o s s } } \cdot g + w _ { \mathrm { l o c a l } } \cdot u$   
4: � ← �.���\_�������� ⋅ �����ℎ\_������   
5: � ← �����(� ⋅ �)   
6: $m _ { \mathrm { m i n } }  ( 0 . 0 5 \cdot L$ .���\_�������� if $g > 0 . 5 ,$ else 0)   
7: $m _ { \mathrm { m a x } }  0 . 7 5$ ⋅ �.���\_��������   
8: � ← max(� , min(�, � ))   
9: if � > 0 then   
10: expand � with � neurons   
11: update connections with other layers   
12: update optimizer   
13: end if   
14: end for

This procedure ensures that the expansion process is task-agnostic, bounded, and driven by representational demand rather than any task identifier, making it directly applicable to the Class-IL setting.

## 3.3. Attention Mechanism

While width expansion increases the model’s representational capacity, it does not explicitly address interference that can arise between previously learned and newly introduced classes. To further stabilize representations during incremental updates, we incorporate a modified linear attention mechanism inspired by Katharopoulos, Vyas, Pappas and Fleuret (2020), augmented with a persistent key-value memory.

Let �, � and � denote the input query, key, and value sequences. These inputs are first projected into a shared attention space through learnable linear transformations:

$$
Q = \phi ( W _ { Q } q ) , K = \phi ( W _ { K } k ) , V = W _ { V } v\tag{5}
$$

where $W _ { Q } , W _ { K } ,$ and $W _ { V }$ are learnable projection matrices and �(⋅) is a positive feature map defined as:

$$
\phi ( x ) = E L U ( x ) + 1\tag{6}
$$

where ���(⋅) is an exponential linear unit function (Clevert, Unterthiner and Hochreiter, 2016), which enables the efficient linear attention formulation by ensuring non-negative kernel values.

![](images/a1040d263c8dcb2cd93b5fa8359df3414ee74a006b31e512dd411418711b8311.jpg)  
Figure 1: Attention Mechanism

To provide a stable reference across incremental learning steps, a persistent memory module is introduced, composed of learnable key-value pairs:

$$
k _ { m e m } \in \mathbb { R } ^ { M \times d } , v _ { m e m } \in \mathbb { R } ^ { M \times d }
$$

where � denotes the memory size and � the attention dimensionality. These parameters are initialized using orthogonal initialization to encourage diversity among stored representations. During the forward pass, the memory vectors are concatenated with the projected keys and values:

$$
K ^ { \prime } = [ K ; k _ { m e m } ] , V ^ { \prime } = [ V ; v _ { m e m } ]
$$

Given the augmented key-value set, the linear attention output is computed as:

$$
K V = K ^ { \prime T } V ^ { \prime }\tag{7}
$$

$$
Z = \frac { 1 } { Q ( K ^ { \prime T } ) }\tag{8}
$$

$$
A t t ( Q , K ^ { \prime } , V ^ { \prime } ) = \left( Q \cdot K V \right) \odot Z\tag{9}
$$

This formulation reduces the quadratic complexity of standard self-attention to linear complexity with respect to sequence length, making it more suitable for the incremental learning setting. Notably, the persistent memory allows the model to attend to stable key-value representations consolidated during previous training stages, thereby mitigating representational drift and helping to preserve previously acquired knowledge as new classes are introduced.

## 4. Experimental Setup

The experiments in this section are designed to answer three interconnected questions. First, is dynamic width expansion better than a fixed-capacity model retaining previously acquired knowledge in the Class-IL setting? Second, does the proposed attention mechanism with persistent key-value memory complement width expansion, or does it provide independent benefit? Third, how do these architectural mechanisms interact with established continual learning strategies - regularization, functional, and replaybased - across benchmarks of varying complexity?

To address these questions, we evaluate four architectural configurations - a fixed-capacity baseline, width expansion alone (WE), attention alone, and their combination (WE + Attention) - across two benchmarks: Split MNIST, a controlled setting using multilayer perceptrons, and Split CIFAR-100, a more demanding visual recognition scenario using a convolutional network. In both cases, the same continual learning strategies (EWC, SI, LwF, LwM, ER, A-GEM) are applied under identical hyperparameter conditions, enabling direct comparison. Lower and upper bounds are provided by sequential training without mitigation and joint training on all data simultaneously, respectively. This structure allows us to isolate the contribution of each architectural component and assess its combined efect across strategy families and task complexities.

## 4.1. MNIST Setup

The split MNIST benchmark (Hsu, Liu, Ramasamy and Kira, 2019; Van de Ven et al., 2022) provides a controlled incremental learning scenario in which the original MNIST dataset is partitioned into five sequential steps, each introducing two new classes. During training, the model receives data exclusively from the current step and must learn to recognize the newly introduced classes while maintaining performance on all previously observed ones. No task identity is provided during inference, following the standard Class-IL protocol.

All models are trained for five epochs per incremental step using the Adam optimizer with a learning rate of 0.001 and momentum parameters $\beta _ { 1 } = 0 . 9$ and $\beta _ { 2 } = 0 . 9 9 9$ . Training is performed with mini-batches of 128 samples, while evaluation uses batches of 64 samples. The optimization objective is the cross-entropy loss, computed over the set of classes observed up to the current incremental step. For models incorporating the attention mechanism, the attention dimensionality is set to $d = 1 2 8 .$ , and both the key and value memory matrices $k _ { m e m }$ and $v _ { m e m }$ contain 32 vectors each.

Two reference training regimes are considered to bound the expected performance range. The Lower Bound corresponds to sequential training without any forgetting mitigation strategy, representing the worst-case scenario. The Upper Bound corresponds to joint training, an idealized regime in which the model is trained simultaneously on data from all steps, representing the performance ceiling that continual learning methods seek to approximate.

To provide a comprehensive comparison, several widely used continual learning strategies are evaluated under identical experimental conditions. For EWC and SI, the regularization strength is set to $\lambda ~ = ~ 1 0 ^ { 9 }$ ; while no universally standard value exists for this hyperparameter, values determined empirically via log-scale grid search typically fall between $1 0 ^ { 3 }$ and $1 0 ^ { 1 } 5$ for both datasets in the Class-IL setting (Kruengkrai and Yamagishi, 2022; Van de Ven et al., 2022). For LwF, the distillation loss uses a temperature parameter $T \ = \ 2$ and a weighting coeficient $\beta ~ = ~ 1$ following the common formulation in literature. Replaybased methods maintain an episodic memory bufer of 1000 samples, storing up to 100 samples per class. In the case of A-GEM, the episodic memory also contains 100 samples per class, and reference gradients are computed using minibatches of 128 samples drawn from the replay bufer.

## 4.1.1. Model Architectures

Four model configurations are evaluated to isolate and analyze the individual and combined efects of width expansion and the proposed attention mechanism.

Baseline. The baseline architecture is a multilayer perceptron with two fully connected hidden layers, each containing 400 neurons, followed by ReLU activations. The network takes a flattened MNIST image as input and produces intermediate feature representations, which are forwarded to a shared incremental classifier that predicts all classes observed up to the current step. The incremental classifier dynamically expands its output dimensionality as new classes are introduced throughout the learning process. The architecture is illustrated in Figure 3.

Width Expansion (WE). The second configuration introduces the width expansion mechanism described in Section 3.2. The standard fully connected layers are replaced with width-dynamic (WD) layers, which are capable of increasing their number of neurons during training based on the normalized loss criterion and neuron utilization statistics. The network is initialized with the same configuration as the baseline model, containing two hidden layers of 400 neurons each. When the expansion criterion is triggered, new neurons are added to the hidden layers while preserving all previously learned weights and connections. The resulting architecture is illustrated in Figure 4.

MLP with Attention. The third configuration augments the baseline architecture with the proposed linear attention module. In this architecture, two fully connected layers are first used to compute intermediate feature representations $h _ { 1 }$ and $h _ { 2 } { \mathrm { : } }$

$$
\begin{array} { l } { h _ { 1 } = R e L U ( W _ { 1 } x ) } \\ { ~ } \\ { h _ { 2 } = R e L U ( W _ { 2 } h _ { 1 } ) } \end{array}
$$

The attention module then receives the second-layer representation as the query, while the first-layer representation

![](images/655428b948ebf2a7f7b6e825f026e926d8767a8e2513347c314ee599cf284585.jpg)  
Figure 2: Example of Split MNIST and classification along the three types of incremental learning.

![](images/a9980d89ae85312d73171bbc44d7ae700fddc1609455e8b4f84e756a94e33e7a.jpg)

Figure 3: MLP model for MNIST  
![](images/7912af6ee8a88eeab28c973c37a2e9f29ecc76bb3c0321aee74019a84087f5fd.jpg)  
Figure 4: WE model for MNIST

serves as both key and value:

$$
Q = h _ { 2 } , K = h _ { 1 } , V = h _ { 1 }
$$

Attention is computed as described in Section 3.3, producing an embedding of dimensionality � = 128, which is subsequently projected back into the feature space via an additional fully connected layer before being passed to the incremental classifier. The architecture is illustrated in Figure 5. In this formulation, the attention module enables deeper representations to selectively focus on earlier ones, integrating information from multiple representation levels before generating the final feature vector for classification.

Width Expansion with Attention. The fourth configuration combines the width expansion mechanism with the attention module described above. The standard fully connected layers used for feature extraction are replaced with width-dynamic layers, allowing the model to increase its representational capacity during incremental learning as described in Section 3.2. The attention module is applied in the same manner as in the previous configuration, with the second-layer serving as query and the first-layer representation as both key and value. The attention output is mapped back into the feature space through a linear projection before being forwarded to the incremental classifier. The architecture is illustrated in Figure 6.

This configuration is designed to combine two complementary mechanisms for continual learning: dynamic width expansion increases model capacity as new classes are introduced, while the attention module with persistent memory stabilizes representations across incremental steps, reducing interference between previously and newly learned classes.

## 4.2. CIFAR-100 Setup

The Split CIFAR-100 benchmark (Krizhevsky; Rebufi et al., 2017) is used to evaluate the proposed approach in a more challenging visual recognition scenario. The original CIFAR-100 dataset is partitioned into 10 sequential incremental steps, each introducing 10 new classes. As in the Split MNIST protocol, the model receives data exclusively from the current step during training and must maintain performance on all previously observed classes without access to task identity during inference.

All models are trained using the Adam optimizer with a learning rate of 0.001 and momentum parameters $\beta _ { 1 } = 0 . 9$ and $\beta _ { 2 } = 0 . 9 9 9$ . Training is performed using mini-batches of 256 samples, while evaluation uses mini-batches of 128 samples. Each incremental step is trained for 30 epochs, except for the upper-bound joint-training configuration, which is trained on the complete dataset for the same number of epochs. The optimization objective is the cross-entropy loss, computed over all classes observed up to the current incremental step. For models that incorporate the attention mechanism, the attention dimensionality is set to $d = 2 5 6 .$ and both the spatial and feature attention memory modules contain 64 key-value vectors each.

![](images/d4afac2729330e24ebd327574fb302b3c3df80f912ec29099895259d981a27f5.jpg)  
Figure 5: MLP model with attention mechanism for MNIST

![](images/a83ce03eeb2d9b6653235bfaedf6447c717d20276974611ab952ddb8c019983b.jpg)  
Figure 6: WE model with attention mechanism for MNIST

![](images/c577001531f92ace5249319a23abf129498c170ec86b604e059d807af5a7191f.jpg)  
Figure 7: Sample images from the CIFAR-100 dataset illustrating the diversity of object categories and visual appearances.

The same continual learning strategies evaluated on split MNIST are considered here under identical hyperparameter conditions: EWC and SI use a regularization coeficient � = $1 0 ^ { 9 }$ , LwF and LwM use distillation temperature � = 2, and weighting coeficient $\beta = 1$ . For replay-based methods, an episodic memory with 10,000 samples is maintained with a maximum of 100 samples per class. In the case of $\mathrm { A } \mathrm { - }$ GEM, the episodic memory also stores 100 samples per class, and reference gradients are computed using batches of 256 samples drawn from the replay bufer.

## 4.2.1. Model Architectures

Four model configurations are evaluated in the convolutional setting to assess the impact of width expansion and the attention mechanism under the greater visual complexity of CIFAR-100.

Baseline. The baseline architecture comprises a convolutional feature extractor, two fully connected layers, and a shared incremental classifier. The convolutional backbone consists of five blocks with 3 × 3 kernels, each followed by batch normalization and ReLU activation. Strided convolutions progressively reduce the spatial resolution from 32×32 to 4×4 while increasing the number of feature channels from 3 to 256 with the first layer mapping to 16 channels and each subsequent doubling that count. The resulting feature map is flattened and processed by two fully connected layers with 400 neurons each, producing the final representation used by the incremental classifier. The convolutional backbone is shared across all model configurations evaluated on CIFAR-100, and the full baseline architecture is illustrated in Figures 8 and 9.

Width Expansion (WE). The second configuration introduces the width expansion mechanism in the convolutional setting. The fully connected layers responsible for feature extraction are replaced by width-dynamic layers, which can increase their number of neurons during training based on the normalized loss and neuron utilization statistics. The model is initialized with the same configuration as the baseline, and new neurons are added to the hidden layers when the expansion criterion is triggered, while previously learned weights and connections are preserved. The resulting architecture is illustrated in Figure 10.

CNN with Attention. The third configuration augments the baseline convolutional architecture with two attention modules. The first operates over the spatial feature maps produced by the final convolutional layer. The feature tensor $\mathbf { \bar { \Psi } } _ { X } \in \mathbb { R } ^ { B \times C \times H \times W }$ is reshaped into a sequence of �⋅� spatial tokens of dimensionality �, over which linear self-attention is applied, allowing the model to capture interactions between spatial regions of the image. The resulting attention output is reshaped back to the original spatial format and combined with the input feature map through a residual connection, followed by batch normalization.

After flattening, two fully connected layers produce intermediate representations $h _ { 1 }$ and $h _ { 2 }$ . A second attention module is then applied in the feature space, following the same cross-level formulation used in the MNIST setting:

$$
Q = h _ { 2 } , K = h _ { 1 } , V = h _ { 1 }
$$

This mechanism allows the deeper representations to selectively attend to early feature representations while leveraging the persistent key-value memory described in Section 3.3. The attention output is projected back into the feature space through an additional linear layer before being forwarded to the incremental classifier. The full architecture is illustrated in Figure 11.

Width Expansion with Attention. The final configuration integrates both the width expansion mechanism and the dual attention modules described above. The fully connected layers used for feature extraction are replaced with widthdynamic layers, allowing the model to increase its representational capacity as new classes are introduced, while both the spatial and feature attention modules are retained. This configuration is designed to jointly exploit two complementary strategies: dynamic capacity growth, which provides additional representational resources when demanded, and attention-based stabilization, which mitigates representational drift and interference across incremental steps. The architecture is illustrated in Figure 12.

![](images/9625f518e5c430f099c6ae299c80aaa608ba4eadca5aa89c52fc31a1fa6460e5.jpg)

Figure 8: Common feature extractor for CIFAR-100 models  
![](images/ba0d2f63390d89b598519ebd026da406191cc30cba00062b3ac5831a41f552c6.jpg)  
Figure 9: CNN model for CIFAR-100

## 5. Results

The experimental results reveal clear and consistent patterns across benchmarks, learning strategies, and architectural configurations. Taken together, they highlight the complementary roles of width expansion and the attention mechanism in addressing catastrophic forgetting within the Class-IL setting.

## 5.1. Split MNIST

As shown in Table 1, the results on Split MNIST expose a sharp divide between the evaluated strategy families. Replay-based methods, particularly Exemplar Replay, achieve accuracy values close to the upper bound, substantially outperforming all other approaches. In contrast, regularization-based methods - EWC and SI - fail to provide meaningful mitigation of catastrophic forgetting, with results comparable to the lower bound regardless of the architectural configuration employed. This outcome is consistent with prior findings suggesting that parameter-based constraints are poorly suited to the Class-IL setting, where all classes share a common output space and the required plasticity tends to exceed what these methods allow.

![](images/a7b2ccf2f515f39f10fc3a19d9b7780e46f488343ae9a9b66779012003e3d3e9.jpg)  
Figure 10: CNN model for CIFAR-100 with width dynamic layers

![](images/9bc846c71b26a2294647f46442d107485857ce348f77b5bb5008d5d6c92d918a.jpg)

Figure 11: CNN model with attention mechanism for CIFAR-100  
![](images/e3acdd90bd1430f046c983ddbd3081ddd4160dbbfe24b4326fa27d52a3a2900d.jpg)  
Figure 12: CNNWE model with attention mechanism for CIFAR-100

Among functional regularization methods, LwF demonstrates a considerably more favorable response to the proposed architectural modifications. The combination of width expansion and attention achieves the highest accuracy within this group, reaching 44.04%, compared to 29.79% for the fixed MLP baseline. This improvement reflects the contribution of both mechanisms: width expansion provides additional representational capacity as new classes are introduced, while the attention module with persistent memory stabilizes feature representations across incremental steps, reducing the interference that distillation alone cannot fully prevent. Notably, the attention mechanism isolation introduces grater training variability, as evidenced by the higher standard deviation observed for the MLP with Attention configuration under LwF. When combined with width expansion, however, the model becomes more stable, suggesting that the two mechanisms interact constructively rather than independently.

A-GEM also benefits consistently from the proposed modifications. Accuracy increases from 28.34% for the fixed MLP to 41.25% for the width expansion with attention configuration, with width expansion contributing a meaningful share of this gain even before the attention module is incorporated. As illustrated in Figure 13, the improvements obtained by architectural enhancements are more pronounced for the LwF than for A-GEM. This asymmetry is interpretable: replay-based methods already mitigate forgetting through explicit sample reuse, and the marginal benefit of representational improvements is therefore smaller than for functional methods, which rely entirely on structural mechanisms to preserve prior knowledge.

Table 1
<table><tr><td rowspan=1 colspan=1>Strategy</td><td rowspan=1 colspan=1>MLP</td><td rowspan=1 colspan=1>WE</td><td rowspan=1 colspan=1> $\overline { M L P } + \mathsf { A t t e n t i o n }$ </td><td rowspan=1 colspan=1> $\overline { { \mathsf { W E } + \mathsf { A t t e n t i o n } } }$ </td></tr><tr><td rowspan=1 colspan=1>Joint</td><td rowspan=1 colspan=1> $9 7 . 7 6 \pm 0 . 1 2$ </td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>一</td></tr><tr><td rowspan=1 colspan=1>None</td><td rowspan=1 colspan=1> $\overline { { 1 9 . 6 2 \pm 0 . 0 7 } }$ </td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>EWC</td><td rowspan=1 colspan=1> $\overline { { 1 9 . 6 0 \pm 0 . 0 9 } }$ </td><td rowspan=1 colspan=1> $\overline { { 1 9 . 4 1 \pm 0 . 1 0 } }$ </td><td rowspan=1 colspan=1> $\overline { { 1 8 . 6 7 \ \pm 3 . 0 6 } }$ </td><td rowspan=1 colspan=1> $\overline { { 1 9 . 7 5 \pm 0 . 1 7 } }$ </td></tr><tr><td rowspan=1 colspan=1>SI</td><td rowspan=1 colspan=1> $\overline { { 1 9 . 6 3 \pm 0 . 0 5 } }$ </td><td rowspan=1 colspan=1> $\overline { { 1 9 . 3 9 \pm 0 . 1 1 } }$ </td><td rowspan=1 colspan=1> $\overline { { 1 9 . 0 7 \pm 2 . 5 9 } }$ </td><td rowspan=1 colspan=1> $\overline { { 1 9 . 6 8 \pm 0 . 0 4 } }$ </td></tr><tr><td rowspan=1 colspan=1>LwF</td><td rowspan=1 colspan=1> $\overline { { 2 9 . 7 9 \pm 0 . 8 5 } }$ </td><td rowspan=1 colspan=1> $2 9 . 2 8 \pm 2 . 0 0$ </td><td rowspan=1 colspan=1> $\overline { { 3 9 . 6 3 \pm 6 . 2 5 } }$ </td><td rowspan=1 colspan=1> $\overline { { 4 4 . 0 4 \pm 3 . 9 3 } }$ </td></tr><tr><td rowspan=1 colspan=1>ER</td><td rowspan=1 colspan=1> $\overline { { 8 8 . 8 0 \pm 0 . 7 3 } }$ </td><td rowspan=1 colspan=1> $\overline { { 8 3 . 4 4 \pm 0 . 9 3 } }$ </td><td rowspan=1 colspan=1> $\overline { { 5 6 . 7 4 \pm 3 3 . 1 2 } }$ </td><td rowspan=1 colspan=1> $\overline { { 8 7 . 7 4 \pm 0 . 6 0 } }$ </td></tr><tr><td rowspan=1 colspan=1>A-GEM</td><td rowspan=1 colspan=1> $2 8 . 3 4 \pm 8 . 5 4$ </td><td rowspan=1 colspan=1>33.97 ±8.55</td><td rowspan=1 colspan=1> $\overline { { 3 8 . 3 8 \pm 1 4 . 7 3 } }$ </td><td rowspan=1 colspan=1> $\overline { { 4 1 . 2 5 \pm 4 . 9 3 } }$ </td></tr></table>

Results for Split MNIST Values represent Mean ± Standard Deviation

![](images/55d47ecc56234629f7d80d793040b1335feac9d128c8ab790c23bf03237e2e04.jpg)  
Figure 13: Comparison between LwF and A-GEM on Split-Mnist

It is worth noting that ER exhibits an unexpected drop in performance under the MLP with attention configuration, with a large associated standard deviation. This instability likely to reflects the sensitivity of the attention module to training dynamics in the absence of additional capacity, and is substantially reduced when attention is combined with width expansion.

## 5.2. Split CIFAR-100 without Pretraining

As shown in Table 2, all methods experience a significant reduction in performance on Split CIFAR-100, reflecting the considerably greater visual complexity of this benchmark relative to Split MNIST. Regularization-based approaches again fail to provide meaningful improvements over the lower bound, and this pattern holds across all architectural configurations, reinforcing the conclusion that parameterbased constraints are insuficient to address the challenges of Class-IL in complex visual settings.

The proposed width expansion yields consistent improvements across all functional and replay-adjacent strategies. The efect is particularly pronounced for A-GEM, where accuracy increases from 11.72% to 17.91% with width expansion alone, and reaches 25.17% when the attention mechanism is added - the best result among all methods that not rely on direct sample replay. This progression suggests that the two mechanisms provide complementary benefits: expansion increases the representational resources available for new classes, while the attention module mitigates the drift in previously learned representations that expansion alone cannot prevent.

For LwF, width expansion also yields a consistent improvement, with accuracy increasing from 12.58% to 14.20% under expansion alone, and to 17.91% when attention is incorporated. As illustrated in Figure 14, the attention mechanism has a stronger impact when combined with width expansion than when applied in isolation, mirroring the pattern observed on Split MNIST and further supporting the interpretation that attention-based stabilization is most efective when suficient representational capacity is available.

Among replay-based methods, ER shows no meaningful benefit from width expansion or attention, with results remaining stable across all architectural configurations. This is consistent with the earlier observation that methods relying on stored samples are less sensitive to representational improvements, since explicit rehearsal already provides a strong mechanism for preserving prior knowledge.

## 5.3. Split CIFAR-100 with Pretraining

Table 3 reports results for the convolutional setting in which the backbone is initialized with weights pretrained on CIFAR-10 and subsequently frozen during incremental training. Although pretraining might intuitively be expected to provide a stronger representational foundation, the results show that models trained from scratch outperform their pretrained counterparts across nearly all strategies and configurations.

<table><tr><td rowspan=1 colspan=1>Strategy</td><td rowspan=1 colspan=1>CNN</td><td rowspan=1 colspan=1>WE</td><td rowspan=1 colspan=1> $\overline { { \mathsf { C N N } + \mathsf { A t t e n t i o n } } }$ </td><td rowspan=1 colspan=1> $\overline { { \mathsf { W E } + \mathsf { A t t e n t i o n } } }$ </td></tr><tr><td rowspan=1 colspan=1>Joint</td><td rowspan=1 colspan=1> $\overline { { 4 8 . 5 6 \pm 0 . 5 1 } }$ </td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>1</td></tr><tr><td rowspan=1 colspan=1>None</td><td rowspan=1 colspan=1> $\overline { { 8 . 0 5 \pm 0 . 1 7 } }$ </td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>EWC</td><td rowspan=1 colspan=1> $\overline { { 6 . 4 6 \pm 0 . 3 4 } }$ </td><td rowspan=1 colspan=1> $\overline { { 7 . 2 7 \pm 0 . 1 1 } }$ </td><td rowspan=1 colspan=1> $\overline { { 6 . 4 9 \pm 0 . 3 0 } }$ </td><td rowspan=1 colspan=1> $\overline { { 7 . 5 3 \pm 0 . 1 1 } }$ </td></tr><tr><td rowspan=1 colspan=1>SI</td><td rowspan=1 colspan=1> $\overline { { 5 . 8 2 \pm 0 . 1 6 } }$ </td><td rowspan=1 colspan=1> $\overline { { 8 . 4 3 \ \pm 0 . 0 8 } }$ </td><td rowspan=1 colspan=1> $\overline { { 5 . 8 9 \pm 0 . 1 7 } }$ </td><td rowspan=1 colspan=1> $\overline { { 8 . 3 0 \pm 0 . 0 9 } }$ </td></tr><tr><td rowspan=1 colspan=1>LwF</td><td rowspan=1 colspan=1> $1 2 . 5 8 \pm 0 . 3 0$ </td><td rowspan=1 colspan=1> $\overline { { 1 4 . 2 0 \pm 0 . 3 4 } }$ </td><td rowspan=1 colspan=1> $\overline { { 1 5 . 1 3 \pm 0 . 5 6 } }$ </td><td rowspan=1 colspan=1> $\overline { { 1 7 . 9 1 \pm 0 . 9 0 } }$ </td></tr><tr><td rowspan=1 colspan=1>LwM</td><td rowspan=1 colspan=1> $\overline { { 1 0 . 6 4 \pm 0 . 2 6 } }$ </td><td rowspan=1 colspan=1> $1 2 . 5 2 \pm 0 . 3 8$ </td><td rowspan=1 colspan=1> $\overline { { 1 0 . 3 9 \pm 0 . 8 5 } }$ </td><td rowspan=1 colspan=1> $\overline { { 1 2 . 1 8 \pm 0 . 8 6 } }$ </td></tr><tr><td rowspan=1 colspan=1>ER</td><td rowspan=1 colspan=1> $\overline { { 3 3 . 3 7 \ \pm 0 . 9 4 } }$ </td><td rowspan=1 colspan=1> $\overline { { 3 1 . 2 1 \pm 0 . 6 2 } }$ </td><td rowspan=1 colspan=1> $\overline { { 3 1 . 7 2 \pm 0 . 8 8 } }$ </td><td rowspan=1 colspan=1> $\overline { { 3 2 . 4 1 \pm 0 . 6 3 } }$ </td></tr><tr><td rowspan=1 colspan=1>A-GEM</td><td rowspan=1 colspan=1> $\overline { { 1 1 . 7 2 \pm 3 . 3 8 } }$ </td><td rowspan=1 colspan=1> $\overline { { 1 7 . 9 1 \pm 3 . 8 4 } }$ </td><td rowspan=1 colspan=1> $\overline { { 1 4 . 3 9 \pm 3 . 5 5 } }$ </td><td rowspan=1 colspan=1> $2 5 . 1 7 \pm 6 . 5 0$ </td></tr></table>

Table 2

Results for Split CIFAR-100 without pretraining Values represent Mean ± Standard Deviation
<table><tr><td rowspan=1 colspan=1>Strategy</td><td rowspan=1 colspan=1>CNN</td><td rowspan=1 colspan=1>WE</td><td rowspan=1 colspan=1> $\mathsf { C N N } + \mathsf { A t t e n t i o n }$ </td><td rowspan=1 colspan=1>WE + Attention</td></tr><tr><td rowspan=1 colspan=1>Joint</td><td rowspan=1 colspan=1> $\overline { { 4 4 . 0 9 \pm 0 . 4 0 } }$ </td><td rowspan=1 colspan=1>-</td><td rowspan=1 colspan=1>-</td><td rowspan=1 colspan=1>-</td></tr><tr><td rowspan=1 colspan=1>None</td><td rowspan=1 colspan=1> $\overline { { 8 . 0 4 \pm 0 . 0 6 } }$ </td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>EWC</td><td rowspan=1 colspan=1> $7 . 5 8 \pm 0 . 2 4$ </td><td rowspan=1 colspan=1> $\overline { { 7 . 6 4 \pm 0 . 1 7 } }$ </td><td rowspan=1 colspan=1> $\overline { { 7 . 8 3 \pm 0 . 1 4 } }$ </td><td rowspan=1 colspan=1>7.86 ±0.15</td></tr><tr><td rowspan=1 colspan=1>SI</td><td rowspan=1 colspan=1> $\overline { { 7 . 3 5 \pm 0 . 1 1 } }$ </td><td rowspan=1 colspan=1> $7 . 3 8 \pm 0 . 0 6$ </td><td rowspan=1 colspan=1>5.61 ±0.21</td><td rowspan=1 colspan=1>5.65 ±0.22</td></tr><tr><td rowspan=1 colspan=1>LwF</td><td rowspan=1 colspan=1> $\overline { { 1 4 . 1 9 \pm 0 . 3 2 } }$ </td><td rowspan=1 colspan=1> $\overline { { 1 4 . 4 3 \pm 0 . 4 0 } }$ </td><td rowspan=1 colspan=1>16.13 ±0.59</td><td rowspan=1 colspan=1> $\overline { { 1 6 . 4 1 \pm 0 . 6 9 } }$ </td></tr><tr><td rowspan=1 colspan=1>ER</td><td rowspan=1 colspan=1> $3 1 . 7 2 \pm 0 . 2 6$ </td><td rowspan=1 colspan=1> $3 1 . 6 0 \pm 0 . 5 6$ </td><td rowspan=1 colspan=1>33.38 ±0.57</td><td rowspan=1 colspan=1>33.28 ±0.38</td></tr><tr><td rowspan=1 colspan=1>A-GEM</td><td rowspan=1 colspan=1> $\overline { { 8 . 1 0 \pm 0 . 1 4 } }$ </td><td rowspan=1 colspan=1> $\overline { { 8 . 1 5 \ \pm 0 . 1 0 } }$ </td><td rowspan=1 colspan=1> $\overline { { 1 3 . 2 3 \pm 1 . 6 7 } }$ </td><td rowspan=1 colspan=1> $1 5 . 3 7 \pm 4 . 9 4$ </td></tr></table>

Table 3  
Results for Split CIFAR-100 with pretraining values represent Mean ± Standard Deviation

This counterintuitive outcome can be attributed to the representational mismatch between CIFAR-10 and CIFAR-100. Because the convolutional layers are frozen after pretraining, the model is constrained to a fixed feature space that was optimized for a simpler and categorically diferent distribution. The resulting representations lack the expressiveness required for the finer-grained distinctions among CIFAR-100 classes, forcing the classifier and the proposed expansion mechanisms to compensate for suboptimal features - a task that fundamentally limits achievable performance.

Pretraining does, however, reduce training variability: pretrained models exhibit smaller standard deviations across runs, acting as a strong implicit regularizer. This stability comes at a clear cost to plasticity, as frozen convolutional layers cannot adapt their representations to the new data distribution encountered during incremental learning. The proposed width expansion, which operates exclusively on the fully connected layers in this configuration, can only partially compensate for the rigidity of the fixed feature extractor.

As illustrated in Figures 15 and 16, pretrained models consistently underperform their non-pretrained counterparts across both LwF and A-GEM, with the gap particularly evident in the A-GEM setting. This pattern reinforces a central conclusion of the study: in the Class-IL setting, the ability to adapt feature representations incrementally is more important than the quality of the initial representation space. A well-initialized but fixed feature extractor introduces constraints that ultimately hinder the kind of flexible adaptation that continual learning demands.

## 6. Threats to Validity

A critical assessment of this study’s methodology and results reveals several potential threats to the validity of its conclusions. Acknowledging these limitations is essential for contextualizing the findings and guiding future research.

## 6.1. Internal Validity

Internal validity concerns whether the observed improvements can be attributed to the proposed mechanisms rather than confounding factors. Several design choices support this claim. Hyperparameters for the baseline strategies $\left( \lambda = 1 0 ^ { 9 } \right.$ for EWC and SI, � = 2 and $\beta = 1$ for LwF and LwM, bufer sizes of 100 samples per class) were set following established practice in the Class-IL literature (Van de Ven et al., 2022; Kruengkrai and Yamagishi, 2022) and held constant across all architectural configurations, ensuring that strategy-level diferences do not inflate observed gains. All results are reported as means and standard deviations across multiple independent runs with diferent random seeds, providing a measure of result stability. However, the expansionspecific hyperparameters - including loss threshold, growth factor, and weighting coeficients $w _ { l o s s }$ and $w _ { l o c a l }$ - were not subjected to systematic held-out grid search, and their interaction with specific strategy-architecture combinations cannot be fully ruled out as a source of variance. The high standard deviation observed for A-GEM under some configurations (e.g., 14.73 on Split MNIST under MLP + Attention) suggests that certain combinations are sensitive to initialization or training dynamics in ways not fully captured by the reported statistics.

![](images/e4f1db3fe6fdb98f827afb73c0710d83cd46b1135ee374ac78f1a124cf769ad3.jpg)  
Figure 14: Comparison between LwF and A-GEM on Split CIFAR-100

![](images/3d914074aba80fcab02ed9980a4a73aa3e3fd3c4f4c8f14b586eddcdc5b83dd8.jpg)  
Figure 15: LwF comparison between with and without pretraining.

## 6.2. External Validity

External validity concerns the generalization of findings beyond the evaluated conditions. Two benchmarks of substantially diferent complexity were used: Split MNIST, a controlled setting, and Split CIFAR-100, a more demanding visual recognition task. This range provides meaningful coverage of the Class-IL landscape. Nevertheless, both benchmarks use balanced class splits of equal and fixed size, a condition that may not hold in real-world deployments, where class arrival frequency, distributional shift, and task duration may vary unpredictably. The convolutional backbone used for CIFAR-100 is relatively shallow compared to architectures used in recent state-of-the-art work; performance on large-scale benchmarks such as Split ImageNet, or in settings based on pre-trained transformer backbones, cannot be directly inferred from the present results. Additionally, all evaluations assume a fixed incremental schedule with clearly delineated steps; behavior of the expansion mechanism under irregular, fine-grained, or blurred task boundaries remains an open question.

## 6.3. Construct Validity

Construct validity concerns whether the evaluation metrics adequately capture the properties of interest. Average classification across all observed classes at the end of training is used as the primary metric, consistent with the standard Class-IL evaluation protocol (Van de Ven et al., 2022). This metric, however, does not distinguish between forgetting of early classes and failure to learn later ones - two failure modes with distinct implications for the stabilityplasticity trade-of. Furthermore, it does not capture the computational cost incurred by dynamic expansion, which is a practically relevant consideration in resource-constrained deployment scenarios. Future evaluations should complement accuracy with backward transfer, forward transfer, and parameter count over time to provide a more complete characterization of each mechanism’s contribution.

## 7. Conclusion

This work investigated catastrophic forgetting in Class-IL through the lens of architectural adaptability, proposing a method based on dynamic width expansion within existing network layers, complemented by a linear attention mechanism with a persistent key-value memory. Unlike network expansion approaches such as DER and DNE, which grow model capacity through the addition of task-specific modules and therefore depend on explicit task identifiers, the proposed method expands representational capacity within existing layers based on a normalized cross-entropy criterion that reflects representational demand. This design makes the approach directly applicable to the Class-IL setting, where task boundaries are not available during inference.

![](images/4513abb2cecc8a4783b7df2981d471b8cac7913e4f9ce73f6a2f0b48d9b4777d.jpg)  
Figure 16: A-GEM comparison between with and without pretraining.

Experimental results across Split MNIST and Split CIFAR-100 demonstrate that the combination of width expansion and the attention mechanism provides consistent improvements in knowledge retention. On Split MNIST, the width expansion with the attention module configuration achieves the highest accuracy among functional regularization methods, reaching 44.04%, while on Split CIFAR-100, the same configuration yields 25.17% - the best result among methods that do not rely on direct sample replay. Across both benchmarks, the attention mechanism with persistent key-value memory plays a central role in stabilizing feature representations and mitigating representational drift during incremental updates. The results also reveal that the two mechanisms interact constructively: attention-based stabilization is most efective when combined with suficient representational capacity, and width expansion alone yields more variable results in the absence of the representational anchoring provided by the persistent memory.

A recurring and significant finding across all experiments is that adaptability in feature extraction layers proves more consequential than the quality of the initial representation space. Models trained from scratch consistently outperformed those with pretrained and frozen convolutional backbones, as the latter introduce representational constraints that limit the plasticity required to accommodate new data distributions over time. This result suggests that, in complex continual learning scenarios, the capacity to revise learned representations incrementally should be prioritized over initialization-based stability.

Several directions emerge from the present work. First, the expansion mechanism currently operates on fully connected layers; extending it to convolutional layers could allow the model to adapt its feature extraction capacity directly, rather than relying solely on downstream dense layers to compensate for representational bottlenecks. Second, selective post-expansion pruning could recover computational eficiency without sacrificing accuracy, addressing the risk of unbounded parameter growth observed in the current results. Third, the normalized global loss criterion used to trigger expansion could be replaced by layer-wise or neuronwise importance signals, enabling finer-grained allocation of representational resources across the network. Finally, evaluation on larger-scale and class-imbalanced benchmarks, as well as in settings with irregular overlapping task boundaries, would provide a more complete picture of the approach’s applicability to real-world continual learning scenarios.

## CRediT authorship contribution statement

André L. S. Conde: Writing – original draft, Visualization, Methodology, Investigation, Software, Formal analysis, Conceptualization. Yehia Elkhatib: review & editing, Investigation, Validation. Cateano M. Ranieri: review & editing, Investigation, Project administration, Supervision, Funding acquisition.

## Acknowledgements

This work was partially supported by the São Paulo Research Foundation (FAPESP), grant 2025/13241-3. The Article Processing Charge (APC) was funded by the Brazilian Federal Agency for Support and Evaluation of Graduate Education – CAPES (ROR identifier: 00x0ma614). For the purposes of open access, the authors have applied a Creative Commons CC BY license to any accepted version of the article.

## References

Cao, L., 2022. Beyond iid: Non-iid thinking, informatics, and learning. IEEE Intelligent Systems 37, 5–17.

Chaudhry, A., Ranzato, M., Rohrbach, M., Elhoseiny, M., 2019. Eficient lifelong learning with a-GEM, in: International Conference on Learning Representations. URL: https://openreview.net/forum?id=Hkf2\_sC5FX.

Clevert, D.A., Unterthiner, T., Hochreiter, S., 2016. Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs). URL: http://arxiv.org/abs/1511.07289, doi:10.48550/arXiv.1511.07289. arXiv:1511.07289 [cs].

De Lange, M., Tuytelaars, T., 2021. Continual Prototype Evolution: Learning Online from Non-Stationary Data Streams, in: 2021 IEEE/CVF International Conference on Computer Vision (ICCV), IEEE, Montreal, QC, Canada. pp. 8230–8239. URL: https://ieeexplore.ieee.org/ document/9711397/, doi:10.1109/ICCV48922.2021.00814.

Dhar, P., Singh, R.V., Peng, K.C., Wu, Z., Chellappa, R., 2019. Learning without memorizing, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 5138–5146.

Dong, M., Zhang, Z., Tan, X., Qiu, J., Xie, Y., 2026. Dynamic expansion orthogonal network for class-incremental learning. Knowledge-Based Systems 341, 115868. URL: https://linkinghub.elsevier.com/ retrieve/pii/S0950705126005940, doi:10.1016/j.knosys.2026.115868.

Fu, Z., Wang, Z., Xu, X., Li, D., Yang, H., 2023. Knowledge aggregation networks for class incremental learning. Pattern Recognition 137, 109310. URL: https://linkinghub.elsevier.com/retrieve/pii/ S0031320323000110, doi:10.1016/j.patcog.2023.109310.

Hadsell, R., Rao, D., Rusu, A.A., Pascanu, R., 2020. Embracing change: Continual learning in deep neural networks. Trends in cognitive sciences 24, 1028–1040.

Hsu, Y.C., Liu, Y.C., Ramasamy, A., Kira, Z., 2019. Re-evaluating Continual Learning Scenarios: A Categorization and Case for Strong Baselines. URL: http://arxiv.org/abs/1810.12488, doi:10.48550/arXiv.1810.12488. arXiv:1810.12488 [cs].

Hu, Z., Li, Y., Lyu, J., Gao, D., Vasconcelos, N., 2023. Dense network expansion for class incremental learning, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 11858–11867.

Hussain, A., Ullah, W., Khan, N., Khan, Z.A., Yar, H., Baik, S.W., 2026. Class-incremental learning network for real-time anomaly recognition in surveillance environments. Pattern Recognition 170, 112064. URL: https://linkinghub.elsevier.com/retrieve/pii/ S0031320325007241, doi:10.1016/j.patcog.2025.112064.

Jiang, K., Bai, X., Zhou, F., 2026. Recurrent Network Expansion for Class Incremental Learning. IEEE Transactions on Neural Networks and Learning Systems 37, 122–135. URL: https://ieeexplore.ieee.org/ document/11142762/, doi:10.1109/TNNLS.2025.3601373.

Katharopoulos, A., Vyas, A., Pappas, N., Fleuret, F., 2020. Transformers are RNNs: Fast autoregressive transformers with linear attention, in: III, H.D., Singh, A. (Eds.), Proceedings of the 37th International Conference on Machine Learning, PMLR. pp. 5156–5165. URL: https: //proceedings.mlr.press/v119/katharopoulos20a.html.

Kim, D., Han, B., 2023. On the stability-plasticity dilemma of classincremental learning, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 20196–20204.

Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins, G., Rusu, A.A., Milan, K., Quan, J., Ramalho, T., Grabska-Barwinska, A., Hassabis, D., Clopath, C., Kumaran, D., Hadsell, R., 2017. Overcoming catastrophic forgetting in neural networks. Proceedings of the National Academy of Sciences 114, 3521–3526. URL: https://www.pnas. org/doi/abs/10.1073/pnas.1611835114, doi:10.1073/pnas.1611835114, arXiv:https://www.pnas.org/doi/pdf/10.1073/pnas.1611835114.

Krizhevsky, A., . Learning Multiple Layers of Features from Tiny Images . Kruengkrai, C., Yamagishi, J., 2022. Mitigating the Diminishing Efect of Elastic Weight Consolidation, in: Calzolari, N., Huang, C.R., Kim, H., Pustejovsky, J., Wanner, L., Choi, K.S., Ryu, P.M., Chen, H.H., Donatelli, L., Ji, H., Kurohashi, S., Paggio, P., Xue, N., Kim, S., Hahm, Y., He, Z., Lee, T.K., Santus, E., Bond, F., Na, S.H. (Eds.), Proceedings of the 29th International Conference on Computational Linguistics,

International Committee on Computational Linguistics, Gyeongju, Republic of Korea. pp. 4568–4574. URL: https://aclanthology.org/2022. coling-1.403/.

Li, Y., Dou, H., Li, G., Gao, G., Zhou, H., 2026. INSERTION: From traditional incremental learning to open-world stream learning. Pattern Recognition 176, 113163. URL: https://linkinghub.elsevier.com/ retrieve/pii/S0031320326001287, doi:10.1016/j.patcog.2026.113163.

Li, Z., Hoiem, D., 2017. Learning without forgetting. IEEE transactions on pattern analysis and machine intelligence 40, 2935–2947.

Liu, H., Zhou, Y., Liu, B., Zhao, J., Yao, R., Shao, Z., 2023. Incremental learning with neural networks for computer vision: a survey. Artificial Intelligence Review 56, 4557–4589. URL: https://doi.org/10.1007/ s10462-022-10294-2, doi:10.1007/s10462-022-10294-2.

Rebufi, S.A., Kolesnikov, A., Sperl, G., Lampert, C.H., 2017. icarl: Incremental classifier and representation learning, in: Proceedings of the IEEE conference on Computer Vision and Pattern Recognition, pp. 2001–2010.

Rolnick, D., Ahuja, A., Schwarz, J., Lillicrap, T., Wayne, G., 2019. Experience replay for continual learning, in: Wallach, H., Larochelle, H., Beygelzimer, A., d'Alché-Buc, F., Fox, E., Garnett, R. (Eds.), Advances in Neural Information Processing Systems, Curran Associates, Inc. URL: https://proceedings.neurips.cc/paper\_files/paper/2019/ file/fa7cdfad1a5aaf8370ebeda47a1ff1c3-Paper.pdf.

Van de Ven, G.M., Tuytelaars, T., Tolias, A.S., 2022. Three types of incremental learning. Nature Machine Intelligence 4, 1185–1197.

Yan, S., Xie, J., He, X., 2021. Der: Dynamically expandable representation for class incremental learning, in: 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 3013–3022. doi:10.1109/CVPR46437.2021.00303.

Yu, L., Twardowski, B., Liu, X., Herranz, L., Wang, K., Cheng, Y., Jui, S., Van De Weijer, J., 2020. Semantic Drift Compensation for Class-Incremental Learning, in: 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), IEEE, Seattle, WA, USA. pp. 6980–6989. URL: https://ieeexplore.ieee.org/document/9156964/, doi:10.1109/CVPR42600.2020.00701.

Zenke, F., Poole, B., Ganguli, S., 2017. Continual learning through synaptic intelligence. Proc Mach Learn Res 70, 3987–3995.

Zhao, B., Xiao, X., Gan, G., Zhang, B., Xia, S.T., 2020. Maintaining Discrimination and Fairness in Class Incremental Learning, in: 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 13205–13214. URL: https://ieeexplore.ieee. org/document/9156766/, doi:10.1109/CVPR42600.2020.01322. iSSN: 2575- 7075.

Zhou, D.W., Sun, H.L., Ye, H.J., Zhan, D.C., 2024a. Expandable Subspace Ensemble for Pre-Trained Model-Based Class-Incremental Learning, in: 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), IEEE, Seattle, WA, USA. pp. 23554– 23564. URL: https://ieeexplore.ieee.org/document/10656913/, doi:10. 1109/CVPR52733.2024.02223.

Zhou, D.W., Wang, F.Y., Ye, H.J., Ma, L., Pu, S., Zhan, D.C., 2022. Forward Compatible Few-Shot Class-Incremental Learning, in: 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), IEEE, New Orleans, LA, USA. pp. 9036–9046. URL: https: //ieeexplore.ieee.org/document/9878986/, doi:10.1109/CVPR52688.2022. 00884.

Zhou, D.W., Wang, Q.W., Qi, Z.H., Ye, H.J., Zhan, D.C., Liu, Z., 2024b. Class-incremental learning: A survey. IEEE Transactions on Pattern Analysis and Machine Intelligence 46, 9851–9873. doi:10.1109/TPAMI. 2024.3429383.