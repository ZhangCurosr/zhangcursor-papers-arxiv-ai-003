# Prototype-guided Bilateral Alignment Multimodal Federated Learning

Tianchi Liao <sup>1</sup> <sup>2</sup> Lele Fu <sup>2</sup> Sheng Huang <sup>2</sup> Qing Hu <sup>2</sup> Hong-Ning Dai <sup>1</sup> Chuan Chen <sup>2</sup>

## Abstract

Multimodal federated learning (MFL) has emerged as a pivotal paradigm for leveraging distributed data to enhance model performance. However, existing methods predominantly rely on idealized assumptions of model homogeneity and balanced modality distributions, rendering them ill-suited for practical scenarios characterized by heterogeneous client architectures and severe modality imbalance. To address these challenges, we propose a Multimodal Federated learning Prototype-guided Bilateral Alignment (MFedPBA) framework. MFedPBA facilitates robust knowledge synergy through a dual alignment mechanism: (i) at the feature level, it aligns heterogeneous feature spaces via a projection encoder optimized by contrastive learning and the Gromov-Wasserstein distance; (ii) at the decision level, it employs an entropy-weighted aggregation of naturally aligned logit prototypes. This novel design achieves robust MFL by jointly tackling heterogeneous feature spaces and collectively aggregating decisions. Extensive experiments demonstrate that our method significantly outperforms state-of-the-art baselines under conditions of model heterogeneity and modality imbalance.

## 1. Introduction

With the rapid proliferation of multi-sensor Internet of Things devices and the growing demand for data privacy, personalized federated learning (FL) has emerged as an important paradigm for training customized local models on decentralized clients (Collins et al., 2021; Meng et al., 2024; Kou et al., 2026; Ma et al.). Recently, the integration of heterogeneous sensory data has further advanced multimodal federated learning (MFL), which aims to exploit complementary information across modalities such as vision, audio, and text to improve model robustness and personalization (Qi et al., 2023; Che et al., 2023; Huang et al., 2024; 2026). However, the majority of existing multimodal federated learning frameworks are predicated on the idealized assumption of model homogeneity, mandating that all participating clients deploy an identical neural network architecture (Hu et al., 2024b; Liao et al., 2024; Cao et al., 2026). In practice, as illustrated in Figure 1(a), this assumption rarely holds true in real-world deployments. Given that practical clients typically exhibit profound heterogeneity in both hardware resource constraints (e.g., computational and storage capacities) and multimodal data distributions (such as missing modalities or data skewness), a single global model struggles to simultaneously accommodate the diverse conditions of all nodes. Consequently, designing and deploying personalized, heterogeneous model architectures tailored to different clients has emerged as an inevitable trend (Chen & Zhang, 2024; Rahman & Nguyen, 2024).

To address model heterogeneity, prior work has explored prototype-based federated learning, where class embeddings are exchanged as communication units (Tan et al., 2022a; Huang et al., 2023; Li et al., 2025). Nevertheless, such approaches encounter severe bottlenecks in multimodal settings. Since heterogeneous encoders map multimodal data into disparate embedding spaces, the direct aggregation of prototypes on the server results in significant representation misalignment (Fu et al., 2025; Zhang et al., 2025c). As illustrated in Figure 1(b), when clients employ different encoders, prototypes corresponding to the same class may reside in incompatible feature spaces. Consequently, forced aggregation causes the global prototype to be close to class decision boundaries. Instead of yielding performance gains, such forced aggregation can lead to significant knowledge contamination. This issue often causes the global model to degrade to the point where its performance falls below that of a baseline trained purely on local data (Li et al., 2024; Chen et al., 2025; Xiao et al., 2026; Hu et al., 2024a). This raises a critical research challenge: How can we overcome feature space incompatibility to facilitate effective knowledge transfer when both client neural architectures and input modality configurations are inconsistent?

Furthermore, existing MFL research typically operates under the assumption of balanced modality distributions.

![](images/ef64add927522889c9a1291199f4c03cddbb9d2a10cfff6539702d60e570f045.jpg)  
Figure 1. (a) Illustration of the multimodal federated learning setting with complex architectural and data inconsistencies. (b) Visualization of feature space incompatibility across heterogeneous encoders, where direct prototype aggregation induces drift towards ambiguous regions, resulting in performance inferior to local training.

While some methods consider scenarios with missing modalities, they are often restricted to bimodal settings or assume that all clients share an identical set of modalities (Che et al., 2024; Wang et al., 2024; Pan et al., 2025; Yu et al., 2025). Such assumptions fail to capture the prevalent imbalance in modality availability across decentralized clients. In real-world applications, systems commonly involve multiple modalities, and certain clients may entirely lack data for specific modalities. This imbalance substantially increases the difficulty of collaborative learning by introducing a higher degree of information heterogeneity (Qi et al., 2025; He et al., 2025). This leads to another key challenge: How can we design an alignment mechanism that ensures precise collaboration in complex scenarios characterized by unbalanced data distributions and modality quantities exceeding the bimodal limit?

To address these challenges, we propose a multimodal federated learning prototype-guided bilateral alignment framework (MFedPBA). The framework implements a bilateral alignment mechanism on the server that integrates knowledge from heterogeneous clients through modality-specific feature prototypes and logit prototypes. ❶ At the feature level, the server constructs an autoencoding alignment framework. It leverages contrastive learning to enhance intra-modal semantic consistency, while employing the Gromov–Wasserstein (GW) distance to align the geometric structures between heterogeneous feature spaces and the global space, enabling effective aggregation within a unified representation space. ❷ At the decision level, unlike feature prototypes that may reside in incompatible spaces, logits naturally reside in a shared semantic space aligned across heterogeneous models, as each dimension directly corresponds to a specific class. Leveraging this property, we employ an entropy-based weighting strategy during aggregation. This mechanism quantifies the reliability of local predictions to construct highly discriminative global logit prototypes. By synergizing feature-level and decision-level insights, our framework facilitates robust and efficient crossclient knowledge transfer while strictly preserving the architectural heterogeneity of local models.

The primary contributions of this work are summarized as follows:

• New perspective: We address a realistic MFL setting characterized by model heterogeneity and modality imbalance, transcending conventional homogeneous assumptions to enable robust knowledge transfer.

• Effective algorithm: We propose MFedPBA, a bilateral alignment framework that achieves dual-level synergy: feature alignment via GW distance and contrastive learning, and decision alignment via entropybased logit aggregation. We further provide theoretical convergence guaranties in non-convex settings.

• Superior performance: Extensive experiments on multiple multimodal benchmarks demonstrate that MFedPBA consistently outperforms state-of-the-art baselines under diverse modality distribution settings.

## 2. Related Work

## 2.1. Multimodal Federated Learning

Multimodal federated learning (Feng et al., 2023; Ouyang et al., 2023; Pan et al., 2024) has emerged as a pivotal paradigm for addressing privacy concerns in mobile sensing systems, finding broad application in real-world scenarios (Zhao et al., 2022; Zheng et al., 2023; Thrasher et al., 2025). Existing MFL literature predominantly addresses two settings: (1) unimodal clients, where each client possesses only a single modality (Zhang et al., 2025b; Deng et al., 2025), and (2) missing modalities, where clients hold incomplete subsets of views (Che et al., 2024; Liu et al., 2025). While these approaches aim to aggregate effective global models, real-world deployments often necessitate distinct feature extractors tailored to specific modalities (Chen et al., 2024). To handle such heterogeneity, recent work like FedM-Bridge (Chen & Zhang, 2024) proposes a topology-aware hypernetwork. However, this method incurs prohibitive communication overhead, which scales poorly with increasing model complexity and modality counts. Consequently, existing methods fail to adequately address the compounding challenges of inter-node modal imbalance and model heterogeneity.

However, such methods incur prohibitive communication overhead and suffer from severe scalability bottlenecks as model complexity and the number of modalities increase. In contrast, our work proposes a communication-efficient heterogeneous federated learning framework via multimodal joint alignment.

## 2.2. Federated Prototype Learning

Federated prototype learning has been extensively investigated across diverse tasks, wherein a prototype is defined as the mean feature vector of instances belonging to a specific class (Huang et al., 2022; Qi et al., 2025; Liao et al., 2025). Within the federated learning literature, prototypes serve as a vital mechanism to abstract knowledge while strictly preserving data privacy (Tan et al., 2022b; Zhao et al., 2024). Due to their robust representational capacity, prototypes are widely adopted to enforce local regularization and enhance communication efficiency (Huang et al., 2023; Yi et al., 2023). To mitigate the feature drift issue in FL, (Fu et al., 2025) proposed a federated domain-independent prototype learning approach, which achieves representation and parameter space alignment under feature shifts. Furthermore, FedPall (Zhang et al., 2025c) leverages prototype-based adversarial learning to unify the feature space and employs collaborative learning to reinforce class-specific information within the features. Conventionally, local prototypes are simply aggregated into a global prototype via weighted averaging on the server, yielding sub-optimal global knowledge that negatively impacts client performance. To address this limitation, FedTGP (Zhang et al., 2024) employs adaptive margin-enhanced contrastive learning to optimize trainable global prototypes at the server level.

However, the feature misalignment arising from multimodal heterogeneity is significantly more severe than standard domain shifts. Consequently, naive aggregation fails to bridge the divergent feature spaces generated by heterogeneous local models. To address this challenge, we propose a learnable prototype framework on the server. By distilling knowledge from client-side prototypes, our approach effectively aligns the heterogeneous feature spaces.

## 3. Preliminary

## 3.1. General multimodal FL Framework

Consider a heterogeneous MFL problem. The system consists of a central server and K clients. There are a total of M modality types (e.g. image, video, text, and audio, etc.) and C classes globally. The k-th client only possesses data from a subset of modalities $\mathcal { M } _ { k } \subseteq \{ 1 , \dots , M \}$ , and has a combinatorial input space $\mathcal { X } _ { \mathcal { M } _ { k } } = ( \mathcal { X } _ { m } \vert \forall m \in \mathcal { M } _ { k } )$ where ${ \mathcal { X } } _ { m }$ is the subspace associated with the modality type m. The data distribution, quantity, and modality configuration of different clients are inconsistent. For client k and modality m $\in \mathcal { M } _ { k }$ , its local data is denoted as $\mathcal { D } _ { k } =$ $\{ ( \boldsymbol { x } _ { k , i } , y _ { k , i } ) \} _ { i = 1 } ^ { N _ { k } }$ , where $y _ { k , i } \in \{ 1 , \ldots , C \}$ . Each sample’s input consists of the modalities $\pmb { x } _ { k , i } = ( \pmb { x } _ { k , i } ^ { m } ) _ { m \in \mathcal { M } _ { k } }$ present as in $\mathcal { M } _ { k }$ , where $\boldsymbol { x } _ { k , i } ^ { m }$ denotes the modality m in ${ \pmb x } _ { k , i }$

Following previous conventions (Tan et al., 2022a), we split each client $k ' \mathrm { s }$ model $\theta _ { k }$ into a feature extractor $f _ { k }$ parameterized by $\varphi _ { k }$ and a classifier $h _ { k }$ parameterized by $w _ { k }$ Each client is equipped with $\vert \mathcal { M } _ { k } \vert$ modality specific feature extractors, denoted as $f _ { k } ^ { m } : \mathcal { X } _ { \mathcal { M } _ { k } }  \mathbb { R } ^ { d _ { D } }$ . The client possesses a single classifier, denoted as $h _ { k } : \mathbb { R } ^ { d _ { D } }  \mathbb { R } ^ { d _ { C } }$ . For inference or training, the client concatenates the feature vectors output by all modality specific feature extractors, and feeds this into the classifier $h _ { k }$ to obtain the final prediction. The global objective of MFL is formulated as:

$$
\operatorname* { m i n } _ { \theta } \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \frac { 1 } { N _ { k } } \sum _ { i = 1 } ^ { N _ { k } } \mathcal { L } _ { k } ( \theta _ { k } ) + \Omega ( \theta _ { 1 } , \theta _ { 2 } , \cdot \cdot \cdot , \theta _ { K } ) ,\tag{1}
$$

which aims to jointly optimize the local objectives of all clients min<sub>θ</sub> $\mathcal { L } _ { k } ( \theta _ { k } ) = \mathbb { E } _ { ( \pmb { x } , \pmb { y } ) \sim \mathcal { D } _ { k } } l \left( \pmb { y } , f _ { k } ( \pmb { x } ; \theta _ { k } ) \right)$ , where $l ( \cdot , \cdot )$ is the loss function, and Eq. (1) utilizes a central server to encourage privacy-preserving knowledge sharing schemes $\Omega ( \cdot )$ among clients in order to improve the local model performance of each client.

## 3.2. Prototype-based Federated Learning

In contrast to conventional FL, which relies on aggregating model parameters, prototype-based FL achieves knowledge sharing by exchanging class prototypes between clients and the server, thereby offering a viable solution for scenarios involving model heterogeneity.

Feature-based prototypes. During the local training phase, each client k computes its local prototype for each class c of each modality, using the following method:

$$
E _ { k , c } ^ { m } = \frac { 1 } { | \mathcal { D } _ { k , c } ^ { m } | } \sum _ { ( \pmb { x } , y ) \in \mathcal { D } _ { k , c } ^ { m } } f _ { k } ^ { m } ( \pmb { x } ) ,\tag{2}
$$

where $\mathcal { D } _ { k , c } ^ { m }$ represents a subset of $\mathcal { D } _ { k } ^ { m }$ for the m-th modality of the local dataset, containing all data samples belonging to class c. Previous studies commonly upload local feature prototypes to the server and aggregate them via weighted averaging. However, in MFL, such direct aggregation is challenging due to heterogeneous feature extractors that project data into incompatible embedding spaces.

![](images/72ad793bb93c326df8aa834f6456354278c9b601cb05f1a342064749f88c831c.jpg)  
Figure 2. The framework of the MFedPBA.

Logit-based prototypes. In contrast to feature prototypes, which suffer from inherent space incompatibility across heterogeneous models, logit prototypes reside in a naturally aligned shared semantic space. Each dimension of a logit prototype directly corresponds to a specific class, thereby providing intrinsic semantic alignment across heterogeneous models. This property effectively avoids the collaborative degradation caused by inconsistent embedding spaces and enables more robust knowledge aggregation in heterogeneous federated learning scenarios. Therefore, the prototype based on logits can be defined as the average of the logit vectors of all nodes under category c:

$$
I _ { k , c } = \frac { 1 } { | \mathcal { D } _ { k , c } ^ { m } | } \sum _ { ( \pmb { x } , \pmb { y } ) \in \mathcal { D } _ { k , c } ^ { m } } h _ { k } ( f _ { k } ^ { m } ( \pmb { x } ) ) .\tag{3}
$$

Considering that logits naturally occupy a common semantic space irrespective of the underlying model architectures, the direct alignment of output-level logit prototypes emerges as a more rational approach compared to utilizing feature-level prototypes.

## 3.3. Gromov Wasserstein Distance

The Gromov-Wasserstein (GW) distance provides a rigorous framework for quantifying the discrepancy between two metric measure spaces (Memoli ´ , 2011). The GW distance is particularly well-suited for heterogeneous settings where a direct correspondence between feature dimensions is unavailable, which facilitates alignment by comparing the intrinsic geometric or relational structures of the data (Saha et al., 2025). Specifically, the GW framework evaluates the internal pairwise distance distributions within each space and seeks an optimal coupling (transport plan) that minimizes the relational distortion between these topologies.

Formally, let $( \mathcal { X } , C _ { X } , \mathbf { p } )$ and $( \mathcal { V } , C _ { Y } , \mathbf { q } )$ be two metric measure spaces, where $C _ { X } , C _ { Y }$ denote the distance metrics of data X and Y, respectively (Peyre et al. ´ , 2016). The p, q represent the probability measures associated with each domain. The GW distance is defined as:

$$
\begin{array} { l } { { \displaystyle { G W _ { p } ( C _ { X } , C _ { Y } , { \bf p } , { \bf q } ) = } } \ ~ } \\ { { \displaystyle \left( \operatorname* { m i n } _ { T \in \Pi ( { \bf p } , { \bf q } ) } \sum _ { i , j , k , l } { \cal L } ( C _ { X } ( i , k ) , ~ C _ { Y } ( j , l ) ) ^ { p } { \cal T } _ { i j } { \cal T } _ { k l } \right) ^ { 1 / p } } \ , } \end{array}\tag{4}
$$

where $\Pi ( \mathbf { p } , \mathbf { q } ) \ = \ \left\{ T \in \mathbb { R } _ { + } ^ { n \times m } \Big | T \mathbf { 1 } _ { m } = \mathbf { p } , T ^ { \top } \mathbf { 1 } _ { n } = \mathbf { q } \right\}$ is a joint distribution of all couplings from p to q, and $p$ is the order of distance (commonly $\scriptstyle p = 2 )$ ).

## 4. Methodology

The proposed MFedPBA aligns prototypes by collecting two types at the server. The MFedPBA framework is depicted in Figure 2, with the following details.

## 4.1. Global Prototype Modeling

Each client first performs local training and then collects its local prototypes, denoted as $\{ E _ { k , c } ^ { m } \} _ { k \in K }$ and $\{ I _ { k , c } \} _ { k \in K }$ according to Eq. (2) and (3), and uploads them to the server. Upon receiving the bilateral prototypes from all clients, the server constructs global prototypes at both the logit level and the feature level, respectively.

## 4.1.1. Logit Prototype Aggregation

Since logits are the final outputs of classifiers, they naturally reside in a semantically aligned shared representation space across heterogeneous models. Therefore, no additional processing is applied to the logit prototypes, and they are directly aggregated. Considering that data distribution discrepancies across clients lead to varying levels of predictive uncertainty in their logit prototypes, an entropy weighted aggregation strategy is adopted. Clients with lower predictive uncertainty, corresponding to lower entropy, are assigned higher weights, resulting in a more reliable global logit prototype. Accordingly, defining $\sigma ( \cdot )$ as the softmax, the entropy of a logit prototype is defined as:

$$
H ( I _ { k , c } ) = - \sum _ { j = 1 } ^ { d _ { C } } \sigma ( I _ { k , c } ) _ { j } \log \sigma ( I _ { k , c } ) _ { j } .\tag{5}
$$

The server weights the values based on the inverse of the entropy, obtaining a global logit prototype for class c:

$$
\mathbf { I } _ { G , c } = \sum _ { k = 1 } ^ { K } q _ { k , c } I _ { k , c } , \quad q _ { k , c } = \frac { \left( H ( I _ { k , c } ) + \epsilon \right) ^ { - 1 } } { \sum _ { k ^ { \prime } } ^ { K } \left( H ( I _ { k ^ { \prime } , c } ) + \epsilon \right) ^ { - 1 } } ,\tag{6}
$$

where ϵ is a very small positive constant that avoids numerical collapse.

## 4.1.2. Feature Prototype Learning

To effectively capture modality-specific discriminative characteristics and facilitate cross-modal collaboration, the server introduces an auto-encoding framework composed of modality-specific encoders $\Phi _ { m } \mathbf { \check { ( } \cdot ) } : \mathbb { R } ^ { d _ { D } }  \mathbb { R } ^ { d _ { D } \mathbf { \hat { / } 2 } }$ and a shared decoder $\psi ( \cdot ) : \mathbb { R } ^ { d _ { D } / 2 } \to \mathbb { R } ^ { d _ { D } }$ . This projects feature prototypes of varying modalities and dimensionalities into a unified and comparable semantic space, while enforcing multiple constraints to achieve representation alignment across both clients and modalities. Therefore, we have:

$$
P _ { k , c } ^ { m } = \psi ( \Phi _ { m } ( E _ { k , c } ^ { m } ) ) .\tag{7}
$$

The modality-specific encoders project features of the same modality from different clients into a unified representation space, while the shared decoder enforces a common semantic structure during reconstruction. This promotes crossmodal consistency and constraining latent representations from different modalities within a shared and informationrich semantic space. Consequently, global modality feature prototypes are derived through the weighted aggregation of these intra-modality aligned prototypes:

$$
\mathbf { P } _ { G , c } ^ { m } = \frac { 1 } { \sum _ { k } \mathbb { I } ( m \in \mathcal { M } _ { k } ) } \sum _ { k , m \in \mathcal { M } _ { k } } P _ { k , c } ^ { m } ,\tag{8}
$$

here I denotes the indicator function.

To make the learned global modal prototypes more discriminative, the server updates the network parameters by minimizing the following joint loss function:

Prototype Reconstruction Loss: This loss ensures that the projection and reconstruction process preserves the essential information of the input prototypes:

$$
\mathcal { L } _ { \mathrm { r e c } } = \frac { 1 } { \vert K \vert } \sum _ { k , m , c } \Vert P _ { k , c } ^ { m } - E _ { k , c } ^ { m } \Vert ^ { 2 } .\tag{9}
$$

Inter-modal Contrastive Loss: This loss enhances semantic consistency across modalities within the same class by

pulling together projected features of different modalities for class c, while pushing apart features from different classes

$$
\mathcal { L } _ { \mathrm { c o n } } = - \sum _ { c } \sum _ { m \neq m ^ { \prime } } \log \frac { \exp \left( \sin \left( \mathbf { P } _ { G , c } ^ { m } , \mathbf { P } _ { G , c } ^ { m ^ { \prime } } \right) / \tau \right) } { \sum _ { c ^ { \prime } } \exp \left( \sin \left( \mathbf { P } _ { G , c } ^ { m } , \mathbf { P } _ { G , c ^ { \prime } } ^ { m ^ { \prime } } \right) / \tau \right) } ,\tag{10}
$$

where sim $( u , v ) = u \top v / ( \| u \| \| v \| )$ , and τ is the temperature coefficient.

Intra-modal Distribution Alignment Loss: This loss aligns the structural consistency of the same class across different modalities using the GW distance. This loss does not require strict dimensional alignment but preserves the proportional relationships of inter-class distances between the feature space and the logit space, thereby maintaining topological consistency.

We compute the GW distance between the modality-specific feature prototypes $\mathbf { P } _ { G , c } ^ { m }$ with the global logit prototypes $\mathbf { I } _ { G , c }$ space. Within each metric space (Memoli´ , 2011), we characterize the intrinsic geometric topology using the pairwise squared Euclidean distance, defined as $C _ { i , j } = \| x _ { i } - x _ { j } \| _ { 2 } ^ { 2 }$ Consequently, we derive the cost matrix $\bar { C } _ { \mathbf { I } _ { G } } \in \mathbb { R } ^ { d _ { C } \bar { \times } d _ { C } }$ and $C _ { \mathbf { P } _ { G } ^ { m } } \in \mathbb { R } ^ { d _ { C } \times d _ { C } }$ to represent the inter-class relational structures within the feature and logit spaces, respectively. Therefore, this distance loss can be expressed as:

$$
\begin{array} { l } { { \displaystyle \mathcal { L } _ { \mathrm { a l i g n } } = \sum _ { m } \left( G W _ { 2 } ^ { 2 } ( C _ { \mathbf { I } _ { G } } , C _ { \mathbf { P } _ { G } ^ { m } } , \mathbf { p } , \mathbf { q } ) - \varepsilon H ( T ^ { m } ) \right) } \ ~ } \\ { { \displaystyle = \sum _ { m } \operatorname* { m i n } _ { T ^ { m } \in \Pi ( \mathbf { p } , \mathbf { q } ) } \sum _ { i , j , k , l } \left( C _ { \mathbf { I } _ { G } } ( i , k ) - C _ { \mathbf { P } _ { G } ^ { m } } ( j , l ) \right) ^ { 2 } T _ { i j } ^ { m } T _ { k l } ^ { m } } \ ~ } \\ { { \displaystyle ~ + \sum _ { m } \varepsilon \sum _ { i j } T _ { i j } ^ { m } \log T _ { i j } ^ { m } } . } \end{array}\tag{11}
$$

To solve this problem, we introduce entropy regularization $H ( T )$ to make the problem differentiable, where ε weights this regularization, and then use the Sinkhorn algorithm (Cuturi, 2013; Sejourn ´ e et al. ´ , 2021) to solve the optimal transmission problem of this regularization.

Therefore, the server optimizes the global loss for S rounds. The total loss is formulated as:

$$
\mathcal { L } _ { \mathrm { s e r v e r } } = \mathcal { L } _ { \mathrm { r e c } } + \mathcal { L } _ { \mathrm { c o n } } + \mathcal { L } _ { \mathrm { a l i g n } } .\tag{12}
$$

The refined client feature prototypes incorporate richer global semantics and modality-specific information. Therefore, the server constructs a global feature prototype for each modality m by averaging the learned prototypes.

Finally, the server sends the global prototypes $\{ \mathbf { P } _ { G , c } ^ { m } \} _ { m = 1 } ^ { M }$ and $\mathbf { I } _ { G , c }$ to each client, transferring complementary crossmodal knowledge to compensate for the information deficiency in clients with missing modalities.

## 4.2. Client Local Update

Upon receiving the prototype set from the server, the objective of local training is to effectively distill knowledge from both the global feature prototypes and the global logit prototypes at the representation and decision levels, thereby maximally injecting global information into local representation learning. To this end, we propose a prototype-based supervised contrastive loss composed of two terms.

Feature alignment loss is introduced to mitigate feature space drift caused by model heterogeneity and biased local data distributions. Since the local feature extractor $f _ { k } ^ { m } ( \cdot )$ tends to overfit limited modal client data, we directly anchor local feature representations to the global feature prototype space, encouraging all clients to optimize toward a shared and consensus feature distribution. Specifically, we employ the mean squared error as the alignment metric, defined as

$$
\mathcal { L } _ { \mathrm { f e a } } = \frac { 1 } { | \mathcal { M } _ { k } | } \sum _ { m \in \mathcal { M } _ { k } } \frac { 1 } { C } \sum _ { c = 1 } ^ { C } \Vert \mathbf { P } _ { G , c } ^ { m } - E _ { k , c } ^ { m } \Vert _ { 2 } ^ { 2 } .\tag{13}
$$

Logit alignment loss performs knowledge distillation at the decision level. The global logit prototypes $\mathbf { I } _ { G , c }$ encode rich information about inter-class relationships and global decision structures. By enforcing the local classifier outputs to match the global logit prototypes, robust and transferable decision knowledge is distilled into local models. Accordingly, we measure the distribution discrepancy using the Kullback–Leibler (KL) divergence:

$$
\mathcal { L } _ { \mathrm { l o g i t } } = \frac { 1 } { C } \sum _ { c = 1 } ^ { C } \mathbb { D } _ { \mathrm { K L } } ( \mathbf { I } _ { G , c } \Vert I _ { k , c } ) ,\tag{14}
$$

where $\begin{array} { r } { \mathbb { D } _ { \mathrm { K L } } ( P \| Q ) = \sum _ { i } P ( i ) \ln ( \frac { P ( i ) } { Q ( i ) } ) } \end{array}$ . Finally, to preserve discriminative capability on local data domains, we incorporate the standard cross-entropy loss between the predicted logits and ground-truth labels:

$$
\mathcal { L } _ { \mathrm { c e } } = \sum _ { ( \boldsymbol { x } _ { i } , y _ { i } ) \in \mathcal { D } _ { k } } - \mathbf { 1 } _ { y _ { i } } \log \bigl ( h _ { k } ( f _ { k } ^ { m } ( \boldsymbol { x } _ { i } ) ) \bigr ) ,\tag{15}
$$

where $\sigma ( \cdot )$ is the softmax function. The total local loss is given as follows:

$$
\mathcal { L } _ { \mathrm { c l i e n t } } = \mathcal { L } _ { \mathrm { c e } } + \lambda _ { 1 } \mathcal { L } _ { \mathrm { f e a } } + \lambda _ { 2 } \mathcal { L } _ { \mathrm { l o g i t } } .\tag{16}
$$

The overall MFL algorithm is shown in Algorithm 1.

## 5. Convergence Analysis

To analyze the convergence of MFedPBA, we define t as the current communication round, $e \in \{ 0 , 1 , \cdots , E \}$ as the number of local iterations, where E denotes the maximum number of local iterations. Thus, $( t E + e )$ represents the eth iteration in the (t+1)-th communication round. We make some assumptions see Appendix B.1. Based on the above assumptions, we have the following lemmas and theorems:

Lemma 5.1. Based on Assumption B.1 and B.2, in the local iteration $e \in \{ 0 , 1 , . . . , E \}$ of the $( t + 1 ) \cdot$ -th training round, the local model loss of any client is bounded by:

$$
\begin{array} { r l } & { \mathbb { E } \left[ \mathscr { L } _ { ( t + 1 ) E } \right] \leq } \\ & { \mathscr { L } _ { t E + 0 } - ( \eta _ { c } - \frac { L \eta _ { c } ^ { 2 } } { 2 } ) \displaystyle \sum _ { e = 0 } ^ { E } \| \nabla \mathscr { L } _ { t E + e } \| _ { 2 } ^ { 2 } + \frac { L E \eta _ { c } ^ { 2 } } { 2 } \sigma ^ { 2 } . } \end{array}\tag{17}
$$

Lemma 5.2. After feature prototype learning and logit prototype aggregation are completed on the server, the loss function ofany client can be constrained asfollows:

$$
\begin{array} { r l } & { \mathbb { E } [ \mathcal { L } _ { ( t + 1 ) E + 0 } ] \leq \mathcal { L } _ { ( t + 1 ) E } } \\ & { ~ + ( M \lambda _ { 1 } + \lambda _ { 2 } ) L _ { r } E \eta _ { c } G _ { c } + 2 \lambda _ { 1 } ( \delta _ { s } + M G _ { s } ) . } \end{array}\tag{18}
$$

Based on Lemma 5.1 and Lemma 5.2, we can further derive the following theorems.

Theorem 5.3. The above assumptions, the expectation of the loss ofan arbitrary client’s local model before the start ofa round oflocal iteration satisfies:

$$
\begin{array} { r l } & { \mathbb { E } [ \mathcal { L } _ { ( t + 1 ) E + 0 } ] \leq \mathcal { L } _ { t E + 0 } - ( \eta _ { c } - \displaystyle \frac { L \eta _ { c } ^ { 2 } } { 2 } ) \sum _ { e = 0 } ^ { E } \| \nabla \mathcal { L } _ { t E + e } \| _ { 2 } ^ { 2 } } \\ & { \qquad + \displaystyle \frac { L E \eta _ { c } ^ { 2 } } { 2 } \sigma ^ { 2 } + \Gamma _ { d } E \eta _ { c } + \Gamma _ { s } . } \end{array}\tag{19}
$$

where $\Gamma _ { \mathrm { d } } = ( M \lambda _ { 1 } + \lambda _ { 2 } ) L _ { r } G _ { c }$ and $\Gamma _ { \mathrm { s } } = 2 \lambda _ { 1 } ( \delta _ { s } + M G _ { s } )$ represent the drift constant and server bias constant.

Theorem 5.4. The above assumptions,for arbitrary client and any $\epsilon > 0 ,$ , if learning rate $\eta _ { c } <$ min $\left\{ \frac { 2 } { L } , \frac { 2 ( \epsilon - \Gamma _ { \mathrm { d } } E ) } { L ( \epsilon + E \sigma ^ { 2 } ) } \right\}$ and $\Gamma _ { \mathrm { s } }  0 ,$ , thefollowing inequality holds:

$$
\begin{array} { r } { \displaystyle \frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \sum _ { e = 0 } ^ { E } \mathbb { E } \left[ \left\| \mathcal { L } _ { t E + e } \right\| _ { 2 } ^ { 2 } \right] \leq \frac { 2 \left( \mathcal { L } _ { t = 1 } - \mathcal { L } ^ { * } \right) } { T \eta _ { c } \left( 2 - L \eta _ { c } \right) } } \\ { + \frac { L E \eta _ { c } ^ { 2 } \sigma ^ { 2 } + 2 \left( \Gamma _ { d } E \eta _ { c } + \Gamma _ { s } \right) } { 2 \eta _ { c } - L \eta _ { c } ^ { 2 } } . } \end{array}\tag{20}
$$

With this, it is evident that the local model of any client of MFedPBA converges at a non-convex convergence rate $\mathcal { O } \left( \textstyle { \frac { 1 } { T } } \right)$ . See Appendix B for a detailed proof.

## 6. Experiments

## 6.1. Experimental Setup

Datasets and Model: We used four benchmark datasets for experiments: Caltech101 <sup>1</sup>, Reuters <sup>2</sup>, $\mathrm { N U S }  – \mathrm { W I D E } ^ { 3 }$ , and

Youtube <sup>4</sup>. The data information is shown in Table 1. All datasets are divided into training and test sets in a ratio of 75%/25%.

Table 1. Dataset statistics. Acronyms for modality types: I (Image), L (Language), A (Audio), T (Time-series), X1 (Style-one X), X2 (Style-two X). The #S: Samples, #C: Classes, #M: Modalities.
<table><tr><td rowspan=1 colspan=1>Datasets</td><td rowspan=1 colspan=1>Types</td><td rowspan=1 colspan=1># S</td><td rowspan=1 colspan=1># C</td><td rowspan=1 colspan=1>#M</td><td rowspan=1 colspan=1>Input modalities</td></tr><tr><td rowspan=1 colspan=1>Caltech101</td><td rowspan=1 colspan=1>{1}</td><td rowspan=1 colspan=1>9144</td><td rowspan=1 colspan=1>102</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>{11}, {I2}, and {I3}</td></tr><tr><td rowspan=1 colspan=1>Reuters</td><td rowspan=1 colspan=1>{L}</td><td rowspan=1 colspan=1>18758</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>{L1}, {L2}, {L3}{L4}, and {L5}</td></tr><tr><td rowspan=1 colspan=1>NUS-WIDE</td><td rowspan=1 colspan=1>{I, L}</td><td rowspan=1 colspan=1>5000</td><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>{11}, {I2}, {I3}, and {L}</td></tr><tr><td rowspan=1 colspan=1>Youtube</td><td rowspan=1 colspan=1>{I, A, T}</td><td rowspan=1 colspan=1>2000</td><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>{11}, {12}, {T} $\{ \mathrm { A 1 } \bar  \} , \{ \mathrm { A 2 } \} , \mathrm { a n d } \ \bar { \{ \mathrm { A 3 } } \}$ </td></tr></table>

To model heterogeneous MFL scenarios, under the assumption that the number of clients K equals the number of modalities M, we consider three client data distribution settings: “M2”, where each client possesses two modalities; “M1+”, where each client has at least one modality; and “M1”, where each client contains exactly one modality. Using the YouTube dataset as an example, the corresponding data distributions are illustrated in Figure 3. We assign heterogeneous models to each modality of every client to simulate realistic model heterogeneity. Detailed descriptions of the local dataset partitions and the corresponding neural architecture configurations are provided in Appendix C.1 & C.2.

![](images/eb3b3ceece48530cd74346c006692904250167f8a80347c74726f7f2b8f19be8.jpg)

![](images/4e1c4e4ec9fb4818f229ae396131921c6b28eed6e2e3e3d521ad2142b8d84547.jpg)

![](images/8d95796232a91883fb1b9a9bc042a0acc9337aec71ec14dc02d03150f2c10a12.jpg)  
Figure 3. Example of dataset distribution partitioning.

Baselines: To ensure fair comparison, we re-implemented all baseline methods within a unified HtFLlib framework (Zhang et al., 2025a) and evaluated them under identical experimental settings. Specifically, we include representative prototype-based federated learning methods FedProto (Tan et al., 2022a), FedTGP (Zhang et al., 2024), and Fed-Pall (Zhang et al., 2025c), as well as state-of-the-art multimodal federated learning approaches Harmony (Ouyang et al., 2023), FedMVP (Che et al., 2024), and FedMobile (Liu et al., 2025), with local training (Local) serving as a reference baseline. For consistency, Local and all prototypebased methods adopt the same local multimodal aggregation strategy as our approach. Moreover, to accommodate model heterogeneity, methods that originally require uploading local models are uniformly modified to upload classifier parameters instead. Details of all baselines are provided in Appendix C.3.

Implementation details: To ensure reliability, all methods are evaluated across five independent trials, and we report the mean performance alongside the standard deviation. SGD is adopted as the optimizer uniformly, with the number of local training epochs set to 2. Additionally, to accommodate varying dataset scales, the total number of global communication rounds is configured adaptively for each dataset. The details are provided in Appendix C.4.

## 6.2. Experimental Result

Test Accuracy. Table 2 reports the test accuracies of comparative methods on four multimodal datasets under the M2, M1+, and M1 partition settings. MFedPBA consistently achieves superior performance in all scenarios, with substantial improvements validating the efficacy of the dualprototype bilateral alignment framework.

To further investigate feature space, we visualize the feature prototypes $E _ { k } ^ { m }$ of three-modal clients in the Reuters dataset using t-SNE (Van der Maaten & Hinton, 2008), as shown in Figure 5. Despite sharing identical class labels, features extracted from different modalities are clearly separated due to heterogeneous encoders, corroborating our motivation on feature space incompatibility. Compared with Local training, our method exhibits more compact and well-structured clusters, indicating more effective representation learning. Moreover, in several settings, FedProto underperforms Local training, further confirming that naively aggregating misaligned prototypes under model heterogeneity can introduce knowledge contamination and lead to severe performance degradation rather than improvement.

Furthermore, we extended experiment to a larger number of clients to assess the scalability of the proposed framework. As summarized in Table 2 (K #Client number), the experimental results demonstrate that MFedPBA consistently maintains its superior performance even in large-scale FL settings. Detailed discussions and per-client performance results are provided in Appendix D.

Communication Efficiency. Theoretically, prototype-based FL significantly reduces communication overhead compared to methods that transmit full model parameters. The standard bidirectional communication cost for FedProto is $K 2 C M _ { k } d _ { D }$ , our framework introduces only a marginal increase for transmitting logit prototypes, bringing the total complexity to $K 2 C ( M _ { k } d _ { D } + d _ { C } )$ . Since the number of classes is typically several orders of magnitude smaller than the total model parameters, this additional overhead remains negligible. Consequently, our approach maintains high communication efficiency without imposing substantial costs relative to methods requiring parameters.

Table 2. Comparison of average performance of different methods in simulations with different data distributions. M#: Data partition (K=M); K#: Number of large clients (in M1+). Best and second-best are bold and underlined respectively.
<table><tr><td colspan="2">Dataset</td><td>Local</td><td>FedProto</td><td>FedTGP</td><td>FedPall</td><td>Harmony</td><td>FedMVP</td><td>FedMobile</td><td>MFedPBA</td></tr><tr><td rowspan="4">Caltech101</td><td>M2</td><td> $4 2 . 6 5 _ { \pm 1 . 4 7 }$ </td><td> $3 8 . 7 2 _ { \pm 0 . 4 2 }$ </td><td> $4 0 . 7 8 { \scriptstyle \pm 1 . 3 1 }$ </td><td> $4 2 . 2 9 _ { \pm 0 . 2 7 }$ </td><td> $4 5 . 2 9 _ { \pm 0 . 3 8 }$ </td><td> $4 0 . 2 3 _ { \pm 0 . 6 5 }$ </td><td> $4 3 . 5 1 _ { \pm 0 . 5 4 }$ </td><td> ${ \bf 4 9 . 0 4 } _ { \pm 0 . 2 1 }$ </td></tr><tr><td>M1+</td><td> $4 0 . 8 3 _ { \pm 1 . 1 2 }$ </td><td> $3 6 . 8 1 _ { \pm 0 . 3 5 }$ </td><td> $4 0 . 7 1 _ { \pm 1 . 2 3 }$ </td><td> $4 1 . 8 _ { \pm 0 . 7 7 }$ </td><td> $\overline { { 4 3 . 0 4 _ { \pm 0 . 6 4 } } }$ </td><td> $3 8 . 9 1 _ { \pm 1 . 2 5 }$ </td><td> $4 2 . 2 9 _ { \pm 0 . 6 }$ </td><td> ${ \bf 4 6 . 8 2 _ { \pm 0 . 5 4 } }$ </td></tr><tr><td>M1</td><td> $3 8 . 4 7 _ { \pm 0 . 4 5 }$ </td><td> $3 1 . 5 7 _ { \pm 1 . 2 3 }$ </td><td> $3 6 . 1 8 _ { \pm 0 . 7 2 }$ </td><td> $4 0 . 7 7 _ { \pm 0 . 3 6 }$ </td><td> $4 0 . 1 _ { \pm 0 . 6 9 }$ </td><td> $3 8 . 9 2 _ { \pm 0 . 6 2 }$ </td><td> $4 0 . 0 9 _ { \pm 0 . 1 3 }$ </td><td> ${ \pm 3 . 6 4 } _ { \pm 0 . 1 1 }$ </td></tr><tr><td>K50</td><td> $2 9 . 3 8 _ { \pm 1 . 6 3 }$ </td><td> $2 5 . 1 7 _ { \pm 1 . 7 8 }$ </td><td> $2 8 . 7 7 _ { \pm 1 . 4 3 }$ </td><td> $\overline { { 3 0 . 5 9 _ { \pm 1 . 2 6 } } }$ </td><td> $3 0 . 5 8 _ { \pm 0 . 6 5 }$ </td><td> $2 7 . 7 4 _ { \pm 1 . 5 }$ </td><td> $3 0 . 8 8 _ { \pm 0 . 8 1 }$ </td><td> $\mathbf { 3 1 . 1 9 _ { \pm 0 . 9 2 } }$ </td></tr><tr><td rowspan="4">Reuters</td><td>M2</td><td> $7 0 . 6 1 { \scriptstyle \pm 0 . 2 8 }$ </td><td> $7 3 . 3 9 _ { \pm 0 . 5 7 }$ </td><td> $6 9 . 7 3 _ { \pm 0 . 8 8 }$ </td><td> $7 3 . 0 8 _ { \pm 1 . 0 4 }$ </td><td> $7 2 . 1 8 _ { \pm 0 . 4 1 }$ </td><td> $7 4 . 0 7 { \scriptstyle \pm 1 . 0 8 }$ </td><td> $7 3 . 1 9 _ { \pm 0 . 2 6 }$ </td><td> $7 5 . 1 6 _ { \pm 0 . 4 5 }$ </td></tr><tr><td>M1+</td><td> $7 0 . 8 5 _ { \pm 0 . 5 9 }$ </td><td> $6 8 . 5 4 _ { \pm 1 . 4 6 }$ </td><td> $6 9 . 8 _ { \pm 0 . 5 9 }$ </td><td> $7 1 . 0 9 { \scriptstyle \pm 1 . 5 6 }$ </td><td> $7 2 . 2 8 _ { \pm 0 . 4 7 }$ </td><td> $6 8 . 0 9 _ { \pm 0 . 8 2 }$ </td><td> $7 2 . 7 9 _ { \pm 0 . 4 4 }$ </td><td> ${ \bf 7 5 . 0 1 { \bf _ { \pm 0 . 2 2 } } }$ </td></tr><tr><td>M1</td><td> $6 7 . 0 5 _ { \pm 1 . 0 4 }$ </td><td> $6 8 . 7 5 _ { \pm 0 . 6 3 }$ </td><td> $6 5 . 8 1 _ { \pm 0 . 3 3 }$ </td><td> $6 6 . 9 6 _ { \pm 0 . 2 2 }$ </td><td> $7 0 . 1 6 _ { \pm 1 . 1 4 }$ </td><td> $7 0 . 1 8 _ { \pm 0 . 5 7 }$ </td><td> $\overline { { 6 9 . 9 2 _ { \pm 0 . 9 5 } } }$ </td><td> $7 2 . 0 9 _ { \pm 0 . 7 9 }$ </td></tr><tr><td>K100</td><td> $5 2 . 8 8 _ { \pm 1 . 0 8 }$ </td><td> $4 6 . 4 8 _ { \pm 0 . 9 3 }$ </td><td> $4 8 . 3 1 _ { \pm 1 . 4 1 }$ </td><td> $5 6 . 5 5 _ { \pm 0 . 5 3 }$ </td><td> $5 3 . 9 6 _ { \pm 0 . 5 1 }$ </td><td> $5 2 . 2 8 _ { \pm 1 . 5 3 }$ </td><td> $5 6 . 6 4 _ { \pm 0 . 2 9 }$ </td><td> ${ \bf 5 7 . 8 3 _ { \pm 0 . 9 7 } }$ </td></tr><tr><td rowspan="4">NUS-WIDE</td><td>M2</td><td> $3 2 . 1 8 _ { \pm 0 . 6 4 }$ </td><td> $2 9 . 2 8 _ { \pm 0 . 4 6 }$ </td><td> $3 1 . 1 9 _ { \pm 0 . 5 1 }$ </td><td> $3 3 . 3 8 _ { \pm 0 . 9 1 }$ </td><td> $3 2 . 9 7 _ { \pm 0 . 2 8 }$ </td><td> $3 1 . 8 9 _ { \pm 0 . 4 }$ </td><td> $3 3 . 8 6 _ { \pm 0 . 2 2 }$ </td><td> $3 5 . 2 _ { \pm 0 . 4 6 }$ </td></tr><tr><td>M1+</td><td> $2 8 . 7 2 _ { \pm 0 . 7 1 }$ </td><td> $2 7 . 7 9 _ { \pm 0 . 5 5 }$ </td><td> $2 7 . 1 3 _ { \pm 0 . 1 8 }$ </td><td> $3 0 . 9 9 _ { \pm 0 . 6 3 }$ </td><td> $3 1 . 5 6 { \scriptstyle \pm 0 . 6 1 }$ </td><td> $2 9 . 8 2 _ { \pm 0 . 5 9 }$ </td><td> $3 0 . 9 5 _ { \pm 0 . 5 7 }$ </td><td> $3 2 . 1 5 _ { \pm 0 . 3 6 }$ </td></tr><tr><td>M1</td><td> $3 1 . 1 5 _ { \pm 0 . 1 4 }$ </td><td> $2 4 . 9 6 _ { \pm 0 . 5 2 }$ </td><td> $2 9 . 5 5 _ { \pm 0 . 7 3 }$ </td><td> $3 1 . 0 2 _ { \pm 0 . 9 6 }$ </td><td> $\overline { { 3 2 . 1 9 _ { \pm 0 . 3 5 } } }$ </td><td> $\underline { { 3 2 . 7 _ { \pm 0 . 5 1 } } }$ </td><td> $3 1 . 2 _ { \pm 0 . 3 6 }$ </td><td> $3 3 . 2 5 _ { \pm 0 . 2 4 }$ </td></tr><tr><td>K20</td><td> $2 4 . 5 2 _ { \pm 0 . 5 }$ </td><td> $2 4 . 4 7 _ { \pm 0 . 9 5 }$ </td><td> $2 4 . 1 5 _ { \pm 0 . 3 7 }$ </td><td> $2 6 . 3 5 _ { \pm 0 . 4 4 }$ </td><td> $2 6 . 2 8 _ { \pm 0 . 4 1 }$ </td><td> $2 6 . 6 1 _ { \pm 0 . 8 6 }$ </td><td> $2 7 . 3 4 _ { \pm 0 . 3 7 }$ </td><td> ${ \bf 2 8 . 0 8 _ { \pm 0 . 3 7 } }$ </td></tr><tr><td rowspan="4">Youtube</td><td>M2</td><td> $4 1 . 7 2 _ { \pm 0 . 6 6 }$ </td><td> $4 1 . 1 2 _ { \pm 0 . 6 8 }$ </td><td> $4 5 . 1 6 _ { \pm 1 . 0 2 }$ </td><td> $4 4 . 2 3 _ { \pm 0 . 3 }$ </td><td> $4 3 . 8 _ { \pm 0 . 7 6 }$ </td><td> $4 5 . 5 3 _ { \pm 0 . 5 2 }$ </td><td> $4 4 . 8 3 _ { \pm 0 . 5 3 }$ </td><td> ${ \bf 4 7 . 9 2 _ { \pm 0 . 9 3 } }$ </td></tr><tr><td>M1+</td><td> $4 1 . 8 9 _ { \pm 0 . 7 6 }$ </td><td> $3 8 . 8 2 _ { \pm 1 . 1 2 }$ </td><td> $4 4 . 5 5 \mathrm { \pm 0 . 6 8 }$ </td><td> $4 3 . 0 4 _ { \pm 1 . 0 9 }$ </td><td> $4 3 . 4 7 _ { \pm 0 . 9 1 }$ </td><td> $\overline { { 4 4 . 2 4 _ { \pm 0 . 3 6 } } }$ </td><td> $4 4 . 2 8 \substack { \pm 0 . 6 4 }$ </td><td> ${ \bf 4 7 . 2 1 { _ { \pm 0 . 5 9 } } }$ </td></tr><tr><td>M1</td><td> $4 0 . 0 1 _ { \pm 0 . 4 1 }$ </td><td> $3 6 . 5 9 _ { \pm 0 . 3 9 }$ </td><td> $\overline { { 3 8 . 6 2 _ { \pm 0 . 6 4 } } }$ </td><td> $3 9 . 1 5 _ { \pm 1 . 5 }$ </td><td> $\underline { { 4 1 . 0 7 _ { \pm 0 . 4 2 } } }$ </td><td> $4 0 . 0 4 _ { \pm 0 . 4 6 }$ </td><td> $3 9 . 4 8 _ { \pm 0 . 9 9 }$ </td><td> $\pm 4 . 5 2 _ { \pm 0 . 5 4 }$ </td></tr><tr><td>K20</td><td> $3 2 . 0 1 _ { \pm 0 . 2 5 }$ </td><td> $3 0 . 5 7 _ { \pm 1 . 4 9 }$ </td><td> $3 4 . 1 5 _ { \pm 0 . 3 1 }$ </td><td> $3 2 . 5 9 _ { \pm 0 . 4 7 }$ </td><td> $3 3 . 5 _ { \pm 0 . 4 9 }$ </td><td> $3 1 . 1 2 _ { \pm 0 . 7 7 }$ </td><td> $3 4 . 2 2 _ { \pm 0 . 6 3 }$ </td><td> ${ \bf 3 5 . 0 7 } _ { \pm 0 . 9 2 }$ </td></tr></table>

![](images/907e25b387621a176f6c3b6fbd4865292c2794070c8a478aa3c5a2f68067e71c.jpg)

![](images/215949dbbe0307f35afb6d4fb22d57c1cd1290c8d408ab58b9a73c509fe4de62.jpg)

![](images/fdb8125d50b341c3b23aba9c04e02aaeb0640756038b36c581b971ea6be07148.jpg)

![](images/632492718194ff5d73c01c9f4f136c50365bfac69555af9490f3336af3634cd5.jpg)  
Figure 4. Test accuracy (%) on four datasets under the M2 data distribution setting with model heterogeneity.

![](images/76b4239b8ddce40c72f7e2a98089d7a5350a629fadeaeac737cdd1ecd612fe9c.jpg)  
Figure 5. Feature visualization results of Local and MFedPBA.

We plotted the performance curves of various methods across four datasets under the M2 configuration, as shown in Figure 4, thereby confirming the stable convergence of our proposed framework. We observe that prototype-based baselines, particularly FedTGP, exhibit pronounced fluctuations during the training process. This instability stems from the erratic server prototype updates in multimodal scenarios, which hinder the acquisition of discriminative representations. In contrast, our feature prototypes are regularized by a multi-faceted objective function, allowing them to consolidate global semantic knowledge while preserving critical modality-specific discriminability.

Computational Cost. Compared to the baseline FedProto, the incremental client overhead is limited to the KL divergence between local and global logit prototypes, yielding a marginal complexity of $\mathcal { O } ( C ^ { 2 } )$ . On the server, logit aggregation follows a linear complexity of $\mathcal { O } ( K C d _ { C } )$ while feature-level operations encompass autoencoder mapping at $\begin{array} { r } { \mathcal { O } ( \sum _ { k } | \mathcal { M } _ { k } | C { d } _ { D } { } ^ { 2 } ) } \end{array}$ , reconstruction loss calculation at $\begin{array} { r } { \mathcal { O } ( \sum _ { k } | \mathcal { M } _ { k } | C { d } _ { D } ) } \end{array}$ , cross-modal contrastive loss at $\mathcal { O } ( C M ^ { 2 } d _ { D } )$ , and intra-modality structural alignment at $\mathcal { O } ( T _ { s } M C ^ { 2 } )$ , where $\mathcal { T } _ { s }$ is the number of iterations of the Sinkhorn algorithm. Although these complexities scale with the class count $C ,$ they remain manageable within the scope of standard classification tasks. Crucially, our framework strategically offloads the primary computational burden to the resource-rich server, leveraging its superior processing power to facilitate high-precision alignment without straining resource-constrained clients. Figure 6 shows the running time of the client in each round on Reuters, further demonstrating that our computational overhead is acceptable.

Ablation Study. Figure 7 presents a comprehensive ablation study conducted under the Caltech101 M2 configuration to evaluate the individual contributions of both server-side and client-side modules. In this analysis, the horizontal axis denotes the exclusion (w/o) of specific algorithmic components, while $\because \mathrm { E A } ^ { \prime \prime }$ indicates the substitution of our proposed entropy-based weighting with standard equal averaging. The empirical results clearly demonstrate that the removal of any server-side module precipitates a significant drop in overall accuracy, thereby validating the necessity and synergy of the integrated design. Notably, eliminating the reconstruction loss $( \mathcal { L } _ { \mathrm { r e c } } )$ triggers the most severe performance degradation. This acute decline occurs because the absence of reconstruction constraints induces representation collapse within the latent space, which subsequently propagates erroneous global knowledge into the local training phase of individual clients.

![](images/195b7d92aa77bcca105b81abe0b105a51c43cfb1ff2f7f200c86f917bc9ebc10.jpg)  
Figure 6. Time consumption.

![](images/3f425ddfb1948b64c566915a0c886391c3a95f50c22133e546aac9b2350a7482.jpg)  
Figure 7. Ablation results.

Impact of Feature Dimensions. We systematically evaluate the impact of the feature dimension $d _ { D }$ on model performance by varying it across a predefined set of values (e.g., $d _ { D } \in \{ 2 4 , 4 8 , 6 4 , 1 2 8 , 2 5 6 \}$ ), as summarized in Table 3. We observe a clear trade-off: smaller dimensions restrict the model’s ability to capture complex data patterns, leading to underfitting, whereas excessively large dimensions introduce redundant noise and complicate classifier training. Specifically, on the YouTube dataset, most evaluated methods achieve their most stable and superior performance at $d _ { D } = 4 8$ . Based on this empirical observation, we independently conduct this sensitivity analysis across all datasets, ultimately selecting the dataset-specific optimal dimensions to strike the best balance between representational capacity and trainability.

Table 3. The test accuracy (%) on Youtube in the M1+ setting. “Fed” is omitted in the method name due to limited space.
<table><tr><td>dD</td><td>Local</td><td>Proto</td><td>TGP</td><td>Pall</td><td>Harm</td><td>MVP</td><td>Mobile</td><td>PBA</td></tr><tr><td>24-d</td><td>41.19</td><td>30.12</td><td>41.19</td><td>42.56</td><td>42.06</td><td>43.85</td><td>40.91</td><td>44.43</td></tr><tr><td>48-d</td><td>42.34</td><td>39.05</td><td>43.85</td><td>44.63</td><td>44.43</td><td>45.21</td><td>44.57</td><td>47.02</td></tr><tr><td>64-d</td><td>43.02</td><td>37.82</td><td>45.22</td><td>42.27</td><td>43.13</td><td>43.21</td><td>44.24</td><td>46.46</td></tr><tr><td>128-d</td><td>42.64</td><td>31.24</td><td>45.01</td><td>42.15</td><td>43.24</td><td>44.33</td><td>44.33</td><td>45.68</td></tr><tr><td>256-d</td><td>41.95</td><td>26.32</td><td>43.22</td><td>41.75</td><td>42.74</td><td>42.84</td><td>42.54</td><td>43.84</td></tr></table>

Hyper-parameter. We analyze the sensitivity of server epochs S and regularization terms $\lambda _ { 1 } , \lambda _ { 2 } .$ As shown in Figure 8 (left). While accuracy improves with larger S, gains diminish significantly from 50 to 1000 epochs. Consequently, we set $S = 1 0$ to balance efficiency and performance. We evaluated the sensitivity of the $\lambda _ { 1 } , \lambda _ { 2 }$ using Youtube in the range $\{ 0 . 0 0 1 , 0 . 0 1 , 0 . 0 5 , 0 . 1 , 0 . 5 , 1 , 5 , 8 , 1 0 \}$ . As shown in Figure 8 (right), MFedPBA demonstrates strong robustness across most parameter choices, consistently achieving accuracy above 46 and outperforming all baselines. Under several favorable configurations, the performance further improves to 48.9. In contrast, excessively large loss weights may introduce overly strong alignment constraints, leading to performance degradation.

![](images/051b3717714356a7b469a6f9daf17fdf60bc1538a986c620bc8ee05df722e530.jpg)

![](images/ebc03d7c43c4d5365857f8506548199d0421d27be401a01c155c89398431384b.jpg)  
Figure 8. Hyperparameter experiments for $\lambda _ { 1 } , \lambda _ { 2 } ,$ and S.

## 7. Conclusion.

This paper addresses the challenges of model heterogeneity and modality imbalance in multimodal federated learning by proposing a prototype-guided bidirectional alignment framework for effective knowledge sharing across heterogeneous feature spaces. We analyze the limitations of conventional prototype aggregation under heterogeneous encoders and introduce a dual alignment mechanism that anchors knowledge transfer through semantically aligned logit prototypes while bridging embedding space discrepancies via GW–based feature alignment. Experimental results show that the proposed method effectively mitigates negative transfer and knowledge contamination, and significantly improves generalization and robustness under imbalanced and missing-modality settings.

## Acknowledgements

The research is supported by Seed Funding for Collaborative Research Grants of HKBU (with Grant No. RC-SFCRG/23-24/R2/SCI/06), the National Key Research and Development Program of China (2023YFB2703700), and the GMCC-SYSU Joint Lab for Smart Applications.

## Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

## References

Cao, M., Chen, T., Hu, M., Qi, Z., Cui, Y., Zhou, J., and Xie, X. Two heads are better than one: Generalized cross-domain federated learning via dual-prototype. In Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pp. 59–70, 2026.

Che, L., Wang, J., Zhou, Y., and Ma, F. Multimodal federated learning: A survey. Sensors, 23(15):6986, 2023.

Che, L., Wang, J., Liu, X., and Ma, F. Leveraging foundation models for multi-modal federated learning with incomplete modality. In Joint European Conference on Machine Learning and Knowledge Discovery in Databases, pp. 401–417. Springer, 2024.

Chen, C., Liao, T., Deng, X., Wu, Z., Huang, S., and Zheng, Z. Advances in robust federated learning: A survey with heterogeneity considerations. IEEE Transactions on Big Data, 11(3):1548–1567, 2025.

Chen, H., Zhang, Y., Krompass, D., Gu, J., and Tresp, V. Feddat: An approach for foundation model finetuning in multi-modal heterogeneous federated learning. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pp. 11285–11293, 2024.

Chen, J. and Zhang, A. Fedmbridge: Bridgeable multimodal federated learning. In Forty-first International Conference on Machine Learning, 2024.

Collins, L., Hassani, H., Mokhtari, A., and Shakkottai, S. Exploiting shared representations for personalized federated learning. In International conference on machine learning, pp. 2089–2099. PMLR, 2021.

Cuturi, M. Sinkhorn distances: Lightspeed computation of optimal transport. Advances in neural information processing systems, 26, 2013.

Deng, Y., He, N., Li, X., Wu, F., Zhang, Y., and Ren, J. Cross-modal federated learning among unimodal devices. Proceedings ofthe ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies, 9(3):1–26, 2025.

Feng, T., Bose, D., Zhang, T., Hebbar, R., Ramakrishna, A., Gupta, R., Zhang, M., Avestimehr, S., and Narayanan, S. Fedmultimodal: A benchmark for multimodal federated learning. In Proceedings ofthe 29th ACM SIGKDD conference on knowledge discovery and data mining, pp. 4035–4045, 2023.

Fu, L., Huang, S., Lai, Y., Zhang, C., Dai, H.-N., Zheng, Z., and Chen, C. Federated domain-independent prototype learning with alignments of representation and parameter spaces for feature shift. IEEE Transactions on Mobile Computing, 24(9):9004–9019, 2025.

He, W., Huang, W., Yang, B., Liu, S., and Ye, M. Spmc: Self-purifying federated backdoor defense via margin contribution. In Forty-second International Conference on Machine Learning, 2025.

Hu, M., Yue, Z., Xie, X., Chen, C., Huang, Y., Wei, X., Lian, X., Liu, Y., and Chen, M. Is aggregation the only choice? federated learning via layer-wise model recombination. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pp. 1096–1107, 2024a.

Hu, M., Zhou, P., Yue, Z., Ling, Z., Huang, Y., Li, A., Liu, Y., Lian, X., and Chen, M. Fedcross: Towards accurate federated learning via multi-model cross-aggregation. In Proceedings ofIEEE International Conference on Data Engineering (ICDE), pp. 2137–2150. IEEE, 2024b.

Huang, S., Fu, L., Chen, Z., Zhang, T., Li, X., and Cui, Z. Adan: Adversarial distribution alignment network for multi-view semi-supervised classification. IEEE Transactions on Image Processing, 35:4861–4876, 2026.

Huang, W., Ye, M., and Du, B. Learn from others and be yourself in heterogeneous federated learning. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pp. 10143–10153, 2022.

Huang, W., Ye, M., Shi, Z., Li, H., and Du, B. Rethinking federated learning with domain shift: A prototype view. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 16312–16322. IEEE, 2023.

Huang, W., Wang, D., Ouyang, X., Wan, J., Liu, J., and Li, T. Multimodal federated learning: Concept, methods, applications and future directions. Information Fusion, 112:102576, 2024.

Kou, Z., Wu, J., Huang, W., He, W., Xie, M.-K., Wang, C., Jia, Y., Jiang, D., Liu, Y., Geng, X., and Yang, Q. Fedharmony: Harmonizing heterogeneous label correlations in federated multi-label learning. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026.

Li, Y., Xu, W., Qi, Y., Wang, H., Li, R., and Guo, S. SR-FDIL: synergistic replay for federated domainincremental learning. IEEE Trans. Parallel Distributed Syst., 35(11):1879–1890, 2024.

Li, Y., Wang, H., Qi, Y., Liu, W., and Li, R. Re-fed+: A better replay strategy for federated incremental learning. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025.

Liao, T., Fu, L., Chen, J., Wang, Z., Zheng, Z., and Chen, C. A swiss army knife for heterogeneous federated learning: Flexible coupling via trace norm. Advances in Neural Information Processing Systems, 37:139886–139911, 2024.

Liao, T., Xie, B., Fu, L., Huang, S., Deng, B., Chen, C., and Zheng, Z. Federated domain generalization with decision insight matrix. In Proceedings ofthe International Joint Conference on Artificial Intelligence, Montreal, QC,´ Canada, pp. 16–22, 2025.

Liu, Y., Wang, C., and Yuan, X. Fedmobile: Enabling knowledge contribution-aware multi-modal federated learning with incomplete modalities. In Proceedings ofthe ACM on Web Conference 2025, pp. 2775–2786, 2025.

Ma, Y., Dai, W., Jiang, G., Zhou, C., Zhang, Y., Luo, F., Wang, J., Zhang, A., et al. Fedmc: Federated manifold calibration. In The Fourteenth International Conference on Learning Representations.

Memoli, F. Gromov–wasserstein distances and the metric´ approach to object matching. Foundations of computational mathematics, 11(4):417–487, 2011.

Meng, L., Qi, Z., Wu, L., Du, X., Li, Z., Cui, L., and Meng, X. Improving global generalization and local personalization for federated learning. IEEE Transactions on Neural Networks and Learning Systems, 36(1):76–87, 2024.

Ouyang, X., Xie, Z., Fu, H., Cheng, S., Pan, L., Ling, N., Xing, G., Zhou, J., and Huang, J. Harmony: Heterogeneous multi-modal federated learning through disentangled model training. In Proceedings of the 21st Annual International Conference on Mobile Systems, Applications and Services, pp. 530–543, 2023.

Pan, H., Zhao, X., He, L., Shi, Y., and Lin, X. A survey of multimodal federated learning: background, applications, and perspectives. Multimedia Systems, 30(4):222, 2024.

Pan, H., Zhao, X., Jiang, Y., He, L., Wang, B., and Shu, Y. Fedvlp: Visual-aware latent prompt generation for multimodal federated learning. Computer Vision and Image Understanding, 259:104442, 2025.

Peyre, G., Cuturi, M., and Solomon, J. Gromov-wasserstein´ averaging of kernel and distance matrices. In International conference on machine learning, pp. 2664–2672. PMLR, 2016.

Qi, Z., Meng, L., Chen, Z., Hu, H., Lin, H., and Meng, X. Cross-silo prototypical calibration for federated learning with non-iid data. In Proceedings of the 31st ACM international conference on multimedia, pp. 3099–3107, 2023.

Qi, Z., Meng, L., Li, Z., Hu, H., and Meng, X. Cross-silo feature space alignment for federated learning on clients with imbalanced data. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pp. 19986–19994, 2025.

Rahman, R. and Nguyen, D. C. Multimodal federated learning with model personalization. In OPT 2024: Optimizationfor Machine Learning, 2024.

Saha, P., Mishra, D., Wagner, F., Kamnitsas, K., and Noble, J. A. Fedpia–permuting and integrating adapters leveraging wasserstein barycenters for finetuning foundation models in multi-modal federated learning. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pp. 20228–20236, 2025.

Sejourn´ e, T., Vialard, F.-X., and Peyr´ e, G. The unbalanced´ gromov wasserstein distance: Conic formulation and relaxation. Advances in Neural Information Processing Systems, 34:8766–8779, 2021.

Tan, Y., Long, G., Liu, L., Zhou, T., Lu, Q., Jiang, J., and Zhang, C. Fedproto: Federated prototype learning across heterogeneous clients. In Proceedings of the AAAI conference on artificial intelligence, volume 36, pp. 8432–8440, 2022a.

Tan, Y., Long, G., Ma, J., Liu, L., Zhou, T., and Jiang, J. Federated learning from pre-trained models: A contrastive learning approach. Advances in neural information processing systems, 35:19332–19344, 2022b.

Thrasher, J., Devkota, A., Siwakoti, P., Chivukula, R., Poudel, P., Hu, C., Tafti, A., Bhattarai, B., and Gyawali, P. Multimodal federated learning in healthcare: a review. Journal of Healthcare Informatics Research, pp. 1–30, 2025.

Van der Maaten, L. and Hinton, G. Visualizing data using t-sne. Journal ofmachine learning research, 9(11), 2008.

Wang, S., Qu, Z., Liu, Y., Kan, S., Liang, Y., and Wang, J. Fedmmr: Multi-modal federated learning via missing modality reconstruction. In 2024 IEEE International Conference on Multimedia and Expo, pp. 1–6, 2024.

Xiao, T., Li, Y., Qi, Y., Liu, Y., Wang, H., Wang, Y., and Li, R. Enhancing privacy in multimodal federated learning with information theory. Advances in Neural Information Processing Systems, 38:142199–142215, 2026.

Yi, L., Wang, G., Liu, X., Shi, Z., and Yu, H. Fedgh: Heterogeneous federated learning with generalized global header. In Proceedings of the 31st ACM international conference on multimedia, pp. 8686–8696, 2023.

Yu, S., Liu, S., Wang, S., Tang, C., Luo, Z., Liu, X., and Zhu, E. Sparse low-rank multi-view subspace clustering with consensus anchors and unified bipartite graph. IEEE Transactions on Neural Networks and Learning Systems, 36(1):1438–1452, 2025.

Zhang, J., Liu, Y., Hua, Y., and Cao, J. Fedtgp: Trainable global prototypes with adaptive-margin-enhanced contrastive learning for data and model heterogeneity in federated learning. In Proceedings ofthe AAAI conference on artificial intelligence, volume 38, pp. 16768–16776, 2024.

Zhang, J., Wu, X., Zhou, Y., Sun, X., Cai, Q., Liu, Y., Hua, Y., Zheng, Z., Cao, J., and Yang, Q. Htfllib: A comprehensive heterogeneous federated learning library and benchmark. In Proceedings ofthe 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2025a.

Zhang, R., Chi, X., Zhang, W., Liu, G., Wang, D., and Wang, F. Unimodal training-multimodal prediction: Crossmodal federated learning with hierarchical aggregation. IEEE Trans. Mob. Comput., 24(10):10009–10023, 2025b.

Zhang, Y., Liang, F., Yuan, G., Yang, M., Li, C., and Hu, X. Fedpall: Prototype-based adversarial and collaborative learning for federated learning with feature drift. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 3111–3120, 2025c.

Zhao, J., Yi, X., Li, R., Li, Y., Wang, H., Li, Y., Deng, Z., and Xu, Z. Fedta: Unsupervised federated prototype learning with temperature adaptation. In 2024 IEEE International Conference on High Performance Computing and Communications (HPCC), pp. 390–397. IEEE, 2024.

Zhao, Y., Barnaghi, P., and Haddadi, H. Multimodal federated learning on iot data. In 2022 IEEE/ACM seventh international conference on internet-of-things design and implementation (ioTDI), pp. 43–54. IEEE, 2022.

Zheng, T., Li, A., Chen, Z., Wang, H., and Luo, J. Autofed: Heterogeneity-aware federated multimodal learning for robust autonomous driving. In Proceedings of the 29th annual international conference on mobile computing and networking, pp. 1–15, 2023.

## Summary of Appendix

In the appendix, the following contents are included:

A Algorithm A.1 Algorithm Framework A.2 Privacy Analysis

B Convergence Analysis B.1 Assumption B.2 Lemmas & Theorems B.3 Proof of Lemma 5.1 B.4 Proof of Lemma 5.2 B.5 Proof of Theorem 5.3 B.6 Proof of Theorem 5.4

C Details of the Experimental Setup C.1 Dataset Description C.2 Heterogeneous Model C.3 Baseline Methods C.4 Parameter Setting Details

D Details of the Experimental Result

## A. Algorithm

## A.1. Algorithm Framework

We summarize the main steps of the proposed in Algorithm 1.

## A.2. Privacy Analysis

Compared with federated learning paradigms that necessitate gradient or parameter sharing (Ouyang et al., 2023; Che et al., 2024; Liu et al., 2025), the proposed MFedPBA framework affords enhanced privacy preservation by constraining communication exclusively to feature and logit prototypes.

Feature prototypes are synthesized via client-specific non-linear encoders whose parameters remain strictly local. Consequently, the inverse reconstruction of raw inputs from embedding vectors presents a mathematically ill-posed problem, thereby significantly impeding data reconstruction attacks. Concurrently, logit prototypes abstract model outputs into class-level decision statistics, revealing minimal information regarding individual data instances. This abstraction renders it arduous for adversaries to distinguish whether a prototype is dominated by a solitary sensitive sample or derived from a diverse aggregate, effectively attenuating the efficacy of inference attacks.

Furthermore, the generation of both prototype modalities via class-wise averaging eliminates sample-level granularity, mitigating the risk of single-instance inference.

Finally, the server-side prototype optimization introduces an additional transformation layer, ensuring that the global prototypes distributed to clients are not direct replicas of any single client’s representations, thus further reducing the probability of cross-client information leakage.

## B. Convergence Analysis

To analyze the convergence of MFedPBA, we follow (Tan et al., 2022a) and make the following assumptions B.1.

Algorithm 1 MFedPBA   
Input: total rounds T, local epochs ${ \overline { { E , } } }$ server epochs S, total number of clients ${ \overline { { K } } } ,$ , sampled number of clients $\overline { { K _ { C } } } .$ , local   
learning rate $\eta _ { c }$ , server learning rate $\eta _ { s } ,$ hyper-parameter for loss $\lambda _ { 1 }$ and $\lambda _ { 2 }$   
Server executes:   
1: Initialize modality-specific encoders $\Phi _ { m } ( \cdot )$ and shared decoder $\psi ( \cdot )$   
2: for each round $t = 1 \cdots T$ do   
3: Server samples subset $K _ { C }$ of clients   
4: for each client $k \in K _ { C }$ in parallel do   
5: $\{ \{ E _ { k , c } ^ { m } \} _ { m = 1 } ^ { | \mathcal { M } _ { k } | } , I _ { k , c } \} \gets$ Client updates $\{ \mathbf { P } _ { G , c } ^ { m } \} , \mathbf { I } _ { G , c } )$   
6: end for   
7: Aggregate logit prototype by Eq. (6)   
8: for each server epoch $s = 1 \cdots S$ do   
9: Project client prototype to a unified space by Eq. (7)   
10: Aggregate modal feature prototype by Eq. (8)   
11: Calculate server loss update by Eq. (12)   
12: end for   
13: end for   
Clients updates:   
1: for each local epoch $e = 1 \cdots E$ do   
2: Sample mini-batch in B:   
3: Calculate sample feature-based prototype $E _ { k , c } ^ { m }$ by Eq. (2) and logit-based prototype $I _ { k , c }$ by Eq. (3)   
4: Calculate local loss by Eq. (16)   
5: Update local model: $\begin{array} { r l } {  { \dot { \theta } _ { k } ^ { t + 1 } \longleftarrow \theta _ { k } ^ { t } - \eta _ { c } \nabla \mathcal { L } _ { k } ( \theta _ { k } ^ { t } ; \mathcal { B } _ { k } ) } \quad } & { { } } \end{array}$   
6: end for   
7: return $\{ E _ { k , c } ^ { m } \} _ { m = 1 } ^ { | \mathcal { M } _ { k } | }$ and $I _ { k , c }$

As for the iteration notation system, we define t as the current communication round, $e \in \{ 0 , 1 , \cdots , E \}$ as the number of local iterations, where E denotes the maximum number of local iterations. Thus, $( t E + e )$ represents the e-th iteration in the (t + 1)-th communication round. The $( t + 0 )$ denotes that at the beginning of the $( t + 1 )$ -th round. Note that $( t E + E )$ corresponds to the last iteration in round (t + 1). Moreover, tE represents the time step before prototype aggregation, and $t E + 0$ represents the time step between prototype learning and the first iteration of the current round. Therefore, we can decompose one round of communication into two stages:

(1) Local update phase: $[ t E + 0  ( t + 1 ) E ]$ indicates that the client has completed the local update.

(2) Server learning phase: $[ ( t + 1 ) E  ( t + 1 ) E + 0 ]$ indicates that the server updates the prototype and distributes it.

Here, we provide detailed mathematical expressions to better represent the process of updating local models. We split each client k’s model $\theta _ { k }$ into a feature extractor $f _ { k } ^ { m }$ parameterized by $\varphi _ { k }$ and a classifier $h _ { k }$ parameterized by $w _ { k }$ . Each client is equipped with $\vert \mathcal { M } _ { k } \vert$ modality specific feature extractors, denoted as $f _ { k } ^ { m } : \mathcal { X } _ { \mathcal { M } _ { k } }  \mathbb { R } ^ { d _ { D } }$ . The client possesses a single classifier, denoted as $\dot { h } _ { k } : \mathbb { R } ^ { d _ { D } }  \mathbb { R } ^ { d _ { C } }$ . So the client loss function can be written as $\mathcal { F } _ { k } = \{ f _ { k } ^ { m } ( \varphi _ { k } ^ { m } ) \} _ { m } \circ h _ { k } ( w _ { k } )$ , and sometimes we use $\theta _ { k }$ to represent $\left( \{ \varphi _ { k } ^ { m } \} _ { m } , w _ { k } \right)$ for short. Therefore, the local loss function of client k can be written as:

$$
\mathcal { L } ( \theta _ { k } ; \mathbf { x } , y ) = \mathcal { L } _ { c e } ( \mathcal { F } _ { k } ( \theta _ { k } ; \mathbf { x } ) , y ) + \lambda _ { 1 } \sum _ { m } | | f _ { k } ^ { m } ( \varphi _ { k } ^ { m } ; \mathbf { x } ^ { m } ) - \mathbf { P } _ { G , c } ^ { m } | | _ { 2 } ^ { 2 } + \lambda _ { 2 } | | h _ { k } ( w _ { k } ; f _ { k } ^ { m } \mathbf { x } ^ { m } ) - \mathbf { I } _ { G , c } | | _ { 2 } ^ { 2 } .\tag{21}
$$

It is worth noting that $\lambda _ { 2 }$ serves as the weighting coefficient for the KL divergence term. Since the KL divergence can be approximated by the quadratic Mean Squared Error (MSE) loss via second-order Taylor expansion near the convergence point, we formulate the objective in Eq. (21) using the MSE form to facilitate the theoretical convergence analysis.

Moreover, the global feature prototype $\mathbf { P } _ { G , c } ^ { m }$ is obtained by the server via S rounds of training with the loss function $\mathcal { L } _ { \mathrm { s e r v e r } }$ while the global logit prototype $\mathbf { I } _ { G , c }$ is aggregated at the server. Therefore, the server optimizes the global loss for S rounds can be written as:

$$
\mathcal { L } _ { \mathrm { s e r v e r } } ( \mathbf { P } ) = \mathcal { L } _ { \mathrm { r e c } } + \mathcal { L } _ { \mathrm { c o n } } + \mathcal { L } _ { \mathrm { a l i g n } } = \frac { 1 } { K } \sum _ { k , m , c } \| P _ { k , c } ^ { m } - E _ { k , c } ^ { m } \| ^ { 2 } + \mathcal { L } _ { \mathrm { c o n } } ( \mathbf { P } ) + \mathcal { L } _ { \mathrm { a l i g n } } ( \mathbf { P } , \mathbf { I } ) .\tag{22}
$$

For ease of analysis, we will represent both loss $\mathcal { L } _ { \mathrm { c o n } }$ and $\mathcal { L } _ { \mathrm { a l i g n } }$ together as $\mathcal { L } _ { \mathrm { m } } , \mathrm { i . e . , } \mathcal { L } _ { \mathrm { m } } = \mathcal { L } _ { \mathrm { c o n } } + \mathcal { L } _ { \mathrm { a l i g n } }$

## B.1. Assumption

Assumption B.1 (Lipschitz Smoothness). The k-th client’s local model loss function L is Lipschitz continuous with Lipschitz constant L, and $L > 0$ with $\begin{array} { r } { \mathcal { L } ( 0 ) = 0 . } \end{array}$ , i.e.,

$$
\begin{array} { r } { \| \nabla { \mathcal L } _ { t _ { 1 } } - \nabla { \mathcal L } _ { t _ { 2 } } \| _ { 2 } \leq L \| \theta _ { k , t _ { 1 } } - \theta _ { k , t _ { 2 } } \| _ { 2 } , \quad \forall t _ { 1 } , t _ { 2 } > 0 , \quad k \in \{ 1 , 2 , \ldots , K \} . } \end{array}\tag{23}
$$

which implies the following quadratic bound,

$$
\mathcal { L } _ { t _ { 1 } } - \mathcal { L } _ { t _ { 2 } } \leq \langle \nabla \mathcal { L } _ { t _ { 2 } } , ( \theta _ { k , t 1 } - \theta _ { k , t 2 } ) \rangle + \frac { L } { 2 } | | \theta _ { k , t 1 } - \theta _ { k , t 2 } | | _ { 2 } ^ { 2 } , \quad \forall t _ { 1 } , t _ { 2 } > 0 , \quad k \in \{ 1 , 2 , \ldots , K \} .\tag{24}
$$

Assumption B.2 (Unbiased Gradient and Bounded Variance). The random gradient $g _ { k , t } = \nabla \mathcal { L } _ { t } \left( \theta _ { k , t } ; B _ { k , t } \right)$ of each client’s local model is unbiased, where B is a batch of local data, i.e.,

$$
\mathbb { E } _ { \mathcal { B } _ { k , t } \subseteq N _ { k } } \left[ g _ { k , t } \right] = \nabla \mathcal { L } ( \theta _ { k , t } ) = \nabla \mathcal { L } _ { t } , \quad \forall k \in \{ 1 , 2 , \dots , K \} ,\tag{25}
$$

and the variance of random gradient ${ g } _ { k , t }$ is bounded by:

$$
\mathbb { E } _ { \mathcal { B } _ { k , t } \subseteq N _ { k } } \left[ \| \nabla \mathcal { L } _ { t } \left( \theta _ { k , t } ; \mathcal { B } _ { k , t } \right) - \nabla \mathcal { L } _ { t } \left( \theta _ { k , t } \right) \| _ { 2 } ^ { 2 } \right] \leq \sigma ^ { 2 } , \quad \forall k \in \{ 1 , 2 , \dots , K \} .\tag{26}
$$

Assumption B.3 (Bounded Gradients). The expectation of the stochastic gradient is bounded by $G _ { c } \mathbf { : }$

$$
\mathbb { E } \left[ | | \nabla \mathcal { L } _ { k } | | ^ { 2 } \right] \leq G _ { c } ^ { 2 } , \quad \forall k \in \{ 1 , 2 , \ldots , K \} .\tag{27}
$$

Assumption B.4 (Lipschitz Continuity). For client k, feature extraction $f _ { k } ^ { m } ( \varphi _ { k } ^ { m } )$ and classifier $h _ { k } ( w _ { k } )$ are Lipschitz continuous with Lipschitz constant $L _ { r }$ , and $L _ { r } > 0 ;$

$$
\begin{array} { r } { \| f _ { k } ^ { m } ( \varphi _ { k , t _ { 1 } } ^ { m } ) - f _ { k } ^ { m } ( \varphi _ { k , t _ { 2 } } ^ { m } ) \| _ { 2 } \leq L _ { r } \| \varphi _ { k , t _ { 1 } } ^ { m } - \varphi _ { k , t _ { 2 } } ^ { m } \| _ { 2 } , } \\ { \| h _ { k } ( w _ { k , t _ { 1 } } ) - \nabla h _ { k } ( w _ { k , t _ { 2 } } ) \| _ { 2 } \leq L _ { r } \| w _ { k , t _ { 1 } } - w _ { k , t _ { 2 } } \| _ { 2 } . } \end{array}\tag{28}
$$

Assumption B.5 (Bounded Server Optimization Drift). The server objective function $\mathcal { L } _ { \mathrm { s e r v e r } }$ is $L _ { s }$ -smooth, and the expectation of the stochastic gradient is bounded, i.e.,

server Lipschitz Smoothness: $\begin{array} { r } { \| \nabla \mathcal { L } _ { \mathrm { s e r v e r } , t _ { 1 } } - \nabla \mathcal { L } _ { \mathrm { s e r v e r } , t _ { 2 } } \| _ { 2 } \leq L _ { s } \| \theta _ { k , t _ { 1 } } - \theta _ { k , t _ { 2 } } \| _ { 2 } , } \end{array}$

gradient bounded: $\mathbb { E } \left\lceil | | \nabla { \mathcal { L } } _ { \mathrm { m } } | | ^ { 2 } \right\rceil \leq G _ { s } ^ { 2 }$ , and $\mathbb { E } \left[ | | \nabla { \mathcal { L } } _ { \mathrm { s e r v e r } } | | ^ { 2 } \right] \leq \delta _ { s } ^ { 2 }$

## B.2. Lemmas & Theorems

Based on the above assumptions, MFedPBA uses a prototype-based approach for local training updates, Lemma B.6 derived by Tan et al., (Tan et al., 2022a) still holds.

Lemma B.6. Let Assumption B.1 and B.2 hold. From the beginning of communication round $t + 1$ to the last local update step, the lossfunction ofan arbitrary client can be bounded as:

$$
\mathbb { E } [ \mathcal { L } _ { ( t + 1 ) E } ] \le \mathcal { L } _ { t E + 0 } - ( \eta _ { c } - \frac { L \eta _ { c } ^ { 2 } } { 2 } ) \sum _ { e = 0 } ^ { E } \| \nabla \mathcal { L } _ { t E + e } \| _ { 2 } ^ { 2 } + \frac { L E \eta _ { c } ^ { 2 } } { 2 } \sigma ^ { 2 } .\tag{29}
$$

Lemma B.7. Let Assumption B.3, B.4 and B.5 hold. After feature prototype learning and logit prototype aggregation are completed on the server, the lossfunction ofany client can be constrained asfollows:

$$
\begin{array} { r } { \mathbb { E } [ \mathcal { L } _ { ( t + 1 ) E + 0 } ] \leq \mathcal { L } _ { ( t + 1 ) E } + ( M \lambda _ { 1 } + \lambda _ { 2 } ) L _ { r } E \eta _ { c } G _ { c } + 2 \lambda _ { 1 } ( \delta _ { s } + M G _ { s } ) . } \end{array}\tag{30}
$$

Theorem B.8 (One-round deviation). Based on the above assumptions, using $\Gamma _ { \mathrm { d } } = ( M \lambda _ { 1 } + \lambda _ { 2 } ) L _ { r } G _ { c }$ and $\Gamma _ { \mathrm { s } } = 2 \lambda _ { 1 } ( \delta _ { s } +$ $M G _ { s } )$ represent the drift constant and server bias constant. The expectation ofthe loss ofan arbitrary client’s local model before the start ofa round oflocal iteration satisfies:

$$
\mathbb { E } [ \mathcal { L } _ { ( t + 1 ) E + 0 } ] \leq \mathcal { L } _ { t E + 0 } - ( \eta _ { c } - \frac { L \eta _ { c } ^ { 2 } } { 2 } ) \sum _ { e = 0 } ^ { E } \| \nabla \mathcal { L } _ { t E + e } \| _ { 2 } ^ { 2 } + \frac { L E \eta _ { c } ^ { 2 } } { 2 } \sigma ^ { 2 } + \Gamma _ { d } E \eta _ { c } + \Gamma _ { s } .\tag{31}
$$

Theorem B.9 (Non-convex convergence rate of MFedPBA). Based on the above assumptions, for an arbitrary client and any $\epsilon > 0$ , thefollowing inequality holds:

$$
\begin{array} { r l } { \displaystyle \frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \sum _ { e = 0 } ^ { E } \mathbb { E } \left[ \| \mathcal { L } _ { t E + e } \| _ { 2 } ^ { 2 } \right] \leq \frac { 2 ( \mathcal { L } _ { t = 1 } - \mathcal { L } ^ { * } ) } { T \eta _ { c } ( 2 - L \eta _ { c } ) } + \frac { L E \eta _ { c } ^ { 2 } \sigma ^ { 2 } + 2 ( \Gamma _ { d } E \eta _ { c } + \Gamma _ { s } ) } { 2 \eta _ { c } - L \eta _ { c } ^ { 2 } } , } & { } \\ { \leq \epsilon . } & { } \\ { s . t . \quad } & { \eta _ { c } < \operatorname* { m i n } \left\{ \frac { 2 } { L } , \frac { ( \epsilon - \Gamma _ { \mathrm { d } } E ) + \sqrt { ( \epsilon - \Gamma _ { \mathrm { d } } E ) ^ { 2 } - 2 L ( \epsilon + E \sigma ^ { 2 } ) \Gamma _ { \mathrm { s } } } } { L ( \epsilon + E \sigma ^ { 2 } ) } \right\} . } \end{array}\tag{32}
$$

## B.3. Proof of Lemma 5.1

Lemma 5.1. Let Assumption B.1 and B.2 hold. From the beginning of communication round $t + 1$ to the last local update step, the loss function of an arbitrary client can be bounded as:

$$
\mathbb { E } [ \mathcal { L } _ { ( t + 1 ) E } ] \le \mathcal { L } _ { t E + 0 } - ( \eta _ { c } - \frac { L \eta _ { c } ^ { 2 } } { 2 } ) \sum _ { e = 0 } ^ { E } \| \nabla \mathcal { L } _ { t E + e } \| _ { 2 } ^ { 2 } + \frac { L E \eta _ { c } ^ { 2 } } { 2 } \sigma ^ { 2 } .
$$

Proof. For arbitrary clients, we have $\theta _ { t + 1 } = \theta _ { t } - \eta _ { c } g _ { t }$ <sub>t</sub>, then

$$
\begin{array} { r l } & { \mathcal { L } _ { t E + 1 } \leq \mathcal { L } _ { t E + 0 } + \langle \nabla \mathcal { L } _ { t E + 0 } , ( \theta _ { t E + 1 } - \theta _ { t E + 0 } ) \rangle + \displaystyle \frac { L } { 2 } \| \theta _ { t E + 1 } - \theta _ { t E + 0 } \| _ { 2 } ^ { 2 } } \\ & { \quad \quad \quad \quad = \mathcal { L } _ { t E + 0 } - \eta _ { c } \langle \nabla \mathcal { L } _ { t E + 0 } , g _ { t E + 0 } \rangle + \displaystyle \frac { L } { 2 } \| \eta _ { c } g _ { t E + 0 } \| _ { 2 } ^ { 2 } . } \end{array}\tag{33}
$$

Taking expectation of both sides of the above equation on the random variable B, we have

$$
\begin{array} { r l } & { \mathbb { E } [ \mathcal { L } _ { t E + 1 } ] \leq \mathcal { L } _ { t E + 0 } - \eta _ { c } \mathbb { E } [ \langle \nabla \mathcal { L } _ { t E + 0 } , g _ { t E + 0 } \rangle ] + \frac { L \eta _ { c } ^ { 2 } } { 2 } \mathbb { E } [ \| g _ { t E + 0 } \| _ { 2 } ^ { 2 } ] } \\ & { \qquad = \mathcal { L } _ { t E + 0 } - \eta _ { c } \| \nabla \mathcal { L } _ { t E + 0 } \| _ { 2 } ^ { 2 } + \frac { L \eta _ { c } ^ { 2 } } { 2 } \mathbb { E } [ \| g _ { t , t E + 0 } \| _ { 2 } ^ { 2 } ] } \\ & { \qquad \leq \mathcal { L } _ { t E + 0 } - \eta _ { c } \| \nabla \mathcal { L } _ { t E + 0 } \| _ { 2 } ^ { 2 } + \frac { L \eta _ { c } ^ { 2 } } { 2 } ( \| \nabla \mathcal { L } _ { t E + 0 } \| _ { 2 } ^ { 2 } + \operatorname { V a r } ( g _ { k , t E + 0 } ) ) } \\ & { \qquad = \mathcal { L } _ { t E + 0 } - ( \eta _ { c } - \frac { L \eta _ { c } ^ { 2 } } { 2 } ) \| \nabla \mathcal { L } _ { t E + 0 } \| _ { 2 } ^ { 2 } + \frac { L \eta _ { c } ^ { 2 } } { 2 } \operatorname { V a r } ( g _ { k , t E + 0 } ) } \\ & { \qquad \leq \mathcal { L } _ { t E + 0 } - ( \eta _ { c } - \frac { L \eta _ { c } ^ { 2 } } { 2 } ) \| \nabla \mathcal { L } _ { t E + 0 } \| _ { 2 } ^ { 2 } + \frac { L \eta _ { c } ^ { 2 } } { 2 } \sigma ^ { 2 } , } \end{array}\tag{34}
$$

where $\operatorname { V a r } ( x ) = \mathbb { E } [ x ^ { 2 } ] - ( \mathbb { E } [ x ] ) ^ { 2 }$ . Take expectation of θ on both sides. Then, by telescoping of E steps, we have,

$$
\mathbb { E } [ \mathcal { L } _ { ( t + 1 ) E } ] \le \mathcal { L } _ { t E + 0 } - ( \eta _ { c } - \frac { L \eta _ { c } ^ { 2 } } { 2 } ) \sum _ { e = 0 } ^ { E } \| \nabla \mathcal { L } _ { t E + e } \| _ { 2 } ^ { 2 } + \frac { L E \eta _ { c } ^ { 2 } } { 2 } \sigma ^ { 2 } ,\tag{35}
$$

which completes the proof.

## B.4. Proof of Lemma 5.2

Lemma 5.2. Let Assumption B.3, B.4 and B.5 hold. After feature prototype learning and logit prototype aggregation are completed on the server, the loss function of any client can be constrained as follows:

$$
\begin{array} { r } { \mathbb { E } [ \mathcal { L } _ { ( t + 1 ) E + 0 } ] \leq \mathcal { L } _ { ( t + 1 ) E } + ( M \lambda _ { 1 } + \lambda _ { 2 } ) L _ { r } E \eta _ { c } G _ { c } + 2 \lambda _ { 1 } ( \delta _ { s } + M G _ { s } ) . } \end{array}
$$

Proof. We define the temporal states of a client. Time $( t + 1 ) E$ denotes the moment when the client has just completed E local training epochs in the t-th round but has not yet received the new prototypes from the server. Time $( t + 1 ) E + 0$ denotes the moment when the client receives the updated prototypes dispatched by the server.

First, we have: $\mathcal { L } _ { ( t + 1 ) E + 0 } = \mathcal { L } _ { ( t + 1 ) E } + \mathcal { L } _ { ( t + 1 ) E + 0 } - \mathcal { L } _ { ( t + 1 ) E }$ . Therefore, our main goal is to derive an upper bound for the difference in loss before and after aggregation:

$$
\Delta \mathcal { L } = \mathbb { E } [ \mathcal { L } _ { ( t + 1 ) E + 0 } - \mathcal { L } _ { ( t + 1 ) E } ] .\tag{36}
$$

Since the model parameter θ does not change during aggregation, the cross-entropy loss $\mathcal { L } _ { \mathrm { c e } }$ cancels out. The difference only comes from the changes in the global prototypes in the regularization term. Thus we have:

$$
\begin{array} { l } { { \displaystyle { \mathcal { L } } _ { ( t + 1 ) E + 0 } - { \mathcal { L } } _ { ( t + 1 ) E } = \lambda _ { 1 } \sum _ { m } \left( | | f _ { k } ^ { m } ( \phi _ { k , ( t + 1 ) E } ^ { m } ) - { \bf P } _ { G , t + 2 } ^ { m } | | - | | f _ { k } ^ { m } ( \phi _ { k , ( t + 1 ) E } ^ { m } ) - { \bf P } _ { G , t + 1 } ^ { m } | | \right) } } \\ { { \displaystyle ~ + ~ \lambda _ { 2 } \left( | | h _ { k } ( w _ { k , ( t + 1 ) E } ) - { \bf I } _ { G , t + 2 } | | - | | h _ { k } ( w _ { k , ( t + 1 ) E } ) - { \bf I } _ { G , t + 1 } | | \right) . } } \end{array}\tag{37}
$$

Using the reverse triangle inequality $\| a - b \| _ { 2 } - \| a - c \| _ { 2 } \leq \| b - c \| _ { 2 }$ , we can bound the above expression as follows:

$$
\mathcal { L } _ { ( t + 1 ) E + 0 } - \mathcal { L } _ { ( t + 1 ) E } = \underbrace { \lambda _ { 1 } \sum _ { m } \Vert \mathbf { P } _ { G , t + 2 } ^ { m } - \mathbf { P } _ { G , t + 1 } ^ { m } \Vert } _ { A } + \underbrace { \lambda _ { 2 } \Vert \mathbf { I } _ { G , t + 2 } - \mathbf { I } _ { G , t + 1 } \Vert } _ { B } .\tag{38}
$$

Next, we define the changes in Part A and Part B separately.

Since the logical value prototype of Part B uses weighted aggregation, we will begin our analysis with Part B. From Eq. (6), we can see that:

$$
\mathbf { I } _ { G , t + 2 } = \sum _ { k = 1 } ^ { K } q _ { k } I _ { k , ( t + 1 ) E } , \quad \mathbf { I } _ { G , t + 1 } = \sum _ { k = 1 } ^ { K } q _ { k } I _ { k , t E } .\tag{39}
$$

Therefore, we have:

$$
\begin{array} { r l } { \mathbb { I } _ { \xi _ { i } \times \xi _ { i + 1 } } = \mathbb { I } _ { \xi _ { i } \times \xi _ { i + 1 } } = \Bigg | } & { \Bigg | \displaystyle \sum _ { j = 1 } ^ { K } g \mathbb { I } _ { \xi _ { i } \xi _ { j } \xi _ { j + 1 } \psi _ { i } } - \displaystyle \sum _ { j = 1 } ^ { K } g \mathbb { I } _ { \xi _ { i } \xi _ { j + 1 } \psi _ { i } } \Bigg | _ { \xi _ { i } = \xi _ { i } } \Bigg | \displaystyle \sum _ { j = 1 } ^ { K } g \mathbb { I } _ { \xi _ { i } \xi _ { j + 1 } \psi _ { i } \xi _ { j + 1 } \psi _ { i } } - \mathbb { I } _ { \xi _ { i } \xi _ { j + 1 } \psi _ { i } } \Bigg | \Bigg | _ { \xi _ { i } = \xi _ { i } } } \\ & { \leq \Bigg | \displaystyle \sum _ { j = 1 } ^ { K } \frac { \mathbb { I } _ { \xi _ { i } } } { N _ { \xi _ { i } } } \displaystyle \sum _ { j = 1 } ^ { K } \mathbb { I } _ { \xi _ { i } \xi _ { j + 1 } \psi _ { i } \xi _ { j + 1 } \psi _ { i } \xi _ { j + 1 } \psi _ { i } } \frac { \mathbb { I } _ { \xi _ { i } } } { \mathbb { I } _ { \xi _ { i } \xi _ { j + 1 } \psi _ { i } } } - \mathbb { I } _ { \xi _ { i } \xi _ { j } \psi _ { i } \xi _ { i + 1 } \psi _ { i } \xi _ { j + 1 } \psi _ { i } } \Bigg | \Bigg | _ { \xi _ { i } } , } \\ &  \times \displaystyle \sum _ { j = 1 } ^ { K } \mu \displaystyle \sum _ { j = 1 } ^ { K } \displaystyle \sum _ { j = 1 } ^ { K } | \mathbb { I } _ { \xi _ { i } \xi _ { j + 1 } \psi _ { i } \xi _ { j + 1 } \psi _ { i } } \mathbb { I } _ { \xi _ { j } \xi _ { j } \psi _ { i } } \mathbb { I } _  \xi _ { j } \psi _ { i } \end{array}\tag{40}
$$

Take expectations on both sides, since $\sum q _ { k } = 1$ , then:

$$
\lambda _ { 2 } \| \mathbf { I } _ { G , t + 2 } - \mathbf { I } _ { G , t + 1 } \| \leq \lambda _ { 2 } L _ { r } \eta _ { c } \sum _ { k = 1 } ^ { K } q _ { k } \sum _ { e = 0 } ^ { E - 1 } \| g _ { k , t E + e } \| _ { 2 } \leq \lambda _ { 2 } L _ { r } E \eta _ { c } G _ { c } .\tag{41}
$$

The derivation for Part A is complete.

Since the global feature prototype is aggregated using non-convex SGD, the server prototype $\mathbf { P } _ { G } ^ { m }$ is the result of the client uploading and optimizing $E _ { k } ^ { m }$ for S rounds. Based on Assumption B.5, we introduce the aggregated mean as an intermediate variable.

Let $\begin{array} { r } { \mathbf { E } _ { t + 2 } = \sum q _ { k } E _ { k , t + 2 } } \end{array}$ be the geometric center of the currently uploaded feature prototypes, and $\begin{array} { r } { \mathbf { E } _ { t + 1 } = \sum q _ { k } E _ { k , t + 1 } } \end{array}$ be the geometric center of the features uploaded in the previous round. Here, we omit the modality superscript m for ease of analysis. Therefore, for part A, we have:

$$
\begin{array} { r l } & { \| \mathbf { P } _ { G , t + 2 } - \mathbf { P } _ { G , t + 1 } \| = \| \mathbf { P } _ { G , t + 2 } - \mathbf { E } _ { t + 2 } + \mathbf { E } _ { t + 2 } - \mathbf { E } _ { t + 1 } + \mathbf { E } _ { t + 1 } - \mathbf { P } _ { G , t + 1 } \| } \\ & { \qquad \leq \underbrace { \| \mathbf { P } _ { G , t + 2 } - \mathbf { E } _ { t + 2 } \| } _ { A 1 } + \underbrace { \| \mathbf { E } _ { t + 2 } - \mathbf { E } _ { t + 1 } \| } _ { A 2 } + \underbrace { \| \mathbf { E } _ { t + 1 } - \mathbf { P } _ { G , t + 1 } \| } _ { A 3 } . } \end{array}\tag{42}
$$

We found that Part A2 is completely consistent with the derivation of the logical value prototype, and the change in the mean is limited by the drift of the client’s local parameters. Therefore, we have:

$$
\| \mathbf { E } _ { t + 2 } - \mathbf { E } _ { t + 1 } \| \leq L _ { r } E \eta _ { c } G _ { c } .\tag{43}
$$

By solving for the gradient of Eq. (22), we obtain:

$$
\begin{array} { l } { \displaystyle \nabla \mathcal { L } _ { s e r v e r } \big ( \mathbf { P } _ { G } ^ { m } \big ) = \frac { 1 } { 2 K } \sum _ { k = 1 } ^ { K } \sum _ { m = 1 } ^ { M } 2 ( P _ { k } ^ { m } - E _ { k } ^ { m } ) + \sum _ { m = 1 } ^ { M } \nabla \mathcal { L } _ { m } \big ( \mathbf { P } _ { G } ^ { m } \big ) } \\ { \displaystyle = M \big ( \mathbf { P } _ { G } - \mathbf { E } _ { a v g } \big ) + M \nabla \mathcal { L } _ { m } \big ( \mathbf { P } _ { G } ^ { m } \big ) . } \end{array}\tag{44}
$$

Through algebraic manipulation, we obtain:

$$
\mathbf { P } _ { G } - \mathbf { E } _ { a v g } = \frac { 1 } { M } \nabla \mathcal { L } _ { \mathrm { s e r v e r } } ( \mathbf { P } _ { G } ^ { m } ) - \nabla \mathcal { L } _ { m } \big ( \mathbf { P } _ { G } ^ { m } \big ) .\tag{45}
$$

By applying the L2-norm and the triangle inequality to the above expression, we obtain:

$$
\| \mathbf { P } _ { G } - \mathbf { E } _ { a v g } \| _ { 2 } \leq \| \frac { 1 } { M } \nabla \mathcal { L } _ { \mathrm { s e r v e r } } ( \mathbf { P } _ { G } ^ { m } ) \| _ { 2 } + \| \nabla \mathcal { L } _ { m } ( \mathbf { P } _ { G } ^ { m } ) \| _ { 2 } .\tag{46}
$$

According to assumption B.5, therefore we have:

$$
\begin{array} { r l } & { \displaystyle \mathbb { E } [ \| \mathbf { P } _ { G } - \mathbf { E } _ { a v g } \| _ { 2 } ] \leq \mathbb { E } [ \| \frac { 1 } { M } \nabla \mathcal { L } _ { \mathrm { s e r v e r } } ( \mathbf { P } _ { G } ^ { m } ) \| _ { 2 } ] + \mathbb { E } [ \| \nabla \mathcal { L } _ { m } ( \mathbf { P } _ { G } ^ { m } ) \| _ { 2 } ] } \\ & { \quad \quad \quad \leq \displaystyle \frac { \delta _ { s } } { M } + G _ { s } . } \end{array}\tag{47}
$$

Accordingly, the derivation process for A1 and A3 is the same as described above. Therefore, considering that there are M modes in total, we can integrate the above process into Eq. (42) to obtain:

$$
\begin{array} { r l } & { \| \mathbf { P } _ { G , t + 2 } - \mathbf { P } _ { G , t + 1 } \| \leq \underbrace { \| \mathbf { P } _ { G , t + 2 } - \mathbf { E } _ { t + 2 } \| } _ { A 1 } + \underbrace { \| \mathbf { E } _ { t + 2 } - \mathbf { E } _ { t + 1 } \| } _ { A 2 } + \underbrace { \| \mathbf { E } _ { t + 1 } - \mathbf { P } _ { G , t + 1 } \| } _ { A 3 } . } \\ & { \qquad \leq 2 M ( \frac { \delta _ { s } } { M } + G _ { s } ) + M L _ { r } E \eta _ { c } G _ { c } = 2 \delta _ { s } + 2 M G _ { s } + M L _ { r } E \eta _ { c } G _ { c } . } \end{array}\tag{48}
$$

Based on the derivation results from Eq. (48) and Eq. (41), substituting them into Eq. (38), and taking expectation of both sides:

$$
\begin{array} { r l } & { \mathbb { E } [ \mathcal { L } _ { ( t + 1 ) E + 0 } ] - \mathbb { E } [ \mathcal { L } _ { ( t + 1 ) E } ] \leq \lambda _ { 1 } ( 2 \delta _ { s } + 2 M G _ { s } + M L _ { r } E \eta _ { c } G _ { c } ) + \lambda _ { 2 } L _ { r } E \eta _ { c } G _ { c } } \\ & { \qquad = ( M \lambda _ { 1 } + \lambda _ { 2 } ) L _ { r } E \eta _ { c } G _ { c } + 2 \lambda _ { 1 } ( \delta _ { s } + M G _ { s } ) , } \end{array}\tag{49}
$$

which completes the proof.

## B.5. Proof of Theorem 5.3

Theorem 5.3 Based on the above assumptions, using $\Gamma _ { \mathrm { d } } = ( M \lambda _ { 1 } + \lambda _ { 2 } ) L _ { r } G _ { c }$ and $\Gamma _ { \mathrm { s } } = 2 \lambda _ { 1 } ( \delta _ { s } + M G _ { s } )$ represent the drift constant and server bias constant. The expectation of the loss of an arbitrary client’s local model before the start of a round of local iteration satisfies:

$$
\begin{array} { r l } & { \mathbb { E } [ \mathcal { L } _ { ( t + 1 ) E + 0 } ] \leq \mathcal { L } _ { t E + 0 } - ( \eta _ { c } - \frac { L \eta _ { c } ^ { 2 } } { 2 } ) \displaystyle \sum _ { e = 0 } ^ { E } \| \nabla \mathcal { L } _ { t E + e } \| _ { 2 } ^ { 2 } + \frac { L E \eta _ { c } ^ { 2 } } { 2 } \sigma ^ { 2 } } \\ & { \qquad + \Gamma _ { d } E \eta _ { c } + \Gamma _ { s } . } \end{array}
$$

Proof. According to the expectation equation, the change in one communication round consists of two phases: the server-side phase and the client-side phase. The total change can be expressed as:

$$
\begin{array} { r } { \mathbb { E } [ \mathcal { L } _ { ( t + 1 ) E + 0 } ] - \mathcal { L } _ { t E + 0 } \leq \underbrace { \left( \mathbb { E } [ \mathcal { L } _ { ( t + 1 ) E + 0 } ] - \mathbb { E } [ \mathcal { L } _ { ( t + 1 ) E } ] \right) } _ { \mathrm { s e r v e r } } + \underbrace { \left( \mathbb { E } [ \mathcal { L } _ { ( t + 1 ) E } ] - \mathbb { E } [ \mathcal { L } _ { t E + 0 } ] \right) } _ { \mathrm { c i e n t } } . } \end{array}\tag{50}
$$

Substituting Lemma B.6 into the second term on the right-hand side of Lemma B.7, we have:

$$
\begin{array} { r l } & { \mathbb { E } [ \mathcal { L } _ { ( t + 1 ) E + 0 } ] \leq \mathbb { E } [ \mathcal { L } _ { ( t + 1 ) E } ] + ( M \lambda _ { 1 } + \lambda _ { 2 } ) L _ { r } E \eta _ { c } G _ { c } + 2 \lambda _ { 1 } ( \delta _ { s } + M G _ { s } ) } \\ & { \qquad \leq \mathcal { L } _ { t E + 0 } - ( \eta _ { c } - \frac { L \eta _ { c } ^ { 2 } } { 2 } ) \displaystyle \sum _ { e = 0 } ^ { E } \| \nabla \mathcal { L } _ { t E + e } \| _ { 2 } ^ { 2 } + \frac { L E \eta _ { c } ^ { 2 } } { 2 } \sigma ^ { 2 } } \\ & { \qquad + \left( M \lambda _ { 1 } + \lambda _ { 2 } \right) L _ { r } E \eta _ { c } G _ { c } + 2 \lambda _ { 1 } ( \delta _ { s } + M G _ { s } ) . } \end{array}\tag{51}
$$

To simplify the notation, we let $\Gamma _ { d } = ( M \lambda _ { 1 } + \lambda _ { 2 } ) L _ { r } G _ { c }$ represent the drift constant, and $\Gamma _ { s } = 2 \lambda _ { 1 } ( \delta _ { s } + M G _ { s } )$ represent the server bias constant, then we have

$$
\begin{array} { r l } & { \mathbb { E } [ \mathcal { L } _ { ( t + 1 ) E + 0 } ] \leq \mathcal { L } _ { t E + 0 } - ( \eta _ { c } - \frac { L \eta _ { c } ^ { 2 } } { 2 } ) \displaystyle \sum _ { e = 0 } ^ { E } \| \nabla \mathcal { L } _ { t E + e } \| _ { 2 } ^ { 2 } + \frac { L E \eta _ { c } ^ { 2 } } { 2 } \sigma ^ { 2 } } \\ & { \qquad + \Gamma _ { d } E \eta _ { c } + \Gamma _ { s } , } \end{array}\tag{52}
$$

which completes the proof.

## B.6. Proof of Theorem 5.4

Theorem 5.4 Based on the above assumptions, for an arbitrary client and any $\epsilon > 0$ , the following inequality holds:

$$
\begin{array} { r l } { \displaystyle \frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \sum _ { e = 0 } ^ { E } \mathbb { E } \left[ \| \mathcal { L } _ { t E + e } \| _ { 2 } ^ { 2 } \right] \leq \frac { 2 ( \mathcal { L } _ { t = 1 } - \mathcal { L } ^ { * } ) } { T \eta _ { c } ( 2 - L \eta _ { c } ) } + \frac { L E \eta _ { c } ^ { 2 } \sigma ^ { 2 } + 2 ( \Gamma _ { d } E \eta _ { c } + \Gamma _ { s } ) } { 2 \eta _ { c } - L \eta _ { c } ^ { 2 } } } & { } \\ { \leq \epsilon . } & { } \\ { s . t . \quad } & { \eta _ { c } < \operatorname* { m i n } \left\{ \displaystyle \frac { 2 } { L } , \frac { ( \epsilon - \Gamma _ { \mathrm { d } } E ) + \sqrt { ( \epsilon - \Gamma _ { \mathrm { d } } E ) ^ { 2 } - 2 L ( \epsilon + E \sigma ^ { 2 } ) \Gamma _ { \mathrm { s } } } } { L ( \epsilon + E \sigma ^ { 2 } ) } \right\} . } \end{array}
$$

Proof. Transform the form of Theorem B.8 into

$$
\sum _ { e = 0 } ^ { E } \left\| \mathcal { L } _ { t E + e } \right\| _ { 2 } ^ { 2 } \leq \frac { \mathcal { L } _ { t E + 0 } - \mathbb { E } \left[ \mathcal { L } _ { ( t + 1 ) E + 0 } \right] + \frac { L E \eta _ { c } ^ { 2 } } { 2 } \sigma ^ { 2 } + \Gamma _ { d } E \eta _ { c } + \Gamma _ { s } } { \eta _ { c } - \frac { L \eta _ { c } ^ { 2 } } { 2 } } .\tag{53}
$$

Take expectations of model $\theta$ on both sides, we have:

$$
\sum _ { e = 0 } ^ { E } \mathbb { E } \left[ \| \mathcal { L } _ { t E + e } \| _ { 2 } ^ { 2 } \right] \leq \frac { \mathbb { E } \left[ \mathcal { L } _ { t E + 0 } \right] - \mathbb { E } \left[ \mathcal { L } _ { ( t + 1 ) E + 0 } \right] + \frac { L E \eta _ { c } ^ { 2 } } { 2 } \sigma ^ { 2 } + \Gamma _ { d } E \eta _ { c } + \Gamma _ { s } } { \eta _ { c } - \frac { L \eta _ { c } ^ { 2 } } { 2 } } .\tag{54}
$$

Summing both sides of $\operatorname { E q . }$ (54) over $T$ rounds, since $\begin{array} { r } { \sum _ { t = 1 } ^ { T } \left( \mathbb E \left[ \mathcal L _ { t E + 0 } \right] - \mathbb E \left[ \mathcal L _ { ( t + 1 ) E + 0 } \right] \right) \le \mathcal L _ { t = 0 } - \mathcal L ^ { * } } \end{array}$ , for each round:

$$
\begin{array} { r l } & { \frac { 1 } { T } \displaystyle \sum _ { t = 0 } ^ { T - 1 } \sum _ { \ell = 0 } ^ { E } \mathbb { E } \left[ \| \mathcal { L } _ { t E + \epsilon } \| _ { 2 } ^ { 2 } \right] \leq \frac { \frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \left( \mathbb { E } \left[ \mathcal { L } _ { t E + 0 } \right] - \mathbb { E } \left[ \mathcal { L } _ { ( t + 1 ) E + 0 } \right] \right) + \frac { L E \eta _ { c } ^ { 2 } } { 2 } \sigma ^ { 2 } + \Gamma _ { d } E \eta _ { c } + \Gamma _ { s } } { \eta _ { c } - \frac { K \eta _ { c } ^ { 2 } } { 2 } } } \\ & { \phantom { \frac { 1 } { T } \displaystyle \sum _ { t = 0 } ^ { T - 1 } \sum _ { \ell = 0 } ^ { E } \frac { 1 } { \eta _ { c } } } \leq \frac { L E \eta _ { c } ^ { 2 } \sigma ^ { 2 } } { 2 } + \Gamma _ { d } E \eta _ { c } + \Gamma _ { s } } \\ & { \phantom { \frac { 1 } { T } \displaystyle \sum _ { t = 0 } ^ { T - 1 } \sum _ { \ell = 0 } ^ { E } \frac { L E \eta _ { c } ^ { 2 } } { 2 } } } \\ &  \phantom { \frac { 1 } { T } \displaystyle \sum _ { t = 0 } ^ { T - 1 } \sum _ { \ell = 0 } ^ { E } \frac { L E \eta _ { c } ^ { 2 } \sigma ^ { 2 } + 2 T \left( \Gamma _ { d } E \eta _ { c } + \Gamma _ { s } \right) } { T \left( 2 \eta _ { c } - L \eta _ { c } ^ { 2 } \right) } } \\ &  \phantom { \frac { 1 } { T } \displaystyle \sum _ { t = 0 } ^ { 2 \left( \mathcal { L } _ { t = 0 } - \mathcal { L } ^ { * } \right) } + \frac { L E \eta _ { c } ^ { 2 } \sigma ^ { 2 } + 2 \left( \Gamma _ { d } E \eta _ { c } + \Gamma _ { s } \right) } { 2 \eta _ { c } - L \eta _ { c } ^ { 2 } } . } \end{array}\tag{55}
$$

Given any $\epsilon > 0$ the above equation satisfies

$$
\frac { 2 \left( \mathcal { L } _ { t = 0 } - \mathcal { L } ^ { * } \right) } { T \eta _ { c } \left( 2 - L \eta _ { c } \right) } + \frac { L E \eta _ { c } ^ { 2 } \sigma ^ { 2 } + 2 ( \Gamma _ { d } E \eta _ { c } + \Gamma _ { s } ) } { 2 \eta _ { c } - L \eta _ { c } ^ { 2 } } \leq \epsilon .\tag{56}
$$

Then, we can obtain:

$$
T \geq \frac { 2 \left( \mathcal { L } _ { t = 0 } - \mathcal { L } ^ { * } \right) } { \eta _ { c } \epsilon \left( 2 - L \eta _ { c } \right) - \eta _ { c } E \left( L \eta _ { c } \sigma ^ { 2 } + 2 \Gamma _ { d } \eta _ { c } \right) - 2 \Gamma _ { s } } .\tag{57}
$$

Since $T > 0 , \mathcal { L } _ { t = 0 } - \mathcal { L } ^ { * } > 0$ , we can further derive:

$$
\eta _ { c } \epsilon \left( 2 - L \eta _ { c } \right) - \eta _ { c } E ( L \eta _ { c } \sigma ^ { 2 } + 2 \Gamma _ { d } \eta _ { c } ) - 2 \Gamma _ { s } > 0 ,\tag{58}
$$

Since Eq. (58) is a quadratic function that opens upwards, we solve it to obtain:

$$
\eta _ { c } < \operatorname * { m i n } \left\{ \frac { 2 } { L } , \frac { ( \epsilon - \Gamma _ { \mathrm { d } } E ) + \sqrt { ( \epsilon - \Gamma _ { \mathrm { d } } E ) ^ { 2 } - 2 L ( \epsilon + E \sigma ^ { 2 } ) \Gamma _ { \mathrm { s } } } } { L ( \epsilon + E \sigma ^ { 2 } ) } \right\} ,\tag{59}
$$

When $\Gamma _ { \mathrm { s } }$ approaches 0, Eq. (59) can be simplified to:

$$
\eta _ { c } < \mathrm { m i n } \left\{ \frac { 2 } { L } , \frac { 2 ( \epsilon - \Gamma _ { \mathrm { d } } E ) } { L ( \epsilon + E \sigma ^ { 2 } ) } \right\} ,\tag{60}
$$

which completes the proof.

## C. Details of the Experimental Setup

## C.1. Dataset Description

This paper focuses on multimodal federated learning under model heterogeneity and imbalanced modality distributions. To evaluate performance in modality-imbalanced scenarios, we select multiple datasets with diverse modalities and classes and construct distributed federated settings, denoted as datasets Caltech101 <sup>5</sup>, Reuters <sup>6</sup>, NUS-WIDE <sup>7</sup>, and Youtube <sup>8</sup>. Detailed descriptions of each dataset are provided below.

The Caltech101 dataset is a widely used benchmark dataset introduced by the California Institute of Technology in 2003 and has been extensively adopted in computer vision, pattern recognition, and machine learning research. The dataset is designed to provide a well-annotated and low-noise image collection for object recognition and image classification tasks. It consists of 102 categories, including 101 object classes and one background category (BACKGROUND Google). To fully exploit the visual information in Caltech101, we adopt a multimodal feature extraction strategy that characterizes images from complementary perspectives. Specifically, we consider the following three modalities:

• I1 (Histogram of Oriented Gradients): A 1984-dimensional Histogram of Oriented Gradients feature, which captures local structural and edge information. Images are partitioned into connected regions, within which gradient orientations are computed and aggregated into histograms. The regional histograms are then concatenated and block-normalized to improve robustness to illumination variations and local deformations.

• I2 (Gist Descriptor): A 512-dimensional Gabor feature designed to model texture and frequency responses. A bank of multi-scale and multi-orientation Gabor filters is applied to each image, followed by average pooling over a 4×4 spatial grid. The pooled responses from all filters and grid cells are concatenated to form the final representation.

• I3 (Local Binary Pattern): A 928-dimensional Local Binary Pattern feature that encodes local texture patterns. For each pixel, binary codes are generated by comparing the center pixel with its neighbors and converted into decimal values. The global image representation is obtained by computing the histogram of LBP codes over the entire image.

The Reuters dataset is a widely adopted benchmark in multi-view learning research. It comprises 18,758 news articles collected by Reuters in 1987, which are categorized into six distinct topics. While the original corpus consists solely of English text, the multi-view variant augments these documents with translations into four additional European languages: French, German, Italian, and Spanish. Consequently, each sample is represented by five distinct views corresponding to these languages. Regarding feature representation, the documents are encoded as Bag-of-Words (BoW) vectors weighted by TF-IDF. The specific feature dimensionality for each view is as follows: 21,531 for English, 24,892 for French, 34,251 for German, 15,506 for Italian, and 11,547 for Spanish.

The NUS-WIDE dataset, released by the National University of Singapore, is a large-scale multi-label image dataset comprising 8 distinct categories. It stands as a milestone benchmark in the fields of web image retrieval, multimodal learning, and semantic annotation research. The dataset provides both visual content and associated textual metadata, including user-contributed tags, titles, and descriptions from Flickr, enabling comprehensive cross-modal analysis. In this work, we extract four complementary feature modalities from NUS-WIDE, each capturing distinct aspects of visual and structural information:

• I1 (Cube Color Histogram): A 64-dimensional color histogram computed in a perceptually robust color space. Images are first transformed from the RGB color space to a color space less sensitive to illumination variations, and color distributions are then quantified to obtain global color representations.

• I2 (Color Correlation Map): A 144-dimensional color correlogram that captures both color statistics and their spatial correlations. This representation models not only the occurrence probability of individual colors but also the co occurrence of identical or different colors at predefined spatial distances.

• I3 (Block-wise Color Moments, BOC): A 225-dimensional block-wise color moments feature that preserves spatial layout information. Each image is uniformly partitioned into a 5×5 grid, and low-order color moments are computed within each block across multiple color channels. The resulting block-level features are concatenated to form the final representation.

• L (Bag-of-Visual-Words): A 500-dimensional bag-of-visual-words representation that treats an image as a document composed of visual words. Local keypoints are first detected and described using local feature descriptors, which are then clustered to construct a visual vocabulary. Each image is subsequently encoded as a histogram over the learned visual words.

The YouTube dataset comprises a large collection of short video clips sourced from the YouTube platform, organized around specific events or actions and spanning 10 distinct categories. As a naturally multimodal data source, it provides synchronized visual, auditory, and temporal information. In this study, we extract six complementary feature modalities to capture diverse aspects of the video content:

• I1 (Spatiotemporal Cuboids): A 2,000-dimensional spatiotemporal interest point descriptor that captures local regions exhibiting significant variations across both spatial and temporal dimensions. This representation models local motion patterns and appearance changes using three-dimensional spatiotemporal cuboids, enabling effective characterization of dynamic visual structures.

• I2 (Optical Flow Histogram): A 1,024-dimensional temporally pooled histogram of oriented gradients representation that encodes global shape and edge information. HOG features are extracted frame-wise and aggregated via tempora pooling operations to obtain a compact global descriptor while preserving spatial structural cues.

• T (Frame-level HOG): A 64-dimensional global optical flow histogram that summarizes pixel-level motion statistics. Optical flow fields are computed between consecutive frames, and the distributions of flow magnitudes and orientations are statistically modeled to form a concise representation of overall motion dynamics.

• A1 (Mel-Frequency Features): A 512-dimensional Mel-frequency cepstral coefficient representation that characterizes the spectral properties of audio signals. The audio stream is segmented into frames and processed through windowing, short-time Fourier transform, Mel-scale filtering, and discrete cosine transform, yielding perceptually robust acoustic features.

• A2 (Volume Streams): A 64-dimensional energy-based descriptor that captures variations in audio intensity over time. Frame-level energy statistics are computed using squared amplitude or root mean square measures and aggregated to reflect global loudness dynamics.

• A3 (Spectrogram-based Features): A 647-dimensional spectrogram-based statistical representation that models the joint time–frequency distribution of the audio signal. Spectrograms are generated via short-time Fourier transform and partitioned along both temporal and frequency axes, with energy statistics aggregated within each region to provide a hierarchical description of audio structure.

To accurately simulate complex data distribution scenarios in heterogeneous multimodal federated learning, this study designs three client modality distribution configurations. The “M2” setup refers to each client randomly selecting two modalities from the complete modality set to form its local dataset; the “M1+” setup requires each client to possess at least one modality, while allowing variations in modality composition across different clients; and the “M1” setup restricts each client to holding only a single modality. The number of clients is equal to the number of modalities. In Figures 9, 10, and 11, we visually present the client modality distribution across four datasets. In these visualizations, each circle represents a client, with the circle’s area proportional to the sample size of the client’s data for the corresponding modality, intuitively illustrating the diversity of modality distribution and the imbalance in data volume under heterogeneous settings.

To evaluate the performance of the algorithm in larger-scale client environments, we scale the number of clients for the four datasets as follows: dataset Caltech101 to 50 clients, dataset Reuters to 100 clients, and datasets NUS-WIDE and YouTube to 20 clients each. The experiments adopt the ”M1+” modality configuration, with a Dirichlet distribution (concentration parameter $\alpha = 0 . 5 )$ used to partition the data among clients in a non-IID manner. The resulting client data distributions for the scaled scenarios are visualized in Figure 12.

![](images/71ef5444d96b1a8c4ea0db6db731f151484425069c919bf9278df9c46302ff4a.jpg)  
(a) Caltech101

![](images/d1819bf8bab810421b4d4beb0ecac287e44ce3778b77612354fb38306e6db879.jpg)  
(b) Reuters

![](images/e93dfcc9f5f7b418e2da63e78da128824b87b2109c02c217ef132e0d5e06fea9.jpg)  
(c) NUS-WIDE

![](images/9c39d18c204885880c569d8fcf7444c7f51a238867d7b5e6fa31f017f6d8783d.jpg)  
(d) Youtube

Figure 9. The distribution of these datasets is set in the M2 scenario. A larger circle means a larger sample size.  
![](images/dff96ff39d5e2b7eef1af771110093746238ead1eebf344edb0c6eeaaffe74d4.jpg)  
(a) Caltech101

![](images/95acd2239c7c6f78d510be2e287437dfe5d47d4ec5bf4f5c9fd144fc8d0d9089.jpg)  
(b) Reuters

![](images/f934a9d55d6e214ef83aeb597ae3a1bfa240444ca93b0c1ca65d11f24666c921.jpg)  
(c) NUS-WIDE

![](images/21f5629c4c5eaa7292fc5327ed0d973d6eecb73c2bb46bec333967d5a9527a93.jpg)  
(d) Youtube  
Figure 10. The distribution of these datasets is set in the M1+ scenario. A larger circle means a larger sample size.

## C.2. Heterogeneous Model.

After the modality-specific feature preprocessing described in Section C.1, we further assign heterogeneous feature extractors to different modalities in order to obtain modality-specific embeddings. Specifically, the model architecture for each modality is stochastically sampled from a candidate pool, as listed in Table 4, thereby simulating realistic model heterogeneity in federated learning. To further enforce architectural diversity, the depth (number of repeating blocks) and width (hidden dimensions) of the networks are also randomized.

![](images/75cd1c38bac249fa4177ef0d2bdd847bd2552620795aa73553420fe92ef46249.jpg)  
(a) Caltech101

![](images/14f6e347ed36f642025faf116b37c19a62c7db09b2a4b8f0e21ce48585783f8d.jpg)  
(b) Reuters

![](images/65cd6dd521d971d3bbd784773ddbbb477bbb06099e1f0953f51b2693043fbc02.jpg)  
(c) NUS-WIDE

![](images/0d4ae88ec450e784b57918ede35da61b563d7fec84c6a6c603ad38ec98063d9b.jpg)  
(d) Youtube  
Figure 11. The distribution of these datasets is set in the M1 scenario. A larger circle means a larger sample size.

Table 4 details the sequential structure of these extractors. The symbol “\*” denotes the number of repeated modules, which takes integer values in the range of [2, 5] in our experiments. The hidden dimension of each model is set as a multiple of 32 and is kept constant or gradually reduced as the number of modules increases, balancing representational capacity and model complexity. This randomized model assignment strategy enables a controlled yet realistic evaluation of multimodal federated learning under heterogeneous architectures.

Table 4. Heterogeneous model architectures for modality-specific feature extraction.
<table><tr><td>Model</td><td>Sequentially Connected Feature Extractors</td><td>Applicable Modality Types</td></tr><tr><td>DNN CNN1D</td><td>[input_dim, *(hidden_dim, Relu), Classifier]</td><td>Image, Language, Audio, Time-series</td></tr><tr><td>CNN2D</td><td>[*(Conv1d, Relu, AdaptiveMaxPool1d), *(hidden_dim, Relu), Classifier] [*(Conv2d, Relu), *(hidden_dim, Relu), Classifier]</td><td>Language, Audio, Time-series Image</td></tr><tr><td>TextCNN</td><td>[*(Conv1d, Relu, AdaptiveMaxPool1d), Concat, *(hidden_dim, Relu), Classifier]</td><td>Language</td></tr><tr><td>Resmodel</td><td>[input_dim, *ResBlock(hidden_dim,BatchNorm1d, Relu), Classifier]</td><td>Audio, Time-series</td></tr></table>

## C.3. Baseline Methods

To ensure fair comparison, we re-implemented all baseline methods within a unified HtFLlib framework (Zhang et al., 2025a) and evaluated them under identical experimental settings. Specifically, we include representative prototype-based federated learning methods FedProto (Tan et al., 2022a), FedTGP (Zhang et al., 2024), and FedPall (Zhang et al., 2025c), as well as state-of-the-art multimodal federated learning approaches Harmony (Ouyang et al., 2023), FedMVP (Che et al., 2024), and FedMobile (Liu et al., 2025), with local training (Local) serving as a reference baseline.

Prior to the main experiments, we conducted a preliminary investigation into local multimodal fusion strategies, comparing feature summation (“sum”) and feature concatenation (“concat”). Results indicated that feature summation (“sum”) yields better performance. Therefore, during local training, for Local, the prototype-based methods, and our proposed MFedPBA, when a client possesses multimodal data, the features extracted by heterogeneous encoders are summed before being sent to the server. In contrast, the multimodal federated learning methods Harmony, FedMVP, and FedMobile follow their origina local learning strategies as described in their respective papers.

Furthermore, we note that no publicly available model-heterogeneous multimodal federated learning methods are directly applicable for comparative evaluation. To establish a reasonable heterogeneous experimental environment, we selected stateof-the-art homogeneous multimodal federated learning methods as baselines and adapted their communication mechanisms by modifying the uploaded parameters from full local models to classifier parameters only, thereby accommodating model-heterogeneous federated learning scenarios.

![](images/a8ff8b50e243048e1525e6528f0ada95df3a6e6735fd360ca6f41050ab72228c.jpg)  
012345678910111213141516171819202122232425262728293031323334353637383940414243444546474849 Client IDs

![](images/39a634a89915745aca78ccce06425cdac13f4b279784c6cf946f1f504f9d399e.jpg)

![](images/5967ded27bc39cb4098cff6954c8166a4d5149bd322bdd50b2e9114ca4d357c6.jpg)

![](images/e4eb0abe149aff21eef7ee6570a40a2c61c0cf3d0f307e99c1153a7e0e086967.jpg)  
Figure 12. In scenarios involving a larger number of clients, the data distribution across these four datasets within the M1+ scenario

## C.4. Parameter Setting Details

The parameters of our method are detailed in Table 5. We ran these experiments on 2 NVIDIA GeForce RTX 4090 GPUs with AMD Ryzen 9 9950X 16-Core Processor (32 threads) for all methods.

## D. Details of the Experimental Result

In the main text, we report the test accuracy of each method across four multimodal datasets under three data partitioning schemes: “M2”, “M1+”, and “M1”. In the appendix, we provide detailed client-level experimental results and performance curves for each data partitioning scheme.

Regarding the evaluation metrics, the accuracy for an individual client k is defined as the ratio of correctly classified samples $( N _ { k , c o r r e c t } )$ to the total number of samples on that client i.e., $\begin{array} { r } { a c c _ { k } = \frac { N _ { k , c o r r e c t } } { N _ { k } ^ { t e s t } } } \end{array}$ . To account for the influence of data volume, the aggregated accuracy (Avg) reported in the tables is calculated as the arithmetic mean of client accuracies, expressed as: $\begin{array} { r } { A v g = \frac { \sum _ { k = 1 } ^ { K } N _ { k , c o r r e c t } } { \sum _ { k = 1 } ^ { K } N _ { k } ^ { t e s t } } } \end{array}$

The organization of the experimental results corresponding to each setting is as follows: Under the M2 partitioning scheme, the data partition is illustrated in Figure 9. The corresponding test results and performance curves are presented in Table 6 and Figure 13, respectively. Similarly, results for the M1+ scheme are presented in Figure 10, Table 7, and Figure 14, while those for the M1 scheme are provided in Figure 11, Table 8, and Figure 15, respectively.

Furthermore, we extend the evaluation to a larger client pool to assess the scalability of the proposed framework. Based on the dataset size, Dataset Caltech101 is distributed across 50 clients, Dataset Reuters is scaled to 100 clients, and Datasets NUS-WIDE and Youtube are partitioned into 20 clients each, following the M1+ partitioning scheme (as illustrated in Figure

Table 5. List of Hyperparameters.
<table><tr><td colspan="2">Descriptions</td><td>Caltech101</td><td>Reuters</td><td>NUS-WIDE</td><td>Youtube</td></tr><tr><td colspan="2">Total rounds T</td><td>400 / 500</td><td>200</td><td>400</td><td>400</td></tr><tr><td rowspan="3">Server</td><td>Server optimizers</td><td colspan="4">SGD</td></tr><tr><td>Server learning rate  $\eta _ { s }$ </td><td colspan="4">0.005~0.01</td></tr><tr><td>Server training epoch S</td><td colspan="4">10</td></tr><tr><td rowspan="6">Client</td><td>Local training epoch  $E$ </td><td colspan="4">2</td></tr><tr><td>Training batch size B</td><td colspan="4">12</td></tr><tr><td>Local optimizers</td><td colspan="4">SGD</td></tr><tr><td>Weighting parameter  $\lambda$ </td><td colspan="4"> $\lambda _ { 1 } \in \{ 0 . 0 1 , 0 . 0 1 , 0 . 1 \} , \lambda _ { 2 } \in \{ 0 . 5 , 1 , 5 \}$ </td></tr><tr><td>Local learning rate  $\eta _ { c }$ </td><td> $0 . 0 0 5 { \sim } 0 . 0 1 0 . 0 0 1 { \sim } 0 . 0 0 5$ </td><td></td><td>0.01</td><td>0.005~0.01</td></tr><tr><td>Feature embedding dimension</td><td>64</td><td>24</td><td>32</td><td>48</td></tr></table>

12). The models are trained until stable convergence is achieved. The detailed experimental configurations are summarized in the row labeled $\mathbf { \ddot { \mu } } ^ { 6 } \mathbf { K } \mathbf { \# } ^ { 5 }$ in Table 2, and the resulting performance curves are depicted in Figure 16. The experimental results substantiate that MFedPBA consistently maintains superior performance and resilience, even within large-scale federated learning scenarios.

![](images/f9c20777eca0207f9ce8ff1481d4bf311c21d4e9ad4bf9460bdbe5298d124613.jpg)  
(a) Caltech101

![](images/914331a8291a58ce03d84f62d600999fefa55c0530df1fa6efe63d1e64e2c9fd.jpg)  
(b) Reuters

![](images/f5c99eb545ef2e87939e6a0f4714f91461d5b641abe120a8efaaf21e482d3a0e.jpg)  
(c) NUS-WIDE

![](images/dfdc029458c7f4e015efcb81a4ccf742ae44fe67cf860d7a7005054d9a658e0a.jpg)  
(d) Youtube

Figure 13. The test accuracy and convergence process of each method in M2 scenario.  
![](images/f1b01645ea3df87f00fb5ffff1f4d976bfcda2b75681082a05a5b8ecaf9e4536.jpg)  
(a) Caltech101

![](images/83f993e67fe1a8d6b16cabd83844ea7e6b18f2841c7de222aecbeba8c06e4b69.jpg)  
(b) Reuters

![](images/9183dd5f358c5624aa3b7f7b7b966068b3c450ee116baa6513379c81aedbf1fd.jpg)  
(c) NUS-WIDE

![](images/d2b6a65603b6fa2c0e36fcb5e376b0a9b0c0d171600e4b44b5c626362fee4954.jpg)  
(d) Youtube  
Figure 14. The test accuracy and convergence process of each method in M1+ scenario.

Table 6. Performance comparison (%) of all compared methods on Caltech101, Reuters, NUS-WIDE, and Youtube using M2 data partitioning, where the number of clients K is equal to the number of modalities M. The k1 represents the client with ID 1.
<table><tr><td colspan="2">Dataset</td><td>Local</td><td>FedProto</td><td>FedTGP</td><td>FedPall</td><td>Harmony</td><td>FedMVP</td><td>FedMobile</td><td>MFedPBA</td></tr><tr><td rowspan="4">Caltech101</td><td>k1</td><td> $4 0 . 4 8 _ { \pm 0 . 8 6 }$ </td><td> $3 7 . 9 6 _ { \pm 0 . 7 5 }$ </td><td> $3 9 . 7 1 _ { \pm 1 . 1 4 }$ </td><td> $3 9 . 8 1 _ { \pm 0 . 9 6 }$ </td><td> $4 2 . 1 4 _ { \pm 0 . 1 7 }$ </td><td> $3 5 . 0 1 _ { \pm 0 . 9 6 }$ </td><td> $4 1 . 8 9 _ { \pm 1 . 0 7 }$ </td><td> $4 7 . 3 5 _ { \pm 0 . 1 9 }$ </td></tr><tr><td>k2</td><td> $4 3 . 2 8 _ { \pm 0 . 4 2 }$ </td><td> $3 8 . 5 6 _ { \pm 0 . 2 1 }$ </td><td> $3 9 . 8 6 _ { \pm 0 . 9 4 }$ </td><td> $4 4 . 1 8 _ { \pm 0 . 1 6 }$ </td><td> $4 6 . 4 4 _ { \pm 0 . 1 5 }$ </td><td> $4 1 . 2 8 _ { \pm 0 . 4 2 }$ </td><td> $4 4 . 7 2 _ { \pm 0 . 2 8 }$ </td><td> $4 8 . 1 9 _ { \pm 0 . 1 8 }$ </td></tr><tr><td>k3</td><td> $4 4 . 2 4 _ { \pm 0 . 4 6 }$ </td><td> $3 9 . 8 9 _ { \pm 0 . 1 5 }$ </td><td> $4 3 . 5 2 _ { \pm 1 . 6 5 }$ </td><td> $4 3 . 3 _ { \pm 0 . 1 5 }$ </td><td> $4 7 . 2 3 _ { \pm 0 . 2 3 }$ </td><td> $4 4 . 8 _ { \pm 0 . 4 9 }$ </td><td> $4 3 . 5 2 _ { \pm 0 . 1 4 }$ </td><td> $5 2 . 4 4 _ { \pm 0 . 2 7 }$ </td></tr><tr><td> $\operatorname { A v g }$ </td><td> $4 2 . 6 5 _ { \pm 1 . 4 7 }$ </td><td> $3 8 . 7 2 _ { \pm 0 . 4 2 }$ </td><td> $4 0 . 7 8 { \scriptstyle \pm 1 . 3 1 }$ </td><td> $4 2 . 2 9 _ { \pm 0 . 2 7 }$ </td><td> $4 5 . 2 9 _ { \pm 0 . 3 8 }$ </td><td> $4 0 . 2 3 \substack { \pm 0 . 6 5 }$ </td><td> $4 3 . 5 1 _ { \pm 0 . 5 4 }$ </td><td> $4 9 . 0 4 _ { \pm 0 . 2 1 }$ </td></tr><tr><td rowspan="6">Reuters</td><td>k1</td><td> $7 7 . 2 2 { \scriptstyle \pm 1 . 5 8 }$ </td><td> $8 2 . 0 6 _ { \pm 0 . 3 2 }$ </td><td> $7 8 . 6 3 _ { \pm 0 . 7 8 }$ </td><td> $7 6 . 9 2 _ { \pm 0 . 6 2 }$ </td><td> $7 6 . 9 4 _ { \pm 0 . 9 1 }$ </td><td> $8 2 . 5 3 _ { \pm 1 . 6 7 }$ </td><td> $8 0 . 8 { \scriptstyle \pm 1 . 0 7 }$ </td><td> $8 2 . 5 5 { \scriptstyle \pm 0 . 7 9 }$ </td></tr><tr><td>k2</td><td> $6 6 . 8 2 _ { \pm 1 . 4 2 }$ </td><td> $6 4 . 4 9 _ { \pm 0 . 6 5 }$ </td><td> $6 2 . 2 3 _ { \pm 1 . 4 6 }$ </td><td> $6 6 . 8 8 _ { \pm 0 . 6 1 }$ </td><td> $6 8 . 1 2 _ { \pm 1 . 0 5 }$ </td><td> $6 5 . 6 2 _ { \pm 1 . 1 1 }$ </td><td> $6 8 . 1 1 _ { \pm 0 . 3 8 }$ </td><td> $6 7 . 8 3 _ { \pm 0 . 7 8 }$ </td></tr><tr><td>k3</td><td> $7 6 . 6 9 _ { \pm 0 . 9 8 }$ </td><td> $7 9 . 8 3 _ { \pm 1 . 3 2 }$ </td><td> $7 3 . 3 3 _ { \pm 1 . 4 7 }$ </td><td> $7 9 . 8 7 _ { \pm 0 . 2 3 }$ </td><td> $7 9 . 4 3 _ { \pm 2 . 1 1 }$ </td><td> $8 2 . 4 4 _ { \pm 0 . 6 1 }$ </td><td> $7 9 . 4 6 _ { \pm 1 . 4 8 }$ </td><td> $7 9 . 6 6 _ { \pm 1 . 2 1 }$ </td></tr><tr><td>k4</td><td> $6 9 . 8 6 _ { \pm 1 . 0 9 }$ </td><td> $7 3 . 7 1 _ { \pm 0 . 5 1 }$ </td><td> $7 0 . 7 8 _ { \pm 0 . 8 2 }$ </td><td> $7 3 . 7 7 _ { \pm 0 . 3 9 }$ </td><td> $6 9 . 1 4 _ { \pm 1 . 0 6 }$ </td><td> $7 1 . 5 2 _ { \pm 1 . 6 7 }$ </td><td> $7 0 . 8 7 _ { \pm 1 . 1 7 }$ </td><td> $7 5 . 7 1 _ { \pm 0 . 7 2 }$ </td></tr><tr><td>k5</td><td> $6 5 . 8 4 _ { \pm 1 . 0 5 }$ </td><td> $7 1 . 1 2 _ { \pm 0 . 4 6 }$ </td><td> $6 6 . 7 6 _ { \pm 1 . 3 4 }$ </td><td> $7 1 . 1 1 { \scriptstyle \pm 0 . 3 3 }$ </td><td> $7 1 . 4 8 _ { \pm 1 . 2 1 }$ </td><td> $7 3 . 7 8 _ { \pm 0 . 5 8 }$ </td><td> $7 0 . 9 7 _ { \pm 1 . 6 6 }$ </td><td> $7 3 . 2 9 _ { \pm 0 . 8 7 }$ </td></tr><tr><td> $\operatorname { A v g }$ </td><td> $7 0 . 6 _ { \pm 0 . 2 8 }$ </td><td> $7 3 . 3 9 _ { \pm 0 . 5 7 }$ </td><td> $6 9 . 7 3 _ { \pm 0 . 8 8 }$ </td><td> $7 3 . 0 8 _ { \pm 1 . 0 4 }$ </td><td> $7 2 . 1 8 _ { \pm 0 . 4 1 }$ </td><td> $7 4 . 0 7 _ { \pm 1 . 0 8 }$ </td><td> $7 3 . 1 9 _ { \pm 0 . 2 6 }$ </td><td> $7 5 . 1 6 _ { \pm 0 . 4 5 }$ </td></tr><tr><td rowspan="5">NUS-WIDE</td><td>k1</td><td> $3 5 . 2 8 _ { \pm 0 . 5 3 }$ </td><td> $3 2 . 3 6 _ { \pm 0 . 5 9 }$ </td><td> $3 5 . 2 8 _ { \pm 0 . 1 7 }$ </td><td> $3 8 . 9 7 _ { \pm 1 . 7 7 }$ </td><td> $3 8 . 4 3 _ { \pm 0 . 9 5 }$ </td><td> $3 7 . 5 7 _ { \pm 0 . 9 8 }$ </td><td> $3 9 . 2 6 _ { \pm 1 . 0 3 }$ </td><td> $3 8 . 9 1 _ { \pm 0 . 7 8 }$ </td></tr><tr><td>k2</td><td> $3 5 . 7 6 { \scriptstyle \pm 1 . 5 4 }$ </td><td> $3 2 . 4 6 _ { \pm 0 . 6 6 }$ </td><td> $3 4 . 2 6 { \scriptstyle \pm 0 . 4 8 }$ </td><td> $3 4 . 7 2 _ { \pm 0 . 8 3 }$ </td><td> $3 5 . 6 8 _ { \pm 1 . 3 7 }$ </td><td> $3 2 . 7 4 _ { \pm 0 . 4 6 }$ </td><td> $3 6 . 4 7 _ { \pm 1 . 0 0 }$ </td><td> $3 9 . 0 3 _ { \pm 1 . 0 0 }$ </td></tr><tr><td>k3</td><td> $2 9 . 5 2 _ { \pm 0 . 8 3 }$ </td><td> $2 6 . 8 6 _ { \pm 0 . 1 2 }$ </td><td> $2 8 . 4 2 _ { \pm 1 . 0 2 }$ </td><td> $2 9 . 9 _ { \pm 0 . 1 8 }$ </td><td> $2 8 . 9 _ { \pm 0 . 8 4 }$ </td><td> $3 0 . 0 8 _ { \pm 0 . 5 7 }$ </td><td> $3 0 . 1 6 _ { \pm 0 . 4 3 }$ </td><td>32.66±0.42</td></tr><tr><td>k4</td><td> $2 9 . 3 1 _ { \pm 0 . 7 1 }$ </td><td> $2 6 . 4 2 _ { \pm 1 . 4 7 }$ </td><td> $2 7 . 6 5 _ { \pm 0 . 6 5 }$ </td><td> $3 0 . 4 8 _ { \pm 1 . 0 4 }$ </td><td> $3 0 . 0 4 _ { \pm 0 . 2 2 }$ </td><td> $2 6 . 7 1 _ { \pm 0 . 5 }$ </td><td> $3 0 . 5 _ { \pm 0 . 5 4 }$ </td><td> $3 1 . 1 4 _ { \pm 1 . 5 3 }$ </td></tr><tr><td> $\operatorname { A v g }$ </td><td> $3 2 . 1 8 _ { \pm 0 . 6 4 }$ </td><td> $2 9 . 2 8 _ { \pm 0 . 4 6 }$ </td><td> $3 1 . 1 9 _ { \pm 0 . 5 1 }$ </td><td> $3 3 . 3 8 _ { \pm 0 . 9 1 }$ </td><td>一  $3 2 . 9 7 _ { \pm 0 . 2 8 }$ </td><td> $3 1 . 8 9 _ { \pm 0 . 4 }$ </td><td> $3 3 . 8 6 _ { \pm 0 . 2 2 }$ </td><td> $3 5 . 2 _ { \pm 0 . 4 6 }$ </td></tr><tr><td rowspan="7">Youtube</td><td>k1</td><td>二  $2 6 . 8 2 _ { \pm 0 . 3 3 }$ </td><td>一  $3 1 . 9 9 _ { \pm 1 . 8 5 }$ </td><td> $3 6 . 9 7 _ { \pm 1 . 3 3 }$ </td><td> $3 3 . 1 4 _ { \pm 0 . 8 8 }$ </td><td> $3 1 . 6 1 _ { \pm 1 . 1 5 }$ </td><td> $3 5 . 4 4 _ { \pm 0 . 3 3 }$ </td><td> $3 4 . 6 7 _ { \pm 0 . 3 3 }$ </td><td> $3 9 . 4 6 _ { \pm 1 . 4 5 }$ </td></tr><tr><td>k2</td><td> $2 3 . 7 7 { \scriptstyle \pm 1 . 1 2 }$ </td><td> $2 2 . 3 _ { \pm 4 . 0 5 }$ </td><td> $2 2 . 5 5 { \scriptstyle \pm 0 . 4 2 }$ </td><td> $2 3 . 0 4 _ { \pm 0 . 4 2 }$ </td><td> $2 1 . 8 1 _ { \pm 1 . 1 2 }$ </td><td> $2 3 . 5 3 \mathrm { \pm 0 . 7 4 }$ </td><td> $2 2 . 0 6 _ { \pm 1 . 2 7 }$ </td><td> $2 1 . 1 2 _ { \pm 0 . 7 4 }$ </td></tr><tr><td>k3</td><td> $4 2 . 2 4 _ { \pm 1 . 4 9 }$ </td><td> $3 5 . 6 3 _ { \pm 1 . 3 2 }$ </td><td> $3 6 . 4 9 _ { \pm 1 . 0 0 }$ </td><td> $3 9 . 9 4 _ { \pm 0 . 5 }$ </td><td> $4 3 . 1 _ { \pm 2 . 2 8 }$ </td><td> $3 8 . 2 2 _ { \pm 0 . 5 }$ </td><td> $4 3 . 6 8 _ { \pm 3 . 0 3 }$ </td><td> $4 6 . 8 4 _ { \pm 1 . 3 2 }$ </td></tr><tr><td>k4</td><td> $5 0 . 8 6 _ { \pm 1 . 5 7 }$ </td><td> $5 1 . 5 5 { \scriptstyle \pm 0 . 8 9 }$ </td><td> $5 5 . 5 { \scriptstyle \pm 3 . 3 1 }$ </td><td> $5 2 . 2 3 _ { \pm 1 . 8 1 }$ </td><td> $5 5 . 1 5 _ { \pm 1 . 7 9 }$ </td><td> $5 8 . 0 8 { \scriptstyle \pm 1 . 6 6 }$ </td><td> $5 2 . 0 6 _ { \pm 0 . 8 9 }$ </td><td> $5 5 . 5 _ { \pm 2 . 8 4 }$ </td></tr><tr><td>k5</td><td> $4 5 . 8 3 _ { \pm 1 . 7 4 }$ </td><td> $4 4 . 8 3 _ { \pm 1 . 5 5 }$ </td><td> $5 1 . 0 1 _ { \pm 2 . 5 3 }$ </td><td> $5 3 . 0 2 _ { \pm 1 . 8 8 }$ </td><td> $5 0 . 1 4 _ { \pm 1 . 5 1 }$ </td><td> $5 2 . 3 _ { \pm 0 . 2 5 }$ </td><td> $5 0 . 5 7 { \scriptstyle \pm 1 . 9 9 }$ </td><td> $5 4 . 1 8 _ { \pm 0 . 5 }$ </td></tr><tr><td>k6</td><td> $5 6 . 2 8 { \scriptstyle \pm 1 . 5 }$ </td><td> $5 3 . 4 6 _ { \pm 0 . 9 9 }$ </td><td> $5 8 . 4 4 _ { \pm 2 . 2 5 }$ </td><td> $5 5 . 4 1 _ { \pm 2 . 4 6 }$ </td><td> $5 3 . 6 8 _ { \pm 3 . 0 7 }$ </td><td> $5 5 . 8 4 _ { \pm 1 . 1 2 }$ </td><td> $5 9 . 5 2 { \scriptstyle \pm 1 . 3 5 }$ </td><td> $6 0 . 9 2 _ { \pm 0 . 2 }$ </td></tr><tr><td> $\operatorname { A v g }$ </td><td> $4 1 . 7 2 _ { \pm 0 . 6 6 }$ </td><td> $4 1 . 1 2 _ { \pm 0 . 6 8 }$ </td><td> $4 5 . 1 6 _ { \pm 1 . 0 2 }$ </td><td> $4 4 . 2 3 _ { \pm 0 . 3 }$ </td><td> $4 3 . 8 _ { \pm 0 . 7 6 }$ </td><td> $4 5 . 5 3 _ { \pm 0 . 5 2 }$ </td><td> $4 4 . 8 3 _ { \pm 0 . 5 3 }$ </td><td> $4 7 . 9 2 _ { \pm 0 . 9 3 }$ </td></tr></table>

![](images/da2dcd3d527c9664b3d113790a7c28e047120299e7609db20a61466cc47817fe.jpg)  
(a) Caltech101

![](images/42a2feb932a8b70ea01a22d2df979fe44a3d12c35d833b292502a07c7c872ef2.jpg)  
(b) Reuters

![](images/8fbf2c5f4261a71db898f5a9723ac207ffc2f80dbf2ebbb8bb5db67bad10463e.jpg)  
(c) NUS-WIDE

![](images/81d49eb729ed0f3dd9eab192036b771befc66eb6d6ae12dcd618ea0fa38fc76b.jpg)  
(d) Youtube

Figure 15. The test accuracy and convergence process of each method in M1 scenario.  
![](images/1205a3320ad14b7b03f0464c18b5480f0347511140d1e36ac270dd7b5620129c.jpg)

![](images/db16dae2565b8221d30a0e35008b8ec33bff989a71d81f9a61ed5b9ac47493df.jpg)  
(a) Caltech101

![](images/84eecc52f2835da6aafe3d15df5bb8586cfea377d545f0499faed6ae5fefe5e1.jpg)  
(b) Reuters  
(c) NUS-WIDE

![](images/ae7b3678c226fdb541756f739b2528d0798ca4ef9bf9f6d41c7e049842beb623.jpg)  
(d) Youtube  
Figure 16. The testing accuracy and convergence process of each method in the M1+ scenario, with a large number of clients participating.

Table 7. Performance comparison (%) of all compared methods on Caltech101, Reuters, NUS-WIDE, and Youtube using M1+ data partitioning, where the number of clients K is equal to the number of modalities M. The k1 represents the client with ID 1.

$$
3 4 . 5 1 _ { \pm 0 . 6 7 }
$$

$$
3 0 . 7 4 _ { \pm 1 . 0 9 }
$$

$$
4 0 . 6 3 _ { \pm 0 . 8 7 }
$$

$$
3 4 . 7 _ { \pm 1 . 5 3 }
$$

$$
3 5 . 0 2 _ { \pm 0 . 3 9 }
$$

$$
3 5 . 1 7 _ { \pm 0 . 7 9 }
$$

$$
3 8 . 5 7 _ { \pm 0 . 7 7 }
$$

$$
3 9 . 3 4 _ { \pm 0 . 6 5 }
$$

$$
2 9 . 5 8 _ { \pm 1 . 0 0 }
$$

$$
4 3 . 4 9 _ { \pm 0 . 5 8 }
$$

$$
4 1 . 6 8 _ { \pm 0 . 8 3 }
$$

$$
3 6 . 5 8 _ { \pm 0 . 6 1 }
$$

$$
4 0 . 4 6 _ { \pm 0 . 2 4 }
$$

$$
4 1 . 0 4 _ { \pm 1 . 0 3 }
$$

$$
4 1 . 1 8 _ { \pm 0 . 2 4 }
$$

$$
4 4 . 5 9 _ { \pm 2 . 0 3 }
$$

$$
3 3 . 6 9 _ { \pm 0 . 5 4 }
$$

$$
4 4 . 5 3 _ { \pm 1 . 2 2 }
$$

$$
4 5 . 9 2 _ { \pm 0 . 5 6 }
$$

$$
4 1 . 0 5 _ { \pm 0 . 5 8 }
$$

$$
4 6 . 3 _ { \pm 1 . 1 }
$$

$$
4 5 . 4 6 _ { \pm 0 . 4 7 }
$$

$$
4 6 . 4 5 _ { \pm 1 . 5 7 }
$$

$$
5 0 . 0 2 _ { \pm 0 . 5 9 }
$$

$$
4 0 . 8 3 _ { \pm 1 . 1 2 }
$$

$$
\operatorname { A v g }
$$

$$
3 6 . 8 1 _ { \pm 0 . 3 5 }
$$

$$
4 0 . 7 1 { \scriptstyle \pm 1 . 2 3 }
$$

$$
4 1 . 8 _ { \pm 0 . 7 7 }
$$

$$
4 3 . 0 4 _ { \pm 0 . 6 4 }
$$

$$
3 8 . 9 1 _ { \pm 1 . 2 5 }
$$

$$
4 2 . 2 9 _ { \pm 0 . 6 }
$$

$$
4 6 . 8 2 _ { \pm 0 . 5 4 }
$$

$$
7 2 . 2 8 { \scriptstyle \pm 1 . 9 2 }
$$

$$
7 5 . 0 0 { \scriptstyle \pm 2 . 5 4 }
$$

$$
7 4 . 6 1 { \scriptstyle \pm 2 . 5 8 }
$$

$$
7 4 . 8 _ { \pm 1 . 7 6 }
$$

$$
7 4 . 7 1 { \scriptstyle \pm 0 . 8 5 }
$$

$$
6 1 . 1 2 _ { \pm 0 . 8 8 }
$$

$$
7 5 . 1 2 _ { \pm 1 . 2 5 }
$$

$$
8 2 . 1 1 { \scriptstyle \pm 0 . 7 7 }
$$

$$
8 1 . 2 _ { \pm 0 . 8 1 }
$$

$$
8 0 . 0 4 _ { \pm 1 . 5 6 }
$$

$$
8 1 . 9 _ { \pm 1 . 1 7 }
$$

$$
8 0 . 8 2 _ { \pm 1 . 0 5 }
$$

$$
8 0 . 7 6 _ { \pm 0 . 2 9 }
$$

$$
8 1 . 6 3 _ { \pm 0 . 2 3 }
$$

$$
8 3 . 3 2 _ { \pm 0 . 1 8 }
$$

$$
6 0 . 5 7 { \scriptstyle \pm 0 . 8 5 }
$$

$$
5 3 . 4 _ { \pm 1 . 2 5 }
$$

$$
5 7 . 8 3 _ { \pm 1 . 8 7 }
$$

$$
6 0 . 1 2 { \scriptstyle \pm 1 . 1 8 }
$$

$$
8 3 . 7 2 _ { \pm 0 . 6 4 }
$$

$$
6 1 . 7 _ { \pm 0 . 6 7 }
$$

$$
6 0 . 0 5 _ { \pm 0 . 3 4 }
$$

$$
6 1 . 3 1 { \scriptstyle \pm 0 . 1 4 }
$$

$$
6 5 . 5 7 _ { \pm 0 . 5 3 }
$$

$$
5 8 . 4 2 _ { \pm 1 . 3 2 }
$$

$$
6 1 . 3 5 _ { \pm 1 . 9 4 }
$$

$$
6 3 . 0 4 _ { \pm 2 . 4 7 }
$$

$$
6 2 . 2 1 { \scriptstyle \pm 0 . 7 }
$$

$$
6 7 . 6 4 _ { \pm 0 . 2 5 }
$$

$$
6 4 . 2 7 _ { \pm 0 . 6 7 }
$$

$$
7 3 . 9 5 { \scriptstyle \pm 2 . 6 5 }
$$

$$
7 7 . 7 4 { \scriptstyle \pm 1 . 0 1 }
$$

$$
6 7 . 0 6 _ { \pm 0 . 4 5 }
$$

$$
7 7 . 3 9 _ { \pm 1 . 5 7 }
$$

$$
7 8 . 4 _ { \pm 1 . 1 4 }
$$

$$
7 7 . 6 1 { \scriptstyle \pm 0 . 4 6 }
$$

$$
6 8 . 3 4 _ { \pm 0 . 4 6 }
$$

$$
7 7 . 2 6 _ { \pm 0 . 2 7 }
$$

$$
7 8 . 9 5 { \scriptstyle \pm 1 . 5 5 }
$$

$$
8 1 . 8 2 _ { \pm 0 . 2 1 }
$$

$$
7 0 . 8 5 _ { \pm 0 . 5 9 }
$$

$$
6 8 . 5 4 _ { \pm 1 . 4 6 }
$$

$$
6 9 . 8 _ { \pm 0 . 5 9 }
$$

$$
7 1 . 0 9 _ { \pm 1 . 5 6 }
$$

$$
7 2 . 2 8 _ { \pm 0 . 4 7 }
$$

$$
6 8 . 0 9 _ { \pm 0 . 8 2 }
$$

$$
7 2 . 7 9 _ { \pm 0 . 4 4 }
$$

$$
7 5 . 0 1 _ { \pm 0 . 2 2 }
$$

$$
2 8 . 6 3 _ { \pm 0 . 7 5 }
$$

$$
3 0 . 2 1 _ { \pm 0 . 4 3 }
$$

$$
2 7 . 4 8 _ { \pm 1 . 1 1 }
$$

$$
3 0 . 7 7 _ { \pm 1 . 4 1 }
$$

$$
2 8 . 1 6 _ { \pm 0 . 9 }
$$

$$
2 8 . 1 7 { \scriptstyle \pm 0 . 2 4 }
$$

$$
3 2 . 8 2 _ { \pm 0 . 5 3 }
$$

$$
2 9 . 8 2 _ { \pm 0 . 6 5 }
$$

$$
2 6 . 2 _ { \pm 0 . 6 8 }
$$

$$
2 5 . 6 7 _ { \pm 1 . 3 9 }
$$

$$
3 3 . 0 2 _ { \pm 1 . 3 1 }
$$

$$
2 9 . 3 5 _ { \pm 0 . 4 2 }
$$

$$
3 1 . 8 3 _ { \pm 1 . 2 4 }
$$

$$
2 9 . 5 1 { \scriptstyle \pm 0 . 5 3 }
$$

$$
2 9 . 7 7 { \scriptstyle \pm 0 . 5 5 }
$$

$$
3 0 . 9 _ { \pm 0 . 8 5 }
$$

$$
2 8 . 7 _ { \pm 1 . 2 6 }
$$

$$
2 5 . 9 4 _ { \pm 0 . 8 1 }
$$

$$
2 6 . 3 7 _ { \pm 2 . 0 1 }
$$

$$
3 1 . 0 1 _ { \pm 0 . 8 1 }
$$

$$
3 2 . 4 2 _ { \pm 0 . 5 9 }
$$

$$
3 0 . 9 2 _ { \pm 0 . 5 1 }
$$

$$
2 9 . 6 1 _ { \pm 0 . 7 3 }
$$

$$
3 1 . 9 3 _ { \pm 0 . 3 3 }
$$

$$
3 4 . 3 2 { \scriptstyle \pm 2 . 7 9 }
$$

$$
2 8 . 7 5 _ { \pm 0 . 5 7 }
$$

$$
2 8 . 5 2 { \scriptstyle \pm 1 . 9 1 }
$$

$$
3 5 . 5 6 { \scriptstyle \pm 0 . 8 6 }
$$

$$
3 4 . 2 1 _ { \pm 1 . 9 7 }
$$

$$
3 2 . 9 6 _ { \pm 1 . 2 }
$$

$$
3 1 . 3 5 _ { \pm 3 . 9 7 }
$$

$$
3 3 . 7 1 { \scriptstyle \pm 2 . 5 3 }
$$

$$
2 8 . 7 2 _ { \pm 0 . 7 1 }
$$

$$
\operatorname { A v g }
$$

$$
2 7 . 7 9 _ { \pm 0 . 5 5 }
$$

$$
2 7 . 1 3 _ { \pm 0 . 1 8 }
$$

$$
3 0 . 9 9 _ { \pm 0 . 6 3 }
$$

$$
3 1 . 5 6 _ { \pm 0 . 6 1 }
$$

$$
2 9 . 8 2 _ { \pm 0 . 5 9 }
$$

$$
3 0 . 9 5 _ { \pm 0 . 5 7 }
$$

$$
3 2 . 1 5 _ { \pm 0 . 3 6 }
$$

$$
3 3 . 5 _ { \pm 1 . 0 2 }
$$

$$
3 3 . 3 3 _ { \pm 0 . 8 5 }
$$

$$
3 3 . 1 7 _ { \pm 2 . 0 4 }
$$

$$
3 5 . 6 2 _ { \pm 1 . 0 2 }
$$

$$
3 5 . 7 8 _ { \pm 1 . 7 }
$$

$$
3 5 . 4 6 _ { \pm 0 . 5 7 }
$$

$$
4 0 . 2 3 _ { \pm 2 . 1 9 }
$$

$$
4 0 . 5 2 _ { \pm 1 . 0 2 }
$$

$$
5 7 . 1 8 _ { \pm 2 . 4 6 }
$$

$$
5 2 . 8 7 _ { \pm 0 . 5 }
$$

$$
5 8 . 0 5 _ { \pm 2 . 2 4 }
$$

$$
5 6 . 7 _ { \pm 1 . 6 3 }
$$

$$
5 7 . 9 5 _ { \pm 1 . 6 8 }
$$

$$
5 9 . 5 8 { \scriptstyle \pm 0 . 6 6 }
$$

$$
5 8 . 1 4 _ { \pm 0 . 6 6 }
$$

$$
6 0 . 2 5 { \scriptstyle \pm 0 . 4 4 }
$$

$$
3 7 . 5 _ { \pm 1 . 3 8 }
$$

$$
3 0 . 5 6 _ { \pm 4 . 3 5 }
$$

$$
3 7 . 5 _ { \pm 1 . 5 6 }
$$

$$
3 6 . 6 3 _ { \pm 1 . 9 7 }
$$

$$
3 7 . 1 5 _ { \pm 1 . 3 1 }
$$

$$
3 4 . 5 5 _ { \pm 0 . 3 }
$$

$$
3 7 . 1 5 _ { \pm 0 . 6 }
$$

$$
3 6 . 8 1 _ { \pm 1 . 0 6 }
$$

$$
4 0 . 3 7 { \scriptstyle \pm 1 . 7 9 }
$$

$$
3 7 . 9 3 _ { \pm 4 . 4 8 }
$$

$$
4 4 . 5 4 _ { \pm 1 . 0 0 }
$$

$$
4 2 . 8 2 _ { \pm 1 . 7 9 }
$$

$$
4 3 . 3 9 _ { \pm 3 . 0 6 }
$$

$$
4 2 . 2 4 _ { \pm 0 . 8 6 }
$$

$$
4 4 . 4 _ { \pm 0 . 7 5 }
$$

$$
4 5 . 4 _ { \pm 0 . 2 5 }
$$

$$
4 4 . 3 2 _ { \pm 1 . 9 7 }
$$

$$
4 0 . 8 7 _ { \pm 4 . 2 }
$$

$$
4 9 . 6 8 _ { \pm 0 . 9 6 }
$$

$$
4 7 . 5 1 _ { \pm 1 . 3 8 }
$$

$$
4 7 . 1 3 _ { \pm 0 . 6 6 }
$$

$$
4 9 . 1 7 _ { \pm 0 . 9 6 }
$$

$$
4 6 . 4 9 _ { \pm 1 . 7 3 }
$$

$$
5 0 . 4 5 _ { \pm 1 . 3 3 }
$$

$$
2 2 . 0 8 _ { \pm 1 . 1 2 }
$$

$$
2 2 . 5 1 { \scriptstyle \pm 5 . 3 2 }
$$

$$
2 9 . 0 6 _ { \pm 2 . 1 5 }
$$

$$
2 2 . 7 3 { \scriptstyle \pm 2 . 8 3 }
$$

$$
2 2 . 7 3 { \scriptstyle \pm 2 . 3 4 }
$$

$$
2 7 . 9 2 _ { \pm 2 . 3 4 }
$$

$$
2 7 . 2 7 _ { \pm 2 . 3 4 }
$$

$$
2 9 . 2 2 _ { \pm 2 . 3 4 }
$$

$$
4 1 . 8 9 _ { \pm 0 . 7 6 }
$$

$$
\operatorname { A v g }
$$

$$
3 8 . 8 2 _ { \pm 1 . 1 2 }
$$

$$
4 4 . 5 5 _ { \pm 0 . 6 8 }
$$

$$
4 3 . 0 4 _ { \pm 1 . 0 9 }
$$

$$
4 3 . 4 7 _ { \pm 0 . 9 1 }
$$

$$
4 4 . 2 4 _ { \pm 0 . 3 6 }
$$

$$
4 4 . 2 8 _ { \pm 0 . 6 4 }
$$

$$
4 7 . 2 1 _ { \pm 0 . 5 9 }
$$

Table 8. Performance comparison (%) of all compared methods on Caltech101, Reuters, NUS-WIDE, and Youtube using M1 data partitioning, where the number of clients K is equal to the number of modalities M. The X1 represents a client that has the X1 modal enabled.

$$
3 9 . 4 7 _ { \pm 0 . 1 3 }
$$

$$
3 8 . 2 7 _ { \pm 1 . 4 }
$$

$$
3 6 . 8 9 _ { \pm 0 . 2 8 }
$$

$$
2 5 . 5 6 { \scriptstyle \pm 2 . 0 4 }
$$

$$
4 1 . 1 6 _ { \pm 0 . 6 7 }
$$

$$
4 2 _ { \pm 0 . 7 1 }
$$

$$
3 1 . 4 7 _ { \pm 1 . 7 3 }
$$

$$
3 9 . 0 7 _ { \pm 1 . 2 2 }
$$

$$
3 9 . 8 4 _ { \pm 0 . 6 2 }
$$

$$
3 8 . 6 2 _ { \pm 0 . 5 }
$$

$$
3 0 . 8 9 _ { \pm 1 . 3 4 }
$$

$$
3 7 . 0 2 { \scriptstyle \pm 0 . 5 6 }
$$

$$
4 1 . 8 7 _ { \pm 0 . 1 3 }
$$

$$
3 5 . 9 1 _ { \pm 0 . 6 6 }
$$

$$
3 9 . 7 8 _ { \pm 1 . 3 }
$$

$$
4 3 . 2 9 _ { \pm 0 . 2 }
$$

$$
4 8 . 7 3 _ { \pm 0 . 9 }
$$

$$
3 8 . 0 9 { \scriptstyle \pm 1 . 1 6 }
$$

$$
4 0 . 6 9 _ { \pm 0 . 8 3 }
$$

$$
3 8 . 1 7 { \scriptstyle \pm 0 . 7 3 }
$$

$$
4 0 . 0 4 _ { \pm 1 . 6 5 }
$$

$$
4 0 . 7 8 _ { \pm 0 . 3 7 }
$$

$$
4 0 . 2 2 _ { \pm 0 . 2 8 }
$$

$$
4 1 . 4 _ { \pm 1 . 0 5 }
$$

$$
\operatorname { A v g }
$$

$$
3 8 . 4 7 _ { \pm 0 . 4 5 }
$$

$$
3 1 . 5 7 { \scriptstyle \pm 1 . 2 3 }
$$

$$
3 6 . 1 8 _ { \pm 0 . 7 2 }
$$

$$
4 0 . 7 7 { \scriptstyle \pm 0 . 3 6 }
$$

$$
4 0 . 1 _ { \pm 0 . 6 9 }
$$

$$
3 8 . 9 2 _ { \pm 0 . 6 2 }
$$

$$
4 0 . 0 9 _ { \pm 0 . 1 3 }
$$

$$
4 3 . 6 4 _ { \pm 0 . 1 1 }
$$

$$
6 4 . 4 7 _ { \pm 1 . 0 7 }
$$

$$
5 8 . 8 7 _ { \pm 0 . 6 1 }
$$

$$
6 0 . 3 9 _ { \pm 0 . 9 3 }
$$

$$
6 5 . 8 5 { \scriptstyle \pm 0 . 5 5 }
$$

$$
6 4 . 3 9 { \scriptstyle \pm 0 . 5 5 }
$$

$$
6 3 . 8 5 _ { \pm 1 . 5 9 }
$$

$$
6 4 . 9 3 _ { \pm 0 . 4 3 }
$$

$$
6 9 . 8 1 _ { \pm 1 . 4 }
$$

$$
7 9 . 5 2 _ { \pm 0 . 6 7 }
$$

$$
6 7 . 2 { \scriptstyle \pm 1 . 3 3 }
$$

$$
7 7 . 5 8 _ { \pm 0 . 8 8 }
$$

$$
7 4 . 3 1 _ { \pm 2 . 0 3 }
$$

$$
7 6 . 4 4 _ { \pm 0 . 1 8 }
$$

$$
7 6 . 4 1 _ { \pm 0 . 2 5 }
$$

$$
7 5 . 3 7 _ { \pm 0 . 3 8 }
$$

$$
6 3 . 1 5 { \scriptstyle \pm 0 . 9 6 }
$$

$$
6 6 . 7 _ { \pm 0 . 7 1 }
$$

$$
7 7 . 8 7 _ { \pm 0 . 8 5 }
$$

$$
4 8 . 4 3 _ { \pm 0 . 9 9 }
$$

$$
5 4 . 5 5 { \scriptstyle \pm 3 . 6 7 }
$$

$$
6 7 . 3 8 { \scriptstyle \pm 0 . 4 3 }
$$

$$
6 8 . 6 1 _ { \pm 1 . 2 1 }
$$

$$
6 7 . 3 1 { \scriptstyle \pm 0 . 4 4 }
$$

$$
7 2 . 3 9 _ { \pm 1 . 5 9 }
$$

$$
7 8 . 2 9 _ { \pm 0 . 1 6 }
$$

$$
7 6 . 6 1 _ { \pm 1 . 6 4 }
$$

$$
6 8 . 2 6 { \scriptstyle \pm 0 . 3 3 }
$$

$$
7 3 . 2 4 _ { \pm 1 . 2 9 }
$$

$$
7 6 . 1 9 _ { \pm 0 . 1 2 }
$$

$$
_ { \textrm L 5 }
$$

$$
7 5 . 0 1 _ { \pm 1 . 1 2 }
$$

$$
6 5 . 4 4 { \scriptstyle \pm 1 . 5 5 }
$$

$$
6 0 . 3 8 { \scriptstyle \pm 0 . 6 5 }
$$

$$
6 6 . 0 4 _ { \pm 0 . 7 7 }
$$

$$
6 6 . 8 4 _ { \pm 0 . 9 8 }
$$

$$
7 5 . 2 _ { \pm 0 . 6 2 }
$$

$$
6 6 . 4 2 _ { \pm 0 . 7 }
$$

$$
6 7 . 0 4 _ { \pm 0 . 1 3 }
$$

$$
7 9 . 8 3 _ { \pm 1 . 6 5 }
$$

$$
6 6 . 8 1 _ { \pm 0 . 3 3 }
$$

$$
6 7 . 3 1 { \scriptstyle \pm 0 . 3 3 }
$$

$$
6 7 . 0 5 _ { \pm 1 . 0 4 }
$$

$$
\operatorname { A v g }
$$

$$
6 8 . 7 5 _ { \pm 0 . 6 3 }
$$

$$
6 5 . 8 1 _ { \pm 0 . 3 3 }
$$

$$
6 6 . 9 6 _ { \pm 0 . 2 2 }
$$

$$
7 0 . 1 6 _ { \pm 1 . 1 4 }
$$

$$
7 0 . 1 8 _ { \pm 0 . 5 7 }
$$

$$
6 9 . 9 2 _ { \pm 0 . 9 5 }
$$

$$
7 2 . 0 9 _ { \pm 0 . 7 9 }
$$

$$
3 4 . 5 _ { \pm 0 . 8 5 }
$$

$$
2 3 . 1 5 _ { \pm 1 . 5 5 }
$$

$$
3 5 . 0 4 _ { \pm 0 . 1 8 }
$$

$$
3 2 . 2 7 _ { \pm 3 . 2 4 }
$$

$$
3 6 . 8 5 _ { \pm 0 . 4 9 }
$$

$$
3 2 . 3 2 _ { \pm 1 . 3 4 }
$$

$$
3 3 . 6 5 _ { \pm 1 . 3 4 }
$$

$$
3 3 . 3 3 _ { \pm 1 . 2 1 }
$$

$$
3 4 . 4 _ { \pm 1 . 1 2 }
$$

$$
3 4 . 0 8 _ { \pm 3 . 0 2 }
$$

$$
3 4 . 2 9 _ { \pm 0 . 8 }
$$

$$
3 6 . 0 0 { \scriptstyle \pm 0 . 4 9 }
$$

$$
3 8 . 7 6 { \scriptstyle \pm 1 . 5 8 }
$$

$$
3 9 . 4 _ { \pm 0 . 3 7 }
$$

$$
3 6 . 5 3 { \scriptstyle \pm 1 . 7 6 }
$$

$$
3 2 . 3 7 _ { \pm 0 . 8 }
$$

$$
4 0 . 7 9 _ { \pm 0 . 8 5 }
$$

$$
2 2 . 1 7 _ { \pm 1 . 6 3 }
$$

$$
3 0 . 3 5 _ { \pm 0 . 8 6 }
$$

$$
3 5 . 4 6 _ { \pm 1 . 1 5 }
$$

$$
3 3 . 0 1 _ { \pm 1 . 2 9 }
$$

$$
3 4 . 6 1 _ { \pm 1 . 2 1 }
$$

$$
3 3 . 5 5 _ { \pm 1 . 6 6 }
$$

$$
2 0 . 8 7 _ { \pm 0 . 4 9 }
$$

$$
3 5 . 4 6 _ { \pm 0 . 6 8 }
$$

$$
2 2 . 1 9 _ { \pm 3 . 0 2 }
$$

$$
1 8 . 7 4 _ { \pm 0 . 8 2 }
$$

$$
2 0 . 3 4 { \scriptstyle \pm 0 . 7 4 }
$$

$$
2 2 . 5 8 { \scriptstyle \pm 1 . 6 1 }
$$

$$
2 3 . 4 3 _ { \pm 0 . 7 8 }
$$

$$
2 1 . 0 9 _ { \pm 0 . 3 2 }
$$

$$
2 2 . 4 7 _ { \pm 0 . 4 9 }
$$

$$
3 1 . 1 5 _ { \pm 0 . 1 4 }
$$

$$
\operatorname { A v g }
$$

$$
2 4 . 9 6 _ { \pm 0 . 5 2 }
$$

$$
2 9 . 5 5 _ { \pm 0 . 7 3 }
$$

$$
3 1 . 0 2 _ { \pm 0 . 9 6 }
$$

$$
3 2 . 1 9 _ { \pm 0 . 3 5 }
$$

$$
3 2 . 7 _ { \pm 0 . 5 1 }
$$

$$
3 1 . 2 _ { \pm 0 . 3 6 }
$$

$$
3 3 . 2 5 _ { \pm 0 . 2 4 }
$$

$$
3 4 . 1 3 _ { \pm 0 . 6 9 }
$$

$$
3 9 . 2 9 _ { \pm 2 . 0 6 }
$$

$$
3 2 . 9 4 _ { \pm 2 . 7 5 }
$$

$$
3 4 . 1 3 _ { \pm 2 . 4 8 }
$$

$$
3 3 . 3 3 _ { \pm 2 . 0 6 }
$$

$$
3 4 . 3 4 _ { \pm 1 . 7 4 }
$$

$$
3 7 . 3 _ { \pm 0 . 6 9 }
$$

$$
4 4 . 4 4 _ { \pm 1 . 3 7 }
$$

$$
3 4 . 1 3 { \scriptstyle \pm 1 . 3 7 }
$$

$$
2 6 . 9 _ { \pm 2 . 8 9 }
$$

$$
2 7 . 7 8 _ { \pm 1 . 3 7 }
$$

$$
3 3 . 3 3 { \scriptstyle \pm 1 . 1 9 }
$$

$$
4 1 . 2 7 _ { \pm 3 . 6 4 }
$$

$$
3 8 . 8 9 _ { \pm 1 . 8 2 }
$$

$$
3 4 . 5 2 _ { \pm 3 . 1 5 }
$$

$$
3 0 . 5 6 _ { \pm 1 . 3 7 }
$$

$$
2 2 . 2 2 _ { \pm 1 . 8 2 }
$$

$$
1 8 . 2 _ { \pm 0 . 5 9 }
$$

$$
2 1 . 8 3 _ { \pm 0 . 6 9 }
$$

$$
2 1 . 4 3 _ { \pm 3 . 1 5 }
$$

$$
2 2 . 6 2 _ { \pm 2 . 0 6 }
$$

$$
2 1 . 3 7 _ { \pm 1 . 2 9 }
$$

$$
2 0 . 2 4 _ { \pm 2 . 3 8 }
$$

$$
2 8 . 5 6 _ { \pm 1 . 0 5 }
$$

$$
6 7 . 0 6 { \scriptstyle \pm 1 . 3 7 }
$$

$$
6 0 . 7 1 { \scriptstyle \pm 1 . 1 9 }
$$

$$
5 7 . 5 4 _ { \pm 3 . 4 4 }
$$

$$
5 7 . 1 4 _ { \pm 2 . 3 8 }
$$

$$
6 4 . 6 8 _ { \pm 2 . 4 8 }
$$

$$
6 0 . 3 2 { \scriptstyle \pm 1 . 8 1 }
$$

$$
5 6 . 4 _ { \pm 0 . 7 7 }
$$

$$
7 1 . 3 7 { \scriptstyle \pm 1 . 2 }
$$

$$
3 2 . 9 4 _ { \pm 3 . 4 4 }
$$

$$
2 4 . 8 1 _ { \pm 1 . 7 3 }
$$

$$
3 2 . 9 4 _ { \pm 2 . 4 8 }
$$

$$
3 5 . 7 1 _ { \pm 2 . 0 6 }
$$

$$
3 4 . 1 3 _ { \pm 1 . 8 2 }
$$

$$
3 3 . 3 3 _ { \pm 1 . 1 9 }
$$

$$
3 4 . 8 7 _ { \pm 1 . 7 4 }
$$

$$
3 0 . 7 5 _ { \pm 2 . 2 2 }
$$

$$
4 9 . 6 _ { \pm 1 . 3 7 }
$$

$$
4 9 . 6 1 _ { \pm 1 . 3 9 }
$$

$$
5 8 . 7 3 { \scriptstyle \pm 0 . 6 9 }
$$

$$
5 3 . 1 7 _ { \pm 2 . 4 8 }
$$

$$
5 0 . 4 _ { \pm 0 . 6 9 }
$$

$$
5 1 . 9 8 _ { \pm 2 . 4 8 }
$$

$$
5 3 . 5 7 _ { \pm 3 . 1 5 }
$$

$$
6 1 . 4 4 _ { \pm 0 . 6 4 }
$$

$$
\operatorname { A v g }
$$

$$
4 0 . 0 1 _ { \pm 0 . 4 1 }
$$

$$
3 6 . 5 9 _ { \pm 0 . 3 9 }
$$

$$
3 8 . 6 2 _ { \pm 0 . 6 4 }
$$

$$
3 9 . 1 5 _ { \pm 1 . 5 }
$$

$$
4 1 . 0 7 _ { \pm 0 . 4 2 }
$$

$$
4 0 . 0 4 _ { \pm 0 . 4 6 }
$$

$$
3 9 . 4 8 _ { \pm 0 . 9 9 }
$$

$$
4 4 . 5 2 _ { \pm 0 . 5 4 }
$$