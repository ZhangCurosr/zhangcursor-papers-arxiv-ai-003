# Semantic Map Sharing and Capability-Aware Coverage Planning for AI-Native 6G Robotic Coordination

Abdulqader Dhafer<sup>1,2</sup>, Qi Wang<sup>1</sup>, and Zhou Daniel Hao<sup>1,2</sup>

Abstract—Search and Rescue (SAR) operations increasingly deploy heterogeneous teams of aerial and ground robots. However, conventional coverage methods typically do not translate perceived terrain into platform-specific reachability, while continuous image exchange imposes a high communication cost. We propose an edge-centric, semantic-aware coverage planning framework that integrates aerial terrain perception, robotspecific traversability reasoning, and payload-efficient semantic state sharing. Aerial observations are converted into compact semantic grid maps, enabling reachability-constrained area decomposition and capability-aware coverage paths that assign only regions admitted by each robot’s capability profile. The resulting perception–sharing–planning loop feeds semantic corrections into traversability reasoning and replanning, forming an application-level mechanism motivated by AI-enabled goaloriented communication envisioned for AI-native 6G networks. For the high-update case, transmitting semantic corrections reduces the application payload by a factor of approximately 82 relative to periodic full-map sharing. Across matched benchmark scenarios, the proposed method achieved 91.5% coverage with no capability-infeasible allocations, compared with 78.8% coverage and a 21.5% capability-infeasible allocation rate for LS-MCPP. Semantic corrections update the shared planning state without requiring repeated transmission of the complete map.

Index Terms—Search and Rescue Robotics, Multi-robot Coverage Path Planning, Semantic Communication, AI-Native 6G Coordination.

## I. INTRODUCTION

Disaster response operations increasingly utilize heterogeneous teams of Unmanned Aerial Vehicles (UAVs) and ground robots [1], [2] to explore environments that are unsafe or inaccessible to human responders. UAVs are more effective for surveying flooded or structurally collapsed areas, while ground robots provide close-range inspection where terrain conditions permit, including regions with loose debris, dense vegetation, or obstructing tree branches. However, these platforms can struggle to function as a coherent team. Aerial robots may observe regions that a ground robot cannot safely traverse due to structural collapse, unstable terrain, or flooding, leading to spatial mismatches, infeasible task assignments, and redundant exploration. These challenges highlight a critical disconnect between conventional Coverage Path Planning (CPP), which typically assumes homogeneous mobility and centralized global maps, and emerging search-and-rescue (SAR) deployments in which heterogeneous robots coordinate through distributed, edge-centric architectures envisioned for 6G-enabled systems.

Effective collaboration therefore depends not only on robot autonomy but also on exchanging mission information needed for coordination. Research on 6G networks envisions communication, edge computing, and AI inference as integrated components of distributed decision loops rather than as isolated functions [3], [4]. For SAR robot teams, the value of an observation lies not in accurately reconstructing the original sensor data, but in whether it changes the semantic state used for terrain traversability and coverage assignment. The proposed mechanism therefore integrates AI-derived semantics across the entire coordination loop: a deep learning model extracts terrain semantics, these AI-derived constraints dictate capability-aware path planning, and the resulting state changes are exchanged through application-level goal-oriented communication rather than continuous imagery [5].

Communication capabilities alone are insufficient if the underlying planner does not account for heterogeneous traversability [6], [7]. Disaster environments often contain adjacent regions with different traversability constraints, including damaged infrastructure, debris, and flooding, requiring platform-specific decisions about which robots can safely and effectively access each area [8], [9]. Coverage planning must therefore treat reachability as a primary constraint rather than rely solely on efficiency or spatial proximity. Existing research has largely considered semantic perception [10], [11], multi-robot coverage planning [12], [13], and communication efficiency [6], [14] as separate problems. However, simply combining these components is insufficient because standard planners typically assume a shared terrain graph and do not enforce robot-specific traversability constraints, potentially resulting in infeasible path assignments. A gap therefore remains between identifying terrain semantics, translating them into robot-specific coverage assignments, and efficiently exchanging the resulting state during distributed operation.

To address these challenges, this paper makes the following contributions: (i) an edge-centric coordination framework that structurally couples semantic perception, capability-aware coverage planning, and semantic information exchange for heterogeneous SAR robot teams, as shown in Figure 1; (ii) a reachability-constrained Voronoi decomposition and coverageplanning method that embeds platform-specific traversability constraints directly into the geodesic expansion to ensure feasible coverage and support local replanning; and (iii) a goaloriented semantic state-sharing mechanism that exchanges incremental semantic-map corrections rather than continuous imagery.

## II. RELATED WORK

## A. Perception and Semantic Mapping for Disaster Robotics

Semantic perception from overhead imagery is increasingly used in search-and-rescue (SAR) operations [10], [15]. Disaster datasets such as RescueNet [8] and FloodNet [9] have advanced the capability of deep learning models to segment structural damage, debris, and flooded areas. However, existing work generally uses these semantic labels for damage assessment rather than translating them into robot-specific traversability constraints for coverage planning. Large-scale satellite datasets such as xBD extend assessment to regional building damage, yet their building-focused annotations do not describe the local access conditions required for ground navigation [16]. In this context, there remains a need for semantic representations that translate overhead scene information into robot-specific traversability constraints for coverage planning.

## B. Coverage and Path Planning for Heterogeneous Multi-Robot Systems

Coverage Path Planning (CPP) aims to achieve systematic exploration of an environment while minimizing redundant motion and execution time. Classical methods primarily address single robots [17] or homogeneous teams [18], [19] and typically use a common motion model over geometrically defined free space. Multi-robot CPP (mCPP) extensions scale coverage through task allocation and area decomposition [20], with DARP and Voronoi partitioning balancing workload across disjoint regions [19], [21]. LS-MCPP applies local search to reduce coverage makespan, while MFC constructs rooted forest covers over weighted or unweighted terrain [22], [23]. Both operate on a shared terrain graph and do not encode robot-specific semantic traversability during allocation. Heterogeneous systems have been addressed through rolespecific UAV-UGV collaboration in partially known environments [24] and graph coverage with explicit movement and proximity constraints [25]. The former coordinates platform roles around predefined coverage paths, whereas the latter optimizes constrained covering tours on an input graph. However, neither translates perception-derived terrain labels into robotspecific reachability constraints. Our framework addresses this gap by embedding platform-specific traversability directly into semantic-grid decomposition.

## C. AI-Native 6G Semantic Coordination

Reliable, low-latency communication is a critical enabler for multi-robot SAR operations, particularly in time-critical and safety-sensitive missions. Research toward sixth-generation (6G) wireless networks envisions communication infrastructures that support edge intelligence, sub-millisecond radio latency, and ultra-reliable connectivity [3], [14]. Within this vision, the impracticality of continuous raw sensor streaming for large heterogeneous teams motivates Semantic Communication (SemCom), which shifts the objective from reconstructing source data to delivering task-relevant information for downstream decisions [5]. Such goal-oriented communication is especially relevant in dynamic disaster environments, where delayed semantic updates can result in inconsistent semantic maps, redundant coverage, or infeasible assignments.

Edge intelligence provides a practical basis for implementing semantic coordination in multi-robot SAR. Edgeenabled architectures allow perception and planning to remain onboard each robot, while nearby computing services can maintain and distribute the shared environmental state [4]. This arrangement aligns with the AI-native 6G vision, in which communication and edge computing support distributed AIbased perception and planning. In the proposed framework, incremental semantic-map corrections reduce the need to transmit continuous imagery while updating the traversability information used for coverage assignment and replanning.

## III. PROPOSED APPROACH

To address capability-infeasible allocations in heterogeneous teams, the proposed framework converts aerial observations into semantic grid maps for reachability-constrained decomposition, local path planning, and incremental state sharing.

## A. Semantic Environment Representation

The disaster environment is represented as a semantic grid map derived from aerial imagery using a YOLOv11 segmentation model [26], which produces pixel masks for terrain and structural elements relevant to search-and-rescue operations. To support coverage planning, the segmented output is partitioned into an operator-defined $R \times C$ grid, whose dimensions control the spatial resolution of the map. Each grid cell is assigned the most frequent predicted semantic class among its labeled pixels, producing a compact representation of obstacles and terrain semantics. Cells without a predicted terrain label are marked as unknown, treated as impassable during planning, and excluded from the coverage denominator. The resulting grid serves as the input for capability-aware area decomposition and path planning, as shown in Figure 2.

## B. Reachability-Constrained Area Decomposition and Path Planning

Area decomposition and path planning are performed over the semantic map to assign disjoint coverage regions and generate feasible paths for each agent. The environment is first partitioned into robot-specific regions based on terrain information from the semantic map and each robot’s traversability constraints. Local coverage targets are then selected independently within each assigned region.

![](images/3d55691492b852677a176dd6dba91798244d2883e64d36ad3227276ca47ae6cc.jpg)  
Fig. 1. Framework overview. Aerial or satellite imagery is converted into a semantic map and combined with robot capabilities and initial positions for are partitioning and coverage planning. The semantic map and path assignments are distributed to aerial and ground robots, where onboard functions support local event detection and replanning. Isaac Sim provides the evaluation testbed.

1) Reachability-Constrained Decomposition: Let $G \in \mathbf { \Omega }$ $\mathbb { Z } ^ { R \times C }$ denote a discrete semantic map, where each cell encodes a terrain class inferred from aerial semantic segmentation. Each robot $r _ { i }$ is associated with a capability set

$$
\mathcal { T } _ { i } = \{ t _ { 1 } , t _ { 2 } , . . . \} ,\tag{1}
$$

representing the semantic classes admitted by the platform’s mobility constraints and the operator-defined requirements of the current mission.

The environment is partitioned into disjoint, robot-specific coverage regions using a discrete geodesic (shortest-path) Voronoi decomposition [27] computed on the semantic grid. Geodesic distances are computed along the grid graph while respecting robot-specific terrain traversability, thereby accounting for obstacles and heterogeneous capabilities. For each robot $r _ { i } ,$ a wavefront expansion is performed using Breadth-First Search (BFS), initialized from the robot’s start cell. During expansion, neighboring cells are explored only if they satisfy the constraint $G ( x , y ) \in \mathcal { T } _ { i }$ . Cells that violate the robot’s capability set are treated as impassable, embedding terraindependent traversability directly into the distance computation. This process yields a distance map $d _ { i } ( x , y )$ representing the minimum number of grid steps required for robot $r _ { i }$ to reach cell $( x , y )$ . Each semantic cell that is reachable by at least one robot is assigned to the robot with the minimum geodesic cost according to

$$
\pi ( x , y ) = \arg \operatorname* { m i n } _ { i : d _ { i } ( x , y ) < \infty } d _ { i } ( x , y ) .\tag{2}
$$

Cells that cannot be reached from any robot’s starting position through terrain permitted by its capability set remain unassigned.

2) Local Coverage Path Planning: Following the area decomposition, an initial coverage path is computed for each robot to cover its assigned region. The robot executes this path locally using the semantic grid map. During execution, onboard observations update the semantic map, and the path is replanned when changes in terrain or obstacles are detected. Path construction proceeds incrementally by selecting the nearest uncovered assigned cell in Manhattan distance, with feasible routes generated using capability-constrained $\mathbf { A } ^ { * }$ search over the semantic grid. Uncovered assigned cells encountered along each path segment are marked as covered, enabling implicit path stitching that reduces backtracking and traversal overhead while covering the reachable cells assigned to each robot.

## C. Goal-Oriented Semantic State Sharing

Semantic information exchange is performed through compact, grid-based updates rather than raw imagery or complete semantic-map broadcasts, reflecting the bandwidth-efficient and latency-sensitive communication patterns envisioned for emerging 6G-enabled collaborative robotic systems. Each ground robot and UAV maintains a local semantic grid, initialized from a shared map and updated through onboard perception. When terrain traversability changes or previously unknown obstacles are observed, only the affected semantic cells are shared. These updates encode the location and revised semantic state of the affected cells for integration into each agent’s local map. Due to the distributed architecture, agents may temporarily operate with incomplete or outdated semantic maps. Each robot therefore plans from its local representation and invokes onboard replanning when newly observed terrain conflicts with the current map. Evaluating robustness to communication delays and inter-agent map inconsistencies is left for future work. Overall, lightweight semantic-cell updates combine local perception and replanning in an applicationlevel coordination loop motivated by the AI-native 6G vision.

## IV. DATASETS

Robust semantic understanding of disaster environments is essential for reliable multi-robot coverage planning. However, labeled data for post-disaster scenarios remain scarce and inconsistent across datasets. To address this limitation, a custom dataset was constructed by integrating three opensource datasets that capture complementary viewpoints, scales, and disaster types. The final dataset comprises a total of 5,234 images partitioned into training, validation, and testing sets using an 80/10/10 ratio. Combining these sources provides varied terrains, viewpoints, spatial scales, and disaster conditions relevant to heterogeneous robot teams. RescueNet provides high-resolution post-Hurricane Michael UAV imagery with dense annotations of structural damage and navigable ground surfaces [8]. The Semantic Segmentation Satellite Imagery Dataset contains 261 high-resolution urban images from Houston, Texas [28] and was included to provide additional examples of intact buildings and urban structures. FloodNet contains approximately 2,343 images collected after Hurricane Harvey, including annotations for flooded roads and buildings [9].

![](images/683a55838b7cb37a0ab573775e3666e384f033a129522b4575023ff281d05b75.jpg)  
Fig. 2. Semantic perception pipeline from aerial imagery to a grid-based semantic map. Pixel masks are aggregated by dominant class into a discrete semantic grid for capability-aware multi-robot planning.

## A. Data Pre-processing

All images and annotation masks were converted to a common format and resized to a uniform resolution compatible with the YOLO-based segmentation architecture. The source datasets use partially overlapping labels at different levels of granularity. We therefore mapped their annotations to a common taxonomy of 10 terrain classes (Table I). The taxonomy retains distinctions in structural damage and terrain accessibility needed for robot-specific traversability reasoning and coverage planning.

## V. EXPERIMENTATION

## A. Segmentation Model

The semantic segmentation model was trained on a composite dataset designed to capture heterogeneous spatial scales and viewpoints typical of aerial disaster assessment. Performance was evaluated using pixel-level precision, recall, and micro

TABLE I  
TERRAIN CLASSES USED FOR SEMANTIC SEGMENTATION
<table><tr><td>ID Class</td><td></td><td>Description</td></tr><tr><td>0</td><td>Water</td><td>Flooded areas, standing water, and ponds</td></tr><tr><td></td><td>1 Building No Damage</td><td>Intact structures with no visible damage</td></tr><tr><td></td><td>2 Building Medium Damage</td><td>Visible structural or roof damage; structure remains standing</td></tr><tr><td></td><td>3 Building Major Damage</td><td>Partial collapse or severe structural failure</td></tr><tr><td>4</td><td>Building Total Destruction</td><td>Rubble or fully collapsed structures</td></tr><tr><td>5 6</td><td>Vehicle</td><td>Cars and trucks</td></tr><tr><td>7</td><td>Clear Road</td><td>Unobstructed roads and lands</td></tr><tr><td>8</td><td>Blocked Road Tree</td><td>Roads obstructed by debris Vegetation and trees</td></tr><tr><td>9</td><td></td><td></td></tr><tr><td></td><td>Pool</td><td>Man-made swimming pools</td></tr></table>

F1-score. Evaluation on 523 held-out test images yielded a precision of 0.883, a recall of 0.837, and a micro F1-score of 0.859. The higher precision indicates fewer false-positive terrain labels, while dominant-class aggregation may limit the influence of isolated pixel-level errors. Lower performance was observed for sparsely represented classes, including Building Medium Damage, Building Major Damage, Building Total Destruction, Blocked Road, and Pool, likely reflecting limited training samples and visual ambiguity. Building damage classes also exhibited lower accuracy under oblique viewpoints, consistent with the dominance of top-down imagery in the training data [10], [11]. However, misclassification of a cell’s dominant terrain class can still affect traversability decisions and subsequent coverage assignments.

## B. Isaac Sim Feasibility Evaluation

A representative post-disaster urban environment was constructed in Isaac Sim [29] for a controlled feasibility evaluation of the proposed semantic perception pipeline and heterogeneous coverage-planning algorithm. The simulated scene emulates search-and-rescue conditions, including partially collapsed buildings and debris-obstructed roads. Figure 3 shows the aerial and ground robot deployment together with the discretized semantic grid representation.

Ten Isaac Sim trials were conducted using heterogeneous teams of quadruped ground robots and aerial robots as a closed-loop feasibility evaluation. Semantic observations were converted into grid maps and used for capability-aware assignment and path planning. The initial planning pass assigned approximately 85% of grid cells, with the remaining cells left unassigned because of conservative capability constraints, segmentation uncertainty, and variations in robot starting positions. During execution, local traversal conflicts were identified in fewer than 10% of assigned cells, primarily where dominant-class aggregation concealed non-traversable content within cells whose dominant label was feasible for the assigned robot. These conflicts were subsequently corrected through semantic-map updates and local replanning. The larger semantic-grid benchmark in Section V-C provides the reproducible quantitative comparison against other planners.

![](images/cfa1fd111de193f729d9f06060095aed4038803560fb0eaab5464e549d9443f5.jpg)  
Fig. 3. Multi-robot semantic coverage in Isaac Sim. Snapshot of a simulated disaster environment showing aerial robots (green) and quadruped ground robots (blue). Red cubes denote discretized semantic grid cells used to encode robot-specific traversability and coverage maps. Magnified insets show one aerial and one ground robot during execution.

## C. Quantitative Planner Benchmark

We evaluate the proposed method against two baseline groups: classical capability-blind decomposition and graphbased mCPP planners. The classical baselines are DARP [19] and a modified form of boustrophedon coverage [17], while the graph-based planners are LS-MCPP [22] and Multi-Robot Forest Coverage (MFC) [23]. The modified baseline divides the grid into equal contiguous column regions without using terrain capabilities, then orders each robot’s feasible targets using an alternating row-wise sweep. The benchmark uses one ground robot and one UAV with ${ \mathcal { T } } _ { \mathrm { g r o u n d } } = \{ 6 , 7 , 8 \}$ and $\mathcal { T } _ { \mathrm { U A V } } = \{ 0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 9 \}$ , using the class identifiers in Table I. Each capability set combines platform mobility constraints, which determine where the robot can physically move, with operator-defined mission constraints. In the proposed method, these sets constrain both decomposition and path construction. Cells outside a robot’s set are neither traversed nor overflown nor assigned for coverage. The ground-robot set describes admissible terrain traversal, while the UAV set describes admissible aerial movement and coverage. These sets are scenario-specific and can be adapted to other platforms or missions. All planners are evaluated on identical 20 × 20 semantic grids across a common set of 814 randomized configurations. Of 900 attempted configurations, 85 were excluded because the DARP implementation either failed on disconnected free-space grids or did not converge, and one configuration was incompatible with MFC’s contracted-root representation. Coverage is the percentage of segmented cells covered by a capable robot. Capability-infeasible allocation is the percentage of robot–cell allocation pairs outside the corresponding capability set; for LS-MCPP and MFC, allocations are recovered from the unique grid cells in each completed coverage walk.

TABLE II  
MEAN PLANNER PERFORMANCE OVER THE COMMON CONFIGURATION SET.
<table><tr><td>Planner</td><td>Cov. (%)↑</td><td>Infeas. alloc. (%)↓</td></tr><tr><td>Proposed</td><td>91.5</td><td>0.0</td></tr><tr><td>DARP [19]</td><td>74.2</td><td>21.0</td></tr><tr><td>Modified boustrophedon [17]</td><td>71.9</td><td>22.6</td></tr><tr><td>LS-MCPP [22]</td><td>78.8</td><td>21.5</td></tr><tr><td>MFC rooted-tree cover [23]</td><td>79.1</td><td>21.5</td></tr></table>

Table II reveals two consistent patterns. First, DARP and the modified boustrophedon baseline produce capability-infeasible allocation rates of 21.0% and 22.6%, respectively, while LS-MCPP and MFC both produce rates of approximately 21.5%. The proposed capability-aware method produces none. Second, these infeasible allocations are accompanied by lower feasible coverage: 74.2% for DARP, 71.9% for the modified boustrophedon baseline, 78.8% for LS-MCPP, and 79.1% for MFC, compared with 91.5% for the proposed method. Although the graph-based planners improve coverage over the classical partitions, their common traversability model does not prevent platform-infeasible allocations. By incorporating robot-specific geodesic reachability during decomposition, the proposed method assigns cells only to robots with a capabilityvalid path from their starting position.

## D. Discretization Resolution Analysis

The semantic grid compresses each cell to its dominant class, so mixed cells can hide small non-traversable regions. To quantify this information loss, we evaluate grid resolution from $8 \times 8 ~ \mathrm { t o } ~ 4 0 \times 4 0$ over 80 scenes. Intra-cell impurity measures the proportion of pixels that differ from a cell’s dominant semantic class, while the hidden non-traversable fraction measures nontraversable pixels concealed by a traversable dominant class. Intra-cell impurity decreases from 15.2% at $8 \times 8$ to 3.3% at $4 0 \times 4 0 .$ , while the hidden non-traversable fraction falls from 7.0% to 1.5%. The simulation experiments use a $2 0 \times 2 0$ grid, for which the corresponding impurity and hidden nontraversable fractions are 6.9% and 3.1%, respectively. Finer grids also increase the number of discrete path steps, with the maximum per-robot path length (makespan) growing from roughly 59 to 1,380 grid-cell steps across the resolution sweep. The appropriate resolution is not fixed, as it scales with the covered area and altitude. Images captured at a higher altitude represent a larger physical area, so the grid dimensions should be increased to keep the area represented by each cell approximately consistent. Figure 4 summarizes the resolution sweep.

![](images/a8a766100a6653ec55327005a12d4b779c5fe7078758cb04e06846036f74334f.jpg)

![](images/a81aecf3b71b9066616b06437bda57311da8436b2f0e2cf78f9203267d990817.jpg)  
Fig. 4. Effect of grid resolution on discretization fidelity and grid-step makespan across 80 scenes. (a) Finer grids reduce intra-cell class mixing, hidden non-traversable area, and the proportion of highly mixed cells. (b) Planning makespan increases rapidly with resolution.

## E. Application-Payload Analysis for Semantic State Sharing

We compare the application payload required for uncompressed aerial-image streaming, periodic and event-triggered full semantic-map sharing, and delta-based semantic updates. Raw image transmission is treated as a reference upper bound on communication cost. An RGB image of resolution $1 9 2 0 ~ \times ~ 1 0 8 0$ requires approximately 6.2 MB per frame, resulting in a data rate of 62 MB/s per robot at 10 Hz, excluding compression and protocol overhead. The aggregate source traffic increases linearly with the number of robots. As lower-complexity baselines, we consider sharing complete semantic grid maps. For $\mathrm { ~ a ~ } ~ 2 0 ~ \times ~ 2 0$ grid with one byte per cell, a full update requires 400 bytes, corresponding to 4.0 kB/s at the same frequency. Full-map sharing therefore reduces the application payload by approximately four orders of magnitude relative to uncompressed image streaming, but repeatedly transmits unchanged cells. To isolate the effect of delta encoding from that of event-triggered transmission, we additionally consider an event-triggered full-map baseline, in which the complete semantic map is transmitted whenever a semantic change is detected.

The proposed exchange mechanism transmits an update only when onboard observation identifies a difference from the stored semantic map. Let f denote the semantic observation frequency, q the probability that an observation produces an update, n¯ the mean number of changed cells in a non-empty update, b the bytes used to encode each changed cell, and h the application-message header. The expected application-payload rate is

$$
R _ { \Delta } = f q ( \bar { n } b + h ) .\tag{3}
$$

Using $f = 1 0 ~ \mathrm { H z }$ $b = 3$ B for a two-byte cell index and one-byte semantic label, $h = 4 \ B ,$ , and $\bar { n } = 1$ , the payload ranges from 0 to 70 B/s depending on the frequency of semantic corrections. Under the high-update assumption $q =$ 0.70, event-triggered full-map sharing requires approximately 2.8 kB/s, while the proposed delta updates require 49 B/s. This corresponds to a payload reduction of approximately $5 7 \times$ relative to event-triggered full-map sharing and $8 2 \times$ relative to periodic full-map sharing.

Each semantic correction updates the terrain state used for traversability reasoning and coverage assignment. The resulting exchange is therefore goal-oriented at the application layer, as robots share planning-relevant changes rather than continuous imagery. These values quantify application payload under the stated encoding and update assumptions, rather than end-to-end network performance.

## VI. CONCLUSION

This paper presented an edge-centric semantic coverage framework for heterogeneous SAR robot teams, in which terrain labels derived from aerial imagery constrain robotspecific reachability before spatial decomposition and incremental corrections update the shared planning state. The segmentation model achieved a micro F1-score of 0.859 on 523 held-out images. In the common benchmark, the proposed method achieved 91.5% coverage with no capabilityinfeasible allocations, compared with 78.8% coverage and a 21.5% capability-infeasible allocation rate for LS-MCPP. Representing the resulting planning state as a $2 0 \times 2 0$ semantic grid reduced the application payload by approximately four orders of magnitude relative to uncompressed imagery. Under the high-update assumptions used in the payload analysis, the delta-update model yields 49 B/s, a reduction factor of approximately 82× relative to periodic full-map sharing. These results show that semantic corrections are exchanged for their effect on traversability reasoning, coverage assignment, and replanning rather than to reconstruct the underlying imagery. Coupled with onboard inference and edge-centric state sharing, this decision loop provides an application-level example of AI-enabled goal-oriented communication aligned with the emerging AI-native 6G vision. Future evaluation will examine the complete loop on physical heterogeneous platforms under communication delay and inter-agent map inconsistency.

## DATA AND CODE AVAILABILITY

Due to licensing constraints on the source datasets, we do not redistribute the merged dataset or its annotations. The label-mapping and preprocessing scripts used to reproduce the merged dataset from the original sources, together with instructions for obtaining those sources, are available at https: //github.com/adhafer/semcap-cpp.

## REFERENCES

[1] I. Munasinghe, A. Perera, and R. C. Deo, “A comprehensive review of UAV–UGV collaboration: Advancements and challenges,” Journal of Sensor and Actuator Networks, vol. 13, no. 6, 2024. [Online]. Available: https://www.mdpi.com/2224-2708/13/6/81

[2] Y. Zhang, H. Yan, D. Zhu, J. Wang, C.-H. Zhang, W. Ding, X. Luo, C. Hua, and M. Q. H. Meng, “Air-ground collaborative robots for fire and rescue missions: Towards mapping and navigation perspective,” 2025. [Online]. Available: https://arxiv.org/abs/2412.20699

[3] M. K. Bahare, A. Gavras, M. Gramaglia, J. Cosmas, X. Li, O. Bulakci,<sup>¨</sup> A. Rahman, A. Kostopoulos, A. Mesodiakaki, D. Tsolkas, M. Ericson, M. Boldi, M. Uusitalo, M. Ghoraishi, and P. Rugeland, “The 6G architecture landscape – European perspective,” Feb. 2023. [Online]. Available: https://zenodo.org/records/7313232

[4] M. Zhang, M. Abdi, V. R. Dasari, and F. Restuccia, “Semantic edge computing and semantic communications in 6G networks: A unifying survey and research challenges,” Computer Networks, vol. 270, p. 111531, Oct. 2025. [Online]. Available: http://dx.doi.org/10. 1016/j.comnet.2025.111531

[5] Y. Wang, H. Han, Y. Feng, J. Zheng, and B. Zhang, “Semantic communication empowered 6G networks: Techniques, applications, and challenges,” IEEE Access, vol. 13, pp. 28 293–28 314, 2025.

[6] J. Gielis, A. Shankar, and A. Prorok, “A critical review of communications in multi-robot systems,” 2022. [Online]. Available: https://arxiv.org/abs/2206.09484

[7] J. Bravo-Arrabal, R. Vazquez-Mart´ ´ın, J. J. Fernandez-Lozano, and´ A. Garc´ıa-Cerezo, “Strengthening multi-robot systems for SAR: Codesigning robotics and communication towards 6G,” 2025. [Online]. Available: https://arxiv.org/abs/2504.01940

[8] M. Rahnemoonfar, T. Chowdhury, and R. Murphy, “RescueNet: A high resolution UAV semantic segmentation dataset for natural disaster damage assessment,” Scientific Data, vol. 10, no. 1, p. 913, 2023.

[9] M. Rahnemoonfar, T. Chowdhury, A. Sarkar, D. Varshney, M. Yari, and R. R. Murphy, “FloodNet: A high resolution aerial imagery dataset for post flood scene understanding,” IEEE Access, vol. 9, pp. 89 644–89 654, 2021.

[10] A. Sirma, A. Plastropoulos, G. Tang, and A. Zolotas, “DRespNeT: A UAV dataset and YOLOv8-DRN model for aerial instance segmentation of building access points for post-earthquake search-and-rescue missions,” 2025. [Online]. Available: https://arxiv.org/abs/2508.16016

[11] N. Le and M. Rahnemoonfar, “3D semantic segmentation for postdisaster assessment,” 2025. [Online]. Available: https://arxiv.org/abs/ 2512.24593

[12] E. Galceran and M. Carreras, “A survey on coverage path planning for robotics,” Robotics and Autonomous Systems, vol. 61, no. 12, pp. 1258–1276, 2013. [Online]. Available: https://www.sciencedirect.com/ science/article/pii/S092188901300167X

[13] R. Almadhoun, T. Taha, L. Seneviratne, and Y. Zweiri, “A survey on multi-robot coverage path planning for model reconstruction and mapping,” SN Applied Sciences, vol. 1, no. 8, 7 2019. [Online]. Available: https://doi.org/10.1007/s42452-019-0872-y

[14] M. Ghassemian, A. M. Valenzuela, A. G. Armada, D. Vukobratovic, P. Chatzimisios, K. Althoefer, and R. R. V. Prasad, “6G empowering future robotics: A vision for next-generation autonomous systems,” 2026. [Online]. Available: https://arxiv.org/abs/2602.12246

[15] T. Ruan, A. Ramesh, H. Wang, A. Johnstone-Morfoisse, G. Altindal, P. Norman, G. Nikolaou, R. Stolkin, and M. Chiou, “A framework for semantics-based situational awareness during mobile robot deployments,” 2025. [Online]. Available: https://arxiv.org/abs/2502.13677

[16] R. Gupta, R. Hosfelt, S. Sajeev, N. Patel, B. Goodman, J. Doshi, E. Heim, H. Choset, and M. Gaston, “xBD: A dataset for assessing building damage from satellite imagery,” 2019. [Online]. Available: https://arxiv.org/abs/1911.09296

[17] H. Choset and P. Pignon, “Coverage path planning: The boustrophedon cellular decomposition,” in Field and Service Robotics, A. Zelinsky, Ed. London: Springer London, 1998, pp. 203–209.

[18] X. Zhou, X. Liu, X. Wang, S. Wu, and M. Sun, “Multi-robot coverage path planning based on deep reinforcement learning,” in 2021 IEEE 24th International Conference on Computational Science and Engineering (CSE), 2021, pp. 35–42.

[19] A. C. Kapoutsis, S. A. Chatzichristofis, and E. B. Kosmatopoulos, “DARP: Divide areas algorithm for optimal multi-robot coverage path

planning,” Journal of Intelligent & Robotic Systems, vol. 86, pp. 663– 680, 01 2017.

[20] J. Gong, H. Kim, and S. Lee, “Resilient multi-robot coverage path redistribution using boustrophedon decomposition for environmental monitoring,” Sensors, vol. 24, no. 23, 2024. [Online]. Available: https://www.mdpi.com/1424-8220/24/23/7482

[21] L. Wu, M. A. Garc´ıa, D. Puig, and A. Sole, “Voronoi-based space´ partitioning for coordinated multi-robot exploration,” Journal ofPhysical Agents, vol. 1, pp. 37–44, 01 2007.

[22] J. Tang and H. Ma, “Large-scale multi-robot coverage path planning via local search,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 38, 2024, pp. 17 567–17 574.

[23] X. Zheng, S. Koenig, D. Kempe, and S. Jain, “Multirobot forest coverage for weighted and unweighted terrain,” IEEE Transactions on Robotics, vol. 26, no. 6, pp. 1018–1031, 2010.

[24] G. G. R. de Castro, T. M. B. Santos, F. A. A. Andrade, J. Lima, D. B. Haddad, L. d. M. Honorio, and M. F. Pinto, “Heterogeneous´ multi-robot collaboration for coverage path planning in partially known dynamic environments,” Machines, vol. 12, no. 3, 2024. [Online]. Available: https://www.mdpi.com/2075-1702/12/3/200

[25] D. Mutzari, Y. Aumann, and S. Kraus, “Heterogeneous multi-robot graph coverage with proximity and movement constraints,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, 2025, pp. 14 646– 14 654.

[26] G. Jocher and J. Qiu, “Ultralytics YOLOv11,” https://github.com/ ultralytics/ultralytics, 2024, version 11.0.0.

[27] J. S. B. Mitchell, D. M. Mount, and C. H. Papadimitriou, “The discrete geodesic problem,” SIAM Journal on Computing, vol. 16, no. 4, pp. 647–668, 1987.

[28] J. Alchimowicz, “Semantic segmentation of satellite imagery,” Jun 2022. [Online]. Available: https://figshare.com/collections/semantic segmentation satellite imagery/6026765/1

[29] NVIDIA, “Isaac Sim,” https://github.com/isaac-sim/IsaacSim, 2025, version 5.1.0, Apache-2.0 License.