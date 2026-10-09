## Neuro-Memory Fuzzy Inference System for Mimicking Human-like Car Following Behavior

## Nazmul Haque

Assistant Professor   
Accident Research Institute (ARI)   
Bangladesh University of Engineering and Technology (BUET)   
Dhaka 1000, Bangladesh   
Email: nhaque@ari.buet.ac.bd

## Md Asif Raihan\*

Professor

Accident Research Institute (ARI)

Bangladesh University of Engineering and Technology (BUET)

Dhaka 1000, Bangladesh

Email: raihan@ari.buet.ac.bd

## Md. Hadiuzzaman

Professor

Department of Civil Engineering

Bangladesh University of Engineering and Technology (BUET)

Dhaka 1000, Bangladesh

Email: mhadiuzzaman@ce.buet.ac.bd

Total number of pages: 20

Submission date: August 1, 2026

\*Corresponding Author

Haque, Raihan, and Hadiuzzaman

## ABSTRACT

This study presents the Neuro-Memory Fuzzy Inference System (NeMeFIS), a hierarchical machine learning architecture that asymmetrically models acceleration and deceleration in car-following behavior by integrating five human memory types—procedural, working, episodic, semantic, and declarative. By linking external variables to memory functions via metaheuristics and validating them through factor and p-value analyses, NeMeFIS uncovers latent cognitive influences across Arterial, Collector, and Rural Highway corridors for different types of vehicles. Results from 54 different trained models emphasize cognitive thresholds shaped by driver perception limits and cognitive load. The trained NeMeFIS models outperform traditional statistical and conventional machine learning models in replicating realistic driving behavior, including comparisons with Linear Regression, ANFIS, and LSTM architectures. Fuzzy rule analysis reveals that declarative memory demands the highest rule, especially during deceleration, indicating complex braking decisions. Procedural memory drives acceleration, while semantic and declarative memory guide deceleration. Risk perception also emerges as a key factor, particularly on urban roads. Validated on both heterogeneous and homogeneous datasets, NeMeFIS offers a robust framework for modeling driver cognition. The findings support psychotherapeutic applications and the development of adaptive, human-like decision systems in Connected and Autonomous Vehicles (CAVs) to enhance traffic safety.

## INTRODUCTION

Car-Following (CF) models are essential for analyzing vehicle interactions and longitudinal dynamics using variables like relative speed, spacing, and velocity (Haque et al. 2024). They typically conceptualize the following vehicle as a reactor to stimuli from the leading vehicle, such as relative speed or desired spacing, with responses reflected in velocity or acceleration adjustments (Khodayari et al. 2010). These models support ITS applications, including Advanced Driver Assistance Systems (ADAS) and Automated Driving Systems (ADS), facilitating advancements in automated and connected vehicles by modeling driver behavior and transferring human skills to intelligent systems (Khodayari et al. 2011). As ITS technologies evolve, understanding vehicle interaction under these systems is vital (Ma and Andreasson 2007). While CF models often define vehicles as "Followers" and "Leaders," real-world traffic involves complex interactions with multiple vehicles, requiring drivers to make continuous decisions for safety and collision avoidance, demanding significant cognitive and psychological engagement (Ranney 1999).

Human memory plays a pivotal role in driving decisions, influencing actions based on past experiences, learned knowledge, and intuitive reasoning (Dougherty et al. 1999; Schooler and Hertwig 2005; Shadlen and Shohamy 2016). However, incorporating human memory into predictive models has been challenging, as it requires capturing intricate mental representations essential for simulating human-like driving behaviors (Zhao et al. 2022). Early CF models, such as Wiedemann (1974) and Fritzsche (1994), introduced psycho-physical thresholds to address perceptual limits, but these models failed to comprehensively account for the psychological processes underlying driver behavior. Boer (1999) criticized these models for their simplification of human behavior and assumption of optimal driver performance. Later advancements integrated factors such as visual angles (Jin et al. 2011) , driving risk (Hamdar et al. 2008 ; Hamdar et al. 2015 ; Talebpour et al. 2011) , and distractions or errors (Lee et al. 2008 ; Treiber et al. 2006). Machine learning techniques, including Artificial Neural Networks (ANN) (Jia et al. 2003) and Adaptive Neuro-Fuzzy Inference Systems (ANFIS) (Chakroborty and Kikuchi 2003), further enriched the field. More recently, deep learning approaches like Recurrent Neural Networks (RNN) (Zhou et al. 2017), Long Short-Term Memory (LSTM) networks (Huang et al. 2018), Gated Recurrent Units (GRU) (Tang et al. 2020), and Deep Q-learning (Tang et al. 2022) have demonstrated potential in enhancing CF models. Chen et al. (2020) introduced Long and Short-Term Driving (LSTD) characteristics to explicitly incorporate human factors.

As Connected Autonomous Vehicles (CAVs) become increasingly prevalent, there is a growing imperative to design systems that mimic human-like decision-making behaviors. Current CAVs primarily rely on rule-based algorithms and machine learning models to perform tasks such as acceleration, braking, and lane changes. While these systems excel in efficiency, safety, and compliance with traffic regulations, they often overlook the complex, adaptive nature of human decision-making, particularly in unpredictable real-world scenarios (Shalev-Shwartz et al. 2017). Human drivers account for various factors—empathy, social context, and situational awareness—when making decisions such as when to yield, how to react to emergencies, and how to navigate intricate traffic conditions (Gogoll and Müller 2017). CAVs, by contrast, lack the ability to replicate these detailed behaviors, which can lead to decisions that, while technically safe, may fail to consider the emotional and social impacts on road users (Maurer et al. 2016). For instance, rigid adherence to safety distances or traffic rules may inadvertently increase stress among human drivers or pedestrians, highlighting the need for CAVs to incorporate human-like behaviors that foster safer, more compassionate interactions. Furthermore, aligning CAV behavior with societal values and ethical considerations is vital for gaining public trust in autonomous driving technologies (Gogoll and Müller 2017; Maurer et al. 2016).

To address these challenges, this study undertakes the development of a Neuro-Memory Fuzzy Inference System (NeMeFIS), a hierarchical machine learning architecture based model that asymmetrically simulates acceleration and deceleration behaviors by integrating five distinct types of human memory—procedural, working, episodic, semantic, and declarative. The model links memory functions to external driving variables using metaheuristic optimization and validates them through factor analysis and p-value assessments. NeMeFIS is trained, tested, and evaluated using UAV-captured naturalistic trajectory data across heterogeneous road environments, including urban arterials, collectors, and rural highways. The architecture allows disaggregated modeling across vehicle types and roadway contexts, enabling the identification of cognitive thresholds and memory-specific rule bases that govern magnitude decisions. Furthermore, numerical simulations confirm the stability and realism of the generated trajectories, establishing NeMeFIS as a robust cognitive framework for mimicking human-like driving.

The remainder of this paper is organized as follows: Section II reviews existing literature relevant to carfollowing behavior and cognitive modeling. Section III introduces the NeMeFIS architecture, detailing its hierarchical structure and the integration of memory types. Section IV describes the data acquisition process, preprocessing techniques, and analytical procedures. Section V presents the training, validation, and testing of the NeMeFIS model, alongside numerical simulations and performance evaluation. Section VI concludes the study with key findings, practical implications, and potential applications.

## RELEVANT WORKS

## Psychology-uninformed Car-Following Models

Traditional car-following models, such as the Gazis–Herman–Rothery (GHR) model (Chandler et al. 1958) , rely on stimulus-response frameworks that predict acceleration based on speed, relative speed, and spacing—largely omitting psychological dimensions. Enhancements like the Intelligent Driver Model (Treiber et al. 2000) , Gipps' model (Gipps 1981) , and Bando's model (Bando et al. 1995) focused on stability and numerical simulation but still lacked behavioral depth. A notable exception is Hamdar et al. (2008) , who incorporated Prospect Theory to model risk-taking under ambiguity.

To overcome these limitations, Machine Learning (ML) methods have gained traction for modeling car-following behavior by capturing nonlinearities and latent psychological factors. Jia et al. (2003) applied a backpropagation neural network using speed and spacing to classify driver risk levels, although validation was limited. Panwai and Dia [33] used a Radial Basis Function Network (RBFN) and fuzzy ARTMAP, mapping perception to action in stop-and-go traffic, but did not model reaction times explicitly. Huang et al. (2018) introduced an LSTM neural network to encode driving memory, improving traffic flow realism, albeit with high data demands. Aghabayk et al. (2013) adopted the LOLIMOT neuro-fuzzy system to reflect perceptual imperfections, revealing the challenge of fuzzy rule complexity and data dependency. More recently, these developments emphasize the shift toward psychologically grounded, data driven modeling to bridge the gap between realism and interpretability in driving behavior models.

## Psychology-informed Car-Following Models

Lee (1966) extended the linear GHR model by incorporating a statistical memory function to capture a driver's reaction to relative speed over time, though it aggregated behavior using basic statistical parameters. Wiedemann (1974) later introduced perceptual thresholds to define behavioral regimes (free-flow, following, emergency) using spacing and speed, marking a psycho-physical advancement. However, his model lacked cognitive components and faced issues like perceptual switching without stimuli.

ML models have since emerged to implicitly encode psychological factors. Fuzzy logic-based approaches by Chakroborty and Kikuchi (2003), Kikuchi and Chakroborty (1992), and Khodayari et al. (2011) mapped external stimuli to driver responses using fuzzy inference systems. The Fadhloun-Rakha model advanced this by integrating randomness in perception and vehicle dynamics (Fadhloun and Rakha 2020). Huang et al. (2018) used LSTM networks to model memory-informed behavior based on past patterns. Hamdar et al. (2008) and Li et al. (2021) incorporated Prospect Theory and fuzzy logic to model decision-making under uncertainty.

Several models have explicitly addressed cognitive mechanisms. Talebpour et al. (2011) introduced a regimeswitching model grounded in psychological decision theories. Tang et al. (2014) extended FVD models to incorporate driving styles and their impact on traffic flow. Jia et al. (2008) proposed a cognitive behavior model structured around perception, memory, and decisions, though lacking empirical validation. van Winsum (1999) modeled adaptive headways under fatigue, while Furutani (1976) introduced "following defense" behavior via catastrophe theory. Yi-Rong et al. (2015) integrated anticipation and response delays.

Despite progress, limitations remain in interpretability, data demands, and realistic psychological integration, underscoring future research needs.

## METHODOLOGY

## Hierarchical Fuzzy Systems

Fuzzy logic systems offer transparency in interpretation and analysis, making them suitable for complex applications. However, conventional fuzzy systems, while being universal approximators—suffer from the "curse of dimensionality," where increasing input parameters exponentially increases rule count, parameters, and data requirements. This often leads to overfitting and loss of system generalizability and transparency. To address this, Raju, Zhou, and Kisner introduced the concept of hierarchical fuzzy systems in the early 1990s (Wang et al. 2006). These systems decompose high-dimensional problems into interconnected, low-dimensional fuzzy logic units organized hierarchically. Outputs from one level serve as inputs for subsequent levels, enabling scalability and reducing complexity. He described such systems using tree structures with multiple levels and units. Mathematically, as described by Sun and Huo (2016), the first unit processes two real inputs, while subsequent units incrementally integrate previous outputs with new inputs, enhancing model interpretability and performance. This process advances till all the real inputs have been used.

Assuming there are �-input parameters $\{ X _ { 1 } , X _ { 2 } , X _ { 3 } , . . . X _ { n } \}$ and $\{ \widehat { X _ { 1 } } , \widehat { X _ { 2 } } , \widehat { X _ { 3 } } , . . . \widehat { X _ { n } } \}$ are the fuzzy variables extracted from input variables. $\cdot _ { \mathrm { m } } ,$ presents total fuzzy rules, membership functions for inputs and outputs are presented by $M _ { { \boldsymbol { \upsilon } } _ { i } ^ { j ( X _ { i } ) } }$ and $M _ { o _ { i } ^ { j ( X _ { i } ) } }$ . With singleton fuzzifier and centroid de-fuzzifier, the hierarchical system can be

represented below by equation $\operatorname { E q . } \left( 1 \right)$ and (2).

$$
I F \left( \widehat { X _ { 1 } } = U _ { 1 } ^ { j } \right) A N D \left( \widehat { X _ { 2 } } = U _ { 2 } ^ { j } \right) T H E N \left( Y _ { 1 } = O _ { 1 } ^ { j } \right)\tag{1}
$$

$$
I F \left( \widehat { X _ { \iota + 1 } } = U _ { i + 1 } ^ { j } \right) A N D \left( \widehat { Y _ { \iota - 1 } } = O _ { i - 1 } ^ { j } \right) T H E N \left( Y _ { i } = O _ { i } ^ { j } \right)\tag{2}
$$

where, ${ } ^ { \cdot } j = \{ 1 , 2 , 3 , \ldots , m \} ; U _ { 1 } ^ { j }$ and $U _ { 2 } ^ { j } ; O _ { i } ^ { j ) }$ represents the fuzzy sets of linguistic labels for input and outputs respectively; $i = \{ 1 , 2 , 3 , \ldots , n - 1 \} . \widehat { X _ { 1 } }$ and $\widehat { X _ { 2 } }$ present the real-input variable and output from the previous fuzzy logic unit respectively. The Output at different level can be recursively calculated using the Eq. (3) and (4).

$$
\begin{array} { r l } & { Y _ { 1 } = \frac { \sum _ { j = 1 } ^ { m _ { 1 } } Q _ { 1 , j } M _ { U _ { 1 } ^ { j } } ( X _ { i } ) M _ { U _ { 2 } ^ { j } } ( X _ { i } ) } { \sum _ { j = 1 } ^ { m _ { 1 } } M _ { U _ { 1 } ^ { j } } ( X _ { i } ) M _ { U _ { 2 } ^ { j } } ( X _ { i } ) } } \\ & { Y _ { i } = \frac { \sum _ { j = 1 } ^ { m _ { 1 } } Q _ { i , j } M _ { o _ { i - 1 } ^ { j } } ( Y _ { i - 1 } ) M _ { U _ { i + 1 } ^ { j } } ( X _ { i + 1 } ) } { \sum _ { j = 1 } ^ { m _ { 1 } } M _ { O _ { i - 1 } ^ { j } } ( Y _ { i - 1 } ) M _ { U _ { i + 1 } ^ { j } } ( X _ { i + 1 } ) } } \end{array}\tag{3}
$$

(4)

� $( M _ { U _ { i , j } }$ or $M _ { O _ { i , j } } )$ can adopt any membership function. If bell shaped membership function is employed, $m _ { A _ { i , j } }$ is given by Eq. (5),

$$
{ M _ { U _ { i , j } } ( X _ { i } ) = \frac { 1 } { 1 + \left[ { { \left( \frac { { X _ { i } } - { { c _ { i , j } } } } { { a _ { i , j } } } \right) } ^ { 2 } } \right] { b _ { i , j } } } , i = 1 , 2 , \dots \dots { n } , j = 1 , 2 , \dots \dots , R }\tag{5}
$$

or the Gaussian membership function by Eq. (6),

$$
M _ { U _ { i , j } } ( X _ { i } ) = e x p \left[ - \left( \frac { X _ { i } - c _ { i , j } } { a _ { i , j } } \right) ^ { 2 } \right] , i = 1 , 2 , \dots \dots n , j = 1 , 2 , \dots \dots , R\tag{6}
$$

where $a _ { i , j } , b _ { i , j } , c _ { i , j }$ are the parameters of the membership functions corresponding to input $X _ { i }$ . The parameters in this layer are usually referred to as premise parameters. $Q _ { i , j }$ is the normalized firing strength.

## The Magnitude Model: Neuro-Memory Fuzzy Inference System (NeMeFIS)

For easiness of application the study considers 3 broader types of memory 1) Sensory 2) Declarative Memory and 3) Non-Declarative Memory. The Declarative Memory are categorized into three types 1) Working memory 2) Episodic Memory and 3) Semantic memory; as per Matthews (2015) and Wolk and Budson (2010). The NeMeFIS (Neuro-Memory Fuzzy Inference System) architecture shown in Figure 1, integrates these human memory types by including Procedural Memory (P), Working Memory (W), Episodic Memory (E), Semantic Memory (S), Declarative Memory (D), and Combined Memory (C). The system is designed to mimic human-like decision-making processes by mapping the functionality of each memory type into fuzzy inference systems (FIS) that process inputs and provide outputs based on specific memory roles.

The architecture consists of multiple subsystems (Figure 1), each representing a memory type, such as P-FIS for Procedural Memory, W-FIS for Working Memory, and so on. Inputs to the system are represented as feature vectors $( X \in \{ { \mathbf { X } } _ { P } , { \mathbf { X } } _ { W } , { \mathbf { X } } _ { E } , { \mathbf { X } } _ { S } , { \mathbf { X } } _ { D } , { \mathbf { X } } _ { C } \} )$ , each containing multiple variables related to their corresponding memory types. The whole architechture is moduled into several Fuzzy Inference System (FIS). The P-FIS, W-FIS, E-FIS and S-FIS are representative of Procedural, Working, Episodic and Semantic memory. The W-FIS, E-FIS and S-FIS connects to D-FIS for Declarative memory. And the P-FIS and the D-FIS connects to C-FIS, the combiner of the memories. Each FIS subsystem computes a specific output $( Y \in \{ { \bf y } _ { P } , { \bf y } _ { w } , { \bf y } _ { E } , { \bf y } _ { S } , { \bf y } _ { D } , { \bf y } _ { C } \} )$ based on its memory function. Procedural Memory (P-FIS) handles habitual tasks and routine actions, while Working Memory (W-FIS) processes information in real-time. Episodic Memory (E-FIS) captures contextual and time-specific information, and Semantic Memory (S-FIS) processes factual and generalized knowledge. Declarative Memory (D-FIS) integrates knowledge from Episodic and Semantic Memories, providing a consolidated representation of explicit knowledge. Combined Memory (C-FIS) further integrates outputs from Declarative Memory and other subsystems, producing a final output that encapsulates the decision-making process across all memory types.

In NeMeFIS, driving memory is encoded during training through the creation and tuning of fuzzy rules and their associated membership functions, each corresponding to one of the five memory types (procedural, working, episodic, semantic, and declarative). As the model learns from vehicle‐following data, it clusters input–output patterns—such as spacing, speed, and angular relationships—into distinct rule sets whose parameters (e.g., rule weights and membership‐function shapes) capture habitual responses, short‐term adaptations, event‐specific recollections, generalized knowledge, and integrated cognitive streams. These rule parameters and membershipfunction boundaries collectively form the “memory” surface: during inference, each new driving situation activates the relevant rules in proportion to how closely current inputs match the stored prototypes, thereby reproducing learned decision patterns.

![](images/acc393206c34e33d49b8a3ae38bbaed3edf891e7c615269ffca051ea64691977.jpg)  
Figure 1 Neuro Memory Fuzzy Inference System (NeMeFIS) Architecture

## DATA COLLECTION, PROCESSING AND ANALYSIS

## Study locations

This study employs a systematic data collection strategy, focusing on three road hierarchies of Dhaka, Bangladesh. The Arterial Road (AR) also features an 8-lane (3.65 m lane width) configuration and primarily accommodates private passenger cars (Lat: 23.833542490523236, Long: 90.41936670104396). The traffic flow rate during the experiment in this corridor was 4522 pcu/hr towards northbound and 4460 pcu/hr towards southbound. The average speed of the traffic stream (AR) was 33.41 kmph. The Collector Road (CR) comprises an 8-lane thoroughfare with four lanes (3.1 m lane width) in each direction, predominantly serving non-motorized transportation, primarily rickshaws (Lat: 23.73904181123112, Long: 90.40997296817643). The traffic flow rate during the experiment in this corridor was 4542 pcu/hr towards northbound and 4214 pcu/hr towards southbound. The average speed of the traffic stream (CR) was 20.77 kmph. The Rural Highway (RH) has a four-lane configuration with two lanes (3.65m lane width) in each direction, where inter-district buses are the primary mode of transportation (Lat: 23.69844939827767, Long: 90.51082733705677). The traffic flow rate during the experiment in this corridor was 2748 pcu/hr towards northbound and 2453 pcu/hr towards southbound. The average speed of this traffic stream (RH) was found to be 45.4 kmph.

## Experimental Set-up and Flight Plan

Video Data has been collected at 30 FPS using a DJI Mini 2 drone weighing 249 grams, equipped with a 4K HDR video recording capability and a 12MP camera sensor. The video resolution has been set to Full HD (1920 x 1080) at 32 Mbps bitrate to ensure optimal video quality for detection and classification while minimizing energy consumption. The data collection process occurred during morning peak hours from 9:00 AM to 10:00 AM on three random weekdays (AR: 8th January 2019; CR: 30th January 2023; RH: 18th February 2023).

Ensuring clear visibility and avoiding unnecessary noise in the data, days with extreme fog have been avoided in the data collection schedule. During the experiments, the drone has been flown at an altitude ranging between 23 to 27 meters (similar to the NGSIM (2007))and the viewing angle with the horizontal plane has been maintained at approximately 17 to 18 degrees. The flight has been initiated from high points at least 20 meters away from the road to minimize the impact on traffic and facilitate quick battery changes. Necessary field measurements have been taken for camera calibration and skew axis correction.

Each detected road user has a certain size represented by its bounding box information (vehicle type, centroid, width, and height) and is extracted using the framework DEEGITS proposed by Islam, Haque, and Hadiuzzaman (2024) for heterogeneous traffic stream. For simplicity, the centroid point has been taken as a simpler representation of the detected vehicle. The measurement of each road user's position in space at tracked time frame constitutes its trajectory. represents the trajectory of -th road users, which is a zipped collection of all points in the desired time interval. The microscopic measurements among the subject vehicle and the lead vehicle within the -vehicle interaction system are described below:

$$
V _ { i } ( t ) = d \big ( T _ { i } ( t ) , T _ { i } ( t - \Delta t ) \big ) / \Delta t\tag{7}
$$

$$
a _ { i } ( t ) = \big ( V _ { i } ( t - \Delta t ) - V _ { i } ( t ) \big ) / \Delta t\tag{8}
$$

$$
h _ { i j } ( x , \mathbf { y } ) = T _ { i } ( x , \mathbf { y } ) - T _ { j } ( x , \mathbf { y } ) , \forall i : i < j\tag{9}
$$

$$
\Delta x _ { i j } ( t ) = d \left( T _ { i } ( t ) , T _ { j } ( t ) \right) , \forall i : i < j\tag{10}
$$

$$
\Delta V _ { i j } ( t ) = V _ { i } ( t ) - V _ { j } ( t ) , \forall i : i < j\tag{11}
$$

$$
\begin{array} { r } { \theta _ { i j } ( t ) = \tan ^ { - 1 } \frac { \left| \mathbf { Q } \left( T _ { i } ( t ) \right) \times \mathbf { R } \left( T _ { i j } ( t ) \right) \right| } { \mathbf { Q } \left( T _ { i } ( t ) \right) . \mathbf { R } \left( T _ { i j } ( t ) \right) } , \theta _ { i j } \in [ 0 , 2 \pi ] } \end{array}\tag{12}
$$

$$
\Delta \mathbf y _ { i j } ( t ) = \Delta x _ { i j } ( t ) \cos \theta _ { i j } ( t )\tag{13}
$$

$$
T T C = \Delta \mathbf { x } _ { i j } ( t ) / \Delta V _ { i j } ( t )\tag{14}
$$

$$
R P = \frac { \left( N T T C - \operatorname* { m a x } ( N T T C ) \right) } { \left( \operatorname* { m a x } ( N T T C ) - \operatorname* { m i n } ( N T T C ) \right) } , N T T C = P _ { 1 0 } ( \omega _ { k } )\tag{15}
$$

Equations (7) to (15) are for measuring speed, acceleration/deceleration, time headway, relative spacing, relative speed, lateral gap, time to collision and risk perception (Haque et al. 2025) respectively. $V _ { i }$ and $V _ { j }$ are the speed of the corresponding vehicles � and �, ∆� is time interval in between two consecutive positions of the vehicles, � and � are the direction vectors. �(. ) is the Euclidean distance function.

## Data Analysis

Table 1 presents the correlation between nine external variables and the response variable—Relative Speed—across three corridor types (AR, CR, RH) for both acceleration (A) and deceleration (D) phases. The results reveal a strong asymmetry in how variables influence acceleration versus deceleration. Speed shows the most dominant pattern, exhibiting strong positive correlations during acceleration (e.g., 0.103 in AR, 0.134 in CR) and nearperfect negative correlations during deceleration (–0.987 in AR, –0.988 in CR, –0.967 in RH), indicating that higher speeds sharply intensify braking responses across all corridor types. Relative speed (Var 8) also plays a major role but in the opposite direction: it shows strong positive correlations during deceleration (0.826 in AR, 0.758 in CR, 0.764 in RH), confirming that larger closing speeds trigger stronger deceleration demands. In contrast, spacing-related variables (Vars 1, 2, and 5) consistently show positive correlations during acceleration and moderate negative correlations during deceleration, highlighting that larger headways promote acceleration whereas reduced spacing induces braking. Angular variables (Vars 3 and 4) display weak correlations in most cases, suggesting limited direct influence on relative speed adjustments. Risk perception (Var 9) remains weakly correlated across all scenarios, indicating a more nuanced or indirect effect on moment-to-moment speed regulation.

TABLE 1 Correlation of the External Variables with the Response Variable (i.e. Relative Speed)
<table><tr><td rowspan=1 colspan=8>AR                                 CR                                RH</td></tr><tr><td rowspan=1 colspan=2>Var</td><td rowspan=1 colspan=2>A                D</td><td rowspan=1 colspan=2>A               D</td><td rowspan=1 colspan=2>A               D</td></tr><tr><td rowspan=1 colspan=2>1</td><td rowspan=1 colspan=1>0.215</td><td rowspan=1 colspan=1>-0.321</td><td rowspan=1 colspan=1>0.282</td><td rowspan=1 colspan=1>-0.230</td><td rowspan=1 colspan=2>0.187           -0.228</td></tr><tr><td rowspan=1 colspan=2>2</td><td rowspan=1 colspan=1>-0.022</td><td rowspan=1 colspan=1>-0.299</td><td rowspan=1 colspan=1>0.061</td><td rowspan=1 colspan=1>-0.219</td><td rowspan=1 colspan=2>0.014          -0.279</td></tr><tr><td rowspan=2 colspan=2>34</td><td rowspan=1 colspan=1>0.007</td><td rowspan=1 colspan=1>0.000</td><td rowspan=1 colspan=1>0.002</td><td rowspan=1 colspan=1>-0.005</td><td rowspan=1 colspan=2>-0.002          0.014</td></tr><tr><td rowspan=1 colspan=1>0.065</td><td rowspan=1 colspan=1>-0.113</td><td rowspan=1 colspan=1>0.027</td><td rowspan=1 colspan=1>-0.058</td><td rowspan=1 colspan=1>-0.020</td><td rowspan=1 colspan=1>-0.074</td></tr><tr><td rowspan=2 colspan=2>56</td><td rowspan=1 colspan=1>0.262</td><td rowspan=1 colspan=1>-0.332</td><td rowspan=1 colspan=1>0.129</td><td rowspan=1 colspan=1>-0.235</td><td rowspan=1 colspan=1>0.130</td><td rowspan=1 colspan=1>-0.212</td></tr><tr><td rowspan=1 colspan=1>-0.002</td><td rowspan=1 colspan=1>-0.326</td><td rowspan=1 colspan=1>0.010</td><td rowspan=1 colspan=1>-0.213</td><td rowspan=1 colspan=1>0.017</td><td rowspan=1 colspan=1>-0.176</td></tr><tr><td rowspan=3 colspan=2>789</td><td rowspan=1 colspan=1>0.103</td><td rowspan=1 colspan=1>-0.987</td><td rowspan=1 colspan=1>0.134</td><td rowspan=1 colspan=1>-0.988</td><td rowspan=1 colspan=1>0.063</td><td rowspan=1 colspan=1>-0.967</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>-0.014</td><td rowspan=1 colspan=1>0.826</td><td rowspan=1 colspan=1>0.033</td><td rowspan=1 colspan=1>0.758</td><td rowspan=1 colspan=1>0.059</td><td rowspan=1 colspan=1>0.764</td></tr><tr><td rowspan=1 colspan=1>0.013</td><td rowspan=1 colspan=1>0.024</td><td rowspan=1 colspan=1>-0.025</td><td rowspan=1 colspan=1>0.018</td><td rowspan=1 colspan=2>0.053          -0.016</td></tr></table>

Corridor Types: AR = Arterial Road, CR = Collector Road, RH = Rural Highway; A=Acceleration, D=Deceleration, 1= Spacing S-L , 2 = Spacing S-NL , 3= Angle S-L , 4 = Angle L-NL , 5= Spacing Change S-L, 6 = Spacing Change S-NL , 7= Speed S 8 = Relative Speed S-NL , 9= Risk Perception; S-L = Subject-Leader, S-NL = Subject-Next Leader, L-NL = Leader-Next Leader, S = Subject Vehicle  
Traffic hysteresis phenomena is widely used to identify the trajectory pairs in the following behavior [51].

Understanding hysteresis allows researchers to properly categorize trajectory data for behavioral modeling, thereby improving the accuracy and reliability of analyses of vehicular interactions. Accordingly, hysteresis patterns were examined for all trajectory pairs to ensure that they conform to true following behavior.

## Integrating External Variables into Human Memory-Based Architecture

The grouping of variables into Procedural, Working, Episodic, and Semantic Memory reflects their theoretical roles in driving behavior and is statistically supported by factor analysis. Procedural Memory variables such as Spacing S-L and S-NL show strong loadings in the AR corridor (0.565 and 0.948), confirming their habitual monitoring function. Episodic Memory variables—Spacing Change S-L and S-NL—exhibit high loadings in the RH corridor (0.697 and 0.736), indicating their relevance for recalling recent spacing variations. Semantic Memory is validated by the very strong loading of Speed S (0.992) in the CR corridor, highlighting its role in knowledge-driven, anticipatory decisions. Further justification is presented in Table II and the subsequent discussion.

TABLE 2 Rationale Behind the Integrating Variables with Corresponding Memory Type
<table><tr><td>Memory Type</td><td>Variables</td><td>Justification</td></tr><tr><td>Procedural Memory (Xp)</td><td> $\begin{array} { r } { 1 \mathrm { - S p a c i n g ~ S - L } ( \Delta x _ { i , i - 1 } ) , } \end{array}$   ${ \ k _ { 2 - \mathrm { S p a c i n g } \mathrm { S - N L } \left( \Delta x _ { i , i - 2 } \right) } }$ </td><td>Habitual and intuitive maintenance of safe spacing, strongly loaded on factors with low uniqueness values.</td></tr><tr><td>Working Memory  $( { \bf { X } } _ { W } )$ </td><td> ${ 3 \mathrm { - A n g l e } } \mathrm { S - L } ( \theta _ { i , i - 1 } ) ,$   $4 \mathrm { - A n g l e } \mathrm { L - N L } ( \theta _ { i , i - 2 } )$ </td><td>Real-time spatial awareness and decision- making, supported by moderate factor loadings and specificity to active tasks.</td></tr><tr><td>Episodic Memory(  $\mathbf { \nabla } [ \mathbf { X } _ { E } )$ </td><td>5-Spacing Change S-L  $( \left( \Delta x _ { i , i - 1 } \right) / \Delta t ) ,$  6-Spacing Change S-NL  $( \left( \Delta x _ { i , i - 2 } \right) / \Delta t )$ </td><td>Interpretation of dynamic spacing changes, associated with event-based memory of recent traffic changes.</td></tr><tr><td>Semantic Memory  $( \mathbf { X } _ { S } )$ </td><td>7-Speed S (Vi), 8-Relative Speed  $\begin{array} { r } { \mathrm { S - N L } ( \Delta V _ { i , i - 2 } ) , } \end{array}$  9-Risk Perception  $\displaystyle \left( R _ { i , i - 1 } \right)$ </td><td>General knowledge of speed and risk, reflecting abstract cognitive processes with high factor loadings.</td></tr></table>

Table 3 summarizes the statistical significance of each external variable across Arterial (AR), Collector (CR), and Rural Highway (RH) corridors using p-value distributions. Variables such as spacing (Var 1), spacing change (Var 5–6), and speed (Var 7) show a high count of p-values < 0.05, indicating strong and consistent predictive influence on relative speed. Angle-related variables (Var 3–4) generally have higher average p-values and fewer significant cases, suggesting weaker or more context-specific effects. Speed (Var 7) is universally significant in CR and RH (14 out of 14 cases), confirming its dominant role in car-following behavior.

TABLE 3 P-Value Analysis of the External Variables
<table><tr><td rowspan=1 colspan=1>CT</td><td rowspan=1 colspan=2>Var</td><td rowspan=1 colspan=4>Min     Max     Avg.     Std.</td><td rowspan=1 colspan=2>2.5%     97.5%</td><td rowspan=1 colspan=1>#P-Value &lt; 0.05</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>1</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.86</td><td rowspan=1 colspan=1>0.19</td><td rowspan=1 colspan=1>0.33</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.86</td><td rowspan=1 colspan=1>10</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>2</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.85</td><td rowspan=1 colspan=1>0.22</td><td rowspan=1 colspan=1>0.27</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.78</td><td rowspan=1 colspan=1>7</td></tr><tr><td rowspan=3 colspan=1>AR</td><td rowspan=1 colspan=2>3</td><td rowspan=1 colspan=1>0.09</td><td rowspan=1 colspan=1>0.93</td><td rowspan=1 colspan=1>0.49</td><td rowspan=1 colspan=1>0.26</td><td rowspan=1 colspan=1>0.11</td><td rowspan=1 colspan=1>0.89</td><td rowspan=1 colspan=1>0</td></tr><tr><td rowspan=2 colspan=2>45</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.79</td><td rowspan=1 colspan=1>0.17</td><td rowspan=1 colspan=1>0.26</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.72</td><td rowspan=1 colspan=1>9</td></tr><tr><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.79</td><td rowspan=1 colspan=1>0.12</td><td rowspan=1 colspan=1>0.24</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.73</td><td rowspan=1 colspan=1>10</td></tr><tr><td rowspan=4 colspan=1></td><td rowspan=4 colspan=2>6789</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.71</td><td rowspan=1 colspan=1>0.16</td><td rowspan=1 colspan=1>0.22</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.67</td><td rowspan=1 colspan=1>7</td></tr><tr><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.88</td><td rowspan=1 colspan=1>0.06</td><td rowspan=1 colspan=1>0.23</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.60</td><td rowspan=1 colspan=1>13</td></tr><tr><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.96</td><td rowspan=1 colspan=1>0.40</td><td rowspan=1 colspan=1>0.39</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.93</td><td rowspan=1 colspan=1>5</td></tr><tr><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.88</td><td rowspan=1 colspan=1>0.33</td><td rowspan=1 colspan=1>0.27</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.84</td><td rowspan=1 colspan=1>3</td></tr><tr><td rowspan=6 colspan=1>CR</td><td rowspan=4 colspan=2>1234</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.66</td><td rowspan=1 colspan=1>0.08</td><td rowspan=1 colspan=1>0.18</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.56</td><td rowspan=1 colspan=1>12</td></tr><tr><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.80</td><td rowspan=1 colspan=1>0.16</td><td rowspan=1 colspan=1>0.26</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.72</td><td rowspan=1 colspan=1>9</td></tr><tr><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.89</td><td rowspan=1 colspan=1>0.31</td><td rowspan=1 colspan=1>0.26</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.80</td><td rowspan=1 colspan=1>4</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.93</td><td rowspan=1 colspan=1>0.20</td><td rowspan=1 colspan=1>0.30</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.88</td><td rowspan=1 colspan=1>8</td></tr><tr><td rowspan=2 colspan=2>56</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.72</td><td rowspan=1 colspan=1>0.10</td><td rowspan=1 colspan=1>0.24</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.70</td><td rowspan=1 colspan=1>12</td></tr><tr><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.55</td><td rowspan=1 colspan=1>0.10</td><td rowspan=1 colspan=1>0.18</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.54</td><td rowspan=1 colspan=1>10</td></tr></table>

<table><tr><td rowspan=1 colspan=1>CT</td><td rowspan=4 colspan=1>Var789</td><td rowspan=1 colspan=4>Min     Max      Avg.     Std.</td><td rowspan=1 colspan=1>2.5%</td><td rowspan=1 colspan=1>97.5%</td><td rowspan=1 colspan=1>#P-Value &lt; 0.05</td></tr><tr><td rowspan=3 colspan=1></td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>14</td></tr><tr><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.86</td><td rowspan=1 colspan=1>0.13</td><td rowspan=1 colspan=1>0.24</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.73</td><td rowspan=1 colspan=1>10</td></tr><tr><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.86</td><td rowspan=1 colspan=1>0.09</td><td rowspan=1 colspan=1>0.22</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.65</td><td rowspan=1 colspan=1>11</td></tr><tr><td rowspan=6 colspan=1>RH</td><td rowspan=2 colspan=1>12</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.85</td><td rowspan=1 colspan=1>0.08</td><td rowspan=1 colspan=1>0.21</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.65</td><td rowspan=1 colspan=1>12</td></tr><tr><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.75</td><td rowspan=1 colspan=1>0.14</td><td rowspan=1 colspan=1>0.23</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.68</td><td rowspan=1 colspan=1>10</td></tr><tr><td rowspan=4 colspan=1>3456</td><td rowspan=1 colspan=1>0.02</td><td rowspan=1 colspan=1>0.95</td><td rowspan=1 colspan=1>0.35</td><td rowspan=1 colspan=1>0.33</td><td rowspan=1 colspan=1>0.02</td><td rowspan=1 colspan=1>0.93</td><td rowspan=1 colspan=1>3</td></tr><tr><td rowspan=2 colspan=1>0.000.00</td><td rowspan=1 colspan=1>0.96</td><td rowspan=1 colspan=1>0.41</td><td rowspan=1 colspan=1>0.39</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.95</td><td rowspan=1 colspan=1>6</td></tr><tr><td rowspan=1 colspan=1>0.96</td><td rowspan=1 colspan=1>0.20</td><td rowspan=1 colspan=1>0.32</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.88</td><td rowspan=1 colspan=1>10</td></tr><tr><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.93</td><td rowspan=1 colspan=1>0.18</td><td rowspan=1 colspan=1>0.31</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.93</td><td rowspan=1 colspan=1>9</td></tr><tr><td rowspan=3 colspan=1></td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.38</td><td rowspan=1 colspan=1>0.05</td><td rowspan=1 colspan=1>0.11</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.35</td><td rowspan=1 colspan=1>14</td></tr><tr><td rowspan=2 colspan=1>89</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.90</td><td rowspan=1 colspan=1>0.18</td><td rowspan=1 colspan=1>0.25</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.76</td><td rowspan=1 colspan=1>8</td></tr><tr><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>0.27</td><td rowspan=1 colspan=1>0.35</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.97</td><td rowspan=1 colspan=1>8</td></tr></table>

\*See Table I for detailed notations

## MODEL DEVELOPMENT AND ASSESSMENTS

The data is split into a training set (70%), a validation set (15%), and a testing set (15%). This randomized division assigns indices to each subset, ensuring no overlap among them. The training set is used to train the architecture, the validation set fine-tunes the model's hyperparameters to enhance its generalization capabilities, and the testing set evaluates the final model's performance on unseen data. For all vehicle types, the system employs five input membership functions, eight output membership functions, 30 combined membership functions, and 36 final membership functions. However, when the focus shifts to individual vehicle types, the parameters adjust to account for heterogeneity: three input MFs, six output MFs, 24 combined MFs, and the same 36 final MFs.

The use of Sugeno-type fuzzy inference systems (SugFIS) across all memory. Gaussian MFs dominate the design for their smoothness and tunability. Each FIS defines its input ranges based on observed data variability, ensuring adaptability to dynamic traffic scenarios. During the training phase, the hierarchical FIS structure, which is designed to represent different types of human memory (e.g., Procedural, Working, Episodic, Semantic, and Declarative), is optimized using Particle Swarm Optimization (Kennedy and Eberhart 1995). The validation phase further refines the trained FIS using the validation dataset. This step adopts a tuning-based approach, employing the Pattern Search optimization (Torczon 1997) method to fine-tune the hyperparameters and ensure the model generalizes well. Finally, the optimized architecture is tested on the testing dataset to evaluate its performance on new, unseen data. Predicted outputs are generated using the test inputs, and the RMSE is calculated to quantify the model's generalization capability. The training-tuning process converges in all three corridors, i.e., AR, CR and RH. Results showed that trained, tested and validated models can adopt the relative speed fluctuations with the parameter sets considered in AR corridor for acceleration and deceleration decisions.

Table 4 presents the Root Mean Square Error (RMSE) for the model's predictions (i.e. normalized Relative Speed (m/s)) across various vehicle types and traffic corridors, categorized by training, testing, and validation. It took 385 hours to train, test, and validate the 54 models on Core i7-10700 CPUs with 24 GB RAM in parallel processing. The results demonstrate that acceleration models (A), which account for differences in acceleration and deceleration behaviors, consistently exhibit lower RMSE compared to deceleration models (D). Moreover, the results of the NeMeFIS models is compared with the traditional Linear Regression (LR) (Helly 1959), ANFIS (Kikuchi and Chakroborty 1992) and LSTM (Huang et al. 2018). NeMeFIS is compared with traditional and machine learning benchmarks. Traditional models like Gipps (1981) and GHR (Chandler et al. 1958) lack support for multivariate psychological inputs, while Helly's model (1959) allows them via its linear structure. ANFIS is selected for its technical similarity to NeMeFIS, despite limited cognitive modeling. The number of membership functions for ANFIS were kept the same as the NeMeFIS models. LSTM is included because it also incorporates human memory through temporal dependency modeling and therefore shares conceptual similarity with the NeMeFIS architecture, making it an appropriate benchmark for comparison. The comparison shows that NeMeFIS outperforms the benchmark models in 81 out of 162 cases, while the remaining 81 cases are distributed among the other three model types.

TABLE 4 Performance (i.e. RMSE) comparison of the NeMeFIS models (a) Training
<table><tr><td colspan="10">Acceleration</td><td colspan="9">Deceleration</td></tr><tr><td>Corridor</td><td>B</td><td>PC</td><td>C M</td><td>P</td><td>R</td><td>AR</td><td></td><td>ALL</td><td>Corridor</td><td>B</td><td>PC</td><td>C</td><td>M</td><td>P R</td><td>AR</td><td>T</td><td>ALL</td></tr><tr><td>AR (HFIS)</td><td>0.11</td><td>0.10</td><td>0.11 0.12</td><td>0.14</td><td>0.28</td><td>0.10</td><td>0.11</td><td></td><td>AR (HFIS)</td><td>0.12</td><td>0.02 0.24</td><td>0.10</td><td>0.04</td><td>0.22</td><td>0.09</td><td></td><td>0.12</td></tr><tr><td>AR (LR)</td><td>0.34</td><td>0.31</td><td>0.37 0.36</td><td>0.36</td><td>0.67</td><td>0.33</td><td>0.33</td><td>AR (LR)</td><td></td><td>0.27 0.33</td><td>0.40</td><td>0.40</td><td>0.39</td><td>0.10</td><td>0.38</td><td></td><td>0.31</td></tr><tr><td>AR (LSTM)</td><td>0.16</td><td>0.14</td><td>0.16 0.12</td><td>0.19</td><td>0.26</td><td>0.14</td><td></td><td>0.16</td><td>AR (LSTM)</td><td>0.19</td><td>0.14</td><td>0.22</td><td>0.14</td><td>0.10 0.29</td><td>0.17</td><td></td><td>0.21</td></tr><tr><td>AR (ANFIS)</td><td>0.16</td><td>0.11</td><td>0.12 0.33</td><td>0.17</td><td>0.10</td><td>0.25</td><td></td><td>0.14</td><td>AR (ANFIS)</td><td>0.02</td><td>0.02</td><td>0.03 0.02</td><td>0.06</td><td>0.02</td><td>0.01</td><td></td><td>0.02</td></tr><tr><td>CR (HFIS)</td><td>0.08</td><td>0.10</td><td>0.12 0.08</td><td>0.08</td><td>0.11</td><td>0.12</td><td></td><td>0.12</td><td>CR (HFIS)</td><td>0.01</td><td>0.05</td><td>0.01 0.08</td><td>0.11</td><td>0.02</td><td>0.03</td><td></td><td>0.02</td></tr><tr><td>CR (LR)</td><td>0.43</td><td>0.35</td><td>0.40 0.35</td><td>0.17</td><td>0.16</td><td>0.40</td><td></td><td>0.12</td><td>CR (LR)</td><td>0.18</td><td>0.12</td><td>0.26 0.42</td><td></td><td>0.46 0.36</td><td>0.28</td><td></td><td>0.21</td></tr><tr><td>CR (LSTM)</td><td>0.10</td><td>0.11</td><td>0.08</td><td>0.15 0.07</td><td>0.17</td><td>0.14</td><td></td><td>0.16</td><td>CR (LSTM)</td><td>0.24</td><td>0.20</td><td>0.27 0.23</td><td></td><td>0.19 0.22</td><td>0.23</td><td></td><td>0.23</td></tr><tr><td>CR (ANFIS)</td><td>0.10</td><td>0.11</td><td>0.15</td><td>0.15 0.07</td><td>0.11</td><td>0.16</td><td></td><td>0.22</td><td>CR (ANFIS)</td><td>0.02</td><td>0.02</td><td>0.01</td><td>0.01</td><td>0.01 0.03</td><td>0.05</td><td></td><td>0.03</td></tr><tr><td>RH (HFIS)</td><td>0.11</td><td>0.22</td><td>0.15</td><td>0.15 0.10</td><td>0.23</td><td>0.12</td><td>0.13</td><td>0.09</td><td>RH (HFIS)</td><td>0.02</td><td>0.11</td><td>0.19</td><td>0.11</td><td>0.09 0.13</td><td>0.10</td><td>0.10</td><td>0.12</td></tr><tr><td>RH (LR)</td><td>0.64</td><td>0.13</td><td>0.64</td><td>0.33 0.31</td><td>0.15</td><td>0.13</td><td>0.32</td><td>0.40</td><td>RH (LR)</td><td>0.47</td><td>0.62</td><td>0.41</td><td>0.43</td><td>0.25 0.47</td><td>0.42</td><td>0.37</td><td>0.41</td></tr><tr><td>RH(LSTM)</td><td>0.10</td><td>0.10</td><td>0.43</td><td>0.13 0.13</td><td>0.17</td><td>0.12</td><td>0.14</td><td>0.20</td><td>RH (LSTM)</td><td>0.24</td><td>0.17</td><td>0.16</td><td>0.15</td><td>0.19</td><td></td><td></td><td></td></tr><tr><td>RH (ANFIS)</td><td>0.10</td><td>0.12</td><td>0.08</td><td>0.17 0.13</td><td>0.14</td><td>0.13</td><td>0.13</td><td>0.28</td><td>RH (ANFIS)</td><td>0.02</td><td>0.04</td><td>0.02</td><td>0.02 0.01</td><td>0.39 0.02</td><td>0.17 0.06</td><td>0.18 0.03</td><td>0.43 0.15</td></tr><tr><td colspan="14">(b) Testing</td></tr><tr><td></td><td></td><td></td><td>Acceleration</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>Deceleration</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Corridor</td><td>B</td><td>PC</td><td>C</td><td>M P</td><td>R</td><td>AR</td><td></td><td>ALL</td><td>Corridor</td><td>B</td><td>PC</td><td>C</td><td>M</td><td>P</td><td>R</td><td>AR 1</td><td>ALL</td></tr><tr><td>AR (HFIS)</td><td>0.10</td><td>0.11</td><td>0.08</td><td>0.10</td><td>0.15 0.30</td><td>0.09</td><td></td><td>0.10</td><td>AR (HFIS)</td><td>0.11</td><td>0.01</td><td>0.06</td><td>0.08</td><td>0.03</td><td>0.25 0.08</td><td></td><td>0.12</td></tr><tr><td>AR (LR)</td><td>0.35</td><td>0.32</td><td>0.36</td><td>0.33</td><td>0.39 0.51</td><td>0.35</td><td></td><td>0.34</td><td>AR (LR)</td><td>0.32</td><td>0.33</td><td>0.32</td><td>0.42</td><td>0.41 0.54</td><td>0.34</td><td></td><td></td></tr><tr><td>AR (LSTM)</td><td>0.29</td><td>0.28</td><td>0.35</td><td>0.29 0.39</td><td>0.40</td><td>0.23</td><td></td><td>0.29</td><td>AR (LSTM)</td><td>0.20</td><td>0.22</td><td>0.50</td><td>0.34</td><td>0.31 0.61</td><td>0.21</td><td></td><td>0.36 0.28</td></tr><tr><td>AR (ANFIS)</td><td>0.11</td><td>0.14</td><td>0.14</td><td>0.12 0.25</td><td>0.59</td><td>0.12</td><td></td><td>0.11</td><td>AR (ANFIS)</td><td>0.03</td><td>0.02</td><td>0.04</td><td>0.02</td><td>0.06 0.02</td><td>0.02</td><td></td><td></td></tr><tr><td>CR (HFIS)</td><td>0.06</td><td>0.09</td><td>0.16</td><td>0.07 0.29</td><td>0.11</td><td>0.13</td><td></td><td>0.12</td><td>CR (HFIS)</td><td>0.07</td><td>0.01</td><td>0.04</td><td>0.06</td><td>0.08 0.02</td><td>0.07</td><td></td><td>0.02 0.08</td></tr><tr><td>CR (LR)</td><td>0.43</td><td>0.38</td><td>0.41</td><td>0.35 0.41</td><td>0.40</td><td>0.39</td><td></td><td>0.37</td><td>CR (LR)</td><td>0.25</td><td>0.31</td><td>0.43</td><td>0.42</td><td>0.47 0.39</td><td>0.45</td><td></td><td>0.29</td></tr><tr><td>CR (LSTM)</td><td>0.29</td><td>0.18</td><td>0.35</td><td>0.45 0.25</td><td>0.15</td><td>0.21</td><td></td><td>0.19</td><td>CR (LSTM)</td><td>0.27</td><td>0.18</td><td>0.67</td><td>0.50</td><td>0.45 0.26</td><td>0.40</td><td></td><td>0.26</td></tr><tr><td>CR (ANFIS)</td><td>0.39</td><td>0.15</td><td>0.23</td><td>0.09 0.53</td><td>0.11</td><td>0.19</td><td></td><td>0.10</td><td>CR (ANFIS)</td><td>0.02</td><td>0.01</td><td>0.01</td><td>0.01</td><td>0.01 0.03</td><td>0.04</td><td></td><td>0.03</td></tr><tr><td>RH (HFIS)</td><td>0.12</td><td>0.17</td><td>0.13 0.36</td><td>0.11 0.13</td><td>0.30</td><td>0.23</td><td>0.13</td><td>0.10</td><td>RH (HFIS)</td><td>0.02</td><td>0.04</td><td>0.32</td><td>0.10</td><td>0.09 0.04</td><td>0.03</td><td>0.08</td><td>0.11</td></tr><tr><td>RH (LR) RH (LSTM)</td><td>0.69</td><td>0.17 0.66 0.34 0.48</td></tr><tr><td>Corridor</td><td>B PC</td><td>C</td><td>M</td><td>P</td><td>R</td><td>AR</td><td></td><td>ALI</td><td>Corridor</td><td>B</td><td>PC</td><td>C</td><td>M</td><td>R 0</td><td>AR</td><td></td><td>ALL</td></tr><tr><td>AR (LR)</td><td>0.36 0.32</td><td>0.37</td><td>0.36</td><td>0.39</td><td>0.57</td><td>0.31</td><td>0.33</td><td>AR (LR)</td><td>0.22</td><td>0.37</td><td>0.30</td><td>0.42</td><td>0.42</td><td>0.54</td><td>0.38</td><td></td><td>0.39</td></tr><tr><td>AR (LSTM)</td><td>0.32 0.28</td><td>0.35</td><td>0.36</td><td>0.62</td><td>0.29</td><td>0.22</td><td>0.28</td><td>AR (LSTM)</td><td>0.31</td><td>0.28</td><td>0.37</td><td>0.39</td><td>0.51</td><td>0.62</td><td>0.20</td><td></td><td>0.24</td></tr><tr><td>AR (ANFIS)</td><td>0.43 0.11</td><td>).29</td><td>0.22</td><td>0.09</td><td>0.46</td><td>0.31</td><td>0.12</td><td>AR (ANFIS)</td><td></td><td>0.02 0.02</td><td>0.07</td><td>0.02</td><td>0.04</td><td>0.02</td><td>0.01</td><td></td><td>0.02</td></tr><tr><td>CR (HFIS)</td><td>0.09 0.09</td><td>5.13</td><td>0.05</td><td>0.13</td><td>0.12</td><td>0.12</td><td>0.11</td><td>CR (HFIS)</td><td></td><td>0.09</td><td>0.02 0.08</td><td>0.01</td><td>0.13</td><td>0.02</td><td>0.03</td><td></td><td>0.08</td></tr><tr><td>CR (LR)</td><td>0.42</td><td>0.40 ).38</td><td>0.37</td><td>0.22</td><td>0.41</td><td>0.39</td><td>0.25</td><td>CR (LR)</td><td></td><td>0.31 0.45</td><td>0.40</td><td>0.42</td><td>0.47</td><td>0.44</td><td>0.40</td><td></td><td>0.36</td></tr><tr><td>CR (LSTM)</td><td>0.20</td><td>0.22 ).35</td><td>0.41</td><td>0.26</td><td>0.26</td><td>0.59</td><td>0.16</td><td>CR (LSTM)</td><td></td><td>0.37</td><td>0.30 0.51</td><td>0.36</td><td>0.42</td><td>0.30</td><td>0.31</td><td></td><td>0.20</td></tr><tr><td>CR (ANFIS)</td><td>0.09 0.15</td><td>).12</td><td>0.14</td><td>0.10</td><td>0.78</td><td>0.09</td><td>0.19</td><td>CR (ANFIS)</td><td></td><td>0.01</td><td>0.02 0.01</td><td>0.01</td><td>0.05</td><td>0.02</td><td>0.04</td><td></td><td>0.03</td></tr><tr><td>RH (HFIS)</td><td>0.26 0.20</td><td>0.41</td><td>0.10</td><td>0.10</td><td>0.44</td><td>0.44</td><td>0.19 0.10</td><td>RH (HFIS)</td><td></td><td>0.08</td><td>0.10</td><td>0.63 0.18</td><td>0.10</td><td>0.18</td><td>0.10</td><td>0.14</td><td>0.12</td></tr><tr><td>RH (LR)</td><td>0.68 0.18</td><td>).49</td><td>0.31</td><td>0.26</td><td>0.17</td><td>0.62</td><td>0.36 0.38</td><td></td><td>RH (LR)</td><td>0.56</td><td>0.66</td><td>0.22</td><td>0.49 0.35</td><td>0.46</td><td>0.42</td><td>0.38</td><td>0.32</td></tr><tr><td>RH (LSTM)</td><td>0.35 0.39</td><td>0.30</td><td>0.39</td><td>0.35</td><td>0.32</td><td>0.32</td><td>0.24 0.43</td><td></td><td>RH (LSTM)</td><td>0.42</td><td>0.36</td><td>0.56</td><td>0.41 0.32</td><td>0.45</td><td>0.33</td><td>0.39</td><td>0.42</td></tr><tr><td>RH (ANFIS)</td><td>0.21 0.59</td><td>).25</td><td>0.88</td><td>0.16</td><td>0.12</td><td>0.13</td><td>0.12</td><td>0.42</td><td>RH(ANFIS)</td><td>0.08</td><td>0.03</td><td>0.13</td><td>0.07</td><td>0.49</td><td>0.12 0.06</td><td>0.03</td><td>0.22</td></tr></table>

1 Vehicle Types: B = Bus, PC =Passenger Car, C = Covered van, M = Motorcycle, P = Pick-up, R = Rickshaw, AR = Auto-Rickshaw, T= Truck; HFIS = Modeled 2 with NeMeFIS; \*hyphen (-) indicates absence of sample in the particular vehicle type

NeMeFIS (HFIS) performs best mainly in deceleration scenarios across all corridors (AR, CR, RH) and vehicle types, showing consistently lower RMSE in training, testing, and validation. Its advantage is strongest for passenger cars, buses, motorcycles, and mixed traffic, particularly on collector roads and rural highways, where braking involves higher cognitive complexity. HFIS also shows strong performance in acceleration on CR and RH, though with smaller margins. This superiority stems from its asymmetric, memory-informed fuzzy structure, which captures nonlinear and context-dependent human driving decisions more effectively than LR, ANFIS, and LSTM.

Table 5 summarizes the number of fuzzy inference system (FIS) rules in the Neuro-Memory Fuzzy Inference System (NeMeFIS) across different driving scenarios. The results reveal that acceleration decisions involve fewer rules under procedural memory, emphasizing skill-based responses, while declarative memory consistently yields the highest rule counts, reflecting its role in complex cognitive reasoning. Deceleration involves significantly higher rule complexity, especially under declarative and combined memory types.

TABLE 5 Number of Rules in the NeMeFIS model in different scenarios and different vehicle types
<table><tr><td rowspan="2">CT FIS</td><td rowspan="2"></td><td colspan="9">Acceleration (A)</td><td colspan="8">Deceleration (D)</td></tr><tr><td>B</td><td>PC</td><td>C</td><td>M</td><td>P</td><td>R AR</td><td>T</td><td>A</td><td>B</td><td>PC</td><td>C</td><td>M</td><td>P</td><td>R</td><td>AR</td><td>T</td><td>A</td></tr><tr><td rowspan="5">AR P</td><td></td><td>7</td><td>7</td><td>8</td><td>8</td><td>8</td><td>10</td><td></td><td>17</td><td>11</td><td>8</td><td>9</td><td>11</td><td>8</td><td>10</td><td>9</td><td>0</td><td>26</td></tr><tr><td>W</td><td>8</td><td>11</td><td>10</td><td>7</td><td>8</td><td>9</td><td></td><td>21</td><td>9</td><td>9</td><td>6</td><td>9</td><td>10</td><td>10</td><td>10</td><td>0</td><td>24</td></tr><tr><td>E</td><td>9</td><td>9</td><td>7</td><td>9</td><td>9</td><td>9</td><td>9</td><td></td><td>21</td><td>9</td><td>7</td><td>10</td><td>11</td><td>10</td><td>10</td><td>0</td><td>25</td></tr><tr><td>S</td><td>27</td><td>32</td><td>42</td><td>43</td><td>36</td><td>42</td><td>34</td><td></td><td>115 37</td><td>37</td><td>38</td><td>40</td><td>39</td><td>40</td><td>41</td><td>0</td><td>133</td></tr><tr><td>D</td><td>160</td><td>151</td><td>167</td><td>167</td><td>169</td><td>195</td><td>170</td><td>0</td><td>332</td><td>190 170</td><td>187</td><td>199</td><td>190</td><td>177</td><td>192</td><td>0</td><td>448</td></tr><tr><td rowspan="6">CR</td><td>C</td><td>71</td><td>65</td><td>69</td><td>74</td><td>72</td><td>94 86</td><td></td><td>128</td><td>83</td><td>84</td><td>90</td><td>93</td><td>97</td><td>76</td><td>95</td><td>0</td><td>166</td></tr><tr><td>P</td><td>9</td><td>9</td><td>11</td><td>11</td><td>10</td><td>10</td><td>0</td><td>14</td><td>9</td><td>8</td><td>8</td><td>10</td><td>10</td><td>11</td><td>10</td><td>0</td><td>23</td></tr><tr><td>W</td><td>9</td><td>9</td><td>8</td><td>8</td><td>6</td><td>10</td><td>0</td><td>17</td><td>12</td><td>7</td><td>9</td><td>10</td><td>10</td><td>11</td><td>9</td><td>0</td><td>25</td></tr><tr><td>E</td><td>8</td><td>10</td><td>8</td><td>11</td><td>8</td><td>10</td><td></td><td>18</td><td></td><td>8</td><td>10</td><td>12</td><td>6</td><td>9</td><td>10</td><td>0</td><td>21</td></tr><tr><td>S</td><td>33</td><td>44</td><td>35</td><td>41</td><td>33</td><td>36</td><td>37</td><td>110</td><td>38</td><td>30</td><td>37</td><td>37</td><td>41</td><td>39</td><td>42</td><td>0</td><td>129</td></tr><tr><td>D</td><td>155</td><td>196</td><td>155</td><td>185</td><td>171</td><td>171</td><td>163</td><td>0</td><td>300 223</td><td>135</td><td>200</td><td>210</td><td>218</td><td>171</td><td>211</td><td>0</td><td>427</td></tr><tr><td rowspan="6">RH</td><td>C</td><td>68</td><td>84</td><td>67</td><td>86</td><td>83</td><td>79 84</td><td>0</td><td>116</td><td>99</td><td>59</td><td>90</td><td>100</td><td>98</td><td>78</td><td>103</td><td>0</td><td>149</td></tr><tr><td>P</td><td>9</td><td>7</td><td>9</td><td>7</td><td>10</td><td>10</td><td>10</td><td>19</td><td>10</td><td>11</td><td>7</td><td>8</td><td>8</td><td>9</td><td>9</td><td>9</td><td>18</td></tr><tr><td>W</td><td>10</td><td>11</td><td>11</td><td>9</td><td>10</td><td>10</td><td>7</td><td>18</td><td>10</td><td>9</td><td>8</td><td>8</td><td>7</td><td>9</td><td>11</td><td>11</td><td>18</td></tr><tr><td>E</td><td>10</td><td>8</td><td>9</td><td>8</td><td>11</td><td>8</td><td></td><td>18</td><td>9</td><td>7</td><td>10</td><td>10</td><td>9</td><td>8</td><td>10</td><td>8</td><td>22</td></tr><tr><td>S</td><td>41</td><td>38</td><td>35</td><td>41</td><td>42</td><td>34 42</td><td>36</td><td>116</td><td>36</td><td>36</td><td>37</td><td>32</td><td>40</td><td>39</td><td>36</td><td>42</td><td>104</td></tr><tr><td>D</td><td>184</td><td>185 79</td><td>156</td><td>183</td><td>209</td><td>200</td><td>192 83</td><td>185</td><td>315 187</td><td>199</td><td>176</td><td>162</td><td>149</td><td>216</td><td>212</td><td>194</td><td>332</td></tr><tr><td></td><td>C</td><td>86</td><td></td><td>84</td><td>86</td><td>107</td><td>89</td><td>90</td><td>111</td><td>92</td><td>97</td><td>89</td><td>74</td><td>66</td><td>109</td><td>98</td><td>83</td><td>129</td></tr></table>

\* CT = Corridor Type, FIS = Fuzzy inference System, FIS types: P = Procedural Memory, W = Working Memory, E = Episodic memory, S = Semantic memory, D = Declarative memory, C = Combined Memory;

Figure 2 illustrates the sensitivity of the Neuro Memory Fuzzy Inference System (NeMeFIS) model's acceleration magnitude (normalized) to various normalized input variables across different traffic scenarios. S-L Spacing strongly influences acceleration magnitude at lower normalized values (0–0.3), emphasizing the need for safe following distances in proximity-dominated conditions. Risk Perception consistently remains a dominant variable across all traffic scenarios, reinforcing its critical role in cautious and human-like decision-making. In AR and RH settings, Speed-S and Relative Speed S-NL exhibit noticeable sensitivity, emphasizing their importance in managing speed mismatches and overtaking behavior.

Figure 3 illustrates the variable importance across vehicle types and road corridors during acceleration and deceleration. This figure reveals that risk perception dominates during acceleration, especially on Arterial (AR) and Collector Roads (CR), with covered vans and motorcycles showing the highest sensitivity due to their contrasting vulnerabilities. In Rural Highways (RH), however, angular spacing becomes more critical for buses aiming to overtake. During deceleration, the emphasis shifts from risk to linear and angular spacing, reflecting adaptive strategies across corridors. In AR, lead spacing ranks highest, followed by angular cues for potential lane changes. In CR, angular distance to the lead vehicle dominates due to weak lane discipline. Finally, in RH corridors, relative spacing with the lead vehicle is most influential, reflecting disciplined traffic flow and fewer overtaking options.

(d) CR-Deceleration S-NL Angle S-L Spacing Change S-NL Spacing Change  
![](images/a5e16691fabadbd7251399ac9a941606e11b24517f59dff033f28a5cf852fdd5.jpg)

![](images/e2888bdbbacefbc8ed40eabb7de663eb45026d10fb2d8805aaf9cdcece37c202.jpg)  
(a) AR-Acceleration

![](images/7c0e741c5a303c554649d2c4d9ed8e04de9959556c8048cfb218b60927d01d5a.jpg)  
(e) RH-Acceleration

(c) CR-Acceleration  
![](images/4ed07794805e4433a300b18a23c07f8988ceab16fbb622e99268a28ae901d3cc.jpg)

![](images/9b763c61bea94d17839b5a006d392dadf2a73c094fe81b59f3c13bcbe03bda1e.jpg)  
(b) AR-Deceleration S-L Spacing S-NL Spacing S-L Angle

![](images/0b916ab0b71b3588edaddb5b09d2be9cbdaeea016ddbdfebd427cb42649eb930.jpg)  
(f) RH-Deceleration S Speed S-NL relative speed Risk Perception

Figure 2 Sensitivity of Different Variables in NeMeFIS Models  
![](images/e157e7f1d954d05d548c76b2bc28238aba6b07a40f02b7e3e78351639ef8f3d7.jpg)  
(a) AR-Acceleration

![](images/1b8eab35acba73dfd8d030636be896ff1adc63c1044dff6358e81a70bd180f95.jpg)

(b) AR-Deceleration  
![](images/04613f553535577029c57d4629d33dc7a7f3ded8eb306a2322be5e82006bfffb.jpg)

(c) CR-Acceleration  
![](images/1a5ed670664c877c9fb0e4a137f909e548490d7af7c30a12c2b83d2041c0460e.jpg)  
(d) CR-Deceleration

![](images/d9210e195814ed047ce5923cb375fe427deed423fabc2d685cbc74900f221075.jpg)

(e) RH-Acceleration  
![](images/56e2d1dbd7351d95c0a35aa129645535c01efb1ceca58ad10cca98628b83527a.jpg)  
(f) RH-Deceleration  
Figure 3 Variable Importance for Different Scenarios and Vehicles  
Figure 4 illustrates the probabilistic influence of various memory types. The centroid values and polygon areas represent the intensity and spread of each memory’s contribution. This figure shows that Procedural (P) memory dominates acceleration on Arterial and Collector Roads (AR, CR), with high centroid values (e.g., P3 on AR) indicating skill-based control at higher speeds. Working (W) memory is also crucial during AR and CR acceleration, supporting real-time adaptations (e.g., clusters W2, W3). During AR deceleration, Semantic (S) memory leads (e.g., S1), aiding in signal anticipation through generalized knowledge. On Collector Roads, Declarative (D) memory dominates deceleration (e.g., D3), reflecting the need to synthesize spatial and experiential inputs in chaotic conditions. In Rural Highways (RH), Semantic memory governs acceleration (e.g., S2, S3), while Declarative memory guides deceleration (e.g., D1, D3) to handle unpredictable hazards. In summary, Procedural memory drives acceleration, while Declarative and Semantic memories dominate deceleration, shaped by road type and traffic complexity.

Furthermore, the stability of the model was evaluated using numerical simulation (results shown in Figure 5). The simulation environment was configured as follows: total simulation time of 100 seconds, a time step of 0.1 seconds, a distance step of 0.1 meters, one lane with a width of 3.65 meters, a road length of 1000 meters, a flow rate resolution of 10 seconds, a flow rate of 1500 vehicles per hour, vehicle composition based on the corridor type, and a perception-reaction time of 1.5 seconds. Simulations were conducted across all three corridor types to assess the stability of the model. Simulations across three corridor types confirmed the model's stability under controlled acceleration, deceleration, and shockwave-inducing conditions. Vehicle trajectories showed no FIFO violations or abrupt speed changes, with minor oscillations damped over time, indicating stable behavior. Emergency braking spikes observed in Figure 5 stemmed from boundary conditions during vehicle entry, not model deficiencies. Thus, the results affirm the proposed model’s robustness and resistance to instability in diverse traffic scenarios.

![](images/705b248bf0cfe26a6ad0061d8727a5d9fb9008f3060059abfcef64cf29179e3d.jpg)

![](images/376ee677cc05eda410a56b8187ee112110798d2c910cf3c1aafc402b455a1288.jpg)  
(a) AR- Acceleration

![](images/899ccd3da98f0d15e869328516fa592010b2156b38d5212363f491e2af183f7d.jpg)

<table><tr><td>-P1(0.81,0.16)</td><td>E2(7.66)</td></tr><tr><td>P1(2.67)</td><td>E3(0.17,0.21)</td></tr><tr><td>P2(0.16,0.13)</td><td>E3(3.9)</td></tr><tr><td>P2(2.83)</td><td>S1(0.72,0.56)</td></tr><tr><td>P3(0.47,0.21)</td><td>S1(6.74)</td></tr><tr><td>P3(4.39)</td><td>S2(0.24,0.29)</td></tr><tr><td>W1(0.16,0.28)</td><td>S2(8.64)</td></tr><tr><td>W1(2.42)</td><td>S3(0.69,0.18)</td></tr><tr><td>W2(0.77,0.2)</td><td>S3(5.66)</td></tr><tr><td>W2(4.87)</td><td>D1(0.49,0.18)</td></tr><tr><td>W3(0.38,0.18)</td><td>D1(3.91)</td></tr><tr><td>W3(4.89)</td><td>D2(0.8,0.26)</td></tr><tr><td>E1(0.48,0.2)</td><td>D2(5.64)</td></tr><tr><td>E1(4.08)</td><td>D3(0.18,0.13)</td></tr><tr><td>E2(0.77,0.34)</td><td>D3(4.75)</td></tr></table>

(b) AR- Deceleration

![](images/1e7b111d62bb54b9c8b26defec849b71cc133b966b9eec34ad4140cc590d6f8a.jpg)

(c) CR- Acceleration  
![](images/fc5315beb83722a2fa370df66704d20a9d9bec24aa77aaa87bae42c356209d5d.jpg)

![](images/fe7e5d3b999f4c3647f15d09a979d74898c801da9332fed692802be9fa60abb6.jpg)

<table><tr><td>P1(0.66,0.11)</td><td>E2(8.28)</td></tr><tr><td>P1(2.5)</td><td>E3(0.18,0.24)</td></tr><tr><td>P2(0.78,0.29)</td><td>E3(4.46)</td></tr><tr><td>P2(3.9)</td><td>S1(0.25,0.25)</td></tr><tr><td>P3(0.26,0.12)</td><td>S1(7.22)</td></tr><tr><td>P3(7.74)</td><td>-S2(0.68,0.15)</td></tr><tr><td>W1(0.45,0.23)</td><td>S2(4.8)</td></tr><tr><td>W1(5.26)</td><td>S3(0.78,0.45)</td></tr><tr><td>W2(0.8,0.17)</td><td>S3(4.11)</td></tr><tr><td></td><td></td></tr><tr><td>W2(3.22)</td><td>D1(0.71,0.2)</td></tr><tr><td>W3(0.16,0.11)</td><td>D1(5.39)</td></tr><tr><td>W3(3.35)</td><td>D2(0.23,0.24)</td></tr><tr><td>E1(0.49,0.13)</td><td>D2(10.48)</td></tr><tr><td></td><td></td></tr><tr><td>E1(6)</td><td>D3(0.72,0.56)</td></tr><tr><td>E2(0.77,0.35)</td><td>D3(6.22)</td></tr></table>

(d) CR- Deceleration

![](images/313e36a38c74a055a852d8a3219fee4ddd6022640c5c6e16e99e1fcf06ac77c1.jpg)

![](images/c6f2908df6332c58a7dbd4ee39f51dc4e388c5d1782508ce161dfde98901d36d.jpg)  
(e) RH- Acceleration

![](images/1b5f46b3b80bcf22a28f86631f17d07b7b0d5111db65f41c803e56cc689c6cf4.jpg)

<table><tr><td>P1(0.36,0.13)</td><td></td><td>E2(4.12)</td></tr><tr><td>P1(4.86)</td><td></td><td>E3(0.48,0.17)</td></tr><tr><td>P2(0.74,0.27)</td><td></td><td>E3(3.59)</td></tr><tr><td>P2(7.76)</td><td></td><td>S1(0.25,0.25)</td></tr><tr><td>P3(0.16,0.24)</td><td></td><td>S1(7.2)</td></tr><tr><td>P3(3.16)</td><td></td><td>S2(0.76,0.46)</td></tr><tr><td>W1(0.44,0.13)</td><td></td><td>S2(4.56)</td></tr><tr><td>W1(5.19)</td><td></td><td>S3(0.68,0.15)</td></tr><tr><td>W2(0.73,0.36)</td><td></td><td>S3(4.94)</td></tr><tr><td>W2(10.42)</td><td></td><td>D1(0.66,0.27)</td></tr><tr><td>W3(0.17,0.25)</td><td></td><td>D1(9.86)</td></tr><tr><td>W3(4.19)</td><td></td><td>D2(0.24,0.2)</td></tr><tr><td></td><td></td><td></td></tr><tr><td>E1(0.82,0.22)</td><td></td><td>D2(9.38)</td></tr><tr><td>E1(4.35)</td><td></td><td>D3(0.78,0.48)</td></tr><tr><td>E2(0.17,0.11)</td><td></td><td>D3(3.88)</td></tr></table>

(f) RH- Deceleration

Figure 4 Contribution of Memory Types in Decision Making at Different Relative Speed Level to Choose Magnitude.

The transferability of the proposed Neuro Memory Fuzzy Inference System (NeMeFIS) architecture was trained and evaluated using sampled trajectory datasets from (a) freeway US-101, sourced from the Next Generation Simulation (US Department of Transportation 2007) program (Figure 6(a)) and, (b) Argoverse 2 (Wilson et al. 2023) which facilitates CAV trajectories (Figure 6(b)). The NeMeFIS architecture was found to be transferable with considerable accuracy and sensitivity in both the datasets

![](images/4e13047dc8d4ca57ac1e26bb81d743e05d8767ff39e1d23e0ca624af45b991f4.jpg)  
(a)

![](images/15bdf3464b8c8079db777299d3f41760dc5b1904b191df00e7436c20e48efcb3.jpg)  
(c)

![](images/778e515fec86670da2e47a0b8ff25d857b628e0b2f949a9f36ad72b75dfe8abe.jpg)  
(e)

![](images/6d5233645f382b4a9574f0d24ac02ee156074baed6daa373f7dcd431d0bd6920.jpg)  
(b)

![](images/735abdc7496e3de0f506f97f2d4fd53f43cfecdfb2b80bcb3a466c431eec262b.jpg)  
(d)

![](images/f1a4555484587df363e1171a82aeaa6b334c42194ec5ec2db3a883758e80b167.jpg)  
(f)  
Private Passenger Car Motor Cycle Bus Truck

Figure 5 Numerical Simulation of the NeMeFIS Models for different vehicle types (a)-(b) AR; (c)-(d) CR; (e)- (f) RH  
![](images/ec5411e16719cf627c1a65cd4663afa90ef956a2ab98154817b09483c0f4aa70.jpg)  
(a)

![](images/826f1c105405af73c6e1d4aeb9fdf56f874de35595b7d6335206aeab8bf404c3.jpg)  
(b)  
Figure 6 Transferability and Convergence of NeMeFIS Architecture in Different Datasets (a) NGSIM (b) Argoverse 2 (CAV) CONCLUSIONS

This study presents the Neuro Memory Fuzzy Inference System (NeMeFIS), a novel model that simulates human-like vehicle-following behavior by incorporating five types of memory: procedural, working, episodic, semantic, and declarative. Acceleration is primarily influenced by procedural and working memory, enabling skillbased and adaptive responses, while deceleration relies on declarative and semantic memory, reflecting complex cognitive demands. The model reveals higher cognitive load for rural highways and larger vehicles, as indicated by increased fuzzy rule counts. Applied on UAV-based (heterogeneous) traffic data and evaluated on US101 (homogenous) and Argoverse 2 (CAV), NeMeFIS demonstrates robustness and transferability. Notably, NeMeFIS consistently outperforms linear regression, ANFIS, and LSTM models, particularly in asymmetric deceleration regimes, highlighting its practical advantage over existing approaches. Future improvements should incorporate extreme case data, tailored loss functions, or models with increased membership functions as well as rules. Expanding the framework to include psychological and physiological factors like stress and fatigue will enhance its realism. NeMeFIS holds potential for driver behavior modification programs and integration into CAVs for context-aware, human-like, and safety-oriented decision-making.

## ACKNOWLEDGMENTS

The author(s) acknowledge the use of Grammarly (Grammarly Inc.), an AI-based writing assistance tool, during the preparation of this manuscript. Grammarly was used solely for grammar, spelling, punctuation, and clarity checks to improve the readability of the written text. The tool was not used to

generate original content, ideas, data, analysis, or conclusions presented in this paper. All intellectual content, research design, data analysis, and interpretations remain the sole responsibility of the author(s).

## AUTHOR CONTRIBUTIONS

The authors confirm contribution to the paper as follows: study conception and design: N. Haque, M. Hadiuzzaman; data collection: N. Haque; deep learning architecture development: N. Haque, analysis and interpretation of results: N. Haque; draft manuscript preparation: N. Haque, M. A. Raihan. All authors reviewed the results and approved the final version of the manuscript.

## DECLARATION OF CONFLICTING INTERESTS

The authors declared no potential conflicts of interest with respect to the research, authorship, and/or publication of this article.

## FUNDING

This research work is supported by the Committee for Advanced Studies and Research (CASR) (Grant No. 371(16)) of Bangladesh University of Engineering and Technology (BUET).

## REFERENCES

Aghabayk, Kayvan, Nafiseh Forouzideh, and William Young. 2013. "Exploring a Local Linear Model Tree Approach to Car-Following." Computer-Aided Civil and Infrastructure Engineering 28 (8): 581–593. https://doi.org/10.1111/mice.12011.

Bando, Masako, Katsuya Hasebe, Akihiro Nakayama, Akihiro Shibata, and Yuki Sugiyama. 1995. "Dynamical Model of Traffic Congestion and Numerical Simulation." Physical Review E 51 (2): 1035–1042. https://doi.org/10.1103/PhysRevE.51.1035.

Boer, Erwin R. 1999. "Car Following from the Driver's Perspective." Transportation Research Part F: Traffic Psychology and Behaviour 2 (4): 201–206. https://doi.org/10.1016/S1369-8478(00)00007-3.

Chakroborty, Partha, and Shinya Kikuchi. 2003. "Calibrating the Membership Functions of the Fuzzy Inference System: Instantiated by Car-Following Data." Transportation Research Part C: Emerging Technologies 11 (2): 91–119. https://doi.org/10.1016/S0968-090X(02)00022-0.

Chandler, Robert E., Robert Herman, and Elliott W. Montroll. 1958. "Traffic Dynamics: Studies in Car Following." Operations Research 6 (2): 165–184.

Anil Chaudhari, Ankit, Karthik K. Srinivasan, Bhargava Rama Chilukuri, Martin Treiber, and Ostap Okhrin. 2022. "Calibrating Wiedemann-99 Model Parameters to Trajectory Data of Mixed Vehicular Traffic." Transportation Research Record 2676 (1): 718–735.

Chen, Xinqiang, Zhibin Li, Yongsheng Yang, Lei Qi, and Ruimin Ke. "High-Resolution Vehicle Trajectory Extraction and Denoising from Aerial Videos." IEEE Transactions on Intelligent Transportation Systems 22 (5): 3190– 3202.

Colyar, James, and John Halkias. 2007. US Highway 101 Dataset. Report FHWA-HRT-07-030. Washington, DC: Federal Highway Administration, US Department of Transportation.

Dougherty, Michael RP, Charles F. Gettys, and Eve E. Ogden. 1999. "MINERVA-DM: A Memory Processes Mode for Judgments of Likelihood." Psychological Review 106 (1): 180–209.

Fadhloun, Karim, and Hesham Rakha. 2020. "A Novel Vehicle Dynamics and Human Behavior Car-Following Model: Model Development and Preliminary Testing." International Journal of Transportation Science and Technology 9 (1): 14–28. https://doi.org/10.1016/j.ijtst.2019.05.004.

Fritzsche, Hans-Thomas, and Daimler-benz Ag. 1994. "A Model for Traffic Simulation." Traffic Engineering + Control 35 (5): 317–321.

Furutani, Naomichi. 1976. "A New Approach to Traffic Behaviour: I. Modelling of 'Following-Defence' Behaviour." International Journal of Man-Machine Studies 8 (5): 597–615. https://doi.org/10.1016/S0020-7373(76)80022- 3.

Gipps, Peter G. 1981. "A Behavioural Car-Following Model for Computer Simulation." Transportation Research Part B: Methodological 15 (2): 105–111. https://doi.org/10.1016/0191-2615(81)90037-0.

Gogoll, Jan, and Julian F. Müller. 2017. "Autonomous Cars: In Favor of a Mandatory Ethics Setting." Science and Engineering Ethics 23 (3): 681–700. https://doi.org/10.1007/s11948-016-9806-x.

Hamdar, Samer H., Martin Treiber, Hani S. Mahmassani, and Arne Kesting.2008. "Modeling Driver Behavior as Sequential Risk-Taking Task." Transportation Research Record 2088 (1): 208–217. https://doi.org/10.3141/2088-22.

Hamdar, Samer H., Hani S. Mahmassani, and Martin Treiber. 2015. "From Behavioral Psychology to Acceleration Modeling: Calibration, Validation, and Exploration of Drivers' Cognitive and Safety Parameters in a Risk-Taking Environment." Transportation Research Part B: Methodological 78: 32–53. https://doi.org/10.1016/j.trb.2015.03.011.

Haque, Nazmul, Md Asif Raihan, Md Mizanur Rahman, and Md Hadiuzzaman. 2024. "A Review on Following Behavioral Models: Regular to Connected Autonomous Vehicle Heterogeneity." IATSS Research 48 (2): 163– 179. https://doi.org/10.1016/j.iatssr.2024.03.002.

Haque, Nazmul, Md Asif Raihan, Farhana Mozumder Lima, and Md Hadiuzzaman. 2025. "Does Deep Learning Architectures Model Human-Like Intelligent Response in Asymmetric Car-Following Behaviour? A Novel Framework for Learning Acceleration–Deceleration Decisions." IET Intelligent Transport Systems 19 (1): e70117. https://doi.org/10.1049/itr2.70117.

Helly, Walter. 1959. "Simulation of Bottlenecks in Single-Lane Traffic Flow." Transport Research International Documentation.

Huang, Xiuling, Jie Sun, and Jian Sun. 2018. "A Car-Following Model Considering Asymmetric Driving Behavior Based on Long Short-Term Memory Neural Networks." Transportation Research Part C: Emerging Technologies 95: 346–362. https://doi.org/10.1016/j.trc.2018.07.022

Islam, Muttahirul, Nazmul Haque, and Md Hadiuzzaman. 2024. "DEEGITS: Deep Learning Based Framework for Measuring Heterogenous Traffic State in Challenging Traffic Scenarios." Paper presented at the 103rd Annual Meeting of the Transportation Research Board, Washington, DC.

Hongfei, Jia, Juan Zhicai, and Ni Anning. 2003. "Develop a Car-Following Model Using Data Collected by 'Five Wheel System.'" In Proceedings of the 2003 IEEE International Conference on Intelligent Transportation Systems, 346–351. IEEE. https://doi.org/10.1109/ITSC.2003.1251975.

Jia, Hongfei, Ming Tang, and Li Li. 2008. "A Car-Following Model Based on Drivers' Cognition Behavior." Journal of Highway and Transportation Research and Development (English Edition) 3 (2): 113–116. https://doi.org/10.1061/JHTRCQ.0000256.

Jin, Sheng, Dian-Hai Wang, Zhi-Yi Huang, and Peng-Fei Tao. 2011. "Visual Angle Model for Car-Following Theory." Physica A: Statistical Mechanics and Its Applications 390 (11): 1931–1940. https://doi.org/10.1016/j.physa.2011.01.012.

Kennedy, James, and Russell Eberhart. 1995. "Particle Swarm Optimization." In Proceedings of ICNN'95 - International Conference on Neural Networks, 1942–1948. IEEE. https://doi.org/10.1109/ICNN.1995.488968.

Khodayari, Alireza, Reza Kazemi, Ali Ghaffari, and Negin Manavizadeh. 2010. "Modeling and Intelligent Control Design of Car Following Behavior in Real Traffic Flow." In 2010 IEEE Conference on Cybernetics and Intelligent Systems, 261–266. IEEE. https://doi.org/10.1109/ICCIS.2010.5518546.

Khodayari, Alireza, Reza Kazemi, Ali Ghaffari, and Reinhard Braunstingl. 2011. "Design of an Improved Fuzzy Logic Based Model for Prediction of Car Following Behavior." In 2011 IEEE International Conference on Mechatronics, 200–205. IEEE. https://doi.org/10.1109/ICMECH.2011.5971281.

Kikuchi, Shinya, and Partha Chakroborty. 1992. "Car-Following Model Based on Fuzzy Inference System." Transportation Research Record, 82-82.

Lee, Gentry.1966. "A Generalization of Linear Car-Following Theory." Operations Research 14 (4): 595–606. https://doi.org/10.1287/opre.14.4.595. [35]

Lee, John D., Michael A. Regan, and K. L. Young. Driver Distraction. CRC Press. https://doi.org/10.1201/9781420007497.

Li, Linbo, Yang Li, and Daiheng Ni. "Incorporating Human Factors into LCM Using Fuzzy TCI Model." Transportmetrica B: Transport Dynamics 9 (1): 198–218. https://doi.org/10.1080/21680566.2020.1837033.

Ma, Xiaoliang, and Ingmar Andreasson. 2007. "Behavior Measurement, Analysis, and Regime Classification in Car Following." IEEE Transactions on Intelligent Transportation Systems 8 (1): 144–156. https://doi.org/10.1109/TITS.2006.883111.

Matthews, Brandy R. 2015. "Memory Dysfunction." CONTINUUM: Lifelong Learning in Neurology 21: 613–626. https://doi.org/10.1212/01.CON.0000466656.59413.29.

Maurer, Markus, J. Christian Gerdes, Barbara Lenz, and Hermann Winner. 2016. Autonomous Driving: Technical, Legal and Social Aspects. Springer Nature.

Panwai, Sakda, and Hussein Dia. 2007. "Neural Agent Car-Following Models." IEEE Transactions on Intelligent Transportation Systems 8 (1): 60–70. https://doi.org/10.1109/TITS.2006.884616.

Ranney, Thomas A. 1999. "Psychological Factors That Influence Car-Following and Car-Following Model Development." Transportation Research Part F: Traffic Psychology and Behaviour 2 (4): 213–219. https://doi.org/10.1016/S1369-8478(00)00010-3.

Schooler, Lael J., and Ralph Hertwig. 2005. "How Forgetting Aids Heuristic Inference." Psychological Review 112 (3): 610–628. https://doi.org/10.1037/0033-295X.112.3.610.

Shadlen, Michael N., and Daphna Shohamy. 2016. "Decision Making and Sequential Sampling from Memory." Neuron 90 (5): 927–939. https://doi.org/10.1016/j.neuron.2016.04.036.

Shalev-Shwartz, Shai, Shaked Shammah, and Amnon Shashua. 2017. "On a Formal Model of Safe and Scalable Self Driving Cars." arXiv preprint arXiv:1708.06374.

Sun, Liang, and Wei Huo. 2016. "Adaptive Fuzzy Control of Spacecraft Proximity Operations Using Hierarchical Fuzzy Systems." IEEE/ASME Transactions on Mechatronics 21 (3): 1629–1640. https://doi.org/10.1109/TMECH.2015.2494607.

Talebpour, Alireza, Hani S. Mahmassani, and Samer H. Hamdar. 2011. "Multiregime Sequential Risk-Taking Model of Car-Following Behavior." Transportation Research Record 2260 (1): 60–66. https://doi.org/10.3141/2260- 07.

Tang, Tie-Qiao, Jia He, Shi-Chun Yang, and Hua-Yan Shang. 2014. "A Car-Following Model Accounting for the Driver's Attribution." Physica A: Statistical Mechanics and Its Applications 413: 583–591. https://doi.org/10.1016/j.physa.2014.07.035.

Tang, Tie-Qiao, Yong Gui, Jian Zhang, and Tao Wang. 2020. "Car-Following Model Based on Deep Learning and Markov Theory." Journal of Transportation Engineering, Part A: Systems 146 (9). https://doi.org/10.1061/JTEPBS.0000430.

Tang, Xiaolin, Jiaxin Chen, Kai Yang, Mitsuru Toyoda, Teng Liu, and Xiaosong Hu. 2022. "Visual Detection and Deep Reinforcement Learning-Based Car Following and Energy Management for Hybrid Electric Vehicles." IEEE Transactions on Transportation Electrification 8 (2): 2501–2515. https://doi.org/10.1109/TTE.2022.3141780

Torczon, Virginia. 1997. "On the Convergence of Pattern Search Algorithms." SIAM Journal on Optimization 7 (1): 1–25. https://doi.org/10.1137/S1052623493250780.

Treiber, Martin, Ansgar Hennecke, and Dirk Helbing. 2000. "Congested Traffic States in Empirical Observations and Microscopic Simulations." Physical Review E 62 (2): 1805–1824. https://doi.org/10.1103/PhysRevE.62.1805

Treiber, Martin, Arne Kesting, and Dirk Helbing. 2006. "Delays, Inaccuracies and Anticipation in Microscopic Traffic Models." Physica A: Statistical Mechanics and Its Applications 360 (1): 71–88. https://doi.org/10.1016/j.physa.2005.05.001.

Van Winsum, Wim. 1999. "The Human Element in Car Following Models." Transportation Research Part F: Traffic Psychology and Behaviour 2 (4): 207–211. https://doi.org/10.1016/S1369-8478(00)00008-5. [41]

Di Wang, Xiao-jun Zeng, and J. Keane. 2006. "A Survey of Hierarchical Fuzzy Systems." International Journal of Computational Cognition 4 (1): 18–29.

Wiedemann, Rainer. 1974. "Simulation des Straßenverkehrsflusses." In Proceedings of the Schriftenreihe des Instituts für Verkehrswesen der Universität Karlsruhe. Germany.

Wilson, Benjamin, William Qi, Tanmay Agarwal, John Lambert, Jagjeet Singh, Siddhesh Khandelwal, Bowen Pan, Ratnesh Kumar, Andrew Hartnett, Jhony Kaesemodel Pontes, Deva Ramanan, Peter Carr, and James Hays. 2023. "Argoverse 2: Next Generation Datasets for Self-Driving Perception and Forecasting." arXiv preprint arXiv:2301.00493.

Wolk, David A., and Andrew E. Budson. 2010. "Memory Systems." CONTINUUM: Lifelong Learning in Neurology 16: 15–28. https://doi.org/10.1212/01.CON.0000368257.30791.3a.

Yi-Rong, Kang, Sun Di-Hua, and Yang Shu-Hong. 2015. "A New Car-Following Model Considering Driver's Individual Anticipation Behavior." Nonlinear Dynamics 82 (3): 1293–1302. https://doi.org/10.1007/s11071- 015-2236-5.

Zhao, Wenjia Joyce, Russell Richie, and Sudeep Bhatia. 2022. "Process and Content in Decisions from Memory." Psychological Review 129 (1): 73–106. https://doi.org/10.1037/rev0000318

Zhou, Mofan, Xiaobo Qu, and Xiaopeng Li. 2017. "A Recurrent Neural Network Based Microscopic Car Following Model to Predict Traffic Oscillation." Transportation Research Part C: Emerging Technologies 84: 245–264. https://doi.org/10.1016/j.trc.2017.08.027.