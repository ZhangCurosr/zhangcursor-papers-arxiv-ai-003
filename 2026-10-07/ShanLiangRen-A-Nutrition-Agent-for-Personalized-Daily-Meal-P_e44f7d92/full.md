# ShanLiangRen: A Nutrition Agent for Personalized Daily Meal Planning

Miao Xie<sup>✉</sup>   
0520shui@163.com   
College of Information and Electrical   
Engineering, China Agricultural   
University   
Beijing, China   
Xiao Zhang   
B20243080802@cau.edu.cn   
College of Information and Electrical   
Engineering, China Agricultural   
University   
Beijing, China   
Ruixin Zhu<sup>✉</sup>   
zhuruixin07@126.com   
Key Laboratory of Precision Nutrition   
and Food Quality, China Agricultural   
University   
Beijing, China   
Yuan Wang   
wy929\_cs@163.com   
College of Information and Electrical   
Engineering, China Agricultural   
University   
Beijing, China   
Chunli Lv<sup>✉</sup>   
lvcl@cau.edu.cn   
College of Information and Electrical   
Engineering, China Agricultural   
University   
Beijing, China

## Abstract

Dietary nutrition planning plays an important role in chronic disease management and maintaining a healthy body. In applications, it must simultaneously satisfy personalized constraints and reasonable multidimensional nutritional goals. These two aspects often conflict, and user constraints evolve with feedback, resulting in a substantial gap between generic guidelines and executable plans. To bridge this gap, we first propose the personalized fully qantified multiobjective dietary planning problem (MDP). To tackle MDP, we develop a nutrition agent, ShanLiangRen. The system first transforms dietary specifications, nutrient data, user attributes and natural language requirements into an individualized constrained planning instance. It then employs an exact retrievalaugmented generation method to shrink the feasible candidate set from a large scale ingredient and recipe space. Finally, it adopts a refinement guided by Pareto principles, where an LLM iteratively revises candidate plans under deterministic nutrition computation and feedback from constraint verification. The system outputs fully quantified meal plans with explicit ingredients and portion sizes, together with reports on nutrition compliance that show constraint satisfaction and nutrient interval attainment. We have released the system online as a WeChat Program, ShanLiangRen. A demo video is available at https://www.youtube.com/watch?v=652OtY5VlGA.

## CCS Concepts

• Information systems → Information retrieval; Recommender systems.

## Keywords

Dietary Planning, Retrieval-Augmented Generation, Agent

ACM Reference Format: Miao Xie, Xiao Zhang, Yuan Wang, Ruixin Zhu, and Chunli Lv. 2026. Shan-LiangRen: A Nutrition Agent for Personalized Daily Meal Planning. In Proceedings ofthe 35th ACM International Conference on Information and Knowledge Management(CIKM’26), November07–11, 2026, Rome, Italy. ACM, New York, NY, USA, 5 pages. https://doi.org/10.1145/3799682.3840265

## 1 Introduction

Unhealthy diets have become an important driving factor of the global burden of chronic diseases: a systematic analysis covering 195 countries estimated that in 2017 about 11 million deaths were associated with dietary risks and caused about 255 million Disability-Adjusted Life Years (DALYs) [1]. Meanwhile, an increasing number of studies indicate that more scientific and individualized dietary interventions can afect metabolic health and disease risk[5, 23]. Existing dietary-planning studies follow two main directions. Health-oriented studies target chronic-disease risk factors, such as obesity and abnormal glucose levels, and develop intervention schemes or clinical nutrition management workflows centered on nutritional goals and food combinations [8, 10, 15]. Computational studies model menus or recipes as combinatorial optimization or recommendation problems under nutritional and preference constraints, enabling the automatic generation of executable diet plans [2, 9, 14, 18, 20, 21]. However, two limitations remain: (1) personalized constraints evolve with users’ physiological states and often conflict with reasonable nutritional goals, making guidelinebased plans dificult to follow and execute at the individual level; and (2) natural-language requirements lack a unified representation and must be reliably normalized into executable constraints and goals, while open-ended dialogue often drifts toward general nutrition Q&A with limited planning actionability [2, 12]. Recent studies have explored integrating LLMs with structured nutrition knowledge or multiobjective optimization to support personalized and explainable dietary recommendations [6, 29].

![](images/9f8abc049ebbf965831053242732554711e08bc96f8301213f28b5333121bcb6.jpg)

Accordingly, we formulate the fully qantified multiobjective dietary planning problem (MDP). Given dietary specifications and user information, MDP aims to dynamically generate a plan suited to the user’s physical condition, quantify its ingredients and portions, and address multidimensional targets beyond calories and macronutrients. Solving MDP involves three coupled challenges. First, the system must formulate an individualized planning instance by converting general dietary specifications into nutrient target intervals based on the user’s physical attributes, while incor porating stable restrictions and daily preferences as requirements or priorities. Second, it must construct feasible meal candidates from structured food and recipe tables with deterministic nutrient composition data, ensuring reliable evidence for ingredients, portions, and nutrient values. Third, it must refine feasible candidates across multiple nutritional objectives and personalized preferences, using deterministic nutrient computation to verify compliance with the target intervals.

To address MDP, we develop ShanLiangRen, an interactive system for personalized daily dietary planning. It focuses on constructing feasible candidates from a large ingredient space and producing executable solutions under multiple objectives through two key designs. First, an exact RAG mechanism based on subgraph matching reduces the feasible search space and generates candidate plans. It represents dietary specifications and user constraints as retrieval filters, abstracts meal structure, category coverage, and combination principles into a query subgraph, and retrieves combinations satisfying user requirements through exact structural matching over the knowledge graph. Second, a Pareto guided mul tiobjective refinement strategy balances constraint satisfaction and nutritional goals. Through a feedback loop of deterministic nutrient computation and constraint verification, it evaluates and optimizes the candidates, improving nutritional objectives while satisfying personalized constraints, and outputs the recommended plan.

This demo makes three contributions: (1) we define the personalized fully qantified multiobjective dietary planning problem (MDP); (2) we design an exact retrieval and Pareto guided refinement pipeline with deterministic nutrient verification; and (3) we develop a system for personalized daily meal planning.

## 2 Problem Definition

We represent a dietary planning task as an instance $I = ( S , T , u )$ The planning instance is grounded in external nutritional knowledge rather than generated nutrition claims. Specifically, $S = ( G , P )$ denotes the dietary specifications, where � is derived from the Chinese Dietary Guidelines (2022) and � follows the Chinese Food Guide Pagoda (2022) [4].� is the ingredient nutrient table built from the China Food Composition Tables [25, 26], and $u = ( u _ { \mathrm { a t t r } } , u _ { \mathrm { r e q } } )$ is the user information. Here, $u _ { \mathrm { a t t r } }$ describes the user’s basic physical signs and health attributes, and $u _ { \mathrm { r e q } }$ describes the user’s dynamic requirements. Therefore, the LLM is used to normalize user requirements and refine candidate plans, while nutrient values, target intervals, and constraint verification are grounded in external specifications and structured data. The system outputs a fully quantified daily plan $x = \{ x _ { m } \mid m \in M \}$ , where � is the set of meals, and each $x _ { m }$ gives the dishes/ingredients and their portion sizes. A feasible solution must satisfy two types of hard constraints: $x \Vdash C _ { u } ( u _ { \mathrm { r e q } } )$ and $x \Vdash C _ { \mathrm { q u a l } } ( S )$ , corresponding respectively to the personalized requirements that the user must satisfy and the qual itative healthy structural principles that the specifications must follow. Based on �, we compute the daily intake nutrition vector $n ( x ) = { \mathrm { C o m p u t e } } ( x ; T )$ , and derive the individualized nutrition recommendation set $R ( u _ { \mathrm { a t t r } } , S ) = \psi ( u _ { \mathrm { a t t r } } , S )$ from $u _ { \mathrm { a t t r } }$ and $S ,$ based on which we construct the nutrition objective set $J _ { \mathrm { n u t } } ( S , u _ { \mathrm { a t t r } } )$ . Here $C _ { u } ( u _ { \mathrm { r e q } } )$ mainly covers individual hard restrictions such as allergies, taboos, meal structure requirements, and explicit daily preferences, while $C _ { \mathrm { q u a l } } ( S )$ captures dietary structure principles derived from guidelines such as category coverage and meal balance.

![](images/fc396c06d8f5e348dbfc7836e6d48db80b6a16aac1e958ed737bdd3e7acbd505.jpg)  
Figure 1: Overall architecture of the ShanLiangRen system.

For each $j \in J _ { \mathrm { n u t } } ( S , u _ { \mathrm { a t t r } } ) , R ( u _ { \mathrm { a t t r } } , S )$ provides its reasonable intake interval $[ L _ { j } , U _ { j } ]$ . To characterize the deviation degree of plan � on objective � relative to this interval, we define the interval deviation:

$$
d _ { j } ( x ) = \frac { \operatorname* { m a x } \{ 0 , L _ { j } - n _ { j } ( x ) , n _ { j } ( x ) - U _ { j } \} } { \operatorname* { m a x } \{ \varepsilon , U _ { j } - L _ { j } \} } ,\tag{1}
$$

where $n _ { j } ( x )$ denotes the intake value of plan � on nutrition objective $j ,$ and � is a very small constant. Under the premise of satisfying the hard constraints, the goal is to make all nutrition objectives fall into their reasonable intervals as much as possible, and minimize the overall deviation:

$$
\begin{array} { l } { \displaystyle \operatorname* { m i n } _ { \begin{array} { l } { L ( x ) = \displaystyle \sum _ { j \in J _ { \mathrm { n u t } } ( S , u _ { \mathrm { a t t r } } ) } w _ { j } \cdot d _ { j } ( x ) } \\ { \mathrm { s . t . } x \mid = C _ { u } ( u _ { \mathrm { r e q } } ) , \quad x \mid = C _ { \mathrm { q u a l } } ( S ) . } \end{array} } } \end{array}\tag{2}
$$

where $w _ { j }$ denotes the importance weight of nutrition objectives.

## 3 System Design

Overview. Given an MDP instance, ShanLiangRen follows a five module online pipeline (Fig.1): (1) individual constraint and objective formulation, (2) exact retrieval for feasible space reduction, (3) quantified candidate generation, (4) deterministic nutrition evaluation, and (5) Pareto guided refinement with explanation. Implementation. ShanLiangRen employs GPT-5.4 as its LLM backbone. The LLM is responsible for requirement normalization, candidate proposal, and targeted plan repair.

(1) Dynamic Constraint Extraction and Task Structuring. The system first receives user information $u = ( u _ { \mathrm { a t t r } } , u _ { \mathrm { r e q } } )$ . The LLM performs structured extraction on $u _ { \mathrm { r e q } }$ to form executable user constraints $C _ { u } ( u _ { \mathrm { r e q } } )$ , covering hard requirements such as taboos/allergies, taste preferences, and meal/course structure (e.g., number of meals, number of dishes per meal, no repetition)[19, 27]. Meanwhile, the system constructs the individualized nutrition objective set $J _ { \mathrm { n u t } } ( S , u _ { \mathrm { a t t r } } )$ based on $u _ { \mathrm { a t t r } }$ and the specifications $S ,$ and generates a recommended intake interval $[ L _ { j } , U _ { j } ]$ . This step uniformly maps user inputs to the formal objects in the problem definition, providing a consistent interface for retrieval reduction and plan generation. In our demo, ShanLiangRen adopts a controlled, plan based interaction design. Stable user attributes and stable restrictions are collected through structured fields, while daily preferences and revision requests can be expressed in natural language with respect to the current meal plan. This design preserves personalized expressiveness while preventing the interaction from drifting into open-ended nutrition consultation.

(2) Exact RAGfor Feasible Space Reduction. To avoid combinatorial search over the full library, the system adopts exact RAG based on subgraph matching to reduce the candidate space to $R _ { u }$ [7, 11, 13, 24, 28] Specifically: (1) exact filtering of hard constraints: it first removes items that violate $C _ { u } ( u _ { \mathrm { r e q } } )$ in the ingredient/recipe library $( \mathrm { e . g . }$ , forbidden ingredients, infeasible meal structures), and combines the qualitative specification constraints $C _ { \mathrm { q u a l } } ( S )$ for structural elimination (e.g., basic feasibility of category coverage/combination templates). For example, a requirement such as avoiding a specific ingredient, including a vegetable category in lunch, and preventing repeated staple foods can be converted into exclusion predicates at the ingredient level, category coverage constraints, and meal structure relations. Exact retrieval means that candidates violating these hard predicates or structural relations are excluded before LLM generation, so the LLM only operates on a user relevant feasible subspace. (2) subgraph matching retrieval: it abstracts “meal structure/category coverage/combination principles” as a query graph, and performs exact matching retrieval under structural constraints on the filtered set to obtain the candidate subspace most relevant to the current user context, yielding $R _ { u } \subseteq D$ After this reduction, subsequent LLM generation and refinement are performed only within $R _ { u }$ . Unlike conventional RAG that primarily retrieves semantically relevant textual contexts, our exact RAG retrieves structured meal candidates that explicitly satisfy query graph relations. If no feasible candidate satisfies all hard constraints, ShanLiangRen does not silently relax safety-critical requirements. Instead, the system reports the infeasible constraint combination and requests revision of lower-priority daily preferences before replanning.

(3) Candidate Generation. Given $R _ { u } ,$ the LLM generates a diverse set of candidate daily plans $\{ x ^ { ( 1 ) } , \ldots , x ^ { ( K ) } \}$ [22]. Each candidate explicitly specifies meal partitioning and ingredient/dish portion sizes, ensuring “fully quantified executability”. Since the generation is constrained by $R _ { u } ,$ , candidate plans inherit the retrieved feasibility conditions and are easier to verify and repair than unconstrained LLM outputs. The generation stage enforces structural hard requirements and promotes diversity for trade-of selection. Deterministic Evaluation. For each candidate $x ,$ the evaluator computes the daily intake vector $n ( x ) = { \mathrm { C o m p u t e } } ( x ; T )$ based on the element table $T ,$ and checks whether hard constraints are satisfied $( x \mid = C _ { u } ( u _ { \mathrm { r e q } } )$ and $x \mid = C _ { \mathrm { q u a l } } ( S ) )$ . Then, according to the recommended interval $[ L _ { j } , U _ { j } ]$ of each objective, the system computes the deviation $d _ { j } ( x )$ and aggregates it to obtain the overall deviation $L ( x )$ . The resulting deficit, excess, and interval attainment status are used both for user explanation and for the next refinement round. The LLM serves as a semantic proposal operator that combines retrieved ingredients according to natural-language preferences and produces targeted modifications while preserving unafected constraints.

(4) Pareto Guided Refinement with LLM. Unlike traditional Pareto optimization, we agentize the candidate generation and repair operators: the LLM performs iterative “minimal modification” driven by deterministic evaluation feedback, rather than using handcrafted search operators [16]. The system maintains an evaluated candidate pool and performs dominance filtering based on the deviation vector $d ( x ) = \{ d _ { j } ( x ) \}$ (the smaller the better): it keeps candidate solutions that are no worse on all objective deviations and strictly better on at least one objective deviation, retains candidates reflecting diferent trade-ofs, and removes dominated solutions that are not better on any considered dimension. Then, the system feeds the interval deviation feedback produced by the evaluator into the LLM, and the LLM performs “minimal-modification” repairs, entering the next round until reaching an iteration limit or the improvement becomes stable.

(5) Recommendation and Explanation. After the candidate pool converges, the system outputs: (1) the recommended plan $x ^ { * }$ selected from the non-dominated set according to the user’s preference priorities; (2) an explainable report, showing the satisfaction of key hard constraints, nutrition computation results �(�), interval attainment for each objective (whether it falls into $[ L _ { j } , U _ { j } ] )$ ) and the corresponding deviation $d _ { j } ( x )$ /deficit or excess.

Backend Knowledge Base. ShanLiangRen maintains a structured ingredient–recipe knowledge base in the backend, and uses the ingredient nutrition element table as the computation basis of the deterministic evaluator[21]. The knowledge base contains structured information including recipe entities, ingredient entities, ingredient category hierarchies, and macro and micro nutrient attributes, enabling the system to abstract requirements such as “meal-type structure/category coverage/combination principles” into query subgraphs and to perform exact matching retrieval within the candidate subspace. Data Scale: the current backend includes 1, 550, 150 recipes and 64, 127 ingredients; each ingredient is annotated with 64 nutrient attribute dimensions, and a category hierarchy is provided. The backend knowledge base supports two diferent functions: structured retrieval and deterministic verification. Recipe and ingredient entities are used to construct candidate meal combinations under restriction and category coverage constraints, while the nutrient table is used to compute the actual intake values of generated plans.

## 4 System in Action

As shown in Fig. 2, ShanLiangRen has been deployed as an online WeChat Mini Program, where users can access the system, submit dietary requirements, and receive plan based dynamic updates [3, 17]: it stores stable restrictions as stable constraints, and incrementally re-plans when users revise daily needs via structured options and comments on the current plan. On the output side, the system generates a fully quantified daily plan with three meals, refined to dishes/ingredients and portion sizes, and performs deterministic nutrition computation based on the ingredient ele ment table, providing interval comparisons for indicators such as energy and macronutrients as well as attainment verification of requirements from dietary specifications, and finally presents an ex ecutable personalized daily plan in the form of “recommended plan + explainable verification report”. In the demo, the user first fills in basic physical signs and stable preferences (Fig.2.2), and inputs daily requirements (Fig.2.1); the system immediately parses them into structured hard constraints and generates a fully quantified three-meal daily plan (Fig.2.3). Then, the interface shows nutrition computation results and specification attainment verification (Fig.2.4), and the user can choose among the recommended plan and several trade-of alternative plans. If the user temporarily adjusts requirements (e.g., taboos or meal structure changes), the system triggers fast incremental replanning and synchronously updates the explainable verification report.

![](images/1804322ef38677327c62daeccd5bcd70bdbccae9df077793c15f917573297ac9.jpg)  
Figure 2: Interface of the ShanLiangRen system.

## 5 Evaluation

We evaluate ShanLiangRen on 300 distinct planning instances covering common personalized requirements, including ingredient taboos, dietary preferences, meal-structure constraints, and nutrition goals. We compare ShanLiangRen with direct LLM generation and constrained prompting, where all methods receive the same user profiles and daily requirements. All generated plans are manually verified by a Ph.D. in nutrition based on the China Food Composition Tables [25, 26], and the verification results are used to compute five metrics. PCS is the percentage of satisfied individual hard constraints. EPC is the percentage of plans with complete dish, ingredient, and portion information. NIA is the percentage of checked nutrient objectives whose verified values fall within the recommended intervals. Plans lacking suficient portion information are marked as incomplete, and their corresponding nutrient objectives are treated as unverifiable. AND is the average normalized nutrient deviation �(�) in Eq. (2), where lower is better. Latency is the average end-to-end generation time in seconds.

Table 1: Overall comparison on 300 user planning cases.
<table><tr><td>Method</td><td>PCS (%)↑</td><td>NIA (%)↑</td><td>EPC (%)↑</td><td>AND (norm.)↓</td><td>Lat. (s)↓</td></tr><tr><td>GPT-5.4</td><td>87.5</td><td>60.4</td><td>85.9</td><td>0.17</td><td>46</td></tr><tr><td>Gemini-3</td><td>75.0</td><td>55.2</td><td>80.2</td><td>0.25</td><td>19</td></tr><tr><td>Doubao-seed 2.0</td><td>87.5</td><td>61.1</td><td>79.9</td><td>0.18</td><td>27</td></tr><tr><td>ShanLiangRen</td><td>100.0</td><td>78.9</td><td>100.0</td><td>0.09</td><td>61</td></tr></table>

As shown in table 1, ShanLiangRen achieves 100.0% PCS and EPC, with the highest NIA of 78.9% and the lowest AND of 0.09, showing full individual constraint satisfaction, complete executable portion information, and lower overall nutrient deviation under full nutrition verification. GPT-5.4 and Gemini-3 provide partially complete portions but show lower PCS and NIA, while Doubaoseed 2.0 performs competitively on NIA but has weaker portion completeness.Although ShanLiangRen incurs a moderate latency overhead, the additional time is acceptable for daily meal planning.

## 6 Conclusion

We presented ShanLiangRen, an interactive nutrition agent for personalized daily meal planning. We first defined the personalized fully qantified multiobjective dietary planning problem (MDP), then designed an exact retrieval and refinement pipeline guided by Pareto principles with deterministic nutrient verification, and finally developed an interactive system that outputs executable meal plans with portion sizes and compliance reports. The evaluation shows that ShanLiangRen improves constraint satisfaction, nutrient attainment, executability, and verifiability over direct LLM generation, at the cost of acceptable additional latency.

## Acknowledgments

This work was supported by the China Agricultural University “Young Researcher” Start-up Fund No. QNYJY2024144 and the Visiting Scholar Program of the China Scholarship Council (CSC) No. 202506350123.

## References

[1] Ashkan Afshin, Patrick John Sur, Kairsten A. Fay, Leslie Cornaby, and Giannina Ferrara. 2019. Health efects of dietary risks in 195 countries, 1990–2017: a systematic analysis for the Global Burden of Disease Study 2017. The Lancet 393, 10184 (2019), 1958–1972. doi:10.1016/S0140-6736(19)30041-8

[2] Grace Ataguba and Rita Orji. 2025. Exploring Large Language Models for Person alized Recipe Generation and Weight-Loss Management. ACM Trans. Comput. Healthcare 6, 2, Article 22 (April 2025), 57 pages.

[3] Weize Chen, Yusheng Su, Jingwei Zuo, Cheng Yang, Chenfei Yuan, Chi-Min Chan, Heyang Yu, Yaxi Lu, Yi-Hsin Hung, Chen Qian, Yujia Qin, Xin Cong, Ruobing Xie, Zhiyuan Liu, Maosong Sun, and Jie Zhou. 2024. AgentVerse: Facilitating Multi Agent Collaboration and Exploring Emergent Behaviors. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net. https://openreview.net/forum?id=EHg5GDnyq1

[4] Chinese Nutrition Society. 2022. Chinese Resident Dietary Guidelines (2022). People’s Medical Publishing House, Beijing

[5] K. D. Corbin, D. Igudesman, S. R. Smith, K. Zengler, and R. Krajmalnik-Brown. 2025. Targeting the Gut Microbiota’s Role in Host Energy Absorption With Precision Nutrition Interventions for the Prevention and Treatment of Obesity. Nutrition Reviews 83, 10 (October 2025), 1928–1943. PMID: 40233201; PMCID: PMC12422011. doi:10.1093/nutrit/nuaf046

[6] Fan Gao, Xinjie Zhao, Ding Xia, Zhongyi Zhou, Rui Yang, Jinghui Lu, Hang Jiang, Chanjun Park, and Irene Li. 2025. HealthGenie: A Knowledge-Driven LLM Framework for Tailored Dietary Guidance. In Proceedings of the 34th ACM International Conference on Information and Knowledge Management. Association for Computing Machinery, 6639–6643. doi:10.1145/3746252.3761479

[7] Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Ming-Wei Chang. 2020. REALM: Retrieval-Augmented Language Model Pre-Training. CoRR abs/2002.08909 (2020). arXiv:2002.08909 https://arxiv.org/abs/2002.08909

[8] Maria Hassapidou, Antonis Vlassopoulos, Marianna Kalliostra, Elisabeth Govers, Hilda Mulrooney, Louisa Ells, Ximena Ramos Salas, Giovanna Muscogiuri, Teodora Handjieva Darleska, Luca Busetto, Volkan Demirhan Yumuk, Dror Dicker, Jason Halford, Euan Woodward, Pauline Douglas, Jennifer Brown, and Tamara Brown. 2023. European Association for the Study of Obesity Position Statement on Medical Nutrition Therapy for the Management of Overweight and Obesity in Adults Developed in Collaboration with the European Federation of the Asso ciations of Dietitians. Obesity facts 16, 1 (2023), 11—28. doi:10.1159/000528083

[9] Tanvir Islam, Anika Rahman Joyita, Md. Golam Rabiul Alam, Mohammad Mehedi Hassan, Md. Rafiul Hassan, and Rafaele Gravina. 2023. Human-Behavior Based Personalized Meal Recommendation and Menu Planning Social System. IEEE Transactions on Computational Social Systems 10, 4 (2023), 2099–2110. doi:10.1109/TCSS.2022.3213506

[10] S. Karjoo, A. Braglia-Tarpey, A. P. Chan, A. G. Ayala Germán, R. E. Herdes, N. Pai, D. Sierra-Velez, B. Whitehead, R. E. Quiros-Tejeira, and D. Duro. 2025. Evidence-based Review of the Nutritional Treatment of Obesity and Metabolic Dysfunction-associated Steatotic Liver Disease in Children and Adolescents. Journal ofPediatric Gastroenterology and Nutrition 81, 3 (September 2025), 485– 496. Epub 2025 Jun 17. PMID: 40525381. doi:10.1002/jpn3.70099

[11] Omar Khattab and Matei Zaharia. 2020. ColBERT: Eficient and Efective Passage Search via Contextualized Late Interaction over BERT. In Proceedings ofthe 43rd International ACM SIGIR conference on research and development in Information Retrieval, SIGIR 2020, Virtual Event, China, July 25-30, 2020, Jimmy X. Huang, Yi Chang, Xueqi Cheng, Jaap Kamps, Vanessa Murdock, Ji-Rong Wen, and Yiqun Liu (Eds.). ACM, 39–48. doi:10.1145/3397271.3401075

[12] D. W. Kim, J. S. Park, K. Sharma, A. Velazquez, L. Li, J. W. Ostrominski, T. Tran, R. H. Seitter Peréz, and J. H. Shin. 2024. Qualitative Evaluation of Artificial Intelligence Generated Weight Management Diet Plans. Frontiers in Nutrition 11 (March 2024), 1374834. PMID: 38577160; PMCID: PMC10991711. doi:10.3389/fnut.2024.1374834

[13] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. In Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual, Hugo Larochelle, Marc’Aurelio Ranzato, Raia Hadsell, Maria-Florina Balcan, and Hsuan-Tien Lin (Eds.). https://proceedings.neurips.cc/paper/2020/hash 6b493230205f780e1bc26945df7481e5-Abstract.html

[14] Diya Li and Mohammed J. Zaki. 2020. RECIPTOR: An Efective Pretrained Model for Recipe Representation Learning. In Proceedings ofthe 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (Virtual Event, CA, USA) (KDD ’20). Association for Computing Machinery, New York, NY, USA, 1719–1727. doi:10.1145/3394486.3403223

[15] Neel H Mehta, Samantha L Huey, Rebecca Kuriyan, Juan Pablo Peña-Rosas, Julia L Finkelstein, Sangeeta Kashyap, and Saurabh Mehta. 2024. Potential Mechanisms of Precision Nutrition-Based Interventions for Managing Obesity. Advances in Nutrition 15, 3 (2024), 100186. doi:10.1016/j.advnut.2024.100186

[16] Tao Meng, Sidi Lu, Nanyun Peng, and Kai-Wei Chang. 2022. Controllable Text Generation with Neurally-Decomposed Oracle. In Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022, Sanmi Koyejo, S. Mohamed, A. Agarwal, Danielle Belgrave, K. Cho, and A. Oh (Eds.). http://papers.nips.cc/paper\_files/paper/2022/hash/ b40d5797756800c97f3d525c2e4c8357-Abstract-Conference.html

[17] Joon Sung Park, Joseph C. O’Brien, Carrie Jun Cai, Meredith Ringel Morris, Percy Liang, and Michael S. Bernstein. 2023. Generative Agents: Interactive Simulacra of Human Behavior. In Proceedings ofthe 36th Annual ACM Symposium on User Interface Software and Technology, UIST 2023, San Francisco, CA, USA, 29 October 2023- 1 November 2023, Sean Follmer, Jef Han, Jürgen Steimle, and Nathalie Henry Riche (Eds.). ACM, 2:1–2:22. doi:10.1145/3586183.3606763

[18] Omur Sahin and Gizem Aytekin-Sahin. 2024. Open-source multi-objective optimization software for menu planning. Expert Systems with Applications 252 (2024), 124213. doi:10.1016/j.eswa.2024.124213

[19] Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. 2023. Toolformer: Language Models Can Teach Themselves to Use Tools. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine (Eds.). http://papers.nips.cc/paper\_files/paper/2023/ hash/d842425e4bf79ba039352da0f658a906-Abstract-Conference.html

[20] Yijun Tian, Chuxu Zhang, Zhichun Guo, Chao Huang, Ronald A. Metoyer, and Nitesh V. Chawla. 2022. RecipeRec: A Heterogeneous Graph Learning Model for Recipe Recommendation. In Proceedings ofthe Thirty-First International Joint Conference on Artificial Intelligence, IJCAI 2022, Vienna, Austria, 23-29 July 2022, Luc De Raedt (Ed.). ijcai.org, 3466–3472. doi:10.24963/IJCAI.2022/481

[21] Yijun Tian, Chuxu Zhang, Zhichun Guo, Yihong Ma, Ronald A. Metoyer, and Nitesh V. Chawla. 2022. Recipe2Vec: Multi-modal Recipe Representation Learning with Graph Neural Networks. In Proceedings of the Thirty-First International Joint Conference on Artificial Intelligence, IJCAI 2022, Vienna, Austria, 23-29 July 2022, Luc De Raedt (Ed.). ijcai.org, 3473–3479. doi:10.24963/IJCAI.2022/482

[22] Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V. Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. 2023. Self-Consistency Improves Chain of Thought Reasoning in Language Models. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net. https://openreview.net/forum?id=1PL1NIMMrw

[23] Miao Xie, Yao Hu, Hao Xu, Xiaoxia Li, Yu Dong, Ran Wang, Jie Guo, Jingjing He, Huiyu Chen, Jiayue Guo, Siyuan Liu, Pengjie Wang, Fazheng Ren, Zhihong Fan, J. Alfredo Martinez, and Ruixin Zhu. 2026. Advances, Challenges, and Future Directions in Precision Nutrition by Machine Learning-Based Prediction of Glycemic Responses to Foods: A Narrative Review. The American Journal of Clinical Nutrition (2026), 101322. doi:10.1016/j.ajcnut.2026.101322

[24] Miao Xie, Xiao Zhang, Yi Li, and Chunli Lv. 2026. Structure Guided Retrieval-Augmented Generation for Factual Queries. In Proceedings ofthe 64th Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers). Association for Computational Linguistics, San Diego, California, United States, 40350–40370. doi:10.18653/v1/2026.acl-long.1873

[25] National Institute for Nutrition and Health, Chinese CDC Yang Yuexin. 2018. Chinese Food Composition Tables (Standard Edition, 6th Edition, Volume 1). Peking University Medical Press.

[26] National Institute for Nutrition and Health, Chinese CDC Yang Yuexin. 2019. Chinese Food Composition Tables (Standard Edition, 6th Edition, Volume 2). Peking University Medical Press

[27] Shunyu Yao,Jefrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R. Narasimhan, and Yuan Cao. 2023. ReAct: Synergizing Reasoning and Acting in Language Models. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net. https://openreview. net/forum?id=WE\_vluYUL-X

[28] Xiao Zhang, Yang Wan, Yi Li, Miao Xie, and Chunli Lv. 2026. MC-RAG System: A Structure-Driven RAG System for Multi-Constraint Queries. arXiv preprint arXiv:2607.10151 (2026). arXiv:2607.10151 [cs.IR] https://arxiv.org/abs/2607.10151

[29] Zheyuan Zhang, Zehong Wang, Tianyi Ma, Varun Sameer Taneja, Sofia Nelson, Nhi Ha Lan Le, Keerthiram Murugesan, Mingxuan Ju, Nitesh V. Chawla, Chuxu Zhang, and Yanfang Ye. 2025. MOPI-HFRS: A Multi-objective Personalized Health-aware Food Recommendation System with LLM-enhanced Interpretation. In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining. Association for Computing Machinery, 2860–2871. doi:10.1145/ 3690624.3709382