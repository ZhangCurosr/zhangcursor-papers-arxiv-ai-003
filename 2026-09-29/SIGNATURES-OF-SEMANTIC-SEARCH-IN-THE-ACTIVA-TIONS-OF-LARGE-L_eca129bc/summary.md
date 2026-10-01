---
title: "SIGNATURES-OF-SEMANTIC-SEARCH-IN-THE-ACTIVA-TIONS-OF-LARGE-L"
source: https://arxiv.org/pdf/2609.35599v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 10:11:05"
field: "语言模型机制解释性"
keywords: ["semantic fluency task", "mechanistic interpretability", "Jacobian-lens", "explore-exploit", "steering vectors", "large language models"]
innovations: ["首次用J-lens揭示LLM语义搜寻中的探索-利用双模态激活签名", "通过J-space耗竭度量证明概念斑块 depletion 驱动切换决策", "构建通用switch direction并实现因果性探索-利用行为调控"]
benchmarks: ["Semantic Fluency Task (SFT)", "动物命名类别任务"]
---

# 论文速读：SIGNATURES OF SEMANTIC SEARCH IN THE ACTIVATIONS OF LARGE LANGUAGE MODELS

## 一句话总结
本研究将语义搜寻框架（semantic foraging framework）延伸至大语言模型，通过Jacobian-lens（J-lens）机制解释性技术发现，LLM在进行语义流畅性任务（SFT）时，其内部激活呈现区分"探索"（cluster switching）与"利用"（item clustering）行为的不同表征签名，且可通过steering vectors因果性地调控这种搜索平衡。

## 研究问题与动机
- 人类在语义流畅性任务中呈现集群-切换模式（clustering-switching），背后由探索-利用的语义搜寻过程驱动，但LLM执行同类任务时是否具备相似的内部表征机制尚不明确。
- 现有机制解释性研究多关注任务级或序列级表征（如sparse autoencoder跟踪事件边界、steering reasoning行为），但未将J-lens等工具用于自然执行语义搜寻任务时的动态分析。
- 理解LLM内部探索-利用平衡的表征对调控生成文本的语义节奏、覆盖度及类搜索任务（工具选择、代码导航、上下文检索）的效率具有潜在应用价值。
- 现有相关研究（Lacosse et al., 2026b）仅指出LLM生成列表时有类似人类聚类模式且switch事件前next-token概率较低，但未建立从表征签名到因果操控的完整证据链。

## 核心贡献（创新点）
- **首次将J-lens用于语义搜寻动态分析**：用J-lens从中间层残差流映射到token级激活，证明概念级激活可预测并前摄switch事件，揭示了LLM内部"探索-利用"区分的表征基础。
- **发现J-space耗竭驱动切换决策**：提出以J-space中当前类别未产出的项目数量（in-category J-space supply）作为"语义斑块"剩余资源的量化指标，证明其耗竭程度显著预测切换概率，且效应强于单纯类别池枯竭效应。
- **揭示高层类别标签的前摄激活与竞争机制**：发现抽象类别相关标签（如"water"、"pet"）的J-lens激活在切换前渐进上升并在switch事件处尖峰，支持类别选择由竞争性激活驱动的假设。
- **构建因果steering向量实现探索-利用调控**：通过两种路径（item-level与category-level）推导steering vectors，成功将switch概率从0.29调控至0.70（正向）或0.07（负向），证实激活签名对switch决策具有因果影响。
- **识别通用switch方向并验证ramping现象**：在Study 2中通过PCA+逻辑回归识别残差流中预测switch/clustering的通用方向，发现该方向加载量随集群内已产出项目数增加而渐进上升（ramping），与人类神经活动中观察到的小脑/海马ramping激活相呼应。

## 方法详解
- **模型与任务设置**：使用5个开源指令微调LLM（Gemma-2-9B/2B、Llama-3.1-8B、Llama-3.2-3B、Qwen2.5-7B），执行SFT提示："Name as many different animals as you can, one after another, separated by commas. Just the list."，使用nucleus sampling（temperature=0.9, top-p=0.95），最大200 tokens，每个实验100次随机种子迭代。
- **J-lens激活提取**：使用预计算的model-specific J-lens（来自Neuronpedia），将中间层残差流映射到token级激活；J-space定义为top-25最强J-lens激活的概念集合。
- **J-space耗竭度量**：计算in-category J-space supply = J-space中属于当前命名类别且尚未产出的项目数量，分析其与switch概率的负相关关系。
- **类别标签激活分析**：识别扩展Troyer类别标签（如"water"、"domestic"、"pet"等），比较target vs. unrelated category的J-lens激活差异，测量switch前后lag值上的激活动态。
- **Steering向量推导**：
  - Item-level: $d_{T,L} = \text{normalize}\left(\frac{1}{|T|}\sum_{w \in T} v_{w,L}\right)$，其中$v_{w,L} = \text{normalize}(J[L]^\top W_U[w]^\top)$
  - Category-level: $d_{c,L} = \text{normalize}(J[L]^\top W_U[c]^\top)$
  - 注入方式：$h_{L,t} \leftarrow h_{L,t} + s \cdot r_L \cdot d_L$，其中$s$为steering系数，$r_L$为参考生成的残差范数均值
- **通用switch方向识别（Study 2）**：在anticipatory（switch前1 token）和event（switch时刻）位置收集残差流表示，经PCA降维至30维后训练L2正则化逻辑回归（$\lambda=0.1$），通过载荷矩阵$W$与回归系数$\beta$重建原始空间方向：$\hat{d}_L = W\beta / \|W\beta\|_2$
- **统计方法**：线性回归（LR）+ Wald检验，Benjamini-Hochberg校正多重比较；Fisher精确检验比较steering vs. noise控制；AUC评估分类性能

## 实验与结果
- **数据集**：动物类别SFT生成数据，使用Troyer扩展类别体系（reptiles/amphibians、water、domestic、pet、forest等），跨类别划分train/test以最小化语义泄露。
- **基线方法**：与Lacosse et al. (2026b) 的发现对比（switch事件前next-token概率较低、残差流表征差异）。
- **主要结果**：
  - 图2A：switch参与项目的next-token概率显著低于cluster项目（所有模型$\bar{\chi^2} \geq 32.39, p < 0.0001$）
  - 图2C：J-lens激活在switch前显著低于cluster（所有模型layer 8+，$\chi^2 \geq 98.17, p < 0.0001$）
  - 图2D：J-space耗竭预测switch概率（所有模型中间层$r \leq -0.31, p < 0.0001$），且效应强于整体池枯竭（$F \geq 19.94, p < 0.0001$）
  - 图3：目标类别标签激活在switch前渐进上升（$F \geq 848.35, p < 0.0001$），unrelated标签在switch处骤降后回升，支持类别竞争假设
  - 图4：Item-level steering在Gemma-9B layer 40使target switch概率达0.68；category-level steering在layer 32达0.45
  - 图5B-C：通用方向预测switch的AUC最高达0.87（event位置，layer 16）；anticipatory方向AUC为0.72（layer 26）
  - 图5D：anticipatory方向加载量随集群内项目数增加而渐进上升（所有模型$\chi^2 \geq 20.44, p < 0.0001$）
  - 图5E-F：正向steering使Gemma-9B switch率从0.29升至0.70；负向steering降至0.07（layer 26，$p < 0.0001$）
- **最强结果**：通用direction steering在Gemma-9B layer 26实现switch率从0.29到0.70的显著提升（+141%相对提升），且负向steering可有效抑制switch至0.07

## 相关工作脉络
- **Hills et al. (2012)** 提出人类语义记忆搜寻的foraging框架，区分clustering（exploit）与switching（explore），本文将其延伸至LLM内部表征层面。
- **Lacosse et al. (2026b)** 发现LLM生成SFT列表时switch事件前next-token概率较低，本文在此基础上进一步定位到中间层J-lens激活并建立因果调控。
- **Gurnee et al. (2026)** 提出J-lens方法与J-space概念（类比全局工作空间），本文首次将其应用于动态search任务分析。
- **Pal et al. (2023) Future Lens** 从隐藏状态预测后续token，与本文J-lens方法思想相近但任务设定不同（本文为主动生成任务中的前摄表征）。
- **Venhoff et al. (2025)** 通过steering vectors操控reasoning行为（如uncertainty表达），本文扩展steering技术至探索-利用行为调控。
- **Lindsey et al. (2025)** 发现LLM在写诗时内部表征候选end-of-line rhyme词，本文与此类似地证明类别标签的前摄激活。

## 局限性与未来方向
- 任务局限于简化的SFT（动物命名），尚不清楚所述动态是否泛化至自然话语生成或其他search任务（如网络搜索、工具选择、代码导航）。
- LLM规模较小（2B-9B），前沿大模型（如175B+）是否呈现相同 dynamics 待验证。
- Steering效果存在模型与层选择性（如Llama-3.2-8B对负向steering不敏感），机制尚不完全清楚。
- 未区分具体semantic content与generic explore-exploit dynamics的相对贡献。
- 未来可探索：在discourse generation中验证switch-associated directions的cross-task泛化性；在更大规模模型中复现；研究正向vs.负向steering不对称性的根源。

## 研究启发与可借鉴点
- **J-lens用于动态任务分析**：将J-lens从静态"概念表征"验证拓展到动态search过程追踪，为后续研究机制解释性技术提供了方法论范式。
- **Steering vectors的因果验证流程**：从"关联发现"（activation predicts behavior）到"因果操控"（steering biases behavior）的完整证据链，可作为机制解释性研究的标准流程参考。
- **Generic vs. specific representations的分离策略**：Study 2通过PCA+逻辑回归识别"generic switch direction"而非针对特定类别的方向，为区分domain-general与domain-specific表征提供了可行方案。
- **Ramping动态的跨物种可比性**：LLM中观察到的anticipatory方向加载量渐进上升与人类小脑/海马ripple burst/ramping激活形成呼应，为跨物种认知机制比较提供了量化接口。
- **应用迁移机会**：调控steering coefficient可能用于控制生成文本的语义覆盖率（高switch=广覆盖低深度）vs. 聚焦深度（低switch=深挖单类别），适用于需要平衡探索与利用的内容生成场景。

## 关键术语表
- **Semantic Fluency Task (SFT)**：要求受试者在限定时间内尽可能多地命名某类别概念（如动物）的经典心理学任务，用于研究语义记忆检索结构。
- **Clustering & Switching**：SFT产出的两个基本行为模式；clustering指连续产出同语义类别项目（exploit），switch指跨类别跳转（explore）。
- **Jacobian-lens (J-lns)**：一种机制解释性技术，通过 Jacobian 矩阵将中间层残差流表示映射回token级激活，揭示模型内部的概念表征。
- **J-space (Jacobian-space)**：由某层top-K最强J-lens激活概念组成的集合，被视为模型的"全局工作空间"，反映当前正在处理的概念集。
- **In-category J-space supply**：J-space中属于当前活跃类别且尚未被产出的项目数量，作为"语义斑块"资源耗竭程度的代理指标。
- **Steering vectors**：从模型激活空间中提取的方向向量，通过向残差流添加该向量可因果性地引导模型行为倾向。
- **Explore-exploit tradeoff**：在搜寻决策中平衡利用已知高价值资源（exploitation）与探索未知潜在资源（exploration）的基本困境。
- **Ramping activation**：神经或人工网络激活随时间渐进上升的模式，在人类SFT中与小脑/海马活动相关，本文在LLM中复现了类似现象。

## 可复现要素
- **数据集**：作者使用自身生成的SFT数据（动物命名），未明确提供公开数据集链接；Troyer类别体系为已建立标准。
- **代码/权重**：使用开源模型（Gemma-2-9B/2B、Llama-3.1-8B、Llama-3.2-3B、Qwen2.5-7B）；J-lens来自Neuronpedia（Lin, 2023）；代码使用Claude Code生成但作者声明手动验证。
- **关键超参**：temperature=0.9, top-p=0.95, max tokens=200, J-space size=25, steering coefficient s∈{0.25, 0.5}（Qwen用0.25，其他用0.5）, PCA降维至30维, L2正则化λ=0.1, 100次随机种子迭代, k=5 fold交叉验证。
