---
title: "SIGNATURES-OF-SEMANTIC-SEARCH-IN-THE-ACTIVA-TIONS-OF-LARGE-L"
source: https://arxiv.org/pdf/2609.35599v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 10:10:56"
field: "大语言模型可解释性与认知建模交叉"
keywords: ["语义流畅性任务", "Jacobian透镜", "机制可解释性", "探索-利用平衡", "steering vectors", "语义搜寻", "J-space", "大型语言模型"]
innovations: ["首次用J-lens量化LLM语义搜寻中item/category级激活的动态变化并证明其预测switch行为", "通过PCA+logistic回归从residual stream中提取通用switch方向并实现双向因果偏置", "揭示抽象category标签激活的竞争-胜出模式及J-space耗尽驱动的foraging-like决策机制"]
benchmarks: ["Semantic Fluency Task (SFT)", "Gemma-2-9B/2B", "Llama-3.1-8B", "Llama-3.2-3B", "Qwen2.5-7B"]
---

# 论文速读：SIGNATURES OF SEMANTIC SEARCH IN THE ACTIVATIONS OF LARGE LANGUAGE MODELS

## 一句话总结
本研究将语义搜寻（semantic foraging）框架拓展至大型语言模型，通过Jacobian透镜（J-lens）与导向向量（steering vectors）等机制可解释性技术，揭示了LLM在语义流畅性任务中区分"聚类利用"与"切换探索"的表征签名，并证明可通过操控内部激活方向因果性地引导切换行为。

## 研究问题与动机
1. 人类在语义流畅性任务（SFT）中表现出"聚类-切换"交替的搜寻模式，已有神经行为学证据支持这是由探索-利用平衡驱动的；但LLM是否也维持类似的表征动力学尚不清楚。
2. 现有研究仅发现LLM输出与人类SFT表面相似（同样出现聚类与切换），但未探究其内部状态是否承载对应的语义探索/利用签名。
3. 机制可解释性技术（J-lens、steering vectors）已在语言结构追踪上取得进展，但尚未被应用于研究LLM执行语义搜寻时的动态表征过程。
4. 理解LLM内部探索-利用动力学对于调控其信息检索、工具选择、代码导航等搜索密集型任务具有潜在实用价值。

## 核心贡献（创新点）
1. **首次用J-lens量化SFT任务中item-level与category-level激活**：证明switch事件前下一token概率显著更低，且switch参与项的J-lens激活低于cluster参与项，首次在LLM内部状态中捕捉到"探索不确定性"的表征。
2. **揭示J-space耗尽预测切换决策**：发现当当前类别在J-space中未使用的同类项逐渐耗尽时，模型向新类别切换的概率显著上升，且该效应强于简单池化耗尽，匹配patch foraging的理论预测。
3. **发现抽象category标签激活的"竞争-胜出"模式**：目标category相关标签在切换前逐渐攀升并在切换时刻激增，而非目标标签同时存在弱上升后被抑制，暗示category选择由竞争机制驱动。
4. **构建item-level与category-level两种因果导向向量**：分别从具体item残差流均值与抽象category标签残差流提取导向向量，成功将模型定向"诱导"切换至目标类别（Gemma-9B item-level steer最高达0.68）。
5. **识别generic switching方向并实现双向偏置**：通过PCA+logistic回归从residual stream中提取通用切换方向，正向steering使Gemma-9B切换率从0.29升至0.70，负向steering降至0.07，证明LLM内部存在表征"探索vs利用"的通用方向。

## 方法详解
**实验模型与设置**：使用5个开源指令微调LLM（Gemma-2-9B/2B、Llama-3.1-8B、Llama-3.2-3B、Qwen2.5-7B），在SFT提示"Name as many different animals as you can, one after another, separated by commas. Just the list."下生成最多200 token输出，使用nucleus sampling（temperature=0.9, top-p=0.95），每个模型100个随机种子，残差流层从第2层起每隔1层采样。

**Switch/Cluster界定**：采用扩展Troyer分类体系（含water、reptiles/amphibians、pets等类别），当连续两项无共享category label时定义为switch事件。

**J-lens激活提取**：利用Neuronpedia预计算的Jacobians $J[L]$将中间层残差流映射到token级激活，J-space定义为top-25最强J-lens激活项的集合；in-category J-space supply指J-space内属于当前类别且尚未产出的item数量。

**Category label激活分析**：选取若干高层抽象标签（如"water"、"domestic"、"pets"等），比较其在target vs unrelated category下的激活差异及切换前后lag-dependent动态。

**Steering vector推导与注入**：
- Item-level：$d_{T,L} = \text{normalize}\left(\frac{1}{|T|}\sum_{w\in T} v_{w,L}\right)$，其中$v_{w,L} = \text{normalize}(J[L]^\top W_U[w]^\top)$。
- Category-level：$d_{c,L} = \text{normalize}(J[L]^\top W_U[c]^\top)$。
- 注入规则：$h_{L,t} \leftarrow h_{L,t} + s \cdot r_L \cdot d_L$，其中$s$为steering系数（item/category steer取0.5，Qwen取0.25），$r_L$为参考生成在该层的平均残差范数。

**Generic switching direction提取（Study 2）**：对anticipatory位置（switch前1 token）的残差流做PCA（降至30维），用L2正则logistic回归学习switch vs cluster分类器，系数$\beta$经PCA载荷$W$投影回原空间得方向$d_L = W\beta$，归一化后用于steering。对照组使用Gram-Schmidt正交化去除沿$d_L$分量的随机噪声方向，验证结果。

## 实验与结果
**数据集与任务**：动物名称SFT（扩展Troyer分类），每模型100次生成，共5个模型。

**Study 1主要结果**：
- 图2A：switch参与项的next-token概率在所有模型中均显著低于cluster参与项（LR, $\bar{\chi^2} \geq 32.39, p < 0.0001$）。
- 图2C：switch项J-lens激活在layer 8以上均显著低于cluster项（$\chi^2 \geq 98.17, p < 0.0001$），且随层深增加效应增强。
- 图2D：in-category J-space supply下降与switch概率呈强负相关（所有模型middle layer blocks, $r \leq -0.31, p < 0.0001$），该效应显著强于整体pool耗尽效应（except Qwen, Table S4）。
- 图3：target category label在切换前逐渐升高，在切换时刻激增（$F \geq 848.35, p < 0.0001$），unrelated label在切换时刻骤降，支持竞争模型。
- 图4B/C：steering成功诱导category切换，Gemma-9B在layer 40 item-level steer使target切换概率达0.68，category-level steer在layer 32达0.45。

**Study 2主要结果**：
- 图5B/C：anticipatory位置AUC最高0.72（Gemma-9B layer 26），event位置AUC最高0.87（layer 16），证明激活可预测switch行为。
- 图5D：cluster内产出item越多，activation在anticipatory方向上的loading越强（ramping），与人类神经ramping现象对应（$\chi^2 \geq 20.44, p < 0.0001$）。
- 图5E：正向steering使Gemma-9B（layer 26）切换率从0.29升至0.70，Qwen-7B从0.40升至0.85，效果一致显著（Fisher exact test, $p < 0.0001$）。
- 图5F：负向steering在Gemma-9B layer 26将切换率降至0.07，但Llama-3.2-3B负向效果较弱。

## 相关工作脉络
1. **Hills et al. (2012)** 提出语义foraging框架，解释人类SFT中聚类-切换行为；本文将其拓展至LLM，填补了人工系统类比验证的空白。
2. **Lacosse et al. (2026b)** 发现LLM处理人类SFT产出时switch事件前概率降低；本文使用J-lens在更深层级揭示了item-level与category-level的激活动态。
3. **Gurnee et al. (2026)** 引入J-lens并定义J-space概念；本文是该方法首次在语义搜寻动态中的系统应用。
4. **Lundin et al. (2023)** 发现人类SFT中后部小脑与海马ramping激活；本文在LLM中发现equivalent的anticipatory direction ramping现象。
5. **Venhoff et al. (2025)** 证明通过steering可调节LLM的不确定性表达；本文扩展至更高层语义行为（category switching）。
6. **Pal et al. (2023)** Future Lens工作追踪后续token表征；本文进一步揭示这些表征与搜索策略（explore/exploit）的耦合关系。

## 局限性与未来方向
1. SFT任务高度受限，尚未验证所述动力学是否泛化至自然语言对话或叙事生成。
2. 部分模型（如Llama-3.2-8B）对generic steering响应较弱，层敏感性与系数敏感性机制未明。
3. 测试模型规模有限（2B-9B），前沿大模型（如百B/千亿参数级）是否呈现相同模式仍是开放问题。
4. 仅考察动物类别SFT，其他语义域（如职业、食物）及cross-task通用性待验证。
5. 未来可在 discourse/narrative生成中检验switch-associated方向的跨任务一致性。

## 研究启发与可借鉴点
1. **J-space depletion作为foraging代理指标**：可将J-space内in-category供给量作为监控LLM内部搜索策略的无侵入式指标，用于检测模型是否陷入过度exploit或频繁无谓切换。
2. **Generic switching direction的可迁移性**：Study 2中通过PCA+logistic回归提取direction的方法可直接迁移至其他探索-利用型任务（如retrieval-augmented generation、multi-step reasoning），调控搜索深度。
3. **竞争式category selection建模**：观察到目标与非目标label激活的"胜者通吃"模式，可启发设计multi-category attention或winner-take-all机制，改进多策略切换的可控性。
4. **双层级steering（item-level vs category-level）**：两类导向向量分别对应细粒度item选择与粗粒度category切换，为分层可控生成提供了可行范式。
5. **ramping现象的测量**：anticipatory direction loading随cluster长度递增的模式可作为质量监控信号，识别模型"卡死"或"过度跳跃"的异常状态。

## 关键术语表
**Semantic Fluency Task (SFT)**：要求受试者在限定时间内尽可能多地命名某类别概念（如动物）的经典认知心理学任务，常用于测量语义记忆搜索效率。
**Semantic Foraging**：将语义记忆检索类比为动物觅食行为，其中"clustering"对应exploit已知区域，"switching"对应explore新区域。
**Jacobian Lens (J-lens)**：一种机制可解释性技术，通过 Jacobian矩阵将中间层残差流激活线性映射到token级概率空间，揭示内部表征的语义内容。
**J-space**：某一层中间表示中最强J-lens激活的前25个概念集合，被视为类似"全局工作空间"的活跃表征池。
**Steering Vector**：通过残差流方向的线性叠加人为干预模型内部激活，以因果性地引导输出行为（如改变topic或策略）。
**Anticipatory Direction**：通过有监督学习（PCA+logistic回归）从residual stream中提取的、能预测即将发生switch行为的通用方向。
**In-category J-space Supply**：J-space中属于当前活跃类别且尚未被产出的item数量，作为"当前语义斑块富集度"的代理指标。
**Ramping Activation**：随时间/序列位置持续增强的激活模式，在人类神经活动中与积累决策相关，本文在LLM中发现equivalent的ramping现象。

## 可复现要素
- **数据集**：SFT由模型自行生成（无外部数据集），提示固定为"Name as many different animals as you can, one after another, separated by commas. Just the list."
- **代码**：论文AI声明中使用Claude Code辅助生成代码，但代码仓库未明确提及是否开源（论文未提及）
- **预计算J-lens**：来源于Neuronpedia平台（Lin, 2023），Gemma/Qwen/Llama模型均有公开版本
- **关键超参**：temperature=0.9, top-p=0.95, nucleus sampling; J-space大小=25; steering系数s=0.5（Qwen=0.25）; PCA降维至30维; L2正则强度=0.1; 100次重复迭代
- **模型权重**：全部使用开源权重（Gemma-2-9B/2B, Llama-3.1-8B, Llama-3.2-3B, Qwen2.5-7B）
