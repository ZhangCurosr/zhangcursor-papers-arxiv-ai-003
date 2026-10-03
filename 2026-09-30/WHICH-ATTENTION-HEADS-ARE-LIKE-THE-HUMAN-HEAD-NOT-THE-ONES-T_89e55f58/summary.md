---
title: "WHICH-ATTENTION-HEADS-ARE-LIKE-THE-HUMAN-HEAD-NOT-THE-ONES-T"
source: https://arxiv.org/pdf/2609.37991v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-03 14:14:31"
field: "神经可解释性 / 脑-AI对齐"
keywords: ["brain-AI alignment", "attention heads", "ablation study", "function vectors", "concept vectors", "attribution patching", "EEG FRP", "mechanistic interpretability"]
innovations: ["首次对LLM注意力头执行脑对齐选择后直接因果消融检验，证明脑对齐与计算因果性解离", "识别出脑对齐头中反复出现的novelty和repetition两类注意力profile及其差异化任务贡献", "建立brain/FV/CV三评分框架系统刻画注意力头，验证attribution patching作为FV筛选的高效代理"]
benchmarks: ["abstract sequence completion (AAABAAA→B task)", "8-extended-task battery (few-shot, prose, retrieval, coreference, etc.)", "Pythia-1.4B training trajectory"]
---

# 论文速读：WHICH-ATTENTION-HEADS-ARE-LIKE-THE-HUMAN-HEAD-NOT-THE-ONES-T

## 一句话总结
本文通过对抽象序列补全任务（如 AAABAAA → B）进行直接因果干预测试，发现 LLM 中与人类额叶 FRP 脑电对齐的注意力头，其因果重要性远低于通过归因修补（attribution patching）选出的功能向量头（FVs），表明**脑对齐与计算因果性存在显著解离**。

## 研究问题与动机
1. **核心问题**：当神经网络某单元能预测人脑在同一任务中的神经信号时，常被视为"两者执行相似计算"的证据；但这些被对齐的单元是否真正因果参与模型计算，从未被直接检验。
2. **现有研究的不足**：先前的验证是间接的——训练使模型偏离脑活动的模型会损害下游性能（Merlin & Toneva, 2026），或语言定位单元同时具有因果重要性和脑对齐性（AlKhamissi et al., 2025），但均未做"按脑对齐选头→删除→测性能"的直接因果测试。
3. **为何用 LLM 注意力头**：头级别的表征可与神经记录比较，且机制可解释性研究已识别出已知功能的头类别（induction heads、copy heads、FV heads），为解读脑对齐分数提供了先验参照。
4. **理论动机**：Opiełka et al. (2026) 在类比任务上区分了概念向量（CVs，跨格式不变的抽象表征）与功能向量（FVs，因果支持任务预测），本文将此框架迁移到序列补全任务，检验脑对齐更贴近哪一类。

## 核心贡献（创新点）
1. **首次对 LLM 注意力头执行"脑对齐选择→零消融→行为测量"的直接因果检验**，与以往间接推论形成本质区别。
2. **发现脑对齐与因果重要性的解离**：跨 17 个模型，FV  Ranked 消融造成的精度损失（峰值 −42.6 pp，移除 12.5% 头）远大于脑 Ranked 消融（峰值 −12.8 pp，移除 24.5% 头），且脑-修补相关系数在所有模型中均接近零（均值 ρ = .017）。
3. **识别出脑对齐头中反复出现的两类注意力配置**：novelty heads（关注序列中仅出现一次的特殊元素）和 repetition heads（关注重复出现的元素），两类在跨模型中稳定再现。
4. **揭示 novelty heads 与人类眼动的对应关系及其功能局限性**：novelty 注意力与人类注视分配呈正相关（均值 ρ = .373，全部 17 个模型一致），但消融 novelty 头平均比随机消融破坏性更小（峰值仅 +0.3 pp），表明其可能仅为架构训练的副产品（spandrel）。
5. **建立三评分框架系统刻画注意力头**：同时计算脑对齐分（brain）、因果贡献分（FV/attribution patching）和概念表征分（CV/RSA），三者在头级别上彼此弱相关，提供了刻画 LLM 头的综合维度。

## 方法详解
- **任务设计**：8 种抽象序列模式（如 AAABAAA→B、ABABCDCD→D 等），文本化版本输入模型，使用三个已解示例作为 few-shot prompt（无 chat template），以原始文本提示评估 exact next-token accuracy。
- **三种头评分**：
  - **Brain score**：对每个头计算其输出的 pattern-level RDM（基于 200 个英文 MC 试验的层头输出取均值后的余弦距离），与人类群体平均额叶 FRP RDM（8×8，28 个唯一比较值）做 Spearman ρ。
  - **Patching score (FV)**：在 200 个英文 open-ended 试验上，对 corrupted prompt（模式信息被破坏）计算正确目标 token 概率相对于各头输出的梯度，AIE = 梯度·(clean_mean − corrupted_activation) 的均值；使用 attribution patching（一阶近似）而非完整 activation patching，验证得 Pearson r = .983。
  - **Concept score (CV)**：在 1,200 个跨 6 种条件（3 种字母表×2 种回答格式）试验上，计算头输出的 RSA 与"同 pattern"设计矩阵的 Spearman ρ。
- **累积零消融（cumulative zero-ablation）**：按各评分降序逐批将头输出置零，与 5 条随机消融轨迹对比，主要指标为准确率变化（pp）。消融粒度覆盖 1 至全量头（最多 5,120 个）。
- **注意力模板匹配**：在 Llama-3.1-70B 中对 top-20 脑对齐头聚类得到 novelty 和 repetition 模板（均值注意力 profile），经位置归一化和中心化后，以 Pearson 相关作为 loading 分数，阈值 ≥ .50 定义为 carrier。
- **人类眼动对比**：将人类被试在七个序列位置上的注视时长汇总为 8×7 地图，与模型各头族平均注意力 profile 做逐单元格相关（均值中心化后计算 ρ）。
- **模型规模**：17 个模型，涵盖 Llama-3.1/3.2/3.3（3B–70B）、Qwen2.5（3B–72B）、Phi-4（14B）、DeepSeek-R1-Distill-Llama-70B，外加 Pythia-1.4B 的训练轨迹分析。

## 实验与结果
- **基线性能**：Llama-3.1-70B base 70.5%，instruct 77.5%；Phi-4（14B）82.0%；Qwen2.5-72B instruct 87.5%；规模并非唯一决定因素（Qwen2.5-32B base 74.5% > Qwen2.5-72B base 54.0%）。
- **脑-因果解离**：
  - 脑-修补相关：Llama-3.1-70B ρ = .03，跨 17 模型均值 ρ = .017（范围 [−.017, .077]），全部模型均接近零。
  - 脑-概念相关：均值 ρ = .10（范围 [−.20, .33]），Llama 全系为正（[.100, .325]），Qwen instruct 全系为负（[−.205, −.037]），Phi-4 ≈ 0。
- **消融曲线（Figure 3）**：
  - **FV-ranked**：峰值超额损失 −42.6 pp（移除 12.5% 头），125 个头即跌破 25% 随机水平，800 个头（15.63%）归零。
  - **CV-ranked**：峰值 −20.7 pp（19.5% 头）。
  - **Brain-ranked**：峰值 −12.8 pp（24.5% 头），15/17 模型在前三成头消融时比随机更有破坏性。
  - **Repetition-ranked**：峰值 −11.2 pp（15.5% 头），整体略大于随机破坏。
  - **Novelty-ranked**：30% 头消融时仅 +14 pp（即比随机破坏更小），全曲线未超过随机 0.3 pp。
- **重复头不影响概念表征**：消融 130 个 repetition 头使准确率下降 5.5 pp，但后续 concept 头的跨格式 RSA 不降反升（0.401→0.427），pattern 分类器保持 100% 准确率。
- **Novelty 头在多任务中均不重要**：8 项扩展任务（few-shot、prose、retrieval、coreference 等）中，novelty 消融在 7/8 任务上低于随机中位数。
- **与人类眼动对应**：Novelty 头与注视地图正相关（均值 ρ = .373，全部 17 模型为正）；Repetition 头负相关（均值 ρ = −.285，全部为负）。在 5/8 模式中（独特元素恰好是答案）对应更集中（novelty ρ = .47），其余 3 模式仅 .12。
- **训练轨迹（Pythia-1.4B）**：随训练进行 novelty 注意力 profile 逐渐成型，copy 强度和因果贡献递增，但始终低于 reference 阈值；最终 checkpoint 移除 32 个 novelty 头仅使准确率从 42.5% 降至 39.5%。

## 相关工作脉络
1. **Function Vector 框架（Todd et al., 2024）**：定义通过激活修补选择的功能向量头；本文沿用 attribution patching 作为 FV 的高效近似，并在序列补全任务上验证其有效性。
2. **概念向量 vs 功能向量区分（Opiełka et al., 2026）**：在言语类比任务上证明 CVs 跨格式泛化而 FVs 因果支持性能；本文将其扩展到抽象序列任务并引入脑对齐作为第三维度。
3. **归因修补（Attribution Patching, Vig et al., 2020）**：基于梯度的头贡献估计方法；本文验证其与完整 activation patching 高度一致（r = .983），可高效用于大规模头筛选。
4. **Induction/Copy Heads（Olsson et al., 2022）**：机制可解释性中已知的头类别；本文的 repetition/novelty profile 发现与之形成对照，但并未将二者归类为已知类别。
5. **脑-AI 对齐文献（Yamins et al., 2014; Schrimpf et al., 2018, 2021; Caucheteux & King, 2022; Goldstein et al., 2022）**：主张模型-大脑相似性意味着共享计算；本文通过因果干预挑战这一推断的逻辑跳跃。
6. **对齐推断批判（Guest & Martin, 2023; Bowers et al., 2023; Antonello & Huth, 2024）**：质疑从表征相似性到计算相同性的逻辑推导；本文提供直接的实证反证。
7. **脑对齐训练的间接因果证据（Merlin & Toneva, 2026；AlKhamissi et al., 2025）**：前者的"训练去对齐损害性能"和后者"语言定位头兼具因果性和脑对齐性"与本文结论不矛盾，因选择标准和干预方式不同；本文聚焦于任务特异性脑对齐的直接因果检验。

## 局限性与未来方向
1. **脑目标的单一性**：仅使用 8 种模式、固定三示例 prompt 和一个群体平均额叶 FRP RDM，"脑对齐"仅反映该特定目标的相似性，不等同于与人类脑活动的广义对应；其他电极组、时间窗口、fMRI 等其他模态可能识别不同组件。
2. **人类 RDM 的统计局限**：仅 28 个依赖比较值，参与者权重不均（数据量多的贡献更大），需更多模式和留出发参与者以提高稳健性。
3. **Novelty 头的功能未完全排除**：在测试任务和 prompt 策略下 novelty 头因果作用小，但不排除在其他计算中存在因果角色，需进一步探索。
4. **消融效应的重叠贡献**：小零消融效应可能源于其他头的冗余补偿，需层匹配控制以分离头选择与网络位置的影响。
5. **模型的泛化范围**：17 个模型虽跨家族和规模，但未覆盖所有架构；不同 prompting 策略（如 chat template）可能改变结果。
6. **未来方向**：扩展任务集和 prompt 策略、测试其他脑信号特征、探索 novelty 头的起源（是否为 softmax 分配的必然副产品）、构建层匹配对照。

## 研究启发与可借鉴点
1. **"选择→干预→测量"的直接因果检验范式**：可推广到其他可解释性单元（如 MLP 层、residual stream components）与其他神经数据的对齐检验，弥补现有文献仅依赖间接相关证据的不足。
2. **三评分框架（Brain/FV/CV）的综合头刻画策略**：同时计算表征相似性、因果贡献和概念不变性，为头功能分类提供多维坐标，可迁移至其他任务和研究场景。
3. **注意力模板匹配发现头家族**：从少数示例头聚类得到模板，再以相关系数在整个模型中扫描 carrier，是一种可复用的头族发现方法，适用于任何具有可可视化 attention profile 的任务。
4. **Attribution patching 作为 FV 筛选的高效代理**：验证其与完整 activation patching 高度一致（r = .983），可将计算成本降低数个数量级，适合大模型的大规模头排序。
5. **跨模型一致性检验协议**：以相等权重对多个模型取均值并报告区间，避免单一模型的偶然性；该协议可直接迁移到其他比较研究中。

## 关键术语表
**Brain alignment**：LLM 注意力头输出表征与人类脑电（FRP）信号在表示 dissimilarity 层面的 Spearman 相关性，衡量模型表征与人类神经表征的相似程度。

**Function vectors (FVs)**：通过归因修补（attribution patching）选出的注意力头集合，其对正确 token 预测概率具有最大平均间接效应（AIE），代表因果上最关键的计算组件。

**Concept vectors (CVs)**：通过在多格式（不同字母表、回答方式）下计算头输出的 RSA 与"同抽象 pattern"设计矩阵的相关性选出的头，代表跨格式不变的抽象模式表征。

**Attribution patching**：以梯度为一阶近似计算单头贡献的方法：AIE = ∇_a P(correct) · (a_clean − a_corrupted)，远快于完整 activation patching 且在此研究中高度一致（r = .983）。

**Fixation-related potentials (FRPs)**：与眼动固定 onset 锁相的 EEG 信号，反映被试对序列元素的自主视觉inspect过程，本文用于构建脑对齐的神经目标 RDM。

**Representational dissimilarity matrix (RDM)**：将各条件（pattern）下的头输出均值两两比较余弦距离，得到一个对称 dissimilarity 矩阵，用于与人类神经 RDM 做相关性比较。

**Novelty heads**：脑对齐头中一种反复出现的注意力 profile，优先关注序列中仅出现一次的特殊元素，与人类注视分布正相关但对任务表现因果贡献极小。

**Repetition heads**：脑对齐头中另一种反复出现的 profile，优先关注序列中重复出现的元素，与 CV 表征共定位且在层分布上重叠，删除时造成适度性能损失但不破坏后续的概念表征。

## 可复现要素
- **数据集**：去标识化的 EEG 和眼动数据来自 Pinier et al. (2025)（25名参与者，400次试验，64导联 BioSemi EEG + EyeLink 1000 Plus）；论文声明释放或共享去标识化数据及衍生表征。
- **代码/权重**：论文未明确声明代码开源状态；模型权重为公开可用的 Llama、Qwen、Phi-4、DeepSeek、Pythia 系列。
- **关键超参**：模板匹配载体阈值 ≥ .50；头消融按 1, 2, 3, 5, 8, 13, 20, 32, 50, 80, 125, 200, 320, 500, 800, 1250, 2000, 3200, 5120 递增；随机对照为 5 条 seed-0 均匀随机排列；归一化为中心化后的 query 内归一化（除以该头在七位置上的总注意力）。
