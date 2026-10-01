---
title: "Reverse-Sequential-Proportional-Approval-Voting-Rule-Proport"
source: https://arxiv.org/pdf/2609.35331v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 10:10:02"
field: "计算社会选择 / 多winner投票理论"
keywords: ["approval-based committee voting", "RevSeqPAV", "Extended Justified Representation", "proportionality degree", "approximation guarantee", "computational social choice"]
innovations: ["首次证明RevSeqPAV比例度与比例排名质量在最坏情况下为零", "确立r≤2时EJR严格成立且r≥3时失效的紧界", "给出k/(m-1)与1/b下两个紧的PAV最优分近似保证"]
benchmarks: ["理论复杂度与近似比分析（无实证基准）"]
---

# 论文速读：Reverse-Sequential-Proportional-Approval-Voting-Rule-Proportional

## 一句话总结
本文对反向序贯比例批准投票规则（RevSeqPAV）进行了系统理论分析，证明其在最坏情况下比例度为零、可能违反EJR且无法保证PAV最优分的常数近似比，但在需淘汰候选数较小（r≤2）或选民批准数有界（b）时能提供紧的公平性与优化保障。

## 研究问题与动机
- RevSeqPAV是Thiele早在1895年提出的两种PAV贪心近似之一，但相较于研究详尽的SeqPAV，其理论性质长期缺乏系统刻画。
- 现有认知呈混合态势：它满足委员会单调性与D’Hondt比例性，却连最基础的JR都可能失败；实践中被LiquidFeedback平台采用，但此前教科书未给出其比例度界，公平性上限未知。
- 本文旨在从两条正交维度厘清该规则的边界：一是比例代表性（EJR、比例度、比例排名质量），二是对PAV最大目标分的近似能力。

## 核心贡献（创新点）
1. **首次确定比例度为零**：证明对任意ℓ≥1，RevSeqPAV的比例度严格为0，填补了Lackner & Skowron教科书中的理论空白。
2. **EJR的紧界划分**：证明当需淘汰候选数r=m−k≤2时，RevSeqPAV严格满足EJR；同时构造反例表明r≥3时EJR必然可被违反，界是紧的。
3. **EJR的近似与群体保障**：给出(1/r)-EJR近似保证，并证明只要ℓ > k(1−1/r)，足够大的ℓ-凝聚组仍可获得完整EJR代表权。
4. **PAV近似比的负结果与两个正区间**：证明最坏情况下PAV近似比为0（即使k=1）；但在k接近m时达到k/(m−1)，在有界批准数b下达到1/b，且两界均紧。

## 方法详解
- **规则机制**：RevSeqPAV从全候选集C_m=C出发，迭代执行：在当前集合C_s中计算每位候选的边际PAV损失Δ_{C_s}(c)=PAV(C_s)−PAV(C_s\{c})，删除损失最小者，直至剩余k个候选构成W。
- **公平性分析技术**：通过反证法结合“边际损失下界”论证r≤2时EJR成立；构造含特定块结构与重复批准的election证明比例度为0；利用调和级数性质与组大小下界推导(1/r)-EJR。
- **近似比分析技术**：证明每次删除最多损失当前PAV分的1/s（因∑Δ≤PAV且候选数为s）；将最优委员会W*与输出W划分为交集I、差集D与R，建立双射配对并利用PAV的单调次模性，导出PAV(W*)≤b·PAV(W)。

## 实验与结果
本文为纯理论工作，未开展实证实验或基准测试。所有结果以定理形式给出：
- 定理2：r≥3时存在反例（k=12, m=15, n=600）使EJR失效。
- 定理5/6：比例度与比例排名质量在最坏情况下均为0。
- 定理7：PAV近似比下界为0（k=1时仍成立）。
- 定理8：PAV(W) ≥ [k/(m−1)]·OPT_k，当r固定且k→∞时趋近1。
- 定理9/10：若最大批准数≤b，则PAV(W) ≥ (1/b)·OPT_k，且该界对任意b≥2紧。
最强结果：在k≈m或b小的正则场景下，RevSeqPAV分别取得逼近最优的PAV分与完整的EJR保障。

## 相关工作脉络
- Thiele (1895) [21] 提出PAV及SeqPAV/RevSeqPAV两种贪心近似，本文延续其经典框架进行系统性性质补全。
- Aziz (2017) [1] 证明RevSeqPAV在r≤2时满足JR，本文将其强化为EJR并证明该条件紧。
- Lackner & Skowron (2023) [15] 教科书指出RevSeqPAV的比例度与PAV近似比均属未知，本文直接回答其中Q9与Q17。
- Skowron et al. (2017) [20] 引入比例排名（proportional rankings）概念，本文借其框架解答了RevSeqPAV在此度量下的最坏保证。
- Aziz et al. (2017) [2] 定义JR/EJR，Skowron (2021) [18] 定义比例度，本文将这些公理化指标拓展至反向淘汰规则。
- Lackner & Skowron (2020) [14] 与 Faliszewski et al. (2023) [10] 的实证研究表明RevSeqPAV与SeqPAV在实践中“几乎难以区分”，本文从理论上解释二者在不利实例下的根本差异。

## 局限性与未来方向
- 最坏情况保障极弱（比例度/近似比均为0），仅在小r或受限偏好宽度下有效，实际泛化能力依赖实例结构。
- 开放问题：有界批准数b是否能同步提升比例度或EJR保证；SeqPAV与RevSeqPAV行为重合的充分条件；两类规则在不同选举族与指标下的实证对比仍需补充。

## 研究启发与可借鉴点
- **“小删除数/小偏好宽度”正则化路径**：从负结果转向正结果的通用技术，适用于其他贪心淘汰型优化规则的分析。
- **边际损失下界+反证法**：将公平性违规转化为删除步的代价矛盾，结构清晰，可复用于其他基于评分的投票规则证明。
- **最优-输出集合配对技巧**（定理9）：利用次模性拆分D/R并利用双射控制损失，对设计近似比分析具有迁移价值。
- **投票规则↔排名质量转化**：将委员会输出等价视为排名前缀，可迁移至排序学习、推荐系统公平性等相邻领域。

## 关键术语表
- **RevSeqPAV**：反向序贯比例批准投票，从全候选集起迭代删除PAV边际损失最小者的多项式时间贪心规则。
- **PAV score**：基于调和数累加的委员会效用评分，最大化该分即得PAV最优委员会。
- **Extended Justified Representation (EJR)**：若存在ℓ-凝聚选民组，则组内至少一人应approve至少ℓ位当选者。
- **Proportionality degree**：衡量ℓ-凝聚组平均满意度下界的公平性指标，本文证明RevSeqPAV该值恒为0。
- **Marginal PAV loss Δ_S(c)**：从候选集S中删除c所导致的PAV总分下降量，等于被删候选每支持选民贡献的1/t之和。
- **Proportional rankings**：基于投票输出全序排列后，用前k名前缀的满意度与正当需求之比衡量比例代表性的框架。

## 可复现要素
- 数据集：无（纯理论分析，未使用公开数据集）。
- 代码/权重：论文未提及开源代码或实现。
- 关键超参：委员会规模k、候选人数m、需淘汰数r=m−k、选民最大批准数b。
- 数据来源：论文未提及。
