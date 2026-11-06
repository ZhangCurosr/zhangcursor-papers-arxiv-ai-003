---
title: "Where-Should-the-Evaluation-Budget-Go-in-LLM-Driven-Agent-Sc"
source: https://arxiv.org/pdf/2610.37276v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-06 08:59:15"
field: "LLM Agent自动化优化"
keywords: ["LLM agent scaffol evolution", "evaluation budget allocation", "sparse feedback", "successive halving", "response model", "item selection"]
innovations: ["将评估预算分配为搜索期稀疏反馈与最终选择两个阶段，针对不同需求分别优化", "提出共享信息核心+固定响应模型机制，在保证跨candidate可比性的同时降低评估成本", "设计热启动逐次减半策略，复用累积观测并结合回归校正实现低成本高可靠性最终选择"]
benchmarks: ["HotpotQA", "HoVer", "DROP", "IFBench"]
---

# 论文速读：Where-Should-the-Evaluation-Budget-Go-in-LLM-Driven-Agent-Sc

## 一句话总结
本文研究了LLM驱动Agent scaffol迭代中评估预算的分配策略，提出EvalAlloc框架，将稀疏搜索反馈与靶向最终确认相结合，在消耗约一半评估cell数的情况下，仍能在多个基准和演化框架上取得与全量评估相当或更优的性能。

## 研究问题与动机
1. **评估成本高昂**：单个task实例可能涉及多次LLM调用、检索步骤和工具交互，对所有candidate进行全面评估成本极高。
2. **评估承担双重角色**：评估既提供指导后续搜索的fitness反馈，又决定最终返回哪个scaffol，两者对评估资源的要求不同。
3. **现有方法不足**：已有稀疏估计和评估级联方法将评估视为单纯的fitness估计问题，忽视了scaffol演化中两个阶段的不同需求。
4. **核心问题**：如何在有限的评估预算下，合理分配到演化搜索和最终选择两个阶段以提升最终scaffol质量？

## 核心贡献（创新点）
1. **揭示评估预算分配的二元性**：系统研究了搜索期反馈与最终选择对评估资源的不同需求，证明稀疏反馈可维持有效演化，而额外靶向评估可提升最终选择质量。
2. **提出EvalAlloc框架**：结合共享信息核心（shared informative core）、固定搜索期响应模型和热启动逐次减半（warm-started successive halving），将评估预算分配到两个不同阶段。
3. **设计可复用的评估分配机制**：通过短全量评估建立共享信息核心并校准响应模型，搜索期保持item参数固定以维护可比性，最终选择时重拟合模型并复用累积观测。
4. **全面实验验证**：在四个基准（HotpotQA, HoVer, DROP, IFBench）和两个演化框架（ShinkaEvolve, AdaEvolve）上验证，EvalAlloc平均节省约一半评估成本的同时达到更具竞争力的TEST性能。

## 方法详解
EvalAlloc框架包含三个阶段：

**1. 核心校准（Core Calibration）**
- 对前W=10个valid candidate在全部N=300 SEARCH items上全量评估
- 估计item信息性：使用Beta后验p_i ~ Beta(a_i+1, n_i-a_i+1)，保守信息性估计I_i = 4(E[X_i] - z√Var(X_i))，其中X_i = p_i(1-p_i)，方差项偏好跨candidate响应变化的items，不确定性惩罚避免证据不足的items
- 校准响应模型：拟合低秩logistic响应模型q_ci = σ(μ + η_i + b_c + u_i^T v_c)，其中(η_i, u_i)为item参数，(b_c, v_c)为candidate参数，使用正则化分数交叉熵
- warm-up candidates的fitted参数同时作为后续candidates的Gaussian prior

**2. 共享核心演化（Shared-Core Evolution）**
- 每个新candidate在k=60个items（20% N）上评估
- 共享信息核心：从前W个candidate的信息性估计中选择top k_0=54个items形成固定核心S_0，所有后续candidate在同一核心上评估
- 自适应组件：剩余k-k_0=6个items通过Gumbel-to-top-k采样选取，logits为I_i/τ（τ=0.001）
- 可比性fitness反馈：使用固定warm-up模型推断candidate特有参数(b_c, v_c)，预测unobserved items响应，计算完成估计F̂(c) = (1/N)(Σ_{i∈S_c} Y_ci + Σ_{i∉S_c} q_ci)，保持跨候选的可比评分尺度

**3. 热启动选择（Warm-Started Selection）**
- 搜索结束后，基于所有累积观测进行逐次减半（SH），共R=4轮，cumulative prefixes分别为30, 60, 120, 300 items
- 复用搜索期观测：Let O_c为candidate c在搜索期的observed items，U_c^r = O_c ∪ P_r，仅执行missing evaluations
- 回归调整分数：第r轮起使用difference estimator F̃_r(c) = (1/|U_c^r|)Σ_{i∈U_c^r} Y_ci + λ_r((1/N)Σ_{i=1}^N q̃_ci - (1/|U_c^r|)Σ_{i∈U_c^r} q̃_ci)，λ_r从surviving candidates数据中估计
- Cross-fitted re-fit：每轮后用所有观测refit响应模型，采用two-fold cross-fitting避免overfitting

## 实验与结果
**数据集**：HotpotQA, HoVer, DROP, IFBench（各含SEARCH 300 / DEV 150 / TEST 300 items）

**演化框架**：ShinkaEvolve, AdaEvolve（后者用SkyDiscover实现）

**基线方法**：
- Skip Eval（无评估）
- Successive Halving（仅最终阶段SH）
- Random（随机子集评估）
- Cascade（级联评估）
- Full（全量评估）

**主要结果**（Table 3）：
| Framework | Method | Avg Score | Avg Cost (%) |
|-----------|--------|-----------|--------------|
| ShinkaEvolve | Full | 0.6973 | 84.4 |
| ShinkaEvolve | **EvalAlloc** | **0.7084** | **43.8** |
| AdaEvolve | Full | 0.7091 | 99.2 |
| AdaEvolve | **EvalAlloc** | **0.7226** | **46.4** |

- EvalAlloc在ShinkaEvolve上以43.8%成本超越Full的0.6973得分（达0.7084）
- EvalAlloc在AdaEvolve上以46.4%成本超越Full的0.7091得分（达0.7226）
- 相比Random和Cascade，EvalAlloc在更低/可比成本下取得更好平均性能

**消融实验**（Table 4）：
- 去除final selection（w/o selection）：Avg下降0.0173，Cost降至27.4%
- 去除response model（w/o response model）：Avg下降0.0119
- 去除informative items（w/o informative items）：Avg下降0.0291（最大退化）
- 证明informative item选择和final confirmation是主要贡献

**Token成本分析**（Table 5）：
- ShinkaEvolve：EvalAlloc使用52.13% cells但64.76% tokens
- AdaEvolve：EvalAlloc使用46.83% cells和45.47% tokens
- High-scoring candidates tend to consume more tokens per item（Spearman corr ≈ 0.25-0.36）

**替代模型配置**（Figure 7）：
- IFBench用gpt-oss-120b proposer，DROP用Llama-3.1-8B-Instruct executor，EvalAlloc均取得竞争性性能

## 相关工作脉络
1. **LLM驱动Agent/Program Evolution**：ShinkaEvolve、AdaEvolve、ADAS、AFlow等工作聚焦candidate generation和演化优化；本文研究其背后评估过程，关注有限评估资源如何支持演化决策和最终选择。
2. **Efficient LLM Evaluation**：TinyBenchmarks、SparseEval、CollabEval等方法利用精心挑选样本和统计估计近似full-benchmark性能；Growing Pains使用固定item-response calibration和anchor items维护跨时期可比性。本文不仅关注估计精度，还考察共享观测和稳定评分参考如何支持可靠演化决策。
3. **Evaluation Allocation and Candidate Selection**：Successive halving、Hyperband、irace等方法通过顺序比较逐步聚焦资源；Common random numbers方法通过共享条件降低pairwise比较方差。本文将这些思想扩展到两阶段分配：演化期稀疏可比反馈 + 最终期复用累积观测的可靠选择。
4. **Code Evolution Evaluation**：Simple baselines are competitive with code evolution工作强调可靠candidate评估的重要性。本文进一步区分搜索期与最终选择的不同需求。

## 局限性与未来方向
1. **Warm-up长度的权衡**：较长warm-up提供更多校准观测但减少后续candidate评估items，存在trade-off（Figure 4）。
2. **搜索期预算的边际效益递减**：搜索期budget从20%增至40%并未持续提升最终性能，说明稀疏反馈已足够（Figure 5）。
3. **模型依赖性**：不同proposer/executor配置下评估策略的相对表现有差异，表明benefits可能依赖于底层模型（Figure 7）。
4. **Token消耗不均**：高score candidates倾向消耗更多tokens per item，导致cell savings与token savings不完全成比例。
5. **未来方向**：可扩展至其他演化框架和更大规模任务，探索动态budget分配策略，以及将此框架应用于更广泛的LLM-driven optimization场景。

## 研究启发与可借鉴点
1. **评估预算二元分配视角**：将评估任务区分为搜索期反馈和最终选择两个角色，针对不同需求分配资源，这一思路可迁移至其他迭代优化场景（如超参搜索、prompt优化）。
2. **共享信息核心设计**：通过common random numbers思想，使用共享items降低pairwise比较方差（Var(Δ_cc') = (σ_c² + σ_c'² - 2σ_cc')/k），这一技术可应用于任何需比较相似candidate的场景。
3. **固定响应模型维护可比性**：搜索期固定item参数避免re-scor档案candidates，保证跨迭代可比评分尺度；这一技巧对需要长期维持评分一致性的演化系统有价值。
4. **热启动逐次减半**：复用累积观测而非丢弃，结合difference estimator校正item难度差异，有效降低final confirmation成本。
5. **离线replay分析**：通过固定candidate序列、仅改变observation policy的replay实验，分离评估策略影响，这一分析方法可用于系统化评估设计。

## 关键术语表
**Scaffol Evolution**：通过LLM迭代生成和评估candidate programs来自动优化agent工作流的方法。
**Evaluation Budget**：可用于评估candidate的总计算资源（如evaluation cells数量）。
**Shared Informative Core**：从前几个full-evaluated candidate中选出的top k_0个高信息性items，所有后续candidate在此核心上评估以保证可比性。
**Response Model**：基于多维项目反应理论(MIRT)拟合的低秩logistic模型，用于从sparse observations估计full-set fitness。
**Successive Halving (SH)**：一种资源分配策略，逐步淘汰表现较差的candidate，将更多评估资源集中于有希望的candidate。
**Archive Regret**：演化过程中，真实top-10 candidate与evaluator ranking的top-10 candidate之间的平均fitness差距。
**Close-pair Agreement**：两个candidate在full-SEARCH mean上相差≤0.03时，evaluator正确排序的比例。
**Difference Estimator**：利用回归模型预测与观测值之间的差异进行校正的估值方法，减少item set不同难度带来的偏差。

## 可复现要素
- **数据集**：HotpotQA, HoVer, DROP, IFBench（均为公开benchmark）
- **代码开源**：论文未明确声明代码开源
- **权重开源**：Qwen3.5-9B, DeepSeek-V4-Flash-0731, gpt-oss-120b, Llama-3.1-8B-Instruct（具体开源状态依各模型而定）
- **关键超参**：W=10（warm-up candidates数），k=60（搜索期items数），k_0=54（shared core size），τ=0.001（Gumbel temperature），R=4（SH rounds），d=2（latent dimension），ρ=1（regularization），z=2（uncertainty penalty）
- **实验设置**：80 proposals per run, 3 seeds per setting, Qwen3.5-9B executor (temp=0, thinking disabled, max 16384 tokens), vLLM serving on 8x NVIDIA H20
