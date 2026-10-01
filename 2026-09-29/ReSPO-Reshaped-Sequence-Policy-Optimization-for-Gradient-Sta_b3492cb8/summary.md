---
title: "ReSPO-Reshaped-Sequence-Policy-Optimization-for-Gradient-Sta"
source: https://arxiv.org/pdf/2609.35433v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 03:20:13"
field: "大语言模型强化学习"
keywords: ["RLVR", "policy optimization", "off-policy learning", "gradient starvation", "alpha-divergence", "sequence-level importance weight"]
innovations: ["从α-divergence variational objective导出sign-dependent two-branch reshaping kernel替代hard clipping", "通过KL projection实现exponential variance-control tilt避免hard boundary", "在dense和MoE Qwen3模型上验证off-policy drift下的优化稳定性"]
benchmarks: ["DAPO-MATH-17k", "AIME 2025", "AIME 2024", "AMC 2023", "OlympiadBench", "MinervaMath", "MATH-500"]
---

# 论文速读：ReSPO - Reshaped Sequence Policy Optimization for Gradient Starvation in Off-Policy Learning

## 一句话总结
本文识别了RLVR中clipped policy optimization存在的sign-dependent gradient starvation问题，并提出ReSPO方法，通过从α-divergence variational objective导出的smooth two-branch kernel替代hard clipping，有效改善off-policy学习中的梯度分配不均问题，在Qwen3模型上实现更快的早期优化和更高的最终性能。

## 研究问题与动机
- **Rollout重用的off-policy drift问题**：RLVR为节省计算成本重复使用rollout，导致当前策略π_θ与数据生成策略π_old产生偏差，序列重要性权重W随训练步骤累积漂移
- **Clipping导致的梯度饥饿现象**：GRPO和GSPO使用hard clipping稳定训练，但存在符号依赖的梯度分配错误——正向响应中低重要性权重尾部的under-generated样本获得几乎为零的梯度权重，而负向响应中高重要性权重尾部的severely over-generated样本权重无界增长
- **现有方法无法同时满足两分支需求**：VESPO虽抑制双端尾部，但在W→0时对正分支产生梯度饥饿；clipping方法在正分支低权重新近于零、负分支高权重无界增长，无法满足四种边界条件
- **Sequence-level与token-level的不匹配**：奖励定义在序列层面，但token-level重要性修正在高序列长度时引入噪声scaling，且GSPO的length-normalized geometric-mean ratio会引入length-dependent bias

## 核心贡献（创新点）
- **提出ReSPO方法，用smooth two-branch kernel替代hard clipping**：通过α-divergence variational objective和exponential variance-control tilt导出 reshaping kernel φ(W)，从根本上解决gradient starvation问题
- **与已有工作的本质区别**：不同于VESPO的单分支KL divergence设计，ReSPO使用α-divergence family并在正/负分支使用不同α参数（α^(+) > 1, α^(-) = 1），使正分支在W→0时保持非零梯度权重、负分支在W→0和W→∞时均趋近于零
- **理论推导满足所有四种边界条件**：通过power-mean interpolation得到unconstrained kernel φ_0，再通过KL projection施加exponential tilt实现variance control，解析证明正分支在W=0处达到全局最大值φ^(+)(0) = e/2 ≈ 1.36
- **实验验证在dense和MoE模型上的优越性**：在Qwen3-1.7B-Base和Qwen3-30B-A3B-Base上，ReSPO在五个六设置中获得最高early peak和late mean训练分数，并在N=16和N=32时获得最高benchmark evaluation平均

## 方法详解
- **问题建模**：定义positive response（ÊÂ > 0）和negative response（ÊÂ < 0），分析importance ratio W = π_θ(o|q)/π_old(o|q)的四种尾部情况，确定所需kernel应满足：φ^(+)(0) > 0（保留under-generated正响应），φ^(+)(∞) = φ^(-)(0) = φ^(-)(∞) = 0（抑制已强化/严重over-generated的响应）
- **Measure change视角**：将raw importance weight W替换为φ(W)，定义unnormalized measure τ(o) = μ(o)φ(W(o))，通过measure-change identity关联策略梯度，设计策略为求解最优τ再反推φ
- **α-divergence variational objective**：使用Bregman generator f(x) = x^α/(α(α-1))求解dual-proximity objective min_τ (1-β)D_f(τ‖μ) + βD_f(τ‖π)，得到power-mean interpolation的unconstrained kernel φ_0(W) = [(1-β) + βW^(α-1)]^(1/(α-1))，满足φ_0(1) = 1
- **Exponential tilt for variance control**：在KL geometry下进行投影，约束二阶矩和总质量，通过KKT条件得到τ*(o) ∝ τ_0*(o)·exp(-λφ_0(W(o)))，提取出最终的reshaping kernel φ(W) = φ_0(W)·exp(λ(1-φ_0(W)))
- **Two-branch design**：正分支使用α^(+) = 2（使φ_0^(+)为算术平均(1+W)/2）、β^(+) = 0.5、λ^(+) = 2，得到φ^(+)(W) = (1+W)/2 · exp(1-W)，在W=0处达到最大值e/2；负分支使用α^(-) → 1极限（几何平均√W）、β^(-) = 0.5、λ^(-) = 2（通过匹配φ^(+)'(1) = -1/2确定），得到φ^(-)(W) = √W · exp(2(1-√W))，在W→0和W→∞时均趋近于零
- **梯度估计**：使用stop-gradient操作符sg(·)使φ(W_i; ÊÂ_i)作为per-sequence scaling coefficient，最终策略梯度为REINFORCE-style估计量：∇_θ I_ReSPO = E[1/G Σ_i φ(W_i; ÊÂ_i) · ÊÂ_i · Σ_t ∇_θ log π_θ(o_{i,t}|q, o_{i,<t})]

## 实验与结果
- **数据集与模型**：训练数据为DAPO-MATH-17k，使用Qwen3-1.7B-Base（dense）和Qwen3-30B-A3B-Base（MoE）两个模型；评估基准包括AIME 2025、AIME 2024、AMC 2023、OlympiadBench、MinervaMath、MATH-500
- **基线方法**：GRPO（token-level clipping）、GSPO（sequence-level clipping with length normalization）、VESPO（KL-based kernel），所有基线使用相同训练配置和数据
- **关键超参数**：rollout reuse ratio N ∈ {8, 16, 32}，每组G=8 rollouts，mini-batch size M=32，学习率10^(-6)，共1024次策略更新；30B模型添加soft length penalty（超过4096 token后线性递减至-1）
- **训练性能**：在1.7B模型N=32设置下，ReSPO late mean达0.276（±0.004），比最高clipped baseline（VESPO 0.230）提升4.6pp；在30B模型N=32设置下，ReSPO达0.588（±0.007），比最高baseline（VESPO 0.523）提升6.5pp
- **评估性能**：在N=16和N=32时，ReSPO在所有四个model-N设置中获得最高macro average；在困难竞赛基准AIME25+AIME24+AMC23上，30B模型N=32时比最高baseline提升3.4-6.3pp
- **跨响应长度分析**：在1.7B模型N=32时，ReSPO在30个response-length bin中24个获得最高accuracy；在短-中长度响应（Q1-Q7）中领先幅度达8.4-24.4pp
- **Positive branch诊断**：在N=32的早期训练（前512步）中，ReSPO在log W < -1区域的positive responses占比22.8%但分配38.5%的coefficient mass（1.69×放大），而VESPO仅分配15.5%（0.93×抑制）；ReSPO的mean positive response长度增长更快（+43% vs VESPO的+10%）
- **鲁棒性与组合**：三seed平均显示ReSPO在1.7B和30B上分别比最佳baseline提升4.07pp和5.62pp；与Routing Replay（R3）正交，在30B模型N=16时R3额外提升late score 1.84pp

## 相关工作脉络
- **GRPO与token-level clipping**：Shao et al. (2024)提出的GRPO使用token-level clipped surrogate，但reward定义在序列层面，token-level修正在高序列长度时引入噪声scaling，本文聚焦sequence-level gradient starvation而非token-level
- **GSPO与length normalization bias**：Zheng et al. (2025)的GSPO使用length-normalized geometric-mean ratio s = W^(1/|o|)稳定MoE训练，但Shen et al. (2026)指出该normalization引入length-dependent bias，使目标policy exponent随长度变化，本文直接使用unnormalized sequence weight W避免此问题
- **VESPO与单分支设计**：Shen et al. (2026)的VESPO从KL divergence导出φ_KL(W) = W^β exp(λ(1-W))，虽然抑制双端尾部，但在W→0时正分支也趋于零导致gradient starvation，本文通过α-divergence family扩展并引入sign-dependent two-branch设计解决此缺陷
- **REAL与gradient starvation的另一种视角**：Zhai et al. (2026)将gradient starvation重构为classification问题，本文保留importance sampling框架以显式处理off-policy staleness，两种方法各有侧重
- **Trust region与divergence measures**：TRPO和PPO通过KL bound或hard clipping实现trust region，ReSPO使用separable power-Bregman geometry从variational objective导出soft kernel，提供平滑的梯度分配而非hard boundary

## 局限性与未来方向
- **Compute budget限制**：30B模型实验使用8192-token response limit和soft length penalty，无法在更长context和更多策略更新下进行完整研究；延长训练可能需要8×H200 GPUs数周时间
- **Hyperparameter未充分搜索**：当前α、β、λ参数主要通过解析tail要求和W=1处derivative matching确定，未进行 exhaustive search；不同模型scale、reuse ratio、response-length regime可能存在更优参数组合
- **仅针对数学推理验证**：实验集中在math reasoning任务（DAPO-MATH、AIME、AMC等），在其他RLVR场景（如代码生成、指令遵循）的有效性待验证
- **单seed实验有限**：除N=32外其他设置未运行多个seed，跨设置的鲁棒性评估可能不充分

## 研究启发与可借鉴点
- **Two-branch kernel设计模式**：根据advantage sign分别设计kernel tail behavior的正/负分支思路，可迁移到其他off-policy learning场景；α-divergence family提供连续可调的shape参数空间，值得在其他policy optimization方法中探索
- **KL projection实现variance control的技巧**：将hard moment constraint通过generalized KL divergence投影转化为exponential tilt，避免hard boundary的同时保持smoothness，这一"变分目标+KL投影"的两阶段推导策略可用于其他kernel设计
- **Positive-branch diagnostic方法**：通过分析log W bins中的coefficient mass allocation和response length dynamics来量化early learning behavior，这种诊断方法可直接复用于其他off-policy RL方法的消融分析
- **与Routing Replay的正交组合**：ReSPO控制importance weight allocation，Routing Replay对齐training/inference router decisions，两者正交且可组合使用；类似的"策略优化+系统优化"组合思路可扩展到其他LLM RL框架
- **从on-policy limit验证一致性**：在W=1时两分支kernel均退化为unit weight，恢复标准on-policy更新，这一一致性验证可作为新kernel设计的必要检查项

## 关键术语表
**ReSPO (Reshaped Sequence Policy Optimization)**：本文提出的方法，通过sign-dependent two-branch kernel替代hard clipping，解决off-policy learning中的gradient starvation问题
**Gradient Starvation**：clipped policy optimization中的sign-dependent梯度分配错误，指under-generated positive responses在低权重尾部被过度抑制，而severely over-generated negative responses在高权重尾部权重无界增长
**Sequence-level importance weight (W)**：序列级别的策略比值W = π_θ(o|q)/π_old(o|q)，聚合token-level比值以匹配序列级reward的granularity
**α-divergence**：参数α控制的Bregman divergence族，本文用于推导power-mean interpolation的unconstrained reshaping kernel
**Exponential tilt**：通过KL projection施加variance control得到的multiplicative factor exp(λ(1-φ_0(W)))，将unconstrained kernel投影到满足moment constraint的集合
**Two-branch kernel**：正分支(α^(+) = 2)和负分支(α^(-) = 1)使用不同参数，分别满足φ^(+)(0) > 0和φ^(-)(0) = φ^(-)(∞) = 0的边界条件
**Rollout reuse ratio (N)**：同一组rollout被用于N个mini-batch更新的比例，N越大表示off-policy drift越严重，是衡量方法稳定性的关键超参数
**Routing Replay (R3)**：Ma et al. (2025)提出的MoE RL稳定技术，在优化时复用rollout-time expert routing decisions，与ReSPO正交可组合

## 可复现要素
- **数据集**：DAPO-MATH-17k（公开），AIME 2025、AIME 2024、AMC 2023、OlympiadBench、MinervaMath、MATH-500评估集
- **代码**：论文声明有accompanying code artifact包含ReSPO实现和launch configurations，训练配置在Tab. 2和Tab. 3中完整记录
- **权重**：使用Qwen3-1.7B-Base和Qwen3-30B-A3B-Base开源base模型
- **关键超参数**：ε=0.2（GRPO clip ratio），β^(±) = 0.5，λ^(+) = λ^(-) = 2，α^(+) = 2，α^(-) = 1；学习率10^(-6)，batch size 32，G=8 rollouts，1024次策略更新
- **评估协议**：temperature=1.0，top-p=0.7，max response length=16384，AIME/AMC每问题16 samples，其他benchmark每问题4 samples，使用Math-Verify evaluator
