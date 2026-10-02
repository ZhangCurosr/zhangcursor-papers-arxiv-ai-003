---
title: "Why-Deterministic-PRM-Guidance-Underperforms-in-Discrete-Dif"
source: https://arxiv.org/pdf/2609.35472v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 08:16:46"
---

# 论文速读：Why-Deterministic-PRM-Guidance-Underperforms-in-Discrete-Dif

## 一句话总结
本文通过构建匹配前向计算预算的诊断协议，系统证明在离散扩散语言模型（dLLM）推理中，确定性PRM引导（PRM Guided）在相同算力度下全面落后于独立采样+最终状态ORM重排（ORM Rerank）。研究揭示性能差距源于两大可分离机制：去噪早期掩码比过高导致PRM判别信号衰减，以及确定性top-1剪枝引发候选池多样性坍塌；同时指出训练于中间状态的PRM不适合作为终局决策器。

## 研究问题与动机
- dLLM在每个去噪步都会暴露部分解码的中间状态（snapshot），看似天然适合用过程奖励模型（PRM）在推理时分配计算以提升准确率，但AR领域PRM的成功无法直接平移至dLLM。
- dLLM的snapshot是分散的掩码token子集而非线性前缀，导致基于因果注意力与固定位置约定的PRM存在架构失配，现有工作多聚焦生成器/采样器改进，缺乏在匹配计算预算下与简单Baseline的系统对比。
- 核心疑问包括：PRM引导在相同forward-pass预算下能否战胜ORM Rerank？高掩码比下的信号是否仍具判别力？确定性剪枝是否破坏候选多样性？跨掩码训练的PRM能否胜任最终选择？mean pooling是否为合适的readout？

## 核心贡献（创新点）
- **提出匹配前向计算预算的诊断协议**：将去噪与scorer调用统一计入单次dLLM scale forward pass，首次在同一算力约束下公平比较PRM引导与ORM重排的准确率与候选质量。
- **系统量化PRM引导的性能缺口并归因分解**：在Dream-7B上证明ORM Rerank@8领先PRM Guided@8达9.95 pp，K=32时扩大至12.69 pp，并将差距拆解为“候选池损伤”（信号衰减+多样性坍塌）与“终局选择缺陷”。
- **揭示跨掩码PRM作为最终裁判的不可靠性**：证明 pooled ROC-AUC 高并不保证 within-problem ranking，跨掩码PRM在N=8时重排表现近似随机，而同一架构重新训练于最终状态后可匹敌ORM。
- **明确因果PRM的readout失配问题并提供修复路径**：证明mean pooling导致因果PRM终态ROC-AUC仅0.61，替换为last-token pooling可修复约70%差距并恢复随N单调上升的重排趋势。
- **开源快照语料与评估工具包**：发布覆盖多掩码比与最终状态的dLLM去噪轨迹语料、PRM/ORM适配器及匹配计算预算的评测脚本，为后续reward-guided dLLM研究提供可比基准。

## 方法详解
- **匹配计算预算协议**：以一次dLLM backbone前向传播为统一计量单位。Vanilla需T=128次；Majority/ORM Rerank@N需128N+N次；PRM Guided分段top-1剪枝需$C_{PRM}=KT + K\lceil T/b\rceil$次（T=128为总去噪步，b为分支间隔）。 headline设置b=64，K=8时约1,040 passes，与ORM Rerank@8（1,032）误差<0.8%。
- **PRM架构与训练**：基于冻结的dLLM骨干（Dream-v0-Instruct-7B / LLaDA-8B-Base），附加LoRA适配器（r=16, α=32作用于q_proj与v_proj）与两层MLP奖励头，输入嵌入256维正弦步长编码。对比全注意力双向PRM与L→R因果PRM。训练数据为GSM8K训练集上的on-policy中间状态，标签为最终答案正确性（binary BCE loss，2,000 steps，batch=32，cosine LR从2e-5衰减至0）。
- **PRM Guided算法（Algorithm 1）**：将T步去噪划分为$\lceil T/b\rceil$个segment，每segment将当前状态复制为K份并行去噪b步，用PRM对所有K个状态打分，仅保留得分最高者进入下一段，最终输出单一轨迹。
- **对照变体**：PRM Hybrid保留全部K个candidate用于测量Oracle上限；Top-M放宽剪枝策略（每段保留前M名）；ESS-tempered SMC用PRM分数对粒子加权重采样以恢复池多样性；Final-state PRM仅在mask=0状态下重新训练以隔离跨掩码训练偏差。
- **评估协议**：采用严格regex答案提取（兼容####、answer is:、\boxed{}及fallback），避免lm-eval最后数字匹配的膨胀；报告Accuracy、Oracle@K、Unique answers per problem、PRM ROC-AUC across mask buckets，并使用paired bootstrap计算95% CI。

## 实验与结果
- **数据集与模型**：主实验GSM8K（1,319测试题），控制实验MATH500与MBPP；主模型Dream-v0-Instruct-7B，交叉验证LLaDA-8B-Base。
- **核心结果（GSM8K）**：ORM Rerank@8达75.13%，PRM Guided@8仅65.18%（差距9.95 pp）；ORM Rerank@32达82.71%，PRM Guided@32达70.02%（差距12.69 pp）。ORM Rerank@8超越所有PRM Guided预算，且加权多数投票（Weighted-Majority）与argmax结果相差<0.4 pp，说明优势不依赖选取规则。
- **机制分解**：
  - **信号衰减**：双向PRM ROC-AUC从低掩码比（[0.0,0.1)）的0.77单调下降至高掩码比的0.54；即使改用fresh rollout重新打标，衰减趋势与精度损失几乎不变。
  - **多样性坍塌**：Top-1剪枝使每问题唯一答案数从4.31降至1.75，Oracle@8从81.05%跌至67.30%（损失13.75 pp）；早期cut移除全部正确谱系的概率高达46.06%。
  - **终局选择缺陷**：SMC恢复池多样性后Oracle@8升至77.89%，但PRM选出的准确率仍仅65.48%，与top-1相当，远低于ORM的75.13%；跨掩码PRM在N=8时重排准确率42.84%（近随机），N=32时提升至65.35%，仍落后ORM 17.4 pp。
  - **Readout失配**：因果PRM mean pooling终态ROC-AUC仅0.61，换last-token pooling升至0.73，修复约70%差距；mean pooling变体在N≥8时出现非单调退化（N=32跌至40.56%），last-token变体则稳定上升至50.64%。
- **跨任务/跨模型泛化**：MATH上ORM Rerank@8达30.65%，PRM Guided仅20.80%（差9.85 pp）；MBPP上ORM Rerank达63.04%，PRM Guided达50.88%（差12.16 pp），但PRM Rerank（65.47%）与ORM持平，表明MBPP的损失完全来自guidance本身而非终局评分；LLaDA上双向PRM引导（31.64%）显著优于因果PRM（22.25%），验证架构优势。
- **最强结果与提升**：ORM Rerank@32以82.71%成为最强方法，较PRM Guided@32提升12.69 pp；即使搭配完美选择器，PRM Guided候选池天花板仅为67.30%，仍低于独立采样+ORM组合。

## 相关工作脉络
- **AR Reward Models & Verifier Reranking**（Lightman 2024; Wang 2024等）：基于线性前缀顺序设计PRM/ORM，不适配dLLM分散掩码状态；本文证明将同类思想直接移植到dLLM会因信号衰减与剪枝策略失效。
- **dLLM Reasoning & Decoding**（Dream, LLaDA, Block Diffusion等）：聚焦生成质量与解码速度优化；本文定位为其测试时计算扩展的公平对照，指出未考虑reward guidance时的性能天花板被低估。
- **Test-Time Scaling for dLLMs**（SMC/Particle samplers, RFG, Remasking等）：提出无PRM或隐式引导的采样策略；本文与之对比，证明显式中间PRM引导在匹配计算下仍不及简单ORM rerank，为后续工作划定更强baseline。
- **Tree Search & Tool-Augmented Reasoning**（ToT, DeepAgent等）：依赖AR前缀树结构；本文的匹配计算协议与pool-damage诊断可迁移至评估任意pruning-based搜索在dLLM上的适用性。
- **定位差异**：本文不提出新采样器或训练目标，而是通过控制
