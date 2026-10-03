---
title: "UNLOCKING-THE-CRITIC-REWARD-FREE-POLICY-OPTIMIZATION-FOR-LLM"
source: https://arxiv.org/pdf/2609.37119v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-03 08:41:54"
field: "LLM 后训练与强化学习"
keywords: ["critic reuse", "reward-free RL", "policy optimization", "chain-of-thought reasoning", "length debiasing", "frozen value network"]
innovations: ["将单次监督训练的 critic 冻结并校准为训练期的唯一奖励/基线/前缀预言器", "单步内循环策略更新即可恢复 critic-based RL 的稳定性", "length-debias + 二值化同时消除连续奖励下的 length exploit 并保留稳定训练"]
benchmarks: ["AIME 2026", "AMC 2023", "GPQA-Diamond", "AIME 2025"]
---

# 论文速读：UNLOCKING THE CRITIC — REWARD-FREE POLICY OPTIMIZATION FOR LLM POST-TRAINING

## 一句话总结
本文提出 RFPO（Reward-Free Policy Optimization），将监督阶段训练并冻结的 critic 重新用作 RL 阶段的唯一奖励、GAE 基线以及对未完成 rollouts 的成功预测器，从而在不引入任何外部标签/验证器的条件下实现与有标签 PPO 相当的训练效果，并将 GPU 耗时缩短约 26%。

## 研究问题与动机
- **RL 后训练对 critic 的依赖与浪费**：PPO 的价值网络在训练结束后常被丢弃，尽管它已学会了预测轨迹结果。现有方法要么用外部 reward model/verifier，要么干脆去掉 critic；本文想回答：一个训练好的 critic 能否本身就成为奖励？
- **外部奖励成本高且不可扩展**：人类反馈、outcome/process reward models 各自需要独立的标注项目；可验证的 verifier reward 又要求每道题都有参考答案。两者的共同点是"为每次 RL 都买一份与训练量成正比的开销"，无法随想要跑的量线性扩展。
- **长链思考中稀疏终端奖励导致的信用分配困难**：推理过程中 verifier 只在最后才给信号（二元成败），早期中间步骤没有监督；这使长 horizon 的 credit assignment 尤为困难。
- **critic 被弃用的根本原因是否真的是价值网络本身**：业界将 critic 移除的原因包括"训练不稳定""内存昂贵""估计不可靠"，但本文认为其中至少部分来自优化配方（多次内循环更新）而非值网络本身，值得重新审视。

## 核心贡献（创新点）
- **Critic 不稳定性是优化 artifacts，不是值网络原罪**：将策略更新限制为每次 batch 单步梯度、低方差即可恢复稳定收敛；这与"critic 不可靠因而被抛弃"的既有判断形成本质区别。
- **单一冻结 critic 兼任三重角色（reward + GAE baseline + 未完成前缀的成功预言器）**：每个角色来自同一 forward pass 的输出，无需额外模型或额外 pass，与"critic 仅作为方差减少器"的传统用法形成本质区别。
- **对 critic 得分做 length-debias + binarization 可阻断策略利用 length bias 的通道**：连续分数能让策略在任一方向上通过"拉长/缩短答案"获利；二值化后超额分数不再带来额外收益，与仅加长度惩罚的做法不同——后者是事后约束，前者从奖励结构上关闭漏洞。
- **零标签 RFPO 在四个基准上与 100% 标签 PPO 持平，且每步节省 ~28% 时间、峰值显存降低 ~9.4 GB/GPU**：性能 parity 建立在"训练循环中没有任何 verifier 标签"的前提下，与 DPO/PRIME 等仍需训练期标签的方法形成对比。
- **可在未完成的 rollout 上提前打分，训练不必等所有轨迹结束**：在 4,096-token cap（~一半 batch 未完成）下，RFPO 仍达到与 5,120-token 完整 rollout 训练相当甚至更优的验证精度，与"必须等结局才能给奖励"的既有做法不同。

## 方法详解
- **目标函数与 critic 的学习信号**：在 γ = λ = 1 设定下，critic 的回归目标在每个 prefix 处都是轨迹的最终 reward，其不动点为 V_φ(x, y_{≤t}) ≈ Pr(success | x, y_{≤t})，即条件成功概率。
- **critic 来源**：从数学推理上的有监督 PPO（500 步 @ 8,192 token + 300 步 @ 5,120 token）中提取 actor（cumulative step 700）与 critic（step 800）两个 checkpoint，此后 critic 不再参与任何训练循环。
- **解释方差作为质量代理指标**：离线检查中，解释方差（explained variance）从初始化 −32 上升到约 0.55，且与 within-problem AUC 的相关系数达 r = 0.91；因此可用它在训练时廉价监控 critic 质量。
- **冻结与校准（核心公式）**：在 512 条 on-policy rollout 上拟合 length-decile 偏移 b(ℓ)（按正确/错误两类分别计算同长度内的得分偏移）和阈值 τ；部署的二值化奖励为：
  r̂(x, y) = 1[ v(x, y) − b(ℓ(y)) > τ ]
  其中 τ = 0.5215，在该校准集上与 verifier 一致性达 93.4%。
- **GAE 坍缩形式**：γ = λ = 1 且奖励为 r̂ 后，advantage 简化为 A_t = r̂ − V̄_φ(x, y_{≤t})（无需归一化即得训练信号）。
- **超参**：actor lr = 10⁻⁶，clip ε = 0.2，温度 1.0，每步 128 题 × 8 回复（= 1,024 rollouts），训练 cap 5,120 tokens（主实验）或 4,096 tokens（短 horizon 消融）；验证在 12,288 tokens 上进行以测量上限能力。
- **部分监督混合**：引入监督比例 p ∈ [0, 1]，以 Bernoulli(p) 独立替换每条 trajectory 的 r̂ 为真实 verifier 标签 r⋆；p = 0 为纯 reward-free，p = 1 退化为有监督 PPO。
- **实现细节**：基于 verl 框架、bf16、FSDP、梯度检查点；critic 冻结后删除其 optimizer states 与 backward pass，reward 本身不产生额外 forward（直接复用 value forward 的同一输出）。

## 实验与结果
- **数据集与模型**：Qwen3-4B-Base；SFT 使用 OpenR1-Math-220k 的子集（45k 长 CoT）；critic 与初始策略来自 DAPO-Math-17k 上的有监督 PPO。
- **评测基准**：AIME 2025/2026（各 30 题）、AMC 2023（40 题）、GPQA-Diamond（198 多选，做 out-of-domain probe）；pass@1/pass@8/maj@16，温度 0.7，12,288 token 验证预算。
- **主要结果（300 步，5,120-token cap）**：
  - RFPO 0% labels：macro-average pass@1 为 41.1%，对比 supervised PPO 100% labels 的 41.8%，差距 0.7 个百分点。
  - AIME 2026 pass@1 峰值：RFPO 0% labels 三重复现均值 25.7%，PPO 100% labels 为 25.0%。
  - GPQA-Diamond 上 RFPO 0% labels 高出 PPO 100% labels 0.7 个百分点。
  - AMC 2023 上 RFPO 略低于 PPO（76.6% vs 78.1% pass@1），作者解释为 40 题中单题权重 2.5 分所致。
- **关键消融与边界条件**：
  - 连续得分作奖励可超过全监督 PPO（AIME 2026 pass@1 峰值 29.2），但训练后期衰退，证明 critic 信息量大、但需二值化来稳住训练。
  - 截断 cap 越低收益越大：4,096-token cap 下 RFPO 均值 47.6%，超过同等 cap 的 PPO 46.6%，且与 5,120-token 完全 rollout 的 RFPO（47.2–47.4%）持平。
  - 1,024-token cap 失败：完成比例仅 0.7%，precision 坍缩至 0.213，验证从 49.1% 降至 39.1%，显示 critic 有" rollout 不再似拟合分布则失效"的下界。
- **效率提升**：每步时间由 582s 降至 421s（−28%），峰值显存降低 9.4 GB/GPU；300 步 GPU-hours 由 388 降至 264（−32%）。
- **训练动力学**：响应长度变化控制在 ±13% 以内，策略熵从 0.32 缓慢降至 0.21–0.22，未出现 collapsing 或 diverging；冻结 critic 对终端正确性的排序在 310 步内无明显退化（AUC 斜率在 −0.006 至 +0.002/100 steps 之间）。
- **部分标签防御**：p = 0.5 时，性能与两端（p = 0 / p = 1）均相当，证明监督比例可作为连续调节旋钮而非开关。

## 相关工作脉络
- **PPO 系列与 critic-free 潮流**：DeepSeek-R1、Kimi K1.5/K2、MiniMax M1 等推理/agent 系统已采用无 critic 方案；本文与它们的核心分歧在于"critic 应被保留并再利用"。
- **过程奖励模型（PRM）**：如 Math-Shepherd（Wang et al., 2024），提供每步密集监督，但需要步级人工/程序标注；critic 则仅从结果中学，并在前缀处给出前瞻性信号，二者信息来源与标注成本截然不同。
- **隐式偏好优化（DPO/PRIME）**：DPO（Rafailov et al., 2023）与 PRIME（Cui et al., 2025）训练期仍需标签；RFPO 在 RL 阶段彻底无标签，区别仅在 pretrain 阶段的一次性监督。
- ** Majority-voting / self-rewarding / confidence-based 替代奖励**：Zuo et al. (2025)、Yuan et al. (2024)、Zhao et al. (2026) 等用自投票、自奖励或模型置信度替代外部奖励，但缺乏 grounded supervision 时易受 reward hacking；本文的 critic 奖励来自 verifier outcome，具有可比的外部锚点。
- **已有恢复 critic 方向的工作**：EVPO（Pan et al., 2026）、Generative Critics（Shan et al., 2026）、ByteDance Seed 等仍将 critic 用于"降低梯度方差"的角色；本文进一步把它升级为"训练期唯一的奖励来源"。
- **处理长 rollout 截断与稀疏奖励**：Sparrow（Zhou et al., 2026）等通过截断 rollout 或稀疏 attention 缓解长 horizon 成本；本文用 frozen critic 的预测直接评分未完成的 rollout，无需改变生成预算结构。

## 局限性与未来方向
- **规模与任务范围受限**：所有实验仅在一个 4B 模型族、数学推理任务上进行；未扩展到更大模型或非推理任务。
- **单次准备成本可观**：获取与校准 critic 需要 800 步监督 PPO（约 901 GPU-hours 训练 + 38 GPU-hours 验证），对资源有限团队构成门槛。
- **连续分数的高天花板未能被稳定复现**：二值化换来稳定性，但丢失了 29.2 这类高于全监督 PPO 的上限；如何安全收回这部分增益需要 validated stopping rule。
- **长 horizon 下限的存在**：当 rollout 严重偏离 critic 拟合分布（如 1,024-token 极短 cap 下）时，预测精度会骤降，说明方法并非无限可扩展至任意短预算场景。
- **answer restatement 作为退化信号尚未被工程化**：Appendix F.3 发现连续奖励下的退化能被 answer restatement 提前检测（r = −0.87），但文中未给出自动停止机制，留给后续工作。
- **代码/权重开源声明**：论文附了训练日志、校准产物与 per-cell 评分摘要，但原始 rollout archive 与 frozen checkpoint 的大小和原始位置需通过 file index 查找，复现仍需额外准备。

## 研究启发与可借鉴点
- **critic 价值的二次挖掘思路具有通用性**：不仅在 RL 后训练可用，凡是有"一次性昂贵 pretrain 价值网络"的场景（如 agent planning、多步验证、多轮对话）都可尝试 freeze-and-repurpose，形成"训练期不花钱、复用期免费"的收益模式。
- **单步内循环 + 大 batch 的稳定性启示**：不仅适用于 RFPO，对其它 critic-based RL recipe（如 VinePPO、Vapo、EVPO）也有参考价值——当出现 collapse/diverge 时，首先排查 inner update 数与 batch 方差，而非直接放弃 critic。
- **length-debias + binarization 的防御范式**：对任何可能携带 length/approximation 相关偏差的自动化奖励，均可套用"按类内去偏 + 二值化封顶"的两段式校准；这种设计比单纯加 length penalty 更干净（从奖励结构上堵死 exploit）。
- **解释方差作为训练期 critic 质量的廉价代理**：r = 0.91 的 correlation 使得可以在不回灌 offline check 的前提下监控 quality，这对大规模 RL 实验的工程监控有普适意义。
- **prefix 打分与截断 budget 的联合使用**：在资源受限（GPU-hours）条件下，可以用更低 cap + 更高未完成率换取更短的 wall-clock；本文定量刻画了"多少未完成仍能保持 precision"的边界，可指导实验设计时的 budget 分配。
- **部分监督比例 p 作为连续旋钮**：既不是全监督也不是纯 reward-free，p ∈ [0, 1] 可按需加入少量 verifier 标签，以防御 drifted reward 带来的退化；这一思路可迁移到其它"弱监督/强监督可混合"的训练管线中。

## 关键术语表
- **RFPO（Reward-Free Policy Optimization）**：用单一冻结并校准的 critic 同时充当 rollout 级奖励、GAE 基线与未完成前缀的成功预言器，使 RL 阶段不再需要外部标签的方法。
- **GAE（Generalized Advantage Estimation）**：Schulman et al. 提出的多步优势估计，γ = λ = 1 时退化为单步 advantage = reward − value。
- **Length-debias（长度去偏）**：按响应长度分十分位、在同一正确性类别内分别拟合偏移 b(ℓ)，以消除 critic 得分中的长度相关性而保留跨类别差距的技术。
- **Within-problem AUC**：固定题目、比较同一题上正确与错误回复的排名质量，用于隔离"识别题目难度"与"判断尝试好坏"两种不同能力。
- **Explained variance（解释方差）**：1 − Var(y − v)/Var(y)，衡量 value head 对终端 reward 波动的解释比例，离线与 within-problem AUC 高度相关（r ≈ 0.91）。
- **Supervision fraction p**：以 Bernoulli(p) 比例把 verifier 真实标签 r⋆ 替换进 reward，p = 0 为全 reward-free，p = 1 为全监督 PPO。
- **Answer restatement**：响应中重申自身答案的次数；在连续奖励下作为退化信号，其上升能提前预示 held-out 准确率的下降（r = −0.87）。
- **Calibration threshold τ**：在校准集上把二值化奖励的正例率匹配 verifier 正例率的分位数，使部署规则在训练起始分布上与 verifier 最大可能一致。

## 可复现要素
- **数据集**：OpenR1-Math-220k（Hugging Face, 2025）的子集 45k 长 CoT；DAPO-Math-17k（Yu et al., 2025）。论文附 archive 含每 cell 评分摘要、benchmark 输出与训练日志；rollout archive 与 frozen checkpoint 位置见 file index。
- **代码/权重开源**：论文声明"every number is produced from artifacts included with the submission"；附有一键重绘所有 figure/table 的脚本。frozen critic 的 calibration artifacts 包含在 data/critic/baselines/；74 条稳定性 ablation 的汇总在 data/derived/stability/runs_summary.csv。
- **关键超参**：actor lr 10⁻⁶、critic lr 10⁻⁵、clip 0.2、γ = λ = 1、group size 8、128 prompts × 8 responses、train cap 5,120 / val cap 12,288、batch = mini-batch（单步）、温度 1.0（采样）/ 0.7（验证）、AdamW β = (0.9, 0.999)、weight decay 0.1、gradient clipping 1.0、bf16 + FSDP + sequence parallelism 2 + gradient checkpointing。
- **硬件**：8 × A100-80GB 单节点；每 300 步 RFPO 用 264 GPU-hours、PPO 100% labels 用 388 GPU-hours。
- **未提及**：具体 vLLM tensor parallelism = 2 之外是否还开 pipeline parallelism、checkpoint 保存频率、是否启用 flash attention 的具体版本、以及除 AIME/AMC/GPQA 外的其他基准。
