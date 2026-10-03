---
title: "UNLOCKING-THE-CRITIC-REWARD-FREE-POLICY-OPTIMIZATION-FOR-LLM"
source: https://arxiv.org/pdf/2609.37119v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-03 02:56:31"
field: "大语言模型强化学习"
keywords: ["reinforcement learning", "LLM post-training", "reward-free optimization", "critic model", "chain-of-thought reasoning", "PPO"]
innovations: ["冻结critic同时作为reward、GAE基线和未完成prefix预测器", "binarization防止policy exploit critic长度偏差", "single-update rule解决critic-based RL不稳定性"]
benchmarks: ["AIME 2025", "AIME 2026", "AMC 2023", "GPQA-Diamond"]
---

# 论文速读：UNLOCKING-THE-CRITIC-REWARD-FREE-POLICY-OPTIMIZATION-FOR-LLM

## 一句话总结
论文提出Reward-Free Policy Optimization (RFPO)，将预训练的critic冻结后作为rollout级reward、GAE基线和未完成prefix的成功预测器，在无需外部label的情况下实现稳定的策略优化，性能匹配监督PPO同时显著降低计算成本。

## 研究问题与动机
- **Critic移除趋势的反思**：当前LLM后训练RL（如DeepSeek-R1、Kimi k1.5）普遍移除critic以降低成本和避免不稳定，但critic已被证明能有效预测未来结果，其价值被浪费。
- **Reward获取成本高**：人类反馈、outcome/process reward models都需要昂贵的标注项目；verifiable rewards需要每个prompt的参考答案，无法随RL规模扩展。
- **长链推理的信���分配难题**：推理trace和agentic episode变长后，generation成本占主导，稀疏的terminal reward使credit assignment更困难，且无法评估未完成的轨迹。
- **Critic不稳定性的归因争议**：现有工作将critic-based RL在长CoT上的不稳定性归咎于value network本身，本文认为这其实是优化配方问题而非网络问题。

## 核心贡献（创新点）
1. **Critic不稳定性是优化artifact**：通过将每次policy update保持较小且低方差（每批一次梯度更新），critic-based RL可以实现稳定收敛，这与"critic inherently unstable"的主流认知不同。
2. **单个冻结critic的三重角色**：预训练critic可同时作为rollout级reward、GAE baseline和未完成prefix的成功预测器，无需外部verifier即可提供dense per-prefix学习信号。
3. **Binarization防止长度偏差利用**：critic的原始分数存在长度偏差，连续reward会让policy exploit这一偏差（缩短或延长响应）；对debiased score进行binarization后关闭了 exploit通道。
4. **零标签匹配监督PPO且成本更低**：binarized RFPO在5,120-token cap下达到与supervised PPO相当的性能（41.1% vs 41.8%平均pass@1），同时将单步时间从582s降至421s（-28%），峰值内存降低9.4GB/GPU。

## 方法详解
**核心公式**：
- Critic固定点：$V_\phi(x, y_{\le t}) \approx \Pr(\text{success} | x, y_{\le t})$（当$\gamma = \lambda = 1$时）
- 校准奖励：$\hat{r}(x, y) = \mathbf{1}[v(x, y) - b(\ell(y)) > \tau]$，其中$b(\ell)$是在正确性类别内估计的长度分位数偏移，$\tau$是匹配正样本率的阈值

**关键设计**：
1. **Critic预训练**：在DAPO-Math-17k上运行supervised PPO，500步8,192-token预算+300步5,120-token cap，从cumulative step 700取初始policy $\pi_{\theta_0}$，step 800冻结critic $\bar{V}_\phi$
2. **一次性校准**：在512个on-policy rollout上拟合$b(\ell)$和$\tau$，不重新拟合
3. **Single-update rule**：每批只做一次梯度更新（mini-batch = batch），避免多内层更新导致的不稳定
4. **监督分数p**：可选混合verifier标签，p=0为完全reward-free，p=1恢复监督PPO

**GAE简化**：当$\gamma=\lambda=1$且使用Eq.(2)奖励时，advantage简化为$A_t = \hat{r} - \bar{V}_\phi(x, y_{\le t})$

## 实验与结果
**实验设置**：
- 模型：Qwen3-4B-Base，在OpenR1-Math-220k的45k long CoT解上SFT 3个epoch
- 基准：AIME 2025/2026、AMC 2023、GPQA-Diamond
- 硬件：8×A100-80GB，verl框架

**主要结果**（Table 2，300步后评估）：
- **RFPO 0% labels**: AIME25=25.4%, AIME26=21.7%, AMC23=70.0%, GPQA=47.3%, Avg=41.1%
- **PPO 100% labels**: AIME25=25.2%, AIME26=21.7%, AMC23=73.6%, GPQA=46.6%, Avg=41.8%
- **RFPO 50% labels**: Avg=41.3%

**关键发现**：
- Continuous reward可达AIME26 pass@1=29.2%（超过PPO的25.0%），但后期下降；binarized后稳定但略低
- 4,096-token cap下：RFPO平均验证47.6% vs PPO 46.6%，GPU-hours 264 vs 327（-19%）
- 约26.5-28.5%的training rollouts在5,120 cap下未完成，但RFPO能同样准确评分

**评估指标**：每10步评估AIME 2026（30题）和AMC 2023（40题），8 samples/problem，temp=0.7，12,288-token budget

## 相关工作脉络
1. **Critic移除方案**（Shao et al. 2024; Ahmadian et al. 2024; Yu et al. 2025）：用group baseline替代critic，本文与之对比证明冻结critic可替代
2. **Critic修复方案**（Yuan et al. 2025; Yue et al. 2025; Shan et al. 2026; Pan et al. 2026）：仍让critic仅作为baseline，本文扩展为reward
3. **Process reward models**（Lightman et al. 2024; Wang et al. 2024）：需step-level标注，本文critic从outcome学习且可预测未完成prefix
4. **Label-free替代**（Zuo et al. 2025; Yuan et al. 2024; Zhao et al. 2026）：易受reward hacking，本文critic受verifier grounding和校准保护
5. **Implicit reward**（DPO/Rafailov et al. 2023; PRIME/Cui et al. 2025）：仍需training labels，本文完全无需

## 局限性与未来方向
- **模型规模与任务范围**：仅在Qwen3-4B数学推理上验证，未扩展至更大模型或其他任务（如agentic、code）
- **连续reward的ceiling未解决**：Continuous RFPO可达到29.2%（超PPO），但后期下降；如何安全利用这一潜力是开放问题
- **长度偏差分离不完全**：debiasing后仍存在残余长度依赖（正确响应中r=-0.26，错误响应中r=+0.22）
- **Calibration drift**：随着policy移动，阈值τ的平衡准确率从0.934下降，需监控或定期重校准
- **极短cap的失效边界**：1,024-token cap下precision降至0.213，说明critic预测能力有下限

## 研究启发与可借鉴点
1. **Single-update stability rule**：每批一次梯度更新+大batch可解决critic-based RL不稳定问题，可直接迁移到其他RLHF场景
2. **Explained variance作为训练代理**：EV与within-problem AUC高度相关（r=0.91），可在训练时低成本监控critic质量
3. **Prefix scoring for truncated rollouts**：Critic在未完成的prefix上仍能提供准确预测（AUC 0.84-0.92 at 4,096 tokens），可推广至其他需要early stopping的场景
4. **Calibration guard against reward hacking**：通过within-class length decile偏移+binarization双重防护，为reward design提供防exploit范式
5. **Partial label protection**：p=0.5的半监督版本可与纯reward-free性能持平，提示在实际应用中可用少量label加固稳定性

## 关键术语表
- **RFPO (Reward-Free Policy Optimization)**：用冻结critic替代verifier作为reward的训练框架
- **GAE (Generalized Advantage Estimation)**：Schulman et al. 提出的advantage估计方法，本文令$\gamma=\lambda=1$使其简化
- **Within-problem AUC**：固定problem后ranking attempts的AUC，比overall AUC更能反映reward质量
- **Debiasing term $b(\ell)$**：在同一correctness class内按长度分位数估计的偏移，分离长度偏差与正确性信号
- **Explained Variance (EV)**：价值头对terminal reward的解释方差，作为critic质量的cheap训练指标
- **Supervision fraction p**：混合verifier标签的比例，p=0完全reward-free，p=1为监督PPO
- **Answer restatement**：response中声称自身answer的次数，是continuous reward exploit的早期信号

## 可复现要素
- **数据集**：OpenR1-Math-220k（SFT用45k subset）、DAPO-Math-17k（critic预训练+RL）、AIME 2025/2026、AMC 2023、GPQA-Diamond；论文声明开源
- **代码/权重**：提供所有训练log、calibration artifacts、benchmark outputs和figure regeneration script；使用verl框架
- **关键超参**：actor LR=$10^{-6}$，critic LR=$10^{-5}$，clip=0.2，$\gamma=\lambda=1$，batch=128 prompts×8 responses，5,120-token cap，temperature=1.0采样
- **Calibration详情**：512 on-policy rollouts拟合，$\tau=0.5215$，93.4% agreement with verifier
- **硬件**：8×A100-80GB，bf16，FSDP，gradient checkpointing
