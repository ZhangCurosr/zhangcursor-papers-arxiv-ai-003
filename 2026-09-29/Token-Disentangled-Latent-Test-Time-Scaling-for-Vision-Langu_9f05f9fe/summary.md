---
title: "Token-Disentangled-Latent-Test-Time-Scaling-for-Vision-Langu"
source: https://arxiv.org/pdf/2609.35228v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 01:14:41"
field: "多模态大模型推理优化"
keywords: ["test-time scaling", "latent reasoning", "vision-language model", "token routing", "test-time compute", "multimodal reasoning"]
innovations: ["提出token-disentangled latent refinement，将视觉与推理反馈路由至不同latent token角色", "设计图像敏感性路由机制，通过patch blackening识别视觉敏感token", "构建双通道奖励（视觉engagement + 推理likelihood margin），避免单一全局奖励的多模态分配错位"]
benchmarks: ["MMStar", "RealWorldQA", "HallusionBench", "ScienceQA-IMG", "MathVista", "LogicVista"]
---

# 论文速读：Token-Disentangled-Latent-Test-Time-Scaling-for-Vision-Language-Reasoning

## 一句话总结
本文提出了**Token-Disentangled Latent Test-Time Scaling (TD-LTTS)**，一种面向冻结多模态大语言模型的推理时潜变量优化框架，通过图像敏感性路由视觉反馈、熵路由推理反馈，实现token角色感知的潜变量微调，在多个视觉-语言推理基准上持续提升宏观准确率。

## 研究问题与动机
1. **多模态推理中的错误来源不同**：MLLM错误可能源于视觉证据不足（感知瓶颈），也可能源于答案选择分歧（推理瓶颈），需针对性计算。
2. **现有潜变量测试时扩展（Latent Test-Time Scaling）使用单一全局标量奖励**，无法区分editable latent tokens的不同角色（视觉感知 vs 推理决策），易将不相关更新压力错误分配。
3. **多模态上下文含混合证据**：部分latent token绑定视觉输入（图像扰动时likelihood变化大），部分对应不确定答案比较/推理转换（LM-head分布熵高），需分离反馈信号。
4. **候选级奖励无法诊断失败原因**：无法判断continuation失败是源于视觉grounding不足还是答案判别失误，导致全局更新可能恶化性能。

## 核心贡献（创新点）
1. **发现现有潜变量测试时扩展在多模态场景的局限**：候选级标量奖励无法区分editable latent序列中的视觉-证据与推理角色。
2. **提出token-disentangled latent refinement机制**：将视觉参与反馈路由至图像敏感token、推理反馈路由至高熵token，实现多模态感知与推理解耦更新。
3. **设计图像敏感性路由（Image-Sensitivity Routing）**：通过图像patch遮蔽（patch blackening）计算likelihood变化得分，识别视觉敏感token。
4. **构建双通道奖励体系**：视觉奖励$R_{vis}$基于图像token注意力engagement相对初始rollout的tanh缩放；推理奖励$R_{rea}$结合文本先验校正的有界分数与候选间margin。
5. **在六个多模态推理基准、两个冻结 backbone（Qwen2.5-VL-7B、InternVL3.5-8B）上实现一致提升**，宏观准确率分别提升+2.57/+1.51，且优于匹配解码候选预算的输出空间扩展基线。

## 方法详解
1. **可编辑潜变量前缀**：从初始CoT生成的hidden state轨迹中截取前$L$个decoder-layer hidden states作为可编辑潜变量$z^{(0)}$，模型参数冻结。
2. **双奖励分解**：
   - 视觉奖励$R_{vis}(x) = \tanh((E(x)-E(y))/\tau_{vis})$，其中$E(x)$为轨迹级图像token注意力engagement均值，$y$为初始rollout。
   - 推理奖励$R_{rea}(x) = \lambda_s \bar{r}(x) + \lambda_{mar} r_{mar}(c(x))$，基于图像条件likelihood与文本先验likelihood之差（有界tanh），再与同步骤候选集内相对margin结合。
3. **Token路由掩码**：
   - 视觉token：$w_i^{vis}=1[i\in \text{TopK}_{\rho_\nu}(\{\nu_j\})]$，$\nu_i=\phi(\tilde{\ell}_i-\ell_i)$，$\phi(x)=\exp(x)-x-1$，$\tilde{\ell}_i$为图像遮蔽后likelihood。
   - 推理token：$w_i^{rea}=1[i\in \text{TopK}_{\rho_r}(\bar{H}_{N_{vis}})]$，$\bar{H}_i$为标准化LM-head熵，仅从非视觉token中选取。
4. **策略损失**：
   $$
   \mathcal{L}_{policy} = -\lambda_{pg}^{vis} R_{vis}(x_k^*) S_{vis}^{(k)} -\lambda_{pg}^{rea} R_{rea}(x_k^*) S_{rea}^{(k)}
   $$
   其中$S_{vis/rea}$为对应掩码下的log-prob求和。
5. **正则化项**：
   - 熵正则：$\mathcal{L}_{ent} = \lambda_{ent}\frac{\sum_i w_i^{rea} H_i}{\sum_i w_i^{rea}+\epsilon}$鼓励推理token保持不确定性。
   - Anchor正则：$\mathcal{L}_{anchor}=\lambda_{anchor}\|z^{(k)}-z^{(0)}\|_2^2$防止潜变量偏离初始轨迹过远。
6. **解码候选生成**：每步从更新后$z^{(k+1)}$解码1个greedy前缀+3个加高斯噪声($\sigma=0.3$)的变体前缀，接原prompt继续autoregressive生成，形成候选集$X^{(k+1)}$。

## 实验与结果
- **Backbone**：Qwen2.5-VL-7B、InternVL3.5-8B为主；扩展至Qwen2.5-VL-3B、InternVL3.5-4B、LLaVA-OV-1.5-8B、MiMO-VL-RL-8B、Qwen3-VL-8B、Qwen2.5-VL-32B共8个模型。
- **基准**：感知类（MMStar、RealWorldQA、HallusionBench）；推理类（ScienceQA-IMG、MathVista、LogicVista）。
- **基线**：CoT、Self-consistency、Best-of-N、Reward-only、LatentSeek(reasoning/perception)、DMLR。
- **主要结果**（Table 1）：
  - Qwen2.5-VL-7B：感知平均65.41→67.96（+2.55），推理平均67.35→69.94（+2.59）。
  - InternVL3.5-8B：感知平均65.84→66.83（+0.99），推理平均70.01→72.05（+2.04）。
- **最强提升**：对Qwen2.5-VL-7B，超越Best-of-N推理指标+1.62，超越Reward-only推理指标+1.95；对InternVL3.5-8B分别+0.77、+0.45。
- **消融**：移除$R_{vis}$或$R_{rea}$均下降；随机路由退化；$\rho_\nu=\rho_r=0.4$最优；$K=4$步达到饱和。

## 相关工作脉络
1. **Latent Test-Time Scaling**（Geiping et al., 2026; Hao et al., 2024; Li et al., 2025a）：将推理计算移至连续潜变量，但多数采用序列级/全局耦合目标，本文引入token角色解耦。
2. **Multimodal Latent Refinement**（Liu et al., 2025a; Jeon et al., 2026）：如DMLR强调保留视觉信息，但未分离视觉与推理反馈路由。
3. **Token-Level Credit Assignment in Multimodal Reasoning**（Huang et al., 2025; Lu et al., 2026; Miao et al., 2026）：训练时感知-推理解耦工作，本文将其延伸至推理时潜变量优化。
4. **Output-Space Test-Time Scaling**（Wang et al., 2022; Brown et al., 2024; Snell et al., 2024）：Self-consistency/Best-of-N依赖采样选择，本文证明在匹配候选预算下潜变量更新更高效。
5. **Visual Reward Design**：本文用图像token注意力engagement替代传统LLM-based reward，避免额外标注需求。
6. **Image Perturbation for Sensitivity Analysis**：沿用patch blackening技术量化token对视觉输入的敏感性，区别于特征可视化方法。

## 局限性与未来方向
1. **推理成本增加**：每样本需多次候选解码、图像扰动、注意力提取与奖励评分，延长延迟。
2. **奖励为代理目标**：视觉engagement与likelihood差并非正确答案验证器，对歧义图像/未指定问题/需外部知识答案可能引入噪声。
3. **Hard Top-K路由**：当前采用硬性top-$\rho$选择保证分支不相交，软路由或学习路由尚未探索。
4. **仅适用于开放权重模型**：需编辑hidden states并透传潜变量，排除纯API闭源模型。
5. **未来方向**：降低奖励估计成本、扩展至多轮长交互场景、探索软路由机制。

## 研究启发与可借鉴点
1. **Token角色感知的奖励解耦**：将单一全局信号按token功能（视觉敏感/高熵）分流，可迁移至纯文本推理的潜变量优化。
2. **图像敏感性路由的轻量化设计**：通过patch blackening计算likelihood变化，无需额外标注，适合多模态感知-推理解耦任务。
3. **双通道奖励的正则平衡**：Anchor正则约束潜变量偏离，熵正则维持推理不确定性，可借鉴至其他潜变量搜索方法。
4. **匹配候选预算的比较范式**：与输出空间基线在相同decoded-candidate budget下对比，排除计算量优势，更公平评估方法有效性。
5. **跨模型家族泛化验证**：在8种不同架构、规模（3B-32B）、训练策略（SFT/RL）模型上验证，增强结论可信度，可作为方法通用性检验模板。

## 关键术语表
**Token-Disentangled Latent Test-Time Scaling**：一种推理时潜变量优化框架，通过图像敏感性和熵路由将视觉与推理反馈分配至不同token角色。
**Image-Sensitivity Routing**：通过比较图像原始与遮蔽后的token likelihood变化，识别对视觉证据敏感的latent token。
**Visual Engagement Reward**：基于图像token注意力均值的相对奖励，衡量候选轨迹对视觉输入的关注程度提升。
**Reasoning Reward**：结合文本先验校正的有界likelihood分数与候选间margin，评估答案质量与判别确定性。
**Anchor Regularizer**：$L_2$惩罚项约束更新后潜变量与初始轨迹的偏离，防止过度优化破坏原有结构。
**Decoded-Candidate Budget**：方法使用的候选生成数量上限，用于与输出空间扩展基线进行公平比较。
**Hard Top-K Routing**：严格选取top-$\rho$比例的token分别路由至视觉/推理分支，确保两类token掩码不相交。
**Modality Gap in MLLMs**：多模态模型中视觉感知与文本推理能力之间的不一致性，本文方法旨在缓解该差距。

## 可复现要素
- **数据集**：MMStar、RealWorldQA、HallusionBench、ScienceQA-IMG、MathVista、LogicVista（均为公开基准）。
- **代码**：已开源，GitHub链接 https://github.com/Qwen-Applications/TD-LTTS。
- **模型权重**：使用开源backbone（Qwen2.5-VL-7B、InternVL3.5-8B等），参数冻结。
- **关键超参**：$K=4$步更新；可编辑前缀长度$L=\min(\lfloor\rho T\rfloor,300)$，$\rho=0.5$；Adam LR=0.05；$\rho_\nu=\rho_r=0.4$；$\tau_{vis}=0.2$；$\lambda_{pg}^{vis}=\lambda_{pg}^{rea}=1.0$；$\lambda_{s}=1.0,\lambda_{mar}=0.5,\lambda_{p}=1.0$；$\lambda_{anchor}=0.05,\lambda_{ent}=0.01$；噪声$\sigma=0.3$；图像遮蔽patch size=14、drop prob=0.5。
- **部署细节**：Editable states取最后decoder-layer hidden states（LM head前），不编辑视觉编码器或KV cache；候选解码每轮独立调用generate()，重置多模态生成状态。
