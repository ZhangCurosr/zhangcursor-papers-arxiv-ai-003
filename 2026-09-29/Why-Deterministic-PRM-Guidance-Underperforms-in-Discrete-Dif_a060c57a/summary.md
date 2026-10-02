---
title: "Why-Deterministic-PRM-Guidance-Underperforms-in-Discrete-Dif"
source: https://arxiv.org/pdf/2609.35472v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 08:16:50"
---

# 论文速读：Why-Deterministic-PRM-Guidance-Underperforms-in-Discrete-Dif

## 一句话总结
本文在严格匹配前向计算预算的条件下评估离散扩散语言模型（dLLM）中的确定性PRM引导策略，证明其因早期去噪阶段信号衰减与top-1剪枝导致的候选池多样性崩溃，在数学与代码推理任务上显著落后于简单的独立采样+ORM重排序。

## 研究问题与动机
- dLLM每步去噪暴露部分解状态，看似天然适合引入过程奖励模型（PRM）进行引导；但AR范式中成功的PRM直接移植到dLLM是否有效尚未验证。
- 现有dLLM研究多聚焦生成器或采样器改进，缺乏在**公平计算预算**下对PRM引导、ORM重排序与最佳之N基线的系统对比。
- PRM的判别信号在高分辨率掩码（mask ratio）下是否仍可依靠？确定性剪枝是否会不可逆地破坏候选池的信息上限？
- 跨掩码训练的PRM能否胜任最终答案裁决？因果注意力与双向注意力在散点状态评分上的本质差异来源是什么？

## 核心贡献（创新点）
- 提出统一的**匹配前向计算预算诊断协议**，将去噪、中间PRM评分与终态ORM评分计入同一单位，消除隐式计算代价对对比结论的混淆。
- 首次定量分解PRM引导性能缺口为**候选池损伤**与**终态选择质量**两个可分离来源，揭示早期弱信号剪枝与跨掩码训练导致的终态判别失配。
- 提出并验证**last-token pooling**可修复因果PRM的读数不匹配，恢复约70%的终态ROC-AUC差距，并解决重排时的非单调坍缩现象。
- 开源快照语料、PRM LoRA适配器权重与评估工具包，为dLLM测试时扩展研究提供可复现的公平比较基准。

## 方法详解
- **匹配计算协议**：定义单次前向传播为基本成本单位。PRM Guided的开销公式为 $C_{\text{PRM}}(K,b) = KT + K\lceil T/b \rceil$，其中T=128为去噪总步数，K为分支宽度，b为评分间隔；ORM Rerank开销为 $128N + N$，两者在 headline 设置下误差<0.8%。
- **PRM架构与训练**：基于冻结的dLLM骨干（Dream-v0-Instruct-7B / LLaDA-8B-Base），接入LoRA适配器（r=16, α=32）+ 两层MLP奖励头 + 256维正弦步长嵌入。使用GSM8K训练集的on-policy中间快照，以二元最终正确性为标签，BCE损失训练2000步。
- **Segmental Top-1 Pruning算法**：每b步将当前状态复制为K份并行去噪，随后用PRM对所有K个状态打分，仅保留得分最高者进入下一段，形成确定性引导轨迹。
- **对照与诊断设计**：包含PRM Hybrid（终段不剪枝以测天花板）、ESS-tempered SMC重采样器（保多样性）、以及相同候选池上的ORM/Final-state PRM/Cross-mask PRM重排对比；通过ROC-AUC掩码曲线、答案唯一性、Top-M截断风险率与读数消融定位瓶颈。

## 实验与结果
- **数据集与模型**：GSM8K测试集（1319题）、MATH500、MBPP；主模型Dream-v0-Instruct-7B，交叉验证LLaDA-8B-Base。
- **核心准确率对比**：匹配预算下ORM Rerank@8达**75.13%**，PRM Guided K=8仅**65.18%**（差距9.95 pp）；K=32时差距扩大至**12.69 pp**（82.71% vs 70.02%）。PRM Guided优于Majority但全面落后于独立采样+ORM。
- **跨任务一致性**：MATH上ORM领先9.85 pp，MBPP上领先12.16 pp；MBPP中PRM终态重排与ORM持平，说明该任务劣势完全来自引导过程的剪枝损耗。
- **机制量化**：PRM ROC-AUC随掩码比例从0.77跌至0.54；Top-1剪枝使Oracle天花板从81.05%降至67.30%（损失13.75 pp）；SMC恢复多数候选后PRM选择准确率仍仅65.48%，证明终态选择是剩余瓶颈。
- **架构诊断**：双向PRM显著优于因果PRM；将因果PRM读数从mean pooling改为last-token pooling后，终态ROC-AUC从0.61升至0.73，重排性能恢复单调上升。

## 相关工作脉络
- **AR奖励模型与验证器重排**：Lightman et al. (2024) PRM、Wang et al. (2024) Math-Shepherd、Uesato et al. (2022)等依赖线性前缀顺序，与dLLM散落掩码状态存在架构错位。
- **dLLM推理与解码**：Dream (Ye et al. 2025)、LLaDA (Nie et al. 2025)、Block Diffusion等主要优化生成质量或速度，未系统回答“PRM引导是否优于best-of-N”这一基础问题。
- **测试时扩展与无奖励引导**：SMC粒子采样 (Ou et al. 2026)、RFG奖励自由引导 (Chen et al. 2025)、重掩码采样 (Wang et al. 2026a)，本文与之对照并指出显式中间PRM+终态ORM解耦更具性价比。
- **匹配计算评估方法学**：传统best-of-N/self-consistency缺乏公平计权；本文建立forward-pass级对齐基准，填补dLLM测试时 scaling 评估的规范空白。
- **因果/双向读出发散**：与AR奖励模型中的last-token读取惯例对照，揭示散点状态评分中pooling策略的关键作用及mean pooling的坍塌风险。

## 局限性与未来方向
- 实验域局限于数学与代码推理，未检验创造性文本生成；猜想弱推理任务的信号衰减可能较轻，但需实证验证。
- PRM仅使用二元最终正确性标签，未引入步骤级细粒度监督；过程监督对早期信号曲线的改善效果未知。
- 仅评估segmental top-1与top-M剪枝，自适应分支调度、lookahead评分、显式多样性核的SM
