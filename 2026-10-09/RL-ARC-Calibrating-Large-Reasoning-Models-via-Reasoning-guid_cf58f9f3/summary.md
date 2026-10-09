---
title: "RL-ARC-Calibrating-Large-Reasoning-Models-via-Reasoning-guid"
source: https://arxiv.org/pdf/2610.11352v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-09 10:21:09"
---

# 论文速读：RL-ARC-Calibrating-Large-Reasoning-Models-via-Reasoning-guid

## 一句话总结
本文提出 RL-ARC，一种在强化学习推理训练中联合利用**推理置信度**与**答案置信度**的可校准框架；通过在正确/错误样本上分别施加正则化与过度自信惩罚，在数学与复杂推理任务上显著缓解大模型的过度自信问题，同时在 ID 与 OOD 设置下保持与纯准确率导向的 RLVR 相当的性能。

## 研究问题与动机
- **RLVR 的校准缺陷**：标准推理强化学习（RLVR）仅以答案正确性为二元奖励，训练后模型易产生系统性过度自信，限制其在医疗、金融等高信赖需求场景的应用。
- **现有校准方法的局限**：已有校准感知训练（如 RLCR、Behavioral Calibration）虽能改善 ID 校准，但在分布外（OOD）场景下仍倾向将高置信度分配给错误答案，且普遍面临准确率下降的 trade-off。
- **忽视推理过程可信度**：现有工作仅对齐“答案层面的置信度”，未显式建模推理链本身的质量； flawed reasoning 即使偶得正确答案也可能被赋予过高置信度，错误推理则得不到有效抑制。
- **实用预测可靠性不足**：高风险应用要求模型在低覆盖率（仅保留高置信输出）时仍能保持低错误率，而现有方法的风险覆盖曲线表现较差。

## 核心贡献（创新点）
1. **提出 RL-ARC 校准感知 RL 框架**，将推理置信度作为辅助信号联合优化答案置信度。与 RLCR 等仅依赖答案置信度的方法本质不同，本文首次把推理过程可信度显式纳入强化学习奖励设计。
2. **设计条件式分段优化目标**，根据预测是否正确动态切换作用机制：正确时最小化 $c_a$ 与 $c_r$ 的差异（推理引导正则化），错误时额外施加 $c_r^2$ 惩罚（过度自信压制）。避免了统一校准损失导致的 accuracy-calibration trade-off。
3. **多维实证验证 ID/OOD 泛化与推理质量提升**。不仅在 Math 与 OOD 基准上取得最佳/次佳校准指标，还通过风险覆盖曲线、置信度分箱分布与浅层推理（SR）分析，系统证明该方法能让模型自适应估计不确定性并提升推理过程可靠性。

## 方法详解
- **基础奖励**：沿用 RLVR 二元奖励 $R_a = \mathbf{1}_{y=y^*}$，结合 Brier Score 构建答案校准奖励 $R(z, c_a) = z - (z - c_a)^2$，其中 $z$ 为预测是否正确（0/1），$c_a$ 为模型自我评估的答案置信度。
- **置信度 Elicitation**：通过定制化 prompt 指令模型在生成推理链的同时，显式输出推理置信度 $c_r$ 与答案置信度 $c_a$（verbalized confidence）。
- **核心目标函数**：$\max_\theta \mathbb{E}_{(x,y^*)\sim\mathcal{D}, (y,c_a,c_r)\sim\pi_\theta} [R_{total}(z, c_a, c_r)]$，其中 $R_{total} = R(z, c_a) - \Omega_{arc}$。
- **分段正则化/惩罚项**：
  $$\Omega_{arc}(z, c_a, c_r) = \begin{cases} \lambda_{pos}(c_a - c_r)^2, & z=1 \text{（预测正确）} \\ \lambda_{neg} c_r^2, & z=0 \text{（预测错误）} \end{cases}$$
- **设计原理**：当预测正确时，$(c_a - c_r)^2$ 强制答案置信度与推理置信度对齐，防止“蒙对却盲目自信”；当预测错误时，原有 $R(z,c_a)=-c_a^2$ 已惩罚高 $c_a$，叠加 $\lambda_{neg} c_r^2$ 进一步压低奖励，迫使模型在错误推理时同时降低答案与推理置信度。默认超参 $\lambda_{pos}=0.4$、$\lambda_{neg}=0.15$（正主导配置表现最优）。

## 实验与结果
- **设置**：主干模型 Qwen2.5-7B、Qwen3-8B、Llama-8B (DeepSeek-R1 Distilled)；训练集 Big-Math（30k 样本）与 HotpotQA；基线涵盖 Base、RLVR、RLCR、Behavioral Calibration、Post-hoc、Probability-based 等；评估指标含 Accuracy、Brier、ECE、AUROC、AURC、Reasoning Calibration、Shallow Reasoning。
- **ID 基准（Qwen2.5-7B）**：RL-ARC 准确率 56.53%（略低于 RLVR 的 56.86%，显著优于 RLCR 的 54.82%）；ECE 0.18、Brier 0.23、AUROC 0.60，均为全场最优或次优。
- **OOD 基准（Qwen2.5-7B）**：RL-ARC 准确率 47.27%（与 RLVR 的 47.58% 基本持平，显著优于
