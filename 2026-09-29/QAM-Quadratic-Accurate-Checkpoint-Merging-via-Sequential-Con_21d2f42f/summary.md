---
title: "QAM-Quadratic-Accurate-Checkpoint-Merging-via-Sequential-Con"
source: https://arxiv.org/pdf/2609.35168v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 03:17:12"
field: "大语言模型预训练与权重合并"
keywords: ["checkpoint merging", "weight averaging", "reconstruction bound", "sequence consistency", "quadratic exactness", "LLM pretraining"]
innovations: ["在局部转移模型下刻画了二阶序列一致的充要矩条件，给出$O(h^3)$极小极大下界；提出QAM系数构造达到该最优率，并在所有二次目标上精确匹配", "通过独立Bernoulli变量和的分布唯一确定合并权重，给出$O(k^2)$卷积算法，无需梯度/Hessian估计"]
benchmarks: ["GSM8K", "MMLU", "BoolQ", "HellaSwag", "PIQA", "COPA", "WinoGrande", "WSC", "WiC", "SST-2", "OpenBookQA", "SciQ", "ARC-Easy", "ARC-Challenge", "MuTual"]
---

# 论文速读：QAM-Quadratic-Accurate-Checkpoint-Merging-via-Sequential-Con

## 一句话总结
本文提出二次精确 checkpoint 合并方法 QAM，通过序列一致性准则与二次函数精确匹配原理，为 LLM 训练过程中的 checkpoint 加权合并提供了具有最优重建误差下界（$O(h^3)$）的理论保证；在 SmolLM3-3B 和 OpenEuroLLM-Prelude-9B 两个公开 Adam 预训练轨迹上的实验表明，QAM 在中长窗口场景显著优于 WSM 基线，尤其数学任务提升显著。

## 研究问题与动机
- **问题**：已保存的 checkpoint 记录了训练轨迹上的状态，但不能决定不同训练计划下会访问的状态处的更新，因此如何从固定长度的梯度下降历史中重建序列参考端点是一个根本性限制问题。
- **现有方法不足**：WSM（Warmup-Stable Merge）给出了 checkpoint 权重与已记录更新强度之间的精确尾部和对应关系，但将其应用于稀疏 Adam 历史时仅依赖启发式规则，未考虑序列重计算带来的状态依赖问题；其理论可到达精度缺乏分析。
- **动机**：需要建立 checkpoint 重建构限的理论下界，并设计一个达到该最优率且可通过二次模型精确选择的合并规则，以区分"设计准则"与"下游评估"两个层面。

## 核心贡献（创新点）
1. **局部序列一致性的充要条件刻画**：在共同局部转移模型下，证明两个索引矩条件（均值与二阶阶乘矩）是凸合并与序列参考在二阶一致性的充要条件，将系数设计问题转化为矩匹配问题。
2. **checkpoint 信息重建的极小极大下界**：证明对非退化 profile，任何仅使用固定长度 GD 历史（步长 $h$）的算法无法实现$o(h^3)$ 以下的端点误差，构造了两个共享相同 checkpoint 历史但端点分离 $\Omega(h^3)$ 的光滑强凸损失函数。
3. **QAM 系数的闭式构造与二次精确性**：通过将独立 Bernoulli 变量之和的分布作为 checkpoint 权重（生成多项式乘积形式），达到匹配的 $O(h^3)$ 上界；进一步利用局部二次损失模型，证明 QAM 在全部固定二次目标上精确匹配序列 GD 参考端点，且在所有线性规则中唯一。
4. **经验验证与诊断设计**：在两个公开 LLM 预训练轨迹（SmolLM3-3B、OpenEuroLLM-Prelude-9B）的 15 个任务、9 种窗口/profile 配置上进行评测；设计匹配矩的 MaxEnt 控制实验，揭示二阶一致性不足以完全决定下游分数。

## 方法详解
- **序列参考定义**：给定有序 checkpoint $\theta_0, \ldots, \theta_k$ 与递降 profile $W_j$，将每条 checkpoint 区间视为一个整体转移，定义 $\phi_0^W = \theta_0$，$\phi_{j+1}^W = \phi_j^W + W_j[F(\phi_j^W) - \phi_j^W]$，其中 $F$ 为公共转移映射。
- **局部转移模型**：假设 $F_h(\theta) = \theta + hf(\theta) + h^2 a(\theta) + O(h^3)$，展开后 base merge 与序列参考在 $f_0$ 和 $a_0$ 项上一致，但交互项差为 $h^2\Delta_2(W)J_0f_0$，其中 $\Delta_2(W) = \sum_{\ell<j}W_j(1-W_\ell) \geq 0$。
- **QAM 系数构造**：将嵌套指示变量 $X_j = \mathbf{1}\{I_c > j\}$ 替换为独立 Bernoulli 变量 $Y_j \sim \mathrm{Bernoulli}(W_j)$，令 $I_q = \sum_j Y_j$，则生成多项式 $Q_W(t) = \prod_{j=0}^{k-1}(1-W_j + W_j t) = \sum_i q_i t^i$，合并权重为 $q_i$。卷积递推公式：$q_i^{(j+1)} = (1-W_j)q_i^{(j)} + W_j q_{i-1}^{(j)}$，计算复杂度 $O(k^2)$，仅需 $O(k)$ 存储。
- **二阶一致定理**：凸权重 $p$ 使 $\sum_i p_i\theta_i - \phi_k^W = O(h^3)$ 对所有满足 Assumption 1 的转移成立当且仅当 $\mathbb{E}_p I = A_W$ 且 $\mathbb{E}_p\binom{I}{2} = B_W$。
- **信息极限定理**：对任意非退化 $W$（$S_W = \sum_{\ell<j}W_jW_\ell(1-W_\ell)>0$），最坏情况重建误差满足 $ch^3 \leq R_h \leq Ch^3$，QAM 达到上界；即使补充 checkpoint 梯度和 loss 值，下界依然成立。
- **二次精确性**：对于固定二次目标 $L(\theta_0+u)=L(\theta_0)+g_0^\top u+\frac{1}{2}u^\top H_0 u$，QAM 系数满足 $\sum_i q_i\theta_i = \phi_k^W$ 精确成立，且该性质唯一确定 $q$。

## 实验与结果
- **数据集/模型**：两个公开 Adam 预训练轨迹——SmolLM3-3B（15 个等间隔 checkpoint，每~100B tokens）和 OpenEuroLLM-Prelude-9B（40 个 checkpoint，每~40B tokens）；窗口分短/中/长三段（Last-5/10/15 和 Last-10/20/40），三种 profile（linear、cosine、$1-\sqrt{\cdot}$）。
- **评估基线**：配对 WSM、单个 constituent checkpoint、constituent 均值。
- **主要结果**：
  - 相对 constituent 均值，QAM 在 105/108 项能力比较中提升（仅 3 项下降，均为 SmolLM3 Reading/Dialogue）。
  - 与配对 WSM 相比：短窗口差异小且混合（QAM 赢 15/36）；中窗口赢 28/36；长窗口赢 29/36。
  - 超过最新 checkpoint：219/270 次（Win），7 模型×15 任务×中/长窗口组合中 45/60 次超越。
  - 超过各任务的 best 单个 checkpoint：147/270 次。
  - **最强结果**：Prelude Last-40 线性 profile GSM8K：QAM 得 45.03 vs WSM 得 42.91（+2.12）；Prelude Last-40 cosine GSM8K 中 WSM 仅 14.03（flexible）而 QAM 达 44.66，差距巨大（WSM 出现严重生成长文本循环退化）。
  - 以 WSM 全局最佳（跨所有 9 配置）为基准，QAM 在 SmolLM3 8/15 任务、Prelude 9/15 任务胜出。
- **诊断发现**：匹配矩的 MaxEnt 控制在 GSM8K 上得分与 QAM 相差不稳定（最大差距达 15.15 点），证明二阶一致性不唯一决定下游表现。

## 相关工作脉络
- **Stochastic Weight Averaging (SWA)**（Izmailov et al., 2018）：最早提出训练过程中取权重平均，本文继承并区分了"设计准则"与"重建精度"两个层次。
- **LAWA**（Sanyal et al., 2023）：研究高学习率下早期平均，本文在理论框架下给出更严格的局部重建分析。
- **Model Soups / Fisher-weighted / TIES-Merging**（Wortsman et al., 2022; Matena & Raffel, 2022; Yadav et al., 2023）：跨模型/任务合并，本文聚焦于同轨迹内按 profile 设计系数。
- **WSM**（Tian et al., 2026）：直接对比基线，给出 tail-sum 对应关系；本文在此基础上引入序列重计算和二次精确性修正。
- **Li et al. (2025)**：提出二次损失展开解释 pretraining averaging，本文以其 quadratic matching 作为系数唯一性选择准则。
- **Sandler et al. (2023)**：将 SGD 轨迹与学习率调度联系，本文用类似局部转移模型但聚焦于 checkpoint 历史的受限信息重建问题。

## 局限性与未来方向
- **模型假设与实际 Adam 历史的差距**：共同局部转移假设 $F_h$ 是理想化模型，稀疏 Adam 历史未必满足该光滑确定性映射。
- **理论未给出最优常数与下游排序**：极小极大界只保证最优阶 $O(h^3)$，未估计常数大小，也未提供不同任务间的排名保证。
- **窗口长度与 granular 未完全解耦**：改变合并粒度同时改变了 QAM 的有效宽度，导致二者效应混淆。
- **未与 Warmup-Stable-Decay (WSD) 直接对比**：需额外训练，论文未涉及。
- **未来方向**：探索该重建视角能多好地近似真实学习率衰减 schedule 的参考端点，需要更强的局部训练动态模型来桥接理论与实际。

## 研究启发与可借鉴点
- **矩匹配作为设计准则的通用框架**：将物理/数值分析中的局部展开思想引入 LLM checkpoint 合并，证明了"序列一致性"可作为无额外训练的系数设计原则，可迁移至其他权重平均场景。
- **信息极限证明技术**：构造两个共享相同 checkpoint 记录但端点 $\Omega(h^3)$ 分离的光滑强凸损失，对分析任何基于历史信息的重建算法的不可约误差有方法论参考价值。
- **MaxEnt 控制的诊断实验设计**：通过固定前两阶矩构造独立对照组，揭示理论一致性与实际性能之间的差距，这种"控制变量+最大熵"方法可推广到其他合并系数的消融分析。
- **生成质量诊断的量化指标**（delimiter/no-newline/loop/median length）：为 WSM 在长窗口的退化提供了超越数值分数的行为解释，可作为 LLM 合并方法的通用评估协议。
- **与团队结合机会**：若团队关注 checkpoint 管理/高效预训练策略，可将 QAM 的二次精确性扩展至带 Hessian 信息的二阶合并或跨任务合并场景。

## 关键术语表
- **QAM (Quadratic-Accurate Merging)**：一种基于序列一致性和二次精确匹配原理的 checkpoint 凸合并方法，系数由独立 Bernoulli 变量和的分布生成。
- **WSM (Warmup-Stable and Merge)**：先将 LR 保持 warmup-stable 再合并 checkpoint 的方法，本文直接对比基线。
- **序列参考 (Sequential Reference)**：在 checkpoint 尺度上将每个区间视为整体转移并按当前状态重算的虚拟训练轨迹，用于定义系数设计的理论基准。
- **局部转移模型 (Local Checkpoint Transition)**：$F_h(\theta)=\theta+hf(\theta)+h^2 a(\theta)+O(h^3)$ 的 Taylor 展开假设，用于理论分析。
- **索引矩条件 (Index Moment Conditions)**：合并权重 $p$ 相对于 checkpoint 索引 $I$ 的均值 $\mathbb{E}[I]=A_W$ 与二阶阶乘矩 $\mathbb{E}\binom{I}{2}=B_W$，是二阶一致性的充要条件。
- **$S_W$ (Profile Interaction Sum)**：$S_W = \sum_{\ell<j}W_jW_\ell(1-W_\ell)$，衡量 profile 的非退化程度；$S_W>0$ 时存在 $\Omega(h^3)$ 信息极限。
- **MaxEnt 控制 (Maximum-Entropy Control)**：在给定均值和方差约束下，与目标合并共享相同前两阶矩的最大熵分布，用于分离矩信息与任务表现的因果。
- **非退化 profile (Nondegenerate Profile)**：$S_W > 0$ 的 update profile，即存在至少一对正权重使得前一个权重不为 1。

## 可复现要素
- **数据集**：SmolLM3-3B 和 OpenEuroLLM-Prelude-9B 的公开 checkpoint 轨迹（论文提供了具体 iter 编号范围）；评测任务来自 lm_eval v0.4.13 的 15 个标准 benchmark。
- **代码/权重**：论文未明确声明代码开源；使用了 fp32 累加、bf16 导出的合并格式，无额外训练。
- **关键超参**：窗口长度（5/10/15 或 10/20/40），profile 形状（linear/cosine/$1-\sqrt{\cdot}$），评估 shot 数（MMLU/GSM8K 用 5-shot，其余 zero-shot）；合并均无调参。
