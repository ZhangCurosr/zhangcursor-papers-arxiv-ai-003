---
title: "Towards-Continual-Learning-Mitigating-Forgetting-With-Subspa"
source: https://arxiv.org/pdf/2610.22293v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-04 10:33:34"
field: "大语言模型持续学习"
keywords: ["continual learning", "catastrophic forgetting", "subspace tuning", "LLM fine-tuning", "singular value decomposition", "subspace constraint"]
innovations: ["提出SST方法，将优化器更新投影到主奇异方向补空间以精确冻结关键子空间", "证明对状态优化器必须投影施加更新而非梯度，否则存在严重泄漏", "引入Basis Refresh机制，在严格保护与持续适应之间提供可控权衡"]
benchmarks: ["GPQA-Diamond", "MMLU-Pro", "GSM8K", "HumanEval", "IFEval", "FinQA", "MedMCQA", "WikiText-2"]
---

# 论文速读：Towards-Continual-Learning-Mitigating-Forgetting-With-Subspa

## 一句话总结
提出 Subspace Tuning（SST），通过将权重更新投影到预训练权重矩阵的底部奇异子空间（即剔除主奇异方向），在全参数微调过程中严格约束更新方向，从而在保持目标任务性能的同时显著缓解大语言模型的灾难性遗忘，且该方法与 SFT、SDFT、Muon 等多种训练配方正交可组合。

## 研究问题与动机
- **核心问题**：LLM 在微调学习新技能时发生灾难性遗忘（catastrophic forgetting），原有能力（推理、代码、指令遵循等）严重退化，而原始预训练数据通常不可用。
- **现有方法不足**：经典持续学习方法依赖历史数据或梯度存储，在 LLM 规模下不可行；SFT 将更新集中在主奇异方向，易引发遗忘；即使 on-policy 方法（如 RLVR、SDFT）通过数据侧间接改善，仍存在显著遗忘（尤其是长推理链场景）。
- **关键洞察**：预训练权重矩阵的主奇异方向与模型核心能力密切相关；RL 等 on-policy 方法天然避免更新这些方向，遗忘更少——这启发直接对更新施加子空间约束。
- **优化器状态导致的投影位置问题**：对于 Adam 等状态优化器，梯度投影不等于更新投影，必须作用于最终施加的更新才能精确约束。

## 核心贡献（创新点）
1. **提出 SST 方法**：通过对优化器施加的更新做双向奇异向量投影（保留 free-free 块），严格冻结主奇异子空间；与 Sculpting/OPLoRA 的因子分解方式本质不同，SST 直接训练全参数权重。
2. **证明投影位置至关重要**：通过受控 toy study 证明，在 Adam 下"投影梯度"会导致 69% 的更新泄漏到保护方向，而"投影施加更新"是唯一同时消除泄漏且不损害泛化能力（NLL 增量仅 0.13）的方案。
3. **引入 Basis Refresh 机制**：每 K 步基于当前权重重新计算 SVD 基准，使受保护子空间随训练动态对齐，在严格保护与持续适应性之间提供可控权衡（R10/R100/no-refresh 三档）。
4. **系统性验证 SST 的多维优势**：在 Qwen3（1.7B–32B）和 Qwen2.5-7B-Instruct 上，跨 Science/FinQA/Pathology/Medicine 四个领域验证了 SST 在学习-遗忘 Pareto 前沿、KL 漂移（2–4× 更低）、序列连续微调累积性等关键性质。

## 方法详解
- **核心投影公式**（Eq. 1/3）：对每个被约束的权重矩阵 $W$，取其 SVD 的前 $k=\lfloor r\cdot\min(d_{out},d_{in})\rfloor$ 个左/右奇异向量组成 $U_k,V_k$，则施加的更新为 $\Pi(\Delta W)=(I-U_kU_k^\top)\Delta W(I-V_kV_k^\top)$，即只保留 free-free 块，丢弃 protected-protected、free-protected、protected-free 三块。
- **投影施加更新的算法流程**（Algorithm 1）：优化器收到完整梯度并正常更新动量（Adam/Muon 均适用），计算出原始更新 $\Delta W$ 后，用 $\Pi$ 投影后再加到权重上；embedding 和 layer norm 等不参与投影。
- **Refresh 机制**（Sec 2.3）：每 K 步将 $W_{ref}$ 重置为当前权重，重新计算 $U_k,V_k$ 及对应投影算子；K=∞ 为 no-refresh，K=100 为 R100，K=10 为 R10。Refresh 打开新适应方向但略微弱化对原始预训练子空间的保护。
- **与相关方法的本质区别**：Sculpting 投影梯度且仅去除 block(1)（两投影方向同时触碰的保护块），遗漏 block(2)(3)；OPLoRA 使用低秩因子 $BA$ 配合固定投影器，保护率低（k=16~128）且步骤缩放失配；SST 直接作用于全参数更新且无低秩近似。
- **计算开销**：在 8×L40S 上，SST 投影每步增加 1.5%，R100 refresh 增加 5.1%，R10 增加 37.6%（未分 GPU）；分 8 GPU 分摊 SVD 后降至 0.6%/4.7%。

## 实验与结果
- **模型与数据**：主实验用 Qwen3-8B（thinking 模式），在 Science（DeepSeek-R1 distilled traces）、FinQA/ConvFinQA、Pathology、Medicine（Claude Sonnet 4.6 生成 CoT）四个领域微调；另用 Qwen2.5-7B-Instruct 和 Olmo-3-7B-Think 做补充验证。
- **评估**：ID accuracy 衡量新任务学习；Prior capability 取 GPQA-Diamond、MMLU-Pro、GSM8K、HumanEval、IFEval 五项平均，衡量遗忘程度；额外用 WikiText-2 上 KL 漂移和思维模型响应长度/幻觉率定性评估。
- **核心结果（Qwen3-8B，Table 4）**：
  - Science 领域：SFT 的 LM avg 从 81.0 降至 55.5（Δ=−25.5），SFT+SST 为 80.0（Δ=−1.0），ID 从 77.6 升至 80.1；SFT+SST+R10 进一步提升 ID 至 84.0。
  - FinQA 领域：SFT Δ=−57.3，SFT+SST Δ=+1.3（反向正提升），ID 72.3 vs 73.0。
  - Pathology 领域：SFT Δ=−46.3，SFT+SST Δ=−3.8。
  - Medicine 领域：SFT Δ=未列全但下降显著，SFT+SST 保持接近基线。
- **模型规模缩放（Table 2/3）**：SFT 在所有尺度上遗忘 30–40pp；SST 从 8B 起遗忘几乎消失（Δ≈−1pp），且在每个尺度上 ID 均优于或持平 SFT。
- **数据缩放（Fig. 3）**：相同 45k 样本预算下，SFT 遗忘随训练单调恶化，SST 保持平坦且持续学习。
- **KL 漂移（Fig. 7/14）**：SST  induce 2–4× 更少的 KL 漂移，且准确率-漂移 Pareto 曲线更优。
- **连续微调（Fig. 9/10）**：两阶段 A→B 和三阶段 FinQA→Pathology→Science 中，SST 各阶段技能持续累积，SFT 则每新阶段遗忘前一阶段（如三阶段中 SFT 的 Pathology 和 Science 分别下降 28pp 和 18pp）。
- **对比 Sculpting/OPLoRA（Table 5）**：no-refresh SST 在四领域均值 LM 为 79.6，Sculpting 为 54.9，OPLoRA 为 63.1；给 OPLoRA 设置 k=0.99d 后仍落后 SST 三个领域。
- **SDFT 组合（Table 4）**：SDFT+SST+R100 在 Science 达成 ID=83.4，LM=80.7（Δ=−0.3），遗忘接近消除。
- **Muon 优化器组合**：SFT+Muon+SST 在 Science 达成 ID=82.1，LM=79.4（Δ=−1.6），验证优化器无关性。

## 相关工作脉络
1. **经典持续学习（EWC、GEM、A-GEM 等）**：依赖历史数据/梯度/任务边界，在 LLM 全参数微调场景不可行；SST 无需这些数据且无任务边界假设。
2. **Self-Distillation Fine-Tuning（SDFT, Shenfeld et al. 2026）**：通过重写监督信号使训练靠近模型自身分布，间接减少遗忘；但 SFT/SDFT 在长推理链场景仍严重遗忘；SST 作用在更新侧，可与 SDFT 正交组合。
3. **Sculpting（Nayak et al. 2026, ICLR 2026）**：冻结主奇异向量并投影梯度，仅去除 block(1)（双重保护方向），遗漏 cross-blocks (2)(3)；且投影在梯度端而非更新端。
4. **OPLoRA（Xiong & Xie 2026, AAAI 2026）**：固定投影器下训练低秩因子 $BA$，本质是将低秩更新投影到自由子空间；保护率极低（k=16–128），与 SST 的 99% 保护形成对比。
5. **PiSSA / MiLoRA / CorDA**：参数高效微调方法从奇异方向初始化/训练 adapter；SST 不依赖 adapter 分解，直接投影全参数更新。
6. **RLVR / On-policy RL（Shenfeld et al. 2026）**：RL 更新天然避开主奇异方向；SST 是对这一现象的结构化建模和显式约束。

## 局限性与未来方向
- 最佳学习率需重新调优，未必与 unconstrained SFT 相同；建立完整的学习-遗忘 Pareto 前沿仍需后续工作。
- 尚未在数百亿参数规模模型上验证（论文测试至 32B），scaling 到更大模型是重要下一步。
- 更长任务序列（超过 3 阶段）下的持续性能未见系统评估。
- Refresh 按固定间隔（K 步）进行，可能不够高效；未来可用每 batch 更新信息动态调整保护子空间。
- 在极小 rank-free 场景（如 r=0.99 尾部过小无法拟合任务）下需要放松保护率（论文仅在 Qwen2.5 Science 场景用过 r=0.90）。

## 研究启发与可借鉴点
1. **投影位置的设计洞见**：对状态优化器（Adam/Muon）而言，投影施加更新比投影梯度更精确，这一结论可迁移到其他子空间约束方法的实现中。
2. **Refresh 作为适配-保留可控开关**：固定 K 步刷新 SVD 基准的思路简洁实用，可用于任何基于 SVD 的子空间约束方法，平衡长期适应与短期保护。
3. **子空间约束与蒸馏/RL 的正交可组合性**：SST 可同时叠加在 SDFT、on-policy RL 之上获得叠加收益，提示子空间约束是一种通用正则化组件。
4. **双投影（output-side + input-side）的必要性**：论文证明单向投影会破坏 free-subspace 等价性，必须两侧同时投影才能精确冻结主奇异三元组，这对未来类似方法有参考价值。
5. **模型融合角度**：附录 B.1 表明独立训练的 SST 专家在 task-arithmetic 融合时干扰远小于 SFT 专家，提示子空间约束可能改善模型合并（model merging）的可组合性。

## 关键术语表
**Subspace Tuning (SST)**：将优化器更新投影到预训练权重矩阵主奇异方向的补空间（free-free block），从而在微调中冻结重要子空间、缓解灾难性遗忘的方法。
**Catastrophic Forgetting**：模型在学习新任务时原有能力严重退化的现象，是持续学习的核心障碍。
**Principal Singular Directions**：预训练权重矩阵最大奇异值对应的左右奇异向量，与模型核心功能（推理、指令遵循等）强相关。
**Rank Fraction (r)**：保护的主奇异方向占总方向的比例（论文中通常取 0.99），决定子空间约束的严格程度。
**Basis Refresh**：每 K 步用当前权重重新计算 SVD 基准，更新受保护的奇异向量集，以维持长期适应能力。
**SDFT (Self-Distillation Fine-Tuning)**：通过将监督信号改写为模型自身输出的分布来减少遗忘的 on-policy 微调方法。
**Free-to-Free Block**：更新矩阵中同时落在左侧和右侧自由子空间的部分，即唯一被 SST 保留的更新分量。
**Policy Drift (KL Divergence)**：微调后模型策略与预训练基座策略之间的分布距离，衡量整体行为偏移程度。

## 可复现要素
- **数据集**：Science（open-r1/Mixture-of-Thoughts，DeepSeek-R1 distilled traces）、FinQA/ConvFinQA、Pathology 和 Medicine（MedMCQA + Claude Sonnet 4.6 生成 CoT）；ToolAlpaca、HuatuoGPT-o1、SciKnowEval 用于 instruct 模型实验；均为公开数据集。
- **代码/权重**：论文未明确声明开源仓库（arXiv 版本为 2025–2026，未见公开链接）；权重使用 Qwen3、Qwen2.5、Olmo-3 官方开源权重。
- **关键超参**：rank fraction r=0.99（仅 Science 用 r=0.90）；refresh 间隔 K∈{∞, 100, 10}；learning rate=5×10⁻⁵；effective batch size=16；optimizer 为 AdamW（部分实验用 Muon）；训练设备 DeepSpeed ZeRO-3，8× NVIDIA L40S。
- **评估工具**：vLLM inference；prior capability 取 GPQA-Diamond、MMLU-Pro、GSM8K、HumanEval、IFEval 五项平均。
