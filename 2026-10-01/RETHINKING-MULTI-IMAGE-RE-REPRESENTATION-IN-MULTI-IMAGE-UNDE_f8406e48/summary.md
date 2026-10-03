---
title: "RETHINKING-MULTI-IMAGE-RE-REPRESENTATION-IN-MULTI-IMAGE-UNDE"
source: https://arxiv.org/pdf/2609.39363v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-03 14:14:51"
field: "多模态理解与视觉推理"
keywords: ["多模态大语言模型", "多图理解", "视觉工具使用", "强化学习", "思维链推理", "视觉中间表征"]
innovations: ["提出多图重表征框架，系统比较文本与视觉中间表征的五种设置", "设计 Mosaic 视觉 harness 和 MosaicBench 细粒度基准", "仅用准确率和格式奖励即通过 RL 训练出多步视觉工具组合能力"]
benchmarks: ["MosaicBench", "M4Bench", "BLINK", "Mantis", "MMIU"]
---

# 论文速读：RETHINKING MULTI-IMAGE RE-REPRESENTATION IN MULTI-IMAGE UNDERSTANDING

## 一句话总结
本文提出"多图重表征"（multi-image re-representation）框架，系统比较文本与视觉两种中间表征方式在多图理解任务中的效用差异，并据此设计了图像操作工具集 Mosaic 及专用基准 MosaicBench；在此基础上，仅用准确性和格式奖励进行 RL 训练，成功让 MosaicAgent-8B 学会多步视觉工具组合与多样化问题解决策略。

## 研究问题与动机
1. **多模态大语言模型（MLLM）在多图任务中仅能被动编码多张图像，缺乏主动重构视觉证据的能力**——跨图像证据往往分散、细微、方向各异，需通过重组才能凸显任务相关细节。
2. **现有方法未厘清"何时视觉重表征比文本推理更有效"**——多数工作将视觉工具使用默认合理，但不同任务对精确视觉证据（如精确定位、方向敏感推理）的依赖程度差异巨大。
3. **已有多图基准混合多种任务需求**（如 BLINK、M4Bench），无法隔离评估视觉重表征本身的增益来源。
4. **如何训练模型自主学习构造有用的视觉中间表示仍不明确**——特别是仅需准确性/格式奖励，无需演示轨迹或工具序列奖励，能否让模型学会多步视觉工具组合。

## 核心贡献（创新点）
1. **形式化"多图重表征"概念，定义五种重表征设置**（No-Re²、T-Re²、PG-Re²、V-Re²、PV-Re²），首次在同框架下系统比较文本与视觉中间表征。
2. **提出 Mosaic，一种可组合的多图视觉 harness**，提供十种确定性图像操作（裁剪、旋转、拼贴、像素差分等），允许模型在持久工作空间中主动构造和复用视觉中间件。
3. **引入 MosaicBench 及配套训练集**，涵盖 32 类 grounding-focused 细粒度多图任务（分辨率、方向、精确比较、假设检验、上下文干扰、空间参考），填补了现有基准在该方向上的空白。
4. **证明仅需准确性与格式奖励的 RL 即可让 MosaicAgent-8B 学会多步视觉工具组合**，且在未见演示轨迹的情况下涌现多样化的问题解决模式（progression、self-correction、后稳定继续处理）。

## 方法详解
**多图重表征的形式化**：给定输入 $x = (q, I_1, \dots, I_n)$ 和答案 $y$，重表征器 $p_\theta^m(z|x)$ 构造中间记录 $z$，求解器 $p_\phi(y|x,z)$ 基于原始输入和中间记录输出答案，总分布为 $p_{\theta,\phi}^m(y|x) = \sum_{z \in \mathcal{Z}_m} p_\theta^m(z|x) \cdot p_\phi(y|x,z)$。

**五种重表征设置**：
- **No-Re²**：直接回答，无显式中间推理。
- **T-Re²**：自由文本 CoT 推理（标准思维链提示）。
- **PG-Re²**：问题引导的文本重描述（描述视觉内容、标注来源、组织跨图比较）。
- **V-Re²**：在线视觉重表征，模型通过交互轨迹 $\tau = (a_1, o_1, \dots, a_T, o_T)$ 逐步调用工具。
- **PV-Re²**：预先生成的视觉中间件，模型无工具访问权限，仅做最终推理。

**Mosaic 视觉 harness**：包含持久化图像工作空间（每个资产有独立透明画布，新操作不修改输入），以及十种可组合操作：crop、rotate、resize、flip、apply affine transformation、draw normalized coordinate grid、compare per-pixel image difference、apply homography transformation、make collage、overlay images。

**RL 训练**：以 Qwen3-VL-8B-Instruct 为初始策略，采用 GRPO，无监督预热或演示轨迹。奖励函数 $r(\tau, y, y^*) = \text{acc}(y, y^*) + \text{fmt}(\tau, y)$，均为二值项。每提示采样 8 个 rollout，共 218 步 RL。训练数据从 12,800 条候选中筛选出 3,509 条（no-tool 失败且 tool-enabled 在 2–7 次中成功的样本）。

## 实验与结果
**数据集与基准**：
- MosaicBench：560 条样本，28 类任务（每类 20 条），与训练集在样本、源、图像三级去重叠。
- M4Bench、BLINK、Mantis、MMIU（公开多图理解基准）。

**主要结果**：
- MosaicAgent-8B 在 MosaicBench 上达 **63.3%**，超越最强开源基线（Qwen3.5-27B 的 60.9%）**+2.4pp**；在 M4Bench 上达 **61.8%**，超越最强开源基线（Qwen3.5-27B 的 57.5%）**+4.3pp**。
- **Hypothesis testing** 子任务提升最大：MosaicAgent-8B 达 62.9%，vs 最强开源基线 31.7%（+31.2pp）。
- Precision comparison：87.0% vs 81.0%（+6.0pp）；Orientation：56.9% vs 51.7%（+5.2pp）。
- Mantis 达 81.6%，接近 Qwen3.5-9B 的 81.9%。

**关键发现**：
- 视觉重表征在"Detailed Difference"任务上（M4Bench）从 7.3%（No-Re²）→ 46.1%（T-Re²）→ 59.9%（V-Re²），显著优于文本方案。
- 但在"State Comparison"任务上所有模型在 V-Re² 下反而低于 T-Re²，说明**视觉重表征的收益具有强任务依赖性**。
- PV-Re²（预先生成的视觉中间件）在某些任务（如 BLINK 的 Relative Depth）上优于 V-Re²，表明在线构造本身存在难度。

**工具使用演变**：训练后总工具调用增加 1.66×；cropping 占比从 36.7%→54.8%，collage 从 0.5%→6.2%，pixel differencing 从 16.1%→3.3%。

**消融**：完整工具集相比仅 crop 在 MosaicBench 上 +9.4pp，凸显多样操作的价值。

## 相关工作脉络
1. **Multi-modal Chain-of-Thought（Multimodal-CoT, PromptCap, QG-CoC）**：基于文本中间表征组织视觉证据，本文将其定位为 Re² 框架中的 T-Re² / PG-Re² 设置，与 V-Re² 形成对照。
2. **Thinking with Images / Visual Agent 方法（Visual Sketchpad, PyVision, DeepEyes, VipAct）**：通过外部工具进行视觉操作，本文的贡献在于系统性比较在线构造 vs 预生成中间件、文本 vs 视觉路径的相对收益，并提出 Mosaic 这一统一 harness。
3. **多图基准（BLINK, M4Bench, MMIU, Mantis）**：这些基准混合语义/时空/感知任务，本文指出其无法隔离细粒度 grounding 需求，从而提出专门针对视觉重表征评估的 MosaicBench。
4. **VTool-R1, OpenThinkIMG, DeepEyes-RL, PyVision-RL**：近期尝试用 RL 训练视觉工具代理，本文与之区别在于：**仅使用准确性+格式奖励**（无工具序列奖励），仍能涌现多步工具组合，并系统分析了解决策略的多样性。
5. **MANTIS**：通过 interleaved instruction tuning 训练多图能力，属于"encode-then-reason"范式，本文强调中间表征的主动构造才是关键缺口。

## 局限性与未来方向
1. **模型规模有限**：实验主要在 8B/32B 模型上进行，更大规模模型的表现及工具使用模式仍有待探索。
2. **奖励信号较为简单**：仅用准确性和格式奖励，未引入工具效率或中间步骤质量奖励，可能限制 agent 在复杂长轨迹任务中的表现。
3. **V-Re² 在部分任务上劣于 PV-Re²**，说明在线视觉重表征的构造策略仍有提升空间，如何降低在线推理难度值得进一步研究。
4. **MosaicBench 任务类型以视觉 grounding 为主**，对高层语义推理任务的覆盖不足，难以全面反映多模态理解的各个维度。
5. **工具调用上限为 10 步**，对于需要更多步骤的复杂任务可能不足。
6. **RL 训练时间较长（218 步）**，对计算资源有一定要求，限制了在更大模型上的可扩展性。

## 研究启发与可借鉴点
1. **"重表征"的理论框架具有高度迁移价值**：将视觉中间件构造抽象为条件分布 $p_\theta^m(z|x)$，可将同一框架迁移到其他多模态场景（视频理解、文档分析、图表推理）。
2. **仅用 accuracy + format 奖励驱动工具学习的思路简洁有效**：无需精心设计的稠密工具奖励或 SFT 预热，值得在工具增强型 agent 训练中复现和扩展。
3. **PV-Re² vs V-Re² 的对照实验设计**揭示了"预生成中间件"与"在线构造"的本质差异，为后续研究如何降低在线推理难度提供了清晰的评估视角。
4. **Prefix answerability 探测方法**（冻结轨迹前缀、观察模型能否从中间状态正确回答）是一种评估"中间表征是否真正有用"的精细手段，可直接迁移到任何 tool-use agent 研究中。
5. **Mosaic 的十种可组合操作的持久化工作空间设计**（每个操作产生新资产，不修改输入，支持叠加透明区域）为构建通用视觉工具集提供了参考架构。

## 关键术语表
**Multi-image re-representation**：将分散在多张图像中的视觉证据重组为中间记录（文本或视觉形式），以支持下游推理的过程。
**Mosaic**：本文提出的多图视觉 harness，提供持久化工作空间和十种可组合图像操作（裁剪、旋转、拼贴、像素差分等）。
**V-Re² / T-Re² / PV-Re²**：三种重表征设置——在线视觉重表征、自由文本思维链、预生成视觉中间件，共享相同最终答案格式。
**Closed-evidence re-representation**：模型与工具仅依赖原始输入，不引入外部证据的设定，此时中间记录与正确答案关于原始输入条件独立。
**GRPO（Group Relative Policy Optimization）**：本文使用的 RL 训练算法，无需 critic 网络，对每组 rollout 进行相对归一化 Advantage 计算。
**MosaicBench**：本文提出的细粒度多图理解基准，含 560 条样本和 28 类任务，专注于 grounding、分辨率、方向、精确比较等视觉挑战。
**Answerability（可答性）**：在固定轨迹前缀条件下，模型能否从当前上下文正确回答问题，用于度量中间表征的信息价值。
**Progression / Self-correction / Post-stabilisation continuation**：三种问题解决模式——单调推进至正确、中途偏离后恢复、到达正确答案后继续多余操作。

## 可复现要素
- **MosaicBench 数据集**：论文声明代码和数据将在 https://github.com/gengyuanmax/Mosaic 开源。
- **训练数据**：12,800 条候选样本经过 rollout 筛选后保留 3,509 条用于训练，覆盖 32 类任务（每类 400 条候选），训练集与 MosaicBench 在样本/源/图像三级去重叠。
- **模型基座**：Qwen3-VL-8B-Instruct。
- **RL 超参**：GRPO，每提示 8 个 rollout，AdamW，学习率 $1 \times 10^{-6}$，batch size=32，1 epoch，KL 系数=0，entropy 系数=0，最大 prompt 长度 16,384 tokens，最大 response 长度 20,480 tokens，每 episode 最多 10 步 assistant turn。
- **硬件**：4 × NVIDIA H100 GPU。
- **代码/权重**：论文未提及代码和模型权重是否已发布，仅声明"Code and data will be released"，待 GitHub 仓库上线后可验证。
