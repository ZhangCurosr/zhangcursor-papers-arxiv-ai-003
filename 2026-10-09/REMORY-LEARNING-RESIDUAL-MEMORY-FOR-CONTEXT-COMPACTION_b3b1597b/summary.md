---
title: "REMORY-LEARNING-RESIDUAL-MEMORY-FOR-CONTEXT-COMPACTION"
source: https://arxiv.org/pdf/2610.11287v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-01 07:18:59"
field: "长上下文 Agent 记忆与压缩"
keywords: ["Residual Memory", "Context Compaction", "Long-Horizon Agent", "Reverse KL Distillation", "Soft Memory Tokens", "Frozen Actor", "DFlash"]
innovations: ["提出在文本摘要后拼接残差软 token 序列的连续补充机制", "冻结 Actor 下两阶段 reverse KL 蒸馏训练记忆网络", "同预算文本对照证明连续补充优于等长生成文本"]
benchmarks: ["SummHay", "AutomationBench", "JobBench", "Terminal-Bench 2.1", "BrowseComp"]
---

# 论文速读：REMORY-LEARNING-RESIDUAL-MEMORY-FOR-CONTEXT-COMPACTION

## 一句话总结
本文提出 REMORY，一种通过神经记忆网络学习「残差软记忆 token」来补充文本摘要的方法，使冻结的 LLM 在上下文压缩后仍能近似完整历史下的决策行为；在 SummHay 上提升了源引用准确率，在 BrowseComp 和 Terminal-Bench 等长程 Agent 基准上实现 3–7.6 分的稳定增益，同时降低重复工具调用和错误率 17–50%。

## 研究问题与动机
- **上下文压缩的单点局限**：现有长程 Agent（如 Claude Code、Codex CLI）在 `/compact` 后仅保留一段文本摘要，但摘要无法覆盖后续每一步决策所需的全部细粒度线索，尤其是随着任务展开，早期上下文的相关性会动态变化。
- **信息可用性与窗口约束的张力**：Agent 上下文窗口有限（通常 20–100 万 token），历史积累后必须压缩；若只靠摘要作检查点，推理时无法同时持有完整交互记录。
- **与人类记忆机制的对照**：人类工作记忆容量极小，却可依赖海马索引和长期记忆支持数月/年的复杂项目——提示将「显式检查点 + 神经连续性表征」配对作为 Agent 记忆架构的合理方向。
- **无需预标注的信息缺口**：论文不假设压缩后缺失的信息可被提前枚举或识别，而是让模型在后续预测过程中动态学习摘要未能覆盖的那部分残余信息。

## 核心贡献（创新点）
1. **残差记忆形式化**：提出在摘要后拼接有界长度的软记忆 token 序列，形成沿序列维度的残差连接，使摘要保持可读性同时补充连续表征。
2. **冻结 Actor 蒸馏训练**：两阶段训练——Stage I 做源文本重建以保留基础信息；Stage II 用摘要条件化后的续写分布误差训练记忆编码器，梯度只流过记忆网络参数，Actor 保持冻结。
3. **层级压缩与边界绑定**：每 1,024 个源位置产 64 个软 token，最终上限 4,096 槽，并在每次压缩边界处重新构造，使记忆跨多次 compaction 持续携带信息。
4. **端到端 Agent 增益验证**：在 SummHay、AutomationBench、JobBench、Terminal-Bench 2.1、BrowseComp 五个基准上统一测试，证明残差记忆在几乎不增加输入规模的前提下提升后续动作质量和工具调用效率。
5. **与已有基线的差异定位**：区别于 AutoCompressors / Gist Tokens / ICAE 等纯连续压缩方案，REMORY 明确保留文本摘要为 readable checkpoint，并维持既有档案检索接口，仅在摘要之上叠加一个可学习的连续补充项。

## 方法详解
- **记忆构造**：在压缩边界 $i$，给定历史 $H_i$（含前一次摘要与残差记忆）和新摘要 $S_i$，记忆网络 $\mathcal{C}_\phi$ 产出 $M_i$，满足 $|M_i| \le m$；$M_i$ 是冻结 Actor 输入空间的连续向量序列。
- **残差拼接**：将 $[S_i; M_i]$ 串联后送入 Actor，构成序列维度的残差类比——摘要提供可读主干，软 token 沿序列维度提供附加预测信号。
- **架构选择**：Qwen3.8-27B 使用 DFlash 架构，GLM-5.3-Flash 使用 DFlash-2（分组动态卷积）；可训练参数分别为 1.94B 与 1.502B，输出为连续 memory 而非 vocab token。
- **Stage I 重建**：Teacher 接收原文本 + 复述指令；Student 将原文替换为软记忆接收相同指令；用 MMFineReason-Full-2.3M 的文本字段训练记忆编码基础源信息的能力。
- **Stage II 摘要条件化续写**：Teacher 接收原始交互历史；Student 接收固定摘要与从该历史构造的残差记忆；两者预测相同的下一步助手响应；记忆编码器不看到未来响应，避免 leakage。
- **损失函数**：Student 到 Teacher 的反向 KL，
  $$\mathcal{L}(\phi) = \frac{\sum_j w_j D_{\mathrm{KL}}(q_j \| p_j)}{\sum_j w_j},$$
  其中 $p_j = p_\theta(\cdot|C_{\text{full}}, y_{<j})$，$q_j = p_\theta(\cdot|C_{\text{mem}}, y_{<j})$，只对 $\phi$ 求梯度。
- **层级压缩**：每 1,024 源位置生成 64 软 token，最终上限 4,096 槽，保留源顺序；中间表示可能大于输出槽位。

## 实验与结果
- **数据集与设置**：SummHay（92 query、10 个集合）、AutomationBench（600 tasks）、JobBench（65 tasks）、Terminal-Bench 2.1（89 tasks）、BrowseComp（1,266 tasks）。
- **SummHay 引用 F1**：Summary 56.98 → Summary + REMORY 61.03（+4.04），Joint score 39.84 → 43.39（+3.55），Coverage 基本不变（67.55 → 67.95）；同 budget 的 Summary++ 仅 56.94，说明连续补充比同长度生成文本更有效。
- **Qwen3.8-27B 端到端**：AutomationBench 35.5 → 45.3（+9.8）；JobBench 33.4 → 41.0（+7.6）；Terminal-Bench 71.9 → 76.4（+4.5）；BrowseComp 74.0 → 77.0（+3.0）。
- **GLM-5.3-Flash 端到端**：Terminal-Bench 84.3 → 87.6（+3.3）；BrowseComp 84.9 → 89.0（+4.1），逼近前沿参考分数。
- **工具调用效率**：重复工具输出下降 17.9–29.8%，工具错误下降 25.3–49.7%（四组 actor-benchmark）。
- **输入规模对比**：原始平均 96,872 token；仅摘要 1,393 token；摘要 + 残差记忆 5,024 token（占原始 5.2%），Joint score 接近完整上下文。
- **首次动作偏好**：488 对 Qwen BrowseComp 轨迹中，残差记忆动作赢得 70.7%（345/488），平 4.3%，输 25.0%。
- **Live Trading 案例**：-11.86% ROI（无残差）→ +5.11% ROI（有残差）；生成成本从 $303.81 降至 $170.30。

## 相关工作脉络
- **ReadAgent / MemGPT**：将 episode gist 与原文 passage 分开，或区分工作记忆与档案记忆；REMORY 保留文本 checkpoint 与检索接口，仅在摘要后叠加连续补充，不替换已有档案通路。
- **ReSum / AgentFold**：前者用周期性摘要做长程搜索，后者在 agent 轨迹内学习多尺度上下文管理；REMORY 明确将补项限定为摘要条件的残差向量序列，不与多尺度管理竞争。
- **AutoCompressors / Gist Tokens / ICAE**：直接学习长文本的连续压缩表征；REMORY 的差异在于不取代摘要，而是在摘要之外学习“摘要未能覆盖的那部分残差”。
- **LCLMs**：端到端可扩展的编码器-解码器压缩并支持选择性展开；REMORY 更轻量、冻结 Actor、用 reverse KL 对齐预测分布，不需要重构完整词表。
- **DFlash / DFlash-2**：块扩散快速推测解码架构，被本文用作记忆网络主干，体现与生成加速框架的兼容性。
- **分布匹配 / Recurrence**：Wingate 等用对比条件做 prompt 压缩，Transformer-XL 用 recurrence 跨段保留信息；本文把分布匹配视角应用到“摘要之后的续写 gap”。

## 局限性与未来方向
- **软 token 可解释性弱**：连续表征难以直接阅读，理解其携带的具体线索仍需辅助分析或可视化。
- **训练成本与规模**：两阶段训练需在真实或合成轨迹上运行 teacher-student 对比，Stag II 使用 Open-PerfectBlend 等大规模数据，训练开销较大。
- **记忆容量上限固定**：4,096 槽对超长历史是否足够仍需验证；对极难任务（如 Live Trading 的 5,336 步 vs 3,013 步）可能仍需配合档案检索。
- **模型特定适配**：目前针对 Qwen3.8-27B 与 GLM-5.3-Flash 分别适配不同 DFlash 变体，泛化到其他架构的迁移成本未充分讨论。
- **实时增量更新缺失**：当前是在压缩边界重构造记忆，尚不确定是否支持在线 incremental update 或在非压缩时刻持续写入。

## 研究启发与可借鉴点
- **残差拼接思路可移植**：将连续补充视为摘要沿序列维度的残差连接，这一抽象可推广到 RAG、agent checkpoint、代码生成续写等场景。
- **冻结 Actor + reverse KL 蒸馏**：保留主模型参数不动、只训接口网络，既降低安全风险又便于工程集成；可用于各种长上下文压缩与续写代理任务。
- **同预算对比实验设计**：用 Summary++（等 position 预算的文本补充）作为强对照，清晰证明连续补充的有效性优于同长度生成文本，这种对照值得在后续工作中沿用。
- **首次动作偏好评估**：采用 rubric-based planning evaluation 衡量 compact 后第一步动作质量，为 compaction 策略提供细粒度诊断信号。
- **结合团队方向的创新机会**：若团队做多模态长上下文或工具调用代理，可将残差记忆扩展到视觉/代码块等多模态 token 序列，或在 agent planning 中引入多级残差（摘要残差 + 子目标残差）。

## 关键术语表
- **Residual Memory（残差记忆）**：在文本摘要后拼接的有界软 token 序列，作为摘要在预测分布上的连续补充，形成序列维度的残差连接。
- **Stage I Reconstruction（重建阶段）**：让记忆网络学习保留源文本信息的初始化阶段，student 用软记忆替代原文本复述源内容。
- **Stage II Summary-Conditioned Continuation（摘要条件化续写阶段）**：在 Stage I 基础上，teacher 看完整历史、student 看摘要 + 残差记忆，共同预测下一步响应，用 reverse KL 训练。
- **Frozen Actor Distillation（冻结 Actor 蒸馏）**：训练过程中 Actor 参数 $p_\theta$ 完全冻结，只更新记忆网络参数 $\phi$。
- **Hierarchical Compression（层级压缩）**：每 1,024 源位置生成 64 软 token，最终上限 4,096 槽，保留源顺序。
- **Reverse KL Alignment（反向 KL 对齐）**：以 student 预测分布 $q_j$ 对 teacher 分布 $p_j$ 的最小化目标驱动记忆网络训练。
- **Summary++（同等预算文本补充）**：使用与残差记忆相同额外 position 预算而生成的文本补充，用于公平对照。
- **Compact Boundary（压缩边界）**：Agent 触发 `/compact` 时的节点，记忆在此处按当前历史和摘要重新构造。

## 可复现要素
- **数据集**：SummHay（92 queries）、MMFineReason-Full-2.3M（Stage I）、Open-PerfectBlend（Stage II 推理/代码轨迹）、AutomationBench、JobBench、Terminal-Bench 2.1、BrowseComp——部分为公开基准，部分需按官方协议获取。
- **代码/权重**：论文未提供开源链接；记忆网络基于 DFlash / DFlash-2 架构，Actor 为 Qwen3.8-27B 与 GLM-5.3-Flash（开源模型但权重获取依官方渠道）。
- **关键超参**：每 1,024 源位置 → 64 软 token；记忆上限 4,096 槽；Qwen 记忆网络 1.94B 参数；GLM 记忆网络 1.502B 参数；Actor 上下文窗口 Qwen 262,144、GLM 1,000,000 token。
