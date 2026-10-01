---
title: "Summarize-Before-Grounding-Query-Guided-Chunk-Condensation-f"
source: https://arxiv.org/pdf/2609.34598v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 22:11:41"
field: "多模态时序定位"
keywords: ["temporal grounding", "large vision-language models", "reinforcement learning", "long video understanding", "chunk condensation", "query-guided retrieval"]
innovations: ["提出先总结后定位的两级块压缩框架", "设计查询引导潜在摘要与关联检索机制", "引入长度感知梯度门控降低长视频训练显存"]
benchmarks: ["Ego4D-NLQ", "TACoS", "Charades-STA", "ActivityNet-Captions"]
---

# 论文速读：Summarize-Before-Grounding-Query-Guided-Chunk-Condensation-f

## 一句话总结
论文提出 **SumGround** 框架，通过“先总结后定位”的两级块压缩策略，将长视频按 chunk 顺序处理并生成查询引导的潜在摘要（KV状态），再利用关联检索筛选相关摘要进行最终时间定位，显著提升了长视频时序定位（VTG）性能并降低了训练显存开销。

## 研究问题与动机
1. **长视频冗余信息干扰**：VTG任务中查询事件仅占视频一小部分，大量冗余视觉信息会误导大视觉语言模型（LVLM），导致定位边界不精确。
2. **显存与证据获取的矛盾**：密集帧采样虽能保留细粒度证据，但会导致显存爆炸；稀疏采样虽可行，却容易遗漏关键证据。
3. **现有RLVR方法的局限**：基于可验证奖励的强化学习（RLVR）训练通常依赖稀疏采样，未解决如何从密集长视频内容中有效挖掘和聚合查询相关证据的问题。
4. **高效长视频处理的瓶颈**：如何在保证证据完整性的同时，避免直接处理全部视觉token带来的高昂计算与存储成本，是现有工作尚未充分探索的方向。

## 核心贡献（创新点）
1. **提出“Summarize-before-grounding”框架**：首次将查询引导的块压缩机制引入VTG，通过顺序chunk处理与潜在摘要累积实现证据聚合，而非传统密集采样或稀疏采样。
2. **设计两级块压缩技术**：引入查询引导的潜在摘要（以KV状态表示）进行信息压缩，并结合关联摘要检索（基于关联分数）动态筛选最可能包含事件区间的摘要，形成聚焦上下文。
3. **开发长度感知梯度门控模块**：根据输入视频长度选择性阻断视觉token的梯度反向传播，在长视频上大幅降低显存占用，同时保留短视频上的端到端视觉到摘要学习能力。
4. **基于RLVR的双目标优化机制**：采用GRPO分别对关联摘要检索和时序定位任务分配独立奖励，联合优化摘要质量与最终边界预测，且无需引入额外参数化模块。

## 方法详解
- **视频分块与顺序处理**：将时长为T的视频均匀划分为K个连续chunk，每个chunk包含时间区间[τ_{i-1}, τ_i)内的帧，并在每帧后插入时间戳文本。
- **查询引导潜在摘要**：在每个chunk处理后，追加一个查询引导的总结提示词（如“总结视频片段以准确定位事件‘切洋葱’的时间区间”），将其KV状态作为该chunk的潜在摘要S_i，压缩冗余视觉token；第i个chunk的输入X_i由系统提示、历史摘要状态和当前chunk组成。
- **关联摘要检索**：在每个chunk处理后，通过关联提示词p_assoc的下一个token logits计算关联分数Δ_i = z_i[yes] - z_i[no]，经过softmax归一化得到分布w_i，从中采样或选取锚点chunk，并扩展选取连续正关联分数的chunk集合作为最终grounding的上下文。
- **长度感知梯度门控**：定义指示函数γ = I[N ≤ W]（N为帧数，W为阈值），当N > W时对视觉token状态应用stop-gradient，仅保留前向传播用于摘要生成；前向过程中仍使用完整视觉信息。
- **RLVR训练目标**：基于GRPO，每个训练样本生成G个rollout，分别计算关联奖励（基于与真实区间的IoU）和定位奖励（预测区间与真实区间的IoU），通过组相对优势归一化后联合优化：max_θ (I_tg + λ I_assoc)。

## 实验与结果
- **数据集**：训练集由NaQ、DiDeMo、QuerYD、HiRest、COIN、Momentor、YouCook2随机采样共42.5K样本组成；测试集为四个标准VTG基准：Charades-STA（短）、ActivityNet-Captions（短）、TACoS（长）、Ego4D-NLQ（长）。
- **评估指标**：R1@.3、R1@.5、R1@.7、mIoU。
- **基线对比**：包括监督/SFT方法（UniVTG、VTG-LLM、TimeSuite、VideoMind、UniTime）和RLVR方法（Time-R1、TimeLens），其中后两者在同一训练数据下重新训练比较。
- **主要结果**：在长视频基准上，SumGround在Ego4D-NLQ的R1@.3达到16.42（较TimeLens†提升约6%，较Time-R1†提升约14%），TACoS的R1@.3达到52.96（较TimeLens†提升约8%）；在短视频基准Charades-STA的R1@.5达到39.57，ActivityNet-Captions的mIoU达到39.54，整体优于所有基线。
- **效率优势**：相比Full-Dense策略，SumGround峰值显存增加仅3.95GB（减少80%），推理延迟降低16%，而mIoU从1.74提升至11.83。

## 相关工作脉络
1. **时序定位的LVLM方法**：UniVTG、VTG-LLM等通过生成式时间戳预测提升定位能力，但未解决长视频中冗余证据的筛选问题。
2. **长视频处理技术**：MovieChat等使用记忆token或摘要token压缩上下文，但通常需要额外模块或大量训练，且未针对精确时间边界设计。
3. **检索增强方法**：LCIRC、ReFrag等通过chunk压缩或选择性检索减少有效上下文，但同样非任务专用设计。
4. **RLVR训练范式**：Time-R1、TimeLens利用重叠奖励优化LVLM，但均依赖固定稀疏采样，未探索从密集内容中动态聚合证据。
5. **定位引导训练**：TimeSuite、VideoMind等采用指令微调或智能体工作流程，可能削弱基础模型通用能力，且受限于初始提案质量。
6. **本文定位**：首次在VTG中结合查询引导的潜在摘要与关联检索，在无需额外模块的情况下实现长视频证据的高效聚合与选择，并通过RLVR统一优化检索与定位两个子任务。

## 局限性与未来方向
- **检索覆盖率瓶颈**：在Ego4D-NLQ上仅52%的样本能完全覆盖真实标注区间，说明当前关联检索仍有提升空间。
- **阈值W的敏感性**：长度感知梯度门控依赖固定帧数阈值，不同视频长度分布可能影响最优值选择。
- **仅使用单一LVLM**：实验基于Qwen2.5-VL-7B-Instruct，其他架构或更大模型的泛化性尚未验证。
- **未来方向**：可扩展至更复杂的视频理解任务（如事件检测、因果推理），或引入更细粒度的时空压缩机制；探索自适应阈值或动态chunk划分策略。

## 研究启发与可借鉴点
1. **潜在摘要作为KV状态**：利用提示词的KV状态压缩视觉信息，避免了额外模块和文本生成开销，可迁移至其他多模态长上下文任务（如视频问答、文档分析）。
2. **双目标RLVR优化**：分别设计检索奖励与定位奖励，通过组相对优势归一化联合训练，为多阶段任务提供了统一的强化学习框架。
3. **长度感知的梯度门控**：根据输入规模动态决定是否阻断梯度，平衡了显存效率与学习能力，适用于任何涉及可变长度输入的长序列模型训练。
4. **关联分数驱动的选择性检索**：基于logit margin的简单关联估计替代复杂检索器，可在保持精度的同时显著降低计算开销。
5. **因果摘要累积设计**：当前chunk摘要 attends to 先前摘要，形成了层次化的信息聚合，值得借鉴到多步推理或时序建模任务中。

## 关键术语表
- **Video Temporal Grounding (VTG)**：视频时序定位，根据自然语言查询在视频中定位对应事件的时间区间。
- **Large Vision-Language Models (LVLMs)**：大视觉语言模型，结合视觉编码器与语言模型的多模态基础模型，具备跨模态推理能力。
- **Reinforcement Learning with Verifiable Rewards (RLVR)**：可验证奖励强化学习，利用任务明确的奖励信号（如时间重叠IoU）对模型进行后训练优化。
- **Group Relative Policy Optimization (GRPO)**：组相对策略优化，一种基于group-wise归一化优势的强化学习算法，用于策略梯度更新。
- **Query-Guided Latent Summary**：查询引导潜在摘要，通过特定提示词的KV状态编码的、与查询相关的视频片段压缩表示。
- **Associative Summary Retrieval**：关联摘要检索，基于每个chunk的关联分数选择最可能与查询相关的摘要序列作为最终定位上下文。
- **Length-Aware Gradient Gating**：长度感知梯度门控，根据输入视频长度动态决定是否阻断视觉token梯度的机制，以平衡显存与学习效果。
- **Intersection-over-Union (IoU)**：交并比，衡量预测时间区间与真实区间重叠程度的指标，常用于定位任务评估。

## 可复现要素
- **数据集**：训练集来自公开数据集（NaQ、DiDeMo、QuerYD、HiRest、COIN、Momentor、YouCook2）的随机采样；测试集为公开基准（Charades-STA、ActivityNet-Captions、TACoS、Ego4D-NLQ）。
- **代码/权重**：论文未提供开源代码与模型权重声明；使用Qwen2.5-VL-7B-Instruct作为基础模型。
- **关键超参**：chunk大小W=42帧，最小检索尺寸M=3，GRPO rollout数G=8，目标权重λ=0.1，学习率1e-6，训练2个epoch，batch size=16（8 GPU×micro-batch 1×gradient accumulation 2），使用DeepSpeed ZeRO-3。
