---
title: "WorldSR-Harnessing-World-Knowledge-Search-for-Grounded-Image"
source: https://arxiv.org/pdf/2610.28423v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-06 08:46:16"
---

# 论文速读：WorldSR: Harnessing World Knowledge Search for Grounded Image Super-Resolution

## 一句话总结
WorldSR 提出了一种代理式（Agent-style）超分框架，将输入图像解构为实体级语义区域，自动从网络与本地知识库检索对齐的参考图，并结合多参考超分模型与迭代细化机制，以显式世界知识约束取代隐式生成先验，有效抑制幻觉并提升重建图像的语义一致性与感知质量。

## 研究问题与动机
- 现有生成先验超分方法（如 StableSR、OSEDiff 等）依赖预训练 T2I 模型的隐式知识，参数化先验具有随机性且与基础模型强绑定，难以随现实世界数据更新，易产生视觉合理但语义错误的幻觉细节。
- 现有参考式超分方法（如 SeeSR+ReFIR、AdaRefSR 等）通常依赖单一或少量预采集参考图，在复杂多实体场景中缺乏区域级对应关系，容易引发结构不一致。
- 直接将检索增强生成（RAG）迁移至图像超分面临挑战：整体图像级相似度无法处理复杂场景中的细粒度实体级变化与局部退化差异。
- 需要一种能够自动识别实体、动态检索外部真实视觉知识，并以细粒度约束方式注入超分过程的端到端自动化框架。

## 核心贡献（创新点）
- 提出实体对齐的世界知识检索框架：利用 MLLM 进行 ROI 规划与边界框预测，将超分问题从“全局图像级重建”重构为“实体级语义 grounding”。
- 设计双引擎检索与三级过滤机制：融合 Google Lens 网络反向搜索与 SigLIP-2 本地向量检索，并通过 IQA 质量评估、MLLM 无关区域剪枝与多模态 reranker 实现高质量参考筛选。
- 构建世界知识驱动的超分网络：引入参考质量细化（RQR）单步 DiT 模块与双 LoRA 架构，显式解耦 LR 域与 Ref 域特征，通过序列拼接与全注意力实现多参考知识高效注入。
- 设计带记忆验证器的迭代细化循环：利用中间超分结果作为视觉反馈，通过 IoU 缓存路由避免冗余检索，逐步收敛至稳定输出。

## 方法详解
- **Agentic ROI Planner**：以 Qwen3.5-35B-A3B 为核心感知工具，通过结构化 Prompt 提取粗粒度实体类别（People/Animals/Landmarks/Vehicles 等）、保守命名及高置信度边界框 $B_i$，裁剪得到图像块 $P_i$。迭代过程中，第 $n$ 步输入可为初始 LR 图 $I^{(0)}$ 或上一轮回退的超分结果 $I^{(n-1)}$，以此逐步抑制幻觉。
- **Memory Verifier**：Memory Bank 缓存历史检索结果 $(B_j, R_j^*)$。决策控制器计算新查询 ROI 与缓存 ROI 的 IoU，若最大 IoU $s_i^* > s_{IoU}=0.5$ 则复用缓存知识；否则触发检索。当所有新查询均命中缓存时判定迭代收敛，输出上一轮 SR 结果。
- **World Knowledge Searcher**：采用双路检索合并为候选集 $\{R_i^j\}_{j=1}^K$。Web 端使用 Playwright 调用 Google Lens 获取 top-$K_{web}$ 结果并下载高清原图；本地库使用 SigLIP-2 编码后计算余弦相似度检索 top-$K_{local}$。
- **World Knowledge Filter**：① Quality-Aware Filtering：使用 RALI 与 Q-Insight 打分，低于阈值 $s_{filter}=3.8$ 直接丢弃；② Irrelevant Region Pruning：MLLM 联合输入查询图与候选图，输出语义对齐的裁剪 bbox 去除无关背景；③ Semantic-Aware Ranking：使用 Qwen3-VL-Reranker 计算语义相似度，选取 top-1 作为最优参考 $R_i^*$。
- **World-Grounded Image Super-Resolution Model**：
  - **Reference Quality Refinement (RQR)**：基于 Flux.2-dev DiT 骨干与 RQR LoRA ($\theta_{RQR}$)，通过速度预测实现单步增强：$v_{ref}^i = \mathcal{F}(z_{ref}^i; \theta_{RQR})$，$\hat{z}_{ref}^i = z_{ref}^i - t^* \cdot v_{ref}^i$，其中 $t^*=0.5$。
  - **Reference-Guided Super-Resolution**：LR 潜码 $z$ 与细化参考潜码 $\{\hat{z}_{ref}^i\}$ 展平后沿序列维度拼接。采用 Dual-LoRA 架构（$\theta_{LR}$ 处理 LR token，$\theta_{Ref}$ 处理 Ref token），共享 MLP 基础权重但通过低秩适配器解耦域间分布，结合 OminiControl 风格适配器调制策略实现跨模态全注意力交互：$\text{Attention}(Q,K,V) = \text{Softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$。
  - **训练目标**：$\mathcal{L} = \mathcal{L}_2 + \lambda_p \mathcal{L}_p + \lambda_g \mathcal{L}_g$，其中 $\lambda_p=5, \lambda_g=0.5$，$\mathcal{L}_p$ 为 OC-EA-DISTS 感知损失，$\mathcal{L}_g$ 为 DINOv3 对抗损失。训练时随机采样 0~$N_{max}=5$ 张参考图进行数据增强，仅更新 LoRA 参数（rank=16），采用 NF4 量化 + QLoRA + Prodigy 优化器，在 8×NVIDIA H800 上训练 30k 步。

## 实验与结果
- **数据集与基线**：在 RealLQ250、Face、Bird、W
