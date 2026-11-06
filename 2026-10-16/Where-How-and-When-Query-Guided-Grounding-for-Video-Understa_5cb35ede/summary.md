---
title: "Where-How-and-When-Query-Guided-Grounding-for-Video-Understa"
source: https://arxiv.org/pdf/2610.21422v1.pdf
model: agnes-2.5-flash
chunks: 15
summarized_at: "2026-11-06 02:59:10"
field: "视频理解与多模态 grounding"
keywords: ["video understanding", "few-shot detection", "moment retrieval", "visual relationship grounding", "open-vocabulary", "dynamic time warping", "query-guided"]
innovations: ["跨帧选择性特征传播的少样本视频目标检测框架", "双阶段可微动态时间规整对齐的视频时刻检索方法", "基于T-partite随机场与信念传播的少样本视觉关系定位"]
benchmarks: ["FSVOD-500", "FSYTV-40", "VidOR", "VidVRD", "ActivityNet-VRL", "SportsMoments"]
---

# 论文速读：Where-How-and-When-Query-Guided-Grounding-for-Video-Understanding

## 一句话总结
本论文提出一套**三层次（Where/How/When）查询引导的视频接地框架**，分别在对象级、关系级和时刻级三个粒度上实现少样本/开放集条件下的可泛化视频理解，在 FSVOD、VidVRD、ActivityNet-VRL 等基准上均达到 SOTA。

---

## 研究问题与动机

1. **可扩展性不足**：传统监督方法依赖对每个可能场景的详细标注（物体类别、关系三元组、动作），面对 YouTube 每天 >20M 视频上传的海量数据无法扩展；object-relation 组合空间指数增长。
2. **开放集泛化缺失**：真实环境持续涌现 unseen 对象/关系/动作，闭集假设下设计的方法难以适应；现有 open-vocabulary 方法（如 Video-OWLViT）缺乏对象感知的时间建模，导致跨帧定位碎片化。
3. **超越静态分析**：视频引入运动模糊、遮挡、形变、视角变化等时序复杂性；逐帧处理忽略时间上下文，导致轨迹碎片化、边界不稳定；需要时空联合推理。
4. **细粒度语义对齐困难**：视频时刻检索中，文本查询难以描述速度、执行风格等细粒度视觉线索，催生 Vid2VidMR 范式但缺乏显式时序对齐。

---

## 核心贡献（创新点）

1. **QDETRv（Query-guided DETR for Videos）**：将 DETR 扩展至视频域，通过 query-context fusion 和 memory module 实现一次定位未见对象；引入无监督预训练（合成轨迹 + recurrent queries + feature-reconstruction），与仅用 CLIP 编码的开集方法本质不同。
2. **Temporal Object-Aware Vision Transformer（TOViT）**：首次在 Few-shot Video Object Detection（FSVOD）中引入大规模预训练 vision-language 模型（OWL-ViT-L/16），并通过**跨帧选择性特征传播筛选机制**（仅高置信度 embedding 传递至下一帧）实现时序一致性，优于专用 FSVOD 方法与 adapted temporal trackers。
3. **基于 T-partite 随机场的少样本视觉关系定位**：将 subject-predicate-object 三元组定位建模为结构化图优化问题，通过 episodic meta-learning 训练 query-conditioned translational embeddings 并 Belief Propagation 消息传递，本质区别在于直接建模谓词语义交互而非仅靠分类头。
4. **MATR（Moment Alignment Transformer）双阶段 DTW 对齐**：在 Transformer 内集成可微动态时间规整（pre- 和 post-fusion alignment），支持视频到视频时刻检索并可扩展至文本查询，与 Moment-DETR/QD-DETR 等仅靠 cross-attention 的文本 VMR 方法形成对比。
5. **系统化三维度视频理解框架**：首次在单一论文中系统性覆盖 Where（对象）、How（关系）、When（时刻）三个接地粒度，桥接检测级与事件级推理鸿沟。

---

## 方法详解

### 3.1 对象级接地（Where）

**QDETRv**（第 3 章，AAAI 2024）
- 扩展 DETR：query 向量与视频帧时序上下文做 cross-attention；引入 memory module 维持跨帧一致性
- 无监督预训练：合成对象轨迹 + recurrent queries + feature-reconstruction 目标，减少对标注依赖

**TOViT / Obj-Wise Vision Transformer**（第 4 章，AAAI 2026）
- 视觉编码器：采用 **OWL-ViT-L/16**（CLIP 预训练）替代 CNN backbone
- Support Set Encoding：每张 support image 提取 patch 级特征 + objectness scores，取最高分 patch 作为 object-centric 表示 $\mathbf{z}_{i,j}$，聚合 K 个样本得 class prototype $\mathbf{z}_i$（均值）
- **Object-Aware Temporal Frame Decoder**：筛选机制——仅当上帧分类置信度 $c^p > \tau$ 时，将对应 embedding 作为下一帧 key-value；跨帧 attention：$\mathbf{A}_t = \mathrm{Softmax}(\mathbf{F}_t \mathbf{V}_{t-1}^{det\,T} / \sqrt{D})$，更新 $\hat{\mathbf{F}}_t = \mathbf{F}_t + \mathbf{A}_t \mathbf{V}_{t-1}^{det}$
- Few-shot Heads：cosine similarity 分类（$\mathbf{v}_t^p$ vs $\mathbf{z}_i$）+ MLP 回归 $(x,y,w,h)$
- 联合损失：$\mathcal{L} = \sum_t [\lambda_{cls}\mathcal{L}_{cls} + \lambda_{box}\mathcal{L}_{box}]$（cross-entropy + L1 + generalized IoU）

### 3.2 关系级接地（How）

**Few-Shot Referring Relationship Grounding**（第 5 章，CVPR 2023）
- 问题：给定 $q = \langle s, p, o \rangle$ 和少量支持视频（含相同 predicate），在测试视频中时空定位 subject 和 object
- 形式化：优化 over $\tau$-partite 随机场的目标函数，顶点 = 候选 subject-object 对，势函数 = 帧级检测势 + episodic meta-training 学习的关系相似性势
- **Metric Learner**：triplet loss 学习 deep metric，正样本为同 predicate pair proposals
- **GSA（Global Semantic Aggregation）**：可学习注意力权重融合所有帧 I3D 特征，纳入长程上下文
- **LLA（Local Localization Aggregation）**：聚焦时间窗口内邻帧空间与 ROI 特征，处理遮挡/部分可见
- **Belief propagation message passing**：高效求解结构化优化，生成 subject-object 轨迹

### 3.3 时刻级接地（When）

**MATR（Moment Alignment Transformer）**（第 6 章，ICCV 2025 / CVIU 2026）
- 任务：Vid2VidMR（用短视频片段作为查询，定位长视频中语义相似片段），可扩展至 text-to-video
- **双阶段动态时间规整（DTW）**：
  - pre-fusion alignment：Transformer 编码前增强查询与目标语义相关性
  - post-fusion alignment：融合后细化目标特征，提升时序对应精度
- Decoder：$k$ 层 Transformer，可学习 query 向量 $\mathbf{Q} \in \mathbb{R}^{l \times d}$；cross-attention 中 encoder 输出对齐特征作 KV
- Prediction Heads：前景分类（3 层 $1\times3$ conv + sigmoid）+ 边界预测（输出左/右偏移 $\hat{d}_i^L, \hat{d}_i^R$）
- 损失函数：
  - $\mathcal{L}_{fg}$：二元交叉熵
  - $\mathcal{L}_{seg}$：smooth L1 + Generalized IoU（仅 foreground）
  - $\mathcal{L}_{align}^{\text{pre/post}}$：对齐代价
  - $\mathcal{L} = \frac{1}{S}\sum_i (\lambda_{fg}\mathcal{L}_{fg} + \lambda_{seg}\mathcal{L}_{seg} + \lambda_{align}^{\text{pre}}\mathcal{L}_{align}^{\text{pre}} + \lambda_{align}^{\text{post}}\mathcal{L}_{align}^{\text{post}})$
- **自监督预训练**：从目标视频随机采样 query clip 构造辅助任务；数据增强（帧反转、高斯噪声、变速，各 50% 概率）；使用 Kinetics-700 无标签视频
- 推理：1D NMS（阈值 0.7），选 $\hat{f}_i$ 最高对应位置

---

## 实验与结果

### 数据集
| 数据集 | 用途 | 规模 |
|--------|------|------|
| FSVOD-500 | Few-shot VOD | 训练/测试类别 disjoint |
| FSYTV-40 | Few-shot VOD | — |
| VidOR | Few-shot VOD | — |
| VidVRD | Few-shot VOD / 关系检测 | — |
| ActivityNet-VRL | Vid2VidMR | 训练 462,872 对 / 验证 829 / 测试 978（160/20/20 类互斥） |
| SportsMoments（本文自建） | Vid2VidMR | 80 场完整比赛（64 板球+16 足球），176.6 小时，4,999 标注片段；训练 ~750K / 验证 10K / 测试 10K（16/4/9 类互斥） |

### 主要结果（5-shot，FSVOD 基准）

| 数据集 | 指标 | Ours | 次优 | 提升 |
|--------|------|------|------|------|
| FSVOD-500 | AP/AP50/AP75 | **30.6 / 42.9 / 32.1** | TACF: 26.9 / 39.2 / 27.9 | **+3.7 / +3.7 / +4.2** |
| FSYTV-40 | AP/AP50/AP75 | **21.2 / 29.8 / 23.5** | TACF: 15.9 / 24.2 / 17.9 | **+5.3 / +5.6 / +5.6** |
| VidOR | AP/AP50/AP75 | **49.4 / 57.3 / 50.2** | FSVOD: 45.1 / 54.3 / 46.2 | **+4.3** |
| VidVRD | AP/AP50/AP75 | **48.7 / 57.8 / 49.2** | FSVOD: 40.7 / 53.5 / 43.7 | **+8.0** |

- 1-shot FSVOD-500：Ours 27.4% vs FSVOD 20.7%（+6.7%）；10-shot 33.2% vs 27.2%（+6.0%）
- 最强结果：VidVRD AP 48.7%，相对次优提升 +8.0 AP

### MATR 结果（Vid2VidMR）

| 数据集 | 指标 | OTAM | Soft-DTW（默认） |
|--------|------|------|------------------|
| ActivityNet-VRL | mIoU / R@1 | 58.2% / 58.7% | ~略低 |
| SportsMoments | mIoU / R@1 | 62.2% / 58.6% | ~略低 |

- 相比 BM-DETR：参数 -27.9%、FLOPs -35.3%、显存 -34.2%、速度 +24.7%、mIoU +9.2%
- 与 Video-LLaMA 系列对比：MATR 远超 Video-LLaVA、TimeChat 等 VLMs（后者需 caption 转换丢失时序信息）
- 鲁棒性：高斯噪声 mIoU 降 -1.0%、变速 -3.5%、帧反转 -14.1%（表明时间方向为重要信号）
- 跨域泛化：DCT-Net 卡通风格 query 下 mIoU 38.3%

### 效率对比（FSVOD-500）

| 方法 | FPS | 显存 (GB) | AP |
|------|-----|-----------|-----|
| Ours（TOViT）| 14 | 8.1 | 30.6 |
| QDETRv | 11 | 7.6 | 26.1 |
| Video OWL-ViT | 10 | 8.4 | 25.8 |
| FSVOD | 16 | 5.8 | 25.1 |
| ByteTrack+CLIP | — | — | 14.7 |

- Ours 比 VideoOWL-ViT 快 40%，比 QDETRv 快 27%（+4.5 AP，仅多 0.5 GB）
- 与 BM-DETR 对比（MATR）：Params 58.6M vs 81.3M，FPS 18.6 vs 14.0

---

## 相关工作脉络

1. **DETR 及扩展**：DETR [10]（集合预测消除 NMS）→ Deformable DETR [47] → 视频 DETR [20,21]；本文 QDETRv 进一步扩展为查询引导的视频 DETR 并加入无监督预训练。
2. **开放词汇检测**：OWL-ViT [6]（去除解码器、利用图文对比预训练）→ Video OWL-ViT [7]；本文 TOViT 在此基础上引入跨帧选择性特征传播和 few-shot heads，区别于纯 open-vocabulary 方法。
3. **视频动作/时刻检索**：Moment-DETR [20]、QD-DETR [21]、UniVTG [24]（文本→视频）；GDP [23]、FFI+SRM [22]（视频→视频）；本文 MATR 针对 Vid2VidMR 设计双阶段 DTW 对齐，优于仅靠 cross-attention 的文本 VMR 方法。
4. **视觉关系检测**：Vid-OR [16]、VidVRD [17]；传统 VRD 依赖全监督三元组标注；本文方法建模为 T-partite 随机场优化 + episodic meta-learning，支持未见谓词泛化。
5. **多目标跟踪**：DeepSORT [143]、TrackFormer [144]、ByteTrack [145]；本文对比表明无显式 few-shot 模块的追踪方法在 few-shot 设置下显著落后（AP 差距达 13–24%）。
6. **Video-LLM**：Video-LLaVA [221]、Qwen3.5-VL [147]；本文指出通用 Video-LLM 缺乏跨帧对象身份维持机制，专用 grounding 模型仍更优。

---

## 局限性与未来方向

1. **遮挡/小目标/相似物敏感**：QDETRv 在 occlusion、small object、visually similar object、overlapping instances 场景表现下降；未来需开发更鲁棒的时空表征。
2. **绝对阈值 τ 的域适应性**：最优 τ=0.94 在四基准上无需调参，但对差异显著的域需重新调优；未来方向包括 per-frame quantile 或 learnable gate。
3. **时间方向假设**：帧反转导致 mIoU 下降 14.1%，表明当前方法依赖单调时间假设，对倒放/非线性叙事视频鲁棒性不足。
4. **外观不变表征**：跨域（卡通风格 query）泛化仅 38.3% mIoU， viewpoint/speed-invariant 表征仍是开放问题。
5. **关系级任务的谓词覆盖率**：ImageNet-VidVRD 中仅 18.9% predicate 有 >100 实例，长尾分布仍需更好建模。

---

## 研究启发与可借鉴点

1. **跨帧选择性特征传播机制**：TOViT 中"仅高置信度 embedding 跨帧传递"的设计可有效抑制噪声累积，可迁移至视频跟踪、时序分割等任务。
2. **双阶段 DTW 对齐策略**：MATR 的 pre- 和 post-fusion alignment 可在保持可微性的同时捕捉粗到细的时序关系，适用于任意需要时序对齐的多模态检索任务。
3. **T-partite 随机场 + Belief Propagation 建模关系定位**：将视觉关系定位形式化为结构化图优化，通过 meta-learning 学习关系势函数，为 few-shot 关系推理提供新范式。
4. **自监督预训练复用任务形式**：MATR 直接复用 Vid2VidMR 任务结构构造辅助预训练样本（随机采样 clip + 时序增强），无需额外标注即可提升初始化，可推广至其他视频理解任务。
5. **三粒度统一框架的评估体系**：论文在对象/关系/时刻三个粒度均提供 SOTA 结果，未来可借鉴此 multi-granularity 评测思路构建统一 benchmark。

---

## 关键术语表

**QDETRv**：Query-Guided DETR for Videos，将 DETR 扩展至视频域，通过查询-上下文融合与时序记忆模块实现一次定位未见对象。

**TOViT / Obj-Wise Vision Transformer**：Temporal Object-Aware Vision Transformer，基于 OWL-ViT 的 few-shot 视频目标检测框架，引入跨帧选择性特征传播筛选机制。

**T-partite 随机场**：将视频中 subject-predicate-object 三元组的候选 bounding box 建模为 T 部图的顶点，通过结构化势函数优化实现关系定位。

**Belief Propagation**：在 T-partite 图上进行的消息传递算法，用于高效求解关系定位的联合优化问题，生成时序一致的 subject-object 轨迹。

**GSA（Global Semantic Aggregation）**：用可学习注意力权重跨所有帧融合 I3D 特征，纳入长程上下文线索的关系聚合技术。

**LLA（Local Localization Aggregation）**：聚焦时间窗口内邻近帧的空间与 ROI 特征，处理遮挡与部分可见对象的关系聚合技术。

**MATR（Moment Alignment Transformer）**：用于 Vid2VidMR 的双阶段动态时间规整 Transformer，支持可微 DTW 对齐与自监督预训练。

**Soft-DTW**：ε-smoothed 版本动态时间规整，处处可微，支持梯度回传，用于 MATR 的预融合与后融合对齐损失。

---

## 可复现要素

- **数据集**：FSVOD、FSYTV-40、VidOR、VidVRD、ActivityNet-VRL 均为公开基准；SportsMoments 由作者自建（论文提供采集与标注流程）
- **代码**：论文未明确声明开源，但列出了各章节对应发表信息（AAAI 2024/2026、CVPR 2023、ICCV 2025、CVIU 2026），可关注作者 GitHub 或会议附录
- **关键超参**：τ=0.94（特征传播阈值）、κ=0.98（检测阈值）、λ_cls=2、λ_box=5、1D NMS 阈值=0.7、采样率=1 frame/2s
- **训练配置**：AdamW、lr=1e-5、weight decay=0.01、cosine scheduling + linear warmup、4 天/2×Nvidia A6000
- **Vision Encoder**：OWL-ViT-L/16（CLIP 预训练）

---
