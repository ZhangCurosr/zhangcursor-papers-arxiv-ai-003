---
title: "VISIONWEAVE-WEAVING-ELASTIC-VISUAL-REPRE-SENTATIONS-AS-A-NAT"
source: https://arxiv.org/pdf/2610.07987v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-07 10:56:05"
field: "多模态大模型高效推理"
keywords: ["Multimodal Large Language Model", "Visual Token Compression", "Elastic Representation", "Self-Distillation", "SGLang Serving", "Adaptive Granularity"]
innovations: ["通过门控空间池化器与多深度粒度路由器将弹性细/粗粒度视觉表示编织为 MLLM 原生能力", "三阶段自蒸馏（全粗→软混合→硬路由）渐进式注入内容自适应 token 分配", "在 SGLang 上实现 block-center MRoPE 与混合粒度序列的端到端部署并验证 2.30× 吞吐增益"]
benchmarks: ["RealWorldQA", "DocVQA", "InfoVQA", "ScreenSpotV2", "HallusionBench", "VideoOCR", "Video-MME", "LongVideoBench"]
---

# 论文速读：VISIONWEAVE-WEAVING-ELASTIC-VISUAL-REPRE-SENTATIONS-AS-A-NAT

## 一句话总结
VisionWeave 通过自蒸馏将弹性视觉表示编织建立为 MLLM 的原生能力，结合可学习的门控空间池化器与粒度路由器，实现内容自适应的细/粗粒度 token 分配；在 Qwen3.8-27B 上平均节省 43.0% 视觉 token 的同时保留 98.9% 原生性能，并在 SGLang 上实现 2.30× 吞吐提升。

## 研究问题与动机
1. **视觉 token 压缩的效率-质量权衡难题**：现代 MLLM 将图像划分为固定尺寸 patch 并编码为密集 token 序列，随分辨率/视频帧数增长带来巨大计算与显存开销；现有降采样方案牺牲细粒度细节，而 token pruning/merging 方法（如 FastV、VisionZip）在信息密集输入和 grounding 任务上性能急剧下降。
2. **固定压缩比缺乏内容自适应**：既有方法依赖预设压缩率，无法根据图像局部信息密度动态调整粒度；ViCO 等 tile 级自适应方案的空间粒度不足，难以处理细节区与背景交错分布的场景。
3. **与现代 serving 基础设施兼容困难**：token pruning/merging 常需 attention weight 提取或在 LLM 内部进行剪枝，难以集成到 SGLang/vLLM 等现代推理引擎，且缺乏端到端部署验证。
4. **缺乏原生弹性视觉处理能力的预训练/后训练范式**：现有工作多在独立视觉任务或需外部指定 budget 的设定下探索内容自适应表示，未将其作为通用 MLLM 的内置能力进行大规模训练验证。

## 核心贡献（创新点）
1. **弹性视觉表示编织的原生能力建立**：首次通过大规模三阶段自蒸馏将内容自适应细/粗粒度 token 分配内化为准 frontier 级 MLLM（Qwen3.5-4B / Qwen3.8-27B）的原生特性，区别于仅做推理时 pruning 的旁路方法。
2. **门控空间池化器（Gated Spatial Pooler）**：以均值池化为起点，引入可学习通道权重与特征精炼，将 2×2 邻域 4 个 patch 映射为单一粗粒度 token，并与原生 fine token 共享同一 MRoPE 坐标空间；相比 pixel-unshuffle 投影在 DocVQA、VideoOCR 等任务上提升 2.64/2.77 分。
3. **多深度探测的粒度路由器（Granularity Router）**：通过 learnable query 在 ViT 早期/中期/末期特征上进行 local-global cross-attention，输出逐块粗粒度选择概率；不同于 ViCO 的 tile 级分辨率分配，支持更细粒度的空间粒度控制。
4. **SGLang 原生部署与端到端效率验证**：实现 block-center 半整数 MRoPE 与混合粒度序列的 serving 支持，在 LongVideoBench 负载下获得 2.30× 吞吐增益、TTFT -54.4%、TPOT -60.6%，填补 token pruning 类方法在工程落地的实证空白。

## 方法详解
**整体架构**：给定 ViT 输出特征 $X \in \mathbb{R}^{H \times W \times d}$，原生 merger $\mathcal{M}$ 将每 $2\times2$ 邻域拼接后经 MLP 生成细粒度 token $V_{\text{fine}} \in \mathbb{R}^{\frac{H}{2}\times\frac{W}{2}\times D}$；同时门控空间池化器 $\mathcal{P}$ 对每 $2\times2$ patch 邻域输出 $\mathbb{R}^{\frac{H}{2}\times\frac{W}{2}\times d}$，再经 $\mathcal{M}$ 得到粗粒度 token $V_{\text{coarse}} \in \mathbb{R}^{\frac{H}{4}\times\frac{W}{4}\times D}$。粗粒度 token 通过位置插值置于对应 $4\times4$ patch 块的几何中心，保留 MRoPE 一致性。

**Gated Spatial Pooler**（公式 4-6）：对 4 个 patch 特征先经 learnable LayerNorm，拼接后经 $f_{\text{glb}}$ 得到全局上下文 $g_k$；gate 分支由 $f_{\text{gate}}$ 计算通道级 softmax 权重 $\alpha_k$；value 分支由 $f_{\text{val}}$ 精炼后加权求和。初始化为均值池化（$f_{\text{gate}}/f_{\text{val}}$ 末层零初始化），逐步学习内容依赖的选择性聚合。

**Granularity Router**（公式 7-9）：每块 $(I,J)$ 维护一 learnable query $h_{IJ}$，在三处 ViT 深度 $\ell \in \{0, L/2, L\}$ 依次做 local cross-attention（关注块内 $4\times4$ patch）和 global cross-attention（扩展至全图），共 6 个独立参数块；末层线性头输出 coarse 概率 $p_{IJ}$。推理时按阈值 $t=0.5$ 硬路由：$p_{IJ}>t$ 选 $V_{\text{coarse},IJ}$，否则保留 4 个 $V_{\text{fine}}$。

**三阶段自蒸馏训练**：
- Stage 1（训练 Pooler）：路由 bypass，所有块强制用粗粒度，优化 $\mathcal{L}_{\text{distill}}=\text{KL}(\hat{p}_\theta(\cdot|V_{\text{fine}}) || \hat{p}_\theta(\cdot|V_{\text{coarse}}))$，样本权重 $\propto \sqrt{\text{响应长度}}$。
- Stage 2（训练 Router）：可微 soft-mix 替代硬路由，对 norm 与方向分别插值（公式 11），加入 MoE-style 平衡损失 $\mathcal{L}_{\text{bal}}$（目标粗路由比例 $\rho=0.8$）。
- Stage 3（联合优化 Pooler+LLM）：使用推理时的硬路由混合序列，冻结 ViT/merger/router，温度=1 的 off-policy 蒸馏。

**SGLang 部署适配**：① Block-center RoPE 支持——将所有位置坐标乘 2、旋转频率除 2，兼容整数索引 cache；② 混合粒度序列支持——在 encoder-disaggregated stage 附加布局元数据，支持 pre-admission embedding slicing。

## 实验与结果
**数据集与基准**：8 项跨域 benchmark（RealWorldQA、DocVQA、InfoVQA、ScreenSpotV2、HallusionBench、VideoOCR、Video-MME、LongVideoBench），覆盖自然图像、文档/信息图理解、GUI grounding、幻觉检测与长视频理解。

**主要结果（Qwen3.8-27B，单图/单帧预算 512 token，默认 $t=0.5$）**：
- VisionWeave 平均节省 **43.0%** 视觉 token，性能保留 **98.9%**（均分 76.73 vs 原生 77.65，-1.08 分）。
- 固定 50% 压缩目标下 FastV† / VisionZip 分别仅保留 88.2% / 87.5% 性能（-11.82% / -12.46%）。
- ScreenSpotV2 上 VisionWeave 以 55.1% 节省仍保持 91.43（-3.17%），对比 VisionZip 在 47.88% 大幅下降。
- DocVQA 上自适应削减至 27.3% 节省仅 -1.23%，体现内容敏感分配。

**分辨率/帧数鲁棒性**：跨 8 项任务、多输入分辨率（图 5）及 256 帧上限视频（图 6）下 consistently 优于简单降采样；256-frame VisionWeave 比 128-frame native 在 LongVideoBench 高 1.20 分且 token 更少。

**SGLang 部署效率（Table 5）**：Native 0.90 req/min → VisionWeave 2.07 req/min（**2.30×**）；Mean TTFT 169.90s→77.54s（-54.4%）；P95 TPOT 715.58ms→289.19ms（-59.6%）。

**推理时控制**：调整 $t$ 可在不重训前提下连续调节 token 节省率（图 8），$t=1.0$ 时退回全细粒度性能。

**消融**：Pooler 相对 pixel-unshuffle 在全 8 榜提升；Router  shuffled routing 使均值下降 5.36 分（Table 6），验证"在哪分配"比"分配多少"更重要；Stage 3 对 7/8 榜提升，ScreenSpotV2 +24.84 分最为显著（Table 7）。

## 相关工作脉络
1. **FastV / VisionZip（token pruning/merging 代表）**：FastV 在 LLM 内层后剪枝、VisionZip 做 token merge，两者均依赖固定压缩比并在文档/grounding 输入上退化严重；VisionWeave 以内容自适应混合粒度替代删减，保持空间覆盖无空洞。
2. **ViCO（tile 级自适应分辨率）**：按低层图像统计将图划分 tile 后为每 tile 分配单一分辨率，空间粒度粗且难以直接接入 Qwen/Kimi 等 native-resolution pipeline；VisionWeave 在块级支持更细粒度切换。
3. **动态 patch 大小方法（Yu et al., Choudhury et al.）**：基于边缘密度/熵等底层统计调整 ViT patch size，主要面向纯视觉分类/检测任务，未优化多模态下游目标。
4. **Matryoshka Multimodal Models / MQT / PAR-CEL**：支持多 token budget 但需外部指定预算；VisionWeave 的预算由路由器隐式产出，无需人工调参。
5. **KV-cache 压缩（DeepSeek-V4 等）**：通过量化/截断存储压缩 KV 状态而非缩短输入序列；作者指出与弹性视觉编织正交可叠加：前者压缩缓存、后者减少 LLM 全程计算。
6. **低层统计驱动 adaptive patch（Choudhury et al. / Yu et al.）**：依赖启发式图像特征，非端到端学习任务感知的粒度决策。

## 局限性与未来方向
1. **训练成本高昂**：Qwen3.8-27B 三阶段自蒸馏消耗 30K+ A100 GPU 小时，限制了对 router 架构变体等更广泛探索。
2. **双粒度限制**：当前仅细/粗两级，作者建议未来扩展至多尺度粒度以进一步提升表示灵活度。
3. **后训练植入方式**：采用 post-training 自蒸馏注入能力，未来可探索在 base model 预训练阶段原生引入弹性视觉编织。
4. **评估任务集有限**：虽覆盖 8 项跨域 benchmark，但对更长上下文视频、3D/多视角场景、多轮交互 agent 任务验证不足。
5. **阈值超参 $t$ 仍需手动设定**：虽支持推理时调节，但未探索任务/租户级别的自动适配策略。

## 研究启发与可借鉴点
1. **三阶段 self-distillation 渐进式能力注入范式**：从"全粗→软混合→硬路由"的调度设计有效缓解梯度不连续，可迁移至其他模块（如 KV eviction、跨模态路由）的增量训练。
2. **块级 soft-mix + MoE-style 平衡损失**：在 training 时避免 hard routing 的不可导问题，同时用 per-sample 统计防止长输入主导分布估计，思路可直接复用于 sequence-level routing。
3. **block-center RoPE 坐标扩展技术**：将半整数位置统一缩放为整数后复用已有 rotary kernel，对任何引入非整格坐标的模块均有参考价值。
4. **SGLang encoder-disaggregated + layout metadata 扩展**：为混合粒度 token 序列提供了可直接复用的 serving 适配模板，有助于打通其他弹性视觉方法到生产环境的最后一公里。
5. **"在哪分配"比"分配多少"更重要**的消融结论提示：后续研究应优先验证路由空间对齐性，而非单纯追求更高压缩比。

## 关键术语表
**Elastic Visual Representation Weaving**：将细粒度与粗粒度视觉表示按内容自适应交织为混合粒度序列的原生能力。
**Gated Spatial Pooler**：以均值池化为起点、通过通道级 gate 与特征精炼将 2×2 patch 邻域聚合为单一粗粒度 token 的可学习模块。
**Granularity Router**：基于多深度 ViT 特征的 learnable query + local-global cross-attention 结构，输出每空间块选择粗/细粒度的概率。
**MRoPE（Multi-dimensional Rotational Position Embedding）**：为视觉 token 赋予时间-空间多维坐标的 RoPE 扩展，VisionWeave 中粗粒度 token 通过位置插值对齐至 $4\times4$ 块的几何中心。
**Self-Distillation（Off-policy）**：以 native 模型 rollout 为教师信号、学生模型在相同文本条件下拟合输出分布的蒸馏方式，训练与推理输入独立。
**$\mathcal{L}_{\text{bal}}$（Balance Loss）**：源自 MoE 训练的 per-sample 路由均衡正则，惩罚粗路由比例偏离目标 $\rho$ 的分布。
**SGLang Encoder-Disaggregated Serving**：将视觉编码器与 LLM 推理分离的部署架构，VisionWeave 在此之上附加 layout metadata 实现混合粒度序列调度。

## 可复现要素
- **数据集**：LLaVA-OneVision（246K）、LLaVA-Video-178K（281K）、UGround（250K），总计约 78 万样本；论文未说明开源状态（内部组合）。
- **代码/权重**：SGLang 集成代码与详细信息将开源至 https://github.com/FFY0/VisionWeave（论文投稿时未公开）；基座 Qwen3.5-4B / Qwen3.8-27B 为开源模型。
- **关键超参**：路由阈值 $t=0.5$（推理默认）；$\rho=0.8$（Stage 2 目标粗路由比例）；Stage 2/3 蒸馏温度 1；Stage 2 $\mathcal{L}_{\text{bal}}$ 权重 0.02；Stage 3 学生最大序列长 24,584；ViT 特征采样深度 [0, 13, 27]。
- **训练硬件**：>30K A100 GPU 小时；Qwen3.8-27B 使用 BF16、Adam（$\beta_1=0.9, \beta_2=0.999$）、cosine LR decay、zero weight decay。
