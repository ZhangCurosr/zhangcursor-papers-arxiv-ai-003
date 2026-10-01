---
title: "SPIDER-Multi-Layer-Semantic-Token-Pruning-and-Adaptive-Sub-L"
source: https://arxiv.org/pdf/2609.34977v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 10:12:28"
field: "多模态大模型推理加速"
keywords: ["Multimodal Large Language Models", "Token Pruning", "Layer Skipping", "Inference Efficiency", "Training-free Acceleration", "Visual Token Redundancy"]
innovations: ["多层语义联合的 Token 剪枝（中间层+深层特征联合聚类）", "面向 Attention/FFN 子层的自适应细粒度跳过机制（在线熵+离线SLC双评分融合）", "统一Token效用衰减视角的跨编码器-解码器联合加速框架"]
benchmarks: ["LLaVA-1.5-7B", "LLaVA-NeXT-7B", "Qwen2.5-VL-3B", "Qwen3-VL-8B", "GQA", "VQAv2", "MME", "POPE", "MMStar", "MMVet", "TextVQA", "DocVQA"]
---

# 论文速读：SPIDER-Multi-Layer-Semantic-Token-Pruning-and-Adaptive-Sub-L

## 一句话总结
SPIDER 是一种训练免费的 MLLM 推理加速框架，通过联合利用视觉编码器中间层与深层的多层语义 Token 剪枝（MSV-Prune），以及在 LLM 解码器中对 Attention/FFN 子层进行自适应跳过的细粒度计算冗余消除（ASL-Skip），在大幅降低 FLOPs 的同时保持与原始模型近乎一致的多模态性能。

## 研究问题与动机
- **视觉 Token 剪枝的语义聚焦偏移问题**：现有 SOTA Token 剪枝方法几乎仅依赖视觉编码器最后一层特征计算 Token 重要性，但中间层 Token 对细粒度任务（如计数、精确定位、OCR）所捕捉的 Object-centric 细节远多于深层，直接丢弃会导致关键语义碎片丢失。
- **LLM 解码器层跳过过于粗粒度**：现有 Layer Skipping 方法通常将整层一次性替换为稀疏版本，忽略了同一层内 Attention 与 FFN 子模块对视觉 Token 贡献的显著差异。
- **数据冗余与计算冗余未被统一建模**：视觉端 Token 数量压缩与解码端计算节省被当作两个独立问题处理，缺乏对 Token 在整个 MLLM 流水线中"效用随深度变化"的统一视角。
- **推理效率仍是 MLLM 大规模落地的核心瓶颈**：高分辨率输入产生的数千级视觉 Token 使序列长度膨胀，显著增加预填充与解码延迟，在边缘/交互场景下尤为突出。

## 核心贡献（创新点）
1. **多层语义 Token 剪枝（MSV-Prune）**：突破"仅看最后一层"的传统范式，将 ViT 中间层与深层特征联合用于语义聚类和相似度计算，使剪枝决策同时兼顾全局语义与细粒度对象细节；区别于 VisPruner 等仅依赖单尾层的做法。
2. **自适应子层跳过（ASL-Skip）**：首次在 MLLM 解码器中引入细粒度的子层（Attention vs. FFN）跳过机制，并结合在线逐 Token 可跳过评分与离线 Sub-Layer Contribution（SLC）评分，动态决定"跳不跳、何时跳、跳过哪个子模块"；区别于 ShortV 等对整个 Decoder Block 一刀切的粗粒度方案。
3. **统一效用分配视角**：将 Token 剪枝和子层跳过视作同一个"Token 效用随推理深度衰减"问题的两个阶段——MSV-Prune 决定哪些 Token 进入解码器，ASL-Skip 决定每个保留 Token 在解码过程中还需多少计算；区别于以往将两者简单拼接的独立优化。
4. **训练免费且跨架构泛化**：全程无需微调，已在 LLaVA-1.5-7B、LLaVA-NeXT-7B、Qwen2.5-VL-3B、Qwen3-VL-8B 等多架构上验证，并在训练感知模式（LoRA fine-tune）下进一步逼近甚至超越需要训练的 PruMerge+。

## 方法详解
- **Multi-layer Semantic Visual Token Pruning (MSV-Prune)**
  - **锚定 Token 选取**：基于视觉编码器自注意力分数（[CLS] 行或 patch 平均注意力）采用动态阈值 τ 筛选出占比为 R×r 的高显著性锚定 Token 集 T_v^anc。
  - **互补 Token 选取**：对非锚定 Token 在高维特征 F_v^L 上做 K-means 粗粒度聚类，按簇大小比例分配配额 N_k；在每簇内用多层余弦相似度 S_ij 排序，S_ij 包含三层内多层相似度（中间层 + 深层）以及与锚定 Token 的相似度两项，保留分数最低的 N_k 个 Token 构成 T_v^cmp，最终与文本 Token 拼接送入 LLM。
- **Adaptive Sub-Layer Skipping (ASL-Skip)**
  - **在线可跳过评分 S_sa(i,ℓ)**：由两部分组成——Semantic Information Entropy E_ii（Token 隐藏态经 unembed 后的 Shannon 熵，衡量语义收敛程度）与 Image-Text Correlation Factor F_itc（Token 隐藏态与文本上下文的余弦相似度），Retention Score R = E_ii + F_itc，S_sa = ReLU(1 − R)。
  - **离线 Sub-Layer Contribution (SLC)**：对每层 ℓ 分别计算跳过 Attention 或 FFN 后输出 logit 的 KL 散度作为贡献度量，按 min-max 归一化得 SLC^Norm，取每层中更高的作为该层跳过的目标子层 m(ℓ)。
  - **分数融合与累加决策**：在 ℓ ≤ L/2 的决策窗口内累加 S_skip(i) = Σ(w₁·S_sa(i,ℓ) + w₂·SLC^Norm(m(ℓ))(ℓ))，一旦超过阈值 T_skip 即进入 Skip Mode，后续层中该 Token 恒 bypass 其目标子层；w₂ 固定为 1，w₁/w₂ 控制在线评分比重。
- **训练感知扩展**：可在原始 checkpoint 基础上用 LoRA 仅微调 1 epoch（665K 指令数据），SPIDER(Train) 进一步将 Aggregate Accuracy 提升至 99.5%，超越训练型的 PruMerge+（99.0%）。

## 实验与结果
- **评估模型**：LLaVA-1.5-7B（576 visual tokens）、LLaVA-NeXT-7B（2880 tokens）、Qwen2.5-VL-3B-Instruct、Qwen3-VL-8B-Instruct。
- **评测基准**：GQA、VQAv2、MME、TextVQA、POPE、MMB（EN/CN）、MMVet、MMStar、DocVQA、ChartQA、VizWiz 等。
- **核心结果**：
  - LLaVA-NeXT-7B：SPIDER 在 ~51% TFLOPs 下达到 Aggregate Accuracy 99.11%（VQAv2=80.2，MMStar=37.7，POPE=88.2）；在 ~21% TFLOPs 时仍保留 96.09% 的基线性能，大幅领先 ShortV（63.56%）等基线。
  - LLaVA-1.5-7B：在 ~56% TFLOPs 下 ACC=99.87%，优于 VisPruner（98.60%）、ShortV（96.08%）和 FastV（95.54%）。
  - Qwen3-VL-8B-Instruct：在 ~55% 与 ~45% TFLOPs 下分别达到 99.06% 和 98.33%，超越 PruneSID、IVC-Prune、iLLaVA、ERASE 等同期方法。
  - **最强提升**：LLaVA-NeXT-7B 在 21% TFLOPs 时相比 ShortV（63.56%）提升 32.53 个百分点；LLaVA-1.5-7B 的 Instance-level POPE 分析显示 SPIDER 使正确率从 85.9% 提升至 86.6%（净增 0.7%）。
  - **效率收益**：LLaVA-1.5-7B 上 Prefill 延迟从 67.4 ms 降至 48.2 ms（−28.5%），Decode 延迟从 23.1 ms 降至 20.6 ms（−10.8%），吞吐从 43.3 升至 45.2 tokens/s。

## 相关工作脉络
1. **FastV**：仅在第 K 层后用固定比例 R 剪枝，依赖文本-视觉注意力，存在位置偏差；SPIDER 的 MSV-Prune 以视觉中间层语义为主，避免位置偏见并更好地覆盖背景信息。
2. **VisPruner / VisionZip / HiPrune**：主张只用视觉编码器最后一层特征做剪枝；本文证明中间层在细粒度任务上更具代表性，MSV-Prune 联合两层特征弥补这一缺陷。
3. **ShortV**：用稀疏版本整体替换整个 Decoder 层；SPIDER 的 ASL-Skip 细粒度到子层（Attention/FFN）并逐 Token 自适应，减少不必要的性能损失。
4. **SparseVLM / Pyramiddrop**：利用文本-视觉注意力进行稀疏化；本文认为视觉自身多层语义足以驱动剪枝，无需依赖易产生位置偏倚的文本注意力。
5. **PruMerge+ / ERASE / IVC-Prune / PruneSID**：后者均多为训练型或独立优化单一阶段；SPIDER 在训练免费条件下达到同水平甚至更强，并可轻量 LoRA 微调进一步拉高上限。
6. **Magnitude Pruning**：按输出范数裁剪子层；本文 SLC 基于 KL 散度度量对最终预测的影响，实验表明 SLC 策略在保真度上显著优于 magnitude pruning。

## 局限性与未来方向
- **对低质量/畸变图像鲁棒性不足**：运动模糊（如车牌识别）和医学影像中异常区域容易被误跳过或剪除，导致幻觉或错误诊断。
- **细粒度 Token 动态稀疏性难以被标准 dense GPU kernel 完全利用**：ASL-Skip 引入的输入依赖型不规则稀疏在 Decode 阶段仍受限于硬件算子，尚需专门的 sparse kernel 支持。
- **离线 SLC 策略虽具备跨基准 Rank 一致性，但仍基于有限采样（150 条样本）构建**，在极端长尾域可能产生偏差。
- **论文提议的未来方向**：任务感知 Token 保留策略、结合轻量分割模块或解剖先验的领域自适应压缩、面向 Decode 阶段的 token regrouping 与硬件感知稀疏计算优化。

## 研究启发与可借鉴点
1. **"多层语义联合"替代"单层贪心"**：MSV-Prune 证明中间层特征在细粒度视觉理解中具有不可替代性，可迁移至视频/多帧场景中的跨帧 Token 选择与压缩设计。
2. **在线 + 离线混合评分框架**：S_sa（在线）负责输入自适应，SLC^Norm（离线）负责结构性冗余先验，二者加权累加后过阈触发，设计简洁且无需额外参数，可直接复用到其他推理加速模块中。
3. **统一"Token 效用随深度衰减"视角**：将编码端和数据端冗余视为同一问题的两个阶段，为今后设计跨组件协同加速框架提供概念模板。
4. **Instance-level 边界漂移分析**：在 POPE 上对比 SPIDER 前后正确/错误样本翻转数，证明合理剪枝反而能减少幻觉，为效率方法的可靠性评估提供了可复用的分析范式。
5. **训练免费 + 轻量训练双模式并存**：同一框架既可做零成本部署，又可通过单轮 LoRA 微调逼近训练型 SOTA，为工程落地提供了灵活的性价比梯度。

## 关键术语表
**Multimodal Large Language Model (MLLM)**：将冻结视觉编码器与 LLM 通过可训练连接器结合的架构，实现图像/视频理解与文本生成的统一推理。
**Token Pruning**：在视觉编码或解码阶段按重要性度量剔除冗余视觉 Token，以降低序列长度与计算开销的训练免费加速策略。
**Layer Skipping**：将部分 LLM Decoder 层替换为稀疏/冻结版本，跳过冗余计算层以减少 FLOPs 的推理加速方法。
**Sub-Layer Contribution (SLC)**：通过 KL 散度量度跳过某一层中 Attention 或 FFN 子模块后对最终输出分布的影响程度，SLC 越低表示该子模块越可被安全跳过。
**Semantic Focus Shift**：ViT 不同层对视觉内容的表征重心差异，中层偏向对象级细粒度，深层偏向全局抽象语义的现象。
**Skippability Score (S_sa)**：结合 Token 隐藏态的语义熵 E_ii 与图文相关性 F_itc 的在线可跳过评分，分数越高表示该 Token 越值得跳过。
**Anchor vs. Complementary Tokens**：MSV-Prune 中将视觉 Token 分为高显著性的锚定集 T_v^anc 与用于补充语义多样性的聚类互补集 T_v^cmp 两类。
**Aggregate Accuracy (Acc.)**：各基准得分相对 Vanilla 性能的归一化均值，用于在跨尺度指标间统一比较不同方法的综合表现。

## 可复现要素
- **数据集**：GQA、VQAv2、MME、TextVQA、POPE、MMB、MMVet、MMStar、DocVQA、ChartQA、VizWiz；论文声明使用公开 benchmark，未提及专门私有数据集。
- **代码/权重**：论文未明确声明开源仓库链接（截至 arXiv 版本），仅说明基于 LLaVA-1.5/NeXT 公开权重与 Qwen2.5-VL/Qwen3-VL 开源模型进行评估；建议关注论文后续补充材料或作者主页获取代码。
- **关键超参**：总体保留比例 R、锚定占比 r（默认 0.7）、簇数 K、累积阈值 T_skip（默认 20）、权重比 w₁/w₂（默认 3）、SLC 离线采样集（150 条：GQA/MMVet/POPE 各 50 条）。
