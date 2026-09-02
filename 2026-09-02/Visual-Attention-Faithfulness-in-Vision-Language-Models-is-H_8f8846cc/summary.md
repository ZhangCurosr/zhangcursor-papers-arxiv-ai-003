---
title: "Visual-Attention-Faithfulness-in-Vision-Language-Models-is-H"
source: https://arxiv.org/pdf/2609.00830v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 19:21:56"
field: "视觉语言模型可解释性"
keywords: ["vision-language models", "attention faithfulness", "causal perturbation", "interpretability", "visual attention", "comprehensiveness", "sufficiency gap"]
innovations: ["提出VLM视觉注意力忠实度的因果扰动评估框架，同时度量充分性与必要性", "发现并定义三种异质性视觉注意力处理模式（Faithful-Sufficient/Distributed/Non-Focal）", "揭示模型视觉依赖与人类标注GT区域之间存在系统性偏差"]
benchmarks: ["VQAv2", "VRDU", "ChartQA"]
---

# 论文速读：Visual-Attention-Faithfulness-in-Vision-Language-Models-is-Heterogeneous

## 一句话总结
本文通过 token 级因果扰动分析，首次系统评估了 VLM 中视觉注意力的忠实度，发现其并非二值属性，而是呈现三种异质性处理模式（Faithful-Sufficient / Faithful-Distributed / Non-Focal），且模型视觉依赖与人类标注存在系统性偏差。

## 研究问题与动机
1. **NLP 领域注意力忠实度之争尚未迁移至视觉模态**：文本注意力是否忠实反映模型推理已有大量辩论（Jain & Wallace, 2019 vs Wiegreffe & Pinter, 2019），但 VLM 中视觉注意力排名是否对应模型对特定视觉区域的因果依赖，尚未被系统验证。
2. **现有效率/解释方法隐含"注意力=忠实"假设**：VLM token 剪枝（Chen et al., 2024a; Shang et al., 2025）直接假设低注意力 token 可安全删除，但这一定理从未被因果验证。
3. **人类标注区域未必等于模型因果证据区**：现有可解释性工作常以 human-annotated GT 区域作为"地面真相"，但模型实际依赖的视觉区域可能与人类直觉存在系统性偏离，这一偏差未被量化。
4. **VLM 架构多样性带来异质性预期**：不同视觉编码器（动态 tile vs 全图编码）和 LM backbone 可能导致同一任务采用不同视觉处理策略，但缺乏跨架构的忠实度比较。

## 核心贡献（创新点）
1. **首次提出 VLM 视觉注意力忠实度的系统性因果评估框架**，同时度量充分性（sufficiency gap）和必要性（comprehensiveness）两个互补维度，区别于仅关注单指标或相关性分析的已有工作。
2. **发现并定义三种视觉注意力处理模式**（Faithful-Sufficient / Faithful-Distributed / Non-Focal），揭示了 VLM 视觉忠实度的异质性本质，打破了"注意力要么忠实要么不忠实"的二元认知。
3. **揭示了模型视觉依赖与人类标注 GT 区域之间的系统性偏差**，证明 human-annotated regions 仅在约 60% 的情况下满足充分性，不可作为模型视觉因果依赖的可靠代理。
4. **将三模式框架从通用 VQA 扩展到文档理解任务（VRDU IE / ChartQA VQA），并验证跨架构（Qwen2.5-VL vs InternVL2.5）与跨模型规模（3B/4B/7B/8B）的稳健性**，表明异质性是模型无关现象而具体模式归属取决于架构。

## 方法详解
**注意力 token 排序**：将视觉 token 在 L 层、H 头的全注意力权重聚合为单重要性分数 $a_i = \sum_l \frac{1}{H}\sum_h \frac{1}{|y|}\sum_s \alpha_{l,h}(i, y_s)$，按降序排列后取 top-k 作为候选因果区域 $S_k$。

**因果扰动**：在第一解码器层入口处对选定 token 的 hidden state 执行零消融（$\mathbf{h}_i^{(0)} \leftarrow \mathbf{0}$），避免删除 token 造成的空间布局扭曲，使预测概率变化仅反映被消融 token 的因果贡献。

**忠实度度量**：
- **Comprehensiveness（必要性）**：$\text{Comp}(S_k) = \frac{P(\mathbf{y}|\mathbf{V},\mathbf{T}) - P(\mathbf{y}|\mathbf{V}\setminus S_k, \mathbf{T})}{P(\mathbf{y}|\mathbf{V},\mathbf{T})}$，高值表示 top-k token 对预测必要。
- **Sufficiency Gap（充分性）**：$\text{Sgap}(S_k) = \frac{P(\mathbf{y}|\mathbf{V},\mathbf{T}) - P(\mathbf{y}|S_k, \mathbf{T})}{P(\mathbf{y}|\mathbf{V},\mathbf{T})}$，低/负值表示 top-k token 自身已足够恢复预测。
- 两者均以 baseline 预测概率归一化，确保跨样本可比。

**三模式划分**：在 $(\text{Comp}, \text{Sgap})$ 平面上，先用 Otsu 方法按 Comp 将所有样本二分为 high-Comp 和 low-Comp，再对 high-Comp 子集按 Sgap 做二次 Otsu 分割，辅以边界样本人工校准，最终得到三种模式。

## 实验与结果
**数据集与基线**：VQAv2（90 个样本，平衡 yes/no、number、other 三类）；VRDU（60 个文档 IE 样本）；ChartQA（100 个文档 VQA 样本）。对比策略：Overall Top-k、GT Object、Non-GT Top-k、Random（消融相同 token 数 k）。

**VQA 主要结果（Qwen2.5-VL-7B）**：

| 策略 | Mean Comp | Mean Sgap |
|------|-----------|-----------|
| Overall Top-k | 0.813 | 0.469 |
| GT Object | 0.505 | 0.846 |
| Non-GT Top-k | 0.659 | 0.653 |
| Random | 0.502 | 0.599 |

Overall Top-k 同时达到最高必要性和最低充分性差距，验证注意力排名的因果有效性。GT Object 与 Top-k 差距显著（0.505 vs 0.813 Comp）。

**三模式分布（Qwen2.5-VL-7B，有效样本 85 个）**：
- **Mode A Faithful-Sufficient（32.9%）**：Comp=0.978，Sgap=−0.026，准确度 92.9%，top-k 既必要又充分。
- **Mode B Faithful-Distributed（47.1%）**：Comp=0.952，Sgap=0.818，准确度 100%，top-k 必要但不足。
- **Mode C Non-Focal（20.0%）**：Comp=0.213，Sgap=0.460，准确度 94.1%，无局部必要区域但视觉信息整体仍为必要触发器。

**问题类型关联**：Yes/No 问题集中于 Mode B（55.6%），Number 问题在 A/B 间分配（48.3%/37.9%），Other 问题在 B/C 占比较高（48.3%/34.5%）。

**文档任务结果**：Qwen2.5-VL 在 VRDU IE 和 ChartQA 上均表现为极端 Mode B（top-1% 消融即降 43.8%，Sgap 始终接近 1.0）；InternVL2.5-8B 在同一任务上趋于 Mode A（top-10% Sgap=0.189）。

**跨模型规模（Table 6）**：Qwen 3B→7B、InternVL 4B→8B 均稳定复现三种模式；Top-k 在所有规模下均显著优于 GT。

**稳定性验证**：多轮重采样（n=50/60/70，各 1000 次）的 Wilson 95% CI 均覆盖稳定比例且不含零（Table 4）。

## 相关工作脉络
1. **Jacovi & Goldberg (2020) / DeYoung et al. (2020)**：首次在 NLP 中将忠实度形式化为需经验验证的 graded 属性，并提出 comprehensiveness 和 sufficiency 度量，本文为其向视觉模态的扩展。
2. **Jain & Wallace (2019) / Serrano & Smith (2019)**：论证注意力权重可被置换而不影响预测，质疑注意力作为解释的价值；本文通过因果扰动证实 VLM 视觉注意力在 token 级别具有真实因果贡献。
3. **Bastings & Filippova (2020)**：提出梯度 saliency 可能优于注意力解释；本文直接聚焦注意力忠实度本身，未与 saliency 对比，但指出二者可互补（Appendix B）。
4. **Abnar & Zuidema (2020)**：提出 attention rollout/flow 聚合多层注意力；本文采用直接求和聚合，避免传播假设，更适合评估视觉忠实度。
5. **Chen et al. (2024a) / Shang et al. (2025) / Zhang et al. (2025)**：利用注意力稀疏性进行 VLM token 剪枝/压缩以实现推理加速，隐含"低注意力 token 可安全丢弃"假设；本文的因果验证揭示此假设在 Mode B/C 下可能不成立。
6. **Uppaal et al. (2026)**：研究慢思考 VLM 中显式推理的视觉忠实度（黑盒方法）；本文聚焦模型内部注意力机制的忠实度（白盒因果扰动），两者研究对象和层级不同。

## 局限性与未来方向
1. **仅覆盖动态分辨率 VLM**：结论可能不适用于固定分辨率输入架构（如 LLaVA-style 模型），需进一步验证。
2. **未涵盖多步推理型 VLM**：perform multi-step deliberation 的模型可能呈现不同的视觉注意力行为，留作未来工作。
3. **全图理解类样本被排除**：约 5.6%（5/90）因 GT 覆盖几乎整图而被排除，absence detection 和 holistic scene recognition 的系统性研究有待后续探索。
4. **未与梯度 saliency 方法对比**：attention vs saliency 的相对归因质量是开放问题，可作为扩展方向。

## 研究启发与可借鉴点
1. **因果扰动范式可迁移至其他模态忠实度评估**：本文的"零消融 + Comp/Sgap 双维度"框架可直接应用于 audio-language models、multimodal reasoning models 的忠实度分析，甚至扩展到跨模态注意力（cross-attention）的评估。
2. **三模式分类为 VLM 效率优化提供理论依据**：Mode A 适合激进 token 剪枝，Mode B 需谨慎保留上下文，Mode C 应放弃局部聚焦策略——不同场景可差异化选择压缩/加速策略，而非一刀切。
3. **人类标注不应直接作为模型忠实度 ground truth**：本文揭示约 40% 场景下 human-annotated 区域与模型因果依赖存在分歧，提醒后续可解释性研究应采用因果验证而非仅相关性对比。
4. **跨架构忠实度差异启示模型设计**：InternVL 的 tile-based 设计使局部证据更集中（趋近 Mode A），而 Qwen 的全图编码倾向于分布式依赖（Mode B），未来可在视觉编码器设计时显式优化注意力忠实度模式。

## 关键术语表
**Comprehensiveness（充分性/必要性）**：衡量被高亮 token 是否为预测的必要条件，值越高说明移除这些 token 对预测的破坏越大。
**Sufficiency Gap（充分性差距）**：衡量仅保留 top-k token 后预测能力的损失程度，低值说明这些 token 自身已足够恢复预测。
**Zero-ablation（零消融）**：将选定 token 的 hidden state 置为零向量而非从序列中删除，以保留空间布局结构的同时阻断内容贡献。
**Faithful-Sufficient Mode（忠实-充分模式）**：top-k 注意力 token 既必要又充分，模型推理高度聚焦于局部证据区域。
**Faithful-Distributed Mode（忠实-分布式模式）**：top-k 注意力 token 必要但不足，模型还需依赖更广泛的视觉上下文信息。
**Non-Focal Mode（非聚焦模式）**：无局部集中的注意力必要区域，视觉信息整体作为上下文触发器，但注意力排名无法定位具体因果区域。
**Otsu's Method**：一种自动阈值分割算法，本文用于在 (Comp, Sgap) 二维平面上自适应地将样本划分为三种处理模式。

## 可复现要素
- **模型**：Qwen2.5-VL-7B（主实验）、InternVL2.5-8B（跨架构验证）、Qwen2.5-VL-3B / InternVL2.5-4B（规模扩展）；均为开源模型。
- **数据集**：VQAv2（90 样本）、VRDU（60 样本）、ChartQA（100 样本），基准公开；VQAv2 的 human-annotated GT 区域由论文团队基于 COCO instance segmentation 半自动标注（详见 Appendix A）。
- **代码/权重**：论文未明确声明代码仓库；模型权重开源（HuggingFace）。
- **关键超参**：greedy decoding（temperature=0）；token 消融位置为第一解码器层入口；k 值取 GT 区域覆盖的 token 数；Otsu 两阶段阈值分割。
- **复现难点**：零消融实现需访问 model 中间层 hidden states；GT region 标注需自行复现 COCO-based pipeline（Appendix A 有详细说明）。
