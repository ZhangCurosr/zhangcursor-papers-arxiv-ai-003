---
title: "SPIDER-Multi-Layer-Semantic-Token-Pruning-and-Adaptive-Sub-L"
source: https://arxiv.org/pdf/2609.34977v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 10:12:40"
---

# 论文速读：SPIDER: Multi-Layer Semantic Token Pruning and Adaptive Sub-Layer Skipping in Multimodal Large Language Models

## 一句话总结
SPIDER 是一个免训练的 MLLM 推理加速框架，通过联合优化视觉编码器的多层语义 Token 剪枝（MSV-Prune）与 LLM 解码器的自适应子层跳过（ASL-Skip），在 LLaVA-NeXT-7B 上将 FLOPs 降低 79% 的同时保持 96% 的基线性能。

## 研究问题与动机
1. **双重冗余并存**：MLLM 推理开销来自视觉输入的数据冗余（数千个 Patch Token）与 LLM 解码过程的计算冗余，现有工作往往只侧重其一。
2. **末层剪枝的语义偏差**：主流 Token 剪枝方法仅依赖视觉编码器最后一层特征计算重要性，忽略了中间层在细粒度、物体中心语义上的独特保留能力，易误删对计数、精确定位、OCR 等任务关键的视觉碎片。
3. **层跳过粒度过粗**：现有加速方法通常将整个 Decoder Block 整体跳过或冻结，未区分同一 Block 内 Attention 与 FFN 对视觉 Token 进化的差异化贡献。
4. **模块孤立缺乏统一视角**：剪枝与跳过常被独立设计，未从“Token 效用随模型深度演进而衰减”的一致视角进行级联分配。

## 核心贡献（创新点）
1. **提出 MSV-Prune**：首次系统性利用视觉编码器中间层与深层特征进行语义聚类与互补 Token 选择，本质区别于仅依赖末层或纯文本-视觉注意力的剪枝范式。
2. **提出 ASL-Skip**：量化 Attention 与 FFN 子层的贡献异质性，设计“在线累积证据+离线 KL 画像”的自适应跳过机制，决定每个 Token 是否跳过、何时跳过以及跳过哪个子层，区别于 ShortV 等整块替换策略。
3. **构建统一免训练加速框架**：将编码器侧数据冗余消除与解码器侧计算冗余分配耦合为深度依赖的效用分配流程，无需梯度微调即可直接嵌入 LLaVA、Qwen 等主流架构。
4. **提供可训练升级路径**：在 LoRA 微调后 SPIDER 性能可达 99.5%，证明其策略与参数微调具有正交兼容性。

## 方法详解
**MSV-Prune（多层语义 Token 剪枝）**
- **锚点 Token 选取**：基于视觉编码器自注意力分数 $\mathbf{a}_v$（[CLS] 行或平均入度），通过动态阈值 $\tau$ 选出占比 $r$ 的高显著性锚点 $\mathbf{T}_v^{\text{anc}}$。
- **多层语义聚类与互补 Token 选取**：对非锚点 Token 在深层特征 $\mathbf{F}^L$ 上做 K-means 聚类，按簇大小比例分配配额 $N_k$。簇内采用多层相似度 $S_{ij}$ 筛选最具代表性 Token：
  $$S_{ij} = \underbrace{\sin(\mathbf{F}_i^L, \mathbf{F}_j^L) + \sin(\mathbf{F}_i^M, \mathbf{F}_j^M)}_{\text{多层簇内相似度}} + \underbrace{\max_{p \in \mathbf{T}_v^{\text{anc}}}\sin(\mathbf{F}_i^L, \mathbf{F}_p^L) + \max_{p \in \mathbf{T}_v^{\text{anc}}}\sin(\mathbf{F}_j^L, \mathbf{F}_p^L)}_{\text{与锚点相似度}}$$
  每簇保留得分最低的 $N_k$ 个 Token 构成 $\mathbf{T}_v^{\text{cmp}}$，最终与文本 Token 拼接送入 LLM。

**ASL-Skip（自适应子层跳过）**
- **在线可跳过分数 $S_{sa}$**：融合隐状态的词汇空间信息熵 $E_{ii}$ 与图文相关性 $F_{itc}$（均为 [0,1] 归一化余弦/熵值）：
  $$\mathbf{S}_{sa}(i,\ell) = \mathrm{ReLU}\big(1 - [\mathbf{E}_{ii}(i,\ell) + \mathbf{F}_{itc}(i,\ell)]\big)$$
  熵越低、图文相关性越弱，跳过信号越强。
- **离线子层贡献度 SLC**：对每层 $\ell$，分别用稀疏层替换 Attention 或 FFN 后计算输出 Logit 的 KL 散度，归一化为 $\mathrm{SLC}_{\text{module}}^{\text{Norm}}$。每层选择归一化分最高的子层作为跳过目标 $m(\ell)$。
- **累积决策**：在前 $L/2$ 层融合分数 $\mathrm{S}_{fuse}=w_1 \mathbf{S}_{sa}+w_2 \mathrm{SLC}^{\text{Norm}}$ 逐层累加为 $\mathrm{S}_{skip}(i)$，超阈值 $T_{\text{skip}}$ 后 Token 进入 Skip Mode，目标子层 $m(\ell)$ 在该层之后永久绕过。$w_2$ 固定为 1，$w_1/w_2$ 控制在线/离线信号比重。

## 实验与结果
- **数据集与基线**：GQA、VQAv2、MME、TextVQA、POPE、MMB、MMVet、MMStar、DocVQA；基线覆盖 FastV、VTW、ShortV、VisPruner、SparseVLM、VisionZip、HiPrune、PruneSID、IVC-Prune、ERASE。
- **核心定量结果**：
  - LLaVA-NeXT-7B 约 50% TFLOPs 下，SPIDER 聚合准确率 **99.11%**，优于 VisPruner (98.81%) 与 ShortV (96.67%)。
  - LLaVA-NeXT-7B 激进压缩至约 20% TFLOPs 时，SPIDER 仍保留 **96.09%** 性能，而 ShortV 跌至 63.56%、FastV 为 81.89%。
  - MSV-Pr
