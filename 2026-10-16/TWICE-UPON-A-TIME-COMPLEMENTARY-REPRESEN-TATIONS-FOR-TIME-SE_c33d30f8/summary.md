---
title: "TWICE-UPON-A-TIME-COMPLEMENTARY-REPRESEN-TATIONS-FOR-TIME-SE"
source: https://arxiv.org/pdf/2610.22327v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 17:18:32"
---

# 论文速读：TWICE UPON A TIME: COMPLEMENTARY REPRESENTATIONS FOR TIME-SERIES REASONING

## 一句话总结
本文提出“互补表征”设计原则，将保留绝对量纲的折线图与剔除正仿射振幅变化的归一化延迟嵌入作为双视图共同输入语言模型，在不牺牲数值读取能力的同时显著增强时序动力学推理；基于 Qwen3-VL-8B 的双塔模型在 TSRBench 达到 53.2% 准确率，超越所有已评测专用 TSLM。

## 研究问题与动机
1. 现有 TSLM 通常仅向语言模型暴露单一时序表示，强制统一归纳偏置，难以同时满足“定量读数需保留原始测量框架”与“动力学推理需抑制仿射扰动”的矛盾需求。
2. 保留绝对振幅与坐标轴的表征利于精确数值提取，而经相位空间重构与归一化的表征能凸显局部时序结构并具备尺度不变性，二者各自擅长不同任务。
3. 通用 VLM 直接渲染折线图已证明具备较强基线能力，但仅在基础感知上表现稳定，在高级推理、预测与决策任务上仍有较大提升空间。
4. 核心疑问：能否通过引入携带不同不变性特征的互补视图，替代“单一全能表示+辅助元数据”的旧范式，系统性提升通用 TSLM 的综合推理能力？

## 核心贡献（创新点）
1. **确立互补表征为 TSLM 设计原则**：将定量读数与结构感知解耦为两个视图，区别于现有工作依赖单一表示或在外挂辅助元数据提供尺度信息的做法。
2. **提出双塔 dual-view TSLM 架构**：折线图沿用冻结的 Qwen3-VL 原生视觉编码器，延迟嵌入引入专用 DINOv3 编码器，并通过 DeepStack 实现多级特征融合，区别于以往仅做输入层拼接或独立后处理头的方案。
3. **构建严格的对照与干预评估范式**：引入同视图控制（Plot+Plot / Delay+Delay）与评估时仿射/噪声/重采样扰动分析，量化证明性能提升源于表征内容的互补而非额外通道容量。
4. **以 8B 参数取得当时 TSLM 最强综合表现**：TSRBench 53.2%（较折线图单视图提升 8.0pp，领先最强专用基线 36.7%）、TSExam 89.9%、ChatTS 多变量数值对齐 86.8%，逼近 GPT-5 (T+V) 的 55.6%。

## 方法详解
- **双视图构造**：(a) 折线图视图 $V_i^{\text{plot}}$：原始观测值直接渲染为二维线图，保留绝对振幅、坐标轴范围与量纲，多变量序列独立绘图各自保留测量框架；(b) 归一化延迟嵌入视图 $V_i^{\text{delay}}$：基于 Takens 定理，将长度 $m$ 的序列切分为 $W$ 个重叠窗口（窗口长 $\ell = \min(\max(\lfloor \rho m \rfloor, \ell_{\min}), \ell_{\max})$），每窗口线性插值至 $H$ 点后按列堆叠为 $H \times W$ 矩阵并作 min-max 归一化。因归一化与线性插值，任意变换 $a\mathbf{x}+b\ (a>0)$ 均产生相同图像，从而剔除正仿射振幅与偏移。
- **双塔编码器与融合**：折线图由冻结的 Qwen3-VL 视觉编码器处理，2×2 patch 合并后生成 230 个 plot token 及 3 层中间特征图；延迟嵌入由引入 LoRA 的 DINOv3 ViT-L/16 编码器处理，分别于 block 6/12/18 抽取中间特征，经独立投影模块 $M^{(i)}$ 生成 3 组 DeepStack 多级特征与 64 个 delay token。两流 token 以 `<|vision start|>`/`<|vision end|>` 包裹后拼接送入 Qwen3-VL-8B 主干，多级特征直接加到语言模型前 3 层的对应 token 位置。
- **两阶段训练**：Stage I（模态对齐）冻结 VLM 与 LM，仅训练 DINOv3 LoRA 与投影模块，使用 ~5.5 万 caption 数据（CaTS-Bench、ChatTS-style、结构生成描述）使延迟流与 LM 对齐；Stage II（指令微调）合并 LoRA 并冻结全部视觉路径，对 LM 施加 LoRA，在 ~18.3 万指令数据上微调。损失为标准自回归 next-token prediction，仅覆盖答案 token。
- **数据与监督**：全合成数据采用 gold-by-construction 策略，标签由生成器参数或精确函数直接导出；训练集与 TSRBench/TSExam/ChatTS 评测集严格隔离，无重叠序列或题目。

## 实验与结果
- **数据集与基线**：TSRBench（4125 题，15 任务，涵盖 Perception/Reasoning/Prediction/Decision）、TSExam（746 题）、ChatTS（单/多变量分类与数值对齐）。基线包含通用 VLM（Qwen3-VL-8B/32B、GPT-5）与专用 TSLM（ChatTS-14B、TS-Reasoner-7B、TimeOmni-1-7B）。
- **主结果**：本文模型 TSRBench 总体准确率 53.2%，较折线图单视图（45.2%）提升 8.0pp，大幅领先最强专用 TSLM（36.7%）；TSExam 89.9%；ChatTS-B 多变量数值对齐 86.8%。
- **消融与干预**：Plot+Plot（44.8%）与 Delay+Delay（42.6%）同视图控制未能复现增益，证明提升来自表征内容互补而非双流容量；延迟视图主要拉升推理（+4.3pp）、预测（+23.7pp）与决策任务，感知任务基本持平；评估时干预显示延迟视图对仿射变换几乎不变（±0.8pp），折线图视图敏感，双视图模型在保持较高绝对准确率的同时呈现差异化鲁棒性。附录 C.5 显示 STFT 作为次级视图在 ChatTS 数值对齐上略优，但 TSRBench 上延迟嵌入更强。
- **跨基准泛化**：在 TSExam、ChatTS-A/B 及 TSRBench 均保持领先，表明双视图设计具备跨任务通用性。

## 相关工作脉络
1. **专用 TSLM（ChatTS、OpenTSLM、TimeOmni-1、TS-Reasoner）**：采用单一时序表示（文本序列化、1D patch、TSFM 特征等）并依赖辅助元数据提供尺度
