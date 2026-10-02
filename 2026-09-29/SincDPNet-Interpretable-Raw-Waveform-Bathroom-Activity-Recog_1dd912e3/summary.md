---
title: "SincDPNet-Interpretable-Raw-Waveform-Bathroom-Activity-Recog"
source: https://arxiv.org/pdf/2609.34907v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 01:13:23"
field: "边缘音频事件识别与可解释 TinyML"
keywords: ["acoustic event classification", "raw-waveform learning", "SincNet", "depthwise-separable convolution", "ambient assisted living", "TinyML", "interpretability", "multi-objective Bayesian optimization"]
innovations: ["将可学习 sinc 带通前端（2Nf 个频率参数）与深度可分离卷积体结合，构建仅数千参数的可解释原始波形分类器", "提出 SnaanGhar7 卫生间声学数据集（7 类、21387 片段、5 环境、环境不交叠划分），覆盖水事件/门/助行器/非目标类", "以训练前信号级声学分析（pairwise Bhattacharyya 重叠）对齐 learned filter 响应与混淆矩阵，建立可解释误差归因框架"]
benchmarks: ["SnaanGhar7 (environment-disjoint)", "LOEO 五折交叉验证", "Raspberry Pi Zero 2 W 硬件推理 benchmark"]
---

# 论文速读：SincDPNet-Interpretable-Raw-Waveform-Bathroom-Activity-Recog

## 一句话总结
本文提出了 SnaanGhar7 数据集（7 类卫生间声学事件，21,387 个标注片段，来自 5 个真实环境）及 SincDPNet 模型——一个仅含数千参数、前端使用可学习 sinc 带通滤波器的紧凑原始波形分类器，在环境不交叠划分下实现最高 80.2% 准确率 / 0.760 macro-F1，兼顾边缘部署与声学可解释性。

## 研究问题与动机
1. **公开卫生间声学数据集匮乏**：ESC-50、UrbanSound8K、AudioSet 等主流数据集缺少卫生间场景覆盖；已有相关工作多集中于水相关事件，缺乏门、助行器等辅助生活关键事件及非目标异质类。
2. **可解释性与紧凑性的双重需求**：传统 Mel/MFCC 基方法准确率高但特征不可解释；标准原始波形卷积层需大量自由参数且无法直接映射到物理频率，不利于隐私敏感场景下的误差归因。
3. **环境迁移鲁棒性不足**：若采用随机 clip 级划分，同一录音室的混响、管道特征会同时出现在训练集和测试集，导致对"真实泛化"的过高估计。
4. **TinyML/边缘资源约束**：实际部署要求模型极小（KB 量级）、推理低延迟；现有工作多从大模型压缩而来，缺乏从小预算出发的结构化设计探索。

## 核心贡献（创新点）
1. **SnaanGhar7 数据集**：收录 7 类卫生间活动声学事件（含未知类），21,387 个 1.5s 窗口，5 环境录音、会话/环境双隔离划分、公开代码与基线，弥补了现有人群数据集对 mobility-aid 事件与非目标类的覆盖空白。
2. **SincDPNet 架构**：将 SincNet 的可学习 sinc 带通前端（仅需 $2N_f$ 个频率参数）与深度可分离卷积体结合，使完整模型仅约 2,848–14,040 参数，约为同族 SincNet 的 1/28。
3. **声学先验驱动的可解释误差分析**：训练前通过 Bhattacharyya 重叠、类内 Spread 等信号级指标刻画类间相似度，并在训练后与 learned filter 响应、混淆矩阵定量关联（pairwise 相关 $\rho=0.39$），首次系统解释卫生间声学事件的混淆根源。
4. **多目标贝叶斯优化设计探索**：以 24 个候选配置在 accuracy–size / accuracy–latency 两个 Pareto 前沿上给出可复现的操作点选择，证明该任务的性能瓶颈在于前端结构而非容量。

## 方法详解
**可学习 sinc 带通前端**
- 每个滤波器由两个频率参数（下截止 $f_1$ 与带宽 $b$）参数化，对应理想矩形频响的两个 sinc 函数之差：
  $$g[n;f_1,f_2] = 2f_2\operatorname{sinc}(2\pi f_2 n) - 2f_1\operatorname{sinc}(2\pi f_1 n)$$
- 通过 $f_1^{(i)} = f_{\min} + |\hat{f}_1^{(i)}|$、$f_2^{(i)} = \min(f_1^{(i)} + f_{\min} + |\hat{b}^{(i)}|, f_s/2)$ 约束正值与 Nyquist 上界；加 Hamming 窗减少频谱泄漏。
- 前端可训练参数总量仅 $2N_f$（与核长 $L$ 无关），$N_f=25$ 时仅 50 个参数，滤波后的幅度经 BN→ReLU→时间 max-pool 得到 $N_f$ 条带通通道。

**深度可分离卷积体**
- 将 $N_f$ 条通道重塑为 $\mathbf{X}\in\mathbb{R}^{N_f\times T\times 1}$ 的"类谱图"，送入 $B$ 个 DS-conv 块（DS-conv → BN → ReLU → 可选 2×2 pool）。
- 每块通道数 $C_b = \min(\alpha 2^b, 8\alpha)$，宽度乘子 $\alpha$ 控制容量；全局平均池化后接线性分类头与 softmax。
- 损失为标准交叉熵；DS-conv 参数量仅为同维标准卷积的 $\frac{1}{C_{\mathrm{out}}}+\frac{1}{k^2}$ 倍（$k=3$ 时约为 1/9）。

**多目标贝叶斯优化（MOBO）**
- 设计变量：$N_f\in[16,128]$、$L\in\{101,251,401\}$、$\alpha\in[2,12]$、$B\in[3,6]$、峰值 LR $\eta$、pooling stride $S$。
- Study A 优化 $(\text{macro-F1}, \text{模型大小})$，Study B 优化 $(\text{accuracy}, \text{推理延迟})$；使用 qEHVI 采集函数 + Matérn-5/2 核，24 次评估预算。
- 预计算质量 coreset（按 SNR/过载率/静默率/谱平坦度/类内离群度加权，每类取 top 25%）加速搜索，但所有最终报告模型均用全量训练集重新训练。

**信号级诊断与困难度评分**
- 训练前计算每类的 RMS、ZCR、峰度、静音比例、能量变异性、两两 Bhattacharyya 重叠，构造归一化加权和 $D_c = \sum_k w_k \tilde{s}_{k,c}$ 作为类难度假设（仅用于描述，不直接预测错误）。

## 实验与结果
**数据集**：SnaanGhar7，7 类（Flush/Shower/Bathroom Tap/Basin Tap/Door/Walker-Crutch/Unknown），21,387 个窗口（1.51125 s，20 kHz 单声道），5 环境（E01–E05，宿舍/住宅/机构），最大类间不平衡比 1.2×。

**评估协议**：环境不交叠划分（E01–E03 训练，E04 验证，E05 测试）+ LOEO 五折交叉；所有模型在相同划分下比较。

**基线**（6 个）：DS-CNN（MFCC，18K 参数）、CRNN（MFCC，237K）、TinyCNN（MFCC，94K）、SincNet（Raw，132K）、ACDNet（Raw，22K）、DPNet（Raw，4.9K）。

**主要结果（环境不交叠，三种子 seed 均值±标准差）**：

| 模型 | 参数量 | Acc.(%) | Macro-F1 | MCC |
|---|---|---|---|---|
| SincNet（最强基线） | 132,327 | 76.7 ± 1.8 | 0.712 ± 0.013 | 0.721 ± 0.023 |
| **SincDPNet best-F1** | **14,040** | **80.2 ± 3.2** | **0.760 ± 0.019** | 0.716 ± 0.052 |
| SincDPNet rank-1（加权最优） | 3,408 | 75.6 ± 1.8 | 0.673 ± 0.029 | 0.714 ± 0.020 |
| **SincDPNet compact ($N_f=25$)** | **2,848** | 75.7 ± 3.8 | 0.661 ± 0.049 | **0.716 ± 0.042** |

- 相对同体量的 DPNet（4,944 参数），SincDPNet compact 在 Macro-F1 上 +0.049（0.612→0.661）、MCC +0.071（0.645→0.716），参数量反降 42%。
- 最强模型 SincDPNet best-F1（14,040 参数）在 Acc.、Bal. Acc.、Macro-F1 三项均居首位。
- **Raspberry Pi Zero 2 W 部署**：compact 模型单 clip 推理 435.6 ms，RTF = 0.29（远低于 1 的实时阈值）；best-F1 模型 RTF = 0.61，仍可实时运行。

**消融**：可学习 sinc 前端优于固定 sinc 前端（+0.007）、无约束卷积前端（+0.036）与 MFCC 前端（+0.015）；去掉幅度增强导致 macro-F1 下降 0.026；增大 $N_f$ 收益递减（16→32→64→128 时分别为 0.613/0.628/0.663/0.672）。

**错误分析**：97.6% 的误差归因于两类声学现象——瞬态/冲击重叠（774 例，主要是 Door↔Walker/Crutch 的强不对称混淆）与宽带水流动模糊（406 例，Flush/Shower/Bathroom Tap/Basin Tap 互混）。单类难度评分 $D_c$ 与错误的相关性弱（$\rho=-0.07$），而 pairwise Bhattacharyya 重叠与混淆对角外条目呈中等正相关（$\rho=0.39$）。

## 相关工作脉络
1. **SincNet (Ravanelli & Bengio, 2018)**：最早提出参数化 sinc 带通前端的原始波形语音识别模型，后续层使用标准卷积+全连接，约 $10^5$ 参数；本文保留其前端思想但用 DS-conv 体大幅压缩后端。
2. **DPNet (Chowdhury et al., 2026)**：团队先前的小 footprint 卫生间声学模型，使用通用可学习卷积前端 + 深度可分离体；本文对比表明 sinc 前端可带来 Macro-F1 +0.049 的提升，并减少 42% 参数。
3. **DS-CNN / Hello Edge (Zhang et al., 2017)**：MCU 级关键词识别的经典轻量模型，基于 Mel 频谱+DS-conv；本文将其纳入 MFCC 基线，验证相同架构在原始波形上经 sinc 结构化的收益。
4. **ACDNet (Mohaimenuzzaman et al., 2023)**：面向极端资源约束环境的端到端原始波形管道；其宏 F1（0.639）显著低于带 sinc 约束的前端版本，凸显前端结构在数据有限场景的关键性。
5. **Öztürk et al. (2025)**：公开卫浴声学数据集（~460 min，11 细粒度事件），但缺少门与助行器事件、非目标类及原始波形基线；SnaanGhar7 在其基础上扩展了 assistive-living 相关类别并给出环境不交叠评测。
6. **DCASE 低复杂度任务（2021–2025）**：规定模型预算（128 k INT8 参数等），广泛使用 DS-conv、量化、感受野设计；本文模型仅占 DCASE 预算的 ~1/10 即达到有竞争力的 MCC，展示 TinyML 场景下的设计空间。

## 局限性与未来方向
1. **Basin Tap 极度难学**（召回仅 8.2%，最佳模型提升至 85.2%）：信号层面与其他水流类高度重叠且自身类内变异大，单类困难评分未能预警。
2. **Door↔Walker/Crutch 瞬态混淆未被解决**：两类共享短宽带冲击特征，现有的时间 max-pool 丢弃了部分精细时序信息；文中自述需更长上下文或二级分类器。
3. **数据集规模与场景覆盖有限**：5 个环境、3 位参与者、约 385 分钟原始录音，跨设备/跨房间泛化仍需在更多变体上验证。
4. **未做模型量化部署**：当前硬件实验仅在 FP32 TFLite 进行，INT8 量化及 MCU 级部署留作未来工作。
5. **单段 1.5s 片段无序列建模**：无法利用卫生间活动的时序连贯性（如 shower→flush），难以支持"长时间静止异常"等高级辅助生活检测。

## 研究启发与可借鉴点
1. **"先验声学分析先行"的设计范式**：在训练前通过 signal-level 指标（类内 spread、pairwise Bhattacharyya 重叠、谱质心/平坦度等）预判困难类，并反向指导前端频段分辨率的配置——这一流程可直接移植到其他小规模音频分类任务。
2. **sinc 参数化前端的极致压缩效果**：$2N_f$ 个频率参数即可替代整段卷积核，将可解释性与参数效率统一；在低数据量场景下，"结构化前端 > 无约束大容量"是一个可靠的设计原则。
3. **环境/会话双隔离划分作为泛化基准**：将 recording session 与 environment 作为划分的原子单元，避免 room-specific 特征泄漏；对于任何室内声学识别任务均可作为更严格的评估协议。
4. **质量 coreset 加速 MOBO 搜索**：用复合质量分数（SNR、过载、静默、谱平坦、类内离群）选取每类 top 25% 样本进行架构搜索， rank 相关保持良好；该策略可复用于其他计算昂贵的 TinyML 设计空间探索。
5. **可解释性即产品**：learned filter bank 的直接 Hz 可读性配合混淆矩阵联动分析，能够回答"哪些事件会被误判及为什么"——这一套可解释性流程对辅助生活等安全关键应用具有示范意义。

## 关键术语表
- **SincDPNet**：结合可学习 sinc 带通前端与深度可分离卷积体的紧凑原始波形分类器，前端仅需 $2N_f$ 个频率参数。
- **SnaanGhar7**：7 类卫生间声学活动数据集，21,387 个标注窗口，覆盖水事件、门、助行器及非目标类，5 环境环境不交叠划分。
- **Learnable Sinc Filter Bank**：以低频截止 $f_1$ 和带宽 $b$ 为可调参数的带通 FIR 滤波器组，每个滤波器仅 2 个可训练参数，响应在 Hz 上可直接读取。
- **Depthwise-Separable Convolution (DS-conv)**：将标准卷积分解为逐通道深度卷积与 1×1 点卷积，参数量降至约标准卷积的 $1/C_{\mathrm{out}}+1/k^2$。
- **Environment-Disjoint Split**：以录音环境为单位划分训练/验证/测试，确保未见环境下的真实泛化评估。
- **Multi-Objective Bayesian Optimization (MOBO)**：以高斯过程为代理模型、qEHVI 为采集函数，在 accuracy–size 或 accuracy–latency 二维目标空间搜索 Pareto 最优架构。
- **Bhattacharyya Coefficient (BC)**：基于高斯近似的两分布重叠度量，取值 [0,1]，越大表示两类越难区分。
- **Real-Time Factor (RTF)**：推理时间与音频片段时长之比，RTF < 1 表示模型可在音频捕获间隙完成推断。

## 可复现要素
- **数据集**：SnaanGhar7，需通过 Dataset Access Form 申请获取（论文声明）。
- **代码/模型**：完全开源，仓库 https://github.com/debolina-34/SnaanGhar7，含模型实现、配置与实验脚本。
- **关键超参（compact 配置）**：采样率 20 kHz；clip 长 30,225 样本（1.51125 s）；hop 10,225（66% 重叠）；$N_f=25$，$L=401$，$\alpha=6$，$B=4$；峰值 LR $10^{-3}$–$3\times10^{-2}$；pooling stride 2/4/8；幅度增强 ±25%（仅训练集）。
- **训练细节**：class-balanced categorical cross-entropy；3 个随机 seed 取均值±标准差；Sinc 前端在 TFLite 转换前冻结为等价的固定卷积核。
- **评估平台**：Raspberry Pi Zero 2 W（512 MB RAM，1 GHz，TFLite FP32）；Intel Core U5 笔记本作为参考。
