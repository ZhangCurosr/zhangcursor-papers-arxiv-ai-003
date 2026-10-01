---
title: "Sol-H3-Recursive-Self-Improvement-for-MiniMax-H3-Inference-A"
source: https://arxiv.org/pdf/2609.35110v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 22:10:50"
field: "视频生成模型推理加速"
keywords: ["video diffusion inference", "cross-resolution generation", "latent adaptation", "recursive self-improvement", "kernel fusion", "quantized communication", "edge deployment"]
innovations: ["跨分辨率两阶段生成管线替代全分辨率去噪", "学习型 latent-to-latent 跨 VAE 适配器替代昂贵 decode-reencode 循环", "基于 RSI 的自动化算子优化与精度保持 LoRA fusion"]
benchmarks: ["MiniMax-H3 baseline (SGLang)", "8xGB200 cloud node", "DGX Spark edge device", "RTX 5090 consumer GPU"]
---

# 论文速读：Sol-H3: Recursive Self-Improvement for MiniMax-H3 Inference Acceleration on Sol-Engine across Cloud and Edge

## 一句话总结
本文针对 MiniMax-H3（33B 参数音视频扩散模型）在云端和边缘侧的推理瓶颈，提出了一套全栈优化推理管线：通过跨分辨率两阶段生成调度 + 自研 latent-to-latent 适配器替代昂贵的 VAE 编解码循环，并借助递归自改进（RSI）循环自动化搜索 kernel fusion 与内存布局，在 8×GB200 上实现约 30× 加速（5 秒视频仅需 1.434s，超实时 3.5×）、内存降低约 20%，并在单卡 DGX Spark 上完成全部常驻 HBM 的推理（56.17s）。

## 研究问题与动机
- **33B 大模型推理开销巨大**：MiniMax-H3 含 330 亿参数且为多步迭代去噪的音视频联合生成模型，云端部署受生成延迟制约吞吐量与成本，边缘部署则受严格 HBM 容量约束。
- **单一优化策略无法同时覆盖云/边**：云端核心瓶颈是吞吐与延迟，边缘的核心瓶颈是内存驻留——一旦超出 HBM 就会触发 CPU offload 导致延迟雪崩。
- **现有加速方法各有局限**：算法加速（如减少采样步数、修改分辨率）往往以损质量为代价；系统加速（kernel fusion、通信调度）虽保持数值一致但不改变生成调度本身。
- **跨 VAE 的分辨率衔接成本高**：H3（24 通道、16×空间压缩）与 LTX-2.5（128 通道、32×压缩）使用独立训练的 VAE，传统方案需先 decode 到像素再 re-encode，引入大量计算与内存开销。

## 核心贡献（创新点）
1. **跨分辨率两阶段生成管线**：将原本 49 步全分辨率（1344×768）的去噪流程重构为"4 步低分辨率（672×384）全局布局 + latent handoff + 3 步高分辨率细节精修"，显著降低均匀分辨率去噪的冗余。与已有工作（如 SGLang baseline）的本质区别在于：**不仅减少了步数，还通过分辨率调度改变了整个计算轨迹**。
2. **学习型 latent-to-latent 跨 VAE 适配器（194.76M 参数）**：完全跳过 VAE decode-reencode 循环，以 63ms 延迟（vs 12.7s 的全 VAE 往返）完成 H3→LTX 坐标转换。与 Tiny AutoEncoder 或传统 VAE 方案的区别在于：**通过像素级 unshuffle 对齐与解码器感知损失，在极低参数量下实现保真转换**。
3. **基于 RSI（Recursive Self-Improvement）的自动化算子优化**：在严格数值正确性验证下自动搜索 kernel fusion、内存布局、通信量化格式，区分 bit-exact 变换与近似算术融合，使得加速收益有明确的数值边界。与 SWE-agent/AutoCodeRover 等通用代码智能体不同：**本系统被严格限制在"不改变生成契约"的范围内搜索**。
4. **精度保持的 LoRA consumer fusion**：保留 base 与 LoRA 分支并融合到下一消费层，避免 weight merge 在 BF16 中因舍入导致的有效更新丢失（6.24% vs 35.26% RMSE），在字节级等价于原始分支的同时恢复近半额外开销。与常规 weight merge 加速的本质区别在于：**不以牺牲数值精度为代价换取速度**。

## 方法详解

### 3.1 跨分辨率两阶段生成
- **Stage 1（内容+运动）**：MiniMax-H3 + FastH3 VSA DataFree LoRA，在 672×384×124 帧上做 4 步去噪，生成核心视觉元素与主音频轨道。
- **Latent Handoff**：H3 ×2 upscaler 将空间分辨率提升至 1344×768 → latent adapter 将 H3 特征映射到 LTX 坐标（128 通道、32×压缩、因果 8×时间网格）。
  - **Token 对齐**：按物理帧位置而非张量索引对齐；固定前端拼接线性插值的 H3 特征与最近邻 bin 源 token；spatial pixel-unshuffle 将 $2\times2$ 邻域展入通道，产出 $24\times(1+3)\times4 = 384$ 通道的 LTX grid。
  - **训练目标**：$\mathcal{L}_{\text{adapter}} = \text{MSE}(\hat{z}_L, z_L) + \alpha \cdot \text{MSE}(D_L(\hat{z}_L), D_L(z_L))$，其中 $\alpha=160$，仅更新残差路径，VAE 与 skip 投影固定。
- **Stage 2（细节精修）**：LTX-2.5 dev backbone（BF16，distilled LoRA 450，strength 0.8），在 1344×768 上做 3 步联合音视频更新。

### 3.2 通信优化
- **Packed QKV collective**：将 3 次 Q/K/V 交换合并为 1 次，总 collective 数从 4 降至 2；stride-aware kernel 一次写入交换缓冲区，bit-identical 于参考 permutation。
- **量化通信格式**：block-INT8 QKV（400 字节/token/head）和 FP8 wire format（128 字节 token/head，无 scale 元数据），均在 NCCL 上测得性能提升。

### 3.3 计算量化
- **MXFP8 GEMM**：对 block 2–46 的 attention 与 FFN 投影使用 E4M3 值 + 每 32 值一个 E8M0 scale；fused RMSNorm/modulation 与 SwiGLU producer 直接写入 E4M3 激活，消除中间 BF16 tensor。
- **Attention→量化输出投影复用**：attention 返回的原始 E4M3 直接作为量化输出投影输入，避免中间 BF16 转换。

### 3.4 稀疏注意力
- **Step-dependent sparsity**：refinement 阶段 3 步分别设 $\alpha=1.0, 1.25, 1.5$，逐步提高阈值增加稀疏度。
- **安全轨设计**：前两层 Transformer 保持 dense；prefix KV 块（含文本+条件视频+音频+目标视频之前的所有 token）始终作为 dense sink，保证音频保真与多模态前缀完整性。

### 3.5 Kernel Fusion 与精度保持 LoRA 融合
- **三类算子组融合**：(1) residual + indexed gating + RMSNorm + indexed scale/shift；(2) QK norm + partial RoPE；(3) SwiGLU split + SiLU + mul。
- **LoRA consumer fusion**：保留 base 与 LoRA 分支，将融合加法嵌入消费者 kernel，确保 BF16 舍入点与原分支一致，字节级等价于 native separate branches。

### 3.6 VAE 解码优化
- **Context 并行分片 + 全局 tile batching**：196 个 tile 分布于 8 rank，全局批量调度减少 launch 次数；编译后 decoder 峰值内存从 18.8GB 降至 15.6GB。

### 4.1–4.3 边缘侧关键优化
- **AdaLN precompute**：预计算整个 modulation table 并缓存，替换 26GB adaln\_proj 权重，节省约 24GB HBM。
- **Prompt caching**：Stage 2 使用离线编码的通用 refinement prompt（INT8 Gemma encoder），不再在线加载 Stage-2 text encoder（节省 16.2 GiB）。
- **Weight quantization**：FP8 H3 draft DiT（23.4 GiB）+ NVFP4 AWQ Qwen prompt encoder（14.6 GiB）确保全部模块常驻 HBM。

### 5. Reference KV Caching
- 首步完整计算 reference KV 并缓存，后续步复用缓存 KV，不做刷新；属近似方法但视觉差异微小。

## 实验与结果
- **数据集/评测设置**：5s/10s/15s 视频（1344×768，24fps），同一 prompt/seed；cross-VAE 适配器在 256 个 held-out 测试视频上评估 PSNR/SSIM。
- **GB200 云端（Tab.1）**：
  - 8×GB200，5s：Base H3 18.25s → Sol-H3 1.434s（**12.73×**），约 3.5× 超实时；15s：99.51s → 6.06s（**16.41×**）。
  - 1×GB200，5s：129.90s → 9.48s（**13.71×**）；10s：376.94s → 24.31s（**15.51×**）。
- **DGX Spark 边缘（5s 5s 1344×768）**：1740.81s → 56.17s（**≈31×**）；内存从 142.4 GiB 降至 116.9 GiB（**-17.9%**），全程驻 HBM，无 CPU offload。
- **RTX 5090 消费级**：1045.40s → 39.80s（**26.3×**）。
- **Cross-VAE 适配器质量（Tab.3）**：最终 checkpoint PSNR 30.618 dB，SSIM 0.8962；较早期 checkpoint 提升 1.205 dB（95% CI [1.166, 1.245]）。
- **跨 VAE 转换延迟（Tab.4）**：Adapter 63.1ms vs 全 VAE 往返 12,748.6ms（**202× 加速**）；内存 0.785 GiB vs 13.79 GiB。
- **Reference KV 缓存（Tab.2）**：最重度条件（2 video）达 1.92× 加速；1 image → 1.14×，1 video → 1.41×。
- **LoRA Consumer Fusion（Tab.5）**：字节级等价于 native；恢复 46.3% 额外耗时，但仍慢于 merged weights 3.24%（后者产生不同数值输出）。

## 相关工作脉络
1. **SGLang / Base H3 服务基线**：原 MiniMax-H3 发布管线与 SGLang serving baseline 采用全分辨率 49 步单阶段去噪；本文提出两阶段跨分辨率调度，从根本上改变了计算轨迹。
2. **Cascade/Multi-resolution 生成范式（如 SDXL、Pyramid Flow）**：本文借鉴了"低分辨率起草+高分辨率精修"的已有范式，但贡献在于完整的 serving 实现与端到端执行栈（含 latent adapter、通信、量化）。
3. **Sparse Attention 系列（Sparse VideoGen、SageAttention、DraftAttention、PISA）**：本文在 refinement 阶段使用 Sol-Attn 在线阈值稀疏注意力，并首次将"prefix dense + target sparse"的安全轨设计应用于多模态音视频模型。
4. **Diffusion Caching（TeaCache、EasyCache、Pyramid Attention Broadcast、TaylorSeer）**：此类方法通过减少步数或缓存去噪计算来加速；本文的参考 KV 缓存与其正交，专门针对"重条件"（参考图/视频）场景。
5. **量化加速（ViDiT-Q、PTQ4DiT、Q-DiT、SVDQuant、AWQ）**：本文使用 MXFP8 GEMM 并联合优化量化通信，强调算术融合与 bit-exact 变换的区分报告。
6. **Sol 视频推理引擎（Li et al., 2026）**：本文是 Sol 框架在单一模型族（MiniMax-H3）上的深度应用，并新增"fail-closed 数值契约"作为优化约束，区别于无约束的自改进智能体。

## 局限性与未来方向
- **近似方法的累积误差**：两阶段分辨率调度、latent adapter、参考 KV 缓存均为近似技术，虽然单点质量损失小，但在更长视频或多轮级联中误差可能累积。
- **适配器的泛化能力待验证**：当前 adapter 仅在 H3→LTX-2.5 的特定 VAE 组合上训练，换用其他模型/VAE 需重新训练。
- **RSI 搜索的计算成本**：自动化 kernel 搜索需要大量 profiling 与验证轮次，部署门槛较高。
- **仅评估了 NVIDIA Blackwell 系列硬件**：未测试 AMD MI300X 等其他架构，通用性存疑。
- **Ref2VA 的质量评估以人类/VLM 偏好为主**，缺乏标准客观指标（如 FVD）的对比。

## 研究启发与可借鉴点
1. **"人类架构设计 + RSI 自动优化"的分层范式**：宏观算法（两阶段调度、latent adapter）由人设计，微观算子（fusion、布局、通信格式）由 RSI 搜索；这一人机协同模式可迁移到其他大模型推理优化场景。
2. **Latent-to-latent 跨 VAE 适配器设计**：通过 pixel-unshuffle 对齐不同 VAE 的时空 grid，并联合 latent MSE + decoded pixel MSE 训练，可有效替代昂贵的 decode-reencode 循环——此思路可推广至任意双 VAE 级联架构。
3. **Precision-preserving LoRA consumer fusion**：在 BF16 下保留 LoRA 分支并融合到消费者层，避免了 weight merge 的数值漂移，同时恢复近半额外开销；可直接复用到任何带 LoRA adapter 的 DiT 推理管线。
4. **Step-dependent 稀疏注意力调度**：根据去噪步数动态调整 sparsity 强度（早期保守、后期激进），在保证全局结构的前提下逐步释放计算预算，这一策略适用于多数多步扩散模型。
5. **Reference KV caching 用于重条件生成**：针对参考图/视频场景的首步 KV 缓存策略，可无缝接入任何支持 cross-attention 的扩散 Transformer，对多参考条件工作负载（如视频编辑）带来近 2× 加速。

## 关键术语表
- **MiniMax-H3**：MiniMax AI 发布的 33B 参数开源音视频联合扩散生成模型，支持文本到视频+音频生成。
- **LTX-2.5**：Lightricks 发布的高分辨率音视频扩散模型，本文作为 Stage 2 精修阶段使用。
- **Cross-Resolution Two-Stage Generation**：将去噪过程分为低分辨率全局布局阶段与高分辨率细节精修阶段的两阶段生成策略。
- **Latent-to-Latent Adapter**：在 H3 与 LTX VAE 的隐空间之间学习的轻量级特征映射模块（194.76M 参数），替代传统的像素级 decode-reencode 循环。
- **Recursive Self-Improvement (RSI)**：一种在严格数值正确性验证下自动搜索 kernel fusion、内存布局和通信格式的训练/优化循环框架（源自 Sol 引擎）。
- **MXFP8 GEMM**：基于 Open Compute Project 微缩放格式的 8-bit 矩阵乘，以 E4M3 值 + 每 32 值一个 E8M0 scale 存储，用于 attention 和 FFN 投影加速。
- **Sol-Attn**：无训练的在线块稀疏注意力方法，通过动态阈值选取 KV 块进行计算，配合 step-dependent 调度控制稀疏度。
- **LoRA Consumer Fusion**：保留 base 与 LoRA 分支、在消费者层融合加法操作的优化技术，保证 BF16 下字节级等价于原始分支。

## 可复现要素
- **数据集**：训练用 80,000 视频（8 个归档），最终 adapter 训练使用 65,536 对 $1344\times768$ 配对视频；评测用 256 个 held-out 测试视频。论文未提及是否公开训练集。
- **代码/权重**：论文提供了 GitHub 链接（Github Code）与 Project Page；latent adapter 权重与 H3/LTX-2.5 模型权重均未在文中明确说明开源状态，需访问项目页面确认。
- **关键超参**：Stage 1 步数=4、分辨率 672×384；Stage 2 步数=3、分辨率 1344×768；稀疏注意力 $\alpha \in \{1.0, 1.25, 1.5\}$；adapter 学习率峰值 $8\times10^{-5}$、$\alpha_{\text{decoded}}=160$；LoRA distilled strength=0.8。
