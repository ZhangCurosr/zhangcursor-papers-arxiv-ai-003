---
title: "VLA-HWBENCH-WHERE-THE-TIME-GOES-WHENVLA-MODELS-RUN-ON-EDGE-H"
source: https://arxiv.org/pdf/2610.34717v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-06 08:47:43"
field: "机器人学习的边缘推理优化"
keywords: ["VLA", "边缘推理", "延迟分析", "推理优化", "低比特量化", "基准评测"]
innovations: ["首个覆盖四种动作生成形式的18模型VLA边缘推理基准", "七阶段细粒度调用分解首次计时CPU预处理", "同模型上逐步叠加编译/后端/低精度的归因分析方法"]
benchmarks: ["LIBERO-10"]
---

# 论文速读：VLA-HWBENCH-WHERE-THE-TIME-GOES-WHEN-VLA-MODELS-RUN-ON-EDGE-HW

## 一句话总结
本文提出了 VLA-HWBench，在两种边缘硬件平台上对 18 个开源 VLA 模型进行端到端推理延迟的系统基准测试与七阶段分解；核心发现是**推理延迟主要由动作生成方式（迭代次数）而非模型参数量决定**，内核融合与 4-bit 权重可提供最大加速（2.6–2.9×），但仅支持极少数模型和平台。

## 研究问题与动机
- **现有研究未能覆盖 VLA 动作生成的四种主要形式**（flow matching、并行解码、token-by-token 自回归、推理后生成动作），仅 5 个模型、2 种形式，难以指导实际部署。
- **调用周期划分过于粗糙**（最多 3 个阶段），无法定位时间究竟消耗在哪一步；CPU 预处理等关键阶段未被计时。
- **不同推理优化在相同模型上缺乏公平对比**：每个后端仅在其目标模型上评估，无法回答"哪种优化最有效、能缩短哪一步、能否迁移到其他模型/平台"。
- **现实部署需求紧迫**：VLA 用于闭环机器人控制时，观测信息随时间变 stale，边缘端低延迟推理对隐私和弱网场景尤为关键。

## 核心贡献（创新点）
1. **首个覆盖四种动作生成形式的 VLA 边缘推理基准**：评测 18 个开源 VLA 模型（基于 HF 下载量与社区星标筛选），涵盖 flow matching（9）、并行解码（4）、自回归 token（3）、推理后动作（2）四种形式，与先前最多覆盖 2 种形成本质区别。
2. **七阶段细粒度调用分解**：将单次推理拆为 S0–S6 七阶段（CPU 预处理、H2D 拷贝、视觉编码、prefill、动作生成、后处理、残余），首次对 CPU 预处理计时，能精确归因时间开销。
3. **同一模型上逐步叠加三种优化进行归因**：对 π0.5 和 GR00T N1.7 两种代表性模型，依次开启编译、专用后端（FlashRT/TensorRT）、低精度（FP8/NVFP4），量化每步加速与时间迁移量。
4. **结合屋顶线模型（Roofline）的后优化瓶颈分析**：区分硬件受限（内存带宽）与软件受限（未融合 kernel、串行 CPU 预处理），给出未来边缘硬件与推理软件的设计建议。

## 方法详解
- **基准设计原则**：固定输入（同一 LIBERO-10 观测）、固定 timing boundary（主机侧观测到主机侧动作 chunk）；各模型使用其作者 pin 的库版本，以 eager PyTorch 为参考基线，可选优化逐个叠加。
- **七阶段定义**（$T = \sum_{k=0}^{6} T_k$）：S0 CPU 预处理（resize、normalize、tokenize）；S1 H2D 拷贝；S2 GPU 视觉编码；S3 GPU prefill（语言模型 KV cache）；S4 GPU 动作生成（N 次迭代的 denoising 步骤/自回归 token 生成）；S5 CPU 后处理；S6 残余（Python 调度等）。
- **指标定义**：Q1 用阶段占比 $T_k/T$、端到端延迟 $T$、每动作耗时 $T/H$（LIBERO 控制器 20 Hz，阈值 50 ms/动作）；Q2 用加速比 $T_0/T$；Q3 用屋顶线模型（计算 bound $F/P$ vs 内存 bound $W/B$）区分硬件/软件瓶颈。
- **平台**：NVIDIA Jetson AGX Thor（Blackwell GPU，2560 CUDA cores，128 GB LPDDR5X 273 GB/s）与 AMD Ryzen AI Max+ PRO 395（RDNA 3.5，40 CUs，128 GB LPDDR5X-8000 256 GB/s），ARM vs x86 宿主架构对比。
- **后端**：LeRobot 0.6.1（eager + `torch.compile`）、各模型作者代码、FlashRT（融合 kernel + FP8/NVFP4，仅 Jetson）、TensorRT（仅 Jetson）。
- **测量协议**：每配置 20 次 warm-up + 最多 100 次计时（≥20 次），取中位数；逐阶段设备同步计时；步成本通过 $T = a + bN$ 线性拟合获得。

## 实验与结果
- **数据集/评测环境**：LIBERO-10 固定观测（两张 256×256 相机图像+机器人本体状态+语言指令），另跑同一 episode 的 24 帧考察输入变化影响。
- **Q1 结果**：Jetson 上 14/19 配置的动作生成占比最高（42–99%）；延迟跨度 **82×**（77 ms X-VLA → 6.3 s VLA-0），AMD 上 **157×**（114 ms → 17.9 s）；参数量仅解释 ≤15% 方差；**10/19**（Jetson）和 **9/19**（AMD）配置能跟上 20 Hz 控制率。
- **Q2 结果**：对 π0.5，编译/后端/低精度分别带来 2.5×/1.8×/1.6× 加速，合计 **7.2×**（318 ms → 44 ms）；GR00T N1.7 合计 **2.6×**（119 ms → 46 ms）；**FlashRT（kernel 融合 + 4-bit）提供最大单步增益（2.6–2.9× over compilation）**，但仅支持这两种模型、仅在 Jetson；优化后 CPU 预处理从 GR00T N1.7 调用的 12% 升至 39%。
- **Q3 结果（瓶颈分析）**：动作生成在 Jetson 上为**内存带宽受限**（denoising 步骤仅处理 10–41 tokens，已达峰值带宽的 75%+）；prefill 和视觉编码为**软件受限**（未融合 kernel、缺少低精度支持）；CPU 预处理受限于**串行 Python 和 idle thread wake-up**。AMD 上视觉编码因 RDNA 3.5 无 FP32/BF16 矩阵单元，性能距屋顶线 8–17×。
- **最强结果**：π0.5 + FlashRT NVFP4 在 Jetson 上 **44 ms/调用**（8.1 ms/动作，远超 50 ms 阈值）；X-VLA 参考配置 Jetson **77 ms**（最快参考配置）。

## 相关工作脉络
1. **Taherin et al. (2026)**：评测 5 个模型在 Jetson Orin 和数据中心 GPU 上，仅 2 种动作生成形式，无 CPU 预处理计时；本文扩展至 18 模型 × 4 形式 × 7 阶段。
2. **Zhou et al. (2026)**：profile 7 个策略（5 个 VLA）于多加速器，发现 action expert 内存受限；本文首次在更多模型上验证并补充 CPU 侧瓶颈。
3. **Vishwanathan et al. (2026)**：单架构分析，发现 action generation 最多占 75% 延迟；本文证明占比可达 99%，且形状因动作生成形式而异。
4. **VLA-Perf (Jiang et al., 2026)**：对 π₀-like 架构做解析性能建模；本文通过实测覆盖更广泛的模型族与优化组合。
5. **推理后端研究**：FlashRT、TensorRT、Embodied.cpp、vla.cpp 等仅针对特定模型；本文在同一模型上公平对比，揭示优化的可迁移性局限。
6. **量化/低精度工作**（LLM.int8、OPTQ、AWQ 等）：主要针对 LLM；本文验证 NVFP4 在 VLA 上保持 0.93+ cosine similarity，为机器人场景提供实证依据。

## 局限性与未来方向
- **优化覆盖有限**：Kernel 融合 + 4-bit 后端（FlashRT）仅支持 π0.5 和 GR00T N1.7，其余 16 个模型无法受益。
- **平台覆盖有限**：AMD 平台仅支持编译优化，后端替换（FlashRT/TensorRT）和量化仅测试于 Jetson；跨平台泛化结论不足。
- **静态评估**：未涉及异步推理（如 Jetson-PI）、chunk 级实时切分、或动作与感知的流水线重叠。
- **输入单一化**：固定 LIBERO-10 观测，不同场景下的延迟分布仅通过 24 帧考察部分变化，缺乏大规模真实部署数据。
- **未探索更多低比特格式**：NVFP4 虽有效，但对更多模型格式的适配与精度-延迟权衡未系统研究。

## 研究启发与可借鉴点
1. **七阶段分解方法论可迁移**：对任何端侧 AI 系统（LLM serving、多模态模型），均可套用"主机侧 → GPU 各 kernel 组 → 主机侧"的阶段划分，精确定位瓶颈。
2. **"动作生成迭代次数比参数量更重要"的结论**：对 VLA/agent 模型设计有直接指导价值——未来应优先减少 denoising steps 或转向并行解码，而非单纯缩小模型。
3. **屋顶线分析+逐 kernel 类分解**的结合方式：先算理论 bound（F/P、W/B），再拆解实际 kernel 时间占比，可复用于其他边缘 ML 系统的性能诊断。
4. **优化逐步叠加归因的设计**：在同一模型/权重/输入上逐层开启优化并记录时间迁移，为后续消融研究提供了可复用的实验范式。
5. **CPU 预处理被忽视的瓶颈**：本文为端侧部署敲响了"主机侧串行 Python 和线程管理"的警钟，可与团队在 host-GPU 协同调度方向结合。

## 关键术语表
- **VLA（Vision-Language-Action）**：直接将视觉观测和语言指令映射到机器人动作的多模态大模型。
- **Flow Matching**：一种生成式扩散模型技术，通过流匹配头在固定步数内去噪生成动作块（action chunk）。
- **Action Chunk**：单次推理调用输出的连续动作序列（长度 H），控制器在执行 H 步后再调用模型。
- **LIBERO-10**：机器人学习基准，包含 10 个长horizon 操作任务，控制器运行频率为 20 Hz（50 ms/步）。
- **FlashRT**：面向 VLA 的低延迟推理运行时，使用融合 kernel 和 CUDA Graph，支持 FP8/NVFP4 权重。
- **Roofline Model**：性能分析框架，以算术强度（F/W）为横轴，绘制计算力和内存带宽两条上限线，判断阶段是否受硬件限制。
- **NVFP4**：NVIDIA 定义的 4-bit 浮点格式，用于 VLA 权重量化，牺牲少量精度换取显著带宽降低。
- **Prefill**：语言模型一次性处理所有输入 token 并填充 KV cache 的前向传播阶段。

## 可复现要素
- **数据集**：LIBERO-10（公开），论文使用固定观测帧（作者代码中有提供）；**公开**。
- **代码**：论文未声明统一基准代码库开源，但各模型的 LIBERO 评测代码链接在 Appendix Table 5 列出（GitHub/HuggingFace）。
- **权重**：18 个模型的 LIBERO checkpoint 均在 HuggingFace 提供，链接见附录 Table 5。
- **关键超参**：LIBERO 控制器频率 20 Hz；warm-up 20 次，计时最多 100 次取中位数；每调用 90 s 截断；两平台统一使用相同 LIBERO-10 观测。
- **平台规格**：Jetson AGX Thor（CUDA 13.2, PyTorch 2.11）；Ryzen AI Max+ PRO 395（ROCm 7.2, PyTorch 2.14）；论文未提及 Docker 容器或具体环境变量。
