---
title: "VLA-HWBENCH-WHERE-THE-TIME-GOES-WHENVLA-MODELS-RUN-ON-EDGE-H"
source: https://arxiv.org/pdf/2610.34717v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-06 08:52:28"
field: "边缘AI系统性能评测"
keywords: ["VLA", "边缘推理", "延迟剖析", "kernel fusion", "低精度推理", "LIBERO"]
innovations: ["提出覆盖4种动作生成形式的7阶段VLA边缘推理基准VLA-HWBench", "发现延迟由动作生成迭代次数而非模型参数量主导，参数量仅解释最多15%方差", "量化编译/专用后端/低精度三层优化收益并揭示优化会转移瓶颈至CPU预处理"]
benchmarks: ["LIBERO-10"]
---

# 论文速读：VLA-HWBENCH: WHERE THE TIME GOES WHEN VLA MODELS RUN ON EDGE HARDWARE

## 一句话总结
本文提出 VLA-HWBench 基准，系统评测 18 个开源 VLA 模型在两种边缘硬件上的端到端推理延迟，将每次调用拆分为七个阶段进行细粒度分析，揭示动作生成方式（而非模型参数量）才是决定延迟的核心因素，并为边缘 VLA 的系统优化提供了实证指导。

## 研究问题与动机
1. **核心问题**：VLA 模型在边缘设备上的推理延迟从何而来？哪些因素决定延迟高低？
2. **现有研究不足一**：已有工作仅覆盖少数动作生成形式（最多 2 种，而实际有 4 种：flow matching、并行解码、token-by-token 自回归、先推理后动作），无法全面反映设计选择的开销差异。
3. **现有研究不足二**：调用阶段划分最多仅 3 段，无法定位时间消耗的具体位置（如 CPU 预处理、数据搬运、视觉编码、prefill、动作生成等）。
4. **现有研究不足三**：未在同一模型上对比不同推理优化（编译、专用后端、低精度）的收益，且各后端通常仅针对特定模型评估，缺乏跨模型、跨平台的统一视角。

## 核心贡献（创新点）
1. **提出 VLA-HWBench 统一基准**：覆盖 18 个主流开源 VLA 模型、4 种动作生成形式、2 种架构不同的边缘平台（Jetson Thor / AMD Ryzen AI Max+），以同一 LIBERO-10 观测输入评估，填补了系统性评测空白。
2. **七阶段细粒度耗时分解**：将单次推理调用拆为 CPU 预处理（S0）、host-to-device 传输（S1）、视觉编码（S2）、prefill（S3）、动作生成（S4）、后处理（S5）及残差（S6），精确归因每个阶段的开销来源。
3. **发现"动作生成迭代次数决定延迟，而非模型规模"**：通过最小二乘拟合证明参数量仅解释最多 15% 的延迟方差，而 flow matching 长 chunk 可跟上下控制率，token-by-token 和 CoT 推理则严重超限。
4. **量化三类优化的收益与瓶颈转移**：编译（torch.compile）、专用后端（FlashRT/TensorRT）、低精度（FP8/NVFP4）在 Jetson 上分别带来 2.5×/1.8×/1.6× 加速，但优化会将时间"挤"到未被加速的阶段（如 CPU 预处理占比从 12% 升至 39%）。

## 方法详解
1. **评估框架**：在 NVIDIA Jetson AGX Thor（Blackwell GPU，128 GB LPDDR5X，273 GB/s 带宽）和 AMD Ryzen AI Max+ PRO 395（RDNA 3.5 GPU，128 GB LPDDR5X-8000，256 GB/s 带宽）上运行，每个模型使用其官方 LIBERO 评估代码与锁定依赖版本，确保公平比较。
2. **七阶段计时**：在每张模型自身代码的阶段边界插入设备级同步计时器，总延迟 $T = \sum_{k=0}^{6} T_k$；动作生成阶段若含 $N$ 次迭代，则 $T_4 \approx N \cdot t$，其中 $t$ 为单次迭代开销。
3. **四种动作生成形式覆盖**：
   - Flow matching：$\pi_{0.5}$、GR00T N1.7、X-VLA、FastWAM 等，通过固定步数去噪生成动作 chunk。
   - 并行解码：OpenVLA-OFT、VLA-Adapter 等，单次 pass 生成全部动作。
   - Token-by-token 自回归：VLA-0、pi0-FAST，每生成一个 token 需一次完整 LLM 前向。
   - 先推理后动作：DeepThinkVLA、MolmoAct2-Think，包含 CoT 推理 token。
4. **优化实验设计**：在 $\pi_{0.5}$ 和 GR00T N1.7 两个拥有全部优化选项的模型上，逐层叠加四类优化——自动编译（LeRobot `torch.compile`）、专用后端（FlashRT 融合 kernel + CUDA Graph、TensorRT）、低精度（FP8 / NVFP4），记录每步加速比与阶段偏移。
5. **瓶颈分析**：使用 Roofline 模型评估各 GPU 阶段的硬件利用率，区分计算瓶颈（compute-bound）与内存瓶颈（memory-bound）；CPU 预处理阶段通过线程数变化与空闲计数器分析宿主软件开销。

## 实验与结果
- **数据集与评测任务**：LIBERO-10 benchmark，固定同一组真实观测（2 张 256×256 相机图像 + 机器人状态 + 语言指令）。
- **基线模型**：18 个开源 VLA，共 19 个参考配置（MolmoAct2 含 flow/discrete 两个动作头）。
- **延迟范围**：Jetson 上 77 ms（X-VLA）至 6.3 s（VLA-0），跨度 82×；AMD 上 114 ms 至 17.9 s，跨度 157×。仅 10/19（Jetson）和 9/19（AMD）配置能跟上 LIBERO 20 Hz 控制率。
- **参数规模相关性**：log-log 线性拟合中，参数量仅解释 Jetson 上 9%、AMD 上 15% 的延迟方差。
- **关键优化收益**：
  - $\pi_{0.5}$：编译 2.5×、后端 1.8×、低精度 1.6×，累计 7.2×（405 ms → 44 ms）。
  - GR00T N1.7：编译 1.02×、后端 1.4×、低精度 1.8×，累计 2.6×（119 ms → 46 ms）。
  - FlashRT 融合 kernel + 4-bit 权重带来最大单步增益（2.6–2.9× over compile），但仅支持 2 个模型且仅在 Jetson。
- **瓶颈分布**：
  - 优化后动作生成（少量 token）在 Jetson 上受内存带宽限制，达到峰值带宽的 50% 以上。
  - 视觉编码和长 prefill 受 unfused kernel 和低精度支持缺失限制，仅达硬件峰值的 6–42%。
  - CPU 预处理（GR00T N1.7 占 39%）受串行 Python 代码与 CPU 线程深度空闲唤醒开销限制。

## 相关工作脉络
1. **Taherin et al. (2026)**：评测 5 个 VLA 在 Jetson Orin 和数据中心 GPU 上的延迟，覆盖 2 种动作生成形式，未分析 CPU 预处理，不支持非 NVIDIA 平台。
2. **Zhou et al. (2026)**：分析 7 个策略（其中 5 个 VLA）在多加速器上的性能，划分为 3 阶段，发现 action expert 内存受限，但未跨平台对比优化收益。
3. **Vishwanathan et al. (2026)**：针对单一架构做 profiling，将最多 75% 延迟归因于动作生成，阶段划分粗、未覆盖全部动作生成形式。
4. **VLA-Perf (Jiang et al., 2026)**：解析式性能建模 $\pi_0$-like 架构，属于理论分析而非实测基准，且仅覆盖一类模型。
5. **定位差异**：VLA-HWBench 是首个同时覆盖 4 种动作生成形式、7 阶段分解、2 个异构边缘平台、并在相同模型上逐层对比多种推理优化的统一基准。

## 局限性与未来方向
1. **优化覆盖不均**：FlashRT 等高性能后端仅支持 2/18 个模型且仅限 Jetson，AMD 平台几乎无优化选项可用，泛化性受限。
2. **实时性评估不完整**：仅以 LIBERO 20 Hz 控制率为基准判定是否"跟上"，未评估不同场景（如高速动态操作）对延迟的更严格需求。
3. **输入多样性有限**：主要使用固定观测和单一 episode 的多帧测试，不同场景/动作 chunk 长度对延迟的影响未充分探索。
4. **未覆盖多模态交互开销**：如异步推理、action chunking 与控制器协同调度等系统级优化未在基准中体现。
5. **未来方向**：扩展融合后端到低功耗 ARM/x86 更多厂商硬件；将 CPU 预处理流水化或与 GPU 推理重叠；探索更少 denoising step、parallel decoding、压缩 token 等算法-系统协同优化。

## 研究启发与可借鉴点
1. **七阶段分解方法可直接迁移**：对任何端侧多模态/Agent 系统的延迟剖析均可采用"预处理→传输→编码→推理→后处理"的分阶段计时范式，精准定位瓶颈。
2. **"动作生成迭代次数优先于模型规模"的结论具有普适性**：设计高效端侧 VLA/Agent 时应优先选择少步数生成策略（并行解码、flow matching 长 chunk），而非单纯缩小模型。
3. **优化会转移瓶颈而非消除瓶颈**：GPU 侧优化后 CPU 预处理占比显著上升，提示端到端优化需 CPU-GPU 协同，不能只关注 GPU kernel。
4. **Roofline 分析用于区分软硬瓶颈**：将各阶段置于算术强度-吞吐量坐标中，可清晰判断阶段是受硬件带宽/算力限制还是软件实现限制，为后续优化指明方向。
5. **统一基准设计原则值得借鉴**：固定输入、锁定依赖版本、逐层叠加优化、交叉平台对比——这一方法论可复用于其他新兴模型族（如 VOA、多模态 Agent）的性能评测。

## 关键术语表
**VLA（Vision-Language-Action）**：直接将图像观测和语言指令映射为机器人动作的端到端深度学习模型，是通用机器人控制的前沿范式。
**Flow matching**：一种生成式建模方法，通过积分一条从噪声到数据的流来生成动作 chunk，只需固定步数去噪即可输出完整动作序列。
**Action chunk**：单次推理调用生成的动作序列（长度为 H），控制器在执行期间无需再次调用模型，chunk 越长单位动作开销越低。
**Kernel fusion**：将多个相邻算子合并为一个 CUDA kernel 执行，减少内存读写和启动开销，是 FlashRT 等专用后端的核心技术。
**NVFP4**：NVIDIA 提出的 4-bit 浮点量化格式，在保持较高数值精度的同时大幅降低权重存储和带宽需求。
**Roofline 模型**：以算术强度（FLOP/Byte）为横轴、吞吐量为纵轴的可视化性能模型，用于区分阶段受计算瓶颈还是内存带宽瓶颈限制。
**LIBERO-10**：一个用于评估机器人长期学习能力的仿真 benchmark，包含 10 个具有不同障碍物布局的任务，常作为 VLA 评测标准。
**CUDA Graph**：将一系列 GPU kernel 调用捕获为图结构以一次性提交，大幅降低 host-side 调度开销。

## 可复现要素
- **数据集**：LIBERO-10 benchmark（公开可用）。
- **代码**：18 个模型的官方仓库均公开（GitHub stars 见附录 Table 5）；FlashRT 开源（https://github.com/flashrt-project/FlashRT）；TensorRT 为 NVIDIA 商业库。
- **硬件平台**：NVIDIA Jetson AGX Thor（T5000 GPU）、AMD Ryzen AI Max+ PRO 395（规格见论文 Table 3）。
- **关键超参**：LIBERO 控制器 20 Hz；相机分辨率 256×256；Warm-up 20 次调用，计时最多 100 次；报告中位延迟与 5th–95th 百分位。
- **精度选项**：BF16（默认）、FP8、NVFP4、FP32（部分模型）。
- **编译选项**：LeRobot `reduce-overhead` / `max-autotune`；Isaac-GR00T 仅编译 action head。
- **论文未提及**：具体的量化校准数据集、FlashRT 的 FP8 scale 校准细节仅说明"在 benchmark 观测上加载时校准一次"。
