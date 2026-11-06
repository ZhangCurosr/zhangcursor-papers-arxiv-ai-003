---
title: "Waypoint-Analytically-Modeling-GEMM-for-CPUs"
source: https://arxiv.org/pdf/2610.34914v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-06 22:23:44"
field: "高性能计算/编译器自动调优"
keywords: ["GEMM", "analytical modeling", "tile sizing", "CPU performance", "LLM inference", "Roofline model", "libzenmm"]
innovations: ["提出 Waypoint-Strict 作为比 Roofline 更紧的 GEMM 性能上界，同时参数化硬件与输入 shape", "提出 Waypoint-Tile 并推导寄存器压力与 load-to-fma 比两个不等式以压缩 tile 搜索空间", "基于 OpenEvolve 演化出 18851× 加速的快速启发式 tile 搜索算法"]
benchmarks: ["AMD Zen 5 EPYC 9755", "AMD Zen 6 EPYC 9996", "AOCL-BLAS", "AOCL-DLP", "SFC", "AMD-PACE LLM inference"]
---

# 论文速读：Waypoint-Analytically-Modeling-GEMM-for-CPUs

## 一句话总结
本文提出了 **Waypoint**，一种参数化的解析性能模型，可同时感知硬件架构特征与输入矩阵形状，自动搜索最优 tile 划分参数以最大化 GEMM 性能；并基于该模型实现了 libzenmm 库，在 AMD Zen 5/Zen 6 CPU 上达到与 SOTA 库相当甚至更优的 GEMM 及 LLM 推理性能。

## 研究问题与动机
- **核心问题**：高性能 GEMM 实现依赖精细的 tile 划分，但现有 SOTA 库（AOCL-BLAS、AOCL-DLP、SFC）的 tile 参数多为针对硬件固定调优或仅适配某类 shape（如 tall-skinny），缺乏对**具体输入矩阵形状**的自适应能力。
- **Roofline 模型的不足**：传统 Roofline 将性能简单划分为 compute-bound 或 memory-bound，忽略指令级并行、缓存层次交互、多线程带宽争用等实际因素，导致对真实性能的估计过于宽松（upper bound 偏乐观）。
- **硬件设计者的预测需求**：厂商提出新 ISA 扩展（AMX/ACE/APX）后，需能**在流片前**准确预估可实现性能，而现有模型要么过于理想化，要么依赖实测 trace 数据。
- **LLM 推理的 shape 多样性**：现代 LLM 推理会触发大量"odd shape" GEMM（非常规 M/N/K 比例），手工调参难以覆盖全部组合，亟需自动化 shape-adaptive 方案。

## 核心贡献（创新点）
1. **Waypoint-Strict**：一种比 Roofline 更紧的 GEMM 性能上界模型，参数化于硬件配置与输入 shape，理论证明其对任意合法 op 满足 $L^r(op) \leq L(op)$，实验误差几何均值仅 1.096×。
   - 本质区别：Roofline 假设纯 compute-bound 或纯 memory-bound，Waypoint-Strict 同时精确建模寄存器零化/加载/存储开销、有效内存带宽及多线程 overhead。

2. **Waypoint-Tile**：在 Waypoint-Strict 基础上引入三层 tile 参数 $(M_r, N_r)$、$(M_c, N_c, K_c)$、$(M_t, N_t)$，精确估算缓存容量 miss 与 reuse overhead，从而为给定 shape 搜索最优 tile 配置，预测最佳 tile 的性能误差几何均值仅 1.012×。
   - 本质区别：此前解析模型（如 [24,26,31]）仅参数化硬件，本文进一步参数化**输入矩阵尺寸**，实现 shape-aware 的 tile 选择。

3. **libzenmm 库 + 快速启发式搜索**：用 Waypoint-Tile 自动为每个 (shape, 硬件) 组合选择 tile 参数；引入 OpenEvolve（AlphaEvolve 开源版）演化出 Fast Heuristic，搜索速度比穷举快 18851×，且最大延迟偏差仅 1.94%。
   - 本质区别：区别于 Libxsmm 等 JIT 库依赖手工 seed 或网格搜索，本文的启发式由 AI 自动生成并逼近穷举最优。

4. **LLM 推理端到端加速**：将 libzenmm 集成至 AMD-PACE 框架，在 Zen 5/Zen 6 上对 prefill-heavy 工作负载获得 **1.36×–1.98×** TTFT 加速，高 batch 下提升更显著（batch=128 时 Zen 6 + Qwen 3 8B 达 9.15×）。
   - 本质区别：之前工作仅关注 GEMM 层面，本文首次将 shape-adaptive GEMM 与 LLM 推理框架打通并报告端到端收益。

## 方法详解
### 2.1 Waypoint-Strict（实现无关的上界）
硬件配置参数 $\mathcal{H}$ 包含：
- 核心指令倒数吞吐 $\tau_{fma}$ 与指令维度 $\delta_{fma}$（支持 AMX/ACE/SME 等矩阵指令）
- 寄存器数量 $N_{reg}$、位宽 $W_{reg}$
- 各级缓存大小 $S_{L1/L2/L3}$ 与访问吞吐 $T_{L1/L2/L3}$
- 每通道峰值带宽 $D_{bw}$、通道数 $D_n$
- 最大核数 $C_{max}$、频率 $f$

对 GEMM op（$C = A \cdot B$，尺寸 $M \times N$，K 维，T 线程，数据类型位宽 $W_d$）：

$$
L(op) = \max\left(\frac{L_c(op)}{op.T}, L_m(op)\right) + L_{mt}(op)
$$

- **计算延迟 $L_c$**：$N_{fma} = M \cdot N \cdot K / \delta_{fma}$，并叠加零化寄存器 $L_z$、加载输入 $L_{din}$、存储输出 $L_{dout}$ 的开销。
- **内存延迟 $L_m$**：
$$
L_m(op) = \frac{(M \cdot K + K \cdot N + M \cdot N) \cdot W_d \cdot f}{D_{bw} \cdot d_n}
$$
其中 $d_n \leq D_n$ 为实际可用通道数，随线程跨 CCD 时降低。
- **多线程开销 $L_{mt}$**：当每线程工作集超出 L1+L2+L3 之和时，多余数据 $\Delta$ 被重复从内存加载：
$$
L_{mt} = \min(\Delta \cdot M_t, \Delta \cdot N_t) \cdot \frac{1}{D_{bw} \cdot d_n}
$$

**理论保证（Theorem 2.1）**：对任意 op 和硬件，$L^r(op) \leq L(op)$，即 Waypoint-Strict 始终比 Roofline 更紧。

### 2.2 Waypoint-Tile（针对 tiled 实现的精化模型）
引入三层 tile：
- 寄存器 micro-tile $(M_r, N_r)$
- 宏 cache-tile $(M_c, N_c, K_c)$
- 多线程 tile $(M_t, N_t)$

总计算延迟分解为 micro-kernel 与 macro-kernel 两部分：
$$
L_c = L_\mu \cdot N_\mu + L_\omega \cdot N_\omega
$$

**Micro-kernel 开销与两个关键不等式**：
- **寄存器压力下界（Lemma 2.2）**：
$$
\frac{A_{reg} + B_{reg}}{C_{reg}} \geq \alpha
$$
- **Load-to-FMA 延迟比上界（Lemma 2.3）**：
$$
\frac{(A_{reg} + B_{reg}) \cdot \tau_{load}}{C_{reg} \cdot \tau_{fma}} \leq 1
$$
违反任一条件会导致 register spilling 或 load 无法被流水线隐藏，需从 L2 重加载，产生 reuse overhead。这两个不等式将 $(M_r, N_r)$ 的搜索空间压缩至极少候选（图 2）。

**Macro-kernel 容量 miss 代价（Algorithm 2）**：
$$
L_\omega = \text{CapacityMissCost}(H, T, S_{A_c} + S_{C_c}) + \beta \cdot \text{CapacityMissCost}(H, T, S_{B_c})
$$
其中 $\beta \in [0,1]$ 反映 B 矩阵因固定步长流式访问时的预取效果；A/C 矩阵无预取优势故 $\beta=1$。

**Packing 策略**：将输入打包为 $(M_p \times K_p)$ 块，$M_p = \text{lcm}(\text{all } M_r)$，$K_p = \text{lcm}(\text{all } K_r)$，保证每次加载完整 cache line 且步长固定，无需为不同 tile 定制 packing。

### 2.3 快速启发式搜索（OpenEvolve）
- Seed 程序来自 6 类先验启发式（数据类型/维度比/L1-L2 缓存 bound/Multi-thread split/算术强度/AI 指令/当前最优）。
- 在 100 轮遗传进化后，得到的 Fast Heuristic 搜索速度比穷举快 **18851×**，最大延迟偏差 **1.94%**。

## 实验与结果
- **硬件平台**：AMD Zen 5 (Turin) EPYC 9755 128C / Zen 6 (Venice) EPYC 9996 256C，固定频率，FP32/BF16/INT8 三种数据类型。
- **评测 shape**：51 种唯一 GEMM shape（涵盖 Small <512、Medium 512-4096、Large >4096、Tall-skinny、Fat-short、Odd、LLM 专用 shape），共 306 次比较。
- **基线**：AOCL-BLAS（手写微内核）、AOCL-DLP（JIT 微内核）、SFC（Space Filling Curve）。

| 指标 | 结果 |
|---|---|
| Waypoint-Strict vs 实测 | 几何均值误差 **1.096×**，相比 Roofline 最多收紧 **3.1%** |
| Waypoint-Tile 预测最佳 tile 性能 | 几何均值误差 **1.012×**（≈1.2%） |
| Waypoint-Tile 排名 Top-1 命中率 | **88.7%** 在 5% 内，**62.7%** 严格最优 |
| libzenmm vs AOCL-BLAS（含 packing） | **1.168×** 更快（纯 GEMM 10.8% 更快） |
| libzenmm vs AOCL-DLP（含 packing） | **1.136×** 更快（纯 GEMM 0.85% 更慢） |
| libzenmm vs SFC（含 packing） | **1.210×** 更快（主要源于 packing 成本低 2.44×） |
| LLM 端到端 TTFT 加速 | **1.36× – 1.98×**（batch=128 时 Zen 6+Qwen3-8B 达 9.15×） |
| ACE 未来硬件性能预测 | 8×/16×/32× 理论 throughput 仅映射到 7.21×/12.84×/21.47×（BF16），揭示 compute 加速后 kernel 转 memory-bound 的瓶颈 |

## 相关工作脉络
1. **GotoBLAS / BLIS**：层状分块 GEMM 的奠基工作，采用手工调优 tile 参数；Waypoint 用解析模型替代人工调参并加入 shape 维度。
2. **自动调优库（ATLAS、OpenTuner、IrGEMM）**：基于网格/搜索的实验调参；Waypoint 避免昂贵的 empirical 搜索，直接解析推导。
3. **解析建模（Low et al. [24]、Martínez et al. [26]、IrGEMM [31]）**：早期解析模型仅参数化硬件或特定 shape 类别；Waypoint 同时参数化硬件与**任意输入尺寸**。
4. **Space-Filling Curve（SFC [13]）**：通过特殊数据布局减少通信；Waypoint 的模型可准确预测 SFC 性能并在 packing 成本上大幅优于它。
5. **Roofline 及其扩展（ECM、instruction-level 模型、DeLTA for GPU）**：多依赖 trace 或面向单一架构；Waypoint 纯解析、跨 x86/ARM 可移植，且无需 trace。
6. **tritonBLAS [29]**：GPU 侧的 GEMM 解析模型；本文将其思想推广至 CPU，并进一步覆盖多线程与 cache 层次。

## 局限性与未来方向
- **多线程调度假设简化**：当前假设均等静态分 tile，未考虑动态调度；实验发现某些 shape 用少于标称线程数反而更快，需更智能的 thread management。
- **固定频率假设**：模型假设 CPU 频率恒定，实际运行时频率会波动，可延伸引入动态频率模型。
- **单指令集假设**：假定全程使用最优 FMA/VNNI 指令，未探索 mixed-instruction 策略。
- **缓存模型简化**：仅区分各级缓存访问延迟，未建模 inclusivity/write-back/replacement policy 等细节。
- **未覆盖 ARM 实测**：理论可移植至 ARM SME/FP-AE，但未在 ARM 平台上验证。

## 研究启发与可借鉴点
1. **双阶段建模思路**：先建立实现无关的严格上界（Waypoint-Strict），再针对具体实现精化（Waypoint-Tile），这种"上下界夹逼"策略可在其他 kernel 优化（如 conv、attention）中复用。
2. **用理论不等式压缩搜索空间**：Lemma 2.2/2.3 将寄存器 tile 从组合爆炸降至 handful 候选，启示在其他内核参数搜索中也可推导类似 structural bound。
3. **AI 驱动的启发式演化（OpenEvolve）**：将 AlphaEvolve 用于系统参数搜索并达成 18851× 加速，证明 LLM/code-gen 方法可切实替代手工调参。
4. **packing 通用化设计**：用 lcm 统一 packing 块尺寸而非为每种 tile 定制，避免了"调参收益被 packing 成本抵消"的陷阱，值得在 tensor core 编程中借鉴。
5. **未来硬件预评估能力**：Waypoint 可用于 ACE/AMX 等 ISA 扩展的性能预研，提示团队在架构评审阶段即可引入解析建模辅助决策。

## 关键术语表
- **GEMM**：General Matrix Multiplication，通用矩阵乘法，形如 $C = \alpha A B + \beta C$，是 HPC 与 ML 中最基础的算子。
- **Tile / Tiling**：将大矩阵划分为若干小 BLOCK 分别计算，以提升 cache 复用率与指令级并行。
- **Micro-kernel**：GEMM 最内层循环，通常用手工汇编实现，负责固定尺寸 $(M_r \times N_r)$ 块的计算。
- **Macro-kernel**：外层循环块，负责将 micro-kernel 组织为覆盖整矩阵的计算任务，主要引入 cache-miss 开销。
- **Instruction reciprocal throughput ($\tau_{fma}$)**：一条指令平均需要多少周期才能发射一次，反映流水线吞吐上限。
- **Operational Intensity**：计算量（FLOPs）与内存访问量（bytes）之比，用于判断 kernel 是 compute-bound 还是 memory-bound。
- **Capacity Miss**：工作集超出缓存容量导致的 miss，与 spatial/temporal locality 失效相关。
- **AMX / ACE / APX**：Intel/AMD/x86 ECAG 提出的矩阵扩展指令集（Advanced Matrix Extensions / AI Compute Extensions / Advanced Performance Extensions），支持大块矩阵乘累加。

## 可复现要素
- **数据集**：51 种自定义 GEMM shape（附录 Table 5 完整列出），非公开数据集。
- **代码/权重**：论文未提及 libzenmm 开源；对比库 AOCL-BLAS、AOCL-DLP、SFC、AMD-PACE 均有 GitHub 仓库。
- **关键超参**：tile 搜索空间由 Lemma 2.2/2.3 约束；OpenEvolve 迭代 100 轮；频率固定（Zen 5 2.7 GHz / Zen 6 2.5 GHz）；每 shape 10 warm-up + 100 次测量取 best。
- **硬件**：AMD Zen 5 EPYC 9755（512 MB L3）、Zen 6 EPYC 9996（1 GB L3）。
