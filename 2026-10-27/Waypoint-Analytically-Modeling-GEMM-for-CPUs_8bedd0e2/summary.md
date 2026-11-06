---
title: "Waypoint-Analytically-Modeling-GEMM-for-CPUs"
source: https://arxiv.org/pdf/2610.34914v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-06 22:23:43"
field: "高性能计算/编译器自动调优"
keywords: ["GEMM", "analytical modeling", "tile optimization", "CPU performance", "LLM inference", "Roofline model", "autotuning"]
innovations: ["提出 Waypoint-Strict 实现无关解析模型，严格优于 Roofline 上界", "提出 Waypoint-Tile 模型，通过 register pressure 和 load-to-fma ratio 剪枝自动搜索最优三层层级 tile 参数", "构建 libzenmm 库并集成 OpenEvolve 快速启发式搜索，实现 shape-adaptive 自适应 GEMM"]
benchmarks: ["AOCL-BLAS", "AOCL-DLP", "SFC", "AMD-PACE LLM推理", "Zen 5/Zen 6 GEMM shape基准"]
---

# 论文速读：Waypoint-Analytically-Modeling-GEMM-for-CPUs

## 一句话总结
论文提出了 Waypoint，一个全新的分析模型，能够根据给定硬件架构和输入矩阵形状自动识别 GEMM 的最优分块（tile）大小；基于该模型构建了 libzenmm 库，在 AMD Zen 5/Zen 6 上实现了与 SOTA GEMM 库相当的性能，并加速了 CPU 上的 LLM 推理（1.36×–1.98×）。

## 研究问题与动机
- **现有 GEMM 库的 tile 大小不能自适应输入形状**：SOTA 库（如 AOCL-BLAS、AOCL-DLP、SFC）的 tile 参数要么针对特定硬件手工调优，要么针对某一类 shape（如 tall-and-skinny），无法随应用实际输入的矩阵维度动态适配。
- **传统 Roofline 模型对 realistic 性能预测过松**：Roofline 仅区分 compute-bound/memory-bound 两种情形，忽略寄存器开销、多核带宽争用、cache miss 等实际微架构效应，预测值往往高于实际可达成性能。
- **硬件设计师缺乏准确估计未来 ISA 扩展性能的工具**：如 AMX、ACE、APX 等新指令集提案推出时，缺乏能刻画"增加计算单元后可能使 kernel 转为 memory-bound"这种非线性缩放的分析模型。
- **分析建模与实证搜索之间的鸿沟**：此前工作 [24, 26, 31] 的分析模型只能给出硬件参数化的上界，缺乏将 tile 参数（register tile / macro tile / thread tile）与内存层级交互相结合以指导自动搜索的方法。

## 核心贡献（创新点）
1. **Waypoint-Strict 模型**：提出一个比 Roofline 更紧致的实现无关性能上界，同时参数化硬件配置与应用输入形状，严格证明 $L^r(op) \le L(op)$（定理 2.1），并在 compute-bound 和 memory-bound 两种场景下分别引入额外计算开销与有效带宽折减。
2. **Waypoint-Tile 模型**：在 Waypoint-Strict 基础上引入三层次 tile 参数（register macro/thread），精确建模 micro-kernel 的 reuse overhead 与 macro-kernel 的 capacity miss 成本，导出了 register pressure（引理 2.2）和 load-to-fma latency ratio（引理 2.3）两个剪枝条件，大幅缩减最优 tile 搜索空间。
3. **libzenmm 库**：基于 Waypoint-Tile 输出构建的自动 GEMM 库，支持 exhaustive search 与基于 OpenEvolve 发现的 fast heuristic search（快 18851×，延迟偏差最大 1.94%）两种 tile 选择策略，综合 packing + GEMM 性能分别优于 AOCL-BLAS / AOCL-DLP / SFC 16.7% / 13.6% / 21.0%。
4. **LLM 推理端到端加速**：将 libzenmm 集成到 AMD-PACE，在多个 batch size、输入长度、模型（OPT 6.7B、Qwen 3 8B）上获得 TTFT 1.36×–1.98× 加速，并在 Zen 6 新硬件上展现更强增益潜力。

## 方法详解

### Waypoint-Strict（实现无关性能上界）
硬件配置 $H$ 参数化：FMA 指令往返时间 $\tau_{fma}$、指令维度 $\delta_{fma}$、寄存器数 $N_{reg}$、各级 cache 容量 $(S_{L1}, S_{L2}, S_{L3})$ 与往返时间 $(T_{L1}, T_{L2}, T_{L3})$、单通道峰值带宽 $D_{bw}$、通道数 $D_n$、最大核数 $C_{max}$、频率 $f$。

GEMM 延迟分解为：
$$L(op) = \max\left(\frac{L_c(op)}{op.T},\; L_m(op)\right) + L_{mt}(op)$$

- **计算延迟 $L_c$**：$N_{fma} = M\cdot N\cdot K / \delta_{fma}$，加上寄存器清零、输入加载、输出存储等辅助指令开销；支持矩阵指令（AMX/ACE）的 $m_{fma}\times k_{fma}\times n_{fma}$ 维度推广。
- **内存延迟 $L_m$**：$L_m(op) = \frac{(MK+KN+MN)\cdot W_d \cdot f}{D_{bw}\cdot d_n}$，其中 $d_n \le D_n$ 为实际可用通道数，刻画多线程下跨 CCD 的带宽争用。
- **多线程开销 $L_{mt}$**：当线程所需数据超出 L1+L2+L3 总容量时，$\Delta = S_{A_t}+S_{B_t}-(S_{L1}+S_{L2}+S_{L3})$ 的部分被重复从内存访问，引入 $L_{mt}=\min(\Delta\cdot M_t,\;\Delta\cdot N_t)/(D_{bw}\cdot d_n)$。

**理论保证（定理 2.1）**：对任意 GEMM 操作和硬件，$L^r(op) \le L(op)$，即 Waypoint-Strict 始终比 Roofline 更紧。

### Waypoint-Tile（分块实现性能预测）
引入三层 tile 参数：
- **Register tile** $(M_r, N_r)$：micro-kernel 内层循环输出维度。
- **Macro tile** $(M_c, N_c, K_c)$：外层数据移动粒度。
- **Thread tile** $(M_t, N_t)$：多线程划分。

总计算延迟分解为：
$$L_c = L_\mu \cdot N_\mu + L_\omega \cdot N_\omega$$
其中 $L_\mu, N_\mu$ 为 micro-kernel 单次延迟与数量，$L_\omega, N_\omega$ 为 macro-kernel 数据搬运延迟与数量。

**Micro-kernel 成本与剪枝**：
- **Reuse overhead**：当微内核所用寄存器超出 L1 命中能力时，部分 load 需从 L2 加载，产生额外开销。
- **引理 2.2（Register pressure）**：$\frac{A_{reg}+B_{reg}}{C_{reg}} \ge \alpha$，违反则寄存器溢出。
- **引理 2.3（Load-to-fma latency ratio）**：$\frac{(A_{reg}+B_{reg})\cdot \tau_{load}}{C_{reg}\cdot \tau_{fma}} \le 1$，违反则部分 load 不能被流水线掩盖。
- 两不等式将 $(M_r, N_r)$ 候选集合急剧缩小（图 2），按每次迭代从内存加载字节数排序选取。

**Macro-kernel 容量 miss 成本（Algorithm 2）**：
$$L_\omega = \text{CapacityMissCost}(H,T,S_{A_c}+S_{C_c}) + \beta\cdot\text{CapacityMissCost}(H,T,S_{B_c})$$
其中 $\beta\in[0,1]$ 为架构相关的预取因子（B 矩阵因连续 stride 访问预取效果好，A/C 矩阵较差）。CapacityMissCost 根据 working set 超出 L2 和 L3 的量分别计算从 L3→L2 和内存→L3 的搬运延迟。

### Packing 策略
libzenmm 采用统一 packing：将输入矩阵打包为 $(M_p\times K_p)$ 块，其中 $M_p = \text{lcm}(\text{所有候选 } M_r)$，$K_p = \text{lcm}(\text{所有候选 } K_r)$，保证每次访问加载完整 cache line 且 successive access 遵循固定 stride，避免因形状依赖的 tile 策略引入额外 packing 开销。

### 快速启发式搜索（OpenEvolve）
对延迟敏感场景（如 LLM 推理），使用 OpenEvolve 在 6 个 seed 启发式（基于数据类型/维度比/L1-L2 cache 边界/多线程划分/算术强度/AI 指令/基线）上进行 100 轮遗传进化，以 Waypoint-Tile 评分，生成 fast heuristic 搜索算法——比 exhaustive search 快 18851×，延迟偏差最大仅 1.94%。

## 实验与结果
- **硬件平台**：AMD Zen 5（Turin，EPYC 9755，128 核，512 MB L3，2.7 GHz）和 Zen 6（Venice，EPYC 9996，256 核，1 GB L3，2.5 GHz）。
- **数据类型**：FP32、BF16、INT8。
- **评估 shape 数**：51 个，覆盖 small（<512）、medium square（512–4096）、large（>4096）、tall skinny、fat short、odd、LLM 共 7 类（306 次对比）。
- **对比基线**：AOCL-BLAS（手工 micro-kernel）、AOCL-DLP（JIT micro-kernel）、SFC（space-filling curve 遍历）。

| 评估项 | 结果 |
|---|---|
| Waypoint-Strict vs 实测（几何均值比） | 1.096×（贴近 Roofline 上界但更紧） |
| Waypoint-Tile 预测最佳 tile vs 实测（几何均值比） | 1.012× |
| Waypoint-Tile 正确排序 top-1 tile 比例 | 88.7%（偏差<5%），严格最优 62.7% |
| libzenmm vs AOCL-BLAS（含 packing） | +16.7% |
| libzenmm vs AOCL-DLP（含 packing） | +13.6% |
| libzenmm vs SFC（含 packing） | +21.0% |
| 纯 GEMM 计算 libzenmm vs AOCL-BLAS | +10.8%；vs SFC 持平（+0.8%） |
| LLM 推理 TTFT 加速（PACE 集成） | 1.36×–1.98×（batch=128 时 Zen 5 达 3.6×，Zen 6 达 9.15×） |
| ACE 指令未来硬件性能预测（1 tile op/cycle vs Zen 5） | BFloat16: 7.21×；Int8: 7.45×（理论峰值 8×/16×/32×，实际因 memory-bound 而折减） |

## 相关工作脉络
- **GotoBLAS / BLIS** [16, 30]：奠定 tile-based GEMM 的分层实现范式；Waypoint 在此基础上增加了输入 shape 自适应和更精确的微架构建模。
- **ATLAS / PHiPAC 等实证调优方法** [7, 10]：通过网格搜索/自动调优找 tile 参数；Waypoint 以解析模型替代大量实测，同时保持 shape 适应性。
- **低精度/形状特化分析模型** [24, 26, 31]：IRGEMM [31] 针对 irregular shape 手工分类设计；Waypoint 统一建模所有 shape，无需人工规则。
- **Roofline 模型及其扩展（ECM）[18, 19]**：ECM 需执行 trace 或针对特定 ISA；Waypoint 纯解析、无 trace 依赖、参数化硬件。
- **Space-Filling Curve GEMM (SFC)** [13]：通过数据重排减少通信；Waypoint 能准确预测 SFC 性能并实现等价效果，但 packing 成本低 2.44×。
- **GPU GEMM 解析模型 (tritonBLAS) [29]**：思路类似但面向 GPU；本文将其思想推广至 CPU 多层 cache 与多线程场景。

## 局限性与未来方向
- **多线程分配策略简化**：假设静态均分 tile（equal distribution），未考虑动态调度；实验观察到某些形状使用少于满线程数反而更快，说明需要更精细的 thread management。
- **固定频率假设**：模型假设 CPU 频率恒定，实际 variable frequency（如 boost 场景）下需扩展频率动态项。
- **微架构细节抽象**：不建模 cache inclusiveness、write-back policy、replacement strategy；更复杂的 cache 模型可进一步提升精度。
- **未探索混合指令集策略**：假设始终使用最佳指令（如 AVX512 VNNI），未考虑多 ISA 混合 dispatch 的收益。
- **Fast heuristic 进化成本**：OpenEvolve 100 轮进化是一次性开销，对全新硬件仍需重新演化 seed。

## 研究启发与可借鉴点
- **"Register pressure + load-to-fma ratio"双剪枝**：两个简单不等式可将寄存器 tile 搜索空间从指数级压缩到常数级候选（图 2），方法简洁且可迁移到其他内核自动调优场景（如 matvec、conv）。
- **统一 packing 策略（LCM 块对齐）**：以所有候选 tile 尺寸的 LCM 作为 packing 粒度，避免了 shape-dependent packing 的高昂代价，对任何自适应 tile 库均有参考价值。
- **OpenEvolve 启发式进化流程**：将解析模型作为 fitness scorer，以遗传算法自动生成快速 tile 选择启发式，这一"模型驱动算法发现"范式可复用于其他系统参数搜索（如 scheduler 策略、memory allocator）。
- **内存争用的通道级建模**：将 $d_n \le D_n$ 的可用通道数作为线程数的函数建模，比单纯使用 peak bandwidth 更贴近真实，可推广到 multi-CCD / NUMA 系统的性能预测。
- **SOTA 库兼容性验证策略**：以 Waypoint-Strict 作为 upper bound、以 SOTA 库实测作为验证锚点，形成"模型预测→库实现→实测闭合"的严谨评测闭环。

## 关键术语表
**Waypoint-Strict**：与实现无关的 GEMM 性能解析上界模型，在 Roofline 基础上额外计入寄存器/加载/存储/多线程开销，严格优于 Roofline。
**Waypoint-Tile**：面向 tile-based GEMM 实现的性能预测模型，显式建模 micro-kernel reuse overhead 和 macro-kernel capacity miss 成本。
**Register pressure（引理 2.2）**：micro-kernel 中输入/输出寄存器数量比的下界约束，违反会导致寄存器溢出。
**Load-to-fma latency ratio（引理 2.3）**：加载指令总延迟与 FMA 指令总延迟之比的上界（≤1），违反会导致部分 load 无法被流水线掩盖。
**CapacityMissCost**：当 working set 超出 L2/L3 容量时，数据从更高层缓存或主存搬运所产生的延迟成本。
**libzenmm**：基于 Waypoint-Tile 自动选择 tile 参数的 GEMM 库，支持 exhaustive search 与 fast heuristic search 两种模式。
**OpenEvolve / AlphaEvolve**：基于遗传算法的 AI 驱动算法发现框架，本文用于自动演化快速 tile 选择启发式。
**ACE（AI Compute Extensions）**：x86 生态提案的新型矩阵指令集，Waypoint 可用于预测其对未来 GEMM 性能的增益（并揭示 memory-bound 瓶颈）。

## 可复现要素
- **数据集**：51 个自定义 GEMM shape（Appendix B Table 5），非公开数据集，但所有 shape 参数已列表。
- **代码**：libzenmm 及 AMD-PACE 集成代码论文未明确开源声明（AMD 内部库）；OpenEvolve [28] 为开源项目（https://github.com/algorithmicsuperintelligence/openevolve）。
- **权重**：无模型权重。
- **关键超参**：L1/L2/L3 cache 容量与延迟（由硬件提供）、$\tau_{fma}$（指令往返时间）、$N_{reg}$（寄存器数）、$D_{bw}$、$D_n$、$\beta$（预取系数，架构相关）、OpenEvolve 进化代数=100、warmup=10、measurement=100 trials。
- **硬件**：Zen 5（EPYC 9755）、Zen 6（EPYC 9996），后者为 AMD 提前提供。
