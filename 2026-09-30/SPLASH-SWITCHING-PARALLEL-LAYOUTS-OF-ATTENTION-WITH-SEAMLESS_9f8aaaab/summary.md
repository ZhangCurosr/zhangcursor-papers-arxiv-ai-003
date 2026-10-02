---
title: "SPLASH-SWITCHING-PARALLEL-LAYOUTS-OF-ATTENTION-WITH-SEAMLESS"
source: https://arxiv.org/pdf/2609.37626v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 14:22:36"
field: "大语言模型推理系统"
keywords: ["LLM serving", "attention parallelism", "tensor parallelism", "context parallelism", "data-parallel attention", "MLA", "runtime switching", "inference system"]
innovations: ["提出注意力布局所有权模型，将权重放置与KV所有权解耦，导出DOP零冗余布局", "设计基于discard/all-gather/all-to-all原语的层内重叠切换，中位开销<0.51%", "构建转换感知调度器，在四种布局间动态选取最快布局，端到端吞吐提升1.3-1.73x"]
benchmarks: ["GLM-5.3 on B200", "DeepSeek-V3.2 on H200", "GLM-5.3-Flash on DCU/BW1000"]
---

# 论文速读：SPLASH: SWITCHING PARALLEL LAYOUTS OF ATTENTION WITH SEAMLESS HANDOFF FOR LLM SERVING

## 一句话总结
SPLASH 是一种 LLM 推理 Serving 系统，能够在请求运行过程中无缝切换注意力并行布局（TP / CP / DP-attention / DOP），通过"所有权模型"将权重放置与 KV 缓存所有权解耦，使中位切换开销低于单步耗时的 0.51%，端到端吞吐较固定布局提升 1.3–1.73×。

## 研究问题与动机
- **负载动态性与布局静态化的矛盾**：Agent、RL-rollout 等场景下，一批请求在生命周期内会从大量短请求演变为少数超长请求，单一布局无法在所有阶段最优。
- **现有切换方案代价过高**：已有引擎切换布局需排空 batch 并重启 worker，最耗时的长请求主导阶段恰好是最需要切换的时刻。
- **MLA/GQA 改变了权重与缓存的耦合关系**：现代注意力（MQA、GQA、MLA）大幅减少 KV head 数，使"权重分片方式"与"每个请求缓存所属 rank"成为两个独立决策，但系统层面尚未被形式化利用。
- **缺乏覆盖全部布局的转换支持**：Shift Parallelism 等前作仅限 TP↔序列并行等少数配对，且切换仍需阻塞；DOP 布局更是现有引擎从未实现的零冗余形态。

## 核心贡献（创新点）
1. **注意力布局所有权模型**：将四种布局统一抽象为"权重放置"和"请求缓存所有权"两个独立维度，推导每种布局的单 rank 内存足迹、最长可行上下文、以及 A²=12 种有向切换各自复用/缺失的状态集与所需原语（discard / all-gather / all-to-all）。
2. **提出 DOP（解耦所有权并行）布局**：对现有三布局组合的穷举自然导出的全新零冗余布局——按 TP 分片投影权重、按 DP-attention 每个请求独居单一 owner，在给定显存下比 DP-attention 多释放 12.69 GiB/GPU 用于 KV（GLM-5.3 FP8、T=8）。
3. **SPLASH 在线切换系统**：利用 discard/all-gather/all-to-all 三种原语在单个 serving step 内分 layer 重叠传输缺失状态，最多只有一层处于传输中，完整切换端到端增加 0.02–11.76 ms（p50），较阻塞切换降低 98.24–98.78%。
4. **转换感知调度器（Transition-aware Scheduler）**：以硬件常量和当前批量 x 为输入，计算各可行布局的单步延迟并维护切换 margin，使部署始终跟随负载变化而选取最快布局。
5. **跨多硬件平台验证**：B200（GLM-5.3）、H200（DeepSeek-V3.2）、DCU（GLM-5.3-Flash）上均复现了相同布局偏好区间，DOP 在 256K–512K 输入及大批量场景显著领先。

## 方法详解
1. **所有权模型**：设投影权重总大小为 W_A、每 token 的 KV 字节数为 k、当前平衡请求数为 B、上下文长度为 s，则四种布局单 rank 内存为：
   - M_TP = W_A / T + Bks
   - M_DOP = M_CP = W_A + Bks / T（注：原文公式排版中 DP/CP 同右侧项，实际 CP 复制权重）
   - M_DP-attention = W_A + Bks / T
   DOP 消除投影副本，KV 按请求独占，因此在固定显存预算下达到最长平衡上下文。
2. **DOP 计算与通信**：分片投影使每 rank 获得全部 N 个 query rows 的 H/T 个 head 切片；owner 侧需要 [N_r, H, d_q] 形状。DOP 在 attention 前后各做一次 variable-size all-to-all 按请求重组，通信量 V_DOP = (1 - 1/T) N H (d_q + d_z) b，与上下文长度 s 无关。
3. **切换原语**：
   - **discard**：目标状态是当前状态的子集时，就地保留需要的块、释放多余块，网络开销为 0。
   - **all-gather**：权重从 TP/DOP 转入 DP/attention 时，每个 rank 接收 (1 - 1/T) 单层投影；或 KV 全汇聚进 TP。
   - **all-to-all**：KV 在 CP 与请求级 owner 布局（DP/DOP）之间搬迁，或 DOP 内部的请求重排。
4. **层内重叠传输与交接**：第 i 层缺失状态在第 i−1 层计算期间并行传输；最多一层在途。内存上满足 max{M_r(a), M_r(b)} + Δ_r^layer + R_r ≤ H_r。交接时 indexer、position、recurrent state 随该层 KV 一并迁移，保证因果前缀完整性。
5. **切换开销建模**：暴露开销
   t̂_exposed = t̂_xfer,1 + Σ_{i=2}^{L} [t̂_xfer,i − t̂_comp,i−1]_+
   权重转移固定 0.29–0.30 ms/layer（B200），KV 转移随 Bs 线性缩放（B=16、256K 约 3.47–3.81 ms/layer）。
6. **调度器 π(x; θ)**：在满足稳态内存约束与切换内存边界的可行集 F(x; θ) 内选 t_ℓ(x; θ) 最小布局；当最优布局领先当前布局超过 margin 时触发切换。

## 实验与结果
- **平台**：B200（GLM-5.3, FP8, 78 层）、H200（DeepSeek-V3.2 FP8, 61 层）、DCU/BW1000（GLM-5.3-Flash INT8, 45 层）；SPLASH 基于 SGLang 0.5.10，开源。
- **Sweep 设计**：输入长度 1K–512K（8 档）、批量 B ∈ {1, 4, 8, 16, 32, 64, 128, 256(·512)}、每请求生成 1,024 tokens；每点 10 次取均值。
- **布局偏好区间**（B200, B=16）：1K–4K → TP；16K–64K → DP-attention；128K → CP；256K–512K → DOP。B=32 时 TP 只在 1K 独占，DOP 同样在 256K–512K 胜出。
- **DOP 最强结果**：B200、64K 输入、B=256 时 26,822 × 10³ tokens/s；128K/B=256 时 30,724 × 10³ tokens/s，较 runner-up（CP 的 27,677）高出 11.0%。
- **KV 容量**（B200, 内存占比 0.85）：DOP 8,271,360 distinct tokens，较 DP-attention 的 6,497,792 多 27.3%，较 CP 的 6,738,944 多 22.7%；DCU 上 DOP 较 DP-attention 多 19.0%（1,118,656 vs 940,224）。
- **切换开销**（B200, B=16, 256K）：12 种有向切换 p50 开销 0.02–11.76 ms，中位占目标就绪步时间比例 <0.51%；相较阻塞切换降低 98.24–98.78%。含 KV 的切换（如 DOP→CP）最重 11.76 ms，纯权重切换最轻 ~0.8 ms。
- **端到端吞吐**：SPLASH 动态切换相较固定布局部署提升 1.3–1.73×。

## 相关工作脉络
- **LoongServe**：弹性序列并行用于长上下文，但不覆盖注意力权重 ↔ KV 所有权的解耦与多布局切换。
- **Flying Serving / Shift Parallelism**：前者仅 TP↔DP 并附带 KV adaptor，后者仅限 TP↔序列并行且共享 head-sharded 布局；两者无法处理 MLA 场景及 DOP 等新布局。
- **Moebius / ReMP / PipeLive**：分别针对 MoE 的 TP/EP、模型并行、流水线并行的运行时重配置；关注维度不同，SPLASH 聚焦注意力这一瓶颈且在单层步骤内完成。
- **Megatron-LM / Ring Attention / Ulysses / SGLang DP-attention**：分别奠定 TP、CP 存储/计算路径与 DP-attention；SPLASH 在其之上叠加布局切换，而非替代任一内核。
- **Pope et al. (Efficiently Scaling Transformer Inference)**：在 TPU 上将 MQA 按 batch 分片以避免 KV head 复制，与 DOP 去除投影副本的思路相近但面向不同硬件与注意力变体。
- **DistServe / Splitwise / Mooncake**：通过预填充/解码分离与 KV 传输组织 serving；SPLASH 则让每个阶段内部可按负载选取最佳注意力布局，两者正交。

## 局限性与未来方向
- **CUDA graph 绑定**：每种布局需独立捕获图，捕获时间与驻留图内存须在部署时预留。
- **显存余量敏感**：切换需额外一层 buffer；显存接近满载时需推迟切换或限流准入。
- **重叠不完全**：传输与同步骤的 collective/计算共享互联与内存带宽，超长历史（如 TP 恢复复制 MLA KV）的每层传输可能超过单层计算能隐藏的量，拉长该步。
- **短生命周期 regime**：若负载振荡快于切换收益窗口，margin 会固化当前布局而错失收益。
- **实现范围限制**：当前要求 TP 与 attention-DP 度相等且一 rank 一 owner；CP 嵌套、投机解码、异构 worker 等均为未来工作。
- **未联合优化**：批量大小与准入策略未与布局选择联合优化，被作者明确列为下一步方向。

## 研究启发与可借鉴点
- **所有权分解思路可迁移**：将并行布局拆解为"数据归属"和"计算归属"两个独立维度的建模方法，可用于分析其他并行组合（如 EP + attention 的混合）。
- **三种切换原语的统一框架**：discard / all-gather / all-to-all 的代价可复用为评估任意两布局切换成本的通用工具，帮助快速比较新布局是否值得加入调度。
- **切换开销的层内隐藏设计**：layer i 传输与 layer i−1 计算重叠、最多一层在途的策略，可推广到 pipeline / MoE expert 切换等需低延迟重配置的场合。
- **DOP 的 all-to-all 重组接口**：variable-size all-to-all + packed query 的实现细节（对齐/不对齐路径、fused no-position/rotary packing）对构建 MQA/GQA 高效推理栈具有直接参考价值。
- **多硬件一致性验证**：B200/H200/DCU 三平台均复现相同 regime 趋势，其 sweep 设计（长度 × 批量正交矩阵 + 10 次重复）可作为 Serving 系统评估的实验模板。

## 关键术语表
- **TP（Tensor Parallelism）**：沿 attention head 维度分片投影权重，每个 rank 持有部分 head；MLA 下 KV 因无 head 轴而被迫全量复制。
- **CP（Context Parallelism）**：沿 token 位置维度切分上下文，复制完整投影权重，适合超长 prompt。
- **DP-attention（Data-Parallel Attention）**：按请求数平分 KV 缓存与 attention 计算，复制完整投影权重，适合大批量独立请求。
- **DOP（Decoupled Ownership Parallelism）**：按 TP 分片投影、按 DP-attention 独享每个请求的 KV；零冗余布局，释放最多 KV 空间。
- **MLA（Merged Linear Attention）**：DeepSeek-V2/V3 的注意力变体，将 Q/K/V 投影融合为少个 latent，消除 head 轴，是驱动所有权解耦的关键背景。
- **Switching 原语（discard / all-gather / all-to-all）**：SPLASH 支持的三种最小转移操作，覆盖全部 12 种有向布局切换。
- **Transition-aware Scheduler**：基于成本模型与当前批量状态实时选布局、并以 margin 防抖的调度函数 π(x; θ)。
- **Exposed switch overhead**：切换中无法被后续层计算隐藏的额外时间，是调度器计入 margin 的实际代价。

## 可复现要素
- **数据集/模型**：GLM-5.3（B200）、GLM-5.3-Flash（DCU）、DeepSeek-V3.2（H200）；评测为合成 workload sweep（输入 1K–512K、输出 1024 tokens）。
- **代码**：已开源，地址 https://github.com/ict-agent/SPLASH-sglang（基于 SGLang 0.5.10）。
- **关键超参**：B200 实验 B=16/32、context 1K–512K、BS ∈ {1,4,8,16,32,64,128,256}，每点 10 次平均；内存占比 0.85（B200/DCU）与 0.90（H200）。调度器参数 θ 经 microbenchmark 校准，论文未披露具体边际阈值数值。
