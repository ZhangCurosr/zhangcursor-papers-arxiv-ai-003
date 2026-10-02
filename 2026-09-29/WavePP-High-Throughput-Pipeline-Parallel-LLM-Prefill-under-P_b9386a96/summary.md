---
title: "WavePP-High-Throughput-Pipeline-Parallel-LLM-Prefill-under-P"
source: https://arxiv.org/pdf/2609.35263v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 14:18:44"
field: "LLM 推理系统优化"
keywords: ["Pipeline Parallelism", "LLM Inference", "Prefix Cache Reuse", "KV Cache Management", "Serving Runtime", "Throughput Optimization"]
innovations: ["异步准入-执行解耦：将 reuse 协商、lease 保护和 escrow 容量预留提前于流水线执行，stage 0 可提前启动", "分片 reuse hint 索引：基于 chained prefix hash 的快速近似查找，将 O(N) 树遍历降至 O(log N)，大幅降低 mutex 竞争", "Local Lease + Capacity Escrow 两段式协议：先锁定复用前缀防止驱逐，再预留 suffix 容量 victim 引用但不物理驱逐，材料化延迟到 loop 线程"]
benchmarks: ["GLM 5.2 PP4", "MiniMax M2.7 PP4", "Kimi K3 TP1×PP8 / TP8×EP8"]
---

# 论文速读：WavePP: High-Throughput Pipeline Parallel LLM Prefill under Prefix Reuse

## 一句话总结
WavePP 是基于 TensorRT-LLM 构建的异步 prefill 运行时，通过将请求准入（reuse 发现、lease 保护、escrow 预留）提前于流水线执行、并动态调整 wave 大小，显著提升了 Pipeline Parallelism 下 LLM 的 prefill 吞吐量；在 Kimi K3 上 c≥8 的 18 个设置中均取得最高吞吐量。

## 研究问题与动机
- **分布式缓存无法保证全局前缀可复用**：在 PP 设置下，各 stage 独立维护 KV 块和 recurrent-state 快照，local cache hit 不保证跨所有 stage 可用，需要一个分布式协议就"公共复用边界"达成一致。
- **传统 admission 序列化阻塞流水线**：在 executor 主路径上完成 cache 遍历、复用协商和空间分配会延迟下一 wave 启动，造成 pipeline bubble。
- **混合注意力架构引入稀疏 recurrent checkpoint**：Kimi K3 等模型同时包含 MLA 和线性注意力层，recurrent snapshot 仅在稀疏边界保存，可复用边界的交集更加受限，进一步放大跨 stage 协调难度。
- **短 suffix 请求在 cache 压力下 admission 成本占比升高**：90% 缓存命中后每个请求只需处理 4K 左右未缓存部分，准备开销成为吞吐瓶颈，亟需将 admission 与执行解耦。

## 核心贡献（创新点）
1. **异步准入-执行重叠设计**：将 reuse 发现、lease 保护、escrow 容量预留与流水线执行并行，stage 0 可提前启动而不必等待后续 stage 完成本地准备。
2. **分片 reuse hint 索引**：用 chained prefix hash 构建索引，通过指数探测+二分查找将 dense attention 遍历从 O(N) 降至约 O(log N)，p99 提示发现时间低至 2.35 ms，大幅降低 mutex 竞争。
3. **Local Lease + Capacity Escrow 协议**：lease 在不进入树的前提下"锁定"复用前缀防止被驱逐；escrow 以 victim 引用（free/offload/eviction）预先保留 suffix 空间，实际 block 分配延迟到本地 materialization 阶段，二者分离避免过早占用内存。
4. **MPU（Maximum Pipeline Utilization）自适应 wave 调度**：根据已准入但未调度的 token 数与 in-flight chunk 动态选择 wave 预算（连续减半序列），在保持流水线充实的同时避免为少量请求增加过多 forward launch 开销。
5. **端到端吞吐提升显著**：在 TensorRT-LLM PP4 相同 kernel 下，GLM 5.2 和 MiniMax M2.7 的 40 个设置中 37 个吞吐提升；c=128 短 suffix 场景下分别达 2.91× 和 2.02×；Kimi K3 28 个设置中 c≥8 的 18 个均优于 TP/EP 和外部 PP 基线。

## 方法详解
- **线程分离**：每个 rank 有两个主线程——WAVE 线程负责请求传输、reuse 协商与容量管理；LOOP 线程运行 executor 并完成 cache block 的本地 materialization。Rank 0 同时担任 leader，在 executor 线程上做 chunk 规划。
- **双通信平面**：data plane 传输 wave schedule 和 activations 支持前向计算；rounds plane 通过 rank 0 broadcast + all-gather 传递 admission 结果，多请求打包以摊薄开销。
- **Fast Reuse Hints**：每个完整 token block 计算 chained prefix hash $H_i = \mathrm{hash}(B_i, \mathrm{extraKeys}_i, \mathrm{salt}, H_{i-1})$；索引按 cache window 和 $H_i$ 维护计数器，block attach/detach 时增减计数。dense attention 使用前向指数探测+二分定位，recurrent 缓存从 attention 边界反向扫描，取所有 rank 的 minimum 作为 reuse candidate。
- **Local Lease**：WAVE 线程在 mutex 外构建 block keys，进入 mutex 后逐块 pin；若不同 rank 的 recurrent 边界不一致（如 rank 0 有 64K/72K/90K 快照、rank 1 有 64K/80K 快照），则逐级回退到最低公共边界，单调收敛。同时限制并发 lease 数量以避免过度 pin 导致无驱逐空间。
- **Capacity Escrow**：lease 之后立即为每层 layer 预留 suffix 空间，escrow 持有 victim 的引用（free block / offload victim / eviction victim），不物理驱逐；commit 时重检容量并处理 victim 替换（若原 victim 被其他请求新增 owner，则选同 window 的 replacement），所有检查通过后才 publish。
- **MPU 调度**：execution ring 大小 $R=2P$，目标 wave 数 $S = R + (F-1)P$（PP4 取 $F=2$）。预算序列 $b_0=M$, $b_{j+1} = \max(b_{\min}, \mathrm{alignDown}_B(b_j/2))$；可用 wave 数估计为 $n(b) = \max(W/b, N_{>b/2})$，选择满足 $n(b) \geq S$ 的最大 $b$；hysteresis 防抖避免频繁切换；FCFS 打包且避免用小于 $\max(b_{\min}, b/2)$ 的碎片填充小余量。
- **Chunk 下发**：rank 0 发送每个 scheduled chunk 的 request ID、起始 token 和长度给 follower ranks，follower 在本地 materialization 完成后对齐执行；若有 mismatch 则执行失败而非处理错误 cache 状态。

## 实验与结果
- **模型与硬件**：GLM 5.2、MiniMax M2.7（NVFP4，4×B300 GPU，PP4）；Kimi K3（2.8T 参数 KDA/MLA MoE，8×GB300 GPU，TP1×PP8 vs TP8/EP8）。
- **基线**：TensorRT-LLM PP4（同 kernels）、TRT-LLM TP8/EP8、SGLang TP8/EP8 & PP8、vLLM TP8/EP8 & PP8。
- **GLM 5.2 / MiniMax M2.7**：40 个设置中 WavePP 在 37 个提升吞吐；冷数据和混合数据提升 1.9–10.7%；90% 短 prefix reuse 在 c=64 提升 39.0%/24.8%；**4K-suffix c=128 场景：GLM 2.91×（416K→1213K tok/s）、MiniMax 2.02×（551K→1113K tok/s）**；p50 prefill 延迟降低 55–71%。
- **Kimi K3**：28 个设置中 WavePP 在 21 个最高吞吐，**c≥8 全部 18 个均最高**；几何均值相对 SGLang PP8 高 9.2%，相对 vLLM PP8 高 15.7%；冷 131K c=32 达 35,878 input tok/s，较 TRT-LLM TP8/EP8 提升 62.6%；90% 262K reuse c=32 达 218,743 input tok/s，较 TRT-LLM TP8/EP8 高 53.3%。
- **Ablation（B300 PP8）**：冷 131K c=16 下 wave admission 使吞吐从 8,826 → 29,692（+236%）；reuse 131K c=8 下 lease+escrow 使 96,796 → 149,021（+54%），MPU 再加 41.7% 至 211,214；但 reuse 262K c=32 下 MPU 反降 9.7%（静态打包更优），说明策略需更自适应。
- **Prefix retention**：wavepp 在 0.5–1.0 req/s 到达率下 good-hit 率 88–100%，对比 TRT-LLM TP8/EP8 的 73–83%。

## 相关工作脉络
- **gLLM**：token throttling 调节 prefill/decode 预算，关注全局平衡；WavePP 同样调整 wave budget，但侧重将 admission 前置以降低调度开销对执行节奏的影响。
- **Sarathi-Serve / Revisiting PP**：前者 chunked prefill + stall-free batching 减少 bubble，后者动态 chunk 减小 stage 间不平衡；WavePP 在此基础上用自适应 wave sizing 配合异步 admission 进一步减少 admission 造成的停顿。
- **SiPipe / gLLM metadata-ahead**：将 host 准备与 GPU 计算重叠；WavePP 也做类似解耦，但额外加入分布式 reuse 协商和 capacity escrow，解决 PP 下跨 stage cache 一致性这一新增难点。
- **vLLM V1 PP / SGLang PP**：两者均支持 successive prefill chunks in-flight；WavePP 聚焦于分布式 admission 协议——在缓存独立驱逐的场景下建立 reusable state 共识并预留 suffix 容量。
- **Marconi / Jenga / LMCache**：关注混合注意力/异构缓存管理；WavePP 在此基础上提出 local lease + escrow 机制，允许各 stage 独立管理各自 cache，同时保障全局 reuse 一致。
- **Mooncake / Disagg serving**：将 prefill/decode 拆分至不同 worker；WavePP 专门针对 prefill-only _worker 的 PP 场景，解决同一 prefill pipeline 内的 admission-执行重叠问题。

## 局限性与未来方向
- **MPU 粒度有限**：当前按固定步长（连续减半）选择 budget，不做单 chunk 执行时间估计；等长 chunk 因 attend 不同 prefix 长度或遇到不同 stage 瓶颈而成本各异，未来可基于执行时间估计做更细粒度的动态 chunking。
- **部分场景 MPU 退回**：ablation 显示 reuse 262K c=32 下 MPU 使吞吐下降 9.7%，说明静态打包在该 workload 更优；需要更灵活的 hysteresis 或 workload-aware 策略避免误判。
- **仅评估 TP1×PP4 / TP1×PP8**：未测试 TP2×PP4、PP16 等拓扑，也未考察 uneven layer placement（stage 间固有执行时间不平衡无法仅靠 admission 优化解决）。
- **FCFS 打包策略**：当前按先到先服务排列请求，未考虑优先级、deadline 或 cache 动态状态；可能在不同 SLO 场景下存在优化空间。
- **单副本评估**：未测试多副本 + cache-aware routing 场景，以及更长生命周期部署下的 SLO 曲线行为。

## 研究启发与可借鉴点
1. **"准入-执行解耦"范式**可迁移：凡涉及分布式状态协商（缓存、路由、配额）的流水线系统，均可将协商前置到独立线程，使 executor 在等待 admission 完成时继续处理已有 microbatch，减少 bubble。
2. **Sharded hint index + 二分定位**是一种通用的"快速近似查询后再精确验证"模式，适用于任何带并发修改的大型树/索引结构，可将 mutex 持有窗口从 O(N) 降至 O(log N)。
3. **Lease + Escrow 两段式容量协议**：先 lease 锁定"可复用部分"防驱逐，再 escrow 预留"未缓存部分"的 victim 引用但不物理驱逐，两段分离可在不确定 final commit 的情况下提前占位，适合任意分布式资源分配场景。
4. **混合 workload 下 throughput 与 p95 的分离趋势**值得注意：WavePP 在 mixed traffic c=16/32 下 p50 优于 PP 基线，但 p95 反而高于基线，说明自适应 wave sizing 在改善中位延迟的同时可能增加尾部波动，后续研究需联合优化两者。
5. **Adaptive wave sizing 的 hysteresis 防抖设计**：通过"升高需连续 R/2 次、降低需连续 P 次"的稳定化机制避免预算在相邻 iteration 来回震荡，可借鉴到任何带宽/容量自适应调节器中。

## 关键术语表
- **Wave / Lane**：wave 是多个 request chunk 沿 pipeline 同步移动的微批次；lane 是单个 request 在不同 wave 中的逻辑位置，贯穿其所有 chunk 被调度完毕。
- **Local Lease**：WAVE 线程在 cache tree mutex 外构建 key，进入 mutex 后 pin 已协商复用的 block 及对应 recurrent snapshot，阻止其被驱逐，但不进行物理 materialization。
- **Capacity Escrow**：lease 之后为 uncached suffix 预留容量，持有 free/offload/eviction victim 的引用；实际驱逐和 block 分配推迟到 LOOP 线程的 materialization 阶段。
- **Reuse Hint Index**：基于 chained prefix hash 的分片索引，维护每 block 的计数；dense attention 用指数探测+二分查找，recurrent 从边界反向扫描，用于快速协商 candidate reuse endpoint。
- **MPU（Maximum Pipeline Utilization）**：基于已 admitted 未调度 token 数和 in-flight chunk 数自适应选择 wave 预算的调度策略，预算从 M 连续减半至 b_min。
- **Chained Prefix Hash**：$H_i = \mathrm{hash}(B_i, \mathrm{extraKeys}_i, \mathrm{salt}, H_{i-1})$，使每 block 的 key 依赖其完整前缀，用于构建 reuse hint index。
- **Kimi Delta Attention（KDA）**：Kimi K3 使用的线性注意力架构，基于 Gated DeltaNet，state 维度与序列长度无关，checkpoint 稀疏存储。
- **Disaggregated Serving**：将 prefill 和 decode 阶段分配给不同类型 worker，使二者可使用不同并行策略和资源配置。

## 关键超参与配置
- **GLM 5.2 / MiniMax M2.7 PP4**：microbatch limit 16,384 tokens；max batch size 128；WavePP ring $R=2P=8$，$F=2$；min wave budget $b_{\min}=8,192$。
- **Kimi K3**：host cache 128 GiB/rank；total work / context chunk budget 16,384 tokens；MPU 候选预算 $M=16,384$, $b_{\min}=8,192$；hysteresis：升高需 $R/2=8$ 次决策稳定，降低需 $P=8$ 次。

## 可复现要素
- **数据集/工作负载**：GLM 5.2、MiniMax M2.7 公开模型（HuggingFace）；Kimi K3 公开模型（arXiv:2607.24653）。负载包括 cold、90% seeded prefix reuse、mixed（75/25 warm/cold），均为合成请求序列，具体 prompt 内容论文未公开。
- **代码**：WavePP 实现在 TensorRT-LLM 内，论文未提供独立开源仓库链接；论文未提及独立代码发布。
- **权重**：使用 NVIDIA NVFP4 量化权重；论文未提及独立权重发布。
- **关键超参**：见"关键超参与配置"节（microbatch limit 16,384；ring 大小 $R=2P$；$F=2$；$b_{\min}=8,192$；host cache 128 GiB/rank 等）。
- **硬件**：NVIDIA B300（GLM/MiniMax 实验）、GB300×8（Kimi K3 实验，双节点各 4 GPU）；论文未提及云实例型号。
- **库版本**：TensorRT-LLM 1.3.0rc26；SGLang v0.5.18（commit 71de97b264b0）；vLLM v0.28.0。
