---
title: "WavePP-High-Throughput-Pipeline-Parallel-LLM-Prefill-under-P"
source: https://arxiv.org/pdf/2609.35263v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 19:55:47"
field: "大模型推理系统"
keywords: ["pipeline parallelism", "LLM inference", "prefix caching", "serving runtime", "throughput optimization"]
innovations: ["异步解耦 admission 与执行，将缓存协调移至 WAVE 线程", "Fast Reuse Hints 索引将缓存遍历从 O(n) 降至 O(log n)", "Local Lease + Capacity Escrow 两级协议实现分布式缓存复用与容量预留"]
benchmarks: ["GLM 5.2", "MiniMax M2.7", "Kimi K3"]
---

# 论文速读：WavePP: High-Throughput Pipeline Parallel LLM Prefill under Prefix Reuse

## 一句话总结
WavePP 是一个构建在 TensorRT-LLM 之上的异步预填充运行时，通过解耦请求 admission 与流水线执行、提出缓存复用协议与容量预留机制，显著提升了多阶段流水线并行 LLM 推理的吞吐量，在高并发与高缓存复用场景下吞吐量最高提升 2.91 倍。

## 研究问题与动机
- **流水线利用率依赖高效调度**：流水线并行（PP）通过并发处理多个请求块来提升预填充吞吐量，但保持流水线满载需要高效的请求调度与准备工作。
- **分布式缓存复用协调成本高**：当各流水线阶段独立管理 KV 缓存时，本地缓存命中无法保证全局前缀可复用，跨阶段协调开销会阻碍请求 admission 速率。
- **传统 lockstep admission 阻塞流水线**：现有方法在 admission 阶段需要所有 rank 同步等待，导致准备工作阻塞在主线执行路径上，延迟下一轮 forward pass。
- **缓存管理与执行路径耦合**：请求的准备（缓存查找、锁存、容量预留）若在主执行循环中串行完成，会引入巨大开销，尤其在并发请求多、缓存压力大的场景下。

## 核心贡献（创新点）
1. **异步解耦的 admission 设计**：将请求 admission 从主执行循环中分离，通过 WAVE 线程与 LOOP 线程的并行协作，允许在早期请求执行时异步完成新请求的 admission。
2. **Fast Reuse Hints 索引机制**：提出基于分片哈希索引的快速缓存复用提示机制，将重复遍历开销从 O(n) 降至约 O(log n)，中位探测时间从 64.4ms 降至 24μs。
3. **Local Leases + Capacity Escrows 协议**：设计本地租约与容量预留两级协议，先锁定可复用前缀再预留后缀空间，避免缓存竞争导致的资源死锁。
4. **MPU 动态波大小调度**：提出最大流水线利用率（Maximum Pipeline Utilization, MPU）调度器，通过自适应缩小波预算增加飞行中的波数，平衡流水线填充率。

## 方法详解
- **双线程架构**：每个 rank 维护 WAVE 线程（负责请求传输、复用协商、容量管理）和 LOOP 线程（负责执行器与本地缓存物化），Rank 0 作为 leader 执行分块规划。
- **双平面通信**：数据平面传输 wave 调度与激活值， Rounds 平面处理 admission 结果；admission rounds 在等待响应时更频繁运行。
- **Fast Reuse Hints**：维护一个无实际块的索引结构，对 dense attention 使用指数探测+二分查找，对稀疏 recurrent snapshots 向后扫描；仅 admission 选中的请求才执行权威遍历。
- **Local Leases**：在 mutex 外构建块键，在 mutex 内验证并 pin 块；按 FCFS 顺序获取租约，限制同时持有租约的请求数（预留 2·P·M token 余量）。
- **Capacity Escrow**：租约获取后立即预留后缀容量；escrow commit 检查容量并提交保留，但不立即分配块或驱逐 victim；实际驱逐/卸载延迟到本地物化阶段。
- **MPU 调度器**：基于 R=2P 的执行环与 S=R+(F−1)P 的目标波数，预算序列 b_j 每次减半至最小值 b_min；可用波数估计为 max(W/b, N_{>b/2})。

## 实验与结果
- **硬件平台**：NV B300 GPU（GLM 5.2/MiniMax M2.7 实验）、双节点八卡 GB300（Kimi K3 实验）。
- **GLM 5.2 & MiniMax M2.7 PP4 对比**：
  - 在 40 个测试设置中，WavePP 在 37 个设置下提升 TensorRT-LLM PP4 的预填充吞吐量。
  - 高缓存复用（90% 短前缀）+ c=128 时，GLM 5.2 吞吐量提升 **2.91×**，MiniMax M2.7 提升 **2.02×**。
  - 4K-suffix 工作负载在 c=128 时，GLM 5.2 达到 1,212,572 tokens/s，较基线提升 191.3%。
- **Kimi K3（2.8T 参数混合模型）跨库对比**：
  - 在 28 个设置中，WavePP 在 21 个设置取得最高吞吐量；c≥8 的 18 个设置全部领先。
  - 几何平均吞吐量较 SGLang PP8 提升 9.2%，较 vLLM PP8 提升 15.7%。
  - 冷请求 c=32 时，65.5K/131K/262K 输入分别较 TRT-LLM TP8/EP8 提升 58.8%/49.1%/37.8%。
  - 前缀缓存命中率 88-100%，高于 TRT-LLM TP8/EP8 的 73-83%。
- **消融实验**：Wave admission 将冷预填充吞吐量提升 236.4%；Leases+Escrows 在复用场景提升 54.0%；MPU 额外提升 41.7%。

## 相关工作脉络
- **gLLM [12]**：通过 token throttling 平衡流水线，但 WavePP 聚焦 admission 提前化而非仅调节 token 预算。
- **Sarathi-Serve [4] / Revisiting PP [9]**：动态分块与调度优化，WavePP 在此基础上将 admission 与执行重叠。
- **vLLM V1 PP [11] / SGLang PP [22]**：支持 pipeline 重叠执行，但 WavePP 强调分布式 admission 协调与本地缓存独立管理。
- **RadixAttention [3] / PagedAttention [2]**：块级 KV 缓存管理，WavePP 采用 TRT-LLM radix tree 策略并扩展其 admission 协议。
- **Marconi [14] / Jenga [15]**：混合注意力架构的缓存管理，WavePP 处理 MLA+线性注意力（如 Kimi Delta Attention）的稀疏 checkpoint 交叉约束。
- **Mooncake [29] / LMCache [16]**：分布式 KV 层级存储，WavePP 聚焦 prefill worker 内的 admission 加速。

## 局限性与未来方向
- **MPU 调度在某些场景存在回退**：静态打包策略可能在特定 workload 下表现更优，需进一步自适应优化。
- **仅评估固定拓扑**：当前仅测试 TP1×PP4 与 TP1×PP8，未探索 TP2×PP4、PP16 等配置下 admission 开销变化。
- **阶段不平衡问题未根本解决**：即使请求就绪，阶段间执行时间差异仍会导致气泡。
- **FCFS 调度缺乏优先级**：未考虑请求 deadline、优先级或缓存动态状态的多目标优化。
- **未评估多副本场景**：结合 cache-aware routing 的多副本部署可能进一步提升 tail latency。

## 研究启发与可借鉴点
- **Admission-Execution 解耦设计**：将分布式协调工作移至独立线程/协程，避免阻塞主执行路径，适用于任何需要多节点协调的并行系统。
- **Hint Index 加速共识机制**：在树形结构中维护轻量级计数器索引，将 O(n) 遍历降为 O(log n)，可推广至其他分布式缓存协商场景。
- **租约+预留的两阶段协议**：先声明后实配的模式（类似数据库两阶段提交思想）可在资源竞争中平衡一致性与延迟。
- **MPU 的折中调度策略**：通过预算折半而非精细切分来增加波数，以较小规划开销换取流水线填充率提升，适合动态负载场景。
- **异构缓存统一协商**：处理 dense KV cache 与 sparse recurrent checkpoint 的交叉约束，为混合注意力架构的缓存复用提供通用框架。

## 关键术语表
**Pipeline Parallelism (PP)**：将模型层划分为连续组分配到不同阶段，各阶段并行处理不同 microbatch 以提升吞吐。
**KV Cache**：自回归模型缓存 keys 和 values，避免重复计算前缀 tokens。
**Radix Tree**：按 token 前缀组织的树形索引，支持高效的前缀缓存查找与共享。
**Local Lease**：在缓存树中 pin 住可复用前缀块，防止被驱逐，但不立即分配后缀空间。
**Capacity Escrow**：为请求的 uncached suffix 预留缓存容量，通过保留 victim 块引用实现延迟物化。
**Wave**：一组并行流过流水线的请求块（chunk），构成调度的基本单位。
**Fast Reuse Hints**：基于分片哈希索引的快速复用提示，减少重复遍历开销。
**MPU (Maximum Pipeline Utilization)**：自适应波预算调度器，通过折半策略增加飞行波数以填满流水线。

## 可复现要素
- **数据集**：使用合成/种子前缀工作负载（冷请求、90%复用、混合流量），未使用公开 benchmark 数据集。
- **代码**：WavePP 基于 TensorRT-LLM 开发，具体开源状态论文未明确提及。
- **权重**：使用 NVFP4 量化权重的 GLM 5.2、MiniMax M2.7、Kimi K3 模型。
- **关键超参**：Block size=16,384 tokens；b_min=8,192；执行环 R=2P；F=2；host cache 128 GiB/rank（Kimi K3）；batch limit=16/32。
