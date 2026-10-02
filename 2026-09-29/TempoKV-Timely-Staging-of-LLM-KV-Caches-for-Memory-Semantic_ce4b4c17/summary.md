---
title: "TempoKV-Timely-Staging-of-LLM-KV-Caches-for-Memory-Semantic"
source: https://arxiv.org/pdf/2609.35065v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 08:13:54"
field: "LLM推理系统与存储优化"
keywords: ["KV Cache", "LLM Serving", "CXL Memory", "Staging Timing", "Memory-Semantic Flash", "Prefix Cache"]
innovations: ["Timing-aware resource commitment separating early reuse knowledge from staging resource acquisition", "Bidirectional time estimation (TTU vs TTR) for just-in-time staging decision", "Protected capacity budget with dynamic fallback to demand retrieval"]
benchmarks: ["NarrativeQA", "LooGLE"]
---

# 论文速读：TempoKV-Timely-Staging-of-LLM-KV-Caches-for-Memory-Semantic

## 一句话总结
TempoKV 提出了一种**时间感知的资源承诺机制**，将"早期发现 KV 复用"与" staging 资源承诺"分离，通过比较运行时估计的"使用截止时间"（TTU）与存储侧估计的"就绪时间"（TTR），在恰当的时机触发 SSD 到快速存储层的 staging，从而在显著降低受保护快层容量占用（减少 63–91%）的同时，保持接近即时 staging 的服务性能。

## 研究问题与动机
- **核心问题**：在 memory-semantic flash 架构下，可复用的 prefix KV 缓存可能超出 GPU HBM 容量，需要存储在 SSD 或 CXL 设备 DRAM 中；但逻辑上的 KV hit 并不等于"已就绪"，仍需 SSD 到快速层的 staging，且 staging 时机直接影响延迟与容量成本。
- **现有方法不足 1**：**即时 staging**（hit 发现即承诺）会过早占用受保护的快速层容量，导致在 retrieval 开始前就长期锁定资源，显著增加 protected fast-tier byte-time。
- **现有方法不足 2**：**延迟 staging**（等到 retrieval 请求时才承诺）会使 SSD staging 延迟直接暴露给客户端，增加首 token 延迟（TTFT）。
- **现有方法不足 3**：基于**队列排名**的启发式（如 Queue-K）不可靠——相同排名的请求因前置请求剩余服务时间、staging  backlog 等因素，实际就绪时间波动极大（从提前 2.67s 到延迟 6.02s）。

## 核心贡献（创新点）
1. **Timing-aware 资源承诺层**：首次将"复用知识"与"staging 资源承诺"解耦，仅在 `TTU ≤ TTR` 时才发起承诺请求，避免过早/过晚。
2. **双向时间估计机制**：运行时估计 TTU（快层到 GPU retrieval 开始前的剩余时间），存储侧估计 TTR（若此刻承诺，使 KV stage-ready 所需时间），二者动态比较。
3. **受保护容量预算的动态管理**：承诺受限于配置的 protected-capacity budget，不足时自动 fallback 到 demand retrieval，且不阻塞其他 claim 的后续承诺尝试。
4. **vLLM + LMCache 的集成实现**：在不修改请求调度策略、不改动设备固件的前提下，将 TempoKV 集成到 vLLM 与 LMCache，验证了其实用性。

## 方法详解
- **架构三组件**：Runtime Adapter（估算 TTU）、TempoKV Controller（比较 TTU/TTR 并决定承诺时机）、Storage-side Staging Provider（估算 TTR 并控制 staging 资源）。
- **TTU 估算**：基于当前 active batch、每个请求的执行进度、调度顺序，映射未完成 prefill 与预期 decode 工作量到时间；调用长度预测采用 QLM 的工作量估计与 Past-Future 的输出长度条件化方法。
- **TTR 估算**：基于已知 fast-tier 驻留状态、队列中排队及进行中的 staging 工作、有效 staging 速率；采用元数据投影，仅对尚未驻留且未被现有 staging 覆盖的 KV object 添加 SSD 读操作。
- **松弛度计算**：`L̂_c(t) = TTU_c(t) - TTR_c(t)`；当 `L̂_c(t) ≤ 0` 或运行时请求 retrieval 时，claim 变为 eligible，提交给 provider 进行有效性 & 容量检查。
- **状态机**：PLANNED → STAGING（承诺成功）→ STAGEREADY（所有 object 驻留且受保护）→ TRANSFERRING（retrieval 开始）→ RELEASED（GPU 传输完成）。
- **Shared staging**：同一 KV object 的 staging 可被多个 claim 共享，避免重复 SSD 读与重复容量预留。

## 实验与结果
- **数据集**：12 篇 NarrativeQA 文档（来自三个长度组），以及 LooGLE 数据集（用于 Figure 6 对比）。
- **模型**：Qwen2.5-14B-Instruct、Llama-3.1-8B-Instruct，BF16 精度。
- **硬件**：Intel Xeon Granite Rapids（72 核）+ NVIDIA H100 PCIe（80GB HBM）+ 128GiB DRAM CXL 设备 + 15.36TB NVMe SSD。
- **核心结果 1**：相较 Immediate staging，TempoKV 将受保护快层 byte-time 减少 **63–91%**；相较 Queue-4 减少 **62–86%**。
- **核心结果 2**：相较未修改的 LMCache Device-DAX L1 配置，TempoKV 将 p95 TTFT 降低 **最高 48.0%**，output throughput 提升 **最高 27.8%**（NarrativeQA）。
- **核心结果 3**：快层容量从 100 GiB 降至 25 GiB 时，TempoKV 的吞吐量与 p95 TTFT **几乎不变**（约 106 token/s、5.9s），展现出优异的容量适应性。
- **对比基线**：Demand、Immediate、Queue-1、Queue-4、TempoKV-Static、TempoKV。

## 相关工作脉络
- **Beluga / TraCT**：使用 CXL 内存池作为共享 KV 缓存底座，但未解决 staging 时机问题。
- **ITME**：软件驱动 prefetching 到 SSD 内部 DRAM 缓存，聚焦于缓存窗口管理而非 commitment timing。
- **HyMCache**：通过 bounded internal-DRAM 窗口流式传输 prefix KV，同样未处理 early knowledge vs. commitment 分离。
- **Bidaw**：storage-aware request scheduling，通过改变请求调度来准备 KV，而 TempoKV 不修改调度策略。
- **IMPRESS / Mooncake / MatKV**：多级 KV 存储系统，侧重存储层级设计，缺乏 timing-aware 的 staging 决策。
- **QLM / Past-Future**：请求调度与 SLA 感知的调度器，TempoKV 与其正交，可与之一同使用。

## 局限性与未来方向
- 实验仅在单 GPU + 单 CXL 设备配置下进行，未评估多 GPU 或多 rack 扩展场景。
- 仅测试了两个模型（Qwen2.5-14B、Llama-3.1-8B），更大规模模型（如 70B+）的 KV footprint 与 staging 行为未验证。
- TTR 估算在某些场景下仍基于静态率或简化假设（TempoKV-Static），完全状态感知的动态估算仍有优化空间。
- 未评估不同 SSD 类型（NVMe vs. SATA）或不同 CXL 设备带宽对 Timing 决策的影响。
- 工作负载为固定间隔请求，真实 traffic 的 bursty 模式下的表现待验证。

## 研究启发与可借鉴点
- **时序解耦思想**：将"知识获取"与"资源承诺"分离是一个通用设计原则，可迁移到任何需要预加载/预热资源的系统（如 RAG 系统、多模态推理）。
- **双向时间比较机制**：TTU vs. TTR 的比较范式简洁有效，可在其他存储层级（如 HBM → DDR → SSD）中复用。
- **受保护容量预算**：动态 budget 而非静态分区是一种更灵活的资源管理方式，值得在多租户 KV 缓存系统中借鉴。
- **与现有系统的正交集成**：不修改 vLLM 调度、不改动 CXL 固件的设计思路，降低了部署门槛。

## 关键术语表
- **Memory-Semantic Flash**：结合 SSD 容量层与快速内存层（如 CXL 设备 DRAM），通过内存语义抽象统一管理的多级存储架构。
- **Prefix KV Cache**：LLM 推理中可复用的前缀 key-value 状态，避免重复 prefill 计算。
- **Staging**：将 KV 对象从 SSD 载入快速存储层（如设备 DRAM）的操作过程。
- **Protected Capacity**：为特定 staging 工作预留、防止被 LRU 驱逐的快层内存配额。
- **TTU (Time-To-Use)**：运行时估计的，从当前时刻到 retrieval 开始前的剩余时间。
- **TTR (Time-To-Ready)**：存储侧估计的，若此刻承诺 staging，使 KV 成为 stage-ready 所需的时间。
- **Stage-Ready**：所有匹配 KV object 均已驻留于快层且受保护的状态。
- **CXL Type-3 Memory**：支持内存语义访问的 CXL 设备，提供 DRAM 快层 + SSD  backing  tier 的架构。

## 可复现要素
- **数据集**：NarrativeQA（公开）、LooGLE（公开）；论文未声明私有数据。
- **代码开源**：论文未明确声明 TempoKV 代码开源；基于 vLLM v0.23.0 与 LMCache v0.5.1 修改。
- **硬件**：Intel Xeon Granite Rapids + NVIDIA H100 PCIe 80GB + CXL 128GiB DRAM 设备 + Samsung PM1753 NVMe SSD。
- **关键超参**：protected-capacity budget = 50 GiB（fast tier 100 GiB 的一半）；controller tick = 50ms；prefix cache ratio = 50%/75%/100%；输出长度上限 = 128 tokens（NarrativeQA）/ 32 tokens（LooGLE）。
- **随机种子**：论文未提及。
