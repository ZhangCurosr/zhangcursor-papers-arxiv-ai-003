---
title: "TempoKV-Timely-Staging-of-LLM-KV-Caches-for-Memory-Semantic"
source: https://arxiv.org/pdf/2609.35065v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 08:13:50"
field: "LLM系统优化与缓存管理"
keywords: ["KV Cache", "LLM Serving", "CXL Memory", "Staging", "Resource Management", "Inference Optimization"]
innovations: ["双维时序估计TTU/TTR对比机制", "Timing-based eligibility松弛度控制", "Metadata-only claims分离知识与资源承诺"]
benchmarks: ["NarrativeQA", "LooGLE"]
---

# 论文速读：TempoKV-Timely-Staging-of-LLM-KV-Caches-for-Memory-Semantic

## 一句话总结
TempoKV是一种感知时间的资源提交层，通过分离"复用知识"与"预取资源承诺"，在SSD-backed CXL内存分层中智能调度KV缓存的预取时机，显著降低显存占用同时保持 Serving 性能。

## 研究问题与动机
- **核心矛盾**：复用prefix KV缓存可超越GPU HBM容量，但SSD-backed缓存检索存在显著延迟，需在SSD与快-tier之间预取（staging）
- **现有方法缺陷**：
  - Immediate staging（立即预取）过早占用保护容量，导致byte-time开销高
  - Demand staging（按需预取）暴露SSD延迟，增加首token等待时间
  - Queue-rank触发不准确（相同队列位置的实际等待时间差异大）
- **关键洞察**：提前知道复用≠需要立即预取；应比较"距检索启动时间"与"预取完成时间"来决定提交时机
- **问题定义**：如何在不改变请求调度策略的前提下，动态确定最优staging commitment时机

## 核心贡献（创新点）
1. **两维时序估计框架**：提出runtime估计的TTU（Time-to-Use）与storage估计的TTR（Time-to-Ready）对比机制，本质区别在于分离"运行时执行状态"与"存储侧预取状态"
2. **Timing-based eligibility机制**：定义松弛度L̂(t) = TTU - TTR，仅当L̂≤0时才提交staging承诺，避免过早占用容量
3. **Metadata-only claims设计**：发现KV命中时仅记录元数据声明而不触发I/O，与现有系统直接启动预取的本质区别
4. **共享staging与容量预算**：支持多claim共享同一KV对象的预取工作，动态保护容量预算而非静态分区
5. **vLLM+LMCache集成实现**：无需求调度器改动，仅需扩展runtime adapter与controller层

## 方法详解
**架构三组件**（Figure 3）：
1. **Runtime Adapter**：
   - 计算TTU：基于active batch、每请求执行进度、调度顺序
   - 使用校准的execution costs映射unfinished prefill与decode工作到时间
   - 考虑并发工作共享timeline，非简单叠加latency
   - 输出长度估计：基于已完成请求的N-n均值，generation limit截断

2. **Storage-side Staging Provider**：
   - 计算TTR：预测使matched KV成为stage-ready所需时间
   - 评估hypothetical commitment的staging queue元数据投影
   - 仅添加SSD reads（对不在fast-tier且未被queued staging覆盖的对象）
   - 使用calibrated有效聚合service rate，考虑concurrent operations共享速率

3. **TempoKV Controller**：
   - **Timing-based eligibility**：L̂_c(t) = TTU_c(t) - TTR_c(t)，L̂≤0时eligible
   - **Commitment流程**：提交eligible claim→provider验证validity+capacity→STAGING/STAGEREADY
   - **Replanning**：按请求顺序与时序更新重评估；committed claim不preempt
   - **Protected capacity budget**：protected-capacity < total capacity，LRU evict unprotected entries

**状态机**：PLANNED→STAGING→STAGEREADY→TRANSFERRING→RELEASED

**关键公式**：
- 保护容量成本：C = (∫₀ᵀ P(t)dt)/N (GiB·s/request)
- 松弛度：L̂_c(t) = TTU_c(t) - TTR_c(t)

## 实验与结果
**实验设置**：
- 硬件：Intel Xeon Granite Rapids (72-core) + NVIDIA H100 PCIe (80GB HBM) + CXL×8 + 15.36TB Samsung PM1753 NVMe SSD
- 设备：128 GiB DRAM fast tier, 100 GiB配置用于主实验
- 模型：Qwen2.5-14B-Instruct, Llama-3.1-8B-Instruct (BF16)
- 工作负载：12份NarrativeQA文档，固定0.5s到达间隔，输出 capped at 128 tokens

**主要结果**：
- **vs Immediate staging**：protected fast-tier byte-time降低63-91%
- **vs Demand staging**：100% prefix cache ratio下，p95 TTFT降低25.7%，output throughput提升20.4%
- **vs Queue-4**：62%更低C，略高TTFT (4.7%)
- **vs LMCache-DAX baseline**：p95 TTFT降低48.0%，throughput提升27.8% (NarrativeQA)
- **Capacity sweep**：100→25 GiB (75%减少)，throughput与p95 TTFT几乎不变(~106 token/s, ~5.9s)
- **State-aware TTR**：Llama 100%下TTFT降低从14.5%提升至20.5%，C从4.0→4.8 GiB·s/request

## 相关工作脉络
1. **Beluga/TraCT**：CXL memory pools as shared KV-cache substrates，但未解决timing问题
2. **ITME**：software-directed prefetching into SSD内部DRAM，关注预取而非commit timing
3. **HyMCache**：streaming reusable prefix KV through bounded internal-DRAM window
4. **Bidaw**：storage-aware request scheduling与KV-read ordering，但需改调度器
5. **LMCache**：enterprise-scale KV cache layer，Device-DAX L1作为baseline
6. **QLM/Past-Future**：work-based waiting-time estimation与output-length conditioning，被TempoKV复用

## 局限性与未来方向
- **TTR估计简化**：Static版本忽略staging queueing与contention，state-aware版本更准确但成本更高
- **CXL特定硬件**：实验基于CXL Type-3 memory device，通用性需验证
- **固定请求间隔**：workload使用synthetic固定间隔，未测试burst traffic
- **单GPU设置**：多GPU扩展性未评估
- **TTR estimator可替换**：框架支持替换，但当前实现可能非最优

## 研究启发与可借鉴点
1. **时序分离思想**：将"知道复用"与"承诺资源"解耦，可迁移至其他缓存系统（如RAG retrieval、multi-agent workflow）
2. **双维估计对比模式**：TTU vs TTR框架可适配不同延迟维度（如网络预取、数据库查询）
3. **保护容量预算机制**：dynamic budget而非static partition，适用于资源受限场景
4. **元数据声明设计**：metadata-only claims避免不必要I/O，可优化其他缓存系统的cold start
5. **无需改调度器的集成**：保持runtime调度策略不变，降低部署门槛

## 关键术语表
- **Memory-semantic flash**：结合SSD-backed capacity与smaller fast-memory tier的存储架构
- **Staging commitment**：授权staging并预留fast-tier容量以protect matched KV的时机点
- **TTU (Time-to-Use)**：运行时估计距fast-tier-to-GPU retrieval开始的时间
- **TTR (Time-to-Ready)**：存储侧估计使matched KV成为stage-ready所需时间
- **Stage-ready**：所有required KV object resident in fast-tier且protected against eviction的状态
- **Protected capacity budget**：小于total capacity的保护容量上限，LRU淘汰unprotected entries
- **Metadata-only claim**：仅记录KV对象manifest而不触发I/O的声明

## 可复现要素
- **数据集**：NarrativeQA (12 documents), LooGLE (随机采样16 contexts)
- **代码**：基于vLLM v0.23.0与LMCache v0.5.1修改，论文未明确开源声明
- **权重**：Qwen2.5-14B-Instruct, Llama-3.1-8B-Instruct (BF16)
- **关键超参**：protected-capacity budget = 50 GiB (half of 100 GiB fast tier), 50ms periodic tick, N=4 active sequences
- **硬件**：H100 PCIe 80GB, 128 GiB CXL device DRAM, 15.36TB NVMe SSD
