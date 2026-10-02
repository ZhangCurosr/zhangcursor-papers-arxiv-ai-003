---
title: "SPLASH-SWITCHING-PARALLEL-LAYOUTS-OF-ATTENTION-WITH-SEAMLESS"
source: https://arxiv.org/pdf/2609.37626v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 14:22:43"
---

# 论文速读：SPLASH-SWITCHING-PARALLEL-LAYOUTS-OF-ATTENTION-WITH-SEAMLESS

## 一句话总结
本文提出 SPLASH，一种在 LLM 推理服务过程中动态切换注意力并行布局的系统。基于现代 Attention 架构中权重分片与 KV 缓存归属解耦的观察，SPLASH 设计了零冗余的 DOP 布局，并通过层间重叠传输将四种布局间的切换开销压至中位数不足 0.51%，最终使端到端吞吐提升 1.3–1.73×。

## 研究问题与动机
- **负载动态波动与固定并行策略的错配**：Agent、RL rollout 等工作负载在单次服务周期内会经历从高并发短请求到低并发长序列的剧烈变化，TP、CP、DP-attention 各有胜负区间，启动时固定一种布局无法跟进实时负载。
- **现有在线重配置代价过高**：传统做法需排空批次并重启 worker 才能切换，而切换收益最大的场景（长尾慢请求拖累整批）恰恰是最需要快速响应的时刻。
- **MLA/MQA/GQA 带来的结构解耦机会**：现代模型削减或消除了 KV heads，使得“注意力投影权重如何放置”与“请求的 KV history 归属哪台设备”成为两个独立维度，为细粒度布局组合与无缝切换提供了理论基础。

## 核心贡献（创新点）
1. **提出注意力布局的所有权模型（Ownership Model）**：将 TP、CP、DP-attention 与 DOP 统一归结为权重放置（分片/复制）与 KV 归属（请求级/位置级）两个独立决策，推导出各布局的单机 footprint、最大可行上下文及 12 种定向切换所需的状态差异与通信原语。与已有工作仅枚举布局不同，该模型给出了切换成本的解析上界与精确分配方案。
2. **首次导出零冗余布局 DOP**：以 TP 方式分片投影权重，以 DP-attention 方式将每个请求的 KV 历史单点归属，消除所有冗余副本。相比 DP-attention，DOP 在 GLM-5.3（B200, FP8, T=8）上每卡释放 12.69 GiB 用于 KV，总容量提升 27–60%，填补了现有引擎缺乏的“纯分片权重+请求级缓存”组合空白。
3. **实现层间重叠的实时切换引擎**：在单次推理 step 内按层异步搬运缺失状态，利用独立 stream 使 layer i 的通信与 layer i−1 的计算重叠，12 种定向切换的中位数开销仅 0.02–11.76 ms（< 0.51% 目标 step 时间），较 blocking 切换降低 98% 以上。与 Shift Parallelism 等仅限两两切换或需长时间阻塞的工作相比，SPLASH 覆盖全布局空间且几乎零停顿。
4. **开发 Transition-aware 调度器**：将硬件常数与当前批次状态（请求数、上下文长度、query rows）映射为单步延迟，引入切换暴露成本与滞回 margin 避免交叉区抖动，使部署始终跟随负载演化。相比仅比较稳态延迟的传统调度，该设计显式计入切换边际成本。

## 方法详解
- **所有权模型与内存公式**：布局由两个维度决定。设投影权重总大小为 $W_A$，每 token KV 字节为 $k$，请求数 $B$，上下文长度 $s$，设备
