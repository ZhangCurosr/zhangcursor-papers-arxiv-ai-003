---
title: "RadixFlow-Filling-Execution-Bubbles-in-Prefix-Cached-Hybrid"
source: https://arxiv.org/pdf/2610.18143v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-01 19:01:10"
field: "大语言模型推理服务与调度"
keywords: ["LLM inference serving", "prefix caching", "hybrid LLM", "scheduling", "Mamba", "KV cache"]
innovations: ["状态就绪遍历：仅阻塞同轮依赖消费者，继续调度独立分支消除执行气泡", "前缀命中与可执行状态的语义分离与形式化", "灵活准入与遍历协同的批量构建策略"]
benchmarks: ["QuALITY", "RACE-M"]
---

# 论文速读：RadixFlow: Filling Execution Bubbles in Prefix-Cached Hybrid LLM Serving

## 一句话总结
论文针对混合LLM（hybrid LLM）推理服务中前缀缓存调度的"执行气泡"问题，提出RadixFlow调度策略：将前缀匹配与状态就绪性解耦，仅阻塞具有同轮依赖的消费者分支，同时继续调度独立的可行分支，从而消除首端阻塞（head-of-line blocking），在五种混合模型上实现平均延迟降低3%~24%、工作轮次减少11%~44%。

## 研究问题与动机
- **核心问题**：混合LLM（如Mamba类模型）的前缀复用不仅需要KV cache，还需recurrent和convolutional状态；当 producer 分支正在计算状态时，consumer 分支虽有前缀匹配但状态不可读，形成同轮依赖（same-round dependency）。
- **现有方法不足**：原生 vLLM 调度器在等待队列扫描中，遇到 BLOCKED 的消费者即停止扫描（prefix-stop），导致本可独立执行的分支被错误排除，产生"可避免的执行气泡"（avoidable execution bubble）——有空闲槽位却未填充可行工作。
- **调度语义差异未被显式建模**：已有前缀缓存工作（RadixAttention、Marconi 等）关注缓存存储与索引，但未区分"前缀命中"与"分支可执行"两种不同事件。
- **评估空白**：在 hybrid LLM 场景下，同一 worker round 内因状态依赖导致的资源浪费尚未被形式化并量化。

## 核心贡献（创新点）
1. **状态就绪遍历（State-ready traversal）**：将 BLOCKED 消费者移入等待队列而非终止扫描，保持对后续独立分支的选择，本质区别于仅靠增大 batch 或延长等待窗口的方案。
2. **前缀命中 ≠ 可执行分支的语义分离**：形式化定义状态就绪性（readiness）与可避免执行气泡，揭示 hybrid LLM 调度中因同轮依赖产生的隐藏浪费。
3. **灵活准入控制（Flexible admission）与遍历协同**：通过短超时（5 ms）收集到达请求，与遍历机制配合提升批次密度；消融实验分离两者贡献。
4. **工程集成**：所有技术已集成到 vllm-jev 框架，在五种 hybrid 模型（Qwen3.5 系列 0.8B/4B/9B/27B 及 Nemotron-9B）及全注意力 Qwen3-8B 控制组上验证。

## 方法详解
**核心调度算法（Algorithm 1）**：
- 对每个 worker round，按原生顺序 ≺ 扫描待调度分支集合 $\mathcal{A}_t$。
- 使用原生状态检查 $\sigma_t(r | B)$ 判断分支结果：
  - **READY**：加入当前批次 $B$。
  - **BLOCKED**：移入等待队列 $D_t$，保持注册状态，继续扫描。
  - **OTHER**（资源超限等）：执行原生处理（可能终止扫描）。
- 执行批次 $B_t = B$，发布状态至 $\mathcal{C}_{t+1}$，并在下一轮唤醒 $D_t$ 中的消费者重新检查。

**关键设计**：
- **不修改缓存存储或状态检查逻辑**，仅改变批次构建策略。
- **阻塞粒度**：仅阻塞 consumer，producer 正常工作；等待队列在下一轮重检。
- **灵活准入**：设定 5 ms  admission deadline，或在已有运行工作/行满/token 预算满时提前释放。
- **碎片控制（Frag 扩展）**：对小残余预算的分支做延迟选择以减少碎片，作为消融实验。

**理论性质（Proposition 2）**：
- 可行性保证：RadixFlow 构造的批次始终满足原生状态检查。
- 包含性：$B_t^{\text{stop}} \subseteq B_t^{\text{RF}}$，即消除了 prefix-stop 下因首端阻塞导致的遗漏。
- 无气泡条件：若无 OTHER 结果终止扫描，则单轮内无可避免气泡。

## 实验与结果
**数据集与工作负载**：
- 基于 QuALITY（62 篇文章，1034 题）和 RACE-M（128 篇文章，517 题）构建，每题为含4个分支的逻辑请求（共享文档前缀，不同候选答案）。
- 三种流类型：cold（跨问题无共享）、reuse（重复问题）、mixed（交错）。
- 9306 个逻辑请求/策略，冻结行预算 $S$ 和 token 预算 $T$（经 vLLM 自动调参器校准）。

**基线方法**：
- Radix（原生 vLLM 调度）、Radix+Timer5/10（有界等待）、Radix+Flex（灵活准入）、RadixFlow（主方法）、RadixFlow+Frag（碎片控制扩展）。

**主要结果**：
| 模型 | Radix 延迟(s) | RadixFlow 延迟(s) | 延迟降低 | 轮次降低 | 复用率变化 |
|------|--------------|------------------|---------|---------|-----------|
| Qwen3.5-0.8B | 0.256 | 0.232 | **9.3%** | 19,803→17,610 (**11%**) | 81.36% → 81.37% |
| Qwen3.5-4B | 13.1 | 10.0 | **23.6%** | 13,424→10,146 (**24%**) | 80.41% → 80.43% |
| Qwen3.5-9B | 26.7 | 24.8 | **7.1%** | 11,042→7,444 (**33%**) | 79.69% → 79.70% |
| Qwen3.5-27B | 164 | 159 | **3.1%** | 11,748→6,529 (**44%**) | 68.50% → 68.51% |
| Nemotron-9B | 2.44 | 2.15 | **12.0%** | 14,338→9,613 (**33%**) | 75.94% → 75.95% |
| Qwen3-8B (FA) | 16.4 | 16.4 | ≈0% | 9,223→6,746 | 85.24% → 85.25% |

- **最强提升**：Qwen3.5-4B 延迟降低 23.6%，Qwen3.5-27B 轮次降低 44%。
- **消融结论**：在 Qwen3.5 系列上，增益主要来自遍历（traversal）；在全注意力控制组上遍历几乎无贡献，验证了机制针对 hybrid 状态依赖的有效性。
- **吞吐量**：Qwen3.5-4B/9B/27B 的 replay throughput 提升 4.0%/3.0%/3.0%。

## 相关工作脉络
1. **PagedAttention (Kwon et al., 2023)**：KV page 共享机制，RadixFlow 保留其缓存管理，仅改变调度逻辑。
2. **RadixAttention (Zheng et al., 2024)**：精确前缀索引，RadixFlow 在其基础上处理 hybrid 状态就绪问题。
3. **Marconi (Pan et al., 2025a)**：hybrid 模型的 prefix caching，关注状态保留成本权衡，RadixFlow 聚焦调度端的状态依赖解耦。
4. **SGLang unified radix cache (Zhang & SGLang Team, 2026)**：统一树结构管理 hybrid 缓存组件，RadixFlow 不修改缓存结构。
5. **Sarathi-Serve (Agrawal et al., 2024)**：平衡 prefill 与 decode 的 chunking，RadixFlow 作用于单 round 内的分支选择。
6. **DistServe/Splitwise (Zhong et al., 2024; Patel et al., 2023)**： disaggregate prefill/decode，与 RadixFlow 作用层面不同（RadixFlow 在 prefill 批次构建内）。

## 局限性与未来方向
- **评估场景有限**：仅针对 candidate-conditioned prefill（单 token 生成），未覆盖长 decode 或完整生成任务。
- **单一配置冻结**：预算和到达速率固定，未做 load sweep 或生产级调度评估。
- **细化依赖追踪缺失**：未记录每轮阻塞/恢复的具体行数，难以精确归因每个 saved round 的来源。
- **异步调度与 CUDA graph**：未评估异步调度下的 dependency fence 语义，以及 CUDA graph 对 per-round cost 的影响。
- **未来方向**：扩展至多 token 生成、探索 producer-first 分组策略、集成异步执行模型、在线自适应预算调参。

## 研究启发与可借鉴点
1. **状态就绪性与前缀匹配的语义分离**：可将此形式化框架迁移至其他需要额外状态（如LoRA适配器、MoE路由）的缓存调度场景。
2. **阻塞粒度优化**：仅阻塞 consumer 而继续扫描的模式，对图调度（DAG scheduling）中类似依赖链有借鉴价值。
3. ** Worker-time 会计方法**：公式 (2) 将总时间分解为"轮次节省收益"与"剩余轮次成本偏移"，可用于精确归因调度优化收益，避免"少轮次=快"的粗糙推断。
4. **消融设计**：通过 Radix vs Radix+Flex vs RadixFlow 的逐层对比，清晰分离 admission 与 traversal 的贡献，值得在系统论文中复用。
5. **Full-attention control**：用 Qwen3-8B (FA) 作为控制组验证 hybrid 特有机制，是区分通用优化与领域特定优化的有效实验设计。

## 关键术语表
**Prefix caching**：将公共 prompt 前缀的 KV cache 保留供后续请求复用，减少重复计算。
**Hybrid LLM**：结合注意力机制与选择性状态空间模型（如 Mamba）的混合架构，需在复用边界维护 recurrent/convolutional 状态。
**Same-round dependency**：producer 分支在同一 worker round 内计算状态，consumer 分支需读取该状态但此时尚未发布，形成同轮阻塞。
**Execution bubble**：批次中存在空闲行/槽位，且有已到达且状态就绪的分支因 head-of-line blocking 未被调度。
**State-ready traversal**：RadixFlow 核心策略，遇到 BLOCKED 分支仅将其放入等待队列，继续扫描并调度独立分支。
**Flexible admission**：在扫描前设定短超时窗口收集到达分支，或满足行/token 预算条件时提前释放，提升批次密度。
**Worker round**：一次全局 worker 调用，包含最多 $S$ 行和 $T$ 个 token 的调度预算。
**Native state check**：vLLM 原生状态检查机制，判断分支恢复边界处的 KV/recurrent/conv 组件是否已发布且可用。

## 可复现要素
- **数据集**：QuALITY 和 RACE-M（公开），通过代码构造工作负载；附录 G 提供匿名化可复现记录计划。
- **代码**：已集成到 vllm-jev（GitHub 仓库未在本文明确给出 URL，附录 G 提及匿名发布版本将包含 pinned model/data revisions）。
- **权重**：使用公开模型权重（Qwen3.5 系列、Nemotron-9B、Qwen3-8B）。
- **关键超参**：行预算 $S$ ∈ {16, 64}，token 预算 $T = 8192$，admission deadline = 5 ms，到达速率 5 req/s（Qwen）/1.5 req/s（Nemotron）。
- **硬件**：A800 GPU，vLLM 0.29，eager execution，禁用异步调度，allocator utilization 0.8。
