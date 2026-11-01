---
title: "RadixFlow-Filling-Execution-Bubbles-in-Prefix-Cached-Hybrid"
source: https://arxiv.org/pdf/2610.18143v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-01 19:00:44"
field: "大语言模型推理服务调度"
keywords: ["prefix caching", "hybrid LLM serving", "Mamba", "batch scheduling", "execution bubble", "KV cache reuse", "state-ready traversal"]
innovations: ["提出状态就绪遍历策略，将同轮依赖的消费者从批次扫描中隔离而非停止扫描，消除可避免的执行气泡", "建立 READY/BLOCKED/OTHER 三态判定框架，形式化 hybrid LLM 前缀缓存中的同轮依赖", "通过 worker cost 分解公式（round credit vs cost shift）定量分析批次密度与延迟收益的权衡"]
benchmarks: ["QuALITY", "RACE-M"]
---

# 论文速读：RadixFlow-Filling-Execution-Bubbles-in-Prefix-Cached-Hybrid

## 一句话总结
RadixFlow 针对混合架构 LLM（如 Mamba）前缀缓存推理中的"头部阻塞"问题，提出状态就绪遍历策略——将被同轮依赖阻塞的消费者分支隔离到等待队列，同时继续扫描并填充批次中其他独立可用分支，从而消除可避免的执行气泡；在 5 个混合模型上降低 3%–24% 平均延迟、减少 11%–44% Worker Round。

## 研究问题与动机
- **问题本质**：前缀命中 ≠ 可执行分支。混合 LLM 的前缀复用不仅需要 KV 缓存，还需要在匹配边界处可用的循环态（recurrent）和卷积态（convolutional）；若该状态由同一轮次中另一分支（生产者）仍在计算，则消费者分支被阻塞。
- **现有调度器的缺陷**：原生 vLLM 调度器在等待队列扫描时遇到 BLOCKED 消费者即停止（prefix-stop），导致"头部阻塞"——即使后面存在能通过原生状态检查且容量足够的独立分支，也会被一并排除。
- **执行气泡的定义**：批次结束后仍有空余行（$b_t = S - |B_t| > 0$），且存在满足 READY 状态却未被调度的已到达分支，即为可避免的空闲容量。
- **收益来源不明**：前作研究多关注缓存命中率提升，但同轮依赖造成的"空转"尚未被量化分析，且现有调度假设"命中即可跑"对 hybrid LLM 不成立。

## 核心贡献（创新点）
1. **问题形式化**：首次将 hybrid LLM 同轮依赖（same-round dependency）严格定义为 READY / BLOCKED / OTHER 三态，给出"执行气泡"（Definition 2）的容量定义，揭示前缀缓存命中但无法调度的"假空闲"现象。
2. **状态就绪遍历（state-ready traversal）**：将阻塞的消费者放入等待队列 $D_t$，保留注册，继续扫描后续独立分支，直至扫描结束再统一发布。与已有工作的本质区别在于不修改缓存存储、不改状态检查逻辑，仅改批次构造策略；理论证明构建的批次严格包含 prefix-stop 批次的子集（Proposition 2）。
3. **灵活 Admission Control（Flex）**：以 5 ms 为上限收集已到达分支，当已有运行工作、行数满或 token 预算满时提前释放；与 Radix+Timer 对比隔离出"同轮依赖信号"vs "普通有限等待"各自的贡献。
4. **实证贡献**：在 5 个 hybrid 模型（Qwen3.5-0.8B/4B/9B/27B + Nemotron-9B）及 full-attention 控制（Qwen3-8B FA）上给出完整消融，揭示 traversal 在大模型上贡献主要收益、admission 在低负载 Nemotron-9B 上贡献主要收益。

## 方法详解
- **三态判定（Definition 1）**：对每个待调度分支 $r$，根据当前部分批次 $B$ 调用原生状态检查，得到 $\sigma_t(r|B) \in \{\text{READY}, \text{BLOCKED}, \text{OTHER}\}$。READY 表示可扩展进批次；BLOCKED 表示其匹配长度 $\hat{h}_t(r) > h_t(r)$（因同轮中某 producer 的循环/卷积态尚未发布）；OTHER 为预算溢出等原生处理逻辑。
- **Algorithm 1（state-ready traversal）**：
  - 按原生顺序扫描 $A_t$；
  - READY → 通过原生分配器选入 $B$；
  - BLOCKED → 加入等待队列 $D_t$，保持注册，**继续扫描**；
  - OTHER → 执行原生动作（可能终止扫描）；
  - 执行完成后发布至 $\mathcal{C}_{t+1}$，唤醒 $D_t$ 中消费者在下轮重新检查。
- **可行性证明（Proposition 2）**：给定固定状态和原生顺序，RadixFlow 构造的批次 $B_t^{\text{RF}}$ 满足 $B_t^{\text{stop}} \subseteq B_t^{\text{RF}}$，且在单调性假设下（无 OTHER 提前终止），无可避免气泡。
- **容量下界（Proposition 1）**：对固定工作碎片，最少轮数 $\geq \max\{\lceil W/T \rceil, \lceil V/S \rceil, D\}$，其中 $D$ 为最长依赖链长度；图 1 的四轮调度即达到此下界。
- **Worker Cost 分解（Equation 2）**：$T_{\text{host},0} - T_{\text{host},1} = \underbrace{(K_0 - K_1)\bar{c}_0}_{\text{round credit}} - \underbrace{K_1(\bar{c}_1 - \bar{c}_0)}_{\text{cost shift}}$，区分"少执行的轮次节省"和"剩余轮次成本变化"两个因素。

## 实验与结果
- **数据集**：QuALITY（62 篇文章，1034 题）+ RACE（128 篇文章，517 题），每题为 4 分支（不同候选答案），共 9306 逻辑请求/策略，三流模式（cold/mixed/reuse）× 两种输入顺序。
- **模型**：Qwen3.5-0.8B/4B/9B/27B（hybrid）、NVIDIA Nemotron-Nano-9B-v2（Mamba2 hybrid）、Qwen3-8B（FA，对照）。
- **关键结果**（相对 Radix 基线）：
  | 模型 | 延迟下降 | Round 下降 | Reuse 变化 |
  |---|---|---|---|
  | Qwen3.5-0.8B | 9.3%（0.256→0.232s）| 11.0%（19,803→17,610）| ≤0.01pp |
  | Qwen3.5-4B | **23.6%**（13.1→10.0s）| **24.4%**（13,424→10,146）| ≤0.03pp |
  | Qwen3.5-9B | 7.1%（26.7→24.8s）| **32.6%**（11,042→7,444）| ≤0.01pp |
  | Qwen3.5-27B | 3.1%（164→159s）| **44.4%**（11,748→6,529）| ≤0.01pp |
  | Nemotron-9B | 12.0%（2.44→2.15s）| 33.0%（14,338→9,613）| ≤0.02pp |
  | Qwen3-8B (FA) | 0.0%（16.4→16.4s）| 3.8%（9,223→6,746）| ≤0.01pp |
- **消融结论**：
  - traversal 贡献绝大部分收益：Qwen3.5-27B 上 Flex 仅省 1.6% rounds，RadixFlow 再省 44%；
  - 对 FA 模型 traversal 几乎无效（循环态检查未激活），验证了方法针对 hybrid 的针对性；
  - RadixFlow+Frag 在 27B 上额外降 5% 延迟（3.6% fewer rounds，+1.66pp reuse）。
- **吞吐**：Qwen3.5-4B/9B/27B 完成吞吐分别提升 4.2%、4.0%、3.0%。

## 相关工作脉络
1. **PagedAttention / RadixAttention**（Kwon 等, Zheng 等）：KV 缓存分页/索引共享；本文不改动缓存存储，仅改批次选择。
2. **Marconi**（Pan 等, 2025a）：hybrid 模型的缓存策略，在 reuse 和 memory cost 间平衡；本文复用其状态检查接口，而非优化缓存结构本身。
3. **SGLang Unified Radix Cache**（Zhang & SGLang Team, 2026）：统一树索引记录各状态组件；本文依赖其原生检查逻辑，不改索引。
4. **Sarathi-Serve**（Agrawal 等, 2024）：prefill-decode chunking 平衡吞吐延迟；本文聚焦 prefill 阶段同轮依赖导致的气泡。
5. **DistServe / Splitwise**（Zhong 等, Patel 等）：阶段分离架构；本文在同一 worker round 内优化 batch 构造，不改变执行拓扑。
6. **FlashAttention 系列**（Dao 等）：kernel 级加速；本文从调度侧补充，两者正交。

## 局限性与未来方向
- 评估限于单轮 prefill + 1 token 生成，未覆盖长 decode 阶段；
- 未做负载 sweeping（不同 offered rate 下的泛化性）；
- 异步调度（async scheduling）、CUDA Graph、TP 场景需单独验证；
- per-request 阻塞计数未记录，无法直接归因特定分支节省；
- 更多 fine-grained dependency trace 可支持更细粒度的调度决策。

## 研究启发与可借鉴点
1. **"前缀命中 ≠ 可执行"** 的区分思路可迁移至其他状态ful模型（如 State-Space Models、RNN-like）的 serving 优化，通用框架：引入 READY/BLOCKED/OTHER 三态，与缓存命中解耦。
2. **Worker Cost 分解公式（Equation 2）** 提供了一种"轮次节省 vs 剩余轮次成本变化"的定量分析方法，可复用于评估其他 batch 构造策略。
3. **实验设计**：统一后端（共享 prefix index、state check、cache primitive）、冻结 budget、分离 admission 与 traversal 的消融链，干净地隔离各组件贡献，值得借鉴。
4. **FA 控制组**的引入（Qwen3-8B FA）是验证"状态就绪依赖"核心假设的关键对照，类似思路可用于验证其他 state-dependent 调度优化。
5. **在线 replay 设置**（真实 benchmark 组合、固定 arrival trace、cold/mixed/reuse 三流）比合成长度测试更能反映实际性能，建议后续工作沿用。

## 关键术语表
- **Prefix reuse / Prefix caching**：将共享 prompt 前缀的 KV 缓存保留并在后续请求中复用，避免重复计算。
- **Hybrid LLM**：结合 full attention 与 selective state-space（如 Mamba/Mamba2）的混合架构，复用边界需要额外的循环态和卷积态。
- **State-ready traversal**：RadixFlow 的核心策略，扫描时将 BLOCKED 消费者移入等待队列而非停止扫描，使独立分支继续入队。
- **Execution bubble**：批次内因头部阻塞造成的合法可填充空闲槽位（行或 token），定义为避免的空闲容量。
- **Same-round dependency**：同一 worker round 中，消费者需要从生产者当前正在计算的循环/卷积态，生产者执行后才可读取。
- **Producer / Consumer**：OS 术语借用，producer 计算所需 recurrent state，consumer 在同轮需要读取该 state。
- **Flexible admission（Flex）**：以 5 ms 为上限收集已到达分支，在行数/token 预算满或有可执行工作时提前释放。
- **Round credit vs cost shift**：少执行轮次带来的时间节省（credit）与剩余轮次变重导致的成本增加（shift）之间的权衡，由 Equation 2 定量刻画。

## 可复现要素
- **数据集**：QuALITY + RACE（公开基准），代码/实验 script 保留，附录 G 声明匿名化复现记录。
- **模型**：Qwen3.5-0.8B/4B/9B/27B（公开权重）、Nemotron-Nano-9B-v2（公开）、Qwen3-8B（公开）。
- **后端**：vLLM 0.29.0，所有策略共享同一 overlay 实现，已集成至 **vllm-jev** 仓库。
- **超参**：行预算 $S \in \{16, 64\}$、token 预算 $T=8192$，通过 64 题预留集 vLLM auto-tuner 校准后冻结；Flex 超时 5 ms；offered rate Qwen 系列 5 req/s，Nemotron-9B 1.5 req/s。
