---
title: "Stable-Token-level-Scheduling-for-Serverless-LLM-Inference"
source: https://arxiv.org/pdf/2610.17689v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-02 09:05:08"
field: "大语言模型推理调度与资源管理"
keywords: ["Serverless LLM Inference", "Token-level Scheduling", "Inter-Token Latency", "Admission Control", "Chunked Prefill", "Feedback Control"]
innovations: ["提出ITL反馈闭环的Token Window自适应admission控制", "双守卫门（token budget+KV-cache headroom）+防饥饿综合排序", "chunked prefill跨decode step注入以降低noisy-neighbor干扰"]
benchmarks: ["GSM8K", "TTFT P99", "ITL Coefficient of Variation", "Throughput (req/s)"]
---

# 论文速读：Stable Token-level Scheduling for Serverless LLM Inference

## 一句话总结
本文针对Serverless LLM推理中请求突发导致解码阶段干扰严重的痛点，提出**Token Window Pacing (TWP)**——一个基于ITL（Inter-Token Latency）反馈闭环的令牌级 admission 控制器，通过动态调节 token 窗口和 chunked prefill 实现稳定低延迟、高吞吐的流式推理。

## 研究问题与动机
1. **TTFT vs ITL 失衡**：现有工作（ServerlessLLM、DynamicBatch等）主要优化 TTFT，但忽视了解码阶段的 ITL 稳定性——ITL 直接决定用户流式体验，高 ITL 会导致卡顿与恢复期延长。
2. **无 admission 反馈闭环**：调度器在请求到达时做 admission 决策，未感知 decode 引擎当前负载状态；突发期间会过度接受请求，使 GPU 超出高效区（$B > B_0$），进入 compute-bound  regime。
3. **Noisy-neighbor penalty**：长 prompt 一次性 prefill 会瞬时抢占共享内存总线，使所有活跃请求的 ITL 短暂翻倍；缺乏分块 staging 机制放大该惩罚。
4. **Serverless 冷启动叠加突发**：冷启动率常超 25%，request-rate 可在短时间内波动数个数量级，open-loop 调度无法区分"有容量"和"已拥塞"两种状态。

## 核心贡献（创新点）
1. **ITL 作为拥塞信号的理论与实证刻画**：证明 $\ell(B_t)$ 存在两阶段模型（带宽受限→计算受限），给出 $B_0$ 的显式拐点；论证 ITL 比 queue depth / KV-cache occupancy 更能及时反映decode pipeline 拥塞。
2. **Token Window 比例反馈控制**：提出自适应 token 窗口 $H_t$（公式 3-4），以归一化慢化 $\sigma_t = \hat{\ell}_t/\ell_{min}$ 为 denominator、队列深度为 numerator，实现"低拥塞扩窗、高拥塞缩窗"的自调节 admission。
3. **双守卫 Admission Gate + 防饥饿综合排序**：token budget 与 KV-cache headroom 双重门槛；请求按 starvation priority、token balance、fairness 加权排序，超长等待强制 admission。
4. **Chunked prefill 参数化接入**：将 admitted 请求的 prompt 拆成固定 chunk 跨 decode step 注入，bound noisy-neighbor penalty 到每步一小 fraction；离线 characterization 确定最优 chunk size $C$。
5. **与现有 serving stack 正交可插拔**：TWP 作为调度层部署于请求队列与后端之间，无需修改模型权重、attention kernel 或 KV-cache 管理，可直接集成 vLLM/Orca 类系统。

## 方法详解
### 系统模型
- **Token Load** $K_t = \sum_{i \in A_t} r_i(t)$：活跃集剩余输出 token 总数，TWP 控制的核心变量。
- **Batch Size** $B_t = |A_t|$：活跃请求数。
- **Saturation 阈值** $B_0$：离线 profiling 得到，对应 ITL 由 $\ell_{min}$ 跳升至 $\ell_{min} + \alpha(B_t - B_0)$ 的拐点。

### ITL 两阶段模型
$$
\ell(B_t) = \begin{cases}
\ell_{min}, & B_t \leq B_0 \\
\ell_{min} + \alpha(B_t - B_0), & B_t > B_0
\end{cases}
$$
当 $B^* \to \infty$ 时，最大 token 处理率收敛到 $1/\alpha$；若到达率 $\lambda \bar{L} > 1/\alpha$，backlog 发散。

### Token Window 自适应控制
- **归一化慢化** $\sigma_t = \hat{\ell}_t / \ell_{min}$（指数平滑消除瞬态 spike）。
- **目标窗口** $H_t^* = H_0 \cdot \frac{1 + \min(1, q_t/Q)}{\max(0.5, \sigma_t)}$：ITL 低且队列增长时扩窗；ITL 升高时缩窗。
- **平滑更新** $H_{t+1} = H_t + \gamma(H_t^* - H_t)$，增益 $\gamma = 0.15$。
- **最小/最大 clamp**：防止过激扩张/收缩。

### 双守卫 Admission Gate
1. **Token budget 守卫**：$K_t + r_i \leq H_t$。
2. **KV-cache headroom 守卫**：剩余可分配 page 数 $> \delta_{kv}$；内存消耗速率快时收紧 $\delta_{kv}$，反之放松。
两条件同时满足才 admission；不跳小补大，保证 starvation 排序不被绕过。

### 请求综合排序
- **Starvation priority**：等待时间 $> T_{max}$ 的请求强制 admission。
- **Token balance** $T(r) = \exp\left(-\frac{|\hat{L}(r) - \bar{r}_t|}{\bar{r}_t + 1}\right)$：偏好与活跃集平均剩余长度相近的请求，减少长尾请求阻塞短请求的频率。
- **Fairness**：wait time 归一化 job size，防止短请求流淹没长请求。
权重 $w_1 > w_2 > w_3$。

### Chunked Prefill 机制
Admitted 请求的 prompt 切为 chunk size $C = 64$ tokens，每 decode step 注入一个 chunk；offline characterization 表明再减小 $C$ 对 ITL 改善边际有限，但 TTFT 持续上升。

### 正交部署
TWP 作为独立调度层挂在请求队列前端，后端仍使用 vLLM/Orca 类 engine；每步观测 $\ell_t, q_t, B_t$，更新 $H_t$，决定下步 admission。

## 实验与结果
### 实验环境
- **硬件**：单卡 NVIDIA Tesla V100 32GB，CUDA 11.8。
- **模型**：Meta-Llama-3.2-3B（FP16）。
- **工作负载**：GSM8K 基准生成请求，Poisson 到达 $\lambda=18$ req/s，60s 区间；prompt 长度异构（60% 短 30-80、30% 中 120-250、10% 长 350-512）；输出长度 20/50/200 tokens 混合。
- **Saturation batch** $B_0 = 8$，$\ell_{min} \approx 43$ ms。

### 基线
FCFS、Round-Robin、Priority（最短 prompt 优先）、DynamicBatch（静态 ITL 阈值截断）、ServerlessLLM（ reimplemented 公平对比）。

### 关键结果
| 指标 | TWP | ServerlessLLM | 提升 |
|---|---|---|---|
| **Throughput** | 0.197 req/s | 0.136 req/s | **+45%** |
| **vs 传统调度** | 0.197 req/s | ~0.121 req/s | **+58%** |
| **TTFT P99** | 66 ms | 78 ms | **-15%** |
| **ITL CV** | 0.0144 | 0.0159-0.0203 | 最低 |
| **ITL 中心值** | 43.5-43.8 ms | 61-62 ms tail | 分布更紧 |
| **TTFT 抖动** | 稳定 ~50 ms | 峰值 77-85 ms | 无 spike |

### 四场景鲁棒性
Low/Medium/High/Burst 四种负载下 TWP 均领先：高负载吞吐 2709 vs 次优 299 req/s（注：表中吞吐量单位应为 token/s 量级，TWP 稳定维持高吞吐且 P99 延迟最低）；burst 场景 TWP P99 延迟 10272 ms 仍显著优于 FCFS/DynamicBatch 的 16894-17869 ms。

### 最强结果
**吞吐提升 58%**（vs 传统调度）、**P99 TTFT 降低 15-20%**、**ITL 变异系数最低 0.0144**，且无需修改底层 serving 栈。

## 相关工作脉络
1. **ServerlessLLM [2]**：multi-tier checkpoint 加载 + live migration 降冷启动；TWP 在其之上补充 admission 反馈控制，解决"加载快但解码拥塞"的次生问题。
2. **Orca [3] / vLLM [4]**：continuous batching + PagedAttention；TWP 复用其 step-synchronous 执行语义，增加 token-level  admission 门。
3. **Sarathi-Serve [5]**：chunked prefill 降 noisy-neighbor penalty，但未将 ITL 观测回灌 admission；TWP 将此机制纳入闭环控制。
4. **Splitwise [14]**：prefill/decode 硬件分离；TWP 在单共享池内实现 phase-aware 调度，避免额外硬件成本。
5. **TokenScale [16]**：Token Velocity 做 autoscaling；TWP 证明 token-level 信号（ITL）比 request-level 更适合做 admission 控制目标。
6. **DynamicBatch [30]**：静态 ITL 阈值截断；TWP 指出静态阈值无法响应突发动态，需自适应窗口。
7. **BAARCH / BATCH [24]**：静态 batch size 绑定 profiled 延迟目标；TWP 扩展为 token load 动态 bound 并耦合 KV-cache 监控。

## 局限性与未来方向
1. **单 GPU 评测**：实验仅在单 V100 上验证，多卡/多节点 serverless pool 下的跨实例协同 admission 未涉及。
2. **离线 profiling 依赖**：$B_0, \ell_{min}, \alpha$ 需离线 sweep 标定；模型/精度切换时需重新校准，动态场景适应性有限。
3. **Poisson 到达假设**：实际 serverless 负载可能呈现 self-similar 长程相关或 cross-tenant 突发耦合，workload 多样性仍有欠缺。
4. **KV-cache 动态管理忽略**：TWP 仅用 headroom 做 admission guard，未与 PagedAttention 的 page 回收/复用策略联动优化。
5. **Starvation 强制 admission 的 TTFT 代价**：$T_{max}$ 保护长尾但可能引入个别高 TTFT；多目标 Pareto 权衡未展开。
6. **未覆盖 speculative decoding / LoRA 多 adapter**：当前仅单模型单权重；多 adapter 场景下 admission 维度需扩展。

## 研究启发与可借鉴点
1. **ITL 作为拥塞信号的理论优势**：可迁移到任何 autoregressive 生成服务（text/speech/image diffusion decode），用 step wall-clock 直接映射用户感知体验，避免 queue depth 的"假拥塞"误判。
2. **Token Load vs Batch Size 的建模选择**：TWP 用剩余 token 总数 $K_t$ 而非请求数 $B_t$ 做控制变量，更贴合 compute/memory 压力的真实度量——建议在同类型推理调度中沿用此指标。
3. **双信号反馈（ITL + KV-cache 消耗速率）**：分离"计算拥塞"和"内存压力"两个维度，各自独立调控 admission 阈值，设计简洁且可解耦调试。
4. **Chunked prefill 与 admission 解耦**：将 chunk size $C$ 作为独立调参项，通过 offline characterization 定位 TTFT/ITL tradeoff 拐点，方法论可复用于其他 staging 策略。
5. **与现有 serving stack 正交集成**：TWP 作为外部调度层无需侵入 kernel，降部署门槛；未来可与 vLLM/Truss 等框架做 PR 级集成验证通用性。

## 关键术语表
- **Inter-Token Latency (ITL)**：相邻两个输出 token 到达用户的时间间隔，直接决定流式体验；TWP 的核心拥塞信号。
- **Time-to-First-Token (TTFT)**：从请求提交到首个输出 token 返回的延迟，含冷启动 + prefill + 首 decode step。
- **Token Window ($H_t$)**：TWP 动态维护的 token load 软上限，通过反馈控制自适应伸缩以预防过载。
- **Saturation Batch Size ($B_0$)**：ITL 由 $\ell_{min}$ 跳升至线性增长的拐点 batch size，offine profiling 标定。
- **Noisy-neighbor Penalty**：新请求一次性 prefill 抢占共享 HBM 总线，导致所有活跃请求 ITL 瞬时升高数倍的干扰效应。
- **Chunked Prefill**：将 admitted 请求的 prompt 切成固定大小 chunk，跨多个 decode step 分批注入，bound 单步干扰。
- **Token Load ($K_t$)**：活跃集中所有请求剩余输出 token 总数，TWP 的控制目标而非 batch size。
- **Coefficient of Variation (CV) of ITL**：ITL 分布的变异系数，用于量化生成稳定性；TWP 在此指标上达最低 0.0144。

## 可复现要素
- **数据集**：GSM8K 数学推理基准生成请求（公开可复现）。
- **代码**：论文未提供开源仓库链接，TWP 实现为 Python 调度层，集成于自定义 serving pipeline（需自行参照算法描述实现）。
- **权重**：Meta-Llama-3.2-3B FP16（HuggingFace 公开）。
- **关键超参**：$H_0 = 400$ tokens、$\gamma = 0.15$、$C = 64$ tokens、$T_{max}$ 未明确数值（论文仅说"超过阈值强制 admission"）。
- **硬件**：NVIDIA Tesla V100 32GB、CUDA 11.8。
- **Profiling 参数**：$B_0$、$\ell_{min}$、$\alpha$ 需离线 sweep 标定（论文未附具体脚本）。
