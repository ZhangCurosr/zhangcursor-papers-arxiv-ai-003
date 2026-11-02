---
title: "Stable-Token-level-Scheduling-for-Serverless-LLM-Inference"
source: https://arxiv.org/pdf/2610.17689v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-02 09:05:17"
field: "LLM serving 系统调度"
keywords: ["serverless LLM inference", "token-level scheduling", "inter-token latency", "admission control", "TTFT", "chunked prefill", "feedback controller"]
innovations: ["提出 ITL 为零开销拥塞信号，建立 decode 步粒度闭环反馈", "Token Window Pacing 以 token 负载为控制变量替代 batch 大小", "双守护准入门（token 预算+KV-cache 余量）结合饥饿安全复合排序"]
benchmarks: ["GSM8K", "Poisson 18 req/s 60s workload", "ServerlessLLM re-implementation"]
---

# 论文速读：Stable-Token-level-Scheduling-for-Serverless-LLM-Inference

## 一句话总结
论文提出了 **Token Window Pacing (TWP)** —— 一个基于运行时 ITL 反馈的令牌级闭环调度框架，通过动态调节 token 窗口大小控制请求准入，在 serverless LLM 推理中实现稳定的 TTFT 和更低的 P99 尾延迟，吞吐量较 ServerlessLLM 提升 45%、较传统调度器提升 58%。

## 研究问题与动机
- **现有调度器只优化 TTFT，忽视 ITL**：当前 serverless LLM 系统（如 ServerlessLLM）以快速响应首令牌为目标，但未考虑解码阶段的逐令牌延迟（ITL），导致高负载下用户体验抖动。
- **无反馈的准入控制导致 GPU 过饱和**：在突发流量下，基于到达时间的开环调度器会过度接受请求，使 active batch 超过饱和度阈值 $B_0$，ITL 线性增长且恢复缓慢。
- **队列深度 ≠ 拥塞信号**：队列可不断增长而 ITL 仍保持低位（带宽受限区），仅凭队列深度做限流会误伤有效容量。
- **KV-cache 占用 ≠ 延迟代价**：大量短输出请求可能占满内存但仍处于带宽受限区，内存信号无法反映计算压力。

## 核心贡献（创新点）
1. **ITL 作为拥塞信号的理论与实证刻画**：证明 ITL 是 decode 步骤内唯一零开销的实时拥塞指标，建立其与服务率的解析关系，给出无控制时的不稳定性证明（定理 §IV-A）。
2. **Token Window Pacing 闭环调度框架**：提出以 token 负载 $K_t$ 而非 batch 大小 $B_t$ 为控制变量的反馈控制器，通过等式 (3)–(4) 动态调整 token 窗口 $H_t$，首次实现 decode 级细粒度的准入闭环。
3. **双守护准入门 + 饥饿安全复合排序策略**：设计 token 预算与 KV-cache 剩余量双条件准入门，排序策略融合等待时间（防饥饿）、token 均衡度（提升批效率）与公平性三项评分，形式化于 Algorithm 1。
4. **Chunked Prefill 噪声邻域惩罚有界化**：将预填充拆分为固定大小 chunk（本文取 $C=64$ tokens），使每步引入的内存总线争用被限定为 $\ell_{\min}$ 的固定比例，从理论上界定了 "噪声邻居" 惩罚。
5. **与模型/内核无关的可插拔部署**：TWP 仅作为调度层集成于现成 serving 后端之上，不修改权重、attention kernel 或 GPU 执行原语，兼容任何暴露 step-level decode 接口的框架。

## 方法详解

### 1. ITL 拥塞信号模型
定义第 $k$ 步的 ITL：$\ell_k = t_k - t_{k-1}$。系统存在饱和度 batch 大小 $B_0$：
$$
\ell(B_t) = 
\begin{cases}
\ell_{\min}, & B_t \leq B_0 \\
\ell_{\min} + \alpha(B_t - B_0), & B_t > B_0
\end{cases}
$$
其中 $\alpha$ 为每增加一个请求带来的额外 decode 步延迟。最大服务率为 $1/\alpha$，当到达率 $\lambda \bar{L} > 1/\alpha$ 时队列发散，证明系统需显式 token 负载控制。

### 2. 动态 Token 窗口
token 负载 $K_t = \sum_{i \in A_t} r_i(t)$（所有活跃请求剩余输出 token 之和），目标窗口 $H_t^*$：
$$
H_t^* = H_0 \cdot \frac{1 + \min(1,\, q_t / Q)}{\max(0.5,\, \sigma_t)}, \quad \sigma_t = \hat{\ell}_t / \ell_{\min}
$$
平滑更新：$H_{t+1} = H_t + \gamma(H_t^* - H_t)$。ITL 低且队列增长时窗口扩张；ITL 升高时窗口收缩。

### 3. 复合请求排序策略
三项评分加权合并（$w_1 > w_2 > w_3$）：
- **饥饿优先级**：等待时间超过 $T_{\max}$ 时强制准入，绕过两重守护门。
- **Token 均衡度**：$T(r) = \exp\!\bigl(-|\hat{L}(r) - \bar{r}_t|/(\bar{r}_t+1)\bigr)$，偏好与当前 batch 平均剩余长度相近的请求，减少长尾阻塞步。
- **公平性**：按作业大小归一化等待时间，防止短请求流长期驱逐大请求。

### 4. 双守护准入门
候选请求仅在同时满足以下条件时才被接纳：
1. $K_t + r_i \leq H_t$（token 负载不超窗口）
2. 可用 KV-cache 内存 $> \delta_{\mathrm{kv}}$（预留内存余量）
二者独立生效：内存信号防 OOM，ITL 信号控饱和。

### 5. Chunked Prefill
准入后不一次性提交整段 prompt，而是按 chunk 大小 $C$ 分步在 decode 间隙执行，使每步的内存总线争用被限制为全量 prefill 的一小部分。

## 实验与结果

- **硬件与模型**：NVIDIA Tesla V100 32GB，Meta-Llama-3.2-3B（FP16）
- **工作负载**：GSM8K 基准，Poisson 到达（λ=18 req/s，60s），提示长度异构（60% 短/30% 中/10% 长），$B_0=8$
- **对比基线**：FCFS、Round-Robin、Priority、DynamicBatch、ServerlessLLM（同环境 re-implement）
- **关键超参**：$H_0=400$ tokens，$\gamma=0.15$，$C=64$ tokens（离线表征确定）

| 指标 | TWP | 次优基线 | 提升 |
|---|---|---|---|
| 吞吐量 | 0.197 req/s | ServerlessLLM 0.136 | **+45%** |
| 较传统调度器 | 0.197 | FCFS 0.121 | **+58%** |
| P99 TTFT | 66 ms | ServerlessLLM 78 ms | ↓ 15% |
| ITL CV | 0.0144 | Round-Robin 0.0203 | ↓ 29% |
| 稳态 ITL | 43.5–43.8 ms | 61–62 ms（基线尾部） | 更稳定 |

- **四场景鲁棒性**（低/中/高/突发负载）：TWP 在所有场景均取得最低 ITL CV 和最高吞吐；高负载下 FCFS 吞吐骤降至 254 req/s，TWP 保持 2709 req/s。
- **TTFT 稳定性**：TWP 全程维持 ~50 ms，Round-Robin/ServerlessLLM 出现 77–85 ms 尖峰。

## 相关工作脉络

- **ServerlessLLM [2]**：多 Tier checkpoint 加载 + 热迁移，大幅降低冷启动，但未解决 warm container 阶段 decode 级拥塞，本文在其之上增加反馈准入层。
- **Orca [3] / vLLM [4]**：Continuous batching 与 PagedAttention，奠定 step-synchronous control 与 per-step 内存观测基础，TWP 借其 measurable state 作反馈变量。
- **Sarathi-Serve [5]**：Chunked prefill 缓解 noisy-neighbour 惩罚，但未将 ITL 观测回灌至准入决策，TWP 将其嵌入闭环。
- **Splitwise [14]**：将 prefill 与 decode 分离至独立硬件池，TWP 在同一共享池内实现 phase-awareness，无需额外硬件。
- **TokenScale [16] / LAPS-SD [17]**：Token-level 信号优于 request-level 的共识，本文从"预测扩缩容"走向"逐 step 准入控制"，实现维度更细。
- **Bari et al. [15]**：Deadline-driven criticality 调度需已知 deadline 且面向单实例，TWP 无需 SLO 先验、面向多租户共享 pool。

## 局限性与未来方向

- **离线表征依赖**：$B_0$、$\ell_{\min}$、$C$ 等参数需一次性离线 sweep 获得，不同模型/显存配置需重新标定。
- **单 GPU 评估**：当前实验在单卡上进行，未验证分布式 serverless 集群下的可扩展性。
- **非冻结冷启动交互**：TWP 的 token 窗口仅作用于 warm container 的 decode 阶段，与冷启动调度（cs_i）解耦，未联合优化。
- **固定权重的排序策略**：$w_1,w_2,w_3$ 与 $T_{\max}$ 为静态超参，未探索在线自适应或强化学习调优。
- **未考虑 LoRA adapter 切换开销**：多 adapter serverless 场景下，context 切换的开销未被纳入 admission gate。

## 研究启发与可借鉴点

- **ITL 作为零开销拥塞信号**的思路可直接移植到任何基于 decode-step 的 serving 后端（vLLM/TGI/SGLang），只需读取 step wall-clock time。
- **token 负载替代 batch 大小**的控制视角更具语义精确性：一个接近完成的短请求与一个新到长请求对未来的压力天差地别，$K_t$ 捕捉了这一差异。
- **双守护门设计**（计算负载 + 内存余量）可泛化到多资源约束场景（如 CPU/GPU/PCIe 带宽），各维度独立设阈值互不干扰。
- **Starvation-first 的复合排序**保障了最坏情况延迟上界，这一"硬约束优先"原则适用于任何需要 fairness 保证的生产调度器。
- **与本团队的结合点**：若团队做 LLM 推理平台，可复用 TWP 的 feedback controller 接口；若做多租户隔离，可将 token 窗口改写为 per-tenant 配额变量，实现细粒度 SLI 保障。

## 关键术语表

- **ITL（Inter-Token Latency）**：相邻两个输出 token 之间的墙钟时间间隔，决定流式体验的平滑度。
- **TTFT（Time-to-First-Token）**：从请求提交到首个输出 token 到达的总延迟，含冷启动、prefill 与首 decode 步。
- **Token Window（H_t）**：系统在当前步允许的最大剩余 token 负载上限，由 ITL 与队列深度反馈动态调节。
- **Saturation Batch Size（B_0）**：decode 从带宽受限区转入计算受限区的临界 batch 大小，超过后 ITL 线性增长。
- **Noisy-Neighbor Penalty**：新请求 prefill 时与活跃 decode 竞争 HBM 总线导致的瞬时 ITL 抬升。
- **Chunked Prefill**：将 prompt 拆分为固定大小 chunk，跨多个 decode 步逐步完成预填充以降低瞬时争用。
- **Token Load（K_t）**：所有活跃请求剩余输出 token 数之和，TWP 的直接控制变量。
- **Slowdown（σ_t）**： smoothed ITL 相对基线 $\ell_{\min}$ 的归一化倍数，作为窗口收缩的反馈系数。

## 可复现要素

- **数据集**：GSM8K（公开）；Poisson λ=18 req/s，60s 试验窗口
- **代码**：论文未提供开源仓库，TWP 以 Python 调度层实现并集成于自定义 pipeline
- **模型/权重**：Meta-Llama-3.2-3B（FP16，HuggingFace 公开）
- **关键超参**：$H_0=400$ tokens，$\gamma=0.15$，$C=64$ tokens，$B_0=8$（V100 32GB  profiling 所得）
- **硬件**：NVIDIA Tesla V100 32GB，CUDA 11.8
