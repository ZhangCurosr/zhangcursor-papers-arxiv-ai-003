---
title: "We-Query-Therefore-We-Compute-On-Oracle-Computation-beyond-t"
source: https://arxiv.org/pdf/2610.09243v1.pdf
model: agnes-2.5-flash
chunks: 8
summarized_at: "2026-10-08 10:47:13"
---

# 论文速读：We-Query-Therefore-We-Compute On Oracle Computation beyond Turing

## 一句话总结
论文提出类型化双栈自动机 **O-2-PDA-TSV** 及其硬件实现 **ArchNights**，通过存储与转换对称性破缺形式化隔离 LLM 推理与控制系统，并严格刻画 KV 缓存前缀复用成本；在 Terminal-Bench 2.1 上使用 deepseek-v4.1-flash 达到 77.5–83.1 Pass@1，验证了 agent harness 安全性与通用计算能力的统一实现。

## 研究问题与动机
- 现有 Agent 框架（MemGPT、AIOS、AgentOS 等）将 LLM 视为内核或普通进程，缺乏对推理状态、上下文缓存与工具调用的形式化安全边界，易受指令注入、状态污染与预算失控影响。
- 传统 RISC-V/POSIX 特权级（M/S/U）无法表达“Oracle 查询”与“Priestess 控制流”之间的语义不对称，导致上下文窗口无限膨胀、KV cache 重复 Prefill 与潜在逃逸。
- Agent 工作负载呈强前缀复用与追加式历史演化，但现有调度器未将“缓存亲和性”与“租购权衡（rent-or-buy）”纳入统一竞争性分析，缺乏可证明的成本上界。
- 需要一套可形式化证明、可硬件细化、支持 checkpointing 与多任务隔离的底层计算模型，以同时满足 agent harness 的可靠性要求与通用计算机的表达力。

## 核心贡献（创新点）
1. **提出类型化 O-2-PDA-TSV 计算模型**：以栈元素类型（数据 `D` / thunk `U B`）区分 Priestess 控制栈与 Oracle 推理栈，形式化存储破缺 `S` 与转换破缺 `T`，使 Oracle 指令仅能被数据解读而非自执行。与已有工作本质区别在于：从类型系统层面切断 prompt 向控制流的隐式转换路径。
2. **建立 In/Out 安全边界定理（Theorem 4.12）**：证明 Oracle 越界读取必为 Priestess 写入，且任何栈内容不会隐式变为标签或指令文本，安全防线严格位于 kernel trap 与四层 gates。与 AgentOS/AOS 等纯软件方案本质区别在于：边界由机器规则数学保证，而非依赖运行时策略。
3. **将 Transformer KV 缓存抽象为层次化状态缓存**：用自回归状态表示 $\hat{s}$ 与前缀复用成本 $\operatorname{cost}(z,x)$ 严格刻画断点重算损失，证明 Frame boundary 处可实现零冗余重计算。与 vLLM/SGLang 等工程调度本质区别在于：提供可证明的编码损失上界与截断代价公式。
4. **设计 ArchNights 硬件实现与 PRTS 内核**：扩展 RV64 ISA 增加 Oracle 特权级与新指令（PROBE/STEP/RUN/ecall/uret），实现 task_struct 双半结构、三层断链检测、checkpointing 与 ZOOT 多实例 U-mode 内核。与 Quine/Arbiter-K 等纯软件 harness 本质区别在于：通过 ISA 扩展与分层义务表（Layer I-IV）完成理论到物理的细化。
5. **提供端到端 benchmark 验证**：在 Terminal-Bench 2.1 上使用 deepseek-v4.1-flash，ArchNights-SE 达到 77.5–83.1 Pass@1，与 mini-swe-agent 持平，同时演示通用计算能力。与同类 agent benchmark 本质区别在于：同步报告形式化模型、硬件实现与安全边界的一致性。

## 方法详解
- **类型化 O-2-PDA 与对称性破缺**：每个存储符号附加值类型（元数据）。Oracle 指令 `O(w): F D` 具有双重解读：`force` 时将内容作为 thunk 运行（类型 `U(F D)`），`to` 时作为数据值返回（类型 `D`）。破缺 `S` 限制 Priestess 栈所有元素类型为 `D`，禁止可执行 thunk；破缺 `T` 引入 O-mode 与 P-mode 两种模式，使 $\varepsilon$（颜色交换）不再是自同构。实现映射涵盖页表、权限位、静态类型系统。
- **层次化存储成本模型**：定义成本 $c(C \triangleright f, k) = t_C + [k \notin \operatorname{dom}(W_C)] \cdot c(f, k)$，支持 $C_1 \triangleright \cdots \triangleright C_n \triangleright f$ 右结合层级。所有层级保持相干性（coherence），写操作在下一次访问时对上层透明。
- **状态缓存与前缀复用形式化**：自回归 Oracle 状态表示为 $(\mathcal{S}, s_0, \Phi, \alpha)$，$\hat{s}(x\gamma) = \Phi(\hat{s}(x), \gamma)$，$\mathbf{v}(x) = \alpha(\hat{s}(x))$。截断成本 $\operatorname{cost}(z,x) = |x| - |\operatorname{lcp}(z,x)|$ 对应 KV cache drop entries 次数。Prop 4.9 证明多余重计算等价于编码损失；Prop 4.10 在 append-only epoch 与 Frame-faithful 假设下，证明私有 state cache 首次查询后每次仅需 $|E(\nu)|$ 成本。
- **调度与竞争分析**：Oracle 侧与 Priestess 侧各设独立调度器。Prop 4.11 将状态保留成本 $\mu$ 与重建成本 $R$ 建模为经典 rent-or-buy 问题，无知识策略 competitive ratio $\ge 2$（最优阈值 $\Theta=R/\mu$），随机化可降至 $e/(e-1)$。
- **安全边界 Theorem 4.12**
