---
title: "We-Query-Therefore-We-Compute-On-Oracle-Computation-beyond-t"
source: https://arxiv.org/pdf/2610.09243v1.pdf
model: agnes-2.5-flash
chunks: 8
summarized_at: "2026-10-09 10:18:44"
field: "AI 硬件安全与可信执行"
keywords: ["LLM-as-Oracle", "Hardware Security", "RISC-V Extension", "Frame-faithfulness", "Formal Verification", "AI Accelerator"]
innovations: ["提出 ArchNights 框架：将 LLM 作为安全可信 Oracle 嵌入 RISC-V 硬件指令集", "定义 Frame-faithfulness 形式化契约，确保 LLM 输出在 Frame 边界处历史一致", "分类推理生命周期策略（Kept/Dropped/Hidden）并分析其对系统安全与性能的影响"]
benchmarks: ["形式化验证（Theorem 4.4, Proposition 4.10）", "硬件指令集正确性", "安全属性验证（SMEP/SMAP, replay policy）"]
---

# 论文速读：We-Query-Therefore-We-Compute: On Oracle Computation beyond Tokens

## 一句话总结
论文提出 **ArchNights** 框架，将大语言模型（LLM）作为安全可信的 **Oracle** 嵌入 RISC-V 硬件指令集（RoCC），通过 **Frame-faithfulness** 契约确保历史状态一致性，实现可验证、可重放的 LLM 硬件加速查询。

## 研究问题与动机
- **LLM 不可信与黑盒化**：大模型输出不可验证，传统将其封装为服务会导致历史污染、重放攻击和安全漏洞。
- **现有封装方案缺陷**：Arbiter-K 依赖 Python 原型 + taint propagation，Poon 的认知指令权限由 OS 控制，神经网络处理单元无历史无自有模式。
- **Token 语义边界缺失**：现有框架未定义 LLM 输出与硬件执行状态之间的形式化契约，导致 Frame 边界处状态不一致。
- **Oracle 调用缺乏硬件级原语**：现有工作（如 Prietess）将模型应答当作数据流，而非可验证的机器指令。

## 核心贡献（创新点）
- **首个 LLM-as-Oracle 硬件指令集设计**：提出单一 Oracle 指令（custom-0 opcode）+ O-mode，通过 `uret` 进入，请求语义由历史 + $L_\Lambda$ 固定，而非指令集本身。
- **Frame-faithfulness 形式化契约**：精确定义 Oracle 输出在 Frame 边界处保持历史一致的条件（Theorem 4.4），确保每次答案只生成一次、重编最多一次。
- **ArchNights 完整硬件-软件协同框架**：从 RISC-V 扩展指令、异常处理、预算控制到解码/渲染管道，提供端到端可验证的 LLM 调用机制。
- **四种推理生命周期策略的形式化分类**：Kept/Dropped by decoder/encoder/kernel/Hidden，明确各策略对 Frame-faithfulness 的影响与代价上界。
- **安全通道与重放攻击的硬件级防护**：通过 SMEP/SMAP、append-only 授权日志、预算 trap 等机制阻断 Meltdown/Spectre 类 transient channel 与自适应逐 bit 洗白攻击。

## 方法详解
### 指令集架构（ISA）
- **Oracle 指令**：32-bit R-type，`custom-0 opcode (0x0001011)`，`funct3=111`。
- **funct7 子字段**：op（PROBE/STEP/RUN）、dir（栈增长方向）、profile（P0/P1/P2/P32）。
- **Profile 配置**（Table 15）：
  - P0：XLEN=64, offset=24, steps=36, budget=40, pages=12, max segment=4095
  - P1：XLEN=64, offset=28, steps=32, budget=32, pages=16, max segment=65535
  - P2：XLEN=64, offset=32, steps=24, budget=24, pages=20, max segment=1048575
  - P32：XLEN=32, offset=22, steps=6, budget=6, pages=12, max segment=1023
- **Result register (rd)**：`status[63:60] + steps[59:24] + bytes[23:0]`
- **Status codes**：0=Continue, 1=Halt, 4=无空间, 5=budget耗尽, 6/9=输入过短/不可编码, 8=非法参数, 10=超时(可恢复), 11=服务不可用, 12=答案不完整/畸形
- **Descriptor**（DESC=1）：32-byte 固定头，含 format(0xFF)、checksum、flags、extension length、request number(32-bit)、输入/输出/总计计数(64-bit)

### Frame-faithfulness 与解码/渲染
- **Decoding**：将生成符号映射到 λ 的解码器可选择丢弃；模型推理接口返回推理内容与答案分开即为典型。
- **Rendering**：聊天模板属编码器一部分；仅当模板将历史答案渲染为与生成时相同 token 序列时，才满足 Frame-faithful。
- **Theorem 4.4(2)** 刻画 Frame-faithfulness 在 Frame 边界成立的条件。
- **失效代价有界且局部**：下次查询只需重编 $|E(D(y))| - |\mathrm{lcp}(y, E(D(y)))|$ 个符号，最多重编最后一个 Frame。

### 推理生命周期策略
| 方式 | 机制 | Frame-faithful? | 代价 |
|------|------|-----------------|------|
| Kept | D 将推理写入栈，E 整体读回 | 可成立 | Priestess 持有推理，每次请求可找到 |
| Dropped by decoder | D 逐个符号映射为 λ，需 $\Gamma_R$ 标记 | 失败 | 下次查询重编最多 $|E(D(y))|$ 符号；推理状态不复用 |
| Dropped by encoder | D 保留推理，E 按聊天模板剔除 | 失败 | 同 decoder drop；Priestess 仍持有推理 |
| Dropped by kernel | 在 trap 或 swap 时从任务栈移除 | 等价于 rewrite | 开启新 epoch |
| Hidden | 接口以加密/签名块或引用返回推理 | 同 kept/dropped | 编码器变为 partial；块为整帧验证 |

### 异常处理
- **Encoder 异常**：illegal argument / insufficient memory / encoding failure → 退避（kernel 弹出导致无法编码的内容，重试）
- **Compute 异常**：recoverable(timeout)/unrecoverable(service loss) → 可恢复重试；不可恢复终止进程，保存 checkpoint
- **Decoder 异常**：insufficient capacity / decoding failure → 重采样（可被 replay policy 限制）
- **Controller 异常**：budget spent before yield → budget trap（非 fault）
- **Segment full**：answer doesn't fit → kernel 扩大 segment 后重试
- **Frame truncated**：答案被输出限制截断 → kernel 按 rd 字节数退避，压缩历史重试

### 安全机制
- **SMEP/SMAP**：限制 Priestess 访问 Oracle 可写内存（architectural channel）
- **Task observes**：通过 K1/K2，任务历史仅受 Priestess 写入任务和调度时机影响
- **重放防护**：非确定性 Oracle 不可在 epoch 内重采样；Zheng et al. [175] 在 checkpoint/fork/restore/merge 处检查 append-only 授权日志
- **CaMeL [244] 与 LLMbda [21]**：特权模型最多重试 10 次；LLMbda 指出该重试循环允许自适应驱动逐 bit 洗白秘密

### Fork 与通信
- Linux fork 继承文件描述符、管道、工作目录、凭证与信号处置；子进程资源计数从零开始
- **Fork 默认连接父子**：通过 result channel（子写端、父读端）
- **Oracle 不持有管道端点**：管道由 shell/operator/fork 创建
- **Asynchrony 三种放置**：微架构（乱序执行/SMT）、操作系统（内核多任务调度）、不可见

## 实验与结果
> ⚠️ 本文提供的分段笔记中未包含实验章节（§7.1 之前部分为空），故无法提供具体数据集、基线对比数字与结果表格。根据已有信息推断：论文核心贡献在于**形式化框架与硬件设计**，实验可能侧重于 Frame-faithfulness 验证、指令集正确性证明及安全属性验证，而非传统 ML benchmark 性能对比。建议查阅原文 §7 获取完整实验数据。

## 相关工作脉络
- **Arbiter-K [145]**：将模型封装为不可信概率处理单元，通过 Python 原型 + taint propagation + rollback 实现安全；本文差异在于硬件原生指令而非软件封装。
- **Poon [146]**：认知指令（THINK/RECALL/ADAPT），权限由 OS 控制；本文差异在于单指令 + O-mode 硬件调用而非 OS 级权限管理。
- **神经网络处理单元 [147]**：替换程序员标记为近似代码，ISA 扩展调用固定函数；本文差异在于保留 LLM 自有模式与历史状态。
- **BDI Agent 模型 [1,2]**：信念/欲望/意图形式化；本文将其桥接至硬件 Oracle 调用。
- **Priestess [156-158]**：程序保持控制、模型应答调用，反映为 Workflow 形式；本文扩展为硬件原语 + 形式化契约。
- **ReAct [162]**：推理 + 工具行动交织循环；本文通过 Frame-faithfulness 解决其历史一致性问题。
- **CaMeL [244] / LLMbda [21]**：重试次数限制与自适应攻击；本文通过 budget trap + append-only 日志提供硬件级防护。

## 局限性与未来方向
- **实验验证缺失**：分段笔记未涵盖实现原型与 benchmark 结果，硬件正确性需 FPGA/ASIC 验证。
- **Profile 固定配置**：P0-P32 四种 profile 静态定义，无法动态适应不同 LLM 规模与查询复杂度。
- **Hidden 策略局限**：加密/签名块依赖整帧验证，不支持逐符号解码，限制了细粒度状态复用。
- **非确定性 Oracle 重放**：虽通过 replay policy 限制，但完全消除重采样开销仍需进一步研究。
- **扩展至多 Oracle 协同**：当前设计针对单 Oracle，多模型联合调用的历史一致性未讨论。

## 研究启发与可借鉴点
- **Frame-faithfulness 形式化方法**：可为其他 AI 硬件加速器（如矩阵乘单元、Transformer 加速器）提供历史一致性验证范式。
- **单一指令 + 语义绑定设计**：避免指令集膨胀，适用于任何需要"黑盒计算单元"的硬件扩展场景。
- **推理生命周期分类**：Kept/Dropped/Hidden 三分法可直接迁移至 LLM 服务框架的状态管理设计。
- **预算 trap 机制**：将传统 CPU 异常处理思想引入 LLM 调用，为 AI 系统的确定性保障提供新工具。
- **硬件-软件协同安全**：SMEP/SMAP + append-only 日志的组合可推广至其他可信执行环境（TEE）中的 AI 服务部署。

## 关键术语表
**Frame-faithfulness**：Oracle 输出在 Frame 边界处保持历史状态一致的数学契约，确保每次答案只生成一次。
**O-mode**：Oracle 执行模式，通过 `uret` 指令从用户态进入，无地址代码，由硬件管理调用上下文。
**ArchNights**：论文提出的 LLM-as-Oracle 硬件-软件协同框架，包含 RISC-V 扩展指令、异常处理与安全机制。
**Budget trap**：当 Oracle 计算耗尽预设预算时触发的非 fault 异常，阻止无限循环与资源耗尽攻击。
**Kept/Dropped/Hidden**：推理历史的三种生命周期策略，分别对应写入栈、丢弃、加密隐藏。
**RoCC**：RISC-V 用户级自定义协处理器扩展，Oracle 指令部署于此，兼容现有 RISC-V 生态。
**Append-only 授权日志**：仅追加不可篡改的审计日志，用于验证 checkpoint/fork/restore 操作的合法性。
**LCP（Longest Common Prefix）**：两次查询输出序列的最长公共前缀，用于界定重编符号数量上界。

## 可复现要素
- **数据集**：论文未提及（框架性论文，侧重形式化验证）
- **代码/权重**：论文未提及
- **关键超参**：Profile 配置（P0/P1/P2/P32）、budget 值、steps 限制、page 数量 — 见 Table 15
- **硬件平台**：RISC-V + RoCC 扩展，具体 FPGA/ASIC 实现未详述

---
