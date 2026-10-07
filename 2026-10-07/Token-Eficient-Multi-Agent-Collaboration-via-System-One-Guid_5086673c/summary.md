---
title: "Token-Eficient-Multi-Agent-Collaboration-via-System-One-Guid"
source: https://arxiv.org/pdf/2610.08155v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-07 22:27:47"
field: "多智能体协作效率优化"
keywords: ["多智能体系统", "System One", "token效率", "协调解耦", "证据定位", "计算分工"]
innovations: ["将协调决策与开放式推理解耦，用预训练System One模型替代LLM做有界协调决策", "决策-证据闭环机制：轻量级控制器+证据Reader+LLM Worker协同", "通过cumulative通信预算与确定性任务编译器实现token-efficient自适应协作"]
benchmarks: ["MMLU", "GSM8K", "AQuA", "MultiArith", "SVAMP", "HumanEval", "WebSRC"]
---

# 论文速读：Token-Eficient-Multi-Agent-Collaboration-via-System-One-Guid

## 一句话总结
本文提出 **S1-MAS**，一种基于 System One 引导的计算分工框架，将多智能体系统中的协调决策（任务选择、角色分配、停止判断）从昂贵的 LLM 推理中解耦，用轻量级 System One 控制器 + 轻量级 Reader 实现高效协调，保留强大 LLM worker 的开放式推理能力，在七个基准上以更低 token 消耗和延迟取得最优或竞争性准确率。

## 研究问题与动机
- **核心问题**：现有 LLM-based 多智能体系统（MAS）将任务推理与协调操作（任务选择、角色分配、消息路由、上下文管理）紧耦合，导致交互次数增长时 token 开销和延迟急剧上升，限制智能体 Web 服务的可扩展性。
- **现有方法不足**：
  1. AgentVerse、MACNET、DyLAN、SelfOrg 等自适应编排方法仍需反复调用生成式 LLM 来做出协调决策，产生大量控制指令生成开销。
  2. 协调类操作本质上是受限动作空间下的有界决策（如选择下一个任务、判断是否停止），却被当作自由文本生成处理，浪费计算资源。
  3. 通信优化工作（AgentPrune、MAD-M²、LLMLingua）主要优化协调决策之后的通信内容压缩，未解决"决定访问/交换什么信息"这一协调决策本身的 LLM 推理成本。
  4. 已有将 System One 模型与 LLM 结合的研究局限于特定应用场景（如动作选择、边缘服务编排、智能体记忆管理），尚未系统探索其在通用 MAS 联合任务编排与证据感知通信中的潜力。

## 核心贡献（创新点）
1. **新视角：协调与推理的计算分工**。首次明确识别 LLM 协调生成为多智能体系统的主要计算瓶颈，并提出将"有界协调决策"与"开放式推理"分离的计算分工框架，而非依赖单一 LLM 完成所有操作。
2. **S1-MAS 框架**：首个系统性探索预训练 System One 模型用于通用 LLM-based 多智能体联合任务编排与证据感知通信的工作；通过轻量级控制器（Laya）+ 证据 Reader（Qwen3-0.6B）+ LLM Worker 的决策-证据闭环，实现无需任务特定训练的可自适应协作。
3. **全面评估验证效率-精度平衡**：在七个多样化基准（数学推理、知识型问答、代码生成、Web 问答）上证明 S1-MAS 达到 state-of-the-art 精度，同时较 AgentVerse/DyLAN/SelfOrg 降低 GPT-4o token 消耗 44.9%–97.2%、端到端延迟降低 37.8%–93.0%。

## 方法详解
**整体架构**：S1-MAS 采用"决策-证据闭环"（decision–evidence loop），包含三个核心组件：
- **System One 控制器**（实例化为 Laya）：负责任务编排，做出有界协调决策（选择检查条件、决定下一个任务、判断是否停止）。
- **轻量级 Reader**（实例化为 Qwen3-0.6B）：从授权响应中检索与条件相关的证据，为控制器决策提供 grounding 信息。
- **LLM Worker**（GPT-4o/GPT-4o-mini）：负责实质性开放式推理、批判与响应生成。

**关键设计**：
1. **状态定义**：$S_t = (q, \mathcal{H}_t, \mathcal{A}_t, B_{\text{src}} - u_t)$，其中 $\mathcal{H}_t$ 为已完成任务-响应对的有序归档，$\mathcal{A}_t$ 为检查触发记录（避免重复检查），$B_{\text{src}} - u_t$ 为剩余通信预算。
2. **检查查询与证据定位**：控制器从问题语句集 $\mathcal{C}(q)$ 中选择未检查的条件形成检查查询 $\chi_t$；Reader 通过字符偏移定位候选答案中相关证据段落，生成证据记录 $\ell_t = \mathcal{R}(q, \chi_t)$，作为控制器的额外上下文。
3. **可行任务构建**：确定性编译器根据状态 $S_t$ 和检查查询 $\chi_t$ 筛选满足前置条件且符合通信预算的任务，形成菜单 $\mathcal{W}_t = \text{Compile}(S_t, \chi_t)$。任务操作类型包括：Solve（独立求解）、Decompose（分解求解）、Reverse（逆向推理）、Review（审查当前答案）、Repair（修复缺陷）、Contrast（对比备选答案）。
4. **System One 引导的动作选择**：控制器使用 $S_t$、$\chi_t$、$\ell_t$ 从 $\mathcal{W}_t$ 中选择下一个任务 $j_{t+1}$ 或选择 finish 终止协作。选择过程通过小规模组内比较并平均得分以降低顺序敏感性。
5. **角色与源分配**：选定的任务 $j_{t+1}$ 决定 worker 角色和所需源响应集合 $\mathcal{P}(j_{t+1})$；对于 Review/Contrast 任务，局部化证据 $\ell_t$ 作为注意力提示高亮相关段落。
6. **通信预算控制**：cumulative 源 token 传输预算约束 $u_t = \sum_{i=1}^{t} \text{tok}(\tilde{e}(j_i)) \leq B_{\text{src}}$，防止重复传输相同响应导致的上下文处理开销。
7. **自适应停止与答案交付**：当存在候选答案且满足停止条件时，控制器可选择 finish；交付规则基于多数投票（多项选择/数值答案）或实用性优先（代码生成），返回归档中的已有响应，无需额外 LLM 调用。

## 实验与结果
**数据集**：MMLU（知识推理）、GSM8K/AQuA/MultiArith/SVAMP（数学问题求解）、HumanEval（代码生成）、WebSRC（结构化网页问答）。

**基线**：单模型方法（Vanilla、CoT、SC-CoT）；固定拓扑多智能体（Chain、Tree、Complete、Random）；5 个 MAS 框架（AgentVerse、MACNET、SelfOrg、MAD-M²(S)、DyLAN）。

**主要结果**（GPT-4o 作为 worker）：
| 指标 | S1-MAS | 最佳基线 | 提升 |
|------|--------|----------|------|
| 平均准确率（6 基准）| **93.97%** | DyLAN 93.12% | +0.85pp |
| MMLU | **91.94%** | DyLAN 91.07% | +0.87pp |
| HumanEval | **91.46%** | MACNET 90.36% | +1.10pp |
| GPT token 消耗（平均）| **455 input / 392 generation** | DyLAN 8006/2220 | 降低 78.3%–94.0% |
| 平均端到端延迟 | **16.8 秒** | DyLAN 约 87 秒 | 降低 58.2%–91.7% |

**WebSRC 结果**：S1-MAS 获得最高精确匹配率（90.11%），token 消耗仅为 AgentVerse 的 15.0%、DyLAN 的 48.8%、SelfOrg 的 17.2%。

**消融实验**：
- 移除 System One 控制器改用随机路由：平均准确率从 84.17% 降至 82.34%（MMLU/AQuA/HumanEval 三项测试集）。
- 用 GPT-4o-mini 替代 System One 控制器：准确率 83.14%（低于 S1-MAS 的 84.17%），但 token 消耗增至 2.19M（vs. 0.54M），节省 75.5% token。
- 移除 Reader（无证据定位）：准确率从 84.17% 降至 82.41%，证明证据定位对性能贡献显著（+1.76pp）。

## 相关工作脉络
1. **AgentVerse（Chen et al. 2023）**：通过招募专业化角色实现协作问题解决；S1-MAS 与其定位差异在于不依赖 LLM 做协调决策，而是用预训练 System One 控制器分离协调与推理。
2. **MACNET（Qian et al. 2025）**：通过有向无环图显式建模通信结构；S1-MAS 强调在通信决策层面降低协调开销，而非仅优化通信结构。
3. **DyLAN（Liu et al. 2024）**：动态调整智能体参与；两者均实现自适应协作，但 DyLAN 仍用 LLM 做协调决策，S1-MAS 用轻量 System One 模型替代。
4. **SelfOrg（Tastan et al. 2026）**：基于 peer contribution 估计重组通信；S1-MAS 通过证据感知的决策-证据闭环实现自适应，避免重复 LLM 调用。
5. **MAD-M²(S)（Tian et al. 2026）**：通过掩码辩论记忆减少通信载荷；S1-MAS 在通信决策之前介入，从源头减少协调决策的 LLM 推理成本。
6. **LLMLingua（Jiang et al. 2023; Pan et al. 2024）**：压缩提供给 LLM 的上下文；属于事后通信内容优化，S1-MAS 从协调决策机制本身降本。

## 局限性与未来方向
- **系统性评估范围**：实验限于六个标准化基准加一个 Web 问答基准，尚未在更复杂的长程多步任务或真实 Web 服务场景中验证。
- **System One 模型依赖**：框架性能依赖所选 System One 模型（Laya）的能力；若模型在处理复杂协调情境时出现决策偏差，可能影响整体效果。
- **任务操作集固定**：当前定义了六种操作（Solve/Decompose/Reverse/Review/Repair/Contrast），对于超出此词汇表的任务类型可能需要扩展。
- **未来方向**：论文指出可扩展至多页 Web 检索与问答场景，进一步提升实际 Web 服务中的适用性。

## 研究启发与可借鉴点
1. **协调-推理分离范式**：将 MAS 中的"有界决策"与"开放式推理"解耦，为后续工作提供了通用的效率优化思路；可迁移至其他需要频繁协调决策的智能体系统（如工具调用编排、多轮对话管理）。
2. **证据定位作为注意力提示**：轻量级 Reader 定位证据段落并以字符偏移形式提供高亮，既保留了完整上下文供 worker 深度推理，又为协调决策提供精确 grounding；该设计可直接复用或改进于其他需要"选择性注意力"的多智能体框架。
3. **通信预算的 cumulative 约束**：通过 $B_{\text{src}}$ 累计预算而非 per-step 预算控制源传输，有效防止重复传输的上下文处理开销；该机制可推广至任何需要限制跨 agent 信息传递的系统。
4. **确定性编译器筛选可行任务**：用确定性规则（前置条件 + 预算约束）过滤任务菜单，将协调决策转化为受限选择题，大幅降低 System One 模型的决策复杂度；此设计模式适用于其他需要将连续决策空间离散化的场景。
5. **无需任务特定训练**：S1-MAS 的所有组件均无需微调即可泛化到不同任务类型，证明预训练 System One 模型的零样本协调潜力；这为快速部署新场景提供了实践路径。

## 关键术语表
**System One**：基于双系统理论的概念，指快速、直觉性、低计算成本的决策模式；本文中特指预训练的轻量级决策模型（如 Laya），用于有界动作选择。
**决策-证据闭环（Decision–Evidence Loop）**：S1-MAS 的核心机制，控制器做出协调决策 → Reader 定位证据 → 决策反馈，形成自适应协作循环。
**检查查询（Inspection Query）**：由控制器生成的 $\chi_t$， pairing 一个待检查的问题条件与当前候选答案，用于指导证据定位。
**通信预算（Communication Budget）**：累积源 token 传输上限 $B_{\text{src}}$，约束跨 worker 传输的历史响应总长度。
**任务操作词汇表（Task Operation Vocabulary）**：S1-MAS 定义的六种协作操作（Solve/Decompose/Reverse/Review/Repair/Contrast），构成协调决策的可选动作空间。
**输出标识符（Output Identifier）**：用于跨响应比较答案的键值，如选项标签、短文本答案或代码片段。
**Attention Cue**：局部化证据段落作为高亮提示，引导 worker 聚焦于相关上下文，同时保留完整响应以供深度推理。

## 可复现要素
- **数据集**：MMLU、GSM8K、AQuA、MultiArith、SVAMP、HumanEval、WebSRC（均为公开基准）
- **代码/权重**：论文未明确声明代码开源状态；System One 控制器使用 Laya（Convai Innovations，Hugging Face 模型库），Reader 使用 Qwen3-0.6B（公开模型）
- **关键超参**：
  - Worker 模型：GPT-4o（主实验）、GPT-4o-mini（消融实验）
  - 最大 worker 执行次数 $T_{\text{max}} = 4$
  - 输出 token 上限 $B_{\text{out}} = 2{,}048$
  - 通信预算 $B_{\text{src}} = 2{,}048$ tokens/问题
  - Worker temperature = 1.0，每次生成单个响应
  - Reader 使用 greedy decoding，non-thinking 模式
