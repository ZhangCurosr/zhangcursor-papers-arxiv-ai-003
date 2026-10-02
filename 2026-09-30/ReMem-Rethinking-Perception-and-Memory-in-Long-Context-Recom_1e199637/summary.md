---
title: "ReMem-Rethinking-Perception-and-Memory-in-Long-Context-Recom"
source: https://arxiv.org/pdf/2609.37311v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 14:19:08"
field: "推荐系统与LLM Agent交叉"
keywords: ["Recommendation Agents", "Long-Context Reasoning", "OCR-based Perception", "Dynamic Memory", "Reinforcement Learning", "GRPO", "LLM"]
innovations: ["首个结合OCR多模态感知与强化学习增强动态记忆的推荐智能体框架", "块级时序演化记忆机制实现线性复杂度长上下文推理", "Multi-Memory GRPO训练策略将最终奖励传播至中间记忆更新"]
benchmarks: ["WebWalkerQA", "HotpotQA", "Amazon Reviews 2023 (MoviesTV/Books/Games)"]
---

# 论文速读：ReMem: Rethinking Perception and Memory in Long-Context Recommendation Agents

## 一句话总结
本文提出了**ReMem**，一个面向长上下文推荐智能体的新框架，通过OCR多模态感知替代 brittle 的 HTML 解析，并结合块级时序动态记忆机制实现线性复杂度的长序列推理，同时设计了 Multi-Memory GRPO 训练策略优化记忆更新过程，在三个推荐任务上平均提升 5.16%。

## 研究问题与动机
- **核心问题**：现有推荐智能体（RecAgents）在感知物品信息和处理长上下文偏好建模两个方面存在根本性不足。
- **现有方法不足一**：主流 RecAgent 依赖解析原始 HTML 获取商品信息，面临跨平台布局异质性、广告噪声干扰、多模态信息丢失等问题，感知能力脆弱且不具备平台通用性。
- **现有方法不足二**：直接拼接完整交互历史会导致上下文超出 LLM 窗口上限，而外部向量检索或静态记忆模块存在分布外泛化差、破坏自回归生成流程、兼容性差等缺陷。
- **设计动机**：借鉴人类行为直觉——通过视觉扫描网页（而非阅读代码）获取信息，并通过选择性抽象关键概念、动态更新认知状态来处理长期记忆，实现有界记忆容量下的长程推理。

## 核心贡献（创新点）
1. **提出首个结合 OCR 多模态感知与强化学习增强动态记忆的长上下文推荐智能体框架 ReMem**；与已有工作的本质区别在于首次将视觉感知与时间演化记忆统一纳入推荐 Agent 架构，而非仅依赖文本解析或静态记忆。
2. **设计块级时序演化记忆（TEM）机制**，支持任意长度交互历史下的线性推理复杂度与有界上下文窗口；与外部记忆模块的本质区别是不需要额外的检索系统或架构修改，直接在标准自回归生成过程中完成记忆更新。
3. **提出 Multi-Memory GRPO 训练算法**，将最终答案优势传播至所有中间记忆更新步骤；与标准 GRPO 的本质区别在于解决了多轮独立对话场景下的信用分配问题，避免跨记忆的人工关联。
4. **构建基于 Amazon Reviews 的三个 Agentic 推荐数据集（MoviesTV、Books、Games）**，覆盖搜索、排序、判断三类任务；填补了现有基准在长上下文多模态感知场景下的评测空白。
5. **实验验证 ReMem 在三个数据集和三类任务上均优于 SOTA 基线，平均相对提升 5.16%**；显著性检验 p < 0.01，证明性能提升稳定可靠。

## 方法详解
- **整体架构**：ReMem 包含三大模块——OCR 多模态感知模块、块级时序演化记忆（TEM）机制、Multi-Memory GRPO 训练策略，整体遵循"感知→记忆更新→最终决策"的流水线。

- **多模态感知模块**：
  - 不使用原始 HTML，而是对商品页面截图调用 DeepSeek-OCR-2 进行解析。
  - 公式 (2)：$\boldsymbol{v}_j = \mathrm{OCR}(\mathrm{Request}(\mathrm{url}(\mathrm{texts}, \mathrm{images}, \mathrm{charts})))$
  - OCR 输出包含结构化文本与自然语言描述的卡片模板，有效过滤广告等噪声，保留设计师刻意突出的关键信息。

- **块级时序演化记忆（TEM）**：
  - 将用户交互历史 $\mathcal{H} = \{v_1, ..., v_J\}$ 划分为 $T$ 个连续块 $C = \{\mathbf{c}^1, ..., \mathbf{c}^T\}$，每个块包含 $W$ 条交互记录及对应商品描述。
  - 记忆更新公式 (3)：$\mathbf{m}^t \sim p_\theta(\mathbf{m}^t | \mathbf{m}^{t-1}, \mathbf{c}^t, Q)$，其中 $|\mathbf{m}^t| = L$ 为固定长度，初始记忆 $\mathbf{m}^0 = \varnothing$。
  - 最终推荐生成公式 (4)：$y \sim p_\theta(y | \hat{\mathbf{c}}, \mathbf{m}^T, Q)$，其中 $\hat{\mathbf{c}} = \mathrm{perception}(\mathcal{Z})$ 为候选物品的 OCR 感知结果。
  - 计算复杂度分析：每步激活上下文长度为 $O(C + L)$，总复杂度为 $O(T(C+L)^2) = O(N)$，线性于历史长度，避免了 $O(N^2)$ 的二次注意力开销。
  - 记忆以自然语言 token 形式存储，具备可解释性、可调试性、可编辑性。

- **Multi-Memory GRPO 训练策略**：
  - 对每个查询采样 $G$ 组推理实例，每组包含 $T$ 个中间记忆 $\mathcal{M}_g = \{\mathbf{m}_g^1, ..., \mathbf{m}_g^T\}$ 和最终答案 $y_g$。
  - 奖励计算：$r_g = r(y_g, y^+)$，基于 HR@K、NDCG@K 或 Accuracy。
  - 组内优势归一化公式 (11)：$A_g = \frac{r_g - \mu_r}{\sigma_r + \epsilon}$。
  - 关键修改：将 $A_g$ 传播至所有中间记忆更新步骤，而非仅优化最终答案。
  - 记忆级策略比率公式 (12)：$\rho_{g,t,l}^m(\theta) = \frac{\pi_\theta(m_{g,t,l} | Q, m_{g,t,<l}, \mathbf{c}^t)}{\pi_{\theta_{\mathrm{old}}}(m_{g,t,l} | Q, c_{g,t,<l}, \mathbf{c}^t)}$。
  - 最终答案与记忆的联合优化目标公式 (16)：$\mathcal{I}_{\mathrm{Mem-GRPO}}(\theta) = \mathcal{I}_{\mathrm{Ans}}(\theta) + \lambda \mathcal{I}_{\mathrm{Mem}}(\theta)$，其中 $\lambda$ 控制记忆优化权重。
  - KL 正则化公式 (17)(18)：$\max_\theta \mathcal{I}(\theta) = \mathcal{I}_{\mathrm{Mem-GRPO}}(\theta) - \beta \mathcal{D}_{\mathrm{KL}}^m$，$\beta = 0.04$。

## 实验与结果
- **数据集**：基于 Amazon Reviews 2023 构建的三个数据集——Games（94,762 用户、25,612 商品）、Books（7,377 用户、120,925 商品）、MovieTV（5,649 用户、28,987 商品），统计详见 Table 2。
- **评估任务**：Searching（检索）、Ranking（排序）、Judging（判断）。
- **评估指标**：HR@K、NDCG@K、Accuracy（LLM-as-Judge，GPT-5-mini）。
- **基线模型**：四类——DeepRec（SASRec、BERT4Rec）、LLMRec（P5、TokenRec）、RecAgent（ToolRec、iAgent）、LongAgent（QwenLong-L1、Mem0）。
- **主干模型**：Qwen3.5-9B，推理温度 0.2，训练温度 1.5，记忆长度 $L_m = 5,000$ tokens，块大小 $W = 3$，最大块长度 100,000 tokens。
- **主要结果**（Table 3）：
  - ReMem 在所有 Searching 和 Judging 任务上取得最佳结果，Ranking 任务上获得 2/3 最佳。
  - **平均相对提升 5.16%**，优于最强基线。
  - Games 数据集 Searching HR@1：ReMem 0.4292 vs iAgent 0.3824（+7.43%）。
  - Books 数据集 Searching HR@1：ReMem 0.3588 vs iAgent 0.2925（+14.55%）。
  - 性能提升具有统计显著性（p < 0.01）。
- **长上下文分析**（Figure 4、Table 6）：随上下文长度增加（Short/Medium/Long），所有模型性能下降，但 ReMem 在所有长度区间保持最优，Long 区间相对 QwenL1 提升达 18.6%~34.1%。
- **消融实验**（Table 4）：移除 OCR（-OCR）、移除 GRPO（-GRPO）、移除 TEM（-TEM）均导致性能下降，其中移除 TEM 下降最大，验证了各模块必要性。
- **推理延迟**（Table 7）：单样本推理时间 Books 224.6s、MovieTV 118.5s、Games 98.3s（vs 基线 Qwen3.5-9B 约 4-10s），作者认为在主动推荐场景下可接受。

## 相关工作脉络
1. **WebWalkerQA / HotpotQA 长上下文基准**：本文 Pilot Study 基于这两个通用 benchmarks 验证感知策略与记忆机制的有效性，为推荐场景的方法设计提供实证依据。
2. **DeepSeek-OCR-2**：作为 OCR 感知工具的核心组件，替代传统 HTML 解析方案，是本文多模态感知的技术基础。
3. **GRPO（DeepSeekMath）**：标准 Group Relative Policy Optimization 算法，本文在其基础上扩展为 Multi-Memory GRPO，解决多轮独立对话的信用分配问题。
4. **DAPO**：本文 Multi-Memory GRPO 的设计灵感来源之一，借鉴其对中间状态的梯度传播思想。
5. **MemAgent / MEM1**：通用长文档推理的记忆 Agent，本文强调与它们的本质差异——推荐场景需分离中间记忆更新与最终决策，且需同时解决上游感知瓶颈。
6. **InstructRec / iAgent / ToolRec**：现有推荐智能体代表性工作，本文在相同数据集和任务设置下进行对比，证明 OCR 感知 + 动态记忆的优越性。
7. **SASRec / BERT4Rec / P5 / TokenRec**：传统序列推荐与 LLM 推荐基线，本文通过对比验证 Agent 范式在开放环境感知与交互推理方面的优势。

## 局限性与未来方向
- **推理延迟较高**：iterative memory construction 和 multi-step reasoning 带来额外开销，单样本推理时间比基线 LLM 高 10-20 倍，在严格实时场景下受限。
- **OCR 模型泛化性待验证**：当前仅使用 DeepSeek-OCR-2，对不同 OCR 模型、跨平台（非 Amazon 风格）的适应性未充分评估。
- **记忆长度与块大小的超参敏感性**：需根据数据集交互历史长度手动调整，缺乏自适应机制。
- **未来方向一**：扩展用户-智能体交互环境，纳入更多细粒度行为（展开描述、搜索补充信息、完成结账等）。
- **未来方向二**：评估不同 OCR 模型（如 PaddleOCR）的性能差异与失败案例。
- **未来方向三**：测试更大规模语言模型（如 GPT-5.6 Sol）和更多记忆 Agent 基线（如 MemAgent）。

## 研究启发与可借鉴点
1. **"人类直觉驱动设计"**：从人类浏览网页和记忆行为的直觉出发，而非直接套用技术解决方案，这种设计理念值得在 Agent 系统设计中推广——先问"人怎么做"，再问"机器如何实现"。
2. **Token-space 显式记忆 vs Feature-space 压缩**：将记忆表示为可读的自然语言 token 而非隐式向量，提升了可解释性和可控性，适合需要人工审核或干预的推荐场景。
3. **优势传播到中间步骤的训练技巧**：Multi-Memory GRPO 将最终奖励传播到所有中间记忆更新的思路，可迁移到任何多阶段推理任务（如多跳问答、规划执行）的强化学习训练中。
4. **线性复杂度长上下文处理**：块级处理 + 固定长度记忆的机制，为在标准 LLM 上实现长上下文推理提供了一种低工程成本的替代方案，无需修改架构或使用外部检索。
5. **数据集构建方法论**：基于真实电商平台网页截图、注入广告噪声、模拟用户 Persona 的数据集构建方式，为 RecAgent 评测提供了可复用的范式。

## 关键术语表
**RecAgent（Recommendation Agent）**：基于 LLM 的自主推荐智能体，能主动感知外部平台、推理用户偏好并执行推荐决策，区别于传统的被动推荐系统。
**OCR（Optical Character Recognition）**：光学字符识别技术，本文指利用 DeepSeek-OCR-2 从商品页面截图中提取结构化文本与多模态信息的能力。
**TEM（Time-Evolving Memory）**：时序演化记忆机制，通过块级顺序处理将任意长交互历史压缩为固定长度、持续更新的语言化记忆。
**Multi-Memory GRPO**：本文提出的强化学习训练变体，将最终推荐结果的优势值传播至所有中间记忆更新步骤，实现端到端的记忆优化。
**InstructRec**：指令遵循推荐范式，本文用于构建用户 Persona 和任务指令的数据集基础。
**Amazon Reviews 2023**：本文构建推荐数据集的源数据，包含商品文本描述、图片 URL 和用户交互历史。
**Needle-in-a-Haystack**：长上下文基准测试任务，本文用于验证不同记忆机制在长文本中的信息定位与提取能力。
**LLM-as-Judge**：使用大语言模型作为评判器评估推荐结果正确性的自动化评测方法。

## 可复现要素
- **数据集**：基于 Amazon Reviews 2023 构建，作者声明代码已开源（https://github.com/Quhaoh233/ReMem），数据集构建细节见 Appendix C.2。
- **代码**：已开源，使用 Python 3.12 和 Unsloth（Version 2026.5.2）实现。
- **硬件**：四卡 NVIDIA H20（96 GB）GPU。
- **关键超参**：主干模型 Qwen3.5-9B，推理温度 0.2，训练温度 1.5，记忆长度 $L_m = 5,000$ tokens，块大小 $W = 3$，最大块长度 100,000 tokens，rollout batch size = 64，group size = 8，KL 系数 $\beta = 0.04$，clip 系数 $\delta = 0.2$，记忆权重 $\lambda$ 预热 2 个 epoch 设为 0，后续设为 0.7。
- **基线实现**：SASRec、BERT4Rec、P5、TokenRec、ToolRec、iAgent、QwenLong-L1、Mem0 等，部分开源模型直接使用官方实现，部分自行复现。
