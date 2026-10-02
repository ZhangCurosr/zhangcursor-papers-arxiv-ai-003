---
title: "RETRIEVAL-AUGMENTED-SKILL-OPTIMIZATION-VIA-CROSS-HARNESS-ADA"
source: https://arxiv.org/pdf/2609.38024v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 08:18:42"
field: "Agent 技能优化与知识迁移"
keywords: ["Agent Skills", "Skill Optimization", "Retrieval-Augmented Generation", "Cross-Harness Adaptation", "LLM Agents"]
innovations: ["提出RASO框架，统一利用外部技能库进行技能初始化与迭代更新", "引入跨Harness适配机制，解决检索技能与目标环境的域/接口不匹配问题", "证明无需rollout的知识grounded初始化与基于反馈的检索更新具有互补增益"]
benchmarks: ["OfficeQA", "SpreadsheetBench", "ALFWorld", "WebShop"]
---

# 论文速读：RETRIEVAL-AUGMENTED-SKILL-OPTIMIZATION-VIA-CROSS-HARNESS-ADA

## 一句话总结
论文提出 RASO（检索增强技能优化）框架，通过检索外部技能库并结合跨 Harness 适配，为 Agent 任务的**技能初始化（RASI）**和**迭代更新（RASU）**提供外部知识先验，解决了现有方法完全依赖昂贵 Agent  rollout 的局限。在 OfficeQA、SpreadsheetBench、ALFWorld 和 WebShop 四个基准上，RASO 均显著优于 TextGrad、GEPA、SkillOpt、WikiSkill 等基线方法。

## 研究问题与动机
1.  **技能价值被忽视**：尽管 GitHub 等平台已积累数百万条公开 Agent 技能（程序性知识），但现有技能优化方法（如 TextGrad、SkillOpt 等）几乎完全依赖目标 Agent 在特定 Harness 下的自身 rollout 经验进行迭代优化，未利用这些外部积累的通用知识。
2.  **直接复用的不匹配问题**：即使尝试检索外部技能，其来源任务的领域（domain）和执行环境（harness，即工具集、文件访问、评分接口）与目标任务常不匹配，直接复用会引入无关或错误的操作指令，效果甚至低于无技能基线。
3.  **初始化成本高**：大多数优化方法假设初始技能已由模型生成或简单获取，未将“高质量、知识 grounding 的初始技能构建”本身纳入优化流程进行专门设计。

## 核心贡献（创新点）
1.  **提出 RASO 框架**：将外部技能库作为先验知识，统一应用于技能的**初始化（RASI）**和**更新（RASU）**两个阶段，而非仅依赖 Agent 自身 rollouts。
2.  **引入跨 Harness 适配（Cross-Harness Adaptation）**：一种共享机制，将检索到的、来自不同领域/Harness 的技能片段，抽象并重新表述为目标任务和目标 Harness 支持的对象、命令和单元，解决知识迁移中的域与接口不匹配问题。
3.  **解耦初始化与更新并验证互补性**：实验证明，无需任何 rollout 的 RASI 即可生成强初始技能；结合执行反馈进行检索更新的 RASU 能进一步精炼技能，两者结合效果最佳。

## 方法详解
RASO 包含两个核心阶段和一个共享机制：
*   **共享机制：技能检索与跨 Harness 适配**
    *   **细粒度检索**：将技能文档按标题分割为段落级别，使用 BM25 根据查询检索最相关的 Top-K 个段落，提高信噪比。
    *   **跨 Harness 适配**：适配代理（$\mathcal{A}_{adaptation}$）接收需求、任务描述（T）、Harness 描述（H）及检索到的段落，生成简洁、可操作的“准则”（lesson）。适配遵循三原则：移除源领域特定名词、忽略目标 Harness 中无对应物的程序、仅在目标 Harness 描述有佐证时才保留关于工具/参数的具体声明。
*   **阶段一：检索增强技能初始化（RASI）**
    1.  **需求与查询生成**：基于 T 和 H，生成一组“需求-查询”对。
    2.  **检索与适配**：对每个查询，检索 Top-K 段落并进行跨 Harness 适配，得到准则集合。
    3.  **技能合成**：技能初始化代理将每个准则整合到执行步骤的对应位置，生成初始技能 $s_0$，**全程无需任何 rollout**。
*   **阶段二：检索增强技能更新（RASU）**
    1.  **文本梯度与查询生成**：用当前技能 $s_t$ 在训练集上执行 rollout，分析失败轨迹，为每种失败模式生成文本梯度 $\delta_i$ 和对应的检索查询 $q_i$。
    2.  **检索与适配**：对每个查询检索并适配，得到更新准则集合。
    3.  **技能更新**：技能更新代理结合当前技能、文本梯度和准则，生成候选技能 $s_{t+1}$。仅在验证集上性能优于 $s_t$ 时接受更新，重复固定迭代次数。

## 实验与结果
*   **数据集**：OfficeQA, SpreadsheetBench, ALFWorld, WebShop。
*   **模型**：GPT-5.6-Luna, Qwen-3.5-9B。
*   **基线**：初始化对比（No Skill, SkillRouter, RFSI）；更新对比（TextGrad, GEPA, SkillOpt, WikiSkill）。
*   **核心结果（GPT-5.6-Luna 为例）**：
    *   **初始化**：RASI 在零 rollout 下显著优于 RFSI，在 OfficeQA (+5.63), SpreadsheetBench (+4.77), ALFWorld (+3.24), WebShop (+1.17) 上取得提升。SkillRouter 直接复用未适配的技能，在 OfficeQA 和 ALFWorld 上甚至低于 No Skill。
    *   **端到端优化**：RASO 在所有基准上均击败最强基线。在 OfficeQA 上比 SkillOpt 高 +3.49 (49.03 vs 45.54)，在 SpreadsheetBench 上比 SkillOpt 高 +6.31 (63.33 vs 57.02)，优势巨大。
    *   **消融实验**：RASU 单独贡献 +6.86 (OfficeQA) / +9.40 (Spreadsheet)；RASI 单独贡献 +5.23 / +6.78；两者结合产生最大增益（+8.33 / +11.66 vs RFSI+RFSU 基线），证明互补性。
    *   **成本**：RASO 在达到更低或可比成本的同时实现最优性能（例如在 WebShop 上成本仅为 TextGrad 的 1/4）。

## 相关工作脉络
1.  **执行经验驱动优化**：TextGrad, GEPA, SkillOpt, WikiSkill 等方法通过 Agent rollout 的反馈（文本梯度、反思、编辑）优化技能。RASO 的定位差异在于，在优化过程中**补充**了来自外部技能库的程序性知识，而非仅依赖 Agent 内部经验。
2.  **外部知识检索与复用**：SkillRouter 等工作侧重于从大型技能库中检索和路由相关技能。RASO 的定位差异在于引入了**跨 Harness 适配**机制，使检索到的异构知识能够安全迁移到目标任务环境，并且将这种检索-适配机制用于**初始化和迭代更新全流程**。
3.  **技能表示与编译**：Agent KB, Anything2Skill 等工作将外部资源（如文档、网页）编译成可复用技能。RASO 的定位差异在于直接利用**已结构化的、针对特定 harness 编写的 Agent 技能库**，并通过适配解决其 harness 不匹配问题，而非从零编译。

## 局限性与未来方向
1.  **依赖外部技能库质量与覆盖度**：性能提升程度受限于 GitSkills 等外部库中与目标任务相关的、且经过适配的知识可用量。知识库过小或领域极偏可能导致收益有限。
2.  **适配过程的 LLM 依赖性**：跨 Harness 适配严重依赖适配器 LLM 的理解与改写能力，若 LLM 在适配过程中引入错误，可能误导技能更新。
3.  **评估范围**：实验主要在四个特定基准上进行，方法的泛化能力在更广泛、更动态的真实世界 harness 环境中有待验证。
4.  **未来方向**：探索更高效的细粒度知识检索策略；研究自适应的检索数量（K 值）；将框架扩展到具有更复杂、动态交互环境的多智能体系统。

## 研究启发与可借鉴点
1.  **检索增强优化的双阶段设计**：将“基于描述的无 rollout 初始化”与“基于执行反馈的迭代更新”解耦并分别增强，是一种高效利用外部知识的通用范式，可迁移至其他需要构建或优化上下文/策略的 LLM 应用。
2.  **跨域/跨接口知识适配机制**：提出的“适配三原则”（移除特定名词、忽略无对应物程序、保留有佐证的特定声明）为其他需要迁移外部知识到新执行环境的研究提供了可直接参考的方法论。
3.  **实验设计的对照严谨性**：论文严格控制了任务描述（T）和 Harness 描述（H）的一致性，并将 RFSI 作为所有更新方法的统一起点，确保了更新算法性能差异的可比性。这种设计值得在对比学习或优化算法论文中借鉴。
4.  **效率与性能的平衡分析**：不仅报告了性能提升，还详细分析了 API 成本和 rollout 次数，证明 RASO 在更少或同等开销下获得更好结果，这种全面评估增强了结论的说服力。

## 关键术语表
*   **Agent Skill**：一种可重用的、以自然语言书写的程序性知识文档，用于指导 Agent 在特定执行环境（Harness）下完成任务。
*   **Execution Harness**：定义了 Agent 可用的工具集、文件访问权限、观察接口和评分规则的底层执行环境。
*   **Cross-Harness Adaptation**：RASO 的核心机制，将检索到的、来自不同领域和 Harness 的技能片段，抽象并重新表述为目标任务和 Harness 所支持的操作语言和对象。
*   **RASI**：Retrieval-Augmented Skill Initialization，RASO 的第一阶段，利用外部知识生成高质量的初始技能，无需任何 Agent rollout。
*   **RASU**：Retrieval-Augmented Skill Update，RASO 的第二阶段，根据执行反馈识别失败模式，检索外部知识进行迭代技能精炼。
*   **Textual Gradient**：在 RASU 中，由 LLM 基于失败轨迹生成的、描述技能应如何改进的文本反馈。
*   **BM25 Retrieval**：论文采用的段落级检索方法，用于从大型技能语料库中查找与查询最相关的知识片段。

## 可复现要素
*   **数据集**：OfficeQA, SpreadsheetBench, ALFWorld, WebShop 均为公开基准测试。外部技能库使用 **GitSkills** 数据集。
*   **代码/权重**：论文未明确声明代码和预训练模型权重的开源情况，但提及使用 GPT-5.6-Luna 和 Qwen-3.5-9B 作为基础模型。
*   **关键超参**：每查询检索段落数 **K=5**；训练轮次 **2 epochs**；每轮训练批次大小 **40 tasks**；chunk 大小 **8**；Qwen-3.5-9B rollout 温度 0.0，优化器温度 0.7；GPT-5.6-Luna 推理努力设为 low。
