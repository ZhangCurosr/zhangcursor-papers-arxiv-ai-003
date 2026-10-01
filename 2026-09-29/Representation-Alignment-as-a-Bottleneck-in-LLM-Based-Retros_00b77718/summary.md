---
title: "Representation-Alignment-as-a-Bottleneck-in-LLM-Based-Retros"
source: https://arxiv.org/pdf/2609.35571v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 22:08:45"
field: "化学信息学与符号规划"
keywords: ["retrosynthesis planning", "representation alignment", "PDDL", "large language models", "intermediate representations", "symbolic planning", "RetroPlan-Bench"]
innovations: ["将LLM逆合成规划失败重新定义为表示对齐问题而非推理能力不足", "提出分子映射-反应映射-PDDL生成三阶段分期分析框架", "构建RetroPlan-Bench多尺度可执行符号规划基准"]
benchmarks: ["RetroPlan-Bench"]
---

# 论文速读：Representation-Alignment-as-a-Bottleneck-in-LLM-Based-Retrosynthesis-Planning

## 一句话总结
论文发现 LLM 在端到端逆合成规划任务中失败的根本原因不是模型推理能力不足，而是**表示对齐（representation alignment）瓶颈**：尽管强模型在各独立子任务上表现接近完美，但在从 SMILES 直接生成 PDDL 的端到端设置下所有模型成功率均为 0%；引入结构化中间表示后性能恢复。

## 研究问题与动机
- **核心问题**：LLM 在逆合成规划中的失败是源于推理能力不足，还是源于非结构化化学表示与符号规划形式之间的表示失配？
- **已有方法的不足**：现有化学 LLM 研究主要关注反应预测或路径生成本身，较少将输出形式化为可由规划器直接执行的可执行符号表示（PDDL）；同时，端到端"SMILES-to-PDDL"方法要求模型同时完成化学结构解释、反应级结构化和符号化编码，单步转换负担过重。
- **关键悖论**：各子任务（分子映射、反应映射、PDDL 生成）上强模型表现近乎完美，但整合为端到端流程后全部失效，说明失败不在推理本身而在表示跨空间组合。
- **缺乏分析工具**：现有关于 LLM 规划的研究多聚焦自然语言或代码形式，缺乏对"非结构化输入→可执行符号表示"转换失败的系统性分解分析框架。

## 核心贡献（创新点）
1. **将 LLM 逆合成规划失败重新定义为表示失配问题而非推理能力不足**：通过对比端到端与分阶段设置的实验结果，证明主瓶颈在于跨表示空间的对齐而非模型容量，与以往归因于推理缺陷的讨论形成本质区别。
2. **提出逆向合成规划的分期分析框架（stage-wise analytical framework）**：将整体任务分解为分子映射、反应映射、PDDL 生成三个可解释的中间变换阶段，逐层定位失败点，区别于以往仅关注最终规划成功率的做法。
3. **构建 RetroPlan-Bench 基准测试**：基于 READRetro 数据集的 368 条多步合成路径，通过符号接地、反应结构化、覆盖率过滤三步流水线构建支持多尺度（100–1000 反应）的系统性评测基准，填补了可执行符号规划评测的空白。
4. **验证中间表示的显著增益与非 oracle 链式设定的鲁棒性**：GPT-5.2 和 DeepSeek V3.2 在 oracle 中间表示下 solve rate 达 1.0，在非 oracle 链式设置下同样保持 Solve Rate=1.0、Path Accuracy=0.986，证明改进并非 oracle 幻觉。

## 方法详解
- **四阶段分层框架**：
  1. **分子映射（Molecule Mapping）**：给定分子符号 ID，模型需生成对应的精确 SMILES 字符串。评估指标为 ID 一致率与 SMILES 精确匹配率，衡量符号到化学结构的接地可靠性。
  2. **反应映射（Reaction Mapping）**：给定参与反应的分子 ID，构建结构化的反应物-产物对 (R, P) 表示。评估反应级精确匹配，检测缺失、重复或错配的组份。
  3. **PDDL 生成（PDDL Generation）**：将结构化反应信息转换为 PDDL domain + problem 文件。评估三维度：①句法有效性（syntax validity）；②结构完整性（actions, predicates, domain/problem 组件是否齐全）；③语义一致性（action 是否正确反映反应物-产物关系）。
  4. **规划执行（Planning Execution）**：使用外部经典规划器 Fast Downward 验证生成的 PDDL 是否能产生有效合成路径，评估 solve rate 与 path accuracy。
- **三种评测设置**：
  - **端到端（End-to-end）**：直接从 SMILES 输入生成 PDDL，单步完成全部转换。
  - **分解+oracle（Decomposed with oracle）**：每个阶段使用真实中间表示进行条件评估，隔离上游误差。
  - **非 oracle 链式（Non-oracle chained）**：前一阶段的模型输出作为下一阶段输入，测试端到端误差传播。
- **RetroPlan-Bench 构建流水线**：
  - **符号接地（Symbolic Grounding）**：将所有 SMILES 替换为唯一符号 ID，迫使模型关注关系/组合结构。
  - **反应结构化（Reaction Structuring）**：将合成路径分解为 (reactant, product) 对，赋予唯一反应 ID。
  - **覆盖率过滤（Coverage Filtering）**：仅保留可用预定义反应集完整表示的路径，确保失败源于模型能力而非数据缺失。

## 实验与结果
- **数据集**：RetroPlan-Bench，基于 READRetro（Kim et al., 2024）368 条多步合成路径构建，涵盖 100–1000 反应规模。
- **评测模型**：GPT-5.2、DeepSeek V3.2、Gemini 3.1、Gemini 2.5 Flash、Qwen3-30B-Thinking、Qwen2.5-14B-Instruct、ChemLLM。
- **关键数字**：
  - **端到端设置**：所有模型 Solve Rate = 0%，Path Accuracy = 0%（Table 9）。
  - **子任务表现**（Table 1，400 反应规模均值）：GPT-5.2 在 Molecule ID 精确匹配=1.0000，SMILES 精确匹配=0.9932，Reaction Mapping=1.0000，PDDL Domain=0.9303，PDDL Problem=1.0000；DeepSeek V3.2 相应为 0.9992/0.7870/1.0000/0.9300/1.0000。
  - **分阶段 oracle 设置**：GPT-5.2、DeepSeek V3.2、Gemini 3.1 达到 Solve Rate = 1.0，Path Accuracy = 0.9864（Table 8）。
  - **非 oracle 链式**：GPT-5.2 和 DeepSeek V3.2 保持 Solve Rate = 1.0，Path Accuracy = 0.986，与 oracle 设置一致。
  - **Gemini 2.5 Flash**：部分子任务表现强，但最终规划阶段失败（Solve Rate=0），说明单阶段准确性不保证可执行性。
  - **ChemLLM**：在所有阶段均失败（Solve Rate=0），主要原因为结构化输出不稳定（无效 JSON、非 PDDL 输出、截断），而非化学推理错误。
- **最强结果**：GPT-5.2 / DeepSeek V3.2 / Gemini 3.1 在分阶段设置下达 Solve Rate=1.0，相对端到端的 0% 为绝对提升。
- **缩放行为**：随着反应数增加，GPT-5.2 和 DeepSeek V3.2 的 action grounding 准确率逐渐下降（从 0.99→0.835@1000），表明结构形式得以维持但语义一致性退化；Qwen 模型在较小规模即快速退化。

## 相关工作脉络
1. **LLM for Planning**（Valmeekam et al., 2023a/b; Ahn et al., 2022; Liu et al., 2023）：现有工作多评估 LLM 从自然语言或代码生成动作序列的能力，本文与之区别在于系统性分析从非结构化化学输入到可执行符号表示（PDDL）的跨表示转换失败点。
2. **LLMs in Chemistry & Retrosynthesis**（Schwaller et al., 2019; Ucak et al., 2022; Han et al., 2024）：已有化学 LLM 工作聚焦反应预测或路径生成本身，较少将输出形式化为可由规划器执行的符号表示，本文将其重新定义为可执行符号规划问题并分析失败阶段。
3. **Intermediate Representations / Modular Reasoning**（Wei et al., 2022; Gao et al., 2023; Yao et al., 2023b）：CoT、Program-of-Thought 等工作已在自然语言推理和代码生成中验证中间表示的有效性，本文将其理念引入跨异构表示空间的规划场景并定量验证表示对齐的核心作用。
4. **PlanBench / BioPlanner**（Valmeekam et al., 2023a; O'Donoghue et al., 2023）：通用规划基准和生物协议规划基准，本文的 RetroPlan-Bench 补充了化学逆合成领域专用的可执行符号规划评测。
5. **LLM+P / PDDLStream**（Liu et al., 2023; Garrett et al., 2020）：将 LLM 与外部规划器结合的工作，本文与其不同，核心贡献不在系统架构而是在揭示失败根源为表示失配而非推理能力。

## 局限性与未来方向
- 研究聚焦逆合成规划单一领域，现象是否泛化至其他规划问题仍需验证。
- 非 oracle 链式评估仅在逆合成基准上进行，未建立跨领域鲁棒性证明。
- 论文未提出解决表示对齐问题的具体模型架构或训练策略，留待未来工作探索。

## 研究启发与可借鉴点
1. **"子任务强但端到端全败"的诊断范式**：本文的分阶段诊断思路（oracle 分解 + 非 oracle 链式 + 端到端三设对比）可直接迁移到其他跨模态/跨表示的生成任务中，用于区分"能力缺失"与"表示对齐瓶颈"。
2. **中间符号表示的工程价值**：在 LLM-based 规划系统中，显式引入结构化中间层（如本论文的分子 ID→反应对→PDDL 三级抽象）比纯端到端方案更可靠，设计pipeline时应优先考虑中间表示的稳定性。
3. **规划可执行性≠子任务准确性**：Gemini 2.5 Flash 在部分子任务高分但规划失败的结果表明，下游可执行性评估（solve rate）是比子任务 accuracy 更严格的评测指标，应在评测体系中纳入。
4. **与团队方向结合机会**：若团队涉及化学信息学、符号规划或跨表示生成任务，可将本论文的分期框架和 RetroPlan-Bench 评测范式推广到其他领域（如材料发现路径规划、生化通路设计）。

## 关键术语表
- **SMILES（Simplified Molecular Input Line Entry System）**：一种用 ASCII 字符串表示分子结构的线性编码方式，是本文的非结构化化学输入表示。
- **PDDL（Planning Domain Definition Language）**：经典规划问题的形式化描述语言，包含 domain（动作定义）和 problem（初始/目标状态）两部分，本文的可执行规划输出格式。
- **Representation Alignment（表示对齐）**：指模型在不同表示空间之间保持一致性和语义对应关系的能力，本文认为是逆合成规划失败的主因。
- **RetroPlan-Bench**：本文提出的基于 READRetro 的可执行符号逆合成规划基准，支持多尺度评测和分阶段分析。
- **Solve Rate**：规划器成功生成有效合成路径的实例比例，是本文衡量最终规划可执行性的核心指标。
- **Oracle Intermediate Representation**：在分阶段评测中直接使用真实中间表示（而非模型生成）的设置，用于测量下游阶段的上界性能。
- **Action Grounding**：衡量生成的 PDDL 动作是否正确包含对应反应的反应物和产物标识符，反映语义一致性。

## 可复现要素
- **数据集**：RetroPlan-Bench，基于 READRetro（Kim et al., 2024）构建，论文未明确声明是否开源，需查看 arXiv 附注或代码仓库。
- **代码/权重**：论文未明确提及代码开源情况，"论文未提及"。
- **关键超参**：论文未详细报告超参数设置，仅提及使用 Fast Downward 作为外部规划器；模型侧使用各模型官方 API，具体 prompt 见附录 Figures 7–10。
