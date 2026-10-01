---
title: "Representation-Alignment-as-a-Bottleneck-in-LLM-Based-Retros"
source: https://arxiv.org/pdf/2609.35571v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 10:08:59"
field: "LLM for Scientific Planning"
keywords: ["Representation Alignment", "Retrosynthesis Planning", "PDDL", "Large Language Models", "Symbolic Planning", "Intermediate Representation", "RetroPlan-Bench"]
innovations: ["重新定义LLM在化学符号规划中的失败为表示对齐问题而非推理能力不足", "提出四阶段分阶段分析框架（Molecule Mapping→Reaction Mapping→PDDL Generation→Planning Execution）并构建RetroPlan-Bench基准", "实证证明通过引入中间符号表示可将端到端0%成功率恢复至100%规划成功率"]
benchmarks: ["RetroPlan-Bench", "READRetro"]
---

# 论文速读：Representation-Alignment-as-a-Bottleneck-in-LLM-Based-Retros

## 一句话总结
论文发现大型语言模型（LLM）在基于化学 retrosynthesis 的符号规划任务中并非缺乏推理能力，而是无法将非结构化化学表示（SMILES）直接转换为可执行符号规划语言（PDDL）——核心瓶颈在于**跨表示空间的对齐（representation alignment）**。通过将任务分解为分阶段子任务并引入中间符号表示，强模型可从端到端 0% 成功率恢复至 100% 规划成功率。

## 研究问题与动机
1. **核心问题**：LLM 在 retrosynthesis 符号规划中表现失败的真正原因是什么？是模型推理能力不足，还是异构表示之间的转换困难？
2. **现有方法的盲区**：已有工作大多直接尝试 SMILES-to-PDDL 的端到端生成，或将任务简单视为反应预测/路径生成，未能系统分析从非结构化化学输入到可执行符号表示的转换失败点。
3. **关键悖论**：各分任务（分子映射、反应映射、PDDL 生成）上强模型表现接近完美，但端到端集成时所有模型完全失败（0% 成功率），说明失败源于表示间组合而非单步操作能力缺失。
4. **研究动机**：需要一种能精确定位失败位置的分析框架，以区分"推理失败"与"跨表示组成失败"。

## 核心贡献（创新点）
1. **重新定义失败归因**：将 LLM-based retrosynthesis 规划的失败重新定性为"表示对齐（representation misalignment）"问题而非推理能力不足，解释了高分项子任务表现与零端到端成功率之间的鸿沟。
2. **提出分阶段分析框架**：将 retrosynthesis 规划分解为分子映射（Molecule Mapping）→ 反应映射（Reaction Mapping）→ PDDL 生成（PDDL Generation）→ 规划执行（Planning Execution）四个阶段，首次系统定位了 LLM 在哪一表示层级失败。
3. **构建 RetroPlan-Bench 基准**：基于 READRetro 重构了 368 条多步合成路径，设计了覆盖分子级→反应级→规划级三层抽象的可执行符号规划基准，支持多尺度（100–1000 个反应）评估。
4. **实证证明中间表示的核心作用**：通过 oracle 和 non-oracle 链式实验表明，提供结构化中间表示后，GPT-5.2 / DeepSeek V3.2 / Gemini 3.1 的规划成功率从 0% 恢复到 100%，且非 oracle 链式设置下仍保持 Solve Rate=1.0、Path Accuracy=0.986。

## 方法详解
1. **四阶段分解框架**：
   - **Molecule Mapping**：将符号分子标识符映射为对应 SMILES 字符串，评估标识符精确匹配率和 SMILES 精确匹配率。
   - **Reaction Mapping**：将分子标识符组合同化为结构化反应表示 (R, P)（反应物/产物集合），评估反应级别精确匹配。
   - **PDDL Generation**：将结构化反应信息形式化为 PDDL domain + problem 文件，从三方面评估：句法有效性（syntax validity）、结构完整性（structural completeness）、语义一致性（semantic consistency）。
   - **Planning Execution**：使用外部经典规划器 Fast Downward 执行生成 PDDL，以 Solve Rate 和 Path Accuracy 衡量可执行性。

2. **RetroPlan-Bench 构建流程**：
   - **Symbolic Grounding**：将所有 SMILES 替换为唯一符号标识符，迫使模型关注符号关系而非化学结构复杂度。
   - **Reaction Structuring**：将每条合成路径分解为 (reactant, product) 对，并为每反应分配唯一 ID。
   - **Coverage Filtering**：仅保留能在预定义反应集内完整表示的路径，排除数据缺失导致的虚假失败。
   - **多尺度规划环境**：按唯一反应数构建 100/200/300/400 反应的问题集，并额外扩充至最大 1000 个 action 的大规模域。

3. **三种评估协议**：
   - **End-to-end（端到端）**：直接从 SMILES 生成 PDDL，单步完成所有转换。
   - **Decomposed with oracle（分阶段+oracle 中间表示）**：各阶段使用人工提供的正确中间表示进行条件评估。
   - **Non-oracle chained（非 oracle 链式）**：前一阶段的模型输出作为下一阶段输入，验证错误传播影响。

4. **关键发现公式**：端到端 Solve Rate = 0（所有模型），而 staged oracle Solve Rate = 1.0（GPT-5.2/DeepSeek V3.2/Gemini 3.1）。

## 实验与结果
- **数据集**：RetroPlan-Bench，源自 READRetro（Kim et al., 2024）的 368 条多步天然产物生物合成路径，覆盖 100–1000 个反应的多尺度评估。
- **评估模型**：GPT-5.2、DeepSeek V3.2、Gemini 3.1、Gemini 2.5 Flash、Qwen3-30B-Thinking、Qwen2.5-14B-Instruct、ChemLLM。
- **主要结果**：
  - **分任务表现**：GPT-5.2 在 Molecule ID 精确匹配达 1.0000、SMILES 精确匹配 0.9932、Reaction Mapping 1.0000、PDDL Domain 0.9303、PDDL Problem 1.0000；DeepSeek V3.2 类似。Gemini 3.1 的 Molecule ID 为 0.6000、SMILES 0.5964（明显低于 GPT/DeepSeek）。
  - **端到端失败**：所有模型在直接 SMILES-to-PDDL 设置下 Solve Rate = 0，Path Accuracy = 0（Table 9）。
  - **Staged oracle 成功**：GPT-5.2、DeepSeek V3.2、Gemini 3.1 在 400 题测试集上 Solve Rate = 1.0，Path Accuracy = 0.986（Table 8）。
  - **Non-oracle 链式鲁棒性**：GPT-5.2 和 DeepSeek V3.2 在非 oracle 链式设置下仍保持 Solve Rate = 1.0，Path Accuracy = 0.986，说明错误传播可控。
  - **缩放行为**：随反应数增加，GPT-5.2/DeepSeek V3.2 的 Action Grounding 从 0.99（100 反应）逐渐降至 0.835（1000 反应），呈现语义漂移但结构保持。
  - **最强结果**：GPT-5.2 在 staged 设置下以 100% Solve Rate 和 98.6% Path Accuracy 达到最优，相比端到端 0% 实现绝对提升。
- **基线对比**：与直接用 LLM 做反应预测（Schwaller et al., 2019; Ucak et al., 2022）或 pathway generation（Zhong et al., 2023）的方法相比，本文独特定位在于以可执行符号规划（PDDL + Fast Downward）为目标，而非仅生成文本描述。

## 相关工作脉络
1. **LLM for Planning**（Valmeekam et al., 2023a,b; Ahn et al., 2022; Huang et al., 2022）：这些工作主要评估 LLM 从自然语言指令生成动作序列的能力，聚焦标准 benchmark（如 Planning Domain Definition Language 相关），未系统分析从非结构化输入到可执行符号表示的转换失败点——本文定位为填补这一空白。
2. **LLM+P 方法**（Liu et al., 2023）：将 LLM 与经典规划器结合，但通常依赖 hand-crafted domain 或 natural language 输入，未探索化学领域的 SMILES-to-PDDL 自动转换。
3. **LLMs in Chemistry**（Schwaller et al., 2019; Ucak et al., 2022; Han et al., 2024）：主要聚焦反应预测和合成路径生成本身，较少将输出形式化为可由规划器直接执行的符号表示（PDDL）。
4. **Intermediate Representations & Modular Reasoning**：Chain-of-thought（Wei et al., 2022）、Program-of-thought（Chen et al., 2023a）、ReAct（Yao et al., 2023b）等已在 NLP/code 领域证明中间表示的价值，但本文首次将其系统应用于跨异构表示空间（化学结构↔符号规划）的对齐问题。
5. **BioPlanner**（O'Donoghue et al., 2023）和 **LAP 格式**（Anhel et al., 2023）：关注生物学协议规划，但使用的是预设 protocol 格式，而非从非结构化分子表示到可执行符号的自动转换。
6. **ChemLLM**（Zhang et al., 2024）：化学领域专用 LLM，本文实验表明即使化学专业化也不能解决跨表示对齐问题——所有阶段均失败，说明通用结构能力比领域知识更重要。

## 局限性与未来方向
1. **领域泛化未验证**：研究聚焦于 retrosynthesis 规划，是否在其他规划问题（如机器人任务规划、生物协议规划）中存在同样的表示对齐瓶颈尚待验证。
2. **非 oracle 链式实验规模有限**：目前仅在同一 retrosynthesis 基准上评估了 GPT-5.2 和 DeepSeek V3.2 的链式表现，缺乏跨领域鲁棒性证据。
3. **小模型在 staged 设置下仍失败**：Gemini 2.5 Flash、Qwen-family、ChemLLM 即使在 oracle 中间表示下也无法生成可执行规划，说明表示对齐能力与模型规模/架构密切相关，机制尚不清晰。
4. **未来方向**：① 将表示对齐问题推广至其他规划领域；② 探索能更好地维持跨表示空间一致性的模型架构或训练策略；③ 研究中间表示的"最小必要粒度"，避免过度分解带来的效率损失。

## 研究启发与可借鉴点
1. **分阶段分析框架可迁移**：将端到端失败任务分解为多个子阶段、逐一注入 oracle/chained 中间表示以定位瓶颈的方法，可广泛适用于任何涉及异构表示转换的 LLM 应用（如 code generation → executable program、natural language → SQL/DSL 等）。
2. **RetroPlan-Bench 的构造思路值得借鉴**：通过 Symbolic Grounding + Reaction Structuring + Coverage Filtering 三步构建受控基准，能有效隔离模型能力与数据质量的影响——这种设计模式可复用于其他科学领域的规划 benchmark。
3. **表示对齐作为独立研究方向**：本文提出"representation alignment"概念，揭示了端到端失败的一个新归因维度，后续研究可将此概念操作化，设计专门评估跨表示对齐能力的评测集。
4. **非 oracle 链式评估的价值**：证明了即使上游存在错误，下游任务仍可能鲁棒地执行（GPT-5.2/DeepSeek 在链式下仍 100% 成功率），提示工程实践中分层 pipeline 设计具有容错优势。
5. **与小团队方向的结合机会**：若团队涉及化学信息学、符号规划或多模态表示学习，可延伸本研究，探索无需 oracle 中间表示的自动表示学习方案，或设计轻量级表示转换器替代 LLM 的各映射阶段。

## 关键术语表
- **SMILES**（Simplified Molecular Input Line Entry System）：一种用 ASCII 字符串线性表示分子结构的标准化化学语法。
- **PDDL**（Planning Domain Definition Language）：AI 规划领域中用于形式化描述动作、前置条件和效果的符号规划语言。
- **Representation Alignment（表示对齐）**：模型在不同异构表示空间（如非结构化字符串↔符号化结构）之间保持一致映射的能力。
- **RetroPlan-Bench**：本文提出的基于 READRetro 重构的符号 retrosynthesis 规划基准，支持多尺度（100–1000 反应）分阶段评估。
- **Molecule Mapping（分子映射）**：将符号分子标识符转换为对应 SMILES 字符串的子任务，是表示对齐的第一步。
- **Reaction Mapping（反应映射）**：将分子级信息组织为结构化反应物-产物集合 (R, P) 的子任务，连接分子级与规划级表示。
- **Oracle 中间表示**：人工提供的正确中间输出，用于条件评估某阶段的理论上限性能。
- **Fast Downward**：一种高效的经典 AI 规划器，用于验证 LLM 生成的 PDDL 是否可产生合法合成路径。

## 可复现要素
- **数据集**：RetroPlan-Bench 基于 READRetro（Kim et al., 2024）的 368 条多步合成路径构建；论文提供了完整的 prompt 示例（Figures 7–10）和评估协议。
- **代码/权重**：论文未提及开源代码或模型权重；使用了商业 API 模型（GPT-5.2、Gemini 3.1 等）和开源模型（DeepSeek V3.2、Qwen 系列、ChemLLM）。
- **关键超参**：规划问题规模设置为 100/200/300/400 个反应（主要实验），以及最多 1000 个 action 的大规模域；使用 Fast Downward 作为外部规划器。
