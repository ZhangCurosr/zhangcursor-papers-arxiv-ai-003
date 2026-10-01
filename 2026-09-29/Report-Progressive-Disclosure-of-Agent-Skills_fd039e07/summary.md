---
title: "Report-Progressive-Disclosure-of-Agent-Skills"
source: https://arxiv.org/pdf/2609.35692v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 03:22:16"
field: "智能体技能管理与检索"
keywords: ["Agent Skills", "Progressive Disclosure", "Lazy Loading", "LLM Agent", "Skill Retrieval", "Prompt Engineering"]
innovations: ["首次在生产环境中实证评估渐进披露技能的Token节省、检索质量与延迟权衡", "提出基于激活码的技能检索评测方法，精确量化LLM是否真正加载目标Skill", "揭示大模型对干扰Skill的鲁棒性差异，为Skill库规模与模型选择提供量化依据"]
benchmarks: ["Qwen3-14B 100% skill retrieval at N=100; Qwen3-8B 15% latency increase at N=50"]
---

# 论文速读：Report: Progressive Disclosure of Agent Skills

## 一句话总结
本文在 Workday 生产环境中实证比较了智能体技能的"主动加载（eager loading）"与"渐进披露（progressive disclosure，即按需懒加载）"两种技能管理范式，发现渐进披露可大幅降低 Token 消耗并提升技能检索质量，代价是整体延迟略有上升。

## 研究问题与动机
- **技能库膨胀导致成本飙升**：随着工作客户对 LLM 智能体的需求增长，Skill 库规模持续扩大；主动加载将所有 Skill 全文注入 prompt 上下文，Token 用量随库大小显著增长（经 Mann–Whitney U 检验确认），直接推高运营费用。
- **主动加载下检索质量恶化**：Skill 库越大，"干扰 Skill（distractor skills）"越容易淹没目标 Skill，LLM 选择出错的概率急剧上升，N=100 时主动加载直接触发 context overflow（crash 率 100%）。
- **安全监控缺口**：大规模 Skill 库存在被恶意植入漏洞的风险，但目前缺乏有效的风险暴露监测手段。
- **渐进披露的理论收益与实际效果尚不明确**：渐进披露作为业界常用手段（如 LangChain 已采用），其对工作延迟、检索质量和成本的综合影响缺乏系统性实证评估。

## 核心贡献（创新点）
- **首次对生产级智能体技能的渐进披露机制进行系统性实证评估**，量化了其在 Token 节省、检索质量与延迟三个维度的影响。
- **设计了基于激活码（activation code）的技能检索评测任务**：为每个 Skill 的正文内容注入随机生成的唯一标识符，要求 LLM 在输出结构化格式时返回正确的激活码，从而精确测量技能检索准确性而非模糊的"任务完成度"。
- **揭示了渐进披露的核心权衡关系**：渐进披露可减少高达 81.7% 的 Token 消耗并完全消除 context overflow，但引入额外 LLM 调用使整体延迟上升约 15%，为工程决策提供量化依据。
- **识别了模型规模对检索鲁棒性的差异**：Qwen3-14B 对简单/中等难度干扰项表现出韧性，但对高难度干扰项仍脆弱；Qwen3-8B 在小库（N=5）下甚至几乎不节省 Token（仅 0.5%），提示小模型下策略收益有限。

## 方法详解
- **技能定义格式**：遵循 Agent Skills 开放标准，每个 Skill 为一个目录，至少包含一个 `SKILL.md` 文件，内含 YAML frontmatter（名称 `name`、描述 `description`）和 Markdown 正文（逐步指令、示例、边界情况等）。
- **主动加载（Eager Loading）**：将全部 N 个 Skill 的完整定义一次性注入 LLM prompt 上下文，LLM 直接基于全量信息检索目标 Skill。
- **渐进披露（Progressive Disclosure）**：
  1. 初始 prompt 仅注入所有 N 个 Skill 的 frontmatter（名称和简短描述），让 LLM 判断最相关的 Skill；
  2. LLM 输出结构化命令（如 `{"action":"load_skill","name":"pptx"}`）；
  3. Harness 在后续 LLM 调用中将选中 Skill 的完整正文加载入上下文，执行任务。
- **评测指标**：
  - **Skill 检索成功率**：输出正确 Skill 名称且激活码匹配；失败类型包括 `wrong_name`、`wrong_code`、`crash`（context overflow 或输出格式错误）。
  - **Token 用量**：记录每轮 prompt 与 completion token 总数。
  - **整体延迟**：从请求到响应完成的 wall-clock 时间（秒）。
- **干扰项设计**：除目标 Skill 外，从 12 个领域家族生成不同难度级别（easy/medium/hard，按触发短语的直接程度分级）的干扰 Skill。

## 实验与结果
- **数据集与配置**：基于 5 个相关 Skill 构建 24 个任务实例，3 种随机种子；Skill 库大小 N ∈ {5, 20, 50, 100}；使用 Qwen2.5-7B-Instruct、Qwen3-8B、Qwen3-14B 三种模型；每种组合 72 次 rollout 取均值；依托 NVIDIA L40S（48GB）部署。
- **Token 节省（渐进披露 vs 主动加载）**：
  | N | Qwen3-14B 节省 | Qwen2.5-7B 节省 |
  |---|---|---|
  | 20 | 74.4% | 67.9% |
  | 50 | 81.7% | 76.6% |
  | 100 | —（主动加载全 crash） | —（主动加载全 crash） |
- **检索质量（成功率的对比）**：
  - Qwen3-14B：N=50 时主动加载 0.13 → 渐进披露 0.72；N=100 时主动加载 0.00（全 crash）→ 渐进披露 1.00。
  - Qwen2.5-7B：N=50 时 0.40 → 0.67；N=100 时 0.00 → 1.00。
  - Qwen3-8B：N=20 时略降（0.68 → 0.38），但 N=50 时回升（0.29 → 0.64），N=100 时达 0.79。
- **延迟影响**：以 Qwen3-8B + N=50 为例，渐进披露使整体延迟从 1.75 秒升至 2.01 秒（约 15% 增幅），主因是额外的 Skill 选择 LLM 调用。
- **稳定性**：N=100 时主动加载在全部模型上发生 100% context overflow crash；渐进披露零 crash。
- **最强结果**：Qwen3-14B + 渐进披露在 N=100 时达到 100% 检索成功率，同时将 Token 消耗从不可行的 crash 状态降至约 4661 tokens。

## 相关工作脉络
- **LangChain 等智能体框架的懒加载机制**：业界已在工程层面采用渐进披露思路，但本文首次在受控生产环境下量化其效果，填补实证空白。
- **Same Task, More Tokens（Levy et al., 2024）**：该工作指出输入长度增加会降低 LLM 推理性能，与本文发现的"主动加载下 Skill 增多导致检索质量下降"结论相互印证，但本文聚焦于 Skill 管理策略这一更具体的场景。
- **PagedAttention（Kwon et al., 2023）**：通过显存分页优化 LLM 服务效率，属于基础设施层面的优化；本文工作在应用层通过减少 prompt 中注入的冗余内容实现等效效果，两者可互补。
- **Skill 作为智能体能力扩展机制**：Workday 内部实践将 Domain Expertise（workflows、scripts、templates）打包为可复用 Skill，与通用工具调用（function calling）范式形成对比——本文强调"文本化规范"而非"代码化接口"的 Skill 管理。
- **Agent 安全性研究**：本文提及恶意 Skill 注入的风险，指向"技能供应链安全"这一新兴方向，当前尚无成熟解决方案。

## 局限性与未来方向
- **单 Skill 检索限制**：评测仅考察每次任务选择 1 个 Skill 的场景；实际应用中可能需要同时加载多个 Skill，且任务本身会决定所需 Skill 数量，这一点在现实中仍未解决。
- **Skill 定义范围有限**：仅考虑了单个 `SKILL.md` 文件；实际 Skill 目录可能包含脚本、参考文档、资产等多个文件，会进一步增加上下文负担。
- **小模型下收益有限**：Qwen3-8B 在 N=5 时 Token 节省仅 0.5%，提示在小规模库或小型模型下策略价值不明显。
- **延迟增高的折中未深入优化**：额外的一次 LLM 调用增加了约 15% 延迟，如何在不牺牲质量的前提下降低这一开销（如并行选择、缓存机制）值得探索。
- **风险暴露监控缺失**：论文提出如何监测恶意 Skill 的风险问题，但未给出解决方案，留作开放研究问题。

## 研究启发与可借鉴点
- **激活码评测范式可直接复用**：将唯一标识符注入 Skill 正文、要求结构化输出匹配的设计，提供了一个干净、可量化的 Skill 检索评估方法，避免依赖任务完成度等模糊指标。
- **渐进披露适用于 Token 成本敏感的生产场景**：对于需要管理大量 Domain Expertise 的智能体系统（如企业内部知识库问答、自动化工作流编排），优先采用渐进披露可带来显著的运营成本下降。
- **小模型下需评估策略收益**：本文提示在选择是否启用渐进披露时需结合模型规模和 Skill 库大小综合考量——若库规模小且使用中等模型，收益可能有限，不值得引入额外调用延迟。
- **可与 RAG 流程结合**：渐进披露的本质是"先粗筛、后精读"的两阶段检索策略，这一思想可直接迁移至文档检索（先检索元数据标题，再加载完整段落）等 RAG 场景。

## 关键术语表
**Progressive Disclosure（渐进披露）**：按需懒加载技能的策略，初始仅向 LLM 上下文注入所有 Skill 的简要描述（frontmatter），根据任务需求逐步加载最相关 Skill 的完整内容。

**Eager Loading（主动加载）**：将所有 Skill 的完整定义一次性注入 LLM prompt 上下文的技能管理方式，简单直接但不具备可扩展性。

**Skill（智能体技能）**：将领域专业知识（工作流、脚本、模板等）打包为可复用的结构化指令集合，每个 Skill 包含名称、描述和正文，遵循 Agent Skills 开放标准。

**Activation Code（激活码）**：为每个 Skill 正文注入的唯一随机标识符，用于在评测中精确验证 LLM 是否真正检索并加载了目标 Skill 的内容，而非仅靠名称猜测。

**Distractor Skill（干扰 Skill）**：在评测中混入的不相关 Skill，用于模拟真实场景中 Skill 库的规模噪声，按 easy/medium/hard 三级难度设置触发短语的直接程度。

**Context Overflow（上下文溢出）**：prompt 总长度超过 LLM 最大上下文窗口导致的错误，表现为 `crash`，是主动加载在大库规模下的主要失效模式。

**Mann–Whitney U Test**：非参数统计检验方法，用于验证 Token 用量差异的统计显著性（本文报告 p < 0.025）。

## 可复现要素
- **数据集**：基于 5 个内部 Skill 自定义构建，**未公开**；任务实例 24 个，含 3 种子种子随机生成的激活码。
- **代码**：**未开源**，使用 vLLM 部署于 NVIDIA L40S（48GB）。
- **关键超参**：
  - 模型：Qwen2.5-7B-Instruct / Qwen3-8B / Qwen3-14B，greedy decoding，32k context window
  - Skill 库大小 N ∈ {5, 20, 50, 100}
  - 每组合 72 次 rollout（4 库大小 × 2 策略 × 24 任务 × 3 种子）
  - 输出格式：`RESULT[<skill-name>]: <activation-code>`
