---
title: "Report-Progressive-Disclosure-of-Agent-Skills"
source: https://arxiv.org/pdf/2609.35692v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 03:21:21"
field: "Agent Skill Management"
keywords: ["agent skills", "progressive disclosure", "skill management", "LLM agents", "context management", "tool selection", "cost-quality tradeoff"]
innovations: ["首次系统实证评估渐进式披露技能管理策略，揭示其在 token 节省（最高 81.7%）和技能检索质量上的优势", "提出 activation code 评估范式和分层难度干扰项生成器，为 skill retrieval 任务提供可量化 benchmark", "建立 token usage、skill-retrieval quality、latency 三维度评估框架，量化成本-性能权衡"]
benchmarks: ["Skill Retrieval Benchmark (activation code based)", "Multi-size library evaluation (N=5/20/50/100)", "Qwen3 series (7B/8B/14B) with 32k context"]
---

# 论文速读：Report-Progressive-Disclosure-of-Agent-Skills

## 一句话总结
Workday AI Research 实证研究了渐进式披露（progressive disclosure）技能管理策略对 LLM-based agent 的影响，发现该策略可显著降低 token 用量（最高节省 81.7%）、改善技能检索质量并提升可靠性，代价是整体延迟略有增加（约 15%）。

## 研究问题与动机
1. **急切加载不可扩展**：agent 技能库随用户增长而扩大，急切加载（eager loading）将所有技能完整内容注入 LLM context，导致 token 用量激增，运营成本显著上升。
2. **上下文溢出风险**：当技能库达到 100 个时，急切加载在全部 rollout 中均发生 context overflow（crash），agent 完全失效。
3. **渐进式披露的效果未知**：虽然业界框架（如 LangChain）已支持懒加载/渐进式披露，但其对技能检索质量、延迟和成本的综合影响尚缺乏系统性实证研究。
4. **检索质量退化问题**：随着技能库增大，干扰项（distractor skills）增多，急切加载下技能检索成功率急剧下降（Qwen3-14B 从 1.00 降至 0.125）。

## 核心贡献（创新点）
1. **首次系统实证评估渐进式披露策略**：建立了一套包含 skill-retrieval success rate、token usage、overall latency 三维度的量化评估框架，填补了技能管理机制对比研究的空白。
2. **揭示技能库规模与检索质量的非线性关系**：发现急切加载下检索质量随 N 增大呈指数级退化，而渐进式披露在 N=100 时仍能保持 100% 成功率（Qwen3-14B），本质区别在于"先筛选再加载"避免了干扰项污染。
3. **量化成本-性能权衡**：给出明确的 tradeoff 曲线——渐进式披露最多节省 81.7% token 但增加约 15% 延迟（N=50, Qwen3-8B），为工程实践提供决策依据。
4. **提出标准化技能管理评估基准**：设计了含 activation code 的 skill-retrieval 任务和三层难度干扰项生成器，为后续研究提供了可复用的评测协议。

## 方法详解
**技能库结构**：每个 skill 是一个目录，核心为 `SKILL.md`，包含 YAML frontmatter（name + description）和 Markdown 主体内容（操作步骤、示例、边界情况等），遵循 open standard format for Agent Skills。

**两种管理策略**：
- **急切加载（Eager Loading）**：将所有 N 个技能的完整 frontmatter + body 一次性注入 LLM prompt context。
- **渐进式披露（Progressive Disclosure）**：
  1. **Phase 1**：仅注入所有 N 个技能的 frontmatter（name + description），prompt LLM 输出结构化命令选择最相关技能，例如 `{"action":"load_skill","name":"pptx"}`。
  2. **Phase 2**：harness 加载选定技能的完整 body 内容，注入 context 后再次调用 LLM 执行任务。

**评估指标设计**：
- **Skill retrieval success**：为每个技能注入唯一 activation code（随机生成），要求 agent 返回正确的 `<skill-name>` 和对应 code。
- **结果分类**：success（完全正确）、wrong_name（选错技能）、wrong_code（选对技能但 code 错误）、crash（context overflow 或格式错误）。
- **干扰项设计**：基于 12 个 domain family 生成三层难度（easy/medium/hard distractors），通过 trigger phrasing 的直接程度区分。

**实验设置**：
- N ∈ {5, 20, 50, 100}
- LLM cores：Qwen2.5-7B-Instruct、Qwen3-8B、Qwen3-14B，greedy decoding，32k context window
- 24 个 task instances × 3 seeds × 2 regimes = 144 conditions，每条件 72 rollouts，共 576 rollouts/LLM core
- 使用 vLLM 服务，单卡 NVIDIA L40S（48GB）

## 实验与结果
**Token 使用（Table 1 Tok↓）**：
- N=20：节省 58.4%~74.4%，N=50：节省 71.3%~81.7%
- N=100 时急切加载全部 crash，无法计算节省比例
- Token 节省随 N 单调递增，统计显著性 p < 10⁻²⁵（Mann–Whitney U test）

**Skill Retrieval（Table 1 Skill retrieval succ）**：
| N | Qwen2.5-7B EL→PD | Qwen3-8B EL→PD | Qwen3-14B EL→PD |
|---|---|---|---|
| 5 | 1.00→1.00 | 1.00→1.00 | 1.00→1.00 |
| 20 | 0.17→1.00 | 0.68→0.38 | 0.42→0.96 |
| 50 | 0.40→0.67 | 0.29→0.64 | 0.13→0.72 |
| 100 | 0.00→1.00 | 0.00→0.79 | 0.00→1.00 |

- 渐进式披露在 N≥20 时全面优于急切加载
- Qwen3-14B 对 easy/medium distractors 鲁棒，但对 hard distractors 仍敏感

**Reliability（crash）**：
- 急切加载 N=100 时 100% crash（三种 LLM 均如此）
- 渐进式披露在所有 N 下 crash rate = 0

**Latency（Section Key Findings）**：
- N=50, Qwen3-8B：从 1.75s 增至 2.01s（+15%）
- 原因：额外一次 LLM invocation 用于 skill 选择

**最强结果**：Qwen3-14B + 渐进式披露，N=100 时 skill retrieval = 1.00，token 节省 = 81.7%（相对 N=50 的 eager loading），且零 crash。

## 相关工作脉络
1. **LangChain 等框架的懒加载机制**：本文是首次对这一业界常用策略进行系统性量化评估，此前缺乏实证数据支撑其 effectiveness。
2. **Levy et al. (2024) "Same Task, More Tokens"**：揭示输入长度对推理性能的负面影响；本文与之呼应，证明技能库膨胀对检索质量的损害更为严重（从 1.00 跌至 0.13）。
3. **Agent skills 标准化努力（open standard for Agent Skills）**：本文遵循该标准设计实验，推动了 skill management 的可比性研究。
4. **Context overflow 研究**：急切加载在 N=100 时 100% crash，与已知"更长 prompt 导致性能下降"现象一致，但本文首次将此问题定位到 skill management regime。
5. **Skill retrieval / tool selection 工作**：本文的 activation code 评估范式为 skill selection 任务提供了新的 benchmark 思路，区别于传统的 function calling 评测。
6. **Workday 生产环境经验**：基于 5,500+ 客户的部署数据，本文具有独特的工业级实验视角，区别于纯学术研究。

## 局限性与未来方向
1. **单 skill 检索假设**：实验限制每个任务仅检索一个 skill，但实际任务可能需要多个技能，如何平衡 skill 定义冗长度与任务复杂度是开放问题。
2. **单一文件限制**：当前每个 skill 仅含 `SKILL.md`，实践中可能包含脚本、参考资料、资产等多文件，文件级粒度管理策略亟待研究。
3. **安全风险未覆盖**：大规模 skill 库可能隐藏恶意或漏洞代码，缺乏对 skill 安全性的监控和评估机制。
4. **延迟优化空间**：渐进式披露增加了一次 LLM call，如何通过优化调度（如并行预加载、缓存命中）降低延迟是可行方向。
5. **仅评估 greedy decoding**：未探索 sampling 或其他解码策略对 skill retrieval quality 的影响。

## 研究启发与可借鉴点
1. **activation code 评估范式**：为 skill/tool 检索任务提供简洁、可量化的评估方法，可迁移至任何需要评估"选择正确接口/技能"的研究。
2. **分层难度干扰项生成**：基于 12 个 domain family 和 trigger phrasing 直接度设计 easy/medium/hard 三级干扰，为 skill retrieval benchmark 建设提供可复用模板。
3. **成本-性能权衡量化框架**：token savings、retrieval quality、latency 三维度联合评估，可作为 agent skill management 研究的标准评测协议。
4. **渐进式披露的工程启发**：对于工具调用（tool calling）场景，可借鉴"先选后载"两阶段策略，避免 context pollution；但需权衡额外 LLM call 的延迟开销。
5. **大规模 skill 库的安全审计**：论文指出安全隐患问题，可结合本团队方向研究 skill 内容的静态分析和风险检测。

## 关键术语表
- **Progressive Disclosure（渐进式披露）**：按需加载技能的策略，初始仅展示 frontmatter，根据任务需求动态加载完整内容，相当于技能管理的"懒加载"。
- **Eager Loading（急切加载）**：一次性将所有技能的完整定义注入 LLM context 的策略，简单但随库规模扩大导致 token 暴增和检索质量下降。
- **Skill Retrieval**：agent 从技能库中选择与当前任务最相关 skill 的能力，是渐进式披露策略的核心环节。
- **Activation Code**：为每个技能注入的唯一随机标识符，用于验证 agent 是否真正加载并理解了目标技能的完整内容。
- **Distractor Skills（干扰技能）**：与任务无关或弱相关的 skill，用于模拟真实场景中的噪音，测试 agent 的检索鲁棒性。
- **Crash**：context overflow 或 malformed output，表示 agent 因超出 LLM 上下文窗口或格式错误而失败。
- **Frontmatter**：SKILL.md 文件开头的 YAML 元数据块，包含 skill 名称和简要描述，是渐进式披露中初始加载的最小信息集。
- **Open Standard for Agent Skills**：由社区制定的技能目录结构标准，规定每个 skill 至少包含 SKILL.md 文件及其 frontmatter 格式。

## 可复现要素
- **数据集**：24 个 task instances + 干扰项生成器（12 个 domain families），基于随机种子生成；未明确声明公开。
- **代码**：论文未提及开源代码。
- **权重**：使用 Qwen2.5-7B-Instruct、Qwen3-8B、Qwen3-14B，可通过 HuggingFace 获取。
- **关键超参**：32k context window，greedy decoding，N ∈ {5, 20, 50, 100}，每条件 72 rollouts。
- **硬件**：单卡 NVIDIA L40S（48GB），vLLM 服务。
