---
title: "SKILLGYM-TRAINING-SKILL-USE-AGENTS-WITH-AU-TOMATIC-VERIFIABL"
source: https://arxiv.org/pdf/2609.37539v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 14:21:31"
field: "智能体技能学习"
keywords: ["skill-use agents", "synthetic data generation", "supervised finetuning", "agent skills", "environment synthesis", "builder-reviewer pipeline"]
innovations: ["提出SkillGym自动化管道，将社区技能转化为可验证任务并训练技能使用能力", "通过builder-reviewer双智能体系统构建覆盖四种推理结构的技能关键任务", "证明SFT训练主要提升智能体查阅技能的倾向，且泛化至未见过的技能与推理结构"]
benchmarks: ["SkillGym test set", "SkillEval", "SkillsBench", "Skill-Use-Bench"]
---

# 论文速读：SKILLGYM-TRAINING-SKILL-USE-AGENTS-WITH-AU-TOMATIC-VERIFIABL

## 一句话总结
论文提出 SkillGym 自动化管道，将互联网社区技能（skills）转化为可验证的任务环境，收集 19k 成功交互轨迹，并通过监督微调（SFT）显著提升了不同规模 LLM 智能体调用、遵循技能指南的能力，且该能力可泛化至未见过的技能与推理结构。

## 研究问题与动机
1. **现有技能训练路径依赖固定环境**：当前技能使用训练大多从智能体在少量固定环境中的交互经验中提取技能，限制了技能学习覆盖的任务领域。
2. **社区技能缺乏训练所需的结构**：公开社区技能（如 skills.sh）虽涵盖众多领域，但均为独立可复用资源，未配套任务、执行环境及成功验证机制，无法直接用于训练。
3. **如何构建技能关键且可验证的任务**：任务需满足“技能关键性”（不使用技能则完成更困难，但通过探索仍可解决）和“结果可验证性”（通过可执行验证器可靠判断对错），现有工作缺乏系统化构建方法。

## 核心贡献（创新点）
1. **提出 SkillGym 全自动化管道**：从爬取的 184k 社区技能中筛选出 11,897 个可用技能，通过 builder-reviewer 双智能体系统自动生成 6.8k 个覆盖四种推理结构的可验证任务，并收集 19k 成功轨迹。
   - *与已有工作的本质区别*：不同于 SKT 等仅使用模板化任务包的方法，SkillGym 以显式推理结构（程序执行、溯因诊断、约束满足、偏序规划）为核心，并通过严格的验证器审查与执行检查确保任务质量。
2. **构建技能关键任务的设计原则与验证机制**：定义“技能关键性”与“可验证性”原则，设计基于 Docker 的环境构建与“有效性门控+质量审查”两阶段任务生成流程，确保每个任务具备可执行的参考解与验证器。
   - *与已有工作的本质区别*：不同于仅依赖静态规则检查的工作，本文通过 builder-reviewer 交互迭代修复信息泄露、验证器过严/过松等问题，使任务真正依赖技能知识。
3. **验证 SFT 训练对技能使用能力的广泛提升与泛化**：在 6 个不同架构、2B–122B 参数的模型上进行 SFT，在所有四个技能使用基准上获得显著提升；9B 微调模型在 SkillGym 测试集和 SkillEval 上超越 397B 未训练模型。
   - *与已有工作的本质区别*：不同于仅在小规模特定环境上训练的方法，本文训练数据跨 18 个领域、3.5k 种技能，且性能增益延伸至训练中未出现的技能（held-out skills）。
4. **深入分析训练带来的行为改变与泛化模式**：发现训练主要教会智能体“查阅技能”（触发率从 28% 提升至 96%），而非直接应用技能内部方法；对不同推理结构的泛化分析揭示了当前任务分布的局限性（领域方法/工具类技能提升有限）。
   - *与已有工作的本质区别*：通过细粒度行为分析（Skill-Use-Bench 三分量）与任务结构标注，明确指出了当前训练数据的盲区，为后续改进提供方向。

## 方法详解
SkillGym 采用三阶段管道：

1. **技能收集与筛选**
   - 从 `skills.sh`（9.7k 顶级技能）和 `claude-skill-registry`（174k 社区技能）爬取，去重后获 51k 唯一技能。
   - 通过规则与 LLM 双重标注（完整性、语言、运行要求、质量等维度），最终筛选出 11,897 个可离线稳定执行、无破坏性操作、文件数≤300 的技能作为任务构建池。

2. **任务构建（Builder-Reviewer 系统）**
   - **环境构建**：Builder 根据技能需求生成 Dockerfile 与依赖文件，Reviewer 启动容器并独立验证依赖正确性，循环直至通过或达到最大迭代次数。
   - **任务构建**：基于四种**推理结构配置文件**（Procedural、Abductive、Constraint satisfaction、Partial-order），每个文件包含难度层（真实输入规模、重新计算验证、技能边缘案例、隐含关键步骤）与任务契约。
   - **三重门控**：
     - **Fit-check**：判断技能是否适合目标推理结构。
     - **有效性门控**：执行验证器检查 `v(x₀)=0`（初始状态失败）且 `v(Exec(ρ; E, x₀))=1`（参考解通过）。
     - **质量审查**：Reviewer 检查信息泄露、验证器过严/过松等问题，要求验证器基于结果值而非实现形式进行判定。

3. **轨迹收集与 SFT**
   - 使用 Kimi-K3、DeepSeek-V4-Flash、GLM-5.2 三个教师模型，在 MiniSwe-Agent、AgentFly、Terminus-2、OpenCode 四个 Agent harness 上收集 19k 成功轨迹。
   - 通过 varied system prompts、tool names、tool schemas 增加交互多样性。
   - SFT 超参数：学习率 1e-5（线性衰减至零），AdamW 优化器，batch size 128，2 epochs，单模型训练约 64 GPU 小时。

## 实验与结果
- **数据集**：构建 6.8k 任务（覆盖 18 个领域），收集 19k 成功轨迹（失败轨迹 15.6k）用于分析。
- **评估基准**：SkillGym 测试集（含 held-in/held-out 技能子集）、SkillEval、SkillsBench、Skill-Use-Bench。
- **主要结果**（Qwen3.5 系列 SFT 前后对比）：
  - SkillGym 测试集：平均提升 13.8 分；9B 模型从 41.3 → 59.5（+18.2），超越 397B 未训练模型的 55.8。
  - SkillEval：平均提升 9.7 分；9B SFT 达 74.7±1.2，接近 SKT 报告的成绩（74.1±1.4）且未使用其数据。
  - SkillsBench：平均提升 9.7 分；122B 模型从 30.1±2.5 → 53.6±5.2。
  - **Skill-Use-Bench SU 分数**：提升最显著，各模型均提升 21–55 分；9B 模型从 14.8 → 49.6。
- **泛化能力**：
  - Held-out 技能成功率从 42.5% 提升至 62.0%（+19.5 分），略高于 held-in 技能提升（+17.0 分）。
  - 技能访问收益翻倍：SFT 后技能带来成功率提升 16.5 分（基础模型仅 7.5 分）。
- **消融实验**：
  - Reviewer 批准的任务（通过有效性+质量审查）训练效果优于仅通过有效性门控的任务（SkillGym +4.7, SkillEval +6.7, Skill-Use-Bench +9.3）。
  - 混合四种推理结构训练与仅用程序执行结构训练在大多数指标上表现相当，说明多结构并未显著拉高平均分数但扩大了覆盖范围。

## 相关工作脉络
1. **SKT（Tan et al., 2026）**：同样从 skills.sh 技能出发合成任务，但采用模板化任务包，侧重技能提供规则的 appliqu，评估集集中于单一推理结构；SkillGym 以显式推理结构为核心，通过 builder-reviewer 交互提升任务质量。
2. **SkillRL（Xia et al., 2026）**：将可重用知识提取为分层 SkillBank，通过 RL 联合演化技能库与智能体；SkillGym 保持技能外部，仅训练通用技能使用能力，训练后技能仍独立存在。
3. **Voyager（Wang et al., 2023）**、**SkillWeaver（Zheng et al., 2025）**：从交互轨迹中构建/检索技能，属于 training-free 方法；SkillGym 聚焦 training-based，通过 SFT 更新权重。
4. **EnvScaler（Song et al., 2026）**、**Agent-World（Dong et al., 2026）**：环境驱动的合成方法，先构建环境再合成任务；SkillGym 是技能驱动的，从技能反推任务与所需环境。
5. **SkillsBench（Li et al., 2026b）**、**Skill-Use-Bench（Han et al., 2026a）**：人工编写的评测基准，规模小（几百任务），用于评估；SkillGym 提供数千个可训练任务，同时用于性能验证。

## 局限性与未来方向
1. **任务类型分布不均**：当前训练数据中“提供领域方法或内置工具”的技能任务改善有限，而“提供参考文档”的技能任务提升显著（+28 分）。
2. **仅使用 SFT**：未探索强化学习；论文指出 SkillGym 环境已具备可执行验证器，可直接提供 outcome reward，适合进一步进行 RL 训练。
3. **推理结构覆盖不均**：训练数据中程序执行结构占主导，其他结构（如溯因诊断、偏序规划）比例较低，可能影响复杂推理技能的训练效果。
4. **教师模型依赖**：轨迹收集依赖 Kimi-K3、DeepSeek-V4-Flash、GLM-5.2 等强教师模型，若教师模型能力不足可能限制数据质量。

## 研究启发与可借鉴点
1. **Builder-Reviewer 双智能体验证管道**：可用于任何需要自动生成高质量训练数据的研究，通过迭代审查修复信息泄露、验证器缺陷等问题，提升数据可靠性。
2. **技能关键性设计原则**：确保任务在不使用技能时仍可解决但更困难，可通过难度层（真实输入规模、重新计算验证、隐含关键步骤）实现，适用于技能相关任务生成。
3. **多 Harness 轨迹收集策略**：使用多个 Agent harness（不同工具接口、协议）和多个教师模型，增加交互多样性，避免模型对特定交互格式过拟合。
4. **推理结构显式划分**：将任务按四种推理结构分类并设计对应配置文件，有助于控制任务多样性，并便于分析不同结构上的泛化表现。
5. **结合本团队方向的潜在机会**：可将 SkillGym 的管道迁移至垂直领域（如医疗、法律技能），或通过 RL 进一步利用验证器作为 reward 信号，训练更复杂的技能组合与规划能力。

## 关键术语表
- **SkillGym**：自动化管道，将社区技能转化为可验证任务环境，收集轨迹并训练技能使用智能体。
- **Skill-critical task**：技能关键任务，要求必须使用技能才能高效完成，但无需技能也可通过探索求解的任务。
- **Builder-Reviewer pipeline**：双智能体构建流程，Builder 生成环境/任务，Reviewer 验证质量并返回反馈，迭代直至通过。
- **Executable verifier**：可执行验证器，自动检查任务输出是否符合要求的脚本，支持多种合法解法。
- **Reasoning structure**：推理结构，包括程序执行、溯因诊断、约束满足、偏序规划四种类型，定义任务求解的逻辑模式。
- **SFT (Supervised Finetuning)**：监督微调，使用成功轨迹（observation-action pairs）对预训练模型进行指令微调。
- **Held-out skill**：未见过的技能，指在训练数据中未出现但在测试中评估的技能，用于检验泛化能力。
- **Agent harness**：智能体框架，如 MiniSwe-Agent、AgentFly 等，提供智能体与环境交互的接口与工具集。

## 可复现要素
- **数据集**：6.8k 任务、19k 成功轨迹，代码与数据已在 GitHub 开源：https://github.com/Reason-Wang/SkillGym。
- **代码/权重**：代码开源；SFT 训练的模型权重未明确提及开源情况（论文未提及）。
- **关键超参数**：学习率 1e-5，线性衰减至零；AdamW 优化器；batch size 128；2 epochs；64 GPU 小时（单模型）；上下文长度 64k（Qwen3.5-9B 消融实验）。
- **教师模型**：Kimi-K3、DeepSeek-V4-Flash、GLM-5.2。
- **Agent harnesses**：MiniSwe-Agent、AgentFly、Terminus-2、OpenCode。
