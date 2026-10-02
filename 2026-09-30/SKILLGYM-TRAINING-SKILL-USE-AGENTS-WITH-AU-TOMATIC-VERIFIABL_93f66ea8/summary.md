---
title: "SKILLGYM-TRAINING-SKILL-USE-AGENTS-WITH-AU-TOMATIC-VERIFIABL"
source: https://arxiv.org/pdf/2609.37539v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 14:21:26"
field: "LLM Agent Training"
keywords: ["Agent Skills", "Synthetic Data", "Supervised Fine-tuning", "Environment Generation", "Skill Use Benchmarking"]
innovations: ["提出 SkillGym 自动化流水线将社区技能转化为可验证训练任务", "引入四种推理结构 Task Profile 和 Builder-Reviewer 双门控质量审查机制", "跨 2B-122B 模型规模验证技能使用能力的 SFT 泛化提升"]
benchmarks: ["SkillGym Test Set", "SkillEval", "SkillsBench", "Skill-Use-Bench"]
---

# 论文速读：SKILLGYM-TRAINING-SKILL-USE-AGENTS-WITH-AU-TOMATIC-VERIFIABL

## 一句话总结
论文提出了 SkillGym，一种自动化流水线，将社区编写的 Agent Skills 转化为可执行、可验证的训练任务环境，通过 builder-reviewer 多轮交互构建涵盖四种推理结构的 6.8k 任务，并利用 19k 条验证通过的轨迹对 LLM 进行 SFT，显著提升了模型在 SkillGym 自测集及 SkillEval、SkillsBench 等多个基准上的技能使用能力。

## 研究问题与动机
- **现有方法局限于少量任务域**：现有 skill-use 训练方法多从 Agent 在固定环境中的经验出发来合成技能，限制了技能学习覆盖的任务领域范围，而社区开源技能覆盖了软件工程、科学、金融、文档处理等广泛领域。
- **技能无法直接用作训练材料**：公开的技能写作目标是作为可复用资源，缺乏配套的任务描述、运行环境和成功判定机制，需要将其转化为"skill-critical"的任务才能用于训练。
- **任务设计需满足双重约束**：任务必须同时满足（1）skill-critical（技能的运用决定任务结果，但无技能仍可尝试解决）和（2）可可靠验证（验证器能识别多种有效解法，区分真正成功与表面合理的输出）。

## 核心贡献（创新点）
1. **提出 SkillGym 自动化流水线，将社区技能转化为可执行训练环境**：与已有工作依赖人工编写或特定环境不同，本文从 184k 个爬取技能中自动构建 6.8k 个具备 executable verifier 的任务。
2. **基于四种推理结构定义任务画像（Task Profiles）**：将技能应用划分为过程执行（Procedural）、溯因诊断（Abductive）、约束满足（Constraint Satisfaction）和部分有序规划（Partial Order），使任务多样化并覆盖不同推理模式。
3. **引入两阶段 Builder-Reviewer 代理系统进行任务构建与质量审查**：相比 SKT 等仅依赖模板的任务合成方法，本文通过执行检查（validity gate）和质量审查（quality gate）双保险，确保任务无信息泄露且验证器合理。
4. **大规模 SFT 验证了跨模型家族的泛化提升**：在 2B 至 122B 参数的多个 LLM 上验证，9B 微调模型在 SkillGym 测试集和 SkillEval 上超越了 397B 未训练的 Qwen3.5。

## 方法详解
**整体流程分为三个阶段：**

1. **技能收集与筛选**：从 skills.sh 和 claude-skill-registry 爬取 184k 条技能，去重后得到 51k 条；通过规则检查（语言为英语、非归档/批量发布、文件数≤300、无需 GPU/网络/交互）筛选出 11,897 条技能池。

2. **任务构建（Builder-Reviewer 两阶段）**：
   - **环境构建**：Builder 根据技能依赖生成 Dockerfile，Reviewer 独立验证镜像可运行；不通过则反馈修改，直至通过或达到迭代上限。
   - **任务编写**：Builder 根据四种 Task Profile 之一构建任务包（instruction.md、初始工作区、参考解 solve.sh、验证器 test.sh）；通过三个门控：
     - **Fit-check**：技能是否适合目标任务类型；
     - **Validity gate**：验证器在初始工作区失败、在参考解执行后通过，即 $v(x_0)=0$ 且 $v(\text{Exec}(\rho;\mathcal{E}, x_0))=1$；
     - **Quality gate**：Reviewer 检查信息泄露、验证器过严或过松等问题。
   - 难度层设计包括：真实规模输入、独立重算验证（recompute-verify）、植入技能文档中的边缘情况、以及仅凭技能知识才能解决的隐含步骤。

3. **轨迹收集与 SFT**：使用 Kimi-K3、DeepSeek-V4-Flash、GLM-5.2 三个 teacher 模型，通过 MiniSwe-Agent、AgentFly、Terminus-2、OpenCode 四种 agent harness 收集轨迹；最终得到 19k 条验证通过的成功轨迹，以 learning rate $10^{-5}$、batch size 128、2 epochs 进行 SFT。

## 实验与结果
- **数据集**：SkillGym 自建测试集（含 held-in 和 held-out 各 200 题）、SkillEval（100 题）、SkillsBench（87 题）、Skill-Use-Bench（多维度评估）。
- **评估基线**：Qwen3.5-397B、GLM-5.2-753B、DeepSeek-V4-Flash-284B/13B、Kimi-K3-2.8T/104B 等大模型。
- **主要结果**：
  - 6 个 backbone（MiniCPM5-2B、Ministral-3-8B、Qwen3.5-4B/9B/27B/122B-A10B）在 22/24 项比较中提升；SkillGym 测试集平均提升 13.8 分，SkillEval +9.7，SkillsBench +9.7，Skill-Use-Bench SU 提升 21-55 分。
  - **Qwen3.5-9B SFT 模型在 SkillGym 测试集（59.5）和 SkillEval（74.7）上超越 Qwen3.5-397B**（55.8 和 73.8）。
  - Held-out skills 提升 19.5 个百分点（42.5%→62.0%），证明跨技能泛化。
  - Skill-Use-Bench 中 Trigger 从 28% 提升至 96%，表明训练教会了模型主动查阅技能。
- **消融实验**：Review-approved 数据优于仅 Validated 数据（SkillGym 测试集 +4.7 分）；四种任务结构混合训练与单一 Procedural 训练表现相当。

## 相关工作脉络
1. **Skill-to-LoRA (Zhang & Qi, 2026)**：将技能合成入适配器；本文保持技能外部化，训练通用技能使用能力并泛化到未见技能。
2. **SkillRL (Xia et al., 2026)**：通过 RL 联合进化技能库和 Agent；本文仅做 SFT，且任务来源为社区公开技能而非 Agent 自身经验。
3. **SKT (Tan et al., 2026)**：同样基于 skills.sh 合成任务，但采用模板驱动且侧重规则应用；本文引入四种推理结构和双重门控审查机制。
4. **SWE-Bench / SkillsBench**：现有基准仅含数百任务、面向评估；本文构建数千任务用于训练，并用这些基准测量迁移效果。
5. **EnvScaler / Agent-World**：环境驱动的任务合成；本文是技能驱动，从技能出发构建围绕其应用的任务。

## 局限性与未来方向
- **部分技能类型提升有限**：SkillsBench 中依赖参考文档的技能提升显著（+28.0），但依赖领域方法、工具或系统修改的任务提升不明显。
- **仅使用 SFT 训练**：作者指出，由于 SkillGym 环境已有可执行验证器提供 outcome reward，未来可利用 RL 进一步训练。
- **部分任务结构在训练中占比低**：如规则应用仅占 12%，但在测试集上仍有提升，说明泛化能力存在，但数据分布差异可能影响某些场景。
- **成本估算**：任务构建阶段消耗较大（约 $5.7k–8.4k），reviewer 成本为假设值。

## 研究启发与可借鉴点
1. **Builder-Reviewer 双代理架构可用于其他合成任务场景**：通过执行验证（validity gate）和质量审查（quality gate）双层过滤，可有效提升合成数据质量，减少对人工标注的依赖。
2. **Recompute-verify 设计原则值得借鉴**：验证器从输入独立重算期望值而非硬编码，使任务对输入扰动鲁棒，同时防止模型通过记忆绕过技能学习。
3. **多 harness 多 teacher 轨迹收集策略**：使用不同交互协议（Bash vs JSON）和不同 reasoning 风格的 teacher 模型，增强了数据的多样性和 SFT 的泛化性。
4. **Skill-critical 任务设计原则**：任务应"有技能时更容易，无技能时仍可尝试"，这一设计确保了训练信号指向技能使用行为而非通用问题解决能力。
5. **跨模型尺寸统一提升**：从 2B 到 122B 均有显著改善，表明技能使用能力的提升具有规模不变性，小模型同样可从高质量轨迹中受益。

## 关键术语表
- **SkillGym**：论文提出的自动化流水线，将社区技能转化为可执行、可验证的训练任务和轨迹。
- **Skill-critical**：任务设计原则，指技能的应用决定任务结果，但任务在无技能时仍可尝试解决。
- **Builder-Reviewer Pipeline**：两阶段代理协作系统，Builder 负责构建环境和任务，Reviewer 负责验证质量和反馈修改。
- **Task Profile**：四种推理结构定义的任务模板，包括 Procedural、Abductive、Constraint Satisfaction、Partial Order。
- **Recompute-verify**：验证器设计原则，从输入独立重算期望输出而非硬编码答案，以接受多种有效解法。
- **Validity Gate**：任务构建的质量门控之一，确保验证器在初始工作区失败、在参考解通过后接受。
- **Held-out Skills**：训练数据中未出现的技能，用于评估模型的跨技能泛化能力。
- **SU Score (Skill-Use Score)**：Skill-Use-Bench 的综合分数，结合 Trigger、Compliance、Boundary 三个维度。

## 可复现要素
- **数据集**：6.8k 任务、19k 轨迹；论文声明代码和数据已开源：https://github.com/Reason-Wang/SkillGym
- **模型**：使用了 Kimi-K3、DeepSeek-V4-Flash、GLM-5.2 作为 teacher；SFT 模型包括 MiniCPM5-2B、Ministral-3-8B、Qwen3.5-4B/9B/27B/122B-A10B
- **超参数**：learning rate $10^{-5}$，linear scheduler 衰减至 0，AdamW optimizer，batch size 128，2 epochs，64 GPU hours
- **Agent Harnesses**：MiniSwe-Agent、AgentFly、Terminus-2、OpenCode
- **训练硬件**：未明确提及具体 GPU 型号
