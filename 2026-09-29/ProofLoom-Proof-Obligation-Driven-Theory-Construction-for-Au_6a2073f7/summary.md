---
title: "ProofLoom-Proof-Obligation-Driven-Theory-Construction-for-Au"
source: https://arxiv.org/pdf/2609.34960v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 03:17:02"
field: "自动化定理证明与形式化数学"
keywords: ["autoformalization", "stochastic optimization", "Lean theorem prover", "LLM agent", "proof obligation", "formal verification", "faithfulness"]
innovations: ["证明义务驱动的理论协同构建与算法模型修订", "签名契约+对抗式Judge审查保障源文献忠实性", "SOptLib跨算法可复用知识与建模经验持续积累"]
benchmarks: ["FOML（Lan教材10算法）", "5个研究论文算法（AMSGrad/STORM/SPIDER/PAGE/SAM）", "33个整体开发评估"]
---

# 论文速读：ProofLoom: Proof-Obligation-Driven Theory Construction for Autoformalizing Research-Level Stochastic Optimization

## 一句话总结
ProofLoom 是一个全自动的 LLM-Agent 系统，针对随机优化算法，以目标证明中的**开放证明义务（Proof Obligation）**为驱动，自动构建 Lean 算法模型与域理论基础设施，并通过签名契约（Signature Contracts）和 Planner–Audit 双重机制确保模型修订与证明构建忠实于源文献。在 15 个教科书与研究论文任务上获得人机双评最高分（6.3/7 与 6.4/7），并揭露了 28 处已发表源材料中的公式错误与证明漏洞。

## 研究问题与动机
- **核心问题**：将研究级随机优化算法及其收敛证明形式化为 Lean 不仅需要翻译定理陈述，还需要构建算法的 Lean 模型（更新规则、随机语义、假设）以及连接 Mathlib 基础库到算法收敛分析的域理论基础设施，而这两者往往大量缺失。
- **自底向上（Bottom-up）风险**：先建理论再证定理的方式固定了算法模型，但模型与证明之间的不匹配可能要到大量开发后才暴露，导致已构建理论需要重做。
- **自顶向下（Top-down）风险**：从目标定理出发可提前暴露不匹配，但为恢复可证性而修订算法模型时，可能无意中引入源文献未声明的假设或弱化结论（即忠实性损失）。
- **已有方法不足**：现有 AutoFormalization 工作（Trellis、LeanMarathon、Archon、OpenGauss 等）缺乏对"模型修订忠实性"和"跨算法知识持续累积"的系统性约束机制。

## 核心贡献（创新点）
1. **证明义务驱动的理论自动构建**：以目标证明中不断开放的 Lean 证明义务为驱动，联合构建算法模型与支撑理论；本质区别在于将理论建设顺序绑定于证明需求而非依赖图先验。
2. **签名契约（Signature Contracts）+ Judge 审查机制**：每次模型修订生成可审查的契约记录（含变化声明、源文献出处、衍生证明义务），独立 Judge 拒绝引入无源假设或弱化结论的修订；与以往方法相比首次将形式化修订置于对抗式语义审查下。
3. **Planner–Audit 双机制**：Planner 自顶向下展开源证明为中间命题蓝图，Audit 自底向上追溯 Lean 依赖与源证明步骤的对应，精确定位缺失的桥梁引理与非必要证明分支；区别于单纯依赖定理生成的方案。
4. **SOptLib 持续知识积累**：每完成一个开发后，Extract 与 Merge 模块将可复用定义/引理泛化、在新开发中实例化验证，并将建模决策与失败证明路线记入自然语言知识库，供后续任务检索使用。
5. **发现 28 处已发表论文中的不一致**：涵盖公式错误、证明漏洞、算法–分析不匹配，并提供 Lean 检查证据；部分发现已得到作者勘误确认（Category B）。

## 方法详解
- **Pipeline 三阶段**：
  - **Model**：基于源材料（算法 A、目标定理 T、源证明 P）与 SOptLib 检索结果，定义规范对象、原始假设、算法状态与更新、随机语义（滤过、适应性）、输出规则与定理陈述；源文献中的假设进入 Setup，其余性质保持为证明义务。
  - **Construct**：Planner 根据当前 Lean 目标、源证明、检索结果与既往尝试生成证明蓝图（ProverStep）或报告阻塞（Blocker）；Prover 在受保护接口下实现蓝图；Audit 回溯 Lean 依赖与源证明步骤的一致性；Refactor 在 Blocker 触发时生成候选修订与签名契约；Judge 对抗审查候选修订。
  - **Learn**：当所有 Lean 义务关闭后，Certify 核查 canonical 入口点、完整依赖、源对应与 Audit 证据；Extract 提取可复用数学对象，泛化算法特定项为参数，在 staging 区重新证明并返回原开发实例化验证；Merge 将通过审查的声明与记录写入 SOptLib。
- **签名契约与忠实性约束**：修订候选 $L'$ 必须满足 $\mathcal{D}_C \subseteq \mathcal{P}(L') \cup \mathcal{O}(L')$ 且 $\mathcal{D}_C \cap \mathcal{A}_{\text{new}}(L') = \emptyset$，即契约所记录的衍生命题不能落入新增假设集合。
- **受保护接口**：审计对比每次 `prove` 编辑的声明头部与 Setup 字段类型，检测 $\Sigma_{\text{lock}}$ 的变更；`reconstruct` 中的每一处修改必须携带源出处标签（如 `source_role_conflict`）与精确引用。
- **SOptLib 分层架构**：GLUE（Mathlib 桥梁层，如条件期望积分）→ MODEL（共享对象，如 Bregman 散度）→ LAYER 0（问题性质，如二阶矩界）→ LAYER 1（收敛引理，如三点不等式）。
- **检索策略**：符号搜索（标识符分割 + 加权词法评分）与 LeanSearch 自然语言查询结合；跨范围重复时优先目标可见声明→项目导入→SOptLib→Mathlib。

## 实验与结果
- **数据集与任务**：15 个任务（10 个来自 Lan 教材 FOML：SMD、SBMD、SCGS、SAPD、NSAGD、SNCCG、RPDG、SCCSP、NSMD、SZO；5 个来自研究论文：AMSGrad、STORM、SPIDER、PAGE、SAM）；共 33 个开发（含 22 个 FOML、8 个研究论文、3 个研究笔记）。
- **评估基线**：Raw Codex、Raw Codex (Goals)、OpenGauss、LeanMarathon、Trellis、Archon（共 6 个），均使用 GPT-5.5 高推理强度。
- **主要结果（人评 1–7 分制）**：
  - FOML：ProofLoom 6.3 vs. Archon（最强基线）4.9；研究论文：ProofLoom 6.4 vs. Archon 5.0。
  - 在所有 16 组（任务群×协议×评审模型）组合中均获最高分。
- **模型评估（G-Eval / FidelityEval，0–100）**：FOML 上 GPT-5.6-sol 给 ProofLoom 92.0（Archon 61.4），研究论文 89.9（Archon 72.7）。
- **消融实验（43 个阻塞案例）**：完整系统正确诊断并提出忠实修复 33 例；去掉 Judge 降为 29 例，错误修复从 1 增至 6 例；去掉 Planner–Audit 亦为 29 例。
- **代码规模**：33 个开发产出 490,693 行算法局部 Lean 代码，零 sorry；SOptLib 含 109,634 行、2,399 个声明（其中 1,983 个定理）。
- **源文献发现**：揭露 28 处不一致，影响 22 个开发，包括 SAM 高斯尺度不匹配、Lan 教材 Lemma 5.8 定义域越界、SPIDER OPTION I/II 命名不一致等，均附 Lean 检查证据。

## 相关工作脉络
1. **OptLib (Li et al., 2024a)**：形式化凸分析与一阶优化速率；本文定位——OptLib 聚焦凸优化理论框架，ProofLoom 面向随机优化研究级算法的端到端证明构建与忠实性控制。
2. **Archon (Ju et al., 2026)**：自主填洞与研究级证明形式化；本文差异——Archon 缺乏签名契约与 Judge 对抗审查机制，且未解决模型修订的忠实性风险。
3. **M2F (Wang et al., 2026c)**：大规模教材自动化形式化；本文差异——M2F 偏重声明翻译与显式依赖管理，ProofLoom 强调"证明义务驱动"的理论与模型协同演化。
4. **LeanMarathon (Zhang et al., 2026b) / Trellis (Pegden, 2026)**：长程证明开发组织；本文差异——二者侧重流程编排，未显式处理算法模型修订与源文献对齐的语义约束。
5. **Open Gauss (Math, Inc., 2026)**：项目级 Lean 工作流；本文差异——Open Gauss 为通用框架，ProofLoom 在此基础上引入 SOptLib 跨算法知识累积与审计回路。
6. **LEGO-Prover (Wang et al., 2023) / LeanAgent (Kumarappan et al., 2025)**：定理证明过程中的库生长与持续学习；本文关联——SOptLib 理念与之相近，但 ProofLoom 额外要求泛化结果在原开发中实例化验证。

## 局限性与未来方向
- **语义审查依赖 LLM 判断**：反复的 LLM 错误可能导致假设或结论逐步偏移源文献。
- **成本与耗时**：构建、语义审查与 Audit 记录均需额外模型调用与 Lean 检查，研究级形式化可能耗时数小时。
- **领域覆盖有限**：当前评估仅覆盖随机优化中的教科书与研究论文任务，提示与控流在更多抽象领域及论证细节较少的文献上的适用性有待验证。
- **迭代开发局限**：提示词与控制流最初在较小 FOML 子集上开发，基础设施 bug 在后续迭代中修复，可能影响早期任务的公平比较。

## 研究启发与可借鉴点
1. **签名契约机制可迁移**：在任意需要"编辑受约束形式化系统"的场景（如程序验证、代码生成）中，引入签名契约 + 对抗审查可有效防止隐性假设注入。
2. **Planner–Audit 双向追溯设计**：前向规划（展开源证明）+ 后向审计（追溯 Lean 依赖对应）的组合，可为其他数学证明自动化任务提供可靠的缺口定位策略。
3. **跨任务知识累积范式**：SOptLib 的"泛化→实例化验证→入库"循环对任何需要多实例开发的 AI 系统（如 API 文档生成、定理证明）均有参考价值。
4. **自动化发现文献不一致**：将形式化过程反向用于科学文献审计，为计算社会学/文献质量评估开辟新路径。
5. **受保护接口（Protected Interface）概念**：将源文献中的声明集合标记为不可篡改，并在每次编辑中强制差分审计，这一设计可用于安全关键的 AI 辅助形式化场景。

## 关键术语表
- **Proof Obligation（证明义务）**：目标证明中尚未在 Lean 中解决的推导步骤或命题，驱动理论与模型的下一步构建。
- **Signature Contract（签名契约）**：记录模型修订中变化的声明、源文献依据与衍生证明义务的不可变记录结构，是忠实性审查的基础。
- **Judge**：独立对抗式审查 Agent，按预设清单检查修订是否引入无源假设或弱化定理结论。
- **Planner–Audit**：Planner 自顶向下展开源证明为中间命题蓝图；Audit 自底向上比对 Lean 实际依赖与源步骤，定位缺失桥梁与非必要分支。
- **SOptLib**：ProofLoom 维护的跨算法可复用随机优化知识库，含分层 Lean 声明与自然语言建模记录。
- **Protected Interface（受保护接口）**：由源文献建立、在证明构建过程中禁止被擅自修改的声明集合，任何变更需经 Refactor–Judge 流程。
- **Bregman Proximal Update（Bregman 近端更新）**：基于 Bregman 散度的迭代更新算子，是随机镜像下降等算法的核心步骤。
- **GLUE / LAYER 0 / LAYER 1**：SOptLib 的三层分类——GLUE 为 Mathlib 桥梁层，LAYER 0 为问题性质层（如噪声矩界），LAYER 1 为收敛引理层（如三点不等式）。

## 可复现要素
- **数据集**：15 个形式化任务（10 个 FOML + 5 个研究论文），来自已发表书籍与论文；附录 B.1 列出了所采用的具体版本与哈希。
- **代码/权重开源**：代码与补充材料见 https://github.com/Trace231/ProofLoom；配套数据包含原始评分与案例标签（`experiments/README.md`）与可复现脚本 `./reproduce.sh tables`。
- **关键超参**：生成模型 GPT-5.5（gpt-5.5，Codex，高推理强度）；每系统–任务累计生成预算 48 小时（含恢复执行）；LLM 评审器 GPT-5.6-sol、Gemini 3.8 Flash、DeepSeek V4 Pro、Claude Opus 5。
- **Lean 环境**：各项目的 `lean-toolchain` 与 `lake-manifest.json` 锁定 Lean 与 Mathlib 版本。
