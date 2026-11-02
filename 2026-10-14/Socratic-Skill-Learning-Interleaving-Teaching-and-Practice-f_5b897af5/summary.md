---
title: "Socratic-Skill-Learning-Interleaving-Teaching-and-Practice-f"
source: https://arxiv.org/pdf/2610.17860v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-02 09:04:30"
---

# 论文速读：Socratic-Skill-Learning-Interleaving-Teaching-and-Practice-f

## 一句话总结
提出 Socratic Skill Learning (SSL) 框架，通过 Educator（只读指导）与 Student（唯一环境操作者与 Skill 作者）的双主体 interleaving 交互，在实时探究阶段动态注入专家经验，显著提升 Agent Skill 的跨任务迁移能力；同时构建 SKILLTRANSFERBENCH 基准并在 Qwen/DeepSeek/Kimi 三组模型对上验证了 Issue-Guided SSL 在降低 61.7% 专家 token 消耗的同时达到最高 0.805 效用。

## 研究问题与动机
- **干预时机不足**：现有 Skill 学习方法（轨迹蒸馏、事后反馈）多在 Student 完成尝试或探索结束后提供指导，无法在假设尚未被证据检验时及时纠偏。
- **指导组织方式缺乏对比**：外部专家应如何穿插于实时探究中？定时 review 与按需 review 对迁移效果与监督成本的影响尚不明确。
- **迁移评估基准局限**：既有基准难以公平区分 Skill 泛化能力与 anchor 过拟合，缺乏公开的变化规格与 fresh solver 冻结评估机制。
- **教学需求演化未建模**：随着 Skill 逐步完善，教学需求显著下降且固定间隔 review 产生大量无效 token，需要按需聚焦的调度策略。

## 核心贡献（创新点）
1. **构建 SKILLTRANSFERBENCH 基准**：该基准通过公开任务族规格（invariant goals 与 permitted variations）与 fresh solver 冻结评估，严格测量 Skill 在未见案例上的迁移效用；与 SkillsBench/EconSkills 的本质区别在于显式分离 anchor 探索与 held-out 验证，防止数据泄露并统一监督成本度量。
2. **提出 Socratic Skill Learning (SSL) 双主体框架**：将外部指导实时穿插于 Student 的探索循环中，Student 保留环境操作权与 Skill 编写权；与 Trace2Skill/SKILL-KD 等离线蒸馏方法的本质区别在于反馈在假设检验阶段即时生效，而非事后修补已固化轨迹。
3. **设计 Issue-Guided SSL 按需调度协议**：维护持久的 LEARNING-ISSUE TREE，仅当未解决问题发生 open/change/reopen 时触发聚焦 review；与 Fixed-Window SSL 及固定步长触发的本质区别在于以“未解疑问”为驱动，避免后期沉默 review 造成的算力浪费。
4. **提供多模型对系统评估与过程可解释分析**：在 Qwen/DeepSeek/Kimi 上验证协议鲁棒性，并量化教学意图、学生响应行为与包修订幅度的统计关联；与仅有最终分数的评估工作的本质区别在于建立“指导→行为→迁移”的可追溯因果链路。

## 方法详解
- **双主体角色划分**：Student 为目标 solver 模型，独占 Docker 工作区与环境操作权，负责探索 anchor、生成证据并撰写最终 `.md` Skill 包；Educator 仅拥有只读权限（查看近期推理、工具调用结果与活体 Skill），可进行解释、挑战、诊断或保持沉默。
- **Fixed-Window SSL**：每隔 $g$（默认 $g=10$）次 Student tool call 触发一次 Educator review，审查本地证据与当前 Skill，提供下一条指导。Student 可澄清、质疑或采纳建议，自主决定何时结束当前 round。每轮结束后由 fresh solver 在重置 anchor 上测试以获取 verifier 反馈，驱动下一轮。
- **Issue-Guided SSL**：每条 Educator 课程附带学习目标与可观测验收标准，形成持久的学习问题节点。问题构成树状结构（父问题未解决时子问题为其后代；闭合后注意力返回最近未解祖先）。仅当未解决问题状态跃迁时触发 review，Educator 仅接收活跃问题链、最新 revision 与局部工具调用记录，大幅压缩上下文。
- **迁移效用定义**：$U_f(\mathcal{K}; \pi_S) = \mathbb{E}_{x \sim \mathcal{D}_f}[r_f(\pi_S(x, \mathcal{K}))]$，其中 $\mathcal{K}$ 为冻结 Skill 包，$\pi_S$ 为 fresh solver，$r_f$ 为私有 verifier 评分；评估时启用 3 次独立 acquisition run 取 family-macro Avg@3。

## 实验与结果
- **数据集**：SKILLTRANSFERBENCH，涵盖 35 个任务族、408 个可执行案例（6 大域：Systems, Services, Data Processing, Scientific Computing, Finance, Office Automation）；主队列 22 族、155 个 held-out 案例。
- **基线**：No Skill、Expert Authoring、Post-hoc Teaching、改编版 Trace2Skill、改编版 SKILL-KD。
- **主要结果（Qwen 主队列）**：Fixed-Window SSL 效用 **0.801**（较 No Skill 0.723 提升 **+7.75pp**）；Issue-Guided SSL 效用 **0.805**（+8.18pp），两项均位列前二。Fixed-Window SSL 较 Trace2Skill (+3.99pp) 与 SKILL-KD (+4.99pp) 优势显著，95% CI 分别为 [1.45, 6.49] 与 [1.79, 8.48]。
- **跨模型一致性**：在 DeepSeek 与 Kimi 共享 lite 子集（10 族/77 案例）上，两种 SSL 协议均保持前二，Fixed-Window SSL 相对 No Skill 分别增益 +3.08pp 与 +8.24pp。
- **效率突破**：Issue-Guided SSL 在 Qwen 主队列实现 0.805 效用，Educator token 减少 **61.7%**，总 acquisition token 减少 **16.9%**（6.89M → 5.72M），迁移收益效率比为 Fixed-Window SSL 的 **2.75 倍**。
- **过程分析**：教学需求随学习推进从首轮 73.0% 降至第三轮 16.1%；指导后 Student 下一调用写/编辑文件的概率提升 **23.9pp**，10 次调用内 Skill 包变更概率提升 **38.6pp**。

## 相关工作脉络
- **Trace2Skill / SKILL-KD**：基于 Teacher-Student 轨迹对比或对比蒸馏生成 Skill 补丁；SSL 将指导前置至实时探究阶段，保留学生自主验证权，避免离线蒸馏丢失上下文假设检验。
- **SkillLearnBench (Teacher Feedback)**：专家在失败后提供参考 Skill 引导修订；SSL 强调在探究进行中动态介入，而非事后补救，使反馈能重新定向证据收集路径。
- **Self-Refine / Reflexion**：依赖模型自身生成的自我反馈迭代；SSL 引入外部 Educator 进行针对性诊断与假设挑战，并维持问题树结构以实现跨轮次知识累积。
- **HASP / SkillHone**：侧重教师审阅学生编写的 skill 程序或持久化修订历史；SSL 通过 issue 触发机制实现“按需聚焦 review”，直接优化监督成本与迁移收益的帕累托前沿。
- **SkillsBench / ContinualSkillBench**：主要评估 Skill 效用或有序任务流；SKILLTRANSFERBENCH 引入公开 variation spec 与 fresh solver 冻结评估，
