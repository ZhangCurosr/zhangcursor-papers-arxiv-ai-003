---
title: "Test-Time-Agent-Evolution-for-Long-Horizon-Legal-Reasoning"
source: https://arxiv.org/pdf/2610.08138v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-07 22:26:43"
---

# 论文速读：Test-Time-Agent-Evolution-for-Long-Horizon-Legal-Reasoning

## 一句话总结
论文提出 **TALA**，一种免训练的测试时智能体演化框架，通过 **Test-Time Memory Evolution (TME)** 实现跨案例经验的检索、上下文适配与有界合并，并结合 **Rubric-Aligned Collaboration (RAC)** 按角色量规校验并修正候选动作，从而在不更新底层模型参数的情况下持续提升长周期法律推理的可靠性、跨阶段一致性与执行鲁棒性。

## 研究问题与动机
1. **静态部署策略难以匹配动态法律环境**：现有法律智能体多依赖部署前预定义的规划/推理/工具调用流程，策略一旦定型便难以随案件事实、证据结构与程序阶段的持续演变进行自适应调整。
2. **案例异构性导致经验直接迁移失效**：即便同类罪名案件，在事实细节、证据链条与程序背景上亦存在显著差异，简单复用 prior experience 易引入错配或误导性指导。
3. **跨角色/跨阶段决策缺乏全局一致性**：长周期法律推理天然耦合多角色（律师、当事人、法官等）与多程序阶段，前期局部合理的决策可能破坏后续阶段的逻辑连贯性，造成误差沿轨迹累积。
4. **逐案微调不具备实际部署可行性**：法律案件持续到达，为每个新案重新微调或强化智能体成本过高，亟需探索仅利用部署时自然产生的信号、无需更新模型参数即可持续进化的适应机制。

## 核心贡献（创新点）
1. **提出免训练的测试时智能体演化范式**：将长周期法律智能体问题从“单次推理优化”转向“持续部署中的在线适应”，形式化刻画了案例异构性与跨角色耦合两大核心挑战，填补了现有 test-time adaptation 工作对非平稳交互场景覆盖的空白。
2. **Test-Time Memory Evolution (TME) 机制**：设计检索-适配-合并的记忆演化流水线，利用 BGE 编码器与余弦相似度从历史案例中提取可复用经验，通过主模型自身将其适配到当前事实与程序上下文，并在固定记忆预算下执行全局语义最相似配对合并，有效抑制冗余与噪声累积。
3. **Rubric-Aligned Collaboration (RAC) 机制**：为不同法律角色定义行为与程序量规集合，对候选动作进行逐条合规校验，违规时将对应量规项作为修订信号交由同一智能体重生成，从机制上阻断局部不一致向后续阶段传播。
4. **在双基准、五 Backbone 上验证强泛化与可解释性**：在 J1-EVAL 与 LegalWorld 上跨 Qwen3 系列、InternLM3、GLM-4 均取得一致提升；消融与 Case Study 证明 TME 与 RAC 在“知识连续性”与“行为对齐”上显著互补，整体框架具备中等测试时开销与高任务完成率。

## 方法详解
- **问题设定**：给定预训练法律智能体 $\pi_\Theta$，在测试时依次处理来自 $P_{\text{test}}$ 的 $N$ 个法律案例，不更新 $\Theta$。每个案例构成多轮交互轨迹 $\tau^{(i)} = \{(s_t, r_t, \mathcal{H}_t, a_t, o_{t+1})\}_{t=1}^{T_i}$，其中 $s_t$ 为当前任务状态，$\mathcal{H}_t$ 为历史交互记录，$r_t$ 为当前活跃角色。
- **Test-Time Memory Evolution (TME)**：
  - **Memory Retrieval**：每轮构造查询 $q_t^{(i)}$，用 BGE 编码器 $f_{\text{BGE}}(\cdot)$ 计算与记忆库 $\mathcal{M}_i$ 中各条目 $m_j$ 的余弦相似度 $\rho_j$，保留 $\rho_j \geq \delta$（$\delta=0.5$）的 Top-K 条目 $\mathcal{R}_t^{(i)}$。
  - **Contextual Memory Adaptation**：将检索集 $\mathcal{R}_t^{(i)}$、当前历史 $\mathcal{H}_t^{(i)}$ 与状态 $s_t^{(i)}$ 输入同一智能体，生成适配记忆 $\hat{m}_t^{(i)} = \pi_\Theta(\mathcal{R}_t^{(i)}, \mathcal{H}_t^{(i)}, s_t^{(i)})$ 及初步动作 $\tilde{a}_t^{(i)}$。
  - **Memory Consolidation**：每完成一个案例，将轨迹蒸馏为简洁可复用条目存入 $\mathcal{M}_i$。当 $|\mathcal{M}_i| > C$ 时触发全局两两合并：选取 $\arg\max_{p<q} \sin_{\cos}(f_{\text{BGE}}(m_p), f_{\text{BGE}}(m_q))$ 的最相似对 $(p^*, q^*)$，用 $\pi_\Theta$ 合并为 $m_{p,q}^* = \pi_\Theta(m_{p^*}, m_{q^*})$ 并更新 $\mathcal{M} \gets (\mathcal{M} \setminus \{m_{p^*}, m_{q^*}\}) \cup \{m_{p,q}^*\}$；合并失败则删除最旧条目。
- **Rubric-Aligned Collaboration (RAC)**：
  - 为场景定义角色专属量规 $\mathcal{G} = \{\mathcal{G}_{\text{lawyer}}, \mathcal{G}_{\text{public}}, \mathcal{G}_{\text{judge}}, \dots\}$。
  - 对候选动作 $\tilde{a}_t^{(i)}$ 进行逐条校验：$v_{t,j} = \text{RubricCheck}(\tilde{a}_t^{(i)}, g_j) \in \{0,1\}$，整体合规性 $v_t = \prod_{j=1}^J v_{t,j}$。
  - 决策规则：若 $v_t=1$ 则直接采用；否则调用 $\pi_\Theta$ 结合违规量规进行修订，生成最终动作 $a_t^{(i)}$。该机制贯穿所有参与角色与整条轨迹，防止局部合法但全局冲突的决策蔓延。

## 实验与结果
- **数据集**：J1-EVAL（6 类交互法律场景，含 KQ/LC/CD/DD/CI/CR 及 Level-I/II/III 多维指标）与 LegalWorld（7 阶段民事诉讼长周期流程 LC→CD→DD→FIT→AD→AR→SIT）。
- **基线**：SourceOnly、ReAct、Plan-and-Solve、Plan-and-Execute、LawThinker、Reflexion。
- **Backbone**：Qwen3.5-4B、Qwen3-8B、Qwen3-32B、InternLM3-8B、GLM-4-9B。
- **主要结果**：
  - TALA 在 J1-EVAL 与 LegalWorld 上对 5 种 Backbone 均实现稳定提升，文档撰写（FOR/DOC）与法庭程序/说理指标（PFS/REA/LAW）增益最为显著。
  - 代表数字：Qwen3-32B 下 J1-EVAL CD-FOR 达 100.0、DD-DOC 53.5、CR-PFS 98.6；LegalWorld LC-IS 54.9、DD-FM 95.5。
  - **消融**：移除 TME 平均下降约 17.5（J1-EVAL）/ 13.0（LegalWorld）分；移除 RAC 平均下降约 15.3 / 9.4 分；SourceOnly 相对 Full 模型平均退化 21.3 分，证明两模块显著互补。
  - **效率与鲁棒性**：LegalWorld 中位耗时约 4.27 分钟/案，交互轮次与工具/Token 消耗保持在合理区间；在 GLM-4-9B 上 LegalWorld 任务完成率达 100%，而基线仅为 26%–72%。
- **结论**：T
