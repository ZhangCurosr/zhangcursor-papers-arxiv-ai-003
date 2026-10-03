---
title: "SEARCH-SHAPES-CONCLUSIONS-AUDITING-EVI-DENCE-SELECTION-BIAS"
source: https://arxiv.org/pdf/2609.39026v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-03 08:46:03"
field: "LLM Agent 可信评估与因果审计"
keywords: ["Deep Research Agents", "Evidence Selection Bias", "Causal Inference", "Off-policy Evaluation", "Adaptive Sampling", "Inverse Probability Weighting"]
innovations: ["提出 CESS 方法，结合预搜索证据预测与序列逆概率加权校正，实现对候选池目标的设计无偏估计；证明校正估计的跨策略对比无法识别搜索政策效应，确立干预实验的必要性"]
benchmarks: ["MS2", "PERSPECTRUM", "Open Deep Research public agent"]
---

# 论文速读：SEARCH SHAPES CONCLUSIONS: AUDITING EVIDENCE SELECTION BIAS IN DEEP RESEARCH AGENTS

## 一句话总结
本文针对 Deep Research agents 的自适应搜索过程中产生的**证据选择偏差（Evidence Selection Bias）**问题，提出了因果证据选择校正方法 CESS（Causal Evidence Selection Correction），利用搜索轨迹中记录的文档选择概率与到达概率对候选池平均证据方向进行无偏估计；同时证明校正目标与搜索政策效应归因属于不同任务，需分别通过干预分析来度量。

## 研究问题与动机
- **自适应搜索导致证据样本不具代表性**：Deep Research agent 的早期发现会引导后续查询方向、文档选择与停止决策，使得实际读取的文档形成有偏的选择样本，而非候选证据池的代表性抽样。
- **现有评估忽略了候选池代表性**：已有工作主要评估引用正确性（citation correctness）、事实准确性（factual correctness）和报告效用（report utility），但这些指标可能在证据集合片面时仍表现良好，掩盖了结论偏差风险。
- **校正估计与政策归因不可兼得**：理论上证明了设计为在任何搜索策略下都能无偏估计候选池目标的估计量，其跨策略对比无法识别搜索政策改变证据阅读的因果效应，二者存在本质不相容性。
- **短预算下方差激增**：当搜索轮次较少时，原始逆概率加权估计的方差极大，需要稳定的收缩机制保障实用性。

## 核心贡献（创新点）
1. **定义了候选池目标与审计框架**：将证据结论形式化为固定候选池中各文档证据得分的等权重平均（candidate-pool target），并与已打开文档平均（Opened Mean）严格区分，为证据选择偏差提供了可审计的量化目标。
2. **提出 CESS 校正方法**：结合预搜索的文档证据预测与轨迹中记录的序列选择/到达概率，构建双重稳健形式的序列校正估计量；引入收缩系数稳定短预算估计，并在部分文档不可达时提供有界区间估计。
3. **证明校正-归因不相容定理**：形式化证明任何满足设计无偏性（design unbiasedness）的候选池目标估计量，其跨策略对比的期望恒为零，因而无法同时识别搜索政策的Opened-evidence效应，确立了干预实验的必要性。
4. **大规模干预实验验证分离性**：在 4,800 条 LLM agent 轨迹（覆盖 2×2 选择-停止交叉干预）上验证了 Theorem 2，证实 CESS 估计与干预效应的 rank correlation 仅 0.269/0.236，两者为互补而非可互换的审计指标。

## 方法详解
- **候选池目标定义**：给定候选文档池 $\mathcal{D}(x) = \{d_1, ..., d_N\}$，每篇文档 $d_i$ 的证据得分为 $Y_i \in [-1, 1]$，目标量为 $\theta^* = \frac{1}{N}\sum_{i=1}^{N} Y_i$，等权重聚合。
- **序列因果结构模型**：用 Pearl 式结构方程刻画搜索过程——$Q_t = f_Q(H_t, U_t^Q)$（查询生成）、$E_t = f_E(Q_t, G_t, \mathcal{D}, U_t^E)$（检索暴露集）、$A_t = f_A(H_t, Q_t, E_t, U_t^A)$（文档选择）、$C_t = f_C(H_{t+1}, U_t^C)$（停止决策）。
- **原始估计量（Sequential DR）**：
  $$\widehat{\theta}_{\mathrm{DR}}^{\mathrm{raw}} = \overline{m} + \frac{1}{K}\sum_{t=1}^{T}\frac{Y_{J_t} - m_\phi(X_{J_t})}{N\,\widehat{p}_t\,\widehat{s}_t}$$
  其中 $\overline{m}$ 为全池预测均值，$p_t$ 为文档选择概率，$s_t$ 为到达第 $t$ 轮的累积概率。分子为预测残差，分母中的逆概率项分别校正文档选择偏差与提前停止偏差。
- **Assumption 1（登录设计与重叠）**：要求预测在搜索前固定、日志概率等于真实控制器概率、每篇候选文档在每个可达历史下选择概率为正、每轮有正概率被到达。
- **Theorem 1（设计无偏性）**：在上述假设下，$\mathbb{E}[\widehat{\theta}_{\mathrm{DR}}^{\mathrm{raw}}|\mathcal{D}] = \theta^*$，即使 outcome model 设定有误也成立。
- **收缩修正**：
  $$\widehat{\theta}_{\mathrm{CESS}} = \Pi_{[-1,1]}\big[\overline{m} + \lambda(\widehat{\theta}_{\mathrm{DR}}^{\mathrm{raw}} - \overline{m})\big]$$
  系数 $\lambda \in [0,1]$ 控制修正量占比，短预算下自动衰减，$\lambda=0$ 退化为纯预测。
- **有限支持区间估计**：当部分文档选择概率为零时，令 $\rho = 1 - |\mathcal{D}_+|/N$ 为不可达质量，给出 $\theta^*$ 的尖锐边界 $[(1-\rho)\theta_+ + \rho L,\; (1-\rho)\theta_+ + \rho U]$。
- **Theorem 2（不变性-归因不相容）**：若 $\widetilde{\theta}(g)$ 对任意支持策略均满足设计无偏性，则 $\mathbb{E}[\widetilde{\theta}(g)-\widetilde{\theta}(g')|\mathcal{D}]=0$，校正对比无法度量 $\tau_O(g,g')$。

## 实验与结果
- **数据集**：MS2（200 个医疗系统综述问题，3,169 篇候选研究）、PERSPECTRUM（36/267 个非医学主题，支持/反对标签化证据）。
- **基线方法**：Tuning-set Mean、Outcome Regression（OR）、Opened Mean、Sequential IPW、Sequential Doubly Robust（无收缩）。
- **MS2 主结果（Table 1，两模型平均）**：CESS 的 MAE 为 0.1583，相对 Opened Mean（0.1743）降低 9.2%；ranking sensitivity 0.1481，相对 Opened Mean（0.2445）降低 39.4%；worst-ranking MAE 0.2165，降低 14.2%。
- **匹配精度下的稳定性比较（Table 2）**：28 个 dataset-model-budget-prediction 设置中，10 个 95% 区间支持 CESS 敏感性更低，0 个支持 CESS 更高，表明增益来自概率校正而非泛化收缩。
- **短预算增益（Figure 3）**：单轮搜索后 Qwen 的 MAE 从 0.3687 降至 0.2157，OLMo 从 0.3522 降至 0.2090；四轮时相对降幅分别为 9.1% 与 15.8%。
- **公开 Agent 迁移（Open Deep Research，PERSPECTRUM，Table 3）**：216 条轨迹，CESS 相对 Opened Mean 将 MAE 从 0.5727 降至 0.2287（↓60.1%），ranking sensitivity 从 0.5849 降至 0.0746（↓87.2%），方向错误率从 0.4907 降至 0.3102（↓18.1pp）。
- **干预实验验证分离性（Table 4）**：4,800 条轨迹的 2×2 选择-停止交叉干预显示，选中效应 ICC 为 0.679/0.685（中等可重复），但与 CESS 估计差值的 rank correlation 仅 0.269/0.236，与 Theorem 2 预测一致。
- **仿真排序干预（Figure 2a）**：未收缩校正将 two-ranking 差异从 0.7478 降至 0.0014（↓99.8%）。
- **MS2 仿真排名敏感性（Figure 2b）**：Task-wise CESS 在 Qwen 上从 0.0415 降至 0.0185（↓55.4%），在 OLMo 上从 0.0391 降至 0.0076（↓80.6%）。

## 相关工作脉络
1. **Deep Research 系统与报告评估**：STORM、Co-STORM（多视角文章组织）、DeepResearcher（强化学习驱动多步搜索）、HypoSearch（探索替代假设）；DREAM、DeepFact、ReportLogic、FActScore 等基准评估报告质量、引用、事实与逻辑支持——这些工作改进证据获取或评估已完成报告，但不从单条自适应轨迹估计候选池目标。
2. **逆概率加权与双重稳健估计**：Cassel et al.（1976）、Robins et al.（1994）、Dudík et al.（2011）奠定了 off-policy 评估基础；本文将其扩展至含自适应停止与连续历史依赖的序列文档选择场景。
3. **主动测试（Active Testing）**：Kossen et al.（2021）研究自适应标签获取下的目标估计；本文面临更复杂的挑战——早期证据影响后续查询生成、检索排名、文档选择与停止决策的全链路依赖。
4. **自适应实验中的置信区间**：Hadad et al.（2021）处理适应性实验中的区间估计；本文在此基础上进一步区分"池目标估计"与"政策效应归因"两个不同审计问题，并提出干预设计的配套方案。
5. **证据/引用评估**：Gao et al.（2023）（引用生成）、Min et al.（2023）（FActScore 事实粒度评估）、Du et al.（2026）（DeepResearch Bench）聚焦局部引用正确性或总体报告质量，忽略候选池代表性这一整体偏差维度。
6. **选择偏差与大数据悖论**：Russo & Zou（2016）、Meng（2018）从信息论与统计范式角度讨论自适应数据收集的偏差；本文将上述理论视角具体化到 LLM agent 的多轮搜索轨迹审计场景。

## 局限性与未来方向
- **依赖日志概率的精确性**：CESS 的理论无偏性要求选择的精确 $p_t$ 与到达概率 $s_t$；在实际部署中这些概率可能由模型估计而非控制器直接记录，预测误差会引入偏差（Appendix G 显示概率估计误差影响较小但存在）。
- **重叠假设的限制**：Assumption 1 要求每篇候选文档在每个可达历史下均有正选择概率；当 policy 对部分文档赋零概率时，只能得到区间估计而非点估计。
- **干预实验成本较高**：验证政策效应需执行完整的 2×2 交叉干预（如 4,800 条轨迹），在实际 agent 审计中可能难以承担。
- **固定候选池假设**：当前框架假定候选池在评估前已预定义，未处理动态扩展或外部网页搜索中候选池未知的场景。
- **未来方向**：① 扩展到动态候选池与 online 审计场景；② 探索无需精确日志概率的近优校正方案；③ 将 CESS 集成到 agent 的 self-audit 回路中实现实时偏差检测；④ 研究多目标聚合规则（如质量加权、来源去重加权）下的泛化。

## 研究启发与可借鉴点
1. **目标分解审计范式**：将"池目标估计"与"政策效应归因"严格分离的审计框架具有普适价值，可迁移至任何 adaptive data collection 场景（如推荐系统的排序偏差评估、临床研究的 adaptive trial 分析）。
2. **序列双重稳健校正的工程实现**：$1/(p_t s_t)$ 的联合逆概率修正、先截断后收缩（或相反）的实现细节、task-wise vs model-wise 系数选择策略，均可复用于其他在线决策系统的 offline evaluation。
3. **排名敏感性作为稳定性指标**：通过固定候选池但反转证据排序（supporting-first vs opposing-first）来量化估计对输入顺序的敏感程度，是一种轻量且信息丰富的鲁棒性评测手段，可融入 agent 基准测试协议。
4. **配对干预设计的因果识别**：2×2 交叉设计（uniform vs agent selection × fixed vs adaptive stopping）配合相同预干预状态与随机数的 paired 执行，可在不完全切断反馈链路的前提下分离选择与停止效应，为 agent 可解释性分析提供模板。
5. **有限支持的区间估计替代方案**：当 overlap 不足时退化为有界区间而非强行点估计，这一保守策略在高风险决策审计（如医疗、法律）中具有实用价值，避免虚假精度。

## 关键术语表
- **Candidate-pool target ($\theta^*$)**：预定义候选文档池中所有文档证据得分的等权重平均值，作为审计目标而非对 open web 或绝对真相的断言。
- **Opened Mean**：仅对被 agent 实际打开的文档证据得分求平均，是未校正的朴素估计量。
- **Ranking sensitivity**：同一候选池在 supporting-first 与 opposing-first 两种排序下估计值的绝对差，量化估计对输入顺序的依赖程度。
- **CESS（Causal Evidence Selection Correction）**：结合预搜索证据预测与序列逆概率加权（校正选择与停止）的双重稳健估计量，经收缩稳定后输出候选池目标估计。
- **Design unbiasedness**：在精确日志概率与重叠假设下，估计量的期望等于真实候选池目标的性质。
- **Invariance-attribution incompatibility**：任何设计无偏的目标估计量的跨策略差值期望为零，故无法同时识别搜索政策的 opened-evidence 效应。
- **Intervention effects ($\Delta_{sel}, \Delta_{stop}, \Delta_{int}$)**：通过 2×2 交叉干预（uniform/agent selection × fixed/adaptive stopping）度量的选择、停止及其交互效应。
- **Limited support / inaccessible mass ($\rho$)**：当部分候选文档在日志设计下选择概率为零时，其累积质量 $\rho$ 决定了点估计不可识别，退化为宽度为 $2\rho$ 的区间估计。

## 可复现要素
- **数据集**：MS2（DeYoung et al., 2021）、PERSPECTRUM（公开 benchmark），论文在 Appendix A.1 给出证据分数构造细节；MS2 使用 200 个 test questions，PERSPECTRUM 使用 36 个 evaluation topics。
- **代码/权重开源情况**：论文 REPRODUCIBILITY STATEMENT 声明提供了主要因果变量、干预设计、测量方式、模型、基准、系数选择与实验流程的完整描述；公开 agent（Open Deep Research）使用固定 commit `1b7d2e80db9faa586165c60e09096dbbfd483a64`；附录声明 released implementation 将报告 data split、tuning criterion 与系数计算方式——但全文未明确给出代码仓库 URL，以官方发布为准。
- **关键超参**：MS2 primary configuration 的 $\lambda$ 分别为 Qwen2.5-32B 的 0.5484、OLMo3-7B 的 0.5649；public-agent 与 ten-round PERSPECTRUM 统一使用 $\lambda=0.5$；five-round PERSPECTRUM variant 使用 0.20 收缩系数；score clipping 范围 $[-1, 1]$。
- **评估协议**：question-level bootstrap（10,000 resamples）、paired trajectory comparison、ICC(1,3) 重复性度量；所有方法设置在使用测试问题证据得分前固定。
