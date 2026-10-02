---
title: "Teacher-Student-Gaps-Are-Not-Enough-Outcome-Guided-On-Policy"
source: https://arxiv.org/pdf/2609.35319v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 08:14:15"
field: "多轮智能体蒸馏"
keywords: ["on-policy distillation", "multi-turn agents", "knowledge distillation", "teacher-student gap", "outcome-guided calibration", "autonomous agents", "ALFWorld", "WebShop", "ScienceWorld"]
innovations: ["揭示监督-收益错配现象：大gap不等价于高指导收益，教师伤害率超过拯救率", "提出OG-OPD：通过配对学生续跑的最终任务结果校准轨迹相对权重，选择性强化真正有益的教师监督", "仅用<1%轮次的稀疏校准获得3.6-17.7pp的显著提升，无需将教师响应作为额外蒸馏目标"]
benchmarks: ["ALFWorld", "ScienceWorld", "WebShop"]
---

# 论文速读：Teacher-Student-Gaps-Are-Not-Enough: Outcome-Guided On-Policy Distillation for Multi-Turn Autonomous Agents

## 一句话总结
论文揭示了多轮智能体蒸馏中"教师-学生gap大≠指导有价值"的监督-收益错配现象，提出了**OG-OPD（结果导向的在策略蒸馏）**方法——通过配对学生续跑对比最终任务结果，有选择性地加强对当前学生真正受益的轮次的教师监督，在ALFWorld、ScienceWorld和WebShop三个基准上均超越现有最强基线。

## 研究问题与动机
- **现有OPD方法的局限**：当前多轮自主智能体的在策略蒸馏（OPD）普遍将大token级分布gap视为强干预信号，认为gap越大越需要教师纠正；但局部gap仅反映当前轮的差异，无法捕捉学生对后续环境的交互能力。
- **监督-收益错配**：实证发现大gap可能是良性的（5.4%决策中师生均成功），也可能是有害的（10.8%的教师动作导致失败 vs 8.8%的教师拯救）；小gap中仍有7.6%为结果关键决策。"向教师靠近"未必提升当前学生的最终任务收益。
- **教师-only完成现象**：约20.6%的大gap决策中，教师能完成任务但当前学生从同一状态几乎无法成功，说明指导价值取决于当前学生的续跑能力，而非教师本身的能力。
- **有效监督应关注结果**：在多轮环境中，当前轮动作决定后续观察与状态，误差会累积传播，因此监督决策必须同时考虑"在哪干预"和"当前学生后续能达成什么结果"。

## 核心贡献（创新点）
1. **揭示监督-收益错配现象**：首次在多轮智能体OPD中系统验证大gap不等价于高指导收益，教师伤害率（10.8%）甚至超过教师拯救率（8.8%），为重新审视干预策略提供了实证基础。
2. **提出OG-OPD框架**：构建轨迹相对权重+结果校准的两阶段机制，仅在学生原始动作失败但教师动作成功时才强化该轮监督，避免了盲目跟随教师的错误引导。
3. **与FutureBridge-OPD的本质区别**：前者用教师偏好token比例评估指导价值并将接受教师响应作为额外蒸馏目标；OG-OPD用配对学生续跑的**最终任务结果**判断是否强化学生原始响应的监督，教师响应仅作校准不进入训练目标。
4. **广泛验证与一致提升**：在三个基准（ALFWorld/ScienceWorld/WebShop）和两组不同师生配置下均取得最高成功率，较vanilla OPD提升3.6–17.7个百分点，较最强基线提升最多7.0个百分点。

## 方法详解
**整体框架**：OG-OPD在vanilla OPD基础上仅修改轮次级权重，训练目标仍为学生自身轨迹上的token。

**Step 1 — 轨迹相对权重（Trajectory-Relative Weighting）**：
- 用每轮token级gap幅度 $\chi_k = \frac{1}{|M_k|}\sum_{j\in M_k}|\psi_{k,j}|$ 的对数 $\nu_k=\log(\varepsilon_0+\chi_k)$ 及其相对前轮变化 $v_k$ 衡量监督信号。
- 以轨迹内首个有效轮 $k_0$ 为参考，定义预校准权重：
  $$\beta_k = \min\{\beta_{\max},\ \exp(\nu_{k_0}-\nu_k)\},\quad \beta_{k_0}=1$$
  即gap高于参考则降权、低于参考则升权（上限$\beta_{\max}$），使整条轨迹的监督尺度趋于一致。

**Step 2 — 候选轮选择（Candidate Selection）**：
- 候选轮需满足：$\nu_k \geq \vartheta_\nu$（gap超过阈值）且（$k=0$或$v_k>0$且$v_k\geq\vartheta_v$，即出现突增）。
- 阈值$\vartheta_\nu,\vartheta_v$按batch分位数确定；按时间顺序检查，选第一个回放合法且教师动作与学生不同的轮$k^\star$做配对评估。

**Step 3 — 结果校准（Outcome-Based Calibration）**：
- 恢复状态$s_{k^\star}$和历史$c_{k^\star}$，分别以**学生原响应**$z^S_{k^\star}$和**教师响应**$z^T_{k^\star}$替代，用同一冻结学生策略$p_\circ$生成两条续跑轨迹$\tilde{\zeta}^S,\tilde{\zeta}^T$。
- 计算$\Delta_\zeta = \mathsf{S}(\tilde{\zeta}^T)-\mathsf{S}(\tilde{\zeta}^S)\in\{-1,0,1\}$，仅当$\Delta_\zeta=1$（学生原动作失败但教师动作成功）时产生正增量：
  $$\mathsf{g}_\zeta = \mathsf{S}(\tilde{\zeta}^T)(1-\mathsf{S}(\tilde{\zeta}^S)) = \mathbb{I}[\Delta_\zeta=1]$$
  $$\gamma_\zeta = \mathsf{g}_\zeta[\beta_\uparrow - \beta_{k^\star}]_+,\quad \omega_{k^\star}=\beta_{k^\star}+\gamma_\zeta$$
- 最终训练目标：$\mathcal{L}_{\text{OG-OPD}}=\frac{1}{Z}\sum_{\zeta,k}sg[\omega_k]Q_k(\theta;\zeta)$，其中$Q_k$为裁剪策略代理损失，$sg[\cdot]$在微分时固定权重与$\psi$。

**关键设计原则**：教师响应和配对续跑**仅用于校准**，不作为额外蒸馏目标；梯度仅来自学生原始轨迹上的token。

## 实验与结果
**基准与模型配置**：
- **ALFWorld**（household tasks，Seen/Unseen/Hard三子集，共395任务）、**ScienceWorld**（1,308 canonical tasks）、**WebShop**（100 held-out sessions）
- 两组配置：(1) Qwen3-32B teacher → Qwen3-1.7B student；(2) Qwen3-8B-RL teacher → Qwen3-4B student（RL教师用GiGPO在对应benchmark上训练）
- 评估指标：成功率SR(%)、任务分Score(0–100)、平均交互轮数Rounds

**主要结果（Table 1，32B→1.7B）**：
| 方法 | ALFWorld Overall SR | ScienceWorld SR | WebShop SR |
|---|---|---|---|
| Vanilla OPD | 20.1±2.1 | 13.0±1.4 | 28.0±2.6 |
| FutureBridge-OPD（最强基线） | 21.4±2.1 | 12.8±0.7 | 38.7±1.5 |
| **OG-OPD** | **24.2±1.5** | **16.6±0.6** | **45.7±1.2** |

- OG-OPD较vanilla OPD提升：ALFWorld +4.1pp、ScienceWorld +3.6pp、**WebShop +17.7pp**
- 较最强基线FutureBridge-OPD提升：ALFWorld +2.8pp、ScienceWorld +3.8pp、**WebShop +7.0pp**

**主要结果（Table 2，8B-RL→4B）**：
- ALFWorld Overall SR: OG-OPD 64.2% vs FutureBridge-OPD 62.7%
- WebShop SR: OG-OPD 56.3% vs FutureBridge-OPD 54.7%

**训练动力学与效率**：
- OG-OPD在WebShop上Teacher NLL下降更快、Parsed恢复更早（Figure 4）
- 候选轮约占合法轮的1/4，实际获额外加权的轮<1%，高效聚焦
- 每步训练开销：ScienceWorld 1.42× vanilla OPD，WebShop 1.48× vanilla OPD（Table 6）

**消融实验（Table 3，Qwen3-32B→1.7B）**：
- 去掉结果校准：ScienceWorld SR 3.8%↓，WebShop SR 32.0%↓
- 不验证直接加权：ScienceWorld 14.9% vs OG-OPD 16.6%；WebShop 35.0% vs 45.7%
- 去掉轨迹相对权重：ScienceWorld 5.5%↓，WebShop 33.3%↓
- 随机采样候选轮：ScienceWorld 15.1%↓，WebShop 37.7%↓
→ 各组件互补，缺一不可

## 相关工作脉络
1. **OPD基础**：Agarwal et al. (2024) 提出OPD核心框架，在学生自身轨迹上施加密集教师监督；本文在此基础上解决多轮智能体中的监督质量筛选问题。
2. **TCOD**（Wang et al., 2026）：通过 temporal curriculum 调节学生rollout深度（F2B/B2F），关注 rollout 构造但不在token级做结果导向的权重校准。
3. **Guided-OPD**（Li et al., 2026）：交错师生轮次并逐步降低教师干预概率，属于turn-level guidance调度，不涉及配对续跑的结果验证。
4. **FutureBridge-OPD**（Chen et al., 2026）：用大gap定位候选轮，以教师偏好token比例评估指导价值，并将接受的教师响应作为额外蒸馏目标——与OG-OPD的核心差异在于验证信号（token比例 vs 最终任务结果）和训练接入方式（额外目标 vs 原始轨迹加权）。
5. **轨迹相对权重先验**（Zhong et al., 2026, SOD）：将gap幅度视为可靠性信号做降权，但未区分"被降权的轮次是否恰恰是教师指导能帮到学生的"。
6. **Step-level credit assignment**（Kazemnejad et al., 2025; Ma et al., 2026）：用Monte Carlo估计或future-KL做step级信用分配，侧重RL训练而非蒸馏场景。

## 局限性与未来方向
- **单次配对评估的方差**：训练时每轮仅用1对续跑结果判断是否加权，而实证分析用5次匹配试验；单次判断可能受随机性影响，未来可引入多次采样的期望估计。
- **额外环境执行的成本**：OG-OPD每步需额外运行配对续跑，训练耗时增加约40–48%；在更长的交互 horizon 或更复杂环境中，这一开销可能进一步放大。
- **仅评估最终二值结果**：校准信号为$\mathsf{S}(\zeta)\in\{0,1\}$，忽略了中间过程质量（如效率、路径合理性），可能错过部分有益但非决定性的指导。
- **阈值设定依赖分位数**：候选选择的分位数阈值$q_\nu,q_v$在当前实验中取0.5/0.6，在不同任务分布下的泛化性有待验证。
- **未探索动态阈值**：当前阈值在整个训练过程中固定，可考虑随训练进度自适应调整。

## 研究启发与可借鉴点
1. **结果导向的校准范式可迁移**：配对续跑+最终结果对比的思想可推广到其他序列决策蒸馏场景（如代码生成、对话管理），只要存在可执行的中间状态 replay 即可复用。
2. **监督-收益错配的普遍性**：大gap≠高价值的发现提醒我们，在任何teacher-student蒸馏中，**干预信号的信噪比需要结果验证**，不能仅依赖局部差异度量做决策。
3. **轨迹相对权重设计精巧**：以首有效轮为参考、对数缩放、上界截断，使整条轨迹的监督尺度归一化——可作为通用预处理模块嵌入其他OPD变体。
4. **效率与效果的权衡策略**：仅对<1%的轮次进行昂贵的配对评估，却获得显著增益；这种"稀疏但精准"的校准策略值得在长序列任务中推广。
5. **可与本团队方向结合的机会**：若团队关注多轮Agent训练，可将OG-OPD的结果校准模块与RAG/工具调用场景结合，用任务完成率替代简单二值SR，评估教师在工具选择/查询策略上的指导价值。

## 关键术语表
- **On-Policy Distillation (OPD)**：在学生自身生成的轨迹上施加密集教师监督的蒸馏方法，减少训练-推理分布偏移。
- **监督-收益错配（Supervision-Benefit Mismatch）**：教师-学生token级gap大并不意味着教师指导对当前学生更有价值，二者不存在单调对应关系。
- **轨迹相对权重（Trajectory-Relative Weighting）**：以轨迹内首个有效轮的gap为参考，按对数比分配各轮预校准权重，使监督尺度跨轮一致。
- **结果校准（Outcome-Based Calibration）**：通过配对续跑对比最终任务结果，仅在学生原动作失败而教师动作成功时才额外强化该轮监督。
- **教师拯救/伤害（Teacher Rescue/Harm）**：拯救指教师动作成功而学生原动作失败；伤害指教师动作失败而学生原动作成功。
- **教师-only完成（Teacher-only Completion）**：教师能完成任务但当前学生从同一后续状态几乎无法成功的情况。
- **配对学生续跑（Paired Student Continuations）**：在同一状态/历史下，分别以学生原响应和教师响应为起点，用同一冻结学生策略继续交互直至终止。
- **Reverse KL Distillation**：本文采用的蒸馏目标，最小化学生分布相对教师分布的KL散度，鼓励学生覆盖教师高概率区域。

## 可复现要素
- **数据集**：ALFWorld（Shridhar et al., 2021）、ScienceWorld（Wang et al., 2022）、WebShop（Yao et al., 2022），均为公开benchmark
- **模型**：Qwen3系列（32B/8B-RL/4B/1.7B），论文声明提供源码在supplementary material中；RL教师用GiGPO（Feng et al., 2025）在各自benchmark上训练
- **关键超参**：$\beta_{\max}=1.2$，$\beta_\uparrow=1.5$，$q_\nu=0.5$，$q_v=0.6$，候选检查上限2轮/轨迹，配对评估1对/轨迹；训练步数200，3个随机种子取均值
- **硬件**：8×A800 GPU
- **开源声明**：论文明确"provide the source code in the supplementary material accompanying this submission"
