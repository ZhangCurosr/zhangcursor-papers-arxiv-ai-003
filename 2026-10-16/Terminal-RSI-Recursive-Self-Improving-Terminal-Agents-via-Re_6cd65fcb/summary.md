---
title: "Terminal-RSI-Recursive-Self-Improving-Terminal-Agents-via-Re"
source: https://arxiv.org/pdf/2610.22192v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-04 10:31:22"
field: "长程 Terminal Agent 强化学习"
keywords: ["terminal agent", "recursive self-improvement", "reinforcement learning", "GAE credit assignment", "reward shaping", "long-horizon agent"]
innovations: ["递归 RSR-RPR 交替框架与 critic 继承 handoff", "Turn-Boundary Skip-Observation GAE（TB-GAE）turn 级信用分配", "FDR→SPR 两阶段奖励演化与固定轨迹离线重评分"]
benchmarks: ["Terminal-Bench 2.1", "Terminal-Bench 3.0", "Terminal-Bench 4.0", "Terminal-Bench Hard", "LHTB", "SWR100"]
---

# 论文速读：Terminal-RSI: Recursive Self-Improving Terminal Agents via Recursive Self-Rewrite and Policy-Refine

## 一句话总结
Terminal-RSI 提出了一种交替进行的递归自重写（RSR）与 critic 继承式递归策略精炼（RPR）框架，结合 Turn-Boundary Skip-Observation GAE（TB-GAE）信用分配机制与动态演化的奖励函数（FDR/SPR），使 Terminal 智能体从部分局部修复逐步进化到严格的终端验证。基于 Qwen3.8-27B 基础模型的五阶段训练后，Terminal-RSI-27B 在 Terminal-Bench 四合一平均成功率上达到 50.58%，较 Base 提升近 30 个百分点。

## 研究问题与动机
- **时间膨胀（Temporal Dilation）**：Token 生成时钟与环境变化时钟不同步，数百个 token 可能只对应一次 shell 动作，而数千个 observation token 不引入额外策略决策，导致标准 GAE 中的衰减因子的信用分配受文本长度而非交互轮次主导。
- **递归监督下的价值漂移**：RSR 重写改变了 actor 偏好的 action，进而改变了 critic 遇到的历史分布；后续 RL 再次改变分布，重置 critic 会丢弃已学结果结构，而直接传递未校准则继承系统性预测误差。
- **奖励对齐难题**：不同任务 family 的语义检查数量不同，失败轨迹可能累积大量通过检查却无法满足完整执行契约；单一全局缩放会过度强调检查项多的 family，纯比例进度奖励则在策略能实现严格成功时仍偏好不完整行为。
- **长程依赖链的严格完成困境**：LHTB 等基准强调局部进展不一定转化为最终完全解决，需要建立从成功探索到可复现执行、从延迟验证到因果归因的闭环。

## 核心贡献（创新点）
1. **递归 Actor-Critic 框架（Base → RSR-1 → RPR-1 → RSR-2 → RPR-2）**：在 RSR 阶段仅更新 actor（通过验证轨迹重写），critic 权重保持不变；RPR 阶段继承 critic 并进行四步critic-only 校准后再联合优化，本质区别在于保留了已学的回报结构，而非每次 actor 转换后重置。
2. **Turn-Boundary Skip-Observation GAE（TB-GAE）**：在 generated token 位置与 interaction turn 边界之间建立独立时钟，跨 turn 衰减因子为 $(\gamma_{\mathrm{turn}} \lambda_{\mathrm{turn}})^B$，与 observation 长度无关；保留精确跨 chunk 续接（需同时传输 $v_{b+1}$ 和 $A_{b+1}$），本质区别在于将 credit 分配单位从 token 级提升到 turn 级，消除文本长度驱动的虚假衰减。
3. **双阶段奖励演化（FDR → SPR）**：FDR 按 family 归一化进度（$P/D_f$），SPR 分离严格成功与有界的部分积分（$S + P/(P+D_f')$），离线重评分将失败奖励占比从 50.09% 降至 28.72%；本质区别是奖励函数随训练阶段动态演化，从探索导向过渡到严格成功导向。
4. **Warm-up Critic Model + 递归 critic handoff**：RSR-1 冻结 actor 后训练一个 SWR epoch 的准备 critic，RPR-2 直接继承 RPR-1 保留的 critic；本质区别是将 critic 初始化/迁移制度化，避免每次 actor 更新后重新拟合值函数。
5. **阶段自适应学习率课程**：RPR-1 actor rate $5\times10^{-7}$ / critic rate $1.5\times10^{-5}$，RPR-2 actor rate 降至 $1\times10^{-7}$，critic rate 不变；本质区别是通过五倍 actor 减速配合继承 critic 的二次精炼，防止策略突变破坏已校准的价值基线。

## 方法详解

**整体流程（五阶段序列）**：
- Base（Qwen3.8-27B）→ RSR-1（actor-only 重写）→ RPR-1（critic-inheriting 联合精炼）→ RSR-2（actor-only 重写）→ RPR-2（critic-inheriting 联合精炼，保留 SPR 分支）→ Terminal-RSI-27B

**（1）Verified Rewriting（RSR 损失）**：
对验证通过的轨迹 $\widetilde{\tau}_{\ell m}$（经过 verifier $\mathcal{V}_x$、leakage screening $q_{\mathrm{leak}}$、runbook planner 和 reviewer 过滤），构建干净策略数据集 $Q_k^{\mathrm{clean}}$，执行 SFT 风格的负对数似然损失：
$$\mathcal{L}_{\mathrm{RSR}}(\theta) = -\mathbb{E}_{Q_k^{\mathrm{clean}}} \sum_j m_j \log \pi_\theta(y_j \mid h_j)$$
其中 $m_j \in \{0,1\}$ 仅在公共前缀 $h_j$ 后选择生成的策略决策 token，critic 权重在此阶段冻结（$\Delta\phi|_{\mathrm{RSR}}=0$）。

**（2）Critic 初始化与传递**：
- RPR-1 的 critic 种子 $\phi_{\mathrm{seed}}$ 由 RSR-1 actor 冻结下 1 个 SWR epoch 训练得到；
- RPR-2 的 critic 继承自 RPR-1 保留权重的 $\phi_k^+$，经过 4 步 critic-only 校准后才进入联合 PPO 更新：
$$\phi_{k+1}^{\mathrm{warm}} = \mathrm{Warmup}_4(\phi_{k+1}^0; \theta_{k+1}^0 \text{ fixed})$$

**（3）TB-GAE 信用分配**：
定义生成位置 $u_j$ 对应 turn ID $t(j)$，建立 turn 级时钟：
$$(\Gamma_j, \Lambda_j) = \begin{cases} (1, 1), & t(j+1) = t(j) \\ (\gamma_{\mathrm{turn}}, \lambda_{\mathrm{turn}}), & t(j+1) \ne t(j) \end{cases}$$
优势估计：
$$\delta_j^{\mathrm{TB}} = \tilde{r}_j + \Gamma_j(1-d_j)v_{j+1} - v_j, \quad A_j^{\mathrm{TB}} = \delta_j^{\mathrm{TB}} + w_j A_{j+1}^{\mathrm{TB}}$$
其中 $w_j = \Gamma_j \Lambda_j (1-d_j)$。跨 turn 衰减 $(\gamma_{\mathrm{turn}}\lambda_{\mathrm{turn}})^B$ 与 observation 长度解耦；当所有 $\Lambda_i=1$ 时，critic 误差仅有 $-e_j$ 局部项。

**（4）奖励演化**：
- **FDR（Phase 1，RPR-1）**：$R_{\mathrm{FDR}} = \max(P/D_f, 0.00)$（valid CTRF）或 $S$（fallback），$D_f = q_{0.90}(N|f)$ 为 family 历史 90 分位数，除法固定于每阶段内。
- **SPR（Phase 2，Long-Reward RPR-2）**：$R_{\mathrm{SPR}} = S + \frac{P}{P + D_f'}$，离线重校准后失败奖励占比由 50.09% 降至 28.72%；$g_f(P) = P/(P+D_f')$ 为一阶导正、二阶导负的凹函数，确保每失败得分 <1、每成功得分 ≥1。

**（5）RPR 联合优化目标**：
$$\mathcal{L}_\pi = -\mathbb{E}_j \min\{\rho_j A_j, \mathrm{clip}(\rho_j, 0.70, 6.00) A_j\}$$
$$\mathcal{L}_V = \mathbb{E}_j \max\{(V_\phi(h_{u_j}) - \widehat{G}_j)^2, (\bar{V}_j - \widehat{G}_j)^2\}$$
其中 $\bar{V}_j = v_j + \mathrm{clip}(V_\phi - v_j, -4.00, 4.00)$；无 KL/熵惩罚；actor 梯度裁剪阈值 1.00。

## 实验与结果

**训练设置**：3,000 任务池（2,400 语义验证 + 600 补充终端任务），Global batch=256（oversampling 至 288），262,144 token 上下文，Megatron 3D + context parallelism 4，per-GPU 8,192 token 预算。

**评估基准**：Terminal-Bench 2.1、3.0、4.0、Hard、LHTB（46 任务）、SWR100（与训练集无重叠）。

| Checkpoint | TB-4.0 | TB-2.1 | TB-3.0 | TB-Hard | LHTB 过程奖励 | LHTB 严格完成 | SWR100 |
|---|---|---|---|---|---|---|---|
| Base | 0.00% | 51.70% | 0.00% | 31.00% | 0.21 | 0/46 | 3.00% |
| RSR-1 | 6.10% ↑6.10 | 70.10% ↑18.40 | 6.80% ↑6.80 | 54.00% ↑23.00 | 0.29 ↑0.08 | 0/46 | 5.00% ↑2.00 |
| RPR-1 | 16.67% ↑10.57 | 77.53% ↑7.43 | 14.29% ↑7.49 | 69.50% ↑15.50 | 0.30 ↑0.01 | 2/46 | **6.00%** ↑1.00 |
| RSR-2 | 17.46% ↑0.79 | **86.52%** ↑8.99 | **15.00%** ↑0.71 | 76.00% ↑6.50 | 0.31 ↑0.01 | 4/46 | 6.00% =0.00 |
| **Terminal-RSI-27B** | **19.84%** ↑2.38 | **86.52%** =0.00 | **16.95%** ↑1.95 | **79.00%** ↑3.00 | **0.37** ↑0.06 | **5/46** | 6.00% =0.00 |

- **四合一 Mean Success**：从 Base 20.68% 递增至 Terminal-RSI-27B 的 50.58%，提升 **29.90pp**（相对 Base 失败率 79.33%→49.42%，降幅 37.70%）。
- **LHTB**：过程奖励 0.21→0.37（+76.19%），严格完成 0/46→5/46。
- **SWR100**：3.00%→6.00%（翻倍），RPR-1 后进入平台期。
- **直接 PPO 对比**（表 4）：Direct PPO 在 TB-4.0 全阶段零成功，TB-2.1 最终 48.31%（低于 RPR-1 的 77.53%），验证 RSR 先导的必要性。
- **难度分层**（图 10）：Medium 23.58%→58.96%，Hard 14.91%→49.12%；Easy 31.78%→72.88%；Extreme 非单调（RPR-1 的 10.98% → RSR-2 的 6.10% → RPR-2 恢复至 10.98%）。
- **与前沿模型比较**：TB-4.0 的 19.84% 超越 Gemini 3.8 Flash（19.70%）和 DeepSeek V4 Pro（14.60%）；TB-2.1 的 86.52% 超越 GPT-6 Sol（85.40%）和 Kimi K3（85.00%）。

## 相关工作脉络
1. **Recursive Self-Rewrite（RSR, Li et al. 2026）**：本文在此基础上引入 critic 继承与 RPR，将 RSR 发现的成功轨迹转化为监督信号并交叉 harness 重写；定位差异：RSR 仅有 actor-only 重写，无 critic 值函数保留与奖励演化。
2. **T1（Yang et al. 2026）**：提供 T1 终端基础设施、Token-In/Token-Out 契约与异步 PPO 模板；本文将其与 RSR 递归耦合，而非单独贡献。
3. **STaR / ReST / ReST-EM / Reflexion**：均为 generate-filter-train 范式；本文与它们的本质区别是用环境级 verifier 而非语言级奖励，且通过 critic handoff 保留值结构。
4. **SAO / Skip-Observation GAE（Hou et al. 2026）**：跳观察节点，但未引入 turn 边界时钟；本文 TB-GAE 进一步引入 turn-level discount 并分析 cross-chunk 精确续接误差。
5. **VAPO / value optimization 相关工作（Yuan et al. 2025; Yue et al. 2025）**：关注长 CoT 中 value 优化的崩溃问题；本文通过 Warm-up 初始化 + 阶段内 4 步 critic-only 校准来缓解该问题。
6. **Terminal-Bench 系列（Merrill et al. 2026; Terminal-Bench Team 2026）**：提供评测环境与基准；本文贡献在于训练框架而非基准构建。

## 局限性与未来方向
- **因果归因未充分隔离**：阶段特定学习率、奖励、初始化同时变化，未进行 compute-matched 的组件消融（TB-GAE vs. 标准 GAE、SPR vs. FDR、inherit vs. reset critic 的交叉控制仍在提案中）。
- **评估覆盖有限**：5 个阶段仅对应 1 个 27B backbone，未建立 scaling law；三重复跑不等价于独立训练 seed，无统计显著性检验。
- **Verification 覆盖缺口**：Rust–C 任务暴露了"通过 C 接口委托 Python 执行"仍可通过 fuzz-test 的架构漏洞；reward calibration 可能保留捷径。
- **训练集泛化审计不完整**：SWR100 与训练集无重叠，但 broader repository / pretraining overlap 尚未充分审计。
- **未来方向**：compute-matched 组件控制实验、独立训练 seed 验证、更严格的 native Rust–C AST 执行路径验证、扩展至更大模型规模。

## 研究启发与可借鉴点
1. **Critic 继承策略（Critic Handoff）**：在多次 actor 更新的递归训练中，保留已校准的 critic 权重并进行阶段内少量 critic-only 微调，比每次都重置或从头训练更高效；此思路可迁移至多阶段 RLHF/RLAIF pipeline。
2. **TB-GAE 的 turn-boundary 时钟思想**：对长交互 agent 任务，将 credit 分配粒度从 token 提升至 interaction turn，可有效消除 observation 长度带来的虚假衰减；适用于任何 token/decision 不同步的 sequential decision-making 场景。
3. **奖励演化（FDR → SPR）的阶段适配**：训练前期用密度进度奖励（FDR）鼓励探索，后期切换到严格成功优先（SPR）抑制虚假高分失败；这一"奖励退火"策略可推广到其他需要平衡探索与严格验证的 agent 训练。
4. **固定轨迹离线重评分（Fixed-Trajectory Rescoring）**：使用同一批 4,352 轨迹在不同奖励函数下重评分，隔离奖励分配效应与训练效应，避免混淆；这是一种可复用的消融诊断方法。
5. **Difficulty-stratified 分层分析**：用结构代理（checker 数、file footprint、instruction length）预先划分 Easy/Medium/Hard/Extreme 四层并冻结，追踪各层非单调表现，可揭示 aggregate 指标掩盖的能力退化；建议纳入后续 benchmarking 流程。

## 关键术语表
- **Terminal Agent**：通过改变可执行环境状态（而非文本响应质量）来完成任务的智能体，需通过 protected verifier 验证。
- **Recursive Self-Rewrite（RSR）**：对已通过 verifier 的轨迹进行 runbook 规划、reviewer 审查与 cross-harness 重新执行，生成干净策略数据供 actor-only SFT。
- **Recursive Policy-Refine（RPR）**：继承前阶段 critic 权重，经少量 critic-only 校准后进行 actor-critic 联合 PPO 优化的阶段。
- **Turn-Boundary Skip-Observation GAE（TB-GAE）**：在 token 级 generated position 间插入 turn 级时钟 $(\gamma_{\mathrm{turn}}, \lambda_{\mathrm{turn}})$，跨 turn 衰减与 observation 长度无关的 GAE 变体。
- **Family Dense Reward（FDR）**：按 family 历史 90 分位数归一化通过检查数 $P/D_f$ 的进度奖励，无 upper bound 和 completion bonus。
- **Success-Prioritized Reward（SPR）**：严格成功信号 $S$ 加上有界凹形进度项 $P/(P+D_f')$，保证失败得分 <1、成功得分 ≥1。
- **Warm-up Critic Model**：在 RSR-1 actor 冻结状态下训练 1 个 SWR epoch 得到的初始 critic $\phi_{\mathrm{seed}}$，用于 RPR-1 的 critc-inheriting 初始化。
- **Mean@3**：对每个 checkpoint 在基准上独立运行 3 次，取算术平均成功率，而非 pass@3（最大成功）。

## 可复现要素
- **数据集**：训练集 3,000 任务（2,400 语义验证 + 600 补充终端任务）；测试集包含 Terminal-Bench 2.1/3.0/4.0/Hard、LHTB（46 任务）、SWR100（100 任务，与训练集无重叠）；训练集构成论文有说明但未提供详细任务清单。
- **代码/权重**：模型权重已开源（论文附有 Model Weights 链接）；训练代码未明确提及开源状态。
- **关键超参**：Actor LR：RPR-1 为 $5\times10^{-7}$，RPR-2 为 $1\times10^{-7}$；Critic LR：两阶段均为 $1.5\times10^{-5}$；Value clip = 4.00；Critic gradient clip = 100.00；Actor gradient clip = 1.00；Policy clip 范围 [0.70, 6.00]（$\epsilon_\ell=0.30, \epsilon_h=5.00$）；KL/熵系数均为 0.00；Global batch=256，oversampling 至 288；RPR-1 保留 20 步，RPR-2 保留 18 步；Critic-only 校准各 4 步；Adam $\beta_1=0.90, \beta_2=0.98$，weight decay=0.10。
- **评估采样**：Temperature 0.60，top-p=0.95，每 turn 最多 8,192 output tokens，每次调用 96,000 input tokens；主动 summarization 在累积 32,000 tokens 时触发；交互上限 100,000 turns。
