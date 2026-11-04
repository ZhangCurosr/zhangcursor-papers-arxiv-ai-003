---
title: "Terminal-RSI-Recursive-Self-Improving-Terminal-Agents-via-Re"
source: https://arxiv.org/pdf/2610.22192v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-04 10:31:29"
field: "终端智能体强化学习"
keywords: ["terminal agent", "recursive self-improvement", "PPO", "GAE credit assignment", "reward shaping", "long-horizon agentic RL", "critic transfer"]
innovations: ["交替递归自重写（RSR）与critic继承递归策略精炼（RPR）的五阶段actor-critic框架", "Turn-Boundary Skip-Observation GAE（TB-GAE）实现跨轮次衰减的token级信用分配", "Success-Prioritized Reward（SPR）通过饱和进度项将严格成功与有界部分信用分离"]
benchmarks: ["Terminal-Bench 2.1", "Terminal-Bench 3.0", "Terminal-Bench 4.0", "Terminal-Bench Hard", "LHTB", "SWR100"]
---

# 论文速读：Terminal-RSI: Recursive Self-Improving Terminal Agents via Recursive Self-Rewrite and Policy-Refine

## 一句话总结
Terminal-RSI 提出了一种递归 actor–critic 框架，通过交替执行**递归自重写（RSR）**与**critic 继承的递归策略精炼（RPR）**，解决终端代理在长交互链中因时序膨胀、价值漂移与奖励错位导致的学习瓶颈；基于 Qwen3.8-27B 经五阶段训练得到的 Terminal-RSI-27B 在 Terminal-Bench 系列与 LHTB 上取得显著突破，四套件平均成功率达 50.58%。

## 研究问题与动机
- **终端代理的评估范式差异**：终端代理以对环境产生持久可执行变更并通过受保护验证器为目标（如 SWE-bench、Terminal-Bench），不同于常规语言模型的"回复合理性"评估；长周期依赖链（如 LHTB）中局部进展未必导向完整解。
- **时序膨胀（Temporal Dilation）**：数百个 token 可对应一次 shell 动作，而数千观测 token 不引入新策略决策；标准 GAE 在交错 token 链上按文本长度衰减远端信用，而非按实际交互轮次衰减，导致"同样决策+相同终端结果但因日志长短不同而获得不同信用"的不公平现象。
- **递归监督下的价值漂移**：RSR 重写改变 actor 偏好分布，进而改变 critic 遭遇的历史；后续强化学习再次改变分布，奖励校准又改变待预测回报。重置 critic 丢弃已有学习结构，直接传递则可能保留系统性预测误差，使更强 actor 无法自动为新规约下的新历史/新horizon 提供可靠价值。
- **奖励对齐困境**：不同任务家族暴露的语义检查数差异巨大；失败轨迹可累积大量通过检查却无法满足完整执行合约。单一起全局尺度会过度强调大验证器家族；纯比例进度奖励则在策略具备严格成功能力后仍偏向不完整行为。论文固定轨迹分析显示：旧奖励下失败轨迹占据 50.09% 奖励总量，其中 48 个失败通过了所有已执行场景检查却未通过终端门。

## 核心贡献（创新点）
1. **递归 actor–critic 框架**：提出 Base → RSR-1 → RPR-1 → RSR-2 → RPR-2 五阶段序列，通过在受控重写中保留 critic 权重、并在每轮 RPR 阶段本地重新校准价值函数，实现跨阶段有价值传递的递归自我改进。
2. **Turn-Boundary Skip-Observation GAE（TB-GAE）**：在保留 token 级基线的同时，沿交互轮次施加衰减；将跨 B 个轮边界的信用衰减独立于观测 token 数量，实现跨计算分块的精确续接。
3. **Family Dense Reward（FDR）与 Success-Prioritized Reward（SPR）**：FDR 按历史家族 90 分位数归一化进度；SPR 通过饱和进度项 $P/(P+D_f')$ 将严格成功与有界部分信用分离，离线重计分将失败奖励占比从 50.09% 降至 28.72%。
4. **两级 Critic 预热与阶段自适应学习率**：初始 SWR epoch 制备 Warm-up Critic Model；每轮 RPR 前冻结 actor 执行 4 次 critic-only 校准；RPR-2 将 actor 学习率降低 5 倍（$5 \times 10^{-7} \to 1 \times 10^{-7}$），防止继承价值基线遭受策略漂移冲击。
5. **多基准系统性评估与诊断**：在 Terminal-Bench 2.1/3.0/4.0/Hard、LHTB、SWR100 上验证；提供失败/成功分解、困难分层、reward 家族雷达图、critic EV 曲线与定性案例，揭示非单调分支与验证覆盖缺口。

## 方法详解
### 2.1 交互、监督与参数交接
- **交互建模**：任务 $x$ 包含公共指令 $x_{\text{pub}}$、沙盒与受保护验证器 $\mathcal{V}_x$；历史 $h_t = (x_{\text{pub}}, o_0, a_0, \dots, o_t)$；Actor 采样 $a_t \sim \pi_\theta(\cdot|h_t)$，环境返回 $o_{t+1}$；轨迹分解为 Actor 决策与环境转移（公式 1）。
- **验证重写（RSR）**：仅当 $\mathcal{V}_x(\tau)=1$ 时保留 $(x, \tau, H)$；规划器接收 $p_{\text{plan}} = I_{\text{plan}} \oplus x_{\text{pub}} \oplus \mathcal{C}(\tau)$；审查者用二进制 schema/artifact/兼容性门 $\boldsymbol{q}(x,b)$ 过滤；新执行 $E_k(x,b_\ell)$ 由保留 actor 在目标 harness $H_0$ 下进行；泄露筛选 $a_x(\tau) = \mathcal{V}_x(\tau) q_{\text{leak}}(\tau)$；干净回放集合 $Q_k^{\text{clean}}$ 均匀采样；RSR 损失为 masked 负对数似然（公式 3）：$\mathcal{L}_{\text{RSR}}(\theta) = -\mathbb{E}_{Q_k^{\text{clean}}} \sum_j m_j \log \pi_\theta(y_j|h_j)$，critic 权重冻结。
- **Critic 初始化与交接**：RSR-1 actor 冻结下进行 1 个 SWR epoch 制备 Warm-up Critic Model $\phi_{\text{seed}}$；RPR-2 继承保留的 RPR-1 critic；交接时 $\Delta \phi|_{\text{RSR}}=0$，每轮 RPR 前冻结 actor 执行 4 次 critic-only 更新后再联合训练（公式 4）。

### 2.2 奖励演化：FDR 与 SPR
- **FDR（公式 6）**：$R_{\text{FDR}} = \max(P/D_f, 0.00)$（验证报告有效时），否则回退到严格成功 $S$；$D_f = q_{0.90}(N|f)$ 为家族 $f$ 历史非卫生检查总数 90 分位数（默认 82）；Phase 内除数冻结；FDR 无单独完成奖励、无上界截断。
- **SPR（公式 7）**：$R_{\text{SPR}} = S + \frac{P}{P + D_f'}$（有效时），否则回退 $S$；$D_f'$ 经离线重校准：对高奖励失败家族除数×2，对长/低奖励家族除数÷4；饱和进度项 $g_f(P) = P/(P+D_f')$ 满足 $g_f' > 0, g_f'' < 0$；保证每次成功比匹配计数的失败恰好多 1，奖励上界 $< 2$。
- **计算时机**：两者均在 rollout 完成时计算；密度意味着进度解析，而非 step-level 推理监督。

### 2.3 TB-GAE：跨轮边界的信用分配
- **时钟设计（公式 8）**：对生成位置 $u_j$ 映射到轮次 $t(j)$；同轮内 $(\Gamma_j, \Lambda_j)=(1,1)$，跨轮时 $(\gamma_{\text{turn}}, \lambda_{\text{turn}})$；终止位置 $(\Gamma_j, \Lambda_j)=(1,1)$ 且不查询不存在的后继轮。
- **递归（公式 9–10）**：$\delta_j^{\text{TB}} = \tilde{r}_j + \Gamma_j(1-d_j)v_{j+1} - v_j$，$A_j^{\text{TB}} = \delta_j^{\text{TB}} + w_j A_{j+1}^{\text{TB}}$，$w_j = \Gamma_j \Lambda_j (1-d_j)$；目标 $\widehat{G}_j = A_j^{\text{TB}} + v_j$。
- **跨边界精确续接**：保留值引导 $v_{b+1}$ 并重置 trace 时误差为 $W_{j,b+1}A_{b+1}^{\text{full}}$；精确续接需同时传输 $v_{b+1}$ 与 $A_{b+1}$；总结改变 conditioning 故不等价。
- **Critic 误差分析（公式 13）**：完整终态 episode 下 $\Delta A_j = -e_j + \sum_{k=j+1}^{L-1} B_{j,k}(1-\Lambda_{k-1})e_k$；观测 bootstrap 项消失，中间同轮误差相消，仅保留初始误差 $-e_j$；所有 $\Lambda_i=1$ 时仅局部 critic 误差残留。

### 2.4 RPR 目标与两级 Critic 预热
- **联合优化（公式 14）**：
  - 策略损失 $\mathcal{L}_\pi = -\mathbb{E}_j \min\{\rho_j A_j, \text{clip}(\rho_j, 0.30, 5.00) A_j\}$（不对称裁剪 $\epsilon_l=0.30, \epsilon_h=5.00$）。
  - 价值损失 $\mathcal{L}_V = \mathbb{E}_j \max\{(V_\phi(h_{u_j})-\widehat{G}_j)^2, (\bar{V}_j-\widehat{G}_j)^2\}$，$\bar{V}_j = v_j + \text{clip}(V_\phi(h_{u_j})-v_j, -4.00, 4.00)$。
  - 无 KL/熵项，系数均为 0.00；样本等权（非 token 等权）。
- **阶段自适应超参（表 1）**：
  - RPR-1：Actor LR $5\times10^{-7}$，Critic LR $1.5\times10^{-5}$，奖励 FDR，保留 local step 20。
  - RPR-2：Actor LR $1\times10^{-7}$（降 5 倍），Critic LR $1.5\times10^{-5}$（不变），奖励 SPR，保留 local step 18。
- **基础设施**：Qwen3.8-27B，262144 token 上下文；T1 基础架构（slime、Megatron、SGLang、Harbor）；异步 1-rollout batch-ahead pipeline；3D parallelism + CP=4；per-GPU token 预算 8192；global batch 256，oversampling 至 288。

## 实验与结果
- **训练数据**：3000 任务池（2400 语义验证任务 + 600 补充终端任务）；SWR100 无重叠。
- **基准**：Terminal-Bench 2.1 / 3.0 / 4.0 / Hard、LHTB（46 任务）、SWR100（100 任务）。
- **主要结果（Terminal-RSI-27B，Mean@3）**：

| 基准 | Terminal-RSI-27B | Base | RSR-1 | 相对 Base 提升 |
|---|---|---|---|---|
| Terminal-Bench 4.0 | **19.84%** | 0.00% | 6.10% | +19.84 pp |
| Terminal-Bench 2.1 | **86.52%** | 51.70% | 70.10% | +34.82 pp |
| Terminal-Bench 3.0 | **16.95%** | 0.00% | 6.80% | +16.95 pp |
| Terminal-Bench Hard | **79.00%** | 31.00% | 54.00% | +48.00 pp |
| LHTB 过程奖励 | **0.37** | 0.21 | 0.29 | +0.16 (+76.19%) |
| LHTB 严格完成 | **5/46 (10.87%)** | 0/46 | 0/46 | 首次达成 |
| SWR100 | **6.00%** | 3.00% | 5.00% | ×2，RPR-1 后 plateau |

- **四套件平均成功率**：Base 20.68% → RSR-1 34.25% → RPR-1 44.50% → RSR-2 48.75% → Terminal-RSI-27B **50.58%**（相对 Base 提升 +29.90 pp）。
- **困难分层**：Easy 从 31.78% 升至 72.88%；Medium 23.58% → 58.96%；Hard 14.91% → 49.12%；Extreme 非单调（RPR-1 10.98% → RSR-2 6.10% → RPR-2 10.98%）。
- **与前沿模型对比**：TB-4.0 上 19.84% 略超 Gemini 3.8 Flash（19.70%）；TB-2.1 上 86.52% 超过 GPT-6 Sol（85.40%）与 Kimi K3（85.00%）。
- **固定轨迹重计分**：FDR → SPR 下失败奖励占比 50.09% → 28.72%，得分为 ≥1 的失败数 158 → 0。
- **Critic EV**：Warm-Up −1.15 → 0.01；RPR-1 0.03 → 0.27；RPR-2 全程保持正值（0.01–0.52）。
- **定性案例**：Embedding-drift 11 断言全通过；Rust–C FFI 双门均 100% 通过（但存在 Python 回退依赖，暴露非原生 AST 路径缺口）。

## 相关工作脉络
1. **递归自训练**：STaR / ReST / ReST-EM 建立 generate–filter–train 范式；Reflexion 通过口头反馈改进行为；本文 RSR 通过 runbook 规划与泄露审查跨 harness 重写已验证轨迹，与 learned value critic 无关。
2. **长视距 Agent 策略优化**：Critic-free 方法（group/leave-one-out/batch norm 优势估计，如 REINFORCE++、DAPO） vs. Critic-based 方法（T1、VAPO、value-preparation 传统）；本文采用固定非对称裁剪 PPO + TB-GAE 信用分配。
3. **异步 / 长交互训练系统**：IMPALA、AReaL、slime 等解耦 actor 轨迹生成与 learner 更新；本文基于 T1 异步基础设施但强调 critic 权重跨监督转换的保留机制。
4. **终端任务与环境构建**：TerminalTraj、CLI-Gym、Terminal-Task-Gen、SkillSynth、CLI-Universe、RST、Environment Evolution、Terminal-Universe 等；本文关注的是在已有监督信号上递归改进策略与价值，而非合成新环境。
5. **奖励 shaping 与 policy invariance**：Ng 等（1999）势函数不变性；本文 SPR 有界进度项不满足 potential-based 不变性，亦不等同最大化成功概率，而是显式分离严格成功与部分信用。
6. **Stale-but-Stable 异步 RL**：Yang et al.（2026）提出 staleness-adaptive trust regions；本文区分 policy drift 与 training–inference implementation mismatch，采用固定异步配置 + 阶段自适应学习率。

## 局限性与未来方向
- **因果归因未完全隔离**：阶段学习率、奖励、初始化同步变化；TB-GAE、孤立 reward、inherited-vs-reset critic 的 matched 对照实验仍需设计。
- **评估统计效力**：三次独立运行不足以替代多 seed 独立训练；checkpoint 选择需依赖受保护验证集而非最终基准。
- **验证器覆盖缺口**：Rust–C 案例暴露 native AST 执行路径未被完全覆盖；reward 校准可能保留捷径；replay 过滤可能丢弃有用行为。
- **泛化范围受限**：SWR100 虽训练无重叠但更广仓库/预训练重叠仍需审计；单 backbone 五阶段不establish scaling law 或普适单调性。
- **未来方向**：matched compute 的组件消融、cross-seed 重复、更长 horizon 的 strict completion、更广泛任务族与更开放 verifier 覆盖、动态 family 校准替代 frozen lookup。

## 研究启发与可借鉴点
1. **Critic 权重跨 supervised transition 保留**：在 actor-only 重写阶段冻结 critic、仅在 RPR 阶段联合训练，并引入 warmup + 4-step 本地校准，可有效缓解递归监督中的价值漂移——该模式可迁移至任何"先 SFT/重写、后 RL 精炼"的两阶段 pipeline。
2. **TB-GAE 的轮次级衰减设计**：将 GAE 的衰减单位从 token 切换到交互轮次，可解决长日志/多步观察导致的不公平信用衰减；任何多轮对话 Agent 均可借鉴该 clock-switching 思路。
3. **SPR 有界饱和进度分离严格成功**：$S + P/(P+D_f')$ 形式在保证探索性部分信用的同时压制"通过多数检查但未通过终端门"的虚假高奖励——适用于任何具多层验证器的任务（代码合成、系统设计、科学工作流重建）。
4. **阶段自适应 Actor LR 衰减**：第二轮 RPR 将 actor LR 降 5 倍而保持 critic LR 不变，保护继承价值基线不被策略漂移冲击——可作为递归精炼的通用调度策略。
5. **固定轨迹离线重计分隔离 reward 效应**：在同一批 trajectory 上切换 reward 函数（FDR → SPR）并比较失败占比，可在不重新 rollouts 的前提下诊断奖励分配偏差——该方法可复用于其他 reward engineering 工作。

## 关键术语表
**Terminal Agent**：以在可执行环境中产生持久变更并通过受保护验证器为目标的 AI 智能体，典型代表为代码修复、CLI 操作等场景。  
**Recursive Self-Rewrite（RSR）**：对已验证轨迹进行 runbook 规划与审查者过滤后的干净回放，仅在 actor 层面进行监督学习，critic 权重冻结。  
**Recursive Policy-Refine（RPR）**：继承上一阶段 critic 权重，在冻结 actor 完成 4 步 critic-only 校准后，联合进行 PPO 策略–价值优化。  
**TB-GAE（Turn-Boundary Skip-Observation GAE）**：在同轮内使用标准 token 级 GAE 递归，跨轮时以 $(\gamma_{\text{turn}}, \lambda_{\text{turn}})$ 衰减，观测节点不参与价值递归。  
**FDR（Family Dense Reward）**：按任务家族历史 90 分位数归一化的进度奖励 $P/D_f$，无单独完成奖励，Phase 内除数冻结。  
**SPR（Success-Prioritized Reward）**：严格成功项 $S$ 加上饱和进度项 $P/(P+D_f')$，将成功与部分信用分离，奖励上界 $< 2$。  
**Warm-up Critic Model**：在 RSR-1 actor 冻结条件下，用 1 个 SWR epoch 预训练的初始价值网络，为 RPR-1 提供良好价值基线。  
**Mean@3**：同一 checkpoint 在基准上独立运行 3 次，取算术平均值报告，不同于 pass@3（取并集）。  
**LHTB（Long-Horizon-Terminal-Bench）**：从 120 候选项目中筛选出的 46 个含参考解与隐藏测试的长依赖链修复任务，报告过程奖励与严格完成数。  
**SWR100**：基于 500 个可执行科学工作流的 100 个重建任务，含隐藏语义配置、结构检查与反捷径检查，与训练池无重叠。

## 可复现要素
- **数据集**：3000 任务训练池（2400 语义验证 + 600 补充）；评估基准 Terminal-Bench 2.1/3.0/4.0/Hard、LHTB、SWR100——基准本身公开，训练池细节见 Appendix F。
- **代码/权重**：论文声明 "Model Weights" 可获取（见首页链接），基础设施引用 T1、Megatron、SGLang、Harbor；具体开源状态需查阅附录与仓库。
- **关键超参**：
  - Actor LR：RPR-1 = $5 \times 10^{-7}$，RPR-2 = $1 \times 10^{-7}$
  - Critic LR：两轮均为 $1.5 \times 10^{-5}$
  - Policy clip：$\epsilon_l=0.30, \epsilon_h=5.00$
  - Value clip：$\epsilon_V = 4.00$
  - Global batch：256，oversampling 至 288
  - Critic-only calibration：每阶段 4 步
  - Retained local steps：RPR-1 = 20，RPR-2 = 18
  - KL/熵系数：均为 0.00
  - 评估采样：temperature=0.60，top-p=0.95，output per turn ≤ 8192，input per call ≤ 96000
- **其他**：262144 token 上下文；Megatron 3D + CP=4；per-GPU token 预算 8192；SGLang 静态 memory fraction 0.75。
