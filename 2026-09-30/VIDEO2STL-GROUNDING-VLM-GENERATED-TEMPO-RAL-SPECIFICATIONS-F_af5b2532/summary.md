---
title: "VIDEO2STL-GROUNDING-VLM-GENERATED-TEMPO-RAL-SPECIFICATIONS-F"
source: https://arxiv.org/pdf/2609.37519v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 23:27:02"
field: "具身智能与机器人强化学习"
keywords: ["Signal Temporal Logic", "VLM", "reward shaping", "cross-embodiment transfer", "reinforcement learning", "robot manipulation", "locomotion"]
innovations: ["两阶段VLM从视频提取实体无关语义轨迹并生成参数化STL规范库", "保号Boltzmann平滑STL语义以保持逻辑满足边界", "两时间尺度奖励：短时滚动窗口稠密鲁棒性与长时因果进度奖励"]
benchmarks: ["ManiSkill3", "Barkour quadruped locomotion"]
---

# 论文速读：VIDEO2STL-GROUNDING-VLM-GENERATED-TEMPORAL-SPECIFICATIONS-FOR-ROBOT-LEARNING

## 一句话总结
Video2STL 提出将纯观测视频转换为参数化信号时序逻辑（STL）规范，并通过两阶段 VLM 提取与实体无关的语义事件轨迹，再用机器人轨迹对阈值和时序界进行实例化，最终将 ST 量化鲁棒性分解为短时滚动窗口稠密奖励与长时因果进度奖励，实现跨形态机器人学习的可解释任务转移。

## 研究问题与动机
- 基于视频的机器人策略学习缺乏"从视频转移到机器人的信息边界"，现有方法要么将视觉观测压缩为标量相似/价值信号，要么让基础模型直接生成奖励代码，但 temporal structure（时序结构）难以检查、接地与复用。
- 标量奖励会掩盖任务在时间上的展开方式；视觉嵌入奖励虽然依赖较少显式结构，但难以追溯高分数是否对应期望行为，跨形态迁移中像素/位姿对应不可靠。
- 已有 VLM/LLM 奖励生成管线将视觉、任务解释、几何与奖励工程耦合到单次调用，无法分离"视频决定什么"与"机器人数据决定什么"。
- 跨形态（人类/动物视频到目标机器人）下可迁移内容常为语义而非运动学，需要一个既能表达时序结构、又能量化评估、还可在专家轨迹上检验的形式化中间表示。

## 核心贡献（创新点）
- 将"仅观测视频"形式化为正式时序任务结构的来源：两阶段 VLM 流水线统一复用同一本体构建操作与时序规范库，替代任务专属奖励模板。
- 分离符号推理与形态相关接地：数值谓词阈值与时间界由成功轨迹拟合，独立保留集仅用于专家一致性过滤，不供应动作监督或定义符号结构。
- 设计两时间尺度规范奖励：短时公式在滑动窗口上的量化鲁棒性提供密集局部反馈，长时序主干的因果前缀监控提供一次性进度奖励。
- 引入保号 Boltzmann 平滑语义：以 smooth min/max 替换硬 min/max 提升可优化性的同时保持逻辑满足边界不变号。
- 在四足 locomotion 与四类 manipulation 任务上实现跨形态迁移，同一表示从视频解释到策略训练全程可检查。

## 方法详解
- 整体管线（Algorithm 1）：输入视频 V 与成功轨迹 $\mathcal{D}^+$ → 阶段一 VLM_event 提取实体无关语义事件轨迹 E → 阶段二 VLM_STL 将 E 映射为参数化 STL 规范库 Φ → 将 $\mathcal{D}^+$ 划分为接地集 $\mathcal{D}_{\mathrm{ground}}$ 与过滤集 $\mathcal{D}_{\mathrm{filter}}$（70/30）→ Ground(Φ, $\mathcal{D}_{\mathrm{ground}}$) 拟合 $\vartheta$ → Filter(Φ_ϑ, $\mathcal{D}_{\mathrm{filter}}$) 保留兼容公式 Φ_keep → 构造局部稠密奖励 $r_t^{\mathrm{local}}$ 与因果进度奖励 $r_t^{\mathrm{progress}}$ → PPO 训练策略 π_θ。
- Stage 1：VLM 根据固定本体输出事件记录 $e_k = (p_k, c_k)$，其中 $p_k \in \mathcal{P}$、$c_k \in$ {essential/terminal/incidental}，并给出成对时序关系与 persistent 属性；若概念超出本体则必须报告 missing concept，将本体不足显式化为可观察失败模式。
  - 操作本体：R={effector, object, receptacle, handle, articulated_part, goal, support}；$\mathcal{P}$ 含 Near, Engaged, Grasped, Released, Displaced, Toward_goal, At_goal, On_support, In_receptacle, Static, Upright, Open 等。
  - 四足本体：R={trunk, front_left, front_right, hind_left, hind_right, support}；$\mathcal{P}$ 含 contact, Swing, Touchdown, Liftoff, Support_count_at_least(k), Trunk_height/pitch/roll_stable, Periodic_leg_motion 等。
- Stage 2：VLM 只接收 Stage 1 的 JSON 轨迹，生成参数化 STL 库 Φ={φ_1,...,φ_K}；语法限定为原子谓词、合取、有界 eventually $\mathbf{F}_{[0,h]}$、有界 always $\mathbf{G}_{[0,H]}$ 及嵌套有界响应 $\mathbf{F}_{[0,H]}(\mu_1 \wedge \mathbf{F}_{[0,h_1]}(\mu_2 \wedge \mathbf{F}_{[0,h_2]}\mu_3))$ 与 eventual persistence $\mathbf{F}_{[0,H]}\mathbf{G}_{[0,h]}\mu$，禁止数值常量，仅允许符号参数 $H_{\mathrm{task}}, h_{\mathrm{goal}}, h_{\mathrm{settle}}$ 等。
- 接地（Grounding）：原子谓词映射为带符号的 STL 鲁棒性 $\rho_p(x_t; \vartheta_p) \ge 0$ 表示满足；通过成功轨迹的 empirical quantile 估计阈值，上限型用高分位（如 95th）、下限型用低分位，缓解落在 $\mathbf{G}$ 内时对瞬时偏差的敏感性；时间参数同理用分位估计。
- 专家一致性过滤（Filtering）：在 $\mathcal{D}_{\mathrm{filter}}$ 上对 $\phi_i$ 的全轨迹鲁棒性 $r_{ij}$ 计算第 α 分位（实验 α=0.10），保留 $Q_\alpha(\{r_{ij}\}) \ge 0$ 的公式；起 held-out 校验作用。
- 保号平滑 STL 语义：定义 $\mathrm{BMin}_\beta(z;A)=\frac{\sum_{i\in A}e^{-\beta z_i}z_i}{\sum_{j\in A}e^{-\beta z_j}}$（BMax 对称取正指数），构造 smin_sp / smax_sp：当 $\min z_i<0$ 仅对负坐标做 BM 聚合，当 $\min z_i>0$ 用全坐标，否则返回 0；满足 $\mathrm{smin}_\beta^{\mathrm{sp}}(z)\ge 0 \iff \min_i z_i\ge 0$，保证逻辑满足边界不被平滑改变。
- 两时间尺度奖励：
  - 短时局部：令 $\Phi_{\mathrm{local}}=\{\phi_i\in\Phi_{\mathrm{keep}}:\mathrm{span}(\phi_i)\le H_{\mathrm{local}}\}$，在尾部窗口 $W_t=[\max(0,t-H_{\mathrm{local}}),t]$ 上用保号平滑鲁棒性 $\rho_{\phi_i}^{\mathrm{sp}}(W_t)$；按 $\mathcal{D}_{\mathrm{ground}}$ 上的 $s_i=Q_\alpha(|\rho^{\mathrm{sp}}_{\phi_i}|)$ 归一化后作 $\bar{\rho}=\tanh(\rho/s_i)\in[-1,1]$；局部奖励 $r_t^{\mathrm{local}}=\frac{1}{|\Phi_{\mathrm{local}}|}\sum_{\phi_i\in\Phi_{\mathrm{local}}}\bar{\rho}_{\phi_i}(W_t)$（公式 7）。
  - 长时因果进度：从 $\Phi_{\mathrm{keep}}$ 中选 backbone φ，将其结构表示为有序语义阶段 $p_1\to\cdots\to p_K$，因果监控仅基于已实现前缀 $x_{0:t}$ 更新最大可达合法前缀长度 $b_t$；进度奖励 $r_t^{\mathrm{progress}}=(b_t-b_{t-1})/K$，每阶段仅发放一次，$\sum_t r_t^{\mathrm{progress}}\le 1$。
  - 总奖励 $r_t=\lambda_L r_t^{\mathrm{local}}+\lambda_P r_t^{\mathrm{progress}}$（公式 8）；四足 locomotion 因以周期性为主而仅使用局部滚动窗口鲁棒性。

## 实验与结果
- 四足 locomotion（MuJoCo XLA + Brax PPO，Google Barkour vb）：
  - 在 0.3–2.1 m/s 的指令速度下，Video2STL（Qwen-3.8 与 GPT-5.6）均以 100% survival 与 100% success 通过全部速度档，包括最高速度 2.1 m/s；Text2Reward 同样 100%/100%，Heuristic 在 2.0 m/s 降至 5%、2.1 m/s 降至 0%。
  - CoT（能量效率）方面：Text2Reward 在低速更优；Video2STL-Qwen 在高速度更具竞争力（1.9/2.0/2.1 m/s 处 CoT 分别为 0.98/1.00/1.02，对比 Text2Reward 的 1.00/1.07/1.14 与 Heuristic 的 1.40）；Gemini-3.1 在 ≥1.6 m/s 时 success 跌至 40%/0%/0%。
- Manipulation（ManiSkill3，4 任务，PPO，128 rollouts）：
  - Video2STL 平均 success-once=85.8%、success-at-end=67.0%，优于 native dense PPO（81.5%/59.5%）与 Text2Reward（65.0%/42.3%）。
  - 分项：PushCube 95%/93%（持平），StackCube 98%/95%（≈Text2Reward 98%/96%，显著优于 PPO 73%/52%），PlaceSphere 91%/75%（对比 PPO 55%/55% 与 Text2Reward 失败），LiftPegUpright 最难，V2S 59%/5%（基准 ~97%/~47–50%）；归因于 Upright 接地容差 15.9° 大于 peg 倾覆角 11.8°，导致释放过早。
- 实验控制严格：所有方法共享相同策略架构、优化过程、训练预算与环境配置，仅 reward formulation 不同。

## 相关工作脉络
- 视觉奖励/表征迁移类（VIP、RoboCLIP、GVL）：主要转移标量相似或进度信号，缺少显式时序结构与可检验边界；Video2STL 转而提取结构化 STL，并以分位接地与独立过滤集验证专家一致性。
- LLM/Code 奖励生成类（Text2Reward、Eureka、ROSETTA、Video2Reward）：通过自然语言或代码直接产出密集奖励程序，语义结构、数值阈值与实现细节纠缠；本文严格分离"视频定符号结构、轨迹定数值参数"，并以 STL 作为可解释中间表示。
- 时序逻辑引导 RL（LTL/STL 量化奖励、从示范挖掘规范）：通常假设给定或人工构造的模板；Video2STL 从零观测视频自动提取模板与结构，并用独立轨迹做 expert-consistency filtering。
- 语言到时序逻辑（Lang2LTL）：起点是显式语言命令，用于规划/执行层面；Video2STL 起点是视频，用于跨形态奖励构造。
- 跨形态视频模仿（Context-translation、domain-adaptive meta-learning）：转移表征或翻译后的示范；Video2STL 转移的是语义事件结构+STL 规范，对形态差异具有更强的解耦能力。
- 四足逻辑驱动控制（DeFazio 等、Gu 等、Atasever 等）：已有工作在给定规范后利用 STL 做 MPC/策略；本文聚焦从视频自动生成规范并完成端到端 RL 训练与跨形态验证。

## 局限性与未来方向
- 本体约束较强：若视频中的重要概念不在预定义 $\mathcal{P}$ 内，只能报告 missing concept 而不能扩展谓词，可能限制对复杂/新任务的覆盖。
- 接地仅依赖少量成功轨迹的分位数，对分布外或噪声较多的 expert 数据敏感；当前仅使用单一 70/30 分割，未系统性探索样本量与划分的稳定性。
- Manipulation 任务中 LiftPegUpright 在 success-at-end 上仍较弱，主因是 Upright 容差大于物理倾覆角，体现"语义规格严格化"可能超出可行域。
- 四足 locomotion 仅用局部滚动窗口鲁棒性，尚未尝试两时间尺度组合；高速度效率最优来自 Qwen-3.8，而 Gemini-3.1 在高速度表现崩塌，VLM 选择的敏感性有待系统化比较。
- 未在大范围跨形态变化（如不同尺寸/DOF 机器人、真实硬件）上验证；当前均在仿真中进行。

## 研究启发与可借鉴点
- 两阶段"语义事件轨迹 → 参数化 STL"解耦思路可直接迁移到其他从视频/语言生成可检查规范的场景；保留 ontology insufficiency 作为显式失败模式有助于诊断模型与本体匹配度。
- 保号 Boltzmann 平滑语义可推广至任意基于 min/max 的逻辑聚合层，在提升梯度可优化性的同时避免硬逻辑边界漂移，值得用于 STL/LTL 奖励的通用实现。
- 两时间尺度奖励（短时稠密鲁棒性 + 长时因果前缀进度）是处理多尺度时序任务的通用脚手架；对周期性任务仅用短时、对序列性任务叠加长时进度，这一设计原则可直接套用。
- 用独立 $\mathcal{D}_{\mathrm{filter}}$ 做分位一致性过滤（而非直接用全部数据训练规格）是一种轻量但有效的"规格鲁棒性校验"，可作为 VLM 生成规范的通用 sanity check。
- 本团队若关注"从多模态演示提取可解释规范"，可将 Video2STL 的 VLM_prompt 模板与本体设计接入自有任务（如装配、抓取），并用相似的两时间尺度奖励框架快速验证。

## 关键术语表
- **Signal Temporal Logic (STL)**：在实值信号上定义含时间边界算子（$\mathbf{F}$ 有界 eventually、$\mathbf{G}$ 有界 always）与时序属性的逻辑语言，支持量化鲁棒度。
- **Parametric STL (PSTL)**：STL 扩展，允许原子谓词阈值与时间区间边界以符号参数出现，待数据接地后再实例化。
- **Quantitative robustness**：STL 公式在轨迹上的实值满足度，正/负分别表示满足/违反，幅度衡量裕量。
- **Sign-preserving smooth semantics**：以带保号约束的 Boltzmann 软 min/max 替换硬 min/max，保证优化时的符号与硬语义一致。
- **Two-timescale reward**：将规范拆为短时局部鲁棒性（密集奖励）与长时因果前缀进度（稀疏、一次性）的组合机制。
- **Causal prefix monitor**：仅基于历史前缀判断当前已达到的最远合法阶段，防止使用未来信息、避免重复奖励已完成阶段。
- **Expert-consistency filtering**：在保留的独立专家轨迹集上以分位鲁棒性阈值检验并剔除与专家行为不符的生成公式。
- **Embodiment-independent semantic trace**：从视频抽取的任务相关实体/事件/时序关系的本体化 JSON 描述，不含机器人特有数值或动作。

## 可复现要素
- 数据集/视频：四足为动物跑步 treadmill 视频（论文中给出链接）；操作任务为 4 类人类示范视频（ManiSkill3 benchmark）。
- 代码/权重：论文 Project webpage 为 video2stl；具体仓库未在正文中声明（论文未提及）。
- 模型：使用 Qwen-3.8、GPT-5.6、Gemini-3.1 等 VLM 生成规范；具体版本/调用方式见附录。
- 关键超参：
  - 四足 PPO：400M 步、unroll=30、32 minibatch、4 updates/batch、γ=0.955、lr=1.5e-4、entropy=0.004、8192 envs、batch=256；domain randomization：摩擦 (0.6,1.4)、增益/偏置 (−5,5)。
  - 操作 PPO：learning rate 3e-4、γ=0.8、GAE λ=0.9、clip=0.2、vf coef=0.5、target KL=0.1、8 epochs、32 minibatch；各任务 2M–30M 步、1024–2048 并行环境。
  - 接地/过滤：70/30 划分；过滤阈值 $Q_{0.10}\ge 0$（四足部分用 $Q_{0.05}$）；tanh 归一化尺度由分位估计。
