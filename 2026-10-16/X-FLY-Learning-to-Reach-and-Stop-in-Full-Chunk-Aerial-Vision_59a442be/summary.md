---
title: "X-FLY-Learning-to-Reach-and-Stop-in-Full-Chunk-Aerial-Vision"
source: https://arxiv.org/pdf/2610.21450v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 17:20:15"
field: "视觉语言导航与具身决策"
keywords: ["Vision-Language Navigation", "Action Chunking", "Flow Matching", "Imitation Learning", "Reinforcement Learning", "Aerial Robotics", "DAgger", "Embodied AI"]
innovations: ["Recovery DAgger在learner-visited决策锚点以smoothstep偏移衰减构造恢复chunk并在到达点标注停止信号", "Flow-GRPO联合优化chunked flow controller与Bernoulli停止策略并引入低频余弦探索噪声", "将SR分解为reachability与termination失败以精确定位全chunk执行的瓶颈"]
benchmarks: ["OpenFly test-seen", "OpenFly test-unseen"]
---

# 论文速读：X-FLY-Learning-to-Reach-and-Stop-in-Full-Chunk-Aerial-Vision

## 一句话总结
X-FLY提出了一种三阶段训练框架，解决航空视觉语言导航（Aerial VLN）中全chunk执行带来的训练分布偏移问题：通过Recovery DAgger在learner-visited决策锚点补全恢复路径与到达停止标签，再经Flow-GRPO联合优化chunked控制与Bernoulli停止策略，最终在OpenFly test-seen上实现74.18%的成功率（SR），相较全chunk行为克隆（BC）的16.45%提升近58个百分点。

## 研究问题与动机
1. **全chunk执行的成本动机**：基于大型VLM的航空VLN策略每次查询开销大（195 ms/waypoint），执行完整预测的8个控制再重新规划可将摊销推理成本降至42 ms/waypoint（4.7×加速）。
2. **决策锚点的分布偏移（support gap）**：全chunk执行使策略查询点从expert trajectory漂移至learner-visited anchors，而expert数据无法覆盖两类关键情况。
3. **控制间隙（Control Gap）**：在偏离专家路径的learner锚点处，不存在从该位姿出发、连贯持续8步且能回归专家路线的expert chunk。
4. **终止间隙（Termination Gap）**：expert数据的正STOP标签仅限于专家终止窗口，learner首次抵达成功区域（20 m内）的锚点处缺乏停止监督信号，导致全chunk BC出现47.78%的"未停止"失败。

## 核心贡献（创新点）
1. **系统化刻画全chunk执行的支持间隙问题**：将导航失败与终止失败解耦分析，揭示全chunk BC的核心瓶颈从"能否到达目标区"转变为"能否在到达后正确停止"，这是已有工作未明确区分的。
2. **Recovery DAgger——learner锚点的恢复与到达停止标签**：以单调游标匹配expert轨迹，构造平滑衰减偏移的目标chunk（Eq.2），并首次在learner自身抵达20 m成功区的锚点上标注正STOP，与expert终端窗口形成互补。
3. **Flow-GRPO面向chunked控制的闭环结果优化**：在共享初始flow latent的8次闭环尝试（6次自由尝试+2次STOP-override）上计算group-relative优势，联合更新rectified-flow controller与Bernoulli STOP head，避免了learned value function的依赖。
4. **低频余弦模式的随机chunk过渡**：引入低维（M=4）余弦基探索噪声，使采样chunk保持平滑的同时提供充分的方向探索（式6），保证闭式对数密度计算。

## 方法详解
### 整体架构（Fast-Slow）
- **慢路径**：冻结Qwen3.5-4B VLM处理指令+每16步刷新的因果关键帧（≤4帧），提取2560维hidden states，通过两个learned query bank（observation bank/48 tokens、navigation bank/48 tokens）跨注意力提取特征。
- **快路径**：仅运行视觉塔处理当前448×448帧，经第三个query bank压缩为48个当前帧token。
- **融合序列**：含ego state（10-D相对量）、anchor token（距上次刷新后的位姿变化+cache age）、history token（已执行动作历史，被预测但未执行的排除）、5类task token，输入3层fusion block（768宽、12头、dropout 0.1）。
- **控制器+停止头**：两层DiT（256宽、8头、3.9M参数）经rectified flow输出H=8个4维控制$(\Delta x, \Delta y, \Delta z, \Delta\psi)$；独立的两层MLP STOP head输出Bernoulli概率，与控制器并行独立计算。

### 三阶段训练流程
**Stage 1: 多域BC初始化**
- 损失函数（Rectified Flow）：$\mathcal{L}_{\text{flow}} = \mathbb{E}[\|v_\theta(X_\tau, \tau, c) - (U^\star - \epsilon)\|_2^2]$，6步Euler采样。
- 辅助损失：STOP BCE(0.1)、endpoint loss(0.1)、控制平滑(0.1)、速度一致性(0.1)、偏航连续性(0.05)、离散动作CE(0.2)。
- 训练数据：UE、3DGS、AirSim三个domain（849,106个expert决策窗口）。
- 超参：AdamW(lr=1e-4, wd=0.01)，warmup 500步，cosine decay，batch=128，10 epochs，bf16。

**Stage 2: Recovery DAgger**
- 部署当前learner以k=H=8执行，记录每个anchor $s_t$处的位姿$p_t$。
- **恢复目标构造**：单调游标$c_t$按实际执行的控制步数推进，偏移$d_t = p_t - p^\star_{c_t}$，target为：
$$p^\text{rec}_{t,j} = p^\star_{c_t+j} + d_t(1 - 3q_j^2 + 2q_j^3), \quad q_j = j/H$$
为smoothstep函数，j=0处偏移为1、j=H处为0，确保chunk从learner位姿出发、在chunk末尾回归专家路线。
- **停止标签规则**：关闭STOP收集，首个进入20 m成功区的anchor标注STOP+悬停目标，此前均标注CONTINUE；游标超过expert终点后仍在外的anchor保留指向goal的CONTINUE标签。
- 训练：3轮迭代，每轮3 epochs，lr=3e-5，recovery:BC=1:1混合，保留最低val control-speed error checkpoint。

**Stage 3: Flow-GRPO闭环优化**
- **Rollout group**：每instruction+start pose采样8次闭环尝试（共享初始flow latent）：6次free attempt + 2次STOP-override（前$n \in \{1,2,3\}$次采样STOP被强制改为CONTINUE）。
- **优势函数**：$A_{i,t} = \frac{R^\text{go}_{i,t} - \mu_t}{\max(s_t, s_\text{floor})}$，s_floor=0.5，在同depth的attempt group内归一化。
- **Actor Loss**：
$$\mathcal{L}_\text{actor} = -\mathbb{E}[A_t(\sum_{s=3}^{5}\log\pi_\theta^s + \log\pi_\theta^z)] + \beta(D^u_\text{KL} + D^z_\text{KL})$$
其中$k=0.01$，控制项作用于执行chunk的决策、停止项作用于legal决策（第2个决策起允许STOP）。
- **排名+停止监督损失**（来自STOP-override轨迹）：
$$\mathcal{L}_\text{rank} = \text{softplus}(1-(\tilde{\ell}^+-\tilde{\ell}^-)) + \frac{1}{2}[\text{softplus}(-\tilde{\ell}^+) + \text{softplus}(\tilde{\ell}^-)]$$
其中$\tilde{\ell}=\ell/1.5$，正样本为20 m内合法决策、负样本为>22 m且后续到达成功的被override停止。
- **Reward设计**：$R = R_\text{task} + 2\mathbb{I}[\text{first entry}] - 1.5\mathbb{I}[\text{leave after entry}] + R_\text{progress} + R_\text{control} - \kappa\mathbb{I}[\text{unsuccessful}]$，其中$k=4.51$保证所有不成功episode的return为负。
- **随机chunk过渡**（探索噪声）：
$$x^{s+1} = \mu_\theta^s + \nu_s(\Phi\epsilon_s + \rho\Psi\xi_s), \quad \rho=0.1$$
最后3个Euler步（s=3,4,5）为随机动作，$\Phi$为4个余弦模式、$\Psi$为其正交补，保证采样chunk平滑且Cov矩阵满秩。
- 训练：3000次更新，lr=1.5e-6（flow controller）/5e-7（其余模块），gradient norm clip=1，每步1 optimizer step。

## 实验与结果
### 数据集与评估设置
- **OpenFly基准**（DreamFly处理版本）：
  - **test-seen**：1,392条轨迹（UE BigCity + 6个AirSim城市）
  - **test-unseen**：404条轨迹（held-out UE SmallCity）
- 成功定义：最终位置在目标20 m内
- 执行设置：k=8（每8个控制执行一次策略查询），控制预算128步
- 评估协议：每次route用STOP关闭的模式执行一次control rollout，基于存储的STOP logit按伯努利first-stop分布解析计算SR/OSR/SPL/NE

### 推理开销
- 慢路径每16 waypoints刷新一次（0.320 s/refresh）
- 视觉塔单次调用0.141 s，融合/控制/STOP调用0.034 s
- k=1：3.12 s/16waypoints → 195 ms/waypoint
- k=8：0.67 s/16waypoints → **42 ms/waypoint**（4.66×加速）

### 主要结果（Table 1）

| 方法 | test-seen SR | test-unseen SR |
|------|-------------|----------------|
| DreamFly [9], k=1 | 32.04% | 29.46% |
| X-FLY BC, k=8 | **16.45%** | **18.43%** |
| + Recovery DAgger, k=8 | 63.46% | 40.63% |
| **+ Flow-GRPO (X-FLY full), k=8** | **74.18%** | **51.73%** |

- **全chunk BC相比单步BC的SR大幅下降**：test-seen从46.57%降至16.45%（导航失败几乎不变28.02%→30.77%，但无停止失败从0.07%暴增至47.78%）。
- **Recovery DAgger的增益**：test-seen SR +47.01 pts（16.45%→63.46%），导航失败从30.77%降至7.14%，无停止失败降至0%。
- **Flow-GRPO进一步增益**：test-seen SR +10.72 pts（63.46%→74.18%），主要减少过早停止（21.21%→9.34%）。
- **test-unseen上单步BC仍落后DreamFly**（22.49% vs 29.46%），两阶段闭环训练带来显著优势。

### 失败归因（Table 3 / Fig.4）
- BC k=8主要失败类型：**No-stop**（47.78% seen, 23.88% unseen）
- Recovery DAgger k=8主要失败类型：**Early stop**（21.21% seen, 37.18% unseen）
- Flow-GRPO k=8主要失败类型：**Late stop**（9.89% seen, 15.84% unseen）+ Early stop（21.04% unseen）
- **所有X-FLY方法的success均以显式in-region STOP结束**（no-STOP success = 0%）。
- 关闭STOP后，Flow-GRPO policy在test-seen 93.41%和test-unseen 88.61%的episode中访问到成功区anchor，剩余差距主要来自停止时机。

### Flow-GRPO训练动态（Fig.5）
- 从Recovery DAgger初始化：3000步内SR从53.4%升至65.1%，提升最快在第260步后SR突破60%，全程无后期回退。
- 从BC初始化对比实验：680步内SR仅在19%~26%波动，OSR持续下降（43%→28.4%），证明Recovery作为初始化对闭环优化至关重要。

## 相关工作脉络
1. **Diffusion/Flow Policy + Action Chunking**：Diffusion Policy[8]、Flow Matching[17]、Action Chunking[34]等基础方法支持连续控制序列预测；X-FLY的关键区别在于关注full-chunk执行引入的训练分布偏移，而非提升chunk本身的质量。
2. **DreamFly [9]**：同领域近期SOTA，采用因果记忆+receding-horizon diffusion planning，但每步仅执行1个控制（k=1）；X-FLY在同一fast-slow backbone上证明全chunk执行（k=8）配合适当训练可获得更高SR与更低推理开销。
3. **DAgger及其变体**：原始DAgger[24]通过learner访问状态重标注缓解covariate shift；HG-DAgger[13]依赖人类纠正、ChauffeurNet[2]合成扰动轨迹、TravelUAV[28]沿参考向前搜索匹配；X-FLY的Recovery DAgger独特之处在于使用单调游标按实际执行步数推进、在learner自身抵达点标注STOP。
4. **Flow-GRPO [18]**：将流匹配模型视为MDP的stochastic transitions进行RL优化；X-FLY将其适配至full-chunk aerial control场景，新增低频余弦探索噪声与factorized STOP objective，解决连续waypoint无内在STOP坐标的问题。
5. **Stop Learning in VLN**：图VLN中STOP作为离散action之一[29]；DreamFly在冻结导航策略上添加轻量终止头[9]；X-FLY的Bernoulli停止头独特之处在于：在learner到达锚点标注正样本、与chunked控制联合从闭环结果优化。
6. **RefineFly [26] & WorldVLN [33]**：均在closed-loop中对 aerial policy进行post-training；X-FLY与它们的区别是明确针对full-chunk执行这一特定部署模式设计训练管道，并将reachability与termination失败显式分离分析。

## 局限性与未来方向
1. **停止时机仍是残余瓶颈**：最终policy在多数episode中能访问到成功区anchor（93%+），但转化率不足，early/late停止构成主要失败来源。
2. **训练seed单一**：每个训练配置仅使用一个随机种子，结果未评估训练seed方差。
3. **仿真到实物的迁移未验证**：OpenFly未捕获真实飞行器动力学、感知噪声及操作安全约束，部署至物理平台需单独验证。
4. **STOP机会稀疏性效应**：全chunk执行将STOP决策频率降至每8步一次，与k=1的native stopping policy存在结构性差异，论文通过replay实验部分分离了此效应但尚未完全解决稀疏决策下的fine-grained停止控制问题。

## 研究启发与可借鉴点
1. **"到达≠停止"的失败分解思路值得迁移**：将SR拆分为reachability（导航能否到达）与termination（到达后能否正确停止）两个独立维度，为后续改进提供明确诊断方向；本团队在室内/车载VLN任务中同样面临类似"到达但未完成交互/取货"的终止问题。
2. **Recovery DAgger的smoothstep偏移衰减构造**：以$(1-3q^2+2q^3)$（$q=j/H$）构造从learner位姿平滑回归专家路线的target，避免了nearest-point匹配的歧义（交叉路径场景），此构造可直接迁移至其他continuous action chunking任务中的off-policy状态恢复。
3. **GROUP-RELATIVE ADVANTAGE无需value function**：Flow-GRPO在同task同depth的8次attempt间计算归一化优势，避免了额外训练critic；该方法适用于样本效率敏感、环境模拟成本高的embodied navigation场景。
4. **低频余弦探索噪声的设计**：将高维chunk空间（H×D=32维）的探索噪声投影到低频余弦基（M=4），既保证chunk平滑性又提供足够探索方向；对任何需要生成时间序列chunk的任务（如机械臂trajectory generation）均有借鉴价值。
5. **STOP-override counterfactual构造**：通过强制override前n次STOP决策为CONTINUE，获得同一start条件下的反事实对比样本，用于构建same-depth的positive/negative停止对，该设计可在任何需要fine-tune终止时机的决策任务中复用。

## 关键术语表
**Full-chunk execution**：策略每次查询预测H个连续控制动作并全部执行后再重新规划，区别于每步执行1个控制后重新查询的单步执行模式。
**Decision anchor**：策略被查询的决策时刻的agent状态（每隔k个控制执行一次查询）。
**Recovery DAgger**：在learner-visited decision anchor处，通过单调游标匹配专家轨迹构造平滑回归目标chunk，并在learner首次进入成功区的anchor处标注正STOP信号的DAgger变体。
**Flow-GRPO**：将flow matching模型的denoising步骤视为MDP的随机动作，在同task group内以闭环outcome计算group-relative优势，联合优化控制chunk与停止策略的RL方法。
**Bernoulli STOP head**：独立于chunked控制器的两层MLP，输出每次决策时停止的概率$p_t=\sigma(\ell_t)$，以伯努利采样决定CONTINUE或STOP。
**Low-frequency cosine exploration noise**：将chunk空间的探索噪声投影到4个正交余弦模式（低频）及其正交补（高频小噪声），使采样chunk保持平滑同时保证Cov矩阵满秩。
**STOP-override attempt**：训练时强制将前$n \in \{1,2,3\}$次采样STOP决策改为CONTINUE的运行轨迹，用于构建反事实对比样本和outcome-verified停止监督对。
**Amortized inference cost**：全chunk执行将慢VLM的推理成本均摊到多个控制步上，从195 ms/waypoint（k=1）降至42 ms/waypoint（k=8），实现4.7×加速。

## 可复现要素
- **数据集**：OpenFly（DreamFly处理版本），包含UE（真实城市航拍）、3DGS（三维高斯重建场景）、AirSim（合成）三个domain；test-seen和test-unseen split已公开。
- **代码/权重**：论文提供了project page（https://juliusju572.github.io/x-fly/），代码开源声明在论文中未明确提及；项目页可能含代码与权重下载链接。
- **关键超参**：
  - Chunk长度H=8，执行步数k=8（full-chunk）
  - BC：AdamW lr=1e-4, wd=0.01, warmup=500, batch=128, 10 epochs
  - Recovery DAgger：lr=3e-5, warmup=50, batch=128, 3 epochs × 3 rounds
  - Flow-GRPO：flow controller lr=1.5e-6, 其余模块lr=5e-7, 3000 updates, β=0.01, ρ=0.1, M=4余弦模式, 6 Euler steps, 3 stochastic tail steps
  - 成功半径：20 m（进入）/ 22 m（离开判定）
  - 控制预算：128步
  - 训练任务采样：50% natural + 20% curriculum（45%→70%最易路线）+ 30% priority
