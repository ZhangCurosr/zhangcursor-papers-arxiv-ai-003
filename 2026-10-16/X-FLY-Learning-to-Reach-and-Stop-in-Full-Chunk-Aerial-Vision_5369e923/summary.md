---
title: "X-FLY-Learning-to-Reach-and-Stop-in-Full-Chunk-Aerial-Vision"
source: https://arxiv.org/pdf/2610.21450v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 17:19:56"
---

# 论文速读：X-FLY-Learning-to-Reach-and-Stop-in-Full-Chunk-Aerial-Vision

## 一句话总结
论文针对大VLM基航拍VLN策略执行完整动作块时产生的分布偏移问题，提出分阶段训练框架X-FLY，通过Recovery DAgger修复离轨决策锚点的控制与停止标签，再经Flow-GRPO联合优化连续控制块与伯努利STOP策略，在OpenFly上达到74.18%（test-seen）/51.73%（test-unseen）成功率，同时将单次决策的摊销推理耗时从195ms降至42ms。

## 研究问题与动机
- 基于大VLM的航拍VLN策略查询昂贵，执行完整预测控制块（full-chunk）可摊销推理成本，但会使策略下次被调用的“决策锚点”偏离专家轨迹分布。
- 专家数据对学习者实际访问的离轨锚点存在双重监督缺失：既没有从该位姿出发的连贯控制块（control gap），也未包含学习者首次进入目标区域时的STOP正样本（termination gap）。
- 实验表明，将单步BC直接切换为全块执行会导致test-seen SR从46.57%暴跌至16.45%，其中47.78%的失败为“到达目标区但未停止”，说明到达能力与终止能力需分别建模。
- 现有方法多聚焦于改变解码策略（如时序集成、双向解码）以提升反应性，本文则从训练分布修复的角度解决全块执行带来的对齐断裂。

## 核心贡献（创新点）
- 系统刻画了全块执行在航拍VLN中产生的支持缺口，定量分离了其对“导航到达”与“停止决策”的独立影响。
- 提出三阶段训练框架X-FLY（多域BC → Recovery DAgger → Flow-GRPO），在冻结VLM骨干的前提下仅微调轻量控制器与STOP头。
- 设计基于单调游标与三次Hermite插值的Recovery标注机制，使学习者轨迹平滑回归专家路线，并在首次入区锚点赋予STOP标签。
- 将Flow-GRPO适配至全块连续控制与独立STOP策略的联合优化，引入低频余弦模态探索噪声与STOP覆盖（stop-override）反事实尝试。
- 提供细粒度的失败归因分析框架，揭示剩余性能瓶颈主要集中在停止时机预测。

## 方法详解
- **策略架构**：采用快慢解耦设计。慢路径每16个控制步刷新一次关键帧缓存（冻结Qwen3.5-4B），快路径实时处理当前帧；融合序列经三层Fusion后输入两个独立模块：2层DiT整流流控制器（输出8维控制块 $U_t=(\Delta x,\Delta y,\Delta z,\Delta\psi)$）与独立双层MLP伯努利STOP头。
- **阶段一：多域行为克隆（BC）**：在OpenFly的UE、3DGS、AirSim域上，以849,106条专家窗口训练策略，目标函数包含整流流损失 $\mathcal{L}_{\text{flow}}=\mathbb{E}[\|v_\theta(X_\tau,\tau,c)-(U^\star-\epsilon)\|^2]$、STOP BCE及控制平滑/速度一致性辅助正则。
- **阶段二：Recovery DAgger**：部署全块执行器（$k=H=8$）运行学习者并记录锚点 $s_t$。使用单调游标 $c_t$ 沿专家路线前进，计算偏移 $d_t=p_t-p_{c_t}^\star$，通过三次Hermite衰减函数构造恢复目标：$p_{t,j}^{\text{rec}}=p_{c_t+j}^\star+d_t(1-3q_j^2+2q_j^3)$，使块终点无缝重连专家路线。首次进入20m成功区域的锚点标记STOP（悬停目标），其余标记CONTINUE。收集与训练交替3轮，每轮将Recovery数据与BC数据1:1混合。
- **阶段三：Flow-GRPO联合优化**：每条任务采样8个闭环尝试（6自由尝试+2 STOP覆盖尝试，覆盖前$n\in\{1,2,3\}$次采样STOP）。基于组内相对优势 $A_{i,t}=(R_{i,t}^{\text{go}}-\mu_t)/\max(s_t,s_{\text{floor}})$ 更新策略，施加KL正则化到初始策略（$\beta=0.01$）。扩散去噪的最后3个Euler步引入随机噪声，探索噪声被约束在4维余弦低频基上以保持轨迹平滑并维持闭式对数密度。额外构造outcome-verified STOP正负对，使用margin-ranking+BCE损失 $\mathcal{L}_{\text{rank}}$ 强化停止时机判别。奖励函数以任务结果为主，辅以进度信用与控制惩罚。

## 实验与结果
- **数据集与基准**：DreamFly处理的OpenFly分桶，test-seen含1,392条轨迹（UE BigCity+六城市AirSim），test-unseen含404条（持留UE SmallCity），成功半径20m。
- **最强结果**：X-FLY (k=8) 在test-seen达到SR 74.18%、OSR 84.07%、SPL 66.34%、NE 17.77m；test-unseen达到SR 51.73%、OSR 68.07%、SPL 47.03%、NE 29.17m，全面超越DreamFly及所有传统基线。
- **阶段增益量化**：全块BC仅16.45% SR；Recovery DAgger将其提升至63.46%（test-seen），几乎消除no-stop失败；Flow-GRPO再增10.72点SR，将early termination从21.21%降至9.34%。
- **推理效率**：全块执行(k=8)将每执行一个航点的摊销模型推理从195ms降至42ms（4.66×加速），且所有成功样例均包含显式in-region STOP。
- **失败归因**：BC的主要退化在于停止时机；Recovery主要修复导航到达能力；Flow-GRPO主要优化停止时序；剩余失败仍集中在early/late stop，策略在93%以上的test-seen轨迹中能访问到目标区内锚点。

## 相关工作脉络
- **DreamFly [9]**：同用快慢VLM骨干，但采用单步执行(k=1)与重收展规划；X-FLY聚焦全块执行引入的分布偏移与停止建模，证明全块+训练修复可达到更优准确率-推理权衡。
- **TravelUAV [28]**：应用连续轨迹DAgger于航拍导航，教师沿参考路线正向搜索匹配；X-FLY采用单调游标直接索引，避免自交路线回溯错误，并在learner首次入区处显式标注STOP。
- **Flow-GRPO [18]**：原用于扩散模型RL训练；本文将其扩展至连续控制块与独立STOP头的联合优化，并加入低频时间基噪声与STOP覆盖机制。
- **Stop策略研究 [29]**：图VLN将STOP作为离散词汇之一；连续航拍VLN缺乏原生STOP坐标，本文采用独立伯努利头并在learner arrival anchor处监督，与专家终端窗形成互补
