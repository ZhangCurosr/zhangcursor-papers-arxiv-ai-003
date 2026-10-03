---
title: "Scale-and-Selection-What-Makes-Automatic-Harness-Evolution-W"
source: https://arxiv.org/pdf/2609.39304v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-03 08:47:17"
field: "robot learning with foundation model agents"
keywords: ["harness evolution", "visual-interface robot agent", "auto prompt engineering", "agent self-improvement", "Champion-Challenger selection", "robot manipulation"]
innovations: ["rollout scale per round governs trust/generalization/steady improvement in harness evolution", "Champion-Challenger selection prevents performance drift under unconditional acceptance", "cross-environment transfer of evolved harness to RoboCasa kitchen tasks"]
benchmarks: ["LIBERO", "robosuite", "RoboCasa"]
---

# 论文速读：Scale-and-Selection-What-Makes-Automatic-Harness-Evolution-W

## 一句话总结
本文研究如何用编码智能体（optimizer agent）自动进化视觉界面机器人智能体的 harness，发现两个关键条件：每轮 optimizer 看到的 rollout 数量决定了进化结果的可信度与泛化性，而 Champion–Challenger 选择机制则是保证持续改进的必要保障。

## 研究问题与动机
- **H harness 通常手工编写**：视觉界面机器人智能体（coding agent 通过截图观察界面、调用少量 MCP 工具控制虚拟机械臂）的能力高度依赖其 harness（prompt、工具定义、控制协议、验证规则），但现有系统均手工设计
- **已有自动化方法不适用于此类系统**：先前自动 harness 优化工作（如 Meta-Harness、Self-Harness、RHO）中每次 rollout 仅数秒且由单元测试评分，而视觉界面机器人 agent 的每次 rollout 含数十至上百个模型调用、产生单一噪声二元结果，优化环路需满足新的要求
- **核心问题**：自动 harness 进化在视觉界面机器人 agent 上可行的前提条件是什么？

## 核心贡献（创新点）
1. **展示视觉界面机器人 agent 的 harness 可被另一个 coding agent 自动进化**，无需训练任一模型，30 轮将 held-out 成功率从 51% 提升至 67%（相对基线提升 16 个百分点）
2. **揭示 rollout 规模是信任、泛化与稳步提升的决定性因素**：每轮 rollout 数从 5 增至 100，held-out 成功率从 47% 升至 67%，小 batch（B=10）出现过拟合——训练成功率 70% 但 held-out 仅 54%
3. **证明无条件接受 optimizer 修订会导致性能衰减**，加入最简防御机制（Champion–Challenger 选择）即可使同一循环稳定提升 30 轮
4. **跨环境迁移验证**：在 RoboCasa 厨房场景中，同一进化环路从基线 2% 快速提升至 37%，且进化后的 harness 对更强模型（GPT-6 Astra）增益更大（+46 点 vs +35 点）

## 方法详解
- **系统设定**：Task agent 为未修改的 Codex（GPT-5.6 Sol），通过截图观察浏览器式 3D 点云 GUI，调用少量 MCP 工具控制虚拟目标夹爪；harness 包含 system prompt、MCP 工具集及其语义、每次工具调用后的观测返回、控制规则与终止逻辑，全部以代码/文本形式存于 agent 仓库
- **进化循环（Algorithm 1）**：每轮固定训练样本 D（B 个 case，相同任务和种子），optimizer agent（同样 Codex/GPT-5.6 Sol，独立会话）读取本轮所有 rollout 完整轨迹（证据清单 manifest M）与已rejected挑战者日志 L，提出一个基于当前 champion H_t 的单次 bounded commit 作为 challenger H'_t，challenger 在相同 B 个 case 上评估得到 s(H'_t)；**Champion–Challenger 选择**规则：仅当 s(H'_t) > s(H_t) 时提升 challenger 为新 champion，否则记录至 L 并保留原 champion；campaign 共 T=30 轮，held-out 评估仅在 campaign 结束后进行一次
- **关键设计细节**：evidence manifest 保留每个 case 的完整轨迹（模型 turn 序列、工具调用、控制器反馈、错误信息与最终 verdict），而非摘要统计；optimizer 被要求识别系统性、task-general 失败机制并提出单次 coherent 修改；challenger 必须通过仓库测试套件且不得改动评估控制平面
- **对照实验**：Unconditional acceptance 规则下 H_{t+1}=H'_t 恒成立，harness 沿单条 commit 链线性演进

## 实验与结果
- **数据集**：训练集 10 个桌面操作任务（robosuite 的 lift/square/stack + LIBERO 的 7 个任务），held-out 集 10 个不同 LIBERO 任务，各任务固定 seed 保证可复现；RoboCasa 厨房场景 10 个任务×10 seeds（B=100）
- **主要结果（Table 1）**：
  | B | 训练成功率（基线→最终） | 轮次内 promotion 数 | Held-out 成功率（最终 champion） | 5 节点均值（范围） |
  |---|---|---|---|---|
  | 5 | 20%→40% | 1 | 47% | 44.6% (35–49) |
  | 10 | 30%→70% | 3 | 54% | 51.4% (48–54) |
  | 20 | 40%→55% | 2 | 51% | 52.8% (50–55) |
  | 100 | 41%→60% | 8 | **67%** | 61.0% (51–67) |
  - 基线 harness held-out 为 51%，仅 B=100 产生显著且稳定的增益（+16 points）
  - 小 batch 过拟合明显：B=10 champion 训练 70% 但 held-out 仅 54%，训练/held-out 相关系数 0.60 由 B=100 节点驱动
- **选择规则对比（Figure 3）**：无条件接受在 B=100 时 10 轮内从 42% 跌至 34%；Champion–Challenger 在 B=100 时 30 轮内从 41% 单调升至 60%，promotions 散布于多轮（第 1,7,16,18,19,23,25,29 轮）
- **RoboCasa 迁移（Table 2,3）**：基线 2/100 → 第 1 轮 champion 26/100 → 第 2 轮 37/100；换用 GPT-6 Astra（进化时从未见过）后，基线 harness 为 37/100，进化后 harness 达 83/100，harness 增益随模型强度放大

## 相关工作脉络
- **VIA（Hu et al. [6]）**：视觉界面机器人 agent 的原始工作，harness 手工编写，本文在其基础上探索自动进化；本文与 VIA 的关键差异是把 harness 作为优化变量而非固定组件
- **Meta-Harness [13] / Self-Harness [21]**：软件 agent 的自动 harness 优化，rollout 秒级且由单元测试评分；本文将其思路移植到 rollout 长达数十分钟且结果高度噪声的视觉界面机器人场景
- **RHO [3]**：最接近的机器人 harness 搜索工作，但把 rollout 数量和选择规则视为固定前提；本文将其视为变量并系统研究
- **Rethinking Harness Evolution [19]**：批判已有 harness 进化报告的提升在匹配预算和 held-out 任务后消失；本文采用严格 train/test split 并直接研究两个受批评的关键变量
- **Guava [16] / Harness VLA [22]**：VLA 模型的 harness 设计；本文聚焦非 VLA 的 general coding agent 作为 robot policy 的场景
- **ASPIRE [17] / ENPIRE [20]**：agentic 机器人自我改进，但进化对象是 skill code 或训练 pipeline，而非 GUI-operating coding agent 的 harness

## 局限性与未来方向
- **rollout 规模的上界未探明**：B=100 时 held-out 成功率仍在上升，未观察是否饱和；受限于仿真并行成本
- **全仿真环境**：物理机器人上每次 rollout 占用实体硬件，难以获得 B=100 规模的 rollout，这是实际部署的主要瓶颈
- **仅研究两种选择规则**：无条件接受 vs Champion–Challenger，其他筛选策略（如基于置信度、多目标权衡）未探索
- **optimizer agent 自身能力依赖**：所有实验使用同一模型（GPT-5.6 Sol），未见 optimizer 能力提升是否进一步放大增益

## 研究启发与可借鉴点
- **rollout 规模作为进化质量的控制变量**可迁移到任何基于 execution feedback 的 agent 优化场景：小 batch 下"成功"信号噪声太大，决策易被运气主导
- **Champion–Challenger 机制的价值被低估**：在最简单的线性 commit 链中，即使 optimizer 质量不高，严格的严格大于判定也能避免累积退化，适合部署资源受限的实时系统
- **evidence manifest 保留完整轨迹而非摘要**使 optimizer 能从具体失败模式推理，这对长 horizon 任务（每次 rollout 含上百 model turns）尤为重要
- **harness 与模型能力的正交互补效应**（Table 3）提示：进化出的 harness 可与更强 foundation model 叠加使用，且增益不随模型增强而饱和，值得组合策略探索
- **跨域迁移验证的价值**：在同一环路不同环境（桌面操作→厨房）快速获得大幅增益，说明进化出的 harness 改进具有 task-general 特性而非过拟合特定场景

## 关键术语表
- **Visual-interface robot agent**：通过截图观察浏览器式 3D 界面、调用少量工具操控虚拟机械臂的编码智能体，harness 决定其行为上限
- **Harness**：位于模型与机器人之间的 prompt、工具接口、观测格式、控制协议与验证规则的集合，替代传统 VLA 模型
- **Optimizer agent**：读取 rollout 证据并提出 harness 修订的 coding agent，本工作中与 task agent 使用相同模型但独立会话
- **Champion–Challenger selection**：仅当 challenger 在相同固定测试集上严格优于当前 champion 时才采纳的决策规则
- **Rollout scale (B)**：每轮进化循环中 optimizer agent 可见的 task agent rollout 数量，决定每个 promotion 决策的信号信噪比
- **Evidence manifest**：由每轮 B 个 rollout 的完整轨迹（模型 turn 序列、工具调用、错误、verdict）构建的证据清单
- **Held-out success**：在训练任务集之外、optimizer 不可见的测试集上的成功率，反映泛化能力
- **Unconditional acceptance**：optimizer 提出的每次修订都被采纳的基准选择规则，实验中表现退化

## 可复现要素
- **数据集**：robosuite（lift/square/stack）+ LIBERO（Object/Goal/Spatial/90/10 suites）训练集；held-out 为不同 LIBERO 任务；RoboCasa 厨房场景——论文声明任务与 seed 固定保证可复现，但未提供公开下载链接
- **代码/权重**：论文未提供开源代码仓库或模型权重
- **关键超参**：每轮 rollout 数 B ∈ {5, 10, 20, 100}；campaign 轮次 T=30；每 rollout 墙钟时限 30 分钟；任务 agent 与 optimizer agent 均为 Codex/GPT-5.6 Sol（最高 reasoning effort）
- **模型**：任务 agent 与 optimizer agent 同模型（GPT-5.6 Sol），评测时额外使用 GPT-6 Astra
