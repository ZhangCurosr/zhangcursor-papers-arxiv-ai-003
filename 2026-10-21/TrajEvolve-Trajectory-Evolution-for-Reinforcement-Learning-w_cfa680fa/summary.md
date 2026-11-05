---
title: "TrajEvolve-Trajectory-Evolution-for-Reinforcement-Learning-w"
source: https://arxiv.org/pdf/2610.28060v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 21:51:58"
---

# 论文速读：TrajEvolve: Trajectory Evolution for Reinforcement Learning with Hindsight Credit Assignment

## 一句话总结
本文提出 TrajEvolve，将长 horizon Agent 的步级信用分配重构为“必要/冗余”二分类任务，利用环境反事实验证获取精确标签，并通过精炼历史轨迹生成高质量演化样本，有效破解了 on-policy RL 中成功轨迹稀缺与冗余步骤堆积的双重瓶颈。

## 研究问题与动机
- **成功轨迹采样稀缺**：长 horizon 任务探索空间巨大，基座模型在有限训练预算内难以采样到足够多的成功轨迹，导致学习信号稀疏、收敛缓慢。
- **末态奖励导致步级信用失准**：Group-based RL 仅依赖终态二元奖励进行组内归一化，对所有步骤一视同仁，无法识别不影响任务结果的冗余步骤，反而 incentivize 冗长操作。
- **现有信用分配方法存在本质缺陷**：启发式信号（如 LLM 置信度）与任务结果无因果关联，易强化“自信但错误”的步骤；基于 outcome 的方法只能识别“导向成功”的步，无法区分必要步与冗余步。
- **性能瓶颈受限于基座模型推理边界**：现有 RL 仅提升已有推理路径的采样效率，缺乏主动构造更优轨迹的机制，难以突破 on-policy 分布的固有约束。

## 核心贡献（创新点）
- **轨迹演化范式**：不仅优化模型自采样轨迹，还主动精炼历史轨迹中的低质量步生成“演化轨迹”，使策略暴露于实际训练中极少采样的优质组合中。
- **信用分配的二元化重构**：将难以量化的连续步级信用评分转化为必要/冗余二分类，把无限解空间压缩为可验证的有限假设空间，显著降低判别器训练难度。
- **基于环境的反事实后验验证**：利用任务执行环境的状态可重播性，通过逐移除步骤并回测任务成败来精确获取 Ground Truth 标签，完全摆脱对额外 Reward Model 或 LLM-judge 的依赖。
- **理论证明演化机制的收敛增益**：在风格化 GRPO 模型下证明 on-policy RL 存在冗余率下界且突破该下界所需迭代次数随 horizon $T$ 指数增长，而引入演化轨迹注入后可将收敛复杂度降至多项式级。

## 方法详解
- **问题重构**：定义步级判别器 $\mathcal{D}_\phi: \mathcal{S} \times \mathcal{A} \times \mathcal{T} \to \{0,1\}$，输出 $\hat{y}_t=1$ 表示步骤 $t$ 对任务完成必要，$\hat{y}_t=0$ 表示冗余。判别器通过 SFT 训练，损失函数为二元交叉熵 $\mathcal{L}_{\mathrm{SFT}} = -\mathbb{E}[\log P_\phi(\hat{y}_t = y_t \mid \mathbf{x}_t)]$。
- **反事实后验验证**：
  - 仅对环境改变型动作进行验证，信息收集型动作默认标记为必要。
  - 定义依赖步集合 $\mathcal{D}(t) = \{ j > t : a_j \notin \mathcal{A}(s_j^{-t}) \}$。
  - 依据充分性/必要性命题：若 $\mathcal{V}(\tau \setminus (\{t\} \cup \mathcal{D}(t))) = 1$ 则 $y_t=0$ 且递归标记 $\mathcal{D}(t)$ 中所有步为冗余；否则 $y_t=1$。
  - 冷启动阶段使用 Qwen2.5-14B-Instruct 预生成约 2000 条成功轨迹构建初始标签集。
- **策略优化融合**：
  - **冗余感知奖励塑形**：$\tilde{R}(\tau_i) = R(\tau_i) - \alpha \cdot \rho(\tau_i)$，其中 $\rho$ 为轨迹中冗余步占比，仅对真正冗余的轨迹施加惩罚。
  - **层次化优势计算**：剧集级优势 $A^E(\tau_i)$ 基于塑形奖励做组内归一化；步级优势 $A^S(a_t^{(i)}) = \mathbb{I}[y_t=1]$；最终优势 $A(a_t^{(i)}) = A^E(\tau_i) + \lambda \cdot A^S(a_t^{(i)})$。失败轨迹对首个错误步前的前缀给予奖励。
  - 策略通过标准 PPO clip 目标 $\mathcal{I}(\theta)$ 更新。
- **轨迹演化机制**：每 $F=5$ 轮触发一次，从历史轨迹中选取低质量样本执行三种策略：① **冗余消除**：仅保留 $y_t=1$ 的步构成 $\tau^*$；② **格式修正**：将非法动作映射到当前状态合法动作空间中的最近有效动作；③ **错误回滚**：在首次错误步处截断并重生成。演化轨迹重执行后与新采样轨迹混合参与优化。

## 实验与结果
- **数据集**：ALFWorld（3,827 个室内多步交互任务）、WebShop（1.18 万真实商品 + 1.2 万众包指令的电商仿真）、ScienceWorld（科学实验仿真，扩展验证）。
- **基线**：GPT-4o、Gemini-2.5-Pro、ReAct、Reflexion、PPO、RLOO、GRPO、GiGPO。
- **核心指标**：ALFWorld 整体成功率 92.2%（Qwen2.5-1.5B）/ 98.5%（7B），超越最强 RL 基线 GiGPO 5.5% / 6.7%；WebShop 成功率 71.9%（1.5B）/ 82.0%（7B），超越 GiGPO 4.7%
