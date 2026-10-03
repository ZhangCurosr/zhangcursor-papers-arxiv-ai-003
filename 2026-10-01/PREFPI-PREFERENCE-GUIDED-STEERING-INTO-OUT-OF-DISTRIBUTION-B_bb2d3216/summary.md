---
title: "PREFPI-PREFERENCE-GUIDED-STEERING-INTO-OUT-OF-DISTRIBUTION-B"
source: https://arxiv.org/pdf/2609.40165v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-03 03:01:07"
---

# 论文速读：PREFPI: PREFERENCE-GUIDED STEERING INTO OUT-OF-DISTRIBUTION BEHAVIORS

## 一句话总结
提出了 PrefPI（Preference-Guided Policy Iteration），一种仅依赖相对偏好反馈的迭代框架，通过分类器无关引导（CFG）将预训练生成式机器人策略逐步推向初始策略有效支持集之外的期望行为。

## 研究问题与动机
- **核心问题**：如何在无需额外演示或手动设计奖励函数的情况下，仅通过用户对自生成轨迹的相对偏好比较，将已具备任务能力的预训练生成策略引导至初始分布中极少或从未出现的期望行为（即 OOD 行为跨越）。
- **现有方法不足**：当前偏好学习方法（如 FDPP、Diffusion-DPO、FlowPRO 等）主要优化策略在数据收集时已能观测或表征的模式，无法有效推动策略进入全新行为区域；单轮偏好更新在目标行为初始不可见时直接失效。
- **理论缺口**：缺乏将相对偏好选择形式化为隐式最优性信号，并与 CFG 结合实现单调策略改进的机制，导致迭代引导缺乏理论保证。

## 核心贡献（创新点）
- **定义超越初始有效支持的偏好学习新设定**：首次系统研究目标行为在初始策略下极难出现、无额外演示供给的强 OOD 引导场景。
- **提出 PrefPI 迭代框架**：将偏好学习重构为偏好条件生成建模，通过 CFG 融合“偏好分支”与“累积成功分支”，实现策略的渐进式行为偏移。
- **建立相对偏好与 CFG 策略改进的理论桥梁**：证明 top-m 相对选择诱导的因子关于潜在效用单调不减，CFG 放大该密度比可保证期望隐式效用的单调提升。
- **跨仿真与现实的大规模验证**：在 LIBERO 仿真与 WidowX 真机实验中，仅用 150 条偏好标签轨迹即实现运输高度从 10.7 cm 跃升至 19.8 cm，且任务成功率保持在 90% 以上。

## 方法详解
- **迭代循环结构**：从预训练生成策略 $\pi_0$ 出发，第 $k$ 轮生成 $N$ 条轨迹 $\mathcal{D}_k$，用户选出 top-m 偏好子集 $\mathcal{D}_k^+$；同时更新累积成功轨迹池 $\mathcal{D}_k^{\text{acc}} = \mathcal{D}_{k-1}^{\text{acc}} \cup \text{Success}(\mathcal{D}_k)$。随后训练双分支并用 CFG 生成下一策略 $\pi_{k+1}$，形成 generate-select-guide-redeploy 闭环。
- **双分支建模**：偏好条件分支（$c=1$）在 $\mathcal{D}_k^+$ 上训练，捕获当前偏好方向；无条件分支（$c=\emptyset$）在 $\mathcal{D}_k^{\text{acc}}$ 上训练，保留历史成功行为的多样性以防止灾难性遗忘。
- **CFG 融合与条件实现**：条件变量通过自然语言 prompt 实现，无条件分支使用原始任务指令，偏好分支指令前加特殊 token `[Cfg]`。生成时沿 CFG 公式 $p_w(\tau) \propto p(\tau) \left( \frac{p(\tau|c=1)}{p(\tau|c=\emptyset)} \right)^w$ 组合两分支，$w$ 为引导强度。
- **理论推导**：设隐式效用函数为 $U(\tau)$，top-m 选择概率 $g_k(U)$ 满足单调性 $u_1 \leq u_2 \implies g_k(u_1) \leq g_k(u_2)$。由贝叶斯规则可得偏好条件分布 $p_k^+(\tau) = \frac{1}{\rho} p_k(\tau) g_k(U(\tau))$，其密度比 $\frac{p_k^+(\tau)}{p_k(\tau)} = \frac{1}{\rho} g_k(U(\tau))$ 构成保序的最优性信号。CFG 应用后分布变为 $\tilde{p}_{k,w}(\tau) \propto p_k(\tau) g_k(U(\tau))^w$，可严格证明 $\frac{\partial}{\partial w} \mathbb{E}_{\tilde{p}_{k,w}}[U(\tau)] \geq 0$，即 CFG 在此设定下等价于单调策略改进算子。

## 实验与结果
- **仿真基准与设定**：在 LIBERO 上使用 PI0.5-LIBERO（flow-matching VLA），偏好目标包括 speed（加速）与 detour（垂直/水平绕高）。每轮收集 40 条轨迹，选取 top 40%。
- **OOD 引导效果**：detour 任务最大抬升
