---
title: "Open-ended-Scientific-Discovery-with-Possibilistic-Reasoning"
source: https://arxiv.org/pdf/2610.11289v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-01 18:55:43"
---

# 论文速读：Open-ended-Scientific-Discovery-with-Possibilistic-Reasoning

## 一句话总结
本文针对物理科学中LLM驱动的开放式假设生成与验证难题，将问题形式化为基于可能论的溯因自主科学发现（AASD），提出利用 anytime-valid e-process 构建证据可能性空间、以固定背景可能性轮廓评估假设兼容性，并设计 Possibility Frontier Search 算法实现动态假设扩展与自适应实验选择，在合成与真实数据集上均实现了 anytime-valid 统计保证与渐近最优的发现进度。

## 研究问题与动机
- 物理科学实验具有噪声性、有限采样与高昂成本，且本质为溯因推理（寻找最佳解释而非演绎验证），现有数学/AI发现工作多依赖形式化验证系统（如 Lean）或固定假设集，难以直接迁移。
- 已有 anytime-valid 证伪方法（如 POPPER）可在自适应实验中安全排除错误假设，但局限于预设假设空间；当初始集合不完整（“最差的批次中最好的”困境）或背景知识需修正时，此类方法无法区分多竞争解释。
- 基于贝叶斯惊讶的探索方法（如 AutoDiscovery）仅在已知合理解释被排除后才具科学意义，且缺乏可单调度量的发现进度指标与理论收敛保证。
- 缺乏能统一处理“假设动态生成—证据累积—背景知识约束—主动实验设计”并提供 anytime 统计有效性与安全停止证书的自动化发现框架。

## 核心贡献（创新点）
1. **AASD 形式化与溯因效用定义**：首次将可能论与 anytime-valid 推断结合，定义单调可计算的溯因效用 $U_t$，为开放式发现提供无需人工干预的进度度量。
2. **两阶段可能论评估框架**：将假设分类为矛盾/开放/认证三类，并以“证据充分 > 背景兼容 > 简洁性”的词典序进行排序，与现有仅依赖惊讶或先验权重的打分机制本质不同。
3. **Possibility Frontier Search 算法与可观测停止证书**：提出基于前沿对 maximin 设计的主动实验策略，并推导仅依赖当前版本空间与背景轮廓的 $\varepsilon$-最优停止条件，弥补了现有方法缺乏安全早停保证的缺陷。
4. **渐近恢复与样本复杂度理论**：证明在合理生成器覆盖率假设下，算法以指数衰减率达到 $\varepsilon$-最优溯因效用，并给出显式实验与生成步数上界，为后续理论分析提供可直接复用的范式。

## 方法详解
- **问题设定**：环境模型 $W_\theta$（$\theta\in\Theta$），动作空间 $\mathcal{A}$，观测 $M_t|A_t,\theta\sim\mathbb{P}_\theta(\cdot|A_t)$，累积记录 $\mathcal{D}_t$。目标是发现含真状态 $\theta^\star$ 且兼容背景知识 $K$ 的假设 $H$。
- **证据可能性（Evidential Possibility）**：基于 anytime-valid e-process $E_t(\theta)$ 构造 $\pi_t(\theta)=\min_{s\leq t}\{1, E_s(\theta)^{-1}\}$，定义版本空间 $V_t=\{\theta:\pi_t(\theta)>\delta\}$。假设按与 $V_t$ 的关系分为矛盾集 $\mathsf{F}_t$、开放集 $\mathsf{O}_t$、认证集 $\mathsf{C}_t=\{H:H\cap V_t\neq\emptyset, V_t\subseteq H, H\neq\Theta\}$。命题1证明在事件 $\mathcal{E}_\delta$ 上真假设永不被剪枝，已认证假设保持认证，错误认证联合概率 $\leq\delta$。
- **背景可能性（Background Possibility）**：固定先验轮廓 $\pi^K$（可由 LLM 或历史 GP 数据生成），定义兼容效用 $u(H)=\inf_{\theta\in H}\pi^K(\theta)$。命题2证明该下确界等价于将 $H$ 内状态强制保留所需的最小轮廓 $L_\infty$ 扰动代价，体现“最小知识修订偏好”。
- **词典序选择**：从 $\mathsf{C}_t$ 中选出 $u(H)$ 最大者，平局时偏好状态数更少的假设（$|H|$），即 $\widehat{H}_t=\arg\min|H|$ s.t. $u(H)=U_t$，对应优先级：证据认证 $>_{\text{lex}}$ 背景兼容 $>_{\text{lex}}$ 简洁性。
- **假设扩展（Expand）**：每轮至多生成 $B$ 个新假设，接受条件为 $H\neq\Theta$ 且 $H\cap V_t\neq\emptyset$；一旦生成即满足 $V_t\subseteq\widetilde{H}$ 则当轮提前停止。
- **前沿搜索与实验设计**：定义可改进开放集 $\mathsf{O}_t^+$ 与前沿对集合 $\mathcal{P}_t$。下一轮动作分布按 maximin 准则选取：$\rho_{t+1}=\arg\max_\rho\min_{(\
