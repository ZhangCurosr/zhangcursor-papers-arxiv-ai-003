---
title: "UNROLLED-FLOW-MODELS-FOR-REASONING"
source: https://arxiv.org/pdf/2610.09759v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-09 03:25:46"
field: "流匹配与神经推理"
keywords: ["flow matching", "reasoning", "latent rollout", "sphere retraction", "recursive reasoning", "structured reasoning"]
innovations: ["终端展开损失：仅在 rollout 终点施加监督，梯度经末段步回传使多步推理有效", "球面重traction：以固定半径投影更新基底稳定长 rollout 动态", "无参 margin 选择：多 rollout 采样后用最大-次大 logit gap 均值选优"]
benchmarks: ["ProsQA", "Sudoku-Hard", "Sudoku-Extreme", "Maze-Hard"]
---

# 论文速读：UNROLLED-FLOW-MODELS-FOR-REASONING

## 一句话总结
论文提出 **Unrolled Flow Model (UFM)**，通过将流匹配模型的训练目标改为"沿自身潜在展开（latent rollout）传递到终端并仅在终点解码"，使额外积分步骤能够在图可达、数独、迷宫等推理任务中切实提升性能；同时引入球面重traction（sphere retraction）稳定长展开的动态，以仅 8.4M 参数的模型大幅超越 28.6M 的标准流语言模型基线。

## 研究问题与动机
- **流模型能否通过更多积分步骤实现更深推理？** 流匹配能以极少步数完成语言生成，但"增加 Euler 步数是否等价于增加推理深度"尚无定论。
- **标准流目标存在结构性缺陷。** FLM / S-FLM 在每个采样时间点 $t \sim \mathcal{U}[0,1]$ 独立监督 denoiser 还原目标 $x_1$，从未评估"某一步产生的状态是否对后续步骤有用"，导致额外步数几乎不带来精度增益。
- **递归/潜在推理已有并行思路。** Zhu et al. (2025) 与 Coconut (Hao et al., 2026) 证明在连续潜在空间中重复应用网络可实现逐步扩展可达集合，但如何将这一性质训练到可泛化的流模型中仍是开放问题。
- **长展开的动态不稳定。** Sudoku/Maze 等任务需要数十步甚至更多积分步，潜在范数会爆炸式增长，需要显式稳定机制。

## 核心贡献（创新点）
1. **构造性定理：两层 Transformer 流可精确求解图可达性。** 第一层缓存边/候选/根节点嵌入，第二层线性注意力头实现邻接算子 $A_\mathcal{G}$ 的速度场；其 Euler 离散化每步恰好将可达集合向外推进一跳，所需步数等于答案到根的距离。
2. **提出 UFM 的终端展开训练范式。** 将损失仅施加在 rollout 终点，梯度流经最后 $N_{\text{back}}$ 步，迫使每个 Euler 步的输出状态成为下一步的有效起点，从而让"步数增多 → 推理加深"在训练层面成立。
3. **球面重traction（sphere retraction）稳定长 rollout。** 以 $\psi(z) = \sqrt{d}\, z/\|z\|_2$ 将每步的更新基底投影到固定半径球面，抑制潜在范数爆炸（无 retraction 时在 Sudoku-Extreme 上范数放大 44,000 倍）。
4. **小参数显著超越大参数流基线。** 8.4M 参数的 UFM 在 Sudoku-Hard 达 86.9%、Sudoku-Extreme 74.4%、Maze-Hard 89.3%，而 28.6M 的 FLM/S-FLM 仅 51.9%/50.9%、10.7%/9.4%、40.4%/49.9%。
5. **多 rollout + 免参数 margin 选择逼近 Pass@K。** 采样 100 条 rollout 并以最大-次大 logit 均值 gap 选优，在 Sudoku-Extreme 上由 74.4% 提升至 98.6%。

## 方法详解
### 核心架构
- 使用双向 DiT 作为 denoiser，输入为当前潜在状态 $z_t \in \mathbb{R}^{L \times d}$、时间 $t$ 与 prompt $c$。
- **速度场定义**沿用条件流匹配：
  $$v_\theta(x,t) = \frac{\widehat{x}_\theta(x,t) - x}{1-t}$$
  Euler 积分：$z_{t_{k+1}} = z_{t_k} + \Delta t_k \cdot v_\theta(z_{t_k}, t_k)$。
- **关键差异**：中间状态 $z_{t_k}$ 不再投影到词表分布，**仅在终点** $z_{t_N}$ 施加线性解码器：
  $$\hat{y} = \text{softmax}(z_{t_N} W_{\text{dec}}^\top) \quad (\text{长序列用 StableMax})$$

### 终端展开损失
训练时从随机子区间起点 $t_0$ 初始化插值状态 $z_{t_0} = (1-t_0)\xi + t_0 \, \text{Embed}(y)$，在 $[t_0, 1]$ 上采样 $N_{\text{train}}$ 步 grid 做 rollout，损失仅计算在 $t_N=1$ 处：
$$\mathcal{L}_{\text{UFM}}(\theta) = \mathbb{E}_{(c,y),\xi,t_0}\!\left[\text{CE}\!\left(\text{softmax}(z_{t_N}W_{\text{dec}}^\top),\, y\right)\right]$$

### 截断反向传播（Truncated Backprop）
- 对长任务取 $N_{\text{train}}=24$，仅对最后 $N_{\text{back}}=6$ 步记录梯度，前 18 步以 stop-gradient 方式计算以节省激活显存。
- 每个样本独立采样递增时间 grid，暴露模型于不同步长配置。

### 球面重traction（Sphere Retraction）
定义 $\psi(z) = \sqrt{d}\, z/\|z\|_2$。对 $k>0$ 的步骤，将更新基底由 $z_{t_k}$ 替换为 $y_{t_k}=\psi(z_{t_k})$，但 denoiser 仍读取原始 $z_{t_k}$：
$$z_{t_{k+1}} = y_{t_k} + \frac{\Delta t_k}{1-t_k}\big(\widehat{x}_\theta(z_{t_k}, t_k, c) - y_{t_k}\big), \quad y_{t_{k+1}} = \psi(z_{t_{k+1}})$$
该投影使欧拉类型的更新基底保持在固定半径球面上，切断范数正反馈。

### 多 rollout 选择
- 推理时采样 $K=100$ 条 rollout，计算无参 margin 分数：
  $$\hat{\delta} = \frac{1}{L}\sum_{i=1}^L \big(\ell_i^{(1)} - \ell_i^{(2)}\big)$$
  其中 $\ell_i^{(1)}, \ell_i^{(2)}$ 为位置 $i$ 上前两大的原始 logit；选取 $\hat{\delta}$ 最高者输出。

## 实验与结果
### 数据集与设置
| 基准 | 任务描述 | 数据规模 |
|---|---|---|
| **ProsQA** | 有向图可达性（2 候选），距离 ≤ 4 hop | 14.8k train / 419 test |
| **Sudoku-Hard** | 30 给定数独，81 格答案 | 48k train / 2k test |
| **Sudoku-Extreme** | 1k base × 1000 aug，81 格 | 422,786 test |
| **Maze-Hard** | 30×30 迷宫，最短路径 > 110 格，900 格输出 | 1k train / 1k test |

所有 FLM/S-FLM 基线使用作者开源代码复现，UFM 与基线共享训练数据。

### ProsQA：步数增益对比（图 2）
- FLM / S-FLM：准确率随 $N$ 增大几乎**平坦**（~12% 无提升）。
- UFM（$N_{\text{train}}=5$，全梯度）：从 $N=1$ 的 12% 升至 $N$ 较大时的 **≈97%**，验证终端展开训练使步数有效。

### 长任务单 rollout（Pass@1，$N_{\text{eval}}=128$，表 3/4）
| 方法 | 参数 | Sudoku-Hard | Sudoku-Extreme | Maze-Hard |
|---|---|---|---|---|
| FLM | 28.6M | 51.9 | 10.7 | 40.4 |
| S-FLM | 28.6M | 50.9 | 9.4 | 49.9 |
| **UFM** | **8.4M** | **86.9 ± 0.9** | **74.4 ± 0.0** | **89.3 ± 0.5** |

UFM 以 **≈1/3 参数** 在三个基准上均大幅领先，Sudoku-Hard 提升约 +35pp。

### 多 rollout 选择（表 5，$K=100$）
- Sudoku-Extreme：74.4% → **98.6%**（margin 选择回收了大部分 Pass@100 差距）。
- Maze-Hard：89.3% → **92.0%**（仍低于 Pass@100 的 95.0%，长答案选择困难）。

### 对比递归推理器（表 4）
UFM 单 rollout 逊于最强的 TRM / EqR / FPRM 等，但优于早期 FLM/S-FLM；多 rollout 后在 Sudoku-Extreme 上接近 GRAM/PTRM。

## 相关工作脉络
1. **Zhu et al. (2025) Reasoning by Superposition。** 证明两层 Transformer 可在连续潜在空间中维护"可达顶点叠加态"并逐跳扩展；本文的理论构造直接基于此工作，但将其从静态超位置推广到可训练的流动力学。
2. **Coconut (Hao et al., 2026)。** 提出在语言模型隐空间中维持连续思考并用 curriculum 逐步替换 token 推理；与 UFM 本质不同：Coconut 保留自回归 token 输出并用课程学习，UFM 完全在潜在空间做端到端展开并仅在终点解码。
3. **FLM / S-FLM (Lee et al., 2026; Deschenaux & Gulcehre, 2026)。** 标准流语言模型基线，均在单时间点训练交叉熵；本文指出其无法从多步中获益的根源，并提出终端损失替代方案。
4. **FRM (Helbling et al., 2026)。** 向流模型引入递归自条件与 fixed-point forcing，同样报告"增加 FLM 采样步不提升数独成绩"；但 FRM 对 carry 停止梯度，而 UFM 对末段 $N_{\text{back}}$ 步允许梯度流通。
5. **Looped Flows (Suleymanzade et al., 2026)。** 同期工作，在 TRM 协议下训练带局部去噪损失的循环流；本文采用单一终端损失 + 截断反向传播，两者训练信号来源不同。
6. **Recursive Reasoners (HRM/TRM/GRAM/PTRM/EqR/FPRM)。** 周定权重的循环网络反复更新隐状态；UFM 与之共享"重复应用同一网络 + 隐状态扩展"范式，但以流匹配的 DiT  backbone 和连续时间 formulation 实现。

## 局限性与未来方向
- **模型规模受限。** 展开训练需保留激活以供反向传播，本文仅做到 8–16M 参数；扩展到大型 LLM 需要更省显存的 rollout 训练方案。
- **理论构造 vs. 训练恢复之间的鸿沟。** 正交嵌入、无 softmax 线性头、固定增益等假设在训练中并不成立；empirical UFM 仅呈现与构造相似的"可达顶点分离"模式，未证明梯度下降能恢复定理 1 的速度场。
- **长答案的多 rollout 选择仍不可靠。** Maze-Hard 的 margin 选择仅覆盖 92.0%（Pass@100 为 95.0%），平均 900 格中仅有 ~110 格与路径相关，稀释了 gap 信号；需要 learned halting / reward head（如 GRAM 的 latent reward model）替代无参选择。
- **合成结构化任务局限。** 尚未在自然语言推理或非结构化语言任务上验证，泛化性待考察。

## 研究启发与可借鉴点
1. **"终端展开损失"可作为通用范式。** 凡涉及"重复应用同一网络逐步精炼隐状态"的架构（diffusion sampler fine-tuning、recurrent reasoner、looped flow），均可借鉴 UFM 的终端 CE + 截断 BPTT 组合，使中间步骤真正服务于最终决策。
2. **球面重traction 是一种廉价稳定的 norm 正则手段。** 当循环/展开动态出现范数正反馈时，以固定半径投影更新基底（而非状态本身）可在不阻断 denoiser 读取原始状态的前提下稳定训练，值得在其他 ODE-based 生成模型中尝试。
3. **多 rollout + 无参 margin 选择在短答案上非常有效。** 对答案长度有限（如数独 81 格）的任务，可以直接用"最大-次大 logit gap 均值"做低成本选优，无需额外训练 scorer；这对需要 Pass@K 评估的推理 benchmark 是实用的 baseline。
4. **训练时随机子区间 rollout + 变步长 grid** 增强了模型对不同积分长度的鲁棒性；推理时只需固定 grid 即可复现，是一种低成本的"尺度不变性"训练技巧。
5. **可与本团队方向结合的创新机会。** 将 UFM 的终端展开思想迁移到 **流匹配强化学习**（从状态-动作轨迹 rollout 末端计算 returns 并回传）、或 **推理型 diffusion policy**（在 latent space 多次 Denoising 后仅在末端解码 action）中，有望获得类似的可扩展推理增益。

## 关键术语表
- **Flow Matching**：将噪声分布通过连续时间 ODE 映射到数据分布的生成建模框架，通过最小化速度场预测误差训练。
- **Euler Rollout**：对 flow 对应的 ODE 做离散 Euler 积分，依次更新潜在状态 $z_{t_{k+1}} = z_{t_k} + \Delta t_k v_\theta(z_{t_k}, t_k)$。
- **Unrolled Flow Model (UFM)**：本文提出的模型，仅在 rollout 终点应用词表解码器并计算 CE 损失，梯度经末段 $N_{\text{back}}$ 步回传。
- **Sphere Retraction**：将更新基底投影到半径 $\sqrt{d}$ 的球面 $\psi(z)=\sqrt{d}\,z/\|z\|_2$，以控制长 rollout 中的潜在范数爆炸。
- **Terminal Rollout Loss**：只在展开的最后一个时间点 $t_N=1$ 施加监督损失，训练步骤之间形成时序依赖。
- **Truncated Backpropagation**：仅对 rollout 末尾 $N_{\text{back}}$ 步记录梯度，之前步骤以 stop-gradient 运行以节省激活显存。
- **Margin-based Rollout Selection**：以各位置最大与次大 logit 的均值差 $\hat{\delta}$ 作为无参得分，从多条 rollout 中挑选最优。
- **Pass@K**：在 $K$ 次独立 rollout 中至少有一条正确的概率，用于评估多轨迹采样的推理可靠性。

## 可复现要素
- **数据集**：ProsQA、Sudoku-Hard（Deschenaux & Gulcehre 2026 划分）、Sudoku-Extreme（Wang et al. 2025 / Jolicoeur-Martineau 2025 协议）、Maze-Hard（30×30，1k train / 1k test）；均为公开合成基准。
- **代码**：论文声明可在 arXiv 页面找到实现（"The implementation can be found here"，见原文 abstract）。
- **权重**：论文未声明开源权重，仅提供 checkpoint selection 与超参细节。
- **关键超参**：
  - DiT：2 层，hidden width 448，8 头，8.4M 参数；pre-norm RMSNorm + SwiGLU（×4）+ adaLN-Zero + 2D RoPE。
  - 训练：AdamW $\beta=(0.9, 0.95)$，grad clip=1.0，dropout=0.1，EMA decay=0.999，cosine LR $10^{-4} \to 10^{-5}$。
  - $N_{\text{train}}=24$，$N_{\text{back}}=6$；$t_0 \sim \mathcal{U}[0, t_{0,\max}]$（$t_{0,\max}$ 从 0.6 线性降至 0.2）。
  - 推理：$N_{\text{eval}}=128$ 均匀 grid；多 rollout 时 $K=100$。
  - 噪声尺度 $\sigma$：Sudoku 任务 $1/\sqrt{448}$，Maze 任务 1。
  - Label smoothing：Sudoku 0.1，Maze 0。
