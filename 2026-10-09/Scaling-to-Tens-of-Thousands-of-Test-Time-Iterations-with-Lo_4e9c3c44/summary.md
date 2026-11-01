---
title: "Scaling-to-Tens-of-Thousands-of-Test-Time-Iterations-with-Lo"
source: https://arxiv.org/pdf/2610.11570v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-01 14:26:08"
field: "递归推理架构"
keywords: ["循环 Transformer", "测试时计算", "残差连接", "递归推理", "推理稳定化", "注意力残差", "自适应步长"]
innovations: ["提出循环原生注意力残差连接 InfiLoop，用内容加权与指数衰减聚合历史迭代状态替代 carry-last 规则", "推导精确流式递推公式使聚合内存恒定为 O(d) 且与迭代次数无关", "揭示模型涌现的自我验证行为：步长在求解时刻突增后冻结状态"]
benchmarks: ["Sudoku-Extreme", "Maze-Hard", "ARC-AGI-1", "ARC-AGI-2"]
---

# 论文速读：Scaling to Tens of Thousands of Test-Time Iterations with Loop-Native Attention Residuals

## 一句话总结
本文提出 **InfiLoop**，一种专为循环 Transformer 设计的"循环原生残差连接"，通过内容加权和指数时间衰减对历史迭代状态进行自适应聚合，解决了 carry-last 规则在深层循环中反复覆盖正确解、导致推理性能下降的问题；在 Sudoku-Extreme 上达到 97.9% 精确准确率，并在超过 20,000 次测试时迭代中持续提升。

## 研究问题与动机
1. **深层循环的精度下降现象**：循环 Transformer（如 TRM）在推理时增加迭代次数并不能持续提升精度——Sudoku-Extreme 上 TRM 在 468 层（18 次外层迭代）达到 62.2%，但在 1,872 层后精度反而降至 58.1%。
2. **carry-last 规则的本质缺陷**：当前主流方法（TRM、HRM 等）采用 carry-last 规则，将每次迭代的最新输出直接作为下一次迭代的输入，导致正确的中间推理结果被后续噪声更新覆盖。
3. **已有稳定化手段的不足**：FPRM 等方法通过预归一化、残差缩放和阻尼固定点更新来稳定循环，但只控制了"每次更新有多大影响"，未改变"下一次循环只能看到最新状态"这一根本架构约束。
4. **错误在深层中的长程传播**：早期循环中的错误会通过递推链向后续迭代传播，一旦前期产生错误，后续循环难以从中恢复。

## 核心贡献（创新点）
1. **发现 carry-last 在深层循环中的不稳定性**：首次通过系统性实验量化了在 Sudoku-Extreme 上随着有效深度增加，carry-last 模型会不断覆盖已解出的正确答案，而 InfiLoop 可永久保持已解状态。
2. **提出循环原生注意力残差连接（InfiLoop）**：用学习的内容分数与指数时间衰减对历史迭代候选状态进行聚合，替代标准 carry-last 规则，使模型能自适应地保留有用中间状态并抑制不可靠更新。
3. **推导精确流式递推公式，实现恒定内存开销**：利用共享评分器和几何衰减的可分离性，推导出仅需维护一个加权向量 $N_t$ 和一个标量 $D_t$ 的精确递推关系，持久聚合内存 $O(d)$ 与迭代次数无关。
4. **揭示涌现的自我验证行为**：学习到的自适应步长 $\gamma_t$ 在搜索阶段被压制（≈0.40），在找到完整解的瞬间突增至 0.91，随后回归基线，表明模型学会了"解出即冻结状态"的策略。

## 方法详解
**基本设定**：循环 Transformer 在每个时间步 $t$ 使用共享块 $f_\theta$，产生候选状态 $z_t = f_\theta(h_{t-1}; x)$，传统方法令 $h_t = z_t$（carry-last）。

**InfiLoop 的聚合公式**：
$$h_t = \sum_{i=0}^{t} \alpha_{t,i} z_i, \quad \alpha_{t,i} = \frac{\exp(s_i - \lambda(t-i))}{\sum_{j=0}^{t}\exp(s_j - \lambda(t-j))}$$
其中内容分数 $s_i = q^\top \text{RMSNorm}(z_i)$，$q \in \mathbb{R}^d$ 为跨迭代共享的学习伪查询，$\lambda > 0$ 为学习的时间衰减率。

**精确流式递推**：令 $e_i = \exp(s_i)$，$\beta = e^{-\lambda}$，则：
$$N_t = \beta N_{t-1} + e_t z_t, \quad D_t = \beta D_{t-1} + e_t, \quad h_t = N_t / D_t$$
初始值 $N_0 = e_0 z_0$，$D_0 = e_0$。只需维护 $O(d)$ 维向量 $N_t$ 和标量 $D_t$。

**自适应残差更新视角**：将递推重写为：
$$h_t = h_{t-1} + \gamma_t(z_t - h_{t-1}), \quad \gamma_t = \frac{e_t}{\beta D_{t-1} + e_t} \in (0,1)$$
$\gamma_t$ 类似于 GRU 的更新门，但由内容分数与历史质量的比值决定，无需额外门控网络。

**嵌套循环扩展**：对 TRM 风格的双轴递归（内层 L 次、外层 H 次），每个轴向独立配备一对 $(q, \beta)$ 和累加器 $(N, D)$，每轴仅增加 $d+1$ 个参数。

## 实验与结果
**数据集**：Sudoku-Extreme（422,786 个唯一谜题）、Maze-Hard（1,000 个唯一样本）、ARC-AGI-1（400 训练/400 评测）、ARC-AGI-2（1,000 训练/120 评测）。

**模型配置**：7M 参数，共享块含 2 层 Transformer（dim=512，8 头注意力，SwiGLU×4，3×3 深度卷积），双轴各配 513 参数的评分器-衰减对。

**主要结果**（Table 1）：

| 方法 | 参数量 | Sudoku-Extreme | Maze-Hard | ARC-AGI-1 | ARC-AGI-2 |
|---|---|---|---|---|---|
| TRM (复现) | 7M | 73.6 | — | — | 4.6 |
| FPRM | 7M | 94.2 | 87.0 | 47.5 | 6.2 |
| **InfiLoop** | **7M** | **97.9** | **87.4** | **49.1** | **13.6** |

- Sudoku-Extreme：InfiLoop 达 **97.9%**，超越 FPRM（94.2%）3.7 个百分点，超越复现 TRM（73.6%）24.3 个百分点。
- ARC-AGI-2：InfiLoop 达 **13.6% pass@2**，接近翻倍超越 TRM（7.8%）。
- 即便与 27M 参数的 HRM（Sudoku 55.0%、Maze 74.5%、ARC-1 40.3%）相比，7M InfiLoop 仍全面领先。

**测试时深度扩展**（Figure 3）：
- 在 1,872 层评估时，InfiLoop 达 90.9%，超出 FPRM 在 3,744 层（90.1%）的性能，**以一半计算量实现更好效果**。
- 在 24,960 层时，InfiLoop 达 92.74%，TRM 仅 83.65%（相差 9 个百分点）。
- 最难的 60 空格 Sudoku：InfiLoop 59.5% vs FPRM 56.5% vs TRM 36.0%。

**消融实验**（Table 2）：
- 双轴（内外层均应用 InfiLoop）优于单轴；H32 L6 时双轴 89.7% vs 仅内层 79.4% vs 仅外层 80.0%。
- 前向衰减（$\beta=0.55$）远优于均匀权重（38.4%）和反向衰减（2.0%）。
- 移除内容分数（$q=0$）使精度从 90.9% 降至约 84%（下降 6–7 个百分点）。
- $\beta=0$（即退化为 carry-last）时精度暴跌至 1.5%。

**状态扰动稳定性**（Figure 5）：InfiLoop 在扰动后保持 100% 已解谜题不被覆盖；TRM 在无扰动情况下已损失 15–20% 的已解谜题。

## 相关工作脉络
1. **Universal Transformer / 循环 Transformer**（Dehghani et al., 2019; Giannou et al., 2023）：用共享块反复执行以增加推理深度；InfiLoop 在此框架上引入跨迭代的记忆聚合机制。
2. **TRM / HRM / URM**（Jolicoeur-Martineau, 2025; Wang et al., 2025; Gao et al., 2025）：嵌套递归推理模型，均采用 carry-last 或类似规则传递状态；InfiLoop 可直接替换其状态传递方式。
3. **FPRM（固定点推理器）**（Movahedi et al., 2026）：通过预归一化、残差缩放和阻尼固定点迭代来稳定深层循环；侧重点在于更新幅度控制，而非历史信息的选择性保留。
4. **Attention Residuals**（Kimi Team et al., 2026）：在 Transformer 层间做跨层注意力聚合，但每层有独立查询、内存随深度线性增长；InfiLoop 将这一思想适配到循环迭代轴并实现恒定内存。
5. **深层网络稳定化方法**（ReZero、LayerScale、DeepNet、Sun et al., 2025）：通过零初始化残差门、深度依赖归一化缩放等缓解深度 curse；这些方法与 InfiLoop 正交，可结合使用。
6. **GRU / Highway Network**（Cho et al., 2014; Srivastava et al., 2015）：门控循环更新的公式形式与 InfiLoop 的 $\gamma_t$ 相似，但 InfiLoop 的门控由内容分数和历史质量比直接决定，无需额外网络参数。

## 局限性与未来方向
1. **仅在较小规模模型（7M 参数）上验证**：未见更大参数规模（如 70M 或更大）的扩展实验，其在更大循环 Transformer 上的有效性待进一步验证。
2. **任务集中于符号推理和视觉变换**：Sudoku、Maze、ARC 均为结构化和离散推理任务；在连续推理或语言生成类任务上的泛化能力未验证。
3. **时间衰减 $\beta$ 的超参数敏感**：消融显示不同初始化 $\beta_0$ 导致显著性能差异（0.10→66.3%，0.55→82.1%，0.90→63.3%），最佳 $\beta$ 依赖任务设定。
4. **内容分数 $s_i$ 的计算开销**：虽仅增加 $O(nd)$ 的 FLOPs，但在超长推理轨迹中累积的标量运算仍有一定开销。
5. **未来方向**：探索 InfiLoop 与 FPRM 等稳定化方法的组合；研究 InfiLoop 在更大循环深度（如 100K+ 次迭代）下的渐进行为；将其应用于更广泛的序列到序列推理任务。

## 研究启发与可借鉴点
1. **"信息保留机制"是深层循环的核心瓶颈**：本文精准定位了 carry-last 规则的缺陷而非一般性的深度 curse，这一诊断思路可直接迁移到其他递归/迭代推理架构的分析中。
2. **精确流式递推的设计范式**：将注意力聚合分解为 $N_t/D_t$ 两个可递推量，使 $O(T^2 d)$ 降至 $O(Td)$ 且内存恒定为 $O(d)$，这一技术路线可复用于其他需要历史记忆的递归模型。
3. **涌现自验证行为的分析框架**：通过对齐求解时刻观察 $\gamma_t$ 的动态变化，揭示了模型自发学会"求解后冻结状态"的策略——这种"对齐里程碑事件→观测内部信号"的分析方法值得在其他推理任务中推广。
4. **双轴独立聚合的模块化设计**：InfiLoop 对内外层分别应用独立连接的设计表明，嵌套递归架构的各轴可独立优化，为后续构建更深的多层递归推理器提供了模块化蓝图。
5. **与现有稳定化技术的正交性**：InfiLoop 解决的是"选什么历史"而非"更新多大"，与 ReZero、FPRM 等方法天然互补，可探索将 InfiLoop 嵌入 FPRM 的固定点迭代中，实现双重稳定。

## 关键术语表
**Looped Transformer（循环 Transformer）**：通过反复应用同一组参数的 Transformer 块来实现深度推理的架构，增加迭代次数即可增加有效深度而不增加参数量。

**Carry-last 规则**：将每次迭代的最新输出直接作为下一次迭代的输入状态（$h_t = z_t$），是 TRM 等模型的标准状态传递方式。

**Attention Residuals（注意力残差）**：Kimi Team (2026) 提出的跨层残差连接，通过伪查询对不同层的输出进行加权聚合；本文将其思想迁移至循环迭代轴。

**Adaptive Step Size $\gamma_t$（自适应步长）**：InfiLoop 中控制新候选状态与历史聚合状态融合比例的系数，形式上与 GRU 更新门相同，但由内容分数和历史质量的比值直接决定。

**Temporal Decay $\beta$（时间衰减）**：控制历史候选状态权重随迭代轮数衰减的指数因子（$\beta = e^{-\lambda}$），前向衰减（$\beta < 1$）使旧状态权重逐渐降低。

**Sudoku-Extreme**：Wang et al. (2025) 引入的高难度数独数据集，包含 422,786 个唯一谜题，训练集约 100 万样本，要求模型输出完整的 81 格填充结果。

**ARC-AGI-1 / ARC-AGI-2**：Chollet 提出的抽象推理与归纳数据集，评估模型从少量示例中学习变换规则并泛化的能力；ARC-AGI-2 更具挑战性，需要组合多个变换且抵抗暴力搜索。

**Exact Recurrence（精确递推）**：InfiLoop 的聚合操作可通过递推公式 $N_t = \beta N_{t-1} + e_t z_t$ 精确计算，无需存储全部历史状态，且无近似误差。

## 可复现要素
- **数据集**：Sudoku-Extreme、Maze-Hard、ARC-AGI-1、ARC-AGI-2（均有公开版本，论文引用了原始来源）。
- **代码**：开源，GitHub 地址 https://github.com/pixeli99/InfiLoop。
- **基础代码库**：基于公开 TRM 代码库（Jolicoeur-Martineau, 2025）实现。
- **关键超参**：AdamW 优化，峰值学习率 $10^{-4}$（Sudoku）/ $3\times10^{-4}$（Maze/ARC），weight decay 1.0，batch size 768；训练调度 H=3 外层、L=6 内层、12 个 deep-supervision segments；隐藏维度 512，8 头注意力，SwiGLU 扩展因子 4，3×3 深度卷积。
- **基线复现**：作者在论文中复现了 TRM（Sudoku-Extreme 73.6%、ARC-AGI-2 4.6%），并使用了 FPRM 官方发布检查点。
