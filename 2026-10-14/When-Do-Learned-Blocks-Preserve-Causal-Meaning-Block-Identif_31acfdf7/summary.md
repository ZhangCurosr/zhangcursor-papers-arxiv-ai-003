---
title: "When-Do-Learned-Blocks-Preserve-Causal-Meaning-Block-Identif"
source: https://arxiv.org/pdf/2610.18155v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-04 10:29:31"
---

# 论文速读：When-Do-Learned-Blocks-Preserve-Causal-Meaning-Block-Identif

## 一句话总结
本文证明了**块对角可逆变换与兼容坐标置换**构成保持任意基于固定块的构造性因果抽象的最大变换类（Thm. 1），并在合成加法自编码器实验中展示：通过拟合的块级映射跨独立训练表征传输高层概念干预，可近似保持其预期因果效果。

## 研究问题与动机
1. **块级表示的现实普遍性**：标准识别理论假设每个潜因子对应一个学习坐标，但实际模型（如SAE、对象中心模型、ISA、加法解码器）常将概念编码为多维块，块内坐标在识别意义下未定。
2. **干预语义的不确定性**：不同训练独立得到的表征可能对同一概念块使用不同内部坐标；在此自由度下进行高层干预，其因果意义是否会随表征变化而漂移尚未厘清。
3. **缺乏跨表征干预传输的理论边界**：现有构造性因果抽象框架提供了低层↔高层的一致性定义，但未刻画“哪些底层重参数化不会破坏该抽象”。
4. **工程实践的支撑需求**：特征组干预、对象槽操控、隐子空间钳位等方法在实践中广泛应用，需要形式化保证以确保跨模型迁移时干预目标仍可达成。

## 核心贡献（创新点）
1. **刻画最大不变变换类**：证明 $\mathcal{T}=\{\sigma \circ D \mid D \text{为块内可逆变换}, \sigma \text{为兼容置换}\}$ 是保证所有基于固定块的构造性因果抽象均被 preserved 的充要类。本质区别在于以往工作仅验证特定变换下的精确性，本文给出“对任意给定块对齐抽象普遍成立”的最强边界（Thm. 1）。
2. **桥接块识别与因果抽象理论**：将 ISA 与加法解码器的块识别结论纳入因果抽象框架，表明块识别遗留的自由度本身不破坏高层因果解释。本质区别在于把代数识别结果转化为对干预传输的结构性约束，而非仅停留在表征恢复层面。
3. **跨表征高层干预传输的实验验证**：在独立训练的加法自编码器上拟合块级仿射映射与置换，将参考模型的概念干预迁移至接收模型并解码评估。本质区别在于首次以定量方式演示“块识别不确定性下干预语义跨模型近似保持”的可行性。

## 方法详解
- **低层/高层SCM与对齐**：低层模型 $L$ 的坐标集 $\mathbf{V}_L$ 按块划分为 $\mathbf{V}^{(1)},\dots,\mathbf{V}^{(B)}$；高层模型 $H$ 含可解释变量 $\mathbf{V}_H=\{Z^{(1)},\dots,Z^{(B)}\}$。对齐 $\langle \Pi, \pi \rangle$ 将每个块 $\mathbf{V}^{(b)}$ 映射为细胞，细胞映射 $\pi_b: \text{Val}_{\mathbf{V}^{(b)}} \twoheadrightarrow \text{Val}_{Z^{(b)}}$ 可为多对一。
- **精确变换与构造性抽象**：若态映射 $\tau_\pi$ 与干预映射 $\omega_\pi$ 满足 $\tau(\text{Solve}(L^I)) = \text{Solve}(H^{\omega(I)})$，则为精确变换；进一步若 $\tau, \omega$ 由对齐诱导，则称 $H$ 为 $L$ 的构造性因果抽象（Defn. 2, 4）。
- **块重参数化群 $\mathcal{D}$**：$D = \text{Diag}(D^{(1)},\dots,D^{(B)})$，每个 $D^{(b)}$ 为块内双射，混合块内坐标但不跨块。
- **兼容置换 $\sigma$ 与输运细胞**：$\sigma$ 为值域一致的坐标置换，将原块 $\mathbf{V}^{(b)}$ 映射至可能交错的输运细胞 $\sigma(\mathbf{V}^{(b)})$。
- **定理1（核心）**：当且仅当 $T = \sigma \circ D$ 时，对任意目标 $H$ 与任意 $(\tau,\omega)\in \mathcal{A}(L,H)$，输运后 $(\tau \circ T^{-1}, \omega^T): L_T \to H$ 仍为构造性抽象，且细胞为 $\sigma(\mathbf{V}^{(b)})$。$\mathcal{T}$ 之外必存在至少一个抽象被破坏（如 XOR 交叉变换会使干预依赖当前状态、细胞不再不相交）。
- **干预实现与传输协议**：因 $\pi_b$ 多对一，采用最近纤维点规则 $\mathbf{x}_0^{(b)*} = \mathbf{x}_0^{(b)} + \frac{s_b^\star - \pi_b(\mathbf{x}_0^{(b)})}{\|\mathbf{w}_b\|^2}\mathbf{w}_b$ 选取低层代表；跨模型通过配对编码拟合仿射映射 $\widehat{D}_r^{(b)}$ 与最优置换 $\widehat{\sigma}_r$，得到 $\widehat{T}_r=\widehat{\sigma}_r \circ \text{Diag}(\widehat{D}_r^{(1)},\widehat{D}_r^{(2)})$，将编辑迁移至接收模型解码后与精确生成观测比较。

## 实验与结果
- **数据生成**：$(s_1,u_1,s_2,u_2) \sim \text{Uniform}[-1,1]$，$\phi(s,u)=(s,u,s^2,su,u^2)^\top$，$\mathbf{o}=A_1\phi(s_1,u_1)+A_2\phi(s_2,u_2)$，$A \in \mathbb{R}^{10\times10}$ 正交固定矩阵。
- **模型架构**：加法自编码器，编码器 $10\to64\to64\to4$，解码器两分支各 $2\to64\to64\to10$（tanh激活）。无因子标签参与训练。
- **训练与筛选**：20,000 训练 / 2,000 验证 / 5,000 测试样本；Adam lr=$10^{-3}$, batch=1024, weight decay=$5\times10^{-4}$；早停200 epochs。仅保留重构 NMSE $\le 0.01$ 的模型（10个独立训练中共4个通过）。
- **映射拟合**：参考块-概念仿射映射 $\pi_b$ 用标签数据拟合，test $R^2 \in [0.9996, 0.9996]$；跨模型配对映射 $\widehat{D}_r$ 用无标签配对编码拟合，validation NMSE $\in [0.0020, 0.0058]$。
- **干预评估**：对每个概念独立采样 $s_b^\star$，共 5,000 次干预/概念。参考模型与4个接收模型解码输出与精确生成观测的 NMSE 均 $\le 0.012$；最强接收模型 Recipient (3) 对 $Z^{(1)}$ 干预 NMSE=0.009342，对 $Z^{(2)}$ 干预 NMSE=0.010307。
- **结论**：在合成加法结构下，基于定理1的块级迁移协议可近似保持高层干预的因果效果，验证了块识别不确定性不会在实践中破坏构造性抽象的有效性。

## 相关工作脉络
1. **Exact transformation & Constructive causal abstraction** (Rubenstein et al., 2017; Beckers & Halpern, 2019; Geiger et al., 2025)：提供跨层因果一致性的基础框架；本文在其上限定“固定块对齐”并求解最大不变类，而非 discovering 新抽象。
2. **DAS** (Geiger et al., 2024)：搜索将高层变量投影至正交子空间的旋转对齐；本文不搜索对齐，而是分析给定对齐在表征变化下的稳定性边界。
3. **Independent Subspace Analysis (ISA)** (Hyvärinen & Hoyer, 2000; Theis, 2006) 与 **Additive Decoders** (Lachapelle et al., 2023)：给出块级代数识别保证；本文证明这些允许变换恰好落在保持构造性抽象的充分必要类内，实现两流派的理论统一。
4. **Sparse Autoencoders (SAEs)** (Bricken et al., 2023; Leask et al., 2025) 与 **Block-sparse featurizers** (Fel et al., 2026)：实践中概念分布于多维块且内部坐标未定；本文为其“块干预跨模型可迁移”提供形式化依据。
5. **Approximate causal abstraction** (Beckers et al., 20
