---
title: "RIDE-REFERENCE-ANCHORED-INFERENCE-TIME-DIFFUSION-EDITING-FOR"
source: https://arxiv.org/pdf/2609.35623v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 22:08:56"
field: "基于扩散模型的药物分子生成"
keywords: ["scaffold hopping", "diffusion editing", "molecular generation", "inference-time optimization", "reference-anchored", "value-guided sampling"]
innovations: ["参考锚定扩散编辑范式：将scaffold hopping reformulate为参考噪声轨迹编辑而非条件生成", "参考最优扰动片段选择：通过MC采样自动定位最佳编辑时间区间", "价值引导骨架采样：基于lookahead samples的值函数估计，适配不可微分子奖励"]
benchmarks: ["CrossDocked2020 test set (60 human disease targets)", "Sim_2D, Sim_3D, Vina S/M/D, QED, SA, Conn.(%)"]
---

# 论文速读：RIDE-REFERENCE-ANCHORED-INFERENCE-TIME-DIFFUSION-EDITING-FOR SCAFFOLD HOPPING

## 一句话总结
RIDE 是一种推理时扩散编辑框架，将 scaffold hopping 重新表述为参考锚定的轨迹编辑过程，通过在参考扩散噪声轨迹上选择最优编辑片段并进行价值引导采样，实现**低 2D 相似性 + 高 3D 相似性**的骨架生成，平均提升 Sim_2D 降低 11.7%、Sim_3D 提升 7.3%。

## 研究问题与动机
1. **核心任务**：Scaffold hopping 需发现与参考配体共享关键功能基团和相似 3D 形状，但 2D 结构不同的新骨架。
2. **现有方法缺陷**：当前扩散方法将问题表述为"给定功能基团的骨架条件生成"，仅依赖采样随机性获得多样性，**缺乏联合保证 2D 新颖性与 3D 形状保持的机制**。
3. **SBDD vs LBDD 的割裂**：基于结构的 SBDD 方法可利用蛋白口袋信息但缺乏显式 2D 新颖性优化；基于配体的 LBDD 方法可探索结构差异但缺乏蛋白结合指导。
4. **推理时编辑的空白**：虽有反演扩散编辑方法，但面向分子生成（离散+连续混合空间）的研究仍稀缺，且难以处理不可微的分子性质奖励。

## 核心贡献（创新点）
1. **参考锚定扩散编辑范式**：将 scaffold hopping 从条件生成 reformulate 为"参考噪声轨迹编辑"，本质区别在于利用参考轨迹作为 3D 形状锚点，而非从零采样。
2. **参考最优扰动片段选择**：通过 MC 采样在每个候选时间步评估编辑潜力，自动定位最适合引入 2D 变化的轨迹区间，解决"在哪编辑"问题。
3. **价值引导骨架采样**：针对分子生成中常见的不可微奖励函数，推导并实现了基于 lookahead samples 的值函数估计，在扰动片段内进行贪婪价值搜索，解决"如何引导编辑"问题。
4. **即插即用推理时框架**：无需重新训练预训练 SBDD 扩散模型，直接将 RIDE 适配至 conDitar、IPDiff、DiffHopp 等多个基线，验证了通用性。

## 方法详解
**整体流程**（三阶段）：

### 阶段一：参考噪声轨迹恢复
- 对参考骨架 $S^{\mathrm{ref}}=(X^{\mathrm{ref}}, V^{\mathrm{ref}})$ 执行反演：
  - 位置轨迹：DDIM 反演公式 (1) 恢复 $\{X_t^{\mathrm{ref}}\}$
  - 类型轨迹：分类扩散反演公式 (3)-(4) 恢复 $\{V_t^{\mathrm{ref}}\}$
  - 导出对应噪声序列 $\{\epsilon_t^{\mathrm{ref}}, g_t^{\mathrm{ref}}\}_{t=1}^T$
- 用算子 $\mathcal{T}_{\theta, t_1:t_2}^{\mathrm{ref}}$ 表示沿参考噪声从 $t_1$ 过渡到 $t_2$ 的反向转移

### 阶段二：参考最优扰动片段选择
- 在 $[t_2, t_1]$ 区间内，将参考噪声替换为采样噪声：
  - $p_{\phi_\epsilon}(\epsilon_t|t_1) = \mathcal{N}(0, I)$（位置）
  - $p_{\phi_g}(g_t|t_1) = \mathrm{Gumbel}(0,1)$（类型）
- 离散化候选 $t_1$，对每个 $t_1$ 生成 $K$ 个 MC 样本，计算评分：
  $$J(t_1|S^{\mathrm{ref}}) = \frac{1}{K}\sum_{k=1}^K \mathcal{R}(S^{(k)}(t_1), S^{\mathrm{ref}})$$
- 选 $t_1^* = \arg\max J(t_1)$，$t_2^* = \max(t_1^*-L, 0)$

### 阶段三：价值引导骨架采样
- **值函数定义**（定理 1）：
  $$V_\phi(S_t|S_{t_1^*}^{\mathrm{ref}}) = \frac{\mathbb{E}[w(S_{t_2^*}; S_t, S_{t_1^*}^{\mathrm{ref}}) \cdot \mathcal{R}(\mathcal{T}_{\theta, t_2^*:0}^{\mathrm{ref}}(S_{t_2^*}), S^{\mathrm{ref}})]}{\mathbb{E}[w(S_{t_2^*}; S_t, S_{t_1^*}^{\mathrm{ref}})]}$$
  其中权重 $w = q(S_t|S_{t_2^*}) / q(S_{t_1^*}^{\mathrm{ref}}|S_{t_2^*})$
- **蒙特卡洛近似**（定理 2）：
  $$\widehat{V}_\phi(S_t) = \sum_{m=1}^M \mathrm{softmax}_m(\{\ell_t^{(j)}\}) \cdot \mathcal{R}(\cdot), \quad \ell_t^{(m)} = \log q(S_t|S_{t_2^*}^{(m)}) - \log q(S_{t_1^*}^{\mathrm{ref}}|S_{t_2^*}^{(m)})$$
- 在每个引导时间步选择最大化 $\widehat{V}_\phi$ 的噪声，迭代更新至 $t_2^*$

### 奖励函数
$$\mathcal{R}(S, S^{\mathrm{ref}}) = \lambda(1 - \mathrm{Sim}_{2\mathrm{D}}) + (1-\lambda)\mathrm{Sim}_{3\mathrm{D}}$$
其中 $\lambda \in [0,1]$ 平衡 2D 新颖性与 3D 保持。

## 实验与结果
**数据集**：60 个与人类主要疾病相关的蛋白-配体复合物（来自 CrossDocked2020 派生测试集，无重叠）

**基线**：conDitar-a、IPDiff-a、DiffHopp、ShEPhERD

**主要结果**（Table 1）：

| 模型 | Sim_2D↓ | Sim_3D↑ | Vina S↓ | Conn.(%)↑ |
|------|---------|---------|---------|-----------|
| conDitar-a | 0.398 | 0.795 | -7.192 | 85.2 |
| **RIDE^(1)+conDitar-a** | **0.354** | **0.884** | -7.508 | **91.5** |
| IPDiff-a | 0.420 | 0.856 | -7.706 | 94.7 |
| **RIDE^(1)+IPDiff-a** | **0.363** | **0.903** | -7.422 | **93.1** |
| ShEPhERD | 0.417 | 0.881 | -6.471 | 29.4 |

- 平均 Sim_2D 降低 **11.7%**，Sim_3D 提升 **7.3%**
- RIDE^(1) 在 conDitar-a 上 Sim_3D 达 **0.884**，超过 ShEPhERD（0.881）
- Vina S 较 conDitar-a 和 DiffHopp 平均提升 **39.1%**
- Conn.(%) 提升 **5.0%**

**关键消融**：
- 价值引导 vs 随机扰动：Sim_2D 降低 8.3%，Sim_3D 提升 1.6%
- 轨迹反演贡献：去除反演后 Sim_3D 从 0.884 降至 0.852（RIDE^(1)）
- 双跳 RIDE^(2)：Sim_2D 进一步降至 0.322-0.325，但 Vina S 从 -7.48 降至 -7.26（存在 trade-off）

## 相关工作脉络
1. **SBDD 扩散模型**（IPDiff, conDitar）：利用蛋白口袋信息生成结合分子——RIDE 在其基础上适配条件并引入参考轨迹
2. **LBDD 扩散模型**（ShEPhERD）：以 3D 形状/静电势为条件的配体生成——RIDE 结合口袋信息实现双重约束
3. **Scaffold hopping 专用模型**（DiffHopp, Turbohopp）：条件骨架生成——依赖随机性，缺乏显式新颖性优化
4. **反演扩散编辑**（Null-text inversion, DDIM inversion）：图像编辑中恢复轨迹再编辑——RIDE 将其扩展到离散+连续混合的分子空间
5. **推理时对齐**（DPS, value-based decoding）：梯度/价值引导扩散生成——RIDE 解决分子领域不可微奖励的挑战

## 局限性与未来方向
1. **多跳 binding affinity 衰减**：RIDE^(2) 中 Vina S 下降，说明属性偏移在多跳中累积，需设计更稳定的迭代策略
2. **原子数固定限制**：编辑后骨架原子数与参考相同，限制了更大幅度的结构变化
3. **奖励函数敏感**：λ 的选择影响 Sim_2D/Sim_3D 权衡，需针对具体任务调参
4. **未验证实验合成可行性**：SA 分数略有下降，实际合成性需进一步验证

## 研究启发与可借鉴点
1. **反演-编辑范式可迁移**：将目标样本反演到扩散轨迹后在片段内编辑，适用于任何需要"保留部分特征+引入变化"的可控生成任务
2. **值引导处理不可微奖励**：通过 lookahead samples 估计值函数，为分子生成、材料设计等领域的黑盒奖励优化提供了实用方案
3. **Sim_3D 隐式保持效应**：即使奖励函数中不含 Sim_3D，参考轨迹机制仍能保持较高 3D 相似性，说明反演恢复本身具有结构保护属性，可借此简化奖励设计
4. **片段扰动策略**：仅在轨迹局部区间扰动而非全路径，平衡了编辑自由度与特征保持，比直接编辑中间状态更稳定

## 关键术语表
**Scaffold Hopping**：药物设计中用不同 2D 骨架替换已知配体骨架，同时保持 3D 形状与结合模式的任务
**Reference-anchored**：以参考配体的完整扩散噪声轨迹为锚点，确保编辑过程不偏离参考的 3D 结合特征
**DDIM Inversion**：将参考样本反演到扩散过程各时间步的中间状态，恢复其对应的噪声序列
**Value-guided Sampling**：通过估计中间状态的条件期望奖励值来引导采样，避免依赖梯度，适用于不可微奖励
**Sim_2D / Sim_3D**：分别基于分子指纹 Tanimoto 距离和 ROCS 计算的 2D/3D 结构相似度
**Vina S/M/D**：AutoDock Vina 预测的结合亲和力得分，分别对应原始 pose、局部优化后和 docking 后
**QED**：Drug-likeness 的定量综合评估指标（0-1 范围）
**RIDE^(2)**：双跳 scaffold hopping，将 RIDE 生成分子作为新参考进行第二轮编辑

## 可复现要素
- **代码**：已开源，https://anonymous.4open.science/r/RIDE-C8A0
- **数据集**：公开数据集（CrossDocked2020 派生测试集，与训练集无重叠）
- **关键超参**：
  - 扩散步数 T：conDitar-a/IPDiff-a 用 1000，DiffHopp 用 500
  - 片段搜索粒度 N：conDitar-a/IPDiff-a 用 10，DiffHopp 用 5
  - 候选范围 n1/n2：conDitar-a/IPDiff-a 用 5/9，DiffHopp 用 3/5
  - MC 样本数 K=100（片段选择），M=1000（值估计）
  - 扰动长度 L=100
  - λ：RIDE^(1) 片段选择=1，价值引导=0.8；RIDE^(2) 分别为 0.7 和 0.5（IPDiff-a）
