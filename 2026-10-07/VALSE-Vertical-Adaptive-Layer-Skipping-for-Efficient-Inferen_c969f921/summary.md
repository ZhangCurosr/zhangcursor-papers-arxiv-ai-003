---
title: "VALSE-Vertical-Adaptive-Layer-Skipping-for-Efficient-Inferen"
source: https://arxiv.org/pdf/2610.07606v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-07 22:28:56"
---

# 论文速读：VALSE-Vertical-Adaptive-Layer-Skipping-for-Efficient-Inferen

## 一句话总结
提出VALSE（Vertical Adaptive Layer Skipping），通过首层浅层特征提取显式难度分数，实现逐样本非连续Transformer层选择性跳过，将条件计算从横向宽度稀疏（MoE）拓展至纵向深度稀疏；构建完整的期望FLOPs闭式解、函数空间包含关系及与MoE的结构对偶性理论，并在原型尺度（12层11.5M参数SST-2分类器）上透明报告了路由坍缩的负结果，为后续大规模验证提供理论基准与故障诊断路径。

## 研究问题与动机
1. **计算冗余与深度利用率不均**：现代LLM无论输入难度高低均须遍历全部数十至数百个Transformer块，简单样本承受与复杂样本完全相同的深度计算代价。
2. **现有自适应深度方法的结构性局限**：Early exit类方法为截断式，只能保留连续前缀；MoD等逐token路由缺乏样本级难度语义；LayerDrop/LayerSkip为静态或逐token选择，非样本驱动。
3. **深度方向稀疏性未被形式化**：横向MoE实现了每层宽度方向专家稀疏，但网络深度固定；垂直方向的条件计算缺乏与宽度稀疏对等的理论框架与路由机制。
4. **缺乏可解释的难度驱动信号**：现有方法多依赖隐式置信度或训练后静态剪枝，未定义显式、单调的难度标量 $\hat{d}$ 来统一控制非连续层保留集合。

## 核心贡献（创新点）
1. **提出VALSE逐样本非连续层跳过机制**：通过轻量难度估计器生成 $\hat{d}$，驱动逐层门控 $g_l$，实现任意跳过中层而保留深层的 retention set $S^{(s)}$，突破截断式early exit的前缀限制。
2. **构建垂直稀疏条件计算的完整理论体系**：证明期望FLOPs闭式解（Theorem 2）、全层函数空间为VALSE函数空间的严格超集（Theorem 10/11），以及VALSE与MoE的管道结构对偶性（Theorem 6）。
3. **设计四维辅助损失与三阶段课程学习范式**：引入层利用率平衡、效率约束、KL输出一致性蒸馏与Z-loss，配合 $\rho_{\text{skip}}$ 渐进退火与Gumbel温度调度，系统缓解路由坍缩与训练不稳定。
4. **确立横向MoE与纵向VALSE的统一二维稀疏框架**：证明宽度稀疏率 $s_w$ 与深度稀疏率 $s_d$ 可相乘实现 $(1-s_w)(1-s_d)$ 的理论节约，完成稀疏Transformer的横纵双向计算图掩码形式化（Theorem 9）。
5. **透明报告原型负结果并给出理论诊断**：明确声明核心假设（难度-深度单调性）在12层原型上未获验证（Pearson $r=-0.017$，路由坍缩至最小深度），并基于Theorem 21给出坍缩盆地的三重成因与可执行的修复路线。

## 方法详解
1. **难度估计模块（Difficulty Estimator）**：从模型前 $K_{\text{est}}=2$ 层提取5维统计特征：隐藏状态范数 $\rho_l$、层间增量范数 $\Delta_l$、自注意力熵 $H_l^{\text{attn}}$、预测熵 $H_l^{\text{pred}}$、奇异值集中度 $\kappa_l$。经轻量MLP（$d_{\text{hidden}}=64$）映射为原始分数 $d(x)$，再经批内min-max归一化得 $\hat{d} \in [0,1]$；理论规格建议使用指数移动平均（EMA, $\lambda=0.9$）校准以实现跨batch可比性。
2. **层路由策略**：提供三种变体。VALSE-Hard（主方案）使用逐层可学习门控 $g_l = \sigma(w_l^\top[r_l;\hat{d}] + b_l)$，推理时硬量化（$\theta_l=0.5$）；VALSE-Soft采用连续门控便于初始化；VALSE-Global通过全局Top-K选择保留层数 $K = \text{round}(B_{\text{target}} - \hat{d}(B_{\text{target}}-L_{\min}))$。跳跃残差默认采用Scheme A恒等映射：$h_{l+1} = h_l + \tilde{g}_l F_l(h_l)$。
3. **训练目标函数**：$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \alpha \mathcal{L}_{\text{balance}} + \beta \mathcal{L}_{\text{eff}} + \delta \mathcal{L}_{\text{consist}} + \gamma \mathcal{L}_{z}$。$\mathcal{L}_{\text{balance}}$ 利用AM-GM不等式鼓励各层均匀激活；$\mathcal{L}_{\text{eff}}$ 为单侧hinge约束激活层数不超过 $B_{\text{target}}$；$\mathcal{L}_{\text{consist}}$ 通过EMA教师与周期性刷新进行KL蒸馏；$\mathcal{L}_{z}$ 稳定门控logits数值分布。
4. **三阶段课程学习与稳定性判据**：Stage 1全层预热（$\rho_{\text{skip}}=0$，偏置 $b_l=2.0$ 使门控饱和开启）；Stage 2渐进跳过（难度感知目标 $\rho_{\text{target}}^{(s)}=\rho^*(1-\hat{d}^{(s)})$，Gumbel温度 $2.0\to0.5$，偏置退火）；Stage 3联合优化。训练反馈回路满足压缩映射条件 $\|\partial\hat{d}/\partial\theta_D\|\cdot\|\partial\mathcal{L}_{\text{eff}}/\partial\hat{d}\|\cdot\|\partial\tilde{g}/\partial\hat{d}\|\cdot\|\partial\mathcal{L}_{\text{task}}/\partial\tilde{g}\|<1$ 时保持稳定。

## 实验与结果
- **实验设置**：12层Transformer编码器（$d_{\text{model}}=256$, 4 heads, $d_{\text{ffn}}=1024$, vocab=8000，11.5M参数），$K_{\text{est}}=2$，CPU上从零训练18 epoch，batch size=32
