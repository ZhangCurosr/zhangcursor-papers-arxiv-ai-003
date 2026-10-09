---
title: "PROBABILITY-SIGNATURE-DYNAMICS-UNPACKING-MODULAR-ADDITION-LE"
source: https://arxiv.org/pdf/2610.11833v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-09 10:16:52"
---

# 论文速读：PROBABILITY-SIGNATURE-DYNAMICS-UNPACKING-MODULAR-ADDITION-LE

## 一句话总结
本文利用**概率签名（probability signatures）**框架，从训练数据分布的条件统计出发正向推导了两层网络在模加任务上学得的参数为何涌现傅里叶结构；同时定量解释了均匀随机标签噪声下受损样本早期损失下降更快的反直觉现象，并将同一算子对角化方法迁移至XOR任务，预测其权重将沿Walsh-Hadamard基底演化。

## 研究问题与动机
- **正向机制缺失**：已有工作通过机制干预证实训练后的网络会形成傅里叶电路并支撑泛化，但未回答“梯度优化为何能从模加数据分布中向前选出这些傅里叶模式”。
- **先验假设依赖**：现有理论分析（如He et al., 2026）基于权重的傅里叶分解，需提前预设傅里叶基底与模加相关；本文希望从零阶数据条件统计自然导出基底，不依赖该先验。
- **早期噪声谜题**：经验观察到均匀随机标签噪声下的受损样本在训练早期损失下降反而比干净样本更快，此前文献仅记录该现象而未给出可计算的数据侧解释。
- **框架泛化性未知**：概率签名能否超越模加任务，用于预测其他代数运算（如XOR）的表征基底，尚未验证。

## 核心贡献（创新点）
1. **数据→梯度的正向推导框架**：提出基于概率签名的主导梯度近似，直接从训练分布的条件概率提取关键统计量，无需预设任何表征基底。
2. **傅里叶基底的自然涌现解释**：证明模加任务的联合概率签名算子构成循环移位交换族，其共同特征向量即离散傅里叶基；在此基础上严格导出主导频率选择、频率匹配与相位对齐的动态方程。
3. **噪声早期损失谜题的定量解析**：通过条件标签碰撞计数与共享坐标强化项，推导出早期损失斜率的闭式表达式，证明噪声数据因碰撞率更高而产生更快的初期下降。
4. **跨运算的基底预测能力**：将同一算子对角化视角迁移至XOR任务，预测权重将沿Hadamard基底演化，并在实验中观测到高稀疏度神经元的涌现与测试精度同步上升。

## 方法详解
- **模型与训练设定**：两层全连接MLP $f(a,b)=W\sigma(U\mathbf{e}_a+V\mathbf{e}_b)$，输入为one-hot编码，$U,V\in\mathbb{R}^{D\times P}$、$W\in\mathbb{R}^{P\times D}$，$D\gg P$。使用softmax交叉熵损失与$L^2$权重衰减$\lambda$，参数高斯初始化方差$D^{-2\gamma}$。采用连续梯度流近似$\frac{d\theta}{dt}=-\eta(\nabla_\theta L+\lambda\theta)$。
- **概率签名定义**：从训练分布$\pi$中采样$(X,Y,Z)$，提取条件统计：一阶$\varphi^{(a)}(x,z)=\mathbb{P}(Z=z|X=x)$、$\varphi^{(b)}(x,z)=\mathbb{P}(Z=z|Y=x)$；三阶联合$\Phi^{(a)}(x,y,z)=\mathbb{P}(Z=z,Y=y|X=x)$等；回溯签名$\psi^{(a)}(x,y)=\mathbb{P}(Y=y|X=x)$等。这些张量/向量仅依赖数据分布，与训练无关。
- **梯度流主导项展开**：对光滑激活函数作$C^3$ Taylor展开$\sigma(h)=c_1 h+\frac{c_2}{2}h^2+o(h^2)$。在小初始化假设$\|\cdot\|_\infty\leq\epsilon$下保留至二阶，得到（Proposition 1）：
  $$\frac{dU^x}{dt}=r_x\eta\left\{c_1W^\top(\varphi_x^{(a)}-\tfrac{1}{P}\mathbf{1})-\tfrac{c_1^2}{P}W^\top\Pi_P W(U^x+V\psi_x^{(a)})+c_2[U^x\odot W^\top\varphi_x^{(a)}-\tfrac{1}{P}(U^x+V\psi_x^{(a)})\odot W^\top\mathbf{1}+\mathrm{diag}(V\Phi_x^{(a)}W)]\right\}+\delta$$
- **傅里叶基底的推导**：在完整干净模加数据上，$\Phi_x^+=P^{-1}S_x$为循环移位矩阵，满足$S_xS_y=S_{x+y}$且彼此交换。求解$S_1F_r=\omega F_r$（$\omega^P=1$）直接得到归一化傅里叶基$F_r(s)=P^{-1/2}e^{-2\pi i rs/P}$。对单隐藏单元做DFT变换后，非零频率$r\neq 0$的演化解耦为独立系统：
  $$\dot{\hat{u}}_r\approx C\hat{w}_r\overline{\hat{v}}_r-\lambda\hat{u}_r,\quad \dot{\hat{v}}_r\approx C\hat{w}_r\overline{\hat{u}}_r-\lambda\hat{v}_r,\quad \dot{\hat{w}}_r\approx C\hat{u}_r\hat{v}_r-\lambda\hat{w}_r$$
  权重衰减提供统一生存阈值，初始随机不对称性决定哪个频率率先跨越阈值成为主导频率；相位满足$\arg U_r+\arg V_r-\arg W_r\to 0\pmod{2\pi}$（Proposition 2）。
- **噪声与有限采样估计**：定理2给出采样比例$\rho$与噪声比例$\alpha$下的梯度误差界，误差方差随$\frac{[(1-\rho)+\alpha]\eta^2}{\rho P^3}(c_1^2\epsilon^2+c_2^2\epsilon^4)$缩放，表明干净样本过少或噪声过高时近似精度显著下降。
- **早期损失下降低定性定理**：定理3推导$\dot{L}_{CE}\propto -\kappa(0)K_0-(\kappa(1)-\kappa(0))K_1$，其中$K_0=\sum_z n_z^2$（同标签碰撞总数），$K_1=P^2(\|\varphi^{(a)}\|_2^2+\|\varphi^{(b)}\|_2^2-2)$。由于$\kappa(1)>\kappa(0)$，噪声数据因条件标签碰撞更强使$\dot{L}$更负，解释早期下降更快的现象。

## 实验与结果
- **数据集与设置**：$\Omega=\{0,\dots,112\}$（$P=113$质数），输入对$P^2$；训练集占80%（含$\alpha=0.1$均匀随机标签噪声），测试集20%；隐藏宽$D=2000$，Adam/AdamW优化器，warmup+cosine decay（$\eta=10^{-3}$，$\eta_0=10^{-5}$），初始化方差$D^{-1}$。
- **权重衰减的作用**：$\lambda=0$时模型在干净/噪声集均达100%但测试为0%（纯记忆）；$\lambda=10^{-4}$实现泛化；$\lambda=5\times10^{-4}$遗忘噪声并保持泛化；$\lambda=10^{-3}$无法收敛。权重衰减被实证为结构选择的生存过滤器。
- **梯度近似精度验证**：截断模型主项与实际CE梯度的余弦相似度$\geq 0.9$；在完整$P^2$样本上使用GELU的50个独立神经元对比中，截断模型预测的主导频率与实际网络收敛频率一致率达93%。
- **频率/相位涌现**：图2A显示多数神经元范数被权重衰减压至0，少数高稀疏度神经元范数同步增长；图2C表明高稀疏度神经元的CE驱动几乎完全来自干净数据，与噪声无关。图3验证$\arg U_\tau+\arg V_\tau\approx\arg W_\tau\pmod{2\pi}$。
- **XOR扩展**：在$\Omega=\{0,\dots,63\}$（$|\Omega|=2^6$）上训练，联合签名算子被Walsh-Hadamard基同时对角化；实验观察到高稀疏度（$\rho>0.7$）且频率匹配的神经元数量随训练单调增加，测试准确率同步上升，与理论预测一致。
- **早期噪声损失对比**：图4显示在几乎所有$\alpha$设置下，噪声样本在$<100$ epoch阶段的损失下降斜率均显著大于干净样本；附录F的模乘实验（$M=59,60,61$）与
