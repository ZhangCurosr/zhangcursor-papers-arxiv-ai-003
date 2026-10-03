---
title: "Scaling-Laws-for-Looped-Mixture-of-Experts"
source: https://arxiv.org/pdf/2609.40316v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-03 08:47:31"
field: "大模型缩放定律与参数高效训练"
keywords: ["scaling laws", "mixture of experts", "looped transformers", "recurrence", "sparsity", "parameter-efficient scaling", "test-time compute"]
innovations: ["首个联合循环递归与MoE稀疏性的统一缩放定律", "有界且稀疏条件的递推有效参数映射", "万亿token实验验证LoopMoE以一半参数追平非循环更大模型"]
benchmarks: ["BBH", "GSM8K", "ARC-C/E", "HellaSwag", "PIQA", "SIQA", "WinoGrande", "BoolQ", "DROP", "MMLU", "Natural Questions", "TriviaQA", "OpenBookQA"]
---

# 论文速读：Scaling-Laws-for-Looped-Mixture-of-Experts

## 一句话总结
本文首次提出统一缩放定律（Loop Scaling Laws），联合建模循环递归深度与 MoE 稀疏性对有效参数增益的影响，并以此为设计依据在万亿 token 训练下使 0.3B 激活/1.3B 总量的 LoopMoE 在推理基准上追平约 2 倍更大的非循环 MoE。

## 研究问题与动机
- 现有缩放定律要么只刻画循环递归（如 Parcae/Iso-Depth），要么只刻画 MoE 稀疏性（如 Unified routed/Joint MoE），二者未被联合建模。
- 循环提升计算深度但复用权重，MoE 提升总容量但保持每 token 激活计算不变，两者被视为互补轴，但相互交互机制不明。
- 已有 Looped MoE 工作多固定循环次数（如 R=2），未能给出随预算选择的依据。
- 实际部署需在训练算力与权重内存双重约束下选择最优 $(R, E)$ 配置。

## 核心贡献（创新点）
- 提出首个联合模型尺寸、数据、循环递归 $R$ 与 MoE 稀疏性的 Loop Scaling Laws，填补两类轴的空白。
- 引入有界、稀疏条件递推映射 $N_{\mathrm{eff}}(R,m)$，以指数饱和刻画每次回路的递减有效增益，并以稀疏度调节渐近上限。
- 证明该定律可还原为 Chinchilla、标准 MoE 与循环缩放定律的特例，具备一致性。
- 在四轴扫参（$N_{\mathrm{act}}, D, E, R$）上拟合，对未见 $R$ 外推的 RMSE 显著低于线性/幂律等无界映射。
- 基于拟合定律给出算力/内存双约束下的 $(R^\star, E^\star, N_{\mathrm{act}}^\star)$ 联合选取策略，并在万亿 token 实验验证实用性。

## 方法详解
- 标准缩放定律：$\mathcal{L}(N, D) = A N^{\alpha} + B D^{\beta} + c$，$\alpha, \beta < 0$，$c$ 为不可约损失，训练算力 $F_{\mathrm{train}} = 6ND$。
- 展开参数：$N_{\mathrm{unroll}}(R) = N_{\mathrm{act}} + (R-1)N_{\mathrm{loop}}$；训练/单 token 推理算力分别为 $6N_{\mathrm{unroll}}(R)D$ 与 $2N_{\mathrm{unroll}}(R)$。
- 三种递推映射比较：线性 $N+(R-1)N_{\mathrm{loop}}$、幂律 $N+(R^{\varphi}-1)N_{\mathrm{loop}}$、有界 $N+\kappa_1 N_{\mathrm{loop}}(1-e^{-(R-1)/\kappa_2})$；前两者 $R\to\infty$ 时无界，后者渐近为 $N+\kappa_1 N_{\mathrm{loop}}$。
- 稀疏条件映射：令 $m=N_{\mathrm{act}}/N_{\mathrm{total}}\in(0,1]$，$\kappa_j(m)=\kappa_j m^{-\theta}$，则 $N_{\mathrm{eff}}(R,m)=N_{\mathrm{act}}+\kappa_1(m)N_{\mathrm{loop}}(1-e^{-(R-1)/\kappa_2(m)})$；$\theta>0$ 时稀疏度越高（$m$ 越小）渐近增益越大且饱和越慢。
- MoE 循环缩放定律：$\mathcal{L}(N_{\mathrm{act}}, D, R, E, m) = A\hat{E}^{\delta} N_{\mathrm{eff}}(R,m)^{\alpha+\gamma\ln\hat{E}} + B\hat{E}^{\omega} D^{\beta+\zeta\ln\hat{E}} + c$，其中 $\hat{E}$ 为单调变换后的有效专家扩张。
- 特例还原：$E=1$ 还原为密集循环定律；$R=1$ 还原为 MoE 定律；$R=1,E=1$ 还原为 Chinchilla 定律。
- 拟合流程：先在 $R=1$ 子集估计基础系数，再用全部四轴数据联合估计 $\{\kappa_1,\kappa_2,\theta\}$；优化采用 L-BFGS-B + log-Huber 目标。

## 实验与结果
- 扫参范围：$N_{\mathrm{act}}\in\{0.3,0.6,1.0\}\mathrm{B}$，$D\in\{100,200,\dots,500\}\mathrm{B}$，$E\in\{1,2,4,8,16\}$，$R\in\{1,2,3,4,6,8\}$；上下文 2048，llama 分词器 202k，AdamW，BF16 权重/FP32 优化器与路由。
- 拟合评估：保留 $R=16、N_{\mathrm{act}}=1.08\mathrm{B}$ 等切片作 held-out；有界稀疏条件映射在所有轴上 RMSE 最低（$R/E/N/D$ 分别为 0.0100/0.0047/0.0050/0.0043）。
- 关键发现：稀疏度提升推高 IsoFLOP 前沿；给定同等激活规模，更多算力时更高 $R$ 更优；内存越紧越倾向高循环、小 $E$，内存宽松越倾向高 $E$、低 $R$。
- 万亿 token 实证：A0.3B-1.3B LoopMoE（$E=8,R=5$）在匹配算力 $1.5\times10^{22}$ FLOPs 下，推理基准 BBH/GSM8K 追平 A0.6B-2.9B MoE；同时支持测试时按需缩放，$R:1\to5$ 使 14 基准 Overall 从 36.6 升至 46.9（+10.3 分）。
- 效率结论：稀疏约带来 3× 激活参数效率，循环约带来 2× 总参数效率（推理任务）。

## 相关工作脉络
- Chinchilla (2022)：仅建模 $N/F$ 轴；本文在其基础上引入 $R$ 与 $S/E$。
- Unified routed MoE (2022) 与 Joint MoE (2025)：建模稀疏轴但固定 $R=1$；本文用 $N_{\mathrm{eff}}(R,m)$ 替代 $N_{\mathrm{act}}$ 扩展其形式。
- Parcae (2604) 与 Iso-Depth (2605)：以线性/幂律无界映射刻画密集循环；本文证明有界指数映射对外推更稳健。
- Sparse Layers (2605) 与 SMELT (2609)：已在 Looped MoE 上实验，但主缩放分析固定 $R=2$；本文给出连续 $R$ 与 $E$ 的联合选择准则。
- MobileMoE (2605) 与 Apple AFM 3 (2026)：资源受限侧 MoE/循环部署；本文的内存-稀疏-循环联动结论可直接用于此类选型。

## 局限性与未来方向
- 仅采用标准中段循环策略（首末各两层不共享），其他循环位置/结构未探索。
- 评估目标以最终回路 Next-Token Prediction 为主，蒸馏/中间层监督虽用于测试时评估但未纳入主缩放拟合。
- 部署内存只计入权重内存，KV cache 与运行时状态未纳入优化。
- 实验规模停留在 0.3–2.4B 激活参数区间，未覆盖云端千亿级前沿模型。
- 外推至 $R>8、E>16$ 时受拟合范围限制，预测不确定性增加。

## 研究启发与可借鉴点
- 将“有效参数”与“展开参数”解耦的思路：用 $N_{\mathrm{eff}}(R)$ 建模权重共享带来的能力增益，而用 $N_{\mathrm{unroll}}(R)$ 建模算力开销，值得迁移到权重复用型架构。
- 有界指数映射替代无界线性/幂律，对饱和型深度扩展更具外推稳定性，可成为后续循环类缩放建模的默认候选。
- 稀疏条件系数 $\kappa_j(m)=\kappa_j m^{-\theta}$ 的参数化简洁且可解释，为“结构变量调制缩放系数”的通用范式。
- 双预算（算力+内存）联合优化的搜索框架可直接复用到端侧 MoE/循环联合设计。
- 测试时按需改变 $R$ 的能力为动态计算分配提供了一条低成本路径。

## 关键术语表
- **Loop Scaling Laws**：联合模型尺寸、训练数据、循环递归与 MoE 稀疏性的统一损失预测定律。
- **Effective parameter count $N_{\mathrm{eff}}$**：权重复用带来的等效能力参数，区别于决定算力的展开参数 $N_{\mathrm{unroll}}$。
- **Bounded recurrence mapping**：以指数饱和刻画循环深度递减收益的映射，渐近上限由 $\kappa_1 N_{\mathrm{loop}}$ 给出。
- **Sparsity-conditional mapping**：将稀疏度 $m$ 作为调制因子，提高循环有效增益渐近线与减缓饱和速度。
- **IsoFLOP frontier**：在固定训练算力下损失随激活参数/稀疏/循环变化的性能前沿曲线。
- **Expert-path diversity $\Psi$**：跨回路由器选择不同专家的平均覆盖数，支撑稀疏条件映射的动机。
- **Compute/memory-optimal $(R^\star, E^\star)$**：在双预算约束下使预测损失最小并满足边际下降阈值的架构选择。
- **Test-time scaling by recurrence**：推理时按需增大 $R$ 以换取更高性能，而不改变存储参数。

## 可复现要素
- 数据集：开放授权、偏 Web 语料混合数学/代码/知识/科学数据；论文未提及独立公开链接。
- 代码/权重：论文未声明开源，未提供权重下载。
- 关键超参：上下文 2048，batch 3072，lr 相关由 AdamW($\beta_1=0.9,\beta_2=0.95,\epsilon=10^{-15}$,wd=0.1,clip=1.0) 控制；MoE 路由用 sigmoid gating、top-k=4、capacity factor=1.5、z-loss $\lambda_z=10^{-4}$、负载均衡 $\lambda_{lb}=10^{-3}$；模型精度 BF16 权重/FP32 优化器与路由。
- 硬件：8 节点 × 64 NVIDIA H100 96GB GPU。
- 拟合工具：`scipy.optimize.curve_fit` 初启 + `scipy.optimize.minimize` L-BFGS-B 精修，log-Huber 目标。
