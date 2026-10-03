---
title: "WHEN-MODELS-DON-T-MANIPULATE-MANIFOLDS-THE-GEOMETRY-OF-A-COM"
source: https://arxiv.org/pdf/2609.37680v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-03 02:57:14"
field: "语言模型机械可解释性"
keywords: ["mechanistic interpretability", "manifold hypothesis", "linear representation", "causal geometry", "number comparison", "attention copy head", "MLP local-global circuit"]
innovations: ["证明数字比较任务中模型使用线性因果方向而非流形维度进行计算", "提出基于加法混合与局部-全局 MLP 分段的比较算法并在因果干预下验证", "将线性比较算法扩展至三数序列并刻画 flag 存储与末 token 汇总机制"]
benchmarks: ["two-digit max comparison (2000 pairs)", "three-digit max comparison (1500 triples)", "position recovery and IIA evaluation on Qwen2.5-7B-Instruct"]
---

# 论文速读：WHEN-MODELS-DON-T-MANIPULATE-MANIFOLDS-THE-GEOMETRY-OF-A-COMPARISON-TASK

## 一句话总结
本文在 Qwen2.5-7B-Instruct 模型上揭示了数字比较任务的因果几何机制：尽管模型激活中存在非线性低维流形（如螺旋），模型在执行比较计算时仍优先使用各数字方向的**线性表征**；通过注意力“拷贝”与残差连接将两个数字的线性向量相加形成共享二维平面，再经 MLP 神经元在局部区间进行分段比较并组合为全局 argmax 答案，该算法可自然推广至三个及以上数字的序列比较。

## 研究问题与动机
- **核心问题**：已知有序概念（如数字）在模型内部呈现低维弯曲流形结构（manifold hypothesis），但具体到某项计算任务时，模型究竟利用流形的哪些几何方面来执行计算尚不清楚。
- **现有工作不足**：先前 GPT-2 比较电路研究（Hanna et al., 2023）虽定位了电路但未因果性证明表征几何的作用；El-Shangiti et al. (2025) 发现了因果影响输出的线性子空间，但未刻画两数比较的具体机制；Yuchi et al. (2026) 仅比较行为准确率与分类器性能，未涉及几何操纵细节。
- **动机**：在机械可解释性领域建立从“流形表征几何”到“算法级计算操作”的因果对应，回答“模型是否/如何在特定任务中实际操控非线性流形”这一开放问题。
- **动机延伸**：比较是决策与数值推理的基础原语；若模型在存在流形结构时仍偏好线性操作，则将对表征几何假设与计算实用性之间的关系提供新的理论 nuance。

## 核心贡献（创新点）
- **线性表征与流形假设的因果调和**：证明即使激活中存在显著的非线性流形结构（PC1/PC3 展示螺旋/弯曲），仍存在单个因果方向（如 u）在层 13 残差流中通过 inter-change intervention（IIA）显著改变模型比较输出，揭示线性分量足以驱动任务表现。
- **基于加法混合的两步局部-全局比较算法**：提出模型先将两数字沿各自方向 ${\boldsymbol{v}}_1, {\boldsymbol{v}}_2$ 的线性表征在残差流中相加形成共享二维平面，再由 MLP 神经元在局部区域执行分段比较、最终组合为全局 argmax 决策，给出完整可验证的计算流程（Alg. 1）。
- **扩展至多数字序列的比较算法**：将上述线性加法 + 局部比较架构推广至三个数字情形（Alg. 2），显示模型在每个时刻 $t$ 维护一个二元 flag $\alpha_t = \mathbb{I}(y_t = \max(y_1,\dots,y_t))$ 并存储在单方向 $\mathbf{d}_t$，最终由末 token 上的单个注意力头汇总获得 argmax 位置，表明线性表征利用具有序列可扩展性。
- **因果层面的算法化解释而非仅描述性几何刻画**：相比以往主要依赖 PCA 可视化与相关性探针的工作，本文通过 interchange intervention、position recovery、attribution freezing、dose-response 等因果度量，逐层验证从方向学习到神经元贡献再到最终输出的因果链，使几何解释具备预测干预行为的精度。
- **揭示“可用 vs. 实用”表征维度的选择性**：明确区分“概念表征空间中丰富的流形维度”与“任务计算实际调用的线性维度”，呼应 Garg et al. (2026) 关于特征可用性与实用性的区分，为后续研究区分表征丰富性与计算简约性提供实证依据。

## 方法详解
- **任务设定与数据**：使用 Qwen2.5-7B-Instruct（28 层，$d_{\text{model}}=3584$，28 个 128 维 head，SwiGLU MLP 宽度 18944）在 two-digit（[10,100)，间隔 $g=10$）与 three-digit（[100,1000)，间隔 $g=100$）两个区间上生成配对/三元组；prompt 以 one-shot 示例固定输出格式，要求模型输出最大值的首 token（leading digit）。
- **因果方向检索**：对每个数字位置，先采用 PCA 寻找主导主成分；若主成分 IIA 较低，则改用 Distributed Alignment Search（DAS）在 fitting 集上训练 rank-1 方向以最大化对扰动后 leading token 的分类交叉熵。得到 $y_1$ 在 L13 的因果方向 ${\bf u}$、${\bf v}_1$（来自 L14.H14 输出）、${\bf v}_2$（L13 PC1）等。
- **加法共享表示的形成**：Layer 14 的 attention head H14 作为 copy head 在 $y_2$ 位置读取 $y_1$ 的表征并将其变换为 ${\bf v}_1 f_1(y_1)$，与原本编码 $y_2$ 的方向 ${\bf v}_2 f_2(y_2)$ 在 pre-MLP 残差流中叠加，形成二维共享平面 $\text{span}({\bf v}_1,{\bf v}_2)$；该叠加可由两个 rank-1 patch 近似等效于 rank-2 平面 patch（IIA 几乎一致），说明模型采用线性相加策略。
- **局部-全局比较的两阶段 MLP 机制**：在层 14 与层 15 的 MLP 中通过 first-order attribution 排序候选神经元并冻结实验定位 12 个关键神经元（两层各 6 个）。层 14 神经元在 $(y_1,y_2)$ 空间的局部区域激活，对应 SwiGLU 门控非线性导致的近似二次响应；层 15 神经元将这些局部比较聚合为全局比较器，最终在 L15 残差流中写出单一比较方向（编码 $\arg\max(y_1,y_2)$ 的位置信号）。
- **三数比较的扩展算法**：在 $y_3$ 位置通过 heads L14.H14、L14.H18 分别拷贝 $y_1$、$y_2$ 表征，叠加到 $y_3$ 的方向 ${\bf v}_3 f_3(y_3)$ 上形成三维 span；MLP 神经元在 $(y_1,y_2,y_3)$ 的局部体积内执行分段比较并输出 flag $\alpha_3$；与此同时在 $y_2$ 位置已保存 $\alpha_2$；最终 token 上 L20 某 attention head 对 $\alpha_2{\bf d}_2$、$\alpha_3{\bf d}_3$ 加权求和得到 ${\bf x}_{\ell'}(\tilde{T})$，其在 top-2 PC 上即可线性分离 argmax 位置。
- **因果评测指标**：
  - **Interchange Intervention Accuracy (IIA)**：将 clean 激活沿目标方向/子空间 patch 到 corrupt 运行中，统计首个生成 token 等于期望 leading digit 的比例。
  - **Position Recovery (PR)**：以 logit 差 $\text{PLD}=\log p(t(r))-\log p(t(b))$ 为基础，计算 patch 后 PLD 相对 clean-corr 差值的归一化偏移，用于评估“答案位置”被干预的程度。
  - **Freezing / Attribution**：按 attribution 分数冻结 top-k 神经元并观测 IIA 衰减；计算上游 MLP 神经元对下游 neuron 的因果边权与虚拟权重（gate projection cosine）。

## 实验与结果
- **行为准确率**：在全部 8010 个不同两位数字对上 greedy 解码，模型达到接近 100% 准确；对三数字序列同样表现优异（附录图 6 显示随 operand 数量增加准确率保持在高位）。
- **两人比较的关键因果数字**：
  - 方向 ${\bf u}$（L13, $y_1$ 位置，$y_1$ 扰动）IIA=0.940，full L13 patch 仅 0.072，说明单一方向高度特异。
  - 共享平面 $\text{span}({\bf v}_1,{\bf v}_2)$ 在 $y_1$ 扰动下 IIA=0.810，$y_2$ 扰动下 IIA=0.948；两个 rank-1 patch 叠加（${\bf v}_1\&{\bf v}_2$）接近平面效果（0.675/0.885），印证加法结构。
  - L15 DAS 比较方向在 $y_1$ 扰动下达到 IIA=1.000，证实最终答案位置信号的高度因果效力。
  - 冻结 12 个关键 MLP 神经元使 IIA 显著下降（图 4e、图 21b/c），随机 12 神经元无此效应。
- **三数比较的因果证据**：
  - ${\bf v}_1,{\bf v}_2,{\bf v}_3$ 三个方向的 PR 在不同扰动 case 下分别显著，且 span  patch 能同时影响多个数字的因果信号（图 5b、图 29）。
  - 层 15 残差中在 $y_2$ 与 $y_3$ 位置均存在编码 flag $\alpha_t$ 的单方向（图 5d）。
  - 末 token 处 L20 残差的 top-2 PC 即可承载完整答案位置信息（图 5e），且这两个 PC 的 PR 与 full residual patch 相当（图 34l）。
- **最强结果**：在 pairwise 设置中，L15 比较方向的 DAS patch 实现 IIA=1.000（first-token），且整个 $(v_1,v_2)$ 平面的 patch 在 $y_2$ 扰动条件下达到 IIA=0.948；三数场景末 token 的 top-2 PC 同样以 minimal subspace 捕获全局 argmax 信息，体现算法的高效性与低秩结构化。

## 相关工作脉络
- **Hanna et al. (2023) GPT-2 比较电路**：本文与其定位差异在于，Hanna 等侧重 circuit 定位与行为拟合，明确承认难以因果证明表征几何的作用；本文则通过 DAS/IIA/PR 建立从流形、线性方向到局部-全局神经元链条的因果解释。
- **Kantamneni & Tegmark (2025) 数字螺旋流形**：该文强调数字在低维流形（helix）上的编码；本文承认该非线性的存在（图 2a 展示 PC1/PC3 螺旋），但证明比较任务中因果有效的子空间是线性方向，提供“流形存在但不被用于该类计算”的反例。
- **Gurnee et al. (2026) When models manipulate manifolds（计数任务）**：该文发现模型确实在 line-breaking 等任务上操控字符计数流形；本文与之形成对照，说明是否操控流形取决于任务结构——比较任务下模型偏好更简单的线性叠加。
- **El-Shangiti et al. (2025) 线性子空间的因果影响**：该文发现影响输出的线性子空间但未分析比较的内部机制；本文细化到具体方向、attention head 与 MLP 神经元的操作顺序。
- **Feucht et al. (2026) Fourier 加法与周期概念**：该文指出星期、月份等周期概念使用 Fourier 特征进行加法；本文进一步支持“任务对称性决定表征利用方式”的视角——有序无周期概念可依赖线性分量。
- **Geiger et al. (2021, 2024) Causal abstractions / DAS**：本文方法论基础，使用 interchange intervention 与分布式对齐搜索定位因果方向，并将其与神经元级冻结、attribution 结合实现算法级重建。

## 局限性与未来方向
- **任务与概念的限定性**：分析仅限数字这一有序概念及 max 计算；对周期概念（如星期）或需流形几何的任务（如加法）是否同样使用线性分量未验证。
- **因果必要性的缺失**：大部分实验证明的是充分性（patching 能改变输出），但未排除模型存在冗余通路（hydra effect, McGrath et al., 2023）的可能性；去除关键方向/神经元后模型的补偿机制未被系统测量。
- **DAS 方法的潜在短板**：DAS 可能找到 shortcut 方向（Wu et al., 2024），尽管作者通过 attention head 定位、neuron 冻结等多重证据缓解该担忧，但仍需更严格的必要性检验。
- **单模型单规模**：仅在一个 7B open-weight 模型上开展，结论是否跨架构（如 LLaMA、PaLM 系列）与跨尺度（1B/70B）成立未知。
- **未来方向**：
  - 将算法扩展到更长序列（$K>3$）并刻画 attention 汇总机制的精确实现。
  - 在周期/非线性任务（加法、日期运算）中对比模型是否真正“操控流形”。
  - 设计 necessity 干预（如因果消融、structural causal ablations）以区分必要与充分组件。
  - 探索线性表征在训练过程中的涌现路径：是优化压力选择还是初始化偏置所致。

## 研究启发与可借鉴点
- **线性-流形双重检验范式**：在同一任务上同时报告 PCA 流形可视化与 DAS 因果方向，并对比二者在 IIA/PR 上的表现，可作为后续研究的标准核查流程，避免将可视化相关性误读为计算因果性。
- **加法混合假设的定量验证**：通过分别 patch rank-1 方向与 rank-2 span 并比较 IIA 差异，可量化模型是否采用线性叠加策略；该设计可直接迁移至其他多输入组合任务（如排序、求和）。
- **局部-全局两段式 MLP 定位**：先用 attribution 筛选候选神经元、再用 freeze-sweep 验证贡献，最后绘制 receptive field 与因果边权，形成可复现的“神经元回路重建”管线，适用于各类算术/逻辑 circuit 的解析。
- **Position Recovery 作为答案位置敏感指标**：相比传统 IIA 对首 token 的敏感性，PR 能更细腻地刻画“模型内部答案位置的偏移”，适合研究 argmax/select 类任务的中间表征。
- **可迁移的创新机会**：将本框架应用于本团队关注的有序概念表征（如时间、程度、评分）任务，检验是否普遍存在“流形丰富但计算线性”的现象；或反向在周期概念任务上验证流形操控假说，形成对照研究。

## 关键术语表
- **Manifold hypothesis（流形假设）**：概念表征在模型高维激活空间中落在低维弯曲子流形上的假设，常见例子包括数字螺旋、日期圆周等。
- **Linear representation hypothesis（线性表征假设）**：主张有序概念被编码为单一方向上的单调函数（常为对数线性），可通过线性探针解码。
- **Interchange Intervention Accuracy (IIA)**：将 clean 运行的激活沿指定方向/子空间 patch 到 corrupt 运行中，统计首 token 正确率的变化，衡量该分量对输出的因果影响。
- **Position Recovery (PR)**：以 logit 差为基数的归一化干预效应指标，反映 patch 后模型对“答案位置/较大 operand 位置”信号的恢复程度。
- **Distributed Alignment Search (DAS)**：通过优化方向矩阵使模型激活与目标因果变量对齐的搜索方法，用于在低因果性主成分之外发现更具解释力的单方向。
- **Argmax position signal**：模型内部表示最大值所在 operand 位置而非其具体数值的编码方式，本文通过 patch 改变该位置信号而非数值本身加以验证。
- **Local-global comparison circuit**：MLP 先在各数字区间的局部区域执行分段比较，再由后续层神经元将局部结果聚合为全局比较判决的两阶段机制。
- **Hydra effect（海德拉效应）**：模型在部分组件被消融后仍能通过备用计算路径维持任务表现的现象，提示单一因果路径的必要性需谨慎判定。

## 可复现要素
- **数据集**：两位数字有序对 2000 个（含 1000 对及镜像），三位数字三元组 1500 个；eval 集 400 对、fitting 集 128 对，三数 quadruples 1400 个（过滤后 1341 个），eval 用前 528 个。数据集为作者采样器生成，**论文未提供公开下载链接**。
- **代码/权重**：使用 Qwen2.5-7B-Instruct（官方开源权重可下载）；因果干预与 DAS 代码依赖内部库，**论文附录提供了实验设置、prompt、超参与采样流程**，但未提供完整开源代码仓库声明（仅在 AI use statement 中提到使用生成式 AI 辅助编写与编辑代码）。
- **关键超参**：Adam 100 步、学习率 0.05、mini-batch 16-32、half-precision loss scale 100；ridge 回归惩罚在 $[10^{-2},10^{4}]$ 取 13 个对数间距值；receptive field 网格步长 4、覆盖 $y\in[11,99]$；dose-response 窗口 64 prompts per cell。
