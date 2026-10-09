---
title: "Scaling-to-Tens-of-Thousands-of-Test-Time-Iterations-with-Lo"
source: https://arxiv.org/pdf/2610.11570v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-09 17:10:59"
field: "循环神经网络与推理架构"
keywords: ["循环Transformer", "残差连接", "测试时计算", "递归推理", "注意力残差", "推理稳定性"]
innovations: ["提出InfiLoop循环原生残差连接，用内容评分与指数衰减替代carry-last规则", "推导精确流式递推形式实现O(d)常量内存聚合", "揭示模型涌现自验证行为：搜索压低步长、解出时跃升、冻结已解状态"]
benchmarks: ["Sudoku-Extreme", "Maze-Hard", "ARC-AGI-1", "ARC-AGI-2"]
---

# 论文速读：Scaling-to-Tens-of-Thousands-of-Test-Time-Iterations-with-Lo

## 一句话总结
论文指出循环Transformer采用carry-last规则在大量测试时迭代后会覆盖正确中间推理、导致性能下降；为此提出InfiLoop——一种循环原生残差连接，通过内容加权评分与指数时间衰减自适应聚合历史候选状态，在7M参数量下于Sudoku-Extreme取得97.9%精确准确率，且在超过20,000层深度时仍能持续提升。

## 研究问题与动机
- **核心问题**：循环Transformer（如TRM）通过共享同一块反复迭代以扩展推理深度，但在迭代次数增加后，推理精度反而下降。
- **carry-last规则的缺陷**：标准做法将上一轮输出直接作为下一轮输入（$h_t = z_t$），新状态完全覆盖旧状态；当某一轮迭代产生噪声或错误更新时，会破坏之前已获得的正确中间推理，且后续迭代难以从已退化的表示中恢复。
- **现有方法的局限**：FPRM等工作通过层缩放、阻尼固定点更新等方式控制残差增长幅度，但仍未改变"下一轮只接收最新状态"这一架构选择，无法显式保留并选择历史输出中的有用信息。
- **实证观察**：在Sudoku-Extreme上，TRM在468层（26层/轮×18轮）达到峰值精度62.2%，但增至1872层后降至58.1%；同时，TRM在首次求解后每轮仍平均改变77个预测、错误单元格数增加0.21，而InfiLoop仅改变24个、减少1.25个错误单元格。

## 核心贡献（创新点）
1. **指出carry-last是深层循环Transformer的主要瓶颈**：通过诊断实验证明新输出不一定优于旧输出，并揭示噪声迭代会导致已求解状态被破坏。
2. **提出InfiLoop循环原生残差连接**：用可学习的注意力聚合替代carry-last，通过伪查询评分与指数时间衰减自适应组合历史候选状态。
3. **推导精确流式递推形式**：证明共享评分器与指数衰减允许仅维护向量$N_t$和标量$D_t$的递推，持久聚合内存独立于迭代次数（$O(d)$而非$O(Td)$）。
4. **揭示自适应残差更新的涌现自验证行为**：模型在搜索阶段压低步长（约0.40），在找到解时步长跃升至0.91，解出后保持冻结；内容评分在无额外监督下自动与解答正确性对齐。
5. **在极小参数下实现最强推理性能**：7M参数的InfiLoop在Sudoku-Extreme达97.9%、ARC-AGI-2达13.6% pass@2，均超越已有同等规模模型。

## 方法详解
- **基本设定**：共享块$f_\theta$在$T$个递归步上重复应用，第$t$步产生候选状态$z_t = f_\theta(h_{t-1}; x)$，初始状态$h_0$给定。
- **注意力残差聚合**（公式2）：
  $$h_t = \sum_{i=0}^{t} \alpha_{t,i} z_i, \quad \alpha_{t,i} = \frac{\exp(s_i - \lambda(t-i))}{\sum_{j=0}^{t}\exp(s_j - \lambda(t-j))}$$
  其中内容分数$s_i = q^\top \text{RMSNorm}(z_i)$，$q$为跨迭代共享的伪查询；时间衰减项$-\lambda(t-i)$由学习得到的$\lambda > 0$控制。
- **精确流式递推**（公式3-4）：令$e_i = \exp(s_i)$、$\beta = e^{-\lambda}$，则维护：
  $$N_t = \beta N_{t-1} + e_t z_t, \quad D_t = \beta D_{t-1} + e_t, \quad h_t = N_t / D_t$$
  每个token仅需保存$d$维向量$N_t$和标量$D_t$，内存与迭代次数无关。
- **自适应残差更新等价形式**（公式5）：
  $$h_t = h_{t-1} + \gamma_t(z_t - h_{t-1}), \quad \gamma_t = \frac{e_t}{\beta D_{t-1} + e_t} \in (0,1)$$
  $\gamma_t$比较新候选的内容权重与保留历史质量：$e_t \gg \beta D_{t-1}$时$\gamma_t \to 1$（新候选主导），$e_t \ll \beta D_{t-1}$时$\gamma_t \to 0$（保留旧状态）。
- **嵌套循环扩展**：对标准循环Transformer替换carry-last为公式4；对TRM/HRM等嵌套结构，在每条递归轴上独立应用相同的 scorer-decay 对$(q,\beta)$及累加器$(N,D)$，每轴仅增加$d+1$个可学习参数。

## 实验与结果
- **数据集**：Sudoku-Extreme（1000基础谜题×1000增强≈1M训练，422,786测试）、Maze-Hard（各1000训练/测试）、ARC-AGI-1（400+400）、ARC-AGI-2（1000+120）。
- **模型规模**：7M参数，共享递归块含2层Transformer、隐藏维度512、8头注意力、SwiGLU（扩展因子4）、深度可分离3×3卷积。
- **主要结果（Table 1）**：
  - Sudoku-Extreme：InfiLoop **97.9%**（FPRM 94.2%，TRM 74.7%，Attractor Model 91.4%）
  - Maze-Hard：InfiLoop **87.4%**（FPRM 87.0%，TRM 85.3%）
  - ARC-AGI-1：InfiLoop **49.1%** pass@2（FPRM 47.5%，TRM 44.6%）
  - ARC-AGI-2：InfiLoop **13.6%** pass@2（TRM 7.8%，FPRM 6.2%）
- **测试时深度缩放（Figure 3）**：
  - InfiLoop在1872层达90.9%，超过FPRM在3744层的90.1%（计算量减半）。
  - 在24,960层时，InfiLoop 92.74% vs FPRM 91.63% vs TRM 83.65%。
  - InfiLoop在936层的86.5%甚至超过TRM在24,960层的83.65%。
- **难度分解**：52个空格的题目InfiLoop超FPRM +3.5pp、超TRM +4.2pp；60个空格时差距扩大至+3.0pp（FPRM）和+23.4pp（TRM）。
- **扰动稳定性（Figure 5）**：注入$\epsilon \in [10^{-4}, 0.3]$后36轮，InfiLoop保留100%已解谜题；TRM即使在无扰动时也丢失15-20%，且微小扰动导致轨迹在12轮内发散至O(1)相对距离。

## 相关工作脉络
- **循环Transformer基础**：Dehghani et al. (2019) Universal Transformers提出共享块迭代；Giannou et al. (2023)、Saunshi et al. (2025)研究其可计算性与潜在推理能力。
- **TRM与HRM**：Jolicoeur-Martineau (2025) TRM、Wang et al. (2025) HRM均基于嵌套递归结构设计紧凑推理器，使用carry-last连接。
- **固定点稳定化方法**：FPRM (Movahedi et al. 2026) 结合pre-norm、学习残差缩放和阻尼固定点更新；Attractor Models (Fein-Ashley & Rashidinejad 2026) 和EqR (Huang et al. 2026) 基于吸引子/平衡点 formulation。
- **深度Transformer稳定化**：DeepNet (Wang et al. 2024)、ReZero (Bachlechner et al. 2021)、LayerScale (Touvron et al. 2021) 等通过残差缩放控制深层网络激活。
- **Attention Residuals**：Kimi Team et al. (2026) 提出跨层注意力残差，但每层需独立伪查询，内存随深度增长；InfiLoop将其适配至循环场景，保证内存恒定。
- **自适应计算时间**：Graves (2016) ACT、Banino et al. (2021) PonderNet、Bae et al. (2025) MoR学习动态计算预算，与InfiLoop关注点不同。

## 局限性与未来方向
- **任务范围**：实验集中在符号推理（Sudoku、Maze）和抽象视觉转换（ARC），未在语言推理、数学证明等任务上验证。
- **参数规模较小**：最大7M参数，模型容量有限，扩展至更大规模时InfiLoop的优势是否保持需进一步验证。
- **超参数依赖**：衰减初始化$\beta_0 = 0.55$表现最佳，极端值（0.10/0.90）显著劣化，说明初始化敏感。
- **嵌套结构适用性**：论文提到可推广至嵌套递归架构，但未系统测试不同嵌套深度与层数的泛化能力。
- **未来方向**：可在更大参数规模、更多样化任务（数学推理、代码生成、多模态）上验证；探索学习衰减率$\lambda$的动态调度策略；将InfiLoop集成至主流循环推理框架中评估工程效率。

## 研究启发与可借鉴点
1. **诊断性实验设计**：通过测量"首次求解后的状态变化幅度"和"已解谜题保留率"来量化carry-last的退化问题，这种分析范式可用于诊断其他递归架构的稳定性。
2. **精确递推替代显式历史存储**：将注意力聚合转化为常数内存的递推形式（维护$N_t$和$D_t$），既保留全局信息又避免$O(T)$内存开销，可迁移至长序列处理、RNN/循环架构设计。
3. **涌现行为的无监督分析**：通过分析步长$\gamma_t$和内容评分$s_i$与任务完成度的关系，揭示模型自发产生的"搜索压制→解发现→状态冻结"行为，为可解释性研究提供范例。
4. **内容评分与任务正确性的隐式对齐**：评分器无显式正确性标签却自动与解质量对齐，提示在学习型循环架构中，简单评分机制可有效充当隐式验证器。
5. **双轴独立聚合的模块化设计**：对内外层分别配置独立$(q,\beta)$对，保持结构简洁的同时提升性能，为嵌套递归系统的接口设计提供参考。

## 关键术语表
- **InfiLoop**：论文提出的循环原生残差连接，通过可学习的内容评分与指数时间衰减自适应聚合历史候选状态。
- **carry-last规则**：标准循环Transformer的连接方式，将上一轮输出直接作为下一轮输入（$h_t = z_t$），无任何历史信息保留。
- **有效深度（effective depth）**：已执行的Transformer层总数，等于监督段数$S$×外循环数$H$×(内迭代数$L$+1)×每块层数$K$。
- **自适应步长$\gamma_t$**：公式5中的更新系数，衡量新候选相对历史累积状态的权重，取值$(0,1)$，由内容评分与衰减质量共同决定。
- **固定点推理（Fixed-point reasoning）**：模型在迭代中趋近稳定状态的推理模式，如FPRM、Attractor Models所采用。
- **ARC-AGI基准**：Chollet提出的抽象推理与归纳_corpus_，要求模型从少量演示中学习变换规则并泛化，分为ARC-AGI-1和ARC-AGI-2两个版本。
- **Attention Residuals**：Kimi Team et al. (2026) 提出的跨层注意力残差机制，每层通过独立伪查询对历史输出加权求和。
- **pass@k**：在ARC等生成任务中，从$k$个候选输出中至少一个正确的比例，本文使用pass@2。

## 可复现要素
- **数据集**：Sudoku-Extreme、Maze-Hard、ARC-AGI-1、ARC-AGI-2均为公开数据集。
- **代码**：已开源，链接为https://github.com/pixeli99/InfiLoop。
- **权重**：论文声明代码可用，但未明确说明模型权重是否开源（需查看GitHub仓库）。
- **关键超参**：
  - 训练Sudoku-Extreme：AdamW，峰值学习率$10^{-4}$，weight decay 1.0，batch size 768，H=3, L=6, S=12段深监督。
  - 训练Maze/ARC：峰值学习率$3\times10^{-4}$，H=3, L=6，最多16监督段。
  - 衰减初始化：$\beta_0 = 0.55$（即$\lambda = -\ln(0.55) \approx 0.598$）。
  - 每轴 scorer-decay 参数：513个（$q \in \mathbb{R}^{512} + \beta \in \mathbb{R}^1$）。
- **模型架构**：2层Transformer共享块，隐藏维度512，8头注意力，SwiGLU扩展因子4，深度可分离3×3卷积。
