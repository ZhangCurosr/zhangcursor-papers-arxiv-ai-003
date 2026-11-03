---
title: "Two-Clocks-Why-Larger-Rollout-Batches-Learn-Slower-but-Fail"
source: https://arxiv.org/pdf/2610.18853v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-03 01:34:47"
field: "大语言模型强化学习训练动力学"
keywords: ["reinforcement learning with verifiable rewards", "GRPO", "batch scaling", "square-root learning rate rule", "training stability", "critical batch size", "ratio clipping", "policy optimization"]
innovations: ["提出双时钟框架统一解释多步GRPO中进度延迟与样本不变崩溃现象", "证明平方根规则保持每样本漂移不变但曲率导致进度下降", "导出基于漂移阈值的失败律并经21次实验验证，优于随机游走假设"]
benchmarks: ["DAPO-Math-17k", "DeepScaleR-Preview", "Qwen2.5-Math-1.5B", "Qwen2.5-Math-7B", "Qwen2.5-1.5B-Instruct"]
---

# 论文速读：Two-Clocks-Why-Larger-Rollout-Batches-Learn-Slower-but-Fail

## 一句话总结
本文提出"双时钟"理论框架，解释多步 GRPO 训练中平方根学习率缩放规则下的大批量延迟现象：平方根规则保持每个样本的漂移不变，但因曲率损失导致进度下降；由此衍生出两个时钟——进度时钟在临界批量以上变慢，失败时钟在平方根规则下保持样本数不变。实验在 Qwen2.5-Math-1.5B 上验证了这两条规律，并据此提出"从小批量起步、在失败前切换到大批量"的调度策略。

## 研究问题与动机
- **多步 GRPO 的平方根规则失效**：单步 GRPO 下，平方根规则可使学习曲线在批量 512–8K 范围内近似重合，但实际系统（如 DAPO）每 rollout 使用 16 步优化器更新，从第二步起 importance ratio 偏离 1，ratio clip 开始截断梯度。
- **三种伴随现象缺乏统一解释**：① 批量越大每样本进度越慢；② 无 clip 时大批量与小批量在同一样本数处崩溃（尽管步骤数相差数十倍）；③ clip 对大批量稳定性增益最大。已有工作分别研究这些现象，但未给出统一框架。
- **平方根规则的实际承诺未被理解**：实践中广泛采用 $\eta \propto \sqrt{B_r}$ 规则，但其是否保持"样本效率"、以及崩溃机制如何随批量缩放，缺乏理论刻画。
- **批量调度缺乏理论指导**：实际训练中如何选择批量及何时切换批量，目前依赖经验，没有来自训练动力学的可计算指导。

## 核心贡献（创新点）
1. **提出双时钟（two-clock）批量缩放框架**：基于局部二次损失模型推导进度时钟（curvature-driven），叠加漂移阈值假设推导失败时钟，统一解释多步 GRPO 中进度延迟与批量无关崩溃两种现象。
2. **揭示平方根规则的实质**：证明平方根规则保持每样本漂移（drift per sample）不变，但不保持每样本进度——大批量以更少、更长的步长累积相同漂移，却因曲率损失更多进度，导出与学习率相关的临界批量 $B^*$。
3. **首次系统性验证失败时钟（failure law）**：通过 21 次无 clip 实验拟合失败律，证明崩溃点服从漂移缩放 $N_{\mathrm{fail}} \propto \sqrt{b}/\eta$，显著优于随机游走假设（$b/\eta^2$），似然比检验 $p \approx 0.3$ 对漂移律成立，对替代假设 $p < 10^{-4}$。
4. **从双时钟导出实用批量调度策略**：提出"临界批量以下起步 → 预测失败前切换至大批量"的 ramp 调度，实验中以大批量生成 75% 样本的同时逼近最佳固定批量性能。
5. **量化 ratio clip 的稳态贡献**：证明 clip 移除的 token 占比虽小（约 0.5%），却携带大批量梯度范数的 50–70%，clip 本质是购买稳定性而非速度，且对大批量的保护效应最强。

## 方法详解
- **平方根规则设定**：rollout 批量 $B_r = PG$（$P$ 个 prompt × $G$ 个 response），拆分为 $\mu$ 个 mini-batch，每步 optimizer batch $b = B_r / \mu$。平方根规则设 $\eta = \kappa\sqrt{b}$，Adam 其他超参固定。
- **每样本漂移（drift per sample）**：在噪声主导归一化假设（A1）和局部二次损失假设（A2）下，一步均值更新 $\mathbb{E}[\Delta\theta] = -\frac{\kappa b}{\sigma}\bar{g}$，每样本漂移 $d = \frac{\kappa\|\bar{g}\|}{\sigma}$ 与批量无关。
- **进度时钟（Proposition 1）**：每样本期望损失下降 $r(b) = r_0 - C_{\mathrm{curv}} \cdot b$，其中 $C_{\mathrm{curv}} = \frac{\kappa^2 \bar{g}^\top H \bar{g}}{2\sigma^2}$ 为曲率项，随批量线性衰减。到达目标精度所需样本 $N_\tau(B_r) = N_\tau^{\min}(1 + B_r/B^*)$，临界批量 $B^* = \mu\left(\frac{2\sigma\|\bar{g}\|^2}{\kappa\bar{g}^\top H\bar{g}} - B_{\mathrm{noise}}\right)$，随学习率增大而缩小（与 McCandlish et al. 2018 的固定学习率临界批量本质不同）。
- **失败时钟（Proposition 2，假设 A3 漂移阈值）**：假设积累漂移 $\|\sum_j \mathbb{E}\Delta\theta_j\|$ 达到阈值 $D$ 时崩溃，推得 $N_{\mathrm{fail}} = C\frac{\sqrt{b}}{\eta}$；在平方根规则下 $N_{\mathrm{fail}} = C/\kappa$，与批量无关。
- **Ratio clip 的作用机制**：每个 rollout 内 log-ratio  spread 随步数约 $\sqrt{k}\eta\sigma_s$ 增长，大批量每步 $\eta$ 更大，spread 更快触发 clip。Clip 限制每个 rollout 的漂移贡献而非撤销已积累漂移，不产生进度间隙的曲率项。
- **Clip 移除梯度份额（Proposition 5）**：通过余弦定理，被 clip 掉的 token 梯度范数占比可达 50–70%（大批量），证明 clip 截断的是主导性梯度分量。

## 实验与结果
- **数据集与模型**：DAPO-Math-17k，Qwen2.5-Math-1.5B；鲁棒性验证涉及 DeepScaleR-Preview、Qwen2.5-1.5B-Instruct、Qwen2.5-Math-7B。
- **基线与设置**：rollout 批量 512/1536/4096/8192（16×范围），$\mu=8$（主实验），学习率 $\eta_S \in \{10^{-6}, 5\times10^{-6}, 10^{-5}\}$，各比例按平方根规则缩放；禁用 KL 惩罚、entropy bonus、动态采样，单独测试 gradient-norm clip。
- **进度时钟（P1）**：$\mu=1$ 时（clip 不生效），大批量仍比小批量多需 20–45% 样本；$\mu=8$ 保守 regime 下样本成本指数 $\alpha=0.15$，活跃 regime 升至 0.44；临界批量 $B^*$ 随学习率从 ~9k 降至 ~2k，远低于梯度噪声尺度。在 7B 模型和通用 checkpoint 上同样复现。
- **失败时钟（P2）**：无 clip 时，4096 批量在 56–80 步后崩溃，512 批量在 648 步后崩溃，但均在 30–50k 样本处崩溃；漂移律拟合 $\beta-\gamma/2=-0.08$（含删失样本为 0.11），90% CI 包含零，排除随机游走假设（$\gamma=2$）。
- **Clip 的作用（P3）**：有 clip 时小批量崩溃延迟 2 倍，大批量延迟 4 倍；放宽 clip 半径获得早期加速但后续衰减且 seed 1 崩溃，验证"clip 买稳定性而非持续速度"。
- **批量调度**：ramp 策略（512 起步，25k–74k 样本处切换至 4096/8192）在所有 switch point 下表现相似，接近最佳固定批量，且 75% 样本在大批量下生成。

## 相关工作脉络
- **Batch-size invariance (Hilton et al. 2022; Li et al. 2026)**：证明 PPO/GRPO 在单步优化器下批量不变性；本文将其扩展至多步场景，揭示不变性仅在单步成立、多步下批量依赖性浮现。
- **Critical batch (McCandlish et al. 2018)**：学习率在每个批量下单独调优时临界批量等于梯度噪声尺度，与学习率无关；本文平方根规则下临界批量 $B^* \propto 1/\kappa$，随学习率增大而缩小，远低于噪声尺度。
- **Training collapse in RLVR (Suk & Duan 2025; Song et al. 2026)**：前者推导步长阈值，后者发现异步 GRPO 崩溃时间与 $1/\eta$ 成正比；本文加入批量维度，沿平方根规则崩溃点近似样本不变。
- **Ratio clipping as bottleneck (Qi et al. 2026; Yang et al. 2026a)**：将 clip 视为学习瓶颈；本文重新定位 clip 为稳定性工具，买取稳定性但牺牲少量早期速度。
- **LLM RL scaling (Yang et al. 2026b; Khatri et al. 2026)**：发现大批量可达更高最终精度但学习更慢；本文双时钟框架将这两个观察统一：大批量精度上限更高（因 clip 保护），但进度受曲率制约。
- **Trust-region alternatives (Dwyer et al. 2025; Jia et al. 2026; Tian et al. 2026)**：用软信任域或 Mu-GRPO 放松硬 clip；本文结果指出 clip 的核心价值是抑制大批量内的 drift spread，为设计新 trust region 提供基准参照。

## 局限性与未来方向
- **单模型家族与单一任务域**：所有主实验基于 Qwen2.5-Math 系列，虽在 7B 和 Instruct checkpoint 上验证但覆盖仍有限，需在其他模型/任务（代码、多模态）上扩展。
- **局部二次模型假设**：理论基于单步局部二次近似，未覆盖长程非二次动力学；直接测量 Hessian 曲率项在大规模下计算昂贵，仅通过可检验预测间接验证。
- **种子间固有方差**：失败点存在跨种子波动，结论依赖 21 次实验的聚合缩放规律而非单次结果。
- **样本效率度量单一**：仅以 response 数衡量样本效率，token 效率、计算效率和 wall-clock 效率受生成长度、硬件和系统实现影响，未纳入分析。
- **未来方向**：将双时钟推广至其他 RLVR 变体（PPO、TRPO）；探索自适应切换点选择策略；研究 clip 替代方案在漂移控制上的等价/优越性。

## 研究启发与可借鉴点
- **"进度时钟 vs 失败时钟"分离思维**：将训练动力学分解为"能学到什么"和"何时崩溃"两个独立维度，可用于分析其他缩放规则（如线性缩放、Cosine schedule）在 RLVR 中的表现，是一个通用的分析框架。
- **多步 GRPO 的临界批量远小于梯度噪声尺度**：实践中应避免在 $B^*$ 以上运行批量，或采用 ramp 调度；这对 DAPO、GRPO 等系统的工程部署有直接指导意义。
- **Ratio clip 的价值重估**：clip 不是必须最小化的瓶颈，而是大批量训练的稳态必需品；放宽 clip 换取短期速度往往得不偿失，此结论可指导 clip 超参选择。
- **漂移阈值假说的普适性检验**：$N_{\mathrm{fail}} \propto \sqrt{b}/\eta$ 形式简单且预测力强，可应用于其他基于 policy gradient 的训练场景（如 offline RL、continued pretraining）的崩溃分析。
- **实验设计的对照策略**：通过 $\mu=1$（clip 不生效）vs $\mu=8$、有/无 clip、固定批量 vs ramp 的多维对照，干净地分离了曲率效应与 clip 效应，为后续工作提供了精妙的消融范式。

## 关键术语表
**Two-clock framework**：将多步 GRPO 训练动力学分解为两个独立时钟——进度时钟（决定达到目标精度所需的样本数）和失败时钟（决定崩溃发生的样本数），两者受不同机制控制。

**Drift per sample**：每消耗一个 response 所积累的确定性参数位移，在平方根规则下与批量无关，是进度时钟和失败时钟的共同基础。

**Progress clock**：基于局部二次损失模型，描述批量增大时曲率损失导致每样本进度线性衰减的现象，临界批量 $B^*$ 以下为近似样本不变，以上则样本效率显著下降。

**Failure clock / drift-threshold hypothesis**：假设崩溃发生在积累漂移达到固定阈值时，沿平方根规则下失败点 $N_{\mathrm{fail}} = C/\kappa$ 与批量无关，服从漂移缩放而非随机游走缩放。

**Critical batch ($B^*$)**：在平方根规则下进度时钟开始显著偏离样本不变的临界批量，随学习率增大而缩小，远低于梯度噪声尺度，区别于 McCandlish et al. 的固定学习率临界批量。

**Square-root rule**：Adam 优化器下的学习率缩放规则 $\eta \propto \sqrt{B_r}$，旨在抵消批量增大带来的梯度噪声降低，使每样本更新步长保持稳定。

**Ratio clip**：GRPO/DAPO 中截断 importance ratio $\rho_t$ 超出 $[1-\varepsilon_\mathrm{low}, 1+\varepsilon_\mathrm{high}]$ 范围的 token 梯度贡献，防止策略更新步长过大；本文发现其主要作用是购买稳定性而非限制速度。

**Sample-cost exponent ($\alpha$)**：描述样本需求随批量增长的幂律指数，$\alpha = \mathrm{d}\log N_\tau / \mathrm{d}\log B_r$，完全不变时 $\alpha=0$，双倍批量需双倍样本时 $\alpha=1$。

## 可复现要素
- **数据集**：DAPO-Math-17k（公开）、DeepScaleR-Preview（公开）、Qwen2.5-Math-1.5B/7B（开源权重）
- **代码**：论文声明开源，仓库为 `two-clocks-rlvr`（链接见 abstract）
- **关键超参**：$G=8$，temperature=1，max response tokens=3072，AdamW ($\beta=(0.9, 0.999)$, weight decay=0.1)，$\varepsilon_\mathrm{low}=0.2$, $\varepsilon_\mathrm{high}=0.28$，dual clip $c=10$，bf16 训练，每 8192 responses 评估一次（512 held-out prompts，mean@4）
- **硬件**：1.5B 模型用 4× NVIDIA RTX A6000，7B 用 4× H100，总计约 2000 GPU-hours
- **框架**：verl 0.9.1 + vLLM 0.19，4-GPU FSDP + parameter/optimizer offload
