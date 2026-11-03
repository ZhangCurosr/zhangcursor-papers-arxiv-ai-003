---
title: "Two-Clocks-Why-Larger-Rollout-Batches-Learn-Slower-but-Fail"
source: https://arxiv.org/pdf/2610.18853v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-03 01:35:17"
---

# 论文速读：Two Clocks: Why Larger Rollout Batches Learn Slower but Fail on Schedule in Reinforcement Learning with Verifiable Rewards

## 一句话总结
本文提出“双时钟”理论框架，解释在多步优化器训练（multi-step）的RLVR中，平方根学习率规则虽能保持每样本的累积漂移不变，却因优化步数减少与曲率损耗叠加而导致大batch样本效率下降（进度时钟），同时崩溃却严格发生在相近的已消耗样本数上（失败时钟）；并据此推导出“先小后大”的batch ramp调度策略以提升实际训练效率。

## 研究问题与动机
- RLVR（如GRPO/DAPO）高度依赖模型生成数据，实践中将一次rollout拆分为多个mini-batch多次更新（DAPO常用μ=16），而非常规的每轮仅1步。
- 既往工作仅验证单步（one-step）场景下平方根规则能保持batch size无关的样本效率，多步训练下该规则的动态机制缺乏统一解释。
- 实践中同时观察到三个现象：①大batch学习速度随batch增大而变慢；②无ratio clip时大batch与小batch在相似样本消耗量后崩溃；③clip对大batch的稳定作用最强。现有文献分别研究batch缩放、临界batch或clip瓶颈，但未能解释三者为何在同一条平方根规则下共生。

## 核心贡献（创新点）
- 提出**双时钟框架**：基于局部二次损失模型推导进度时钟（受曲率项拖累，超过学习率相关的临界batch后样本效率线性退化），并用漂移阈值假设刻画失败时钟（沿平方根规则样本数近似不变）。
- 在16倍rollout batch范围内对Qwen2.5-Math-1.5B进行受控GRPO实验，系统验证三项预测；证明进度差距在关闭clip、更换数据集/通用checkpoint/7B模型后依然稳健复现。
- 基于双时钟导出实用的**batch ramp调度策略**：先在临界batch以下高效积累进度，接近预测失败点前切换至较大batch兼顾吞吐与稳定性，实验显示该策略接近最优固定batch性能且75%样本由大batch生成。

## 方法详解
- **漂移与进度分解**：设优化步Δθ = -ηĝ/ν，在平方根规则(η = κ√b)下，每个consumed response的均值漂移 d = κ‖ḡ‖/σ 与batch大小无关。
- **进度时钟（Progress Clock）**：loss局部二次展开后，每样本损失下降 r(b) = r₀ - C_curv·b，曲率项 C_curv = κ²(ḡᵀHḡ)/(2σ²)。到达目标accuracy所需样本 N_τ ≈ N_min(1 + B_r/B*)，临界batch B* = μ(2σ‖ḡ‖²/(κḡᵀHḡ) - B_noise) 随学习率κ增大而缩小，且远低于梯度噪声尺度。
- **失败时钟（Failure Clock）**：引入漂移阈值假设(A3)，崩溃发生在累积漂移 ‖∑EΔθ_j‖ 达到阈值D时。推导得 N_fail = C·√b/η，沿平方根规则退化为 N_fail = C/κ，与batch无关；拟合21次无clip实验支持漂移规律（γ≈0.7），拒绝噪声驱动随机游走（γ≈2）与batch无关假设。
- **Clip的作用机制**：rollout内log-ratio spread随步数扩散，大batch每步更长导致spread快√(B_r^L/B_r^S)倍；clip截断偏离advantage方向的token。replay实验表明clip仅移除大batch约0.5%的token，却截去了>50%的梯度范数，故大batch从中获得更强稳定性；但clip不改变曲率项，无法弥补进度差距。
- **调度策略**：当吞吐需求迫使使用大batch时，采用ramp schedule——在B*以下先用小batch高效训练，接近预测失败点前重置Adam状态并切换至大batch继续，切换时机在25k~74k样本区间内表现稳健。

## 实验与结果
- **设置**：Qwen2.5-Math-1.5B + DAPO-Math-17k，GRPO，G=8，temp=1，response≤3072，AdamW (β=(0.9,0.999), wd=0.1)，关闭KL/entropy/动态采样。Rollout batch: 512/1536/4096/8192 (S/M/L/XL)，μ=8，η_S∈{1e-6, 5e-6, 1e-5}按√规则缩放。崩溃判据：准确率降至历史最佳一半，或梯度范数持续>早期中位数4倍。
- **进度时钟**：μ=1（clip完全无效）时大batch仍比小batch多需20-45%样本达到15%准确率；μ=8时样本成本指数α从1e-6时的0.15升至1e-5时的0.44，临界batch从~9k降至~2k。进度差距在DeepScaleR-Preview、Qwen2.5-1.5B-Instruct及Qwen2.5-Math-7B上均复现。
- **失败时钟**：无clip时4096 batch在56-80步后崩溃，512 batch在648步后崩溃，但两者均在30k-50k样本处失败；Tobit拟合β-γ/2≈0.11（含censored），支持漂移缩放。
- **Clip效应**：开启clip后小batch失败时间延长2倍，大batch延长4倍（未达98k预算上限）；拓宽clip半径可换取早期加速但丧失大batch稳定性（1个seed崩溃），验证“clip购买稳定性而非持续速度”。
- **调度策略**：512→4096/8192的ramp策略在25k-74k样本处切换表现相似，最终曲线接近最优固定batch，且75%样本由大batch生成。

## 相关工作脉络
- **Batch-size invariance (Hilton et al., 2022; Li et al., 2026)**：证明单步下PPO/GRPO可实现batch无关；本文指出多步下该不变性失效，根源在于曲率损失。
- **Critical batch (McCandlish et al., 2018)**：针对每batch重调LR场景，临界batch等于梯度噪声尺度且与LR无关；本文证明固定√规则下临界batch随LR增大而缩小，可远低于噪声尺度。
- **Training collapse (Suk & Duan, 2025; Song et al., 2026)**：关注step size阈值或异步GRPO崩溃时间；本文补充batch维度，证明沿√规则崩溃时间由累积漂移决定且在样本空间近似不变。
- **Ratio clipping (Qi et al., 2026; Yang et al., 2026a)**：视clip为学习瓶颈；本文定量证明clip主要移除大batch中梯度范数主导的少数token，作用是稳定而非限速。
- **RLVR scaling (Khatri et al., 2026; Yang et al., 2026b)**：发现大batch可达更高准确率但学习更慢；本文用双时钟统一解释“更快但最终更高”与“更慢但更早崩溃”并存的观测。

## 局限性与未来方向
- 理论基于局部二次近似，未直接测量Hessian曲率项，仅通过可检验预测间接验证。
- 实验仅覆盖单一模型族（Qwen2.5）与数学推理任务，未测试代码、对话或更长序列场景。
- 崩溃点存在种子间固有波动，且仅衡量response级样本效率，未纳入token级、算力与wall-clock成本。
- 未来方向可包括
