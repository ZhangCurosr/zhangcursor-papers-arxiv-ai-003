---
title: "TEACHING-LLMS-TO-GENERATE-CHALLENGING-MILP-INSTANCES-VIA-SOL"
source: https://arxiv.org/pdf/2609.37356v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 19:59:07"
field: "组合优化与机器学习交叉"
keywords: ["MILP instance generation", "solver feedback", "reinforcement learning", "large language models", "challenging problem generation", "branch-and-bound", "self-play"]
innovations: ["固定求解器 verifier 的非对称自对弈训练范式，使难度保持绝对标度", "割后读取的复合奖励函数（节点数+间隙+有效性闸+多样性），防御虚假难度", "紧凑索引集模板实现 token 高效的长 MILP 实例生成"]
benchmarks: ["MIPLIB 2010/2017", "D-MIPLIB", "MILP-Evolve", "Ecole synthetic generators", "MIRPLIB", "DIG-MILP"]
---

# 论文速读：TEACHING-LLMS-TO-GENERATE-CHALLENGING-MILP-INSTANCES-VIA-SOL

## 一句话总结
论文提出 OptiScribe，一种基于求解器反馈的无种子非对称自对弈训练框架，将 GRPO 强化学习与固定分支定界求解器结合，直接教授 LLM 从自然语言指令生成**可行且计算挑战性高**的混合整数线性规划（MILP）实例。

## 研究问题与动机
- **现有无 LLM 生成器的缺陷**：手写随机生成器（如 Ecole）多产出小/易实例；基于种子的方法（G2MILP、ACM-MILP）依赖既有种子实例，输出始终贴近种子分布，无法脱离种子独立生成目标家族实例。
- **现有 LLM 生成器的缺陷**：MILP-Evolve、MILP-Retrieval 让 LLM 演化/检索生成器程序，但 LLM 本身从未直接学习"什么让实例更难"，每次面对新家族或新规模均需重新运行求解器循环的参数搜索。
- **现有 RL 验证器反馈未对齐难度目标**：Absolute Zero、R-Zero、STP 等仅返回二元正确性信号（pass/fail），难度相对于共训练的 solver 漂移，没有绝对标度；而 MILP 求解器在求解管道各阶段报告可用努力量度。
- **公共基准库覆盖稀疏且双峰化**：同规模公共实例（MIPLIB、D-MIPLIB 等）多在极易（根节点即解出）或极难（逼近 50,000 节点上限）两端，中间难度区间严重空缺。

## 核心贡献（创新点）
1. **无种子非对称自对弈训练范式**：LLM 作为 challenger 持续产出实例，固定配置的分支割求解器（SCIP）作为 verifier 评估可行性与求解努力，二者不对称——仅 challenger 在线更新，彻底免除对种子实例与推理时求解器调用的依赖；与 Absolute Zero/R-Zero 等共训练双模型的相对难度标度本质不同，难度在所有检查点保持一致的绝对量纲。
2. **防御"虚假难度"的求解器努力奖励函数**：设计包含有效性闸（validity gate）、支点后割平面间隙、分支定界节点数、规模保持与结构多样性四项的复合奖励；关键机制是**在预处理和根割平面之后读取难度信号**，使得依赖弱 formulated big-M 链接或病态系数制造的表面难度被求解器自动修复而不获奖励，与仅奖励原始 LP 间隙的前序工作形成本质区别。
3. **紧凑索引集模板（Compact Index-Set Template）表示**：实例以结构块+JSON 数据块描述，token 开销随参数数量而非约束条数增长（最大规模下仅为显式矩阵编码的约 5–10%），配合确定性解析器还原为显式 MILP，使长实例在 LLM 生成长度限制内仍可行。
4. **语言可控的硬实例生成**：模型在保持基础指令遵循能力的前提下，能通过追加自然语言句子（如"生成更难的实例"、"更稀疏的图"）定向偏移难度分布；同时可生成公共库缺失的特定家族/规模组合以支持求解器参数调优。
5. **跨求解器鲁棒性验证**：在推理时用 HiGHS 和 Gurobi 两个训练外求解器重新求解全部生成实例，证实难度排序不因求解器算法差异而逆转，表明学到的是问题本身的结构困难而非单一求解器的盲区。

## 方法详解
**整体框架（图 1）**：给定含 MILP 家族名与目标规模区间的提示词 $p$，策略 $\pi_\theta$ 采样 $G=64$ 个完成 $o_i$；确定性解析器将其展开为实例 $\mathcal{T}_i$；冻结的 SCIP（单线程、50,000 节点上限、20 s 时间限制）求解并返回有效性闸 $V$、割后间隙 $g_{\mathrm{cut}}$ 与节点数 $N$；据此计算奖励后以 LoRA + GRPO 更新 $\pi_\theta$，KL 惩罚系数 $\beta=0.1$。

**奖励函数（公式 2–5）**：
$$R_i = V(\mathcal{T}_i) \cdot \big[ w_H H(\mathcal{T}_i) + w_v r_{\mathrm{var}}(\mathcal{T}_i) + w_d r_{\mathrm{div}}(\mathcal{T}_i) \big]$$
- **有效性闸** $V \in \{0,1\}$：六重条件合取——可解析、有可行有限最优值、所有连续变量有有限上界、禁止聚合型 big-M 单二进制链接、系数范围不超 $10^4$、家族合规。任一失败则得零分。
- **难度** $H = 0.25\, r_{\mathrm{cut}} + 0.75\, r_{\mathrm{node}}$：割后相对间隙 $r_{\mathrm{cut}}=\mathrm{clip}(g_{\mathrm{cut}},0,1)$（$g_{\mathrm{cut}}=|z^*-z_{\mathrm{cut}}|/(0.10|z^*|)$）与对数缩放的节点项 $r_{\mathrm{node}}=\mathrm{clip}(\log N/\log N_{\mathrm{ref}},0,1)$（$N_{\mathrm{ref}}=5000$）。
- **规模保持** $r_{\mathrm{var}}=\exp(-\alpha|n_i-n^\star|/n^\star)$（$\alpha=3$，$n^\star$ 为区间中点），软惩罚避免模型通过随意增大规模"购买"难度。
- **结构多样性** $r_{\mathrm{div}}$：基于约束行指纹（行方向、非零元数、整数列占比、系数符号集）在多集合内的排名，归一化至 $[0,1]$，恰好平均为 0.5，仅在组内重分配优势，防止模式坍塌。默认权重 $(w_H,w_v,w_d)=(0.70,0.15,0.15)$。

**GRPO 优势**：$A_i = (R_i - \mathrm{mean}_j R_j)/(\mathrm{std}_j R_j + \epsilon)$，仅在同组内比较，跨家族不混排。

**规模课程**：三阶段依次训练 76–110→111–170→171–225 变量，每阶段 61 步，前阶段检查点初始化下一阶段；直接训练（OPTISCRIBE-12B-D）仅在目标区间训练。

## 实验与结果
- **数据集与基线**：三个 MILP 家族（CFL、Max-Cut、多重背包）；对比基线为未微调的 Gemma-4-12B-it / Qwen3.5-4B、直接训练版、标准合成生成器（Ecole 四族）、八个公共库（MIPLIB 2010/2017、MILP-Evolve、D-MIPLIB 等）的 111–500 变量实例；推理时额外使用 HiGHS 和 Gurobi 13.0.3。
- **OptiScribe-12B vs Gemma-4-12B（SCIP）**（Table 2）：
  - CFL：中位节点从 22 升至 109（B1）→ 391（B2）→ 710（B3）→ 1014（B4），较基线提升 **1.7–5.0 倍**；割后间隙提升 **1.4–1.7×10⁻³**；可行性率提升 **9–19 个百分点**。
  - Max-Cut：中位节点从 41 升至 142（B1）→ 364（B2）→ 1002（B3）→ 2578（B4），提升 **1.9–4.5 倍**；间隙提升 **1.1–1.7×10⁻³**。
  - B3/B4 外推仍保持显著优势（CFL 2.4 倍、Max-Cut 2.4/1.9 倍）。
- **OptiScribe-4B vs Qwen3.5-4B（SCIP）**（Table 3）：
  - CFL 中位节点最高达基线的 **26.68 倍**（xH 版本），Gap 提升最高 **2.75 个百分点**；Max-Cut 最高 **7.22 倍**，Gap 提升 **13.40 个百分点**。
  - xH 变体（去多样性项、提高 $w_H$）在所有单元格的节点数最高。
- **跨求解器鲁棒性**（Figure 3, Table 13）：HiGHS 与 Gurobi 上 OptiScribe-12B 难度排序不变；Gurobi 在 CFL 上根节点解出比例较高，但总体趋势一致。
- **对比公共基准**（Figure 2）：公共 111–500 变量实例呈明显双峰（极易或触 50k 上限）；OptiScribe 生成池覆盖整个难度谱，填补中间空白。
- **语言控制**（Figure 4）："Harder" 提示在 Max-Cut B2 上将节点从 344 提升至 400；密度指令（$\kappa=0.8/0.5/0.2$）两模型均能遵循次序；"easier" 同样有效。
- **求解器调优应用**（Section 4.6）：用 40 个生成实例（226–350 变量）在 MILP-Evolve 无 CFL 同规模数据的条件下，为 SCIP 选出最优配置使平均求解时间降低 **16.0%**（95% CI 6.3–25.5%）；跨域迁移时 PAR2¹ 从 15.15 降至 7.50 s。
- **多重背包**：所有模型均未能使其变难（中位节点仅 3–5，间隙约 0.5%），为方法边界案例。

## 相关工作脉络
1. **Verifier-in-the-loop RL（Absolute Zero, R-Zero, STP）**：这三者difficulty 相对共训练的 solver 漂移，verifier 仅返回 pass/fail 二元信号；本文使用固定求解器，奖励含绝对难度的连续量（节点数、间隙），且 verifier 不参与训练。
2. **Seed-based MILP 生成器（G2MILP, ACM-MILP, MILP-StuDio）**：依赖种子实例编辑，输出分布贴近种子；本文完全无种子、无需推理时求解器调用。
3. **LLM-based 代码演化生成器（MILP-Evolve, MILP-Retrieval）**：LLM 不直接学习难度，仅演化/检索生成器程序并通过求解器循环搜索参数；本文 LLM 自身直接学习难度特征，输入为自然语言。
4. **ORLM（Huang et al., 2025）/ SIRL（Chen et al., 2025）**：两者均针对用户提供的已有优化问题进行验证，奖励正确性而非求解努力，不产生新问题；本文专为生成新挑战性问题设计。
5. **Synthetic generators (Ecole)**：标准合成生成器在 111–500 变量范围内 55% 于根节点解出，最多 189 节点；OptiScribe 在同一规模中位节点 347–2578，显著更难。
6. **Instance-space 进化方法（Smith-Miles & Bowly, 2015; Bowly, 2019）**：在实例特征空间中搜索目标区域，需人工设定难度标度；本文通过求解器自动提供难度信号，无需手工调参。

## 局限性与未来方向
- **家族覆盖有限**：仅训练了 CFL、Max-Cut、多重背包三个家族；其中多重背包对所有模型均保持极易，方法在该家族未奏效。
- **实例规模上限 500 变量**：未探索更大规模下的生成能力与难度标度稳定性。
- **每臂仅训练一次**：缺乏重复实验统计，调优结论（求解器调优）基于少量实例，泛化性待验证。
- **课程依赖模型能力**：Gemma-4-12B 需规模课程才稳定收敛，直接训练在目标区间崩溃（有效完成率从 >90% 降至 <60%）；Qwen3.5-4B 更稳健但课程仍带来额外收益。
- **语言控制在 CFL 上失效**：CFL 密度指令未被模型遵循（"sparser" 提示反而使实例更容易且原因不明），推断受问题数学结构约束（Max-Cut 的 LP 界等于总边权，稀疏图天然缩小界gap；CFL 无类似结构）。
- **未来方向**：扩展到更多 MILP 家族、更大规模、更广泛的求解器评估、探索课程对不同架构模型的普适性、挖掘 CFL 密度控制失效的根源。

## 研究启发与可借鉴点
1. **固定 verifier + 连续努力信号**替代二元 pass/fail 用于 RL 训练的思路可迁移至其他需要优化"难度/质量梯度"而非单纯"正确性"的生成任务（如 SAT 实例生成、约束满足问题、程序合成中的性能导向训练）。
2. **复合奖励设计的防作弊机制**：有效性闸（六重规则）+ 割后读取 + 系数范围约束，形成一套"阻止模型走捷径制造虚假难度"的通用范式；在优化问题以外的领域可类比设计"防止生成器通过格式/数值技巧作弊"的验证条件。
3. **规模课程触发模型稳定收敛**：大模型直接在目标尺度训练易崩溃（token cap 导致大量零分），从小尺度逐步扩展可显著提升训练稳定性——此策略适用于任何输出尺度影响完成率的生成任务。
4. **跨求解器/跨环境验证协议**：训练用一种 solver，推理用两种外置 solver 检验难度排序是否保持，是验证"学到了真实难度而非 solver 盲区"的黄金标准，可推广至任何 solver-in-the-loop 方法。
5. **语言控制保留指令遵循能力**：证明在 RL 训练后追加未见过的自然语言提示仍可定向调节输出分布，为"可交互生成器"提供了可行的范式；对于本团队方向，可探索在同一框架下通过 instruction tuning 加入多目标（如"高难度+高稀疏性"）联合控制。

## 关键术语表
- **Branch-and-cut（分支割）**：MILP 求解的主流算法，在分支定界框架下周期性添加割平面以收紧线性松弛。
- **Post-cut root gap（割后根间隙）**：根节点割平面添加完毕后线性松弛最优值与整数最优值的相对差，反映切割后仍存的整数间隙。
- **GRPO（Group Relative Policy Optimization）**：无 critic 的强化学习算法，以组内标准化奖励作为 advantage 信号更新策略。
- **Validity gate（有效性闸）**：乘法门控函数，实例须通过六项结构性与可行性检查方获得非零奖励。
- **Asymmetric self-play（非对称自对弈）**：challenger 在线学习、verifier 固定不动的对抗训练模式，区别于双方共同进化的对称自对弈。
- **Compact index-set template（紧凑索引集模板）**：以结构化文本+JSON 数据描述 MILP，结构块不随规模增长，token 效率远高于显式矩阵。
- **Node reference $N_{\mathrm{ref}}$**：节点奖励项的对数饱和参考值（默认 5,000），决定节点数的尺度归一化。
- **Structural diversity reward（结构多样性奖励）**：基于约束行指纹稀有度的组内排名奖励，鼓励生成结构多样的实例。

## 可复现要素
- **数据集**：三个自建 MILP 家族（CFL、Max-Cut、多重背包）；公共基准使用 MIPLIB 2010/2017、MILP-Evolve、D-MIPLIB、DIG-MILP、MIPcc23、MIPLearn、MIRPLIB 等，从各源筛选 111–500 变量实例；Ecole 标准合成生成器。
- **代码/模型开源声明**：论文明确声明"will release our code and models publicly on acceptance"（接受后公开）；截至论文发表时尚未开源。
- **关键超参**：LoRA rank=16；GRPO 组大小 $G=64$；学习率 $5\times10^{-5}$；KL 系数 $\beta=0.1$；温度 $T=1.0$、top-p=0.95；每阶段 61 步；$N_{\mathrm{ref}}=5000$；节点上限 50,000；训练时间上限 20 s；奖励权重 $(w_H, w_v, w_d)=(0.70, 0.15, 0.15)$；规模课程三阶段 76–110→111–170→171–225 变量。
- **训练环境**：SCIP 10.0（单线程，训练内求解器）；HiGHS、Gurobi 13.0.3（推理评估）。
