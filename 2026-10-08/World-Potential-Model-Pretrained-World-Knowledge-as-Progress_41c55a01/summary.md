---
title: "World-Potential-Model-Pretrained-World-Knowledge-as-Progress"
source: https://arxiv.org/pdf/2610.09560v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-08 22:38:36"
field: "长视野智能体过程监督"
keywords: ["World Potential Model", "long-horizon agent", "process supervision", "pretrained world knowledge", "sparse reward", "ordinal progress", "potential-based reward shaping", "group-relative policy optimization"]
innovations: ["提出WPM抽象，利用预训练模型世界知识零样本评估任务相对进展", "锚定里程碑将序数判断转化为任务局部标量世界势能", "引入参数ρ控制过程敏感回报的时间衰减，显式保留中间进展信号"]
benchmarks: ["ALFWorld", "ScienceWorld"]
---

# 论文速读：World-Potential-Model-Pretrained-World-Knowledge-as-Progress

## 一句话总结
本文提出World Potential Model（WPM），利用预训练模型固有的世界知识从零样本评估长视野Agent的任务相对进展，并将其转化为过程敏感的步级监督信号；实验表明WPM引导的策略优化在所有配置下均优于仅依赖终端结果的GRPO基线。

## 研究问题与动机
1. **长视野智能体的稀疏监督问题**：长视野语言智能体通常只在任务终点获得二元结果反馈，难以区分推进性中间行为与停滞/倒退。
2. **现有评估器构建成本高**：传统价值函数和过程奖励模型需针对每个任务单独标注和训练，难以扩展到异构任务环境。
3. **预训练知识是否可被"萃取"为进展评估能力**：预训练模型已蕴含大量程序流程、因果关系的知识，能否无需任务特定微调直接识别相对进展？
4. **如何从序数判断过渡到可优化的标量信号**：如何在不需要全局校准数值的前提下，将相对进展判断转化为可用于策略梯度的过程敏感回报？

## 核心贡献（创新点）
1. **提出WPM抽象**：将预训练模型的形式化为目标条件化的任务相对进展评估器，首次将"进展识别"定义为预训练模型的可复用能力而非任务特定学习器。
2. **序数进展接口设计**：以有序比较（ordinal comparison）作为WPM的主接口，避免了开域任务中绝对数值进度难以定义的问题。
3. **锚定标量化机制**：通过单次专家轨迹生成任务特定的里程碑锚点（milestone ladder），将预训练模型的序数判断锚定到任务局部标量坐标系，实现进展评估与标量化解耦。
4. **过程敏感回报构造**：引入参数ρ控制中间势能在时间上的衰减权重，使回报显式包含各中间状态进展而不仅依赖首尾差分，突破了经典势函数奖励 shaping 的策略无关性限制。

## 方法详解

**WPM基本形式化**：给定任务目标g，WPM评估agent上下文xₜ中已实现的与g相关的进展量，输出任务相对的进展结构（world potential）。不预测未来回报，而是回顾性地刻画"已完成什么"。

**序数进展接口**：对于同一任务的两个上下文xᵢ和xⱼ，WPM判断xᵢ ≽_g xⱼ（xᵢ至少具有与xⱼ同样多的目标相关进展）。无需绝对数值，只需相对排序。

**锚定标量世界势能**：
- 任务特定的里程碑锚点集 M_g = {(m_k, φ_k)}_{k=0}^K，其中φ₀=0，φ_K=1，严格递增
- 给定上下文xₜ，找到最大满足xₜ ≽_g m_k的锚点索引z_t
- 标量世界势能 Φ_g(xₜ) = φ_{z_t}

**世界势能转移信号**（公式9）：
r_t^WP = r_t^env + γΦ_g(x_{t+1}) − Φ_g(x_t)
正向进展获得正贡献，停滞或倒退获得轻微负信号。

**过程敏感回报**（公式11-12）：
引入ρ∈[0,1]控制时间衰减尺度，β=γρ。回报展开为四项：
- 环境回报的折现和
- 当前上下文势能的负贡献
- 中间世界势能的加权求和（权重w_k(ρ)=γ^k(1−ρ)ρ^{k−1}，几何衰减）
- 终端上下文势能的折现贡献

关键性质：0<ρ<1时中间势能显式出现；ρ=0退化为单步差分；ρ=1时中间项 telescoping 抵消。

**Step-level Group-Relative Advantages**（公式17）：
对每个有效环境步计算WP回报，在同一任务组的 rollout 内做group-relative标准化得到Â_{i,t}，应用于GRPO策略梯度更新。

**超参设置**：γ=0.99，ρ=0.95，有效回报折现β=0.9405，等效时间尺度≈16.8步。

## 实验与结果

**能力评估（Table 1）**：
- 数据集：ALFWorld和ScienceWorld
- 评估1（Pairwise Progress Comparison）：所有模型远超50%随机基线，平均准确率92.52%，最佳Qwen3-VL-30B-A3B在ScienceWorld达98.00%，ALFWorld达95.33%
- 评估2（Anchored Progress Localization）：平均准确率74.71%，最佳Qwen3-VL-30B-A3B在ALFWorld达88.00%，ScienceWorld达85.33%
- 证明零样本预训练模型具备可靠的进展识别能力

**策略优化（Table 2）**：
- 评估器：冻结的Qwen3-VL-32B（无需微调）
- Actor模型：Qwen2.5-3B/7B、Qwen3-4B
- 每组3个随机种子取平均

| 模型 | 环境 | Vanilla | GRPO | WPM |
|------|------|---------|------|-----|
| Qwen2.5-3B | ALFWorld | 10.93% | 53.90% | **60.50%** (+6.60pp) |
| Qwen2.5-3B | ScienceWorld | 8.59% | 51.56% | **55.39%** (+3.83pp) |
| Qwen2.5-7B | ALFWorld | 15.62% | 68.70% | **72.10%** (+3.40pp) |
| Qwen2.5-7B | ScienceWorld | 13.28% | 66.40% | **74.73%** (+8.33pp) |
| Qwen3-4B | ALFWorld | 22.65% | 48.43% | **53.64%** (+5.21pp) |
| Qwen3-4B | ScienceWorld | 19.53% | 42.96% | **46.09%** (+3.13pp) |

- 最优结果：Qwen2.5-7B在ScienceWorld从66.40%提升至74.73%（+8.33pp）
- 平均提升：5.08个百分点
- 在所有6组配置下一致优于outcome-only GRPO

## 相关工作脉络

1. **Process Reward Models**（如AgentPRM、ProgRM）：通过子目标/进展标注训练任务特定的奖励模型；WPM不训练评估器参数，直接从预训练知识中萃取进展判断。
2. **Foundation Models as Zero-shot Reward Models**（Text2Reward、Eureka、RL-VLM-f）：利用预训练模型生成奖励函数或 judgments；本文聚焦"相对进展识别"这一特定接口，强调ordinal比较而非绝对数值打分。
3. **Subgoal-driven Frameworks**（如Subgoal-driven long-horizon agents）：显式规划子目标序列；WPM的里程碑锚点仅需单次专家演示生成，不要求在线分解或搜索。
4. **Potential-based Reward Shaping**（Ng et al.）：经典理论保证策略不变性；本文对0<ρ<1的情形，中间势能显式保留在过程敏感回报中，打破了经典shaping的策略无关性，使回报能区分"何时达成进展"。
5. **Intrinsic Credit Assignment / Hindsight approaches**：从轨迹回放或反事实推理获取信用；WPM完全零样本、无需额外rollout数据，依赖预训练模型本身的知识。
6. **LLMs as Zero-shot Planners**（Huang et al.）：证明预训练模型蕴含可提取的程序性知识；本文将此观察推进到"进展评估"层面，并连接到策略优化的具体信号构造。

## 局限性与未来方向

1. **实验范围有限**：仅在ALFWorld和ScienceWorld两个结构化环境中验证，尚未在更开放的web导航、GUI操作或多智能体协作场景中检验。
2. **锚点构造依赖单次演示**：仅用一条成功专家轨迹生成里程碑，若演示路径非最优或遗漏关键状态分支，锚点集可能不完整。
3. **序数判断的不确定性**：即便准确率较高（80-90%），WPM的误判仍会引入噪声信号；未讨论评估器误差对策略收敛的影响。
4. **未涉及探索效率**：WPM评估"已实现进展"而非"潜在可达进展"，可能无法区分冗余探索与有效探索。
5. **未来方向**：扩展到更广泛的agent设定、研究自适应锚点生成、分析ρ的敏感性、探索与value function的结合。

## 研究启发与可借鉴点

1. **预训练知识的"零样本能力萃取"**：将预训练模型视为蕴含世界知识的知识库，无需微调即可用于监督信号构造，降低了长视野agent过程监督的部署成本。
2. **"序数接口+锚定标量化"的两阶段设计**：先在序数空间做判断（更鲁棒），再参考任务特定的锚点映射到标量（更易优化），这一解耦思路可迁移到其他需要进展评估的场景。
3. **参数ρ控制时间尺度的灵活性**：通过单一超参调节对短期vs长期进展的敏感度，且理论上明确了最优敏感度与目标距离的关系（ρ*_k=(k−1)/k），为后续调参提供指导。
4. **冻结评估器+策略学习的分离架构**：评估器全程冻结，只产生监督信号不参与梯度回传，避免了额外训练开销，适合资源受限的部署场景。
5. **可结合团队现有方向的创新机会**：将WPM与团队在过程奖励建模、agent planning或long-horizon reasoning方向的工作结合，尤其是用预训练模型的进展感知替代或增强task-specific process reward。

## 关键术语表

**World Potential Model (WPM)**：利用预训练世界知识评估agent上下文中已实现的与目标相关的进展的抽象评估器，输出任务相对的进展结构。

**Ordinal Progress Interface**：WPM的主接口，通过成对比较判断哪个上下文具有更多目标相关的已实现进展，无需绝对数值。

**Anchored Scalar World Potential**：将WPM的序数判断锚定到任务特定的有序里程碑集{m_k}，得到标量坐标φ_{z_t}作为世界势能值。

**Process-Sensitive Return**：通过引入参数ρ将世界势能的时间差分整合为步级回报，显式保留中间状态的进展信息，区别于仅依赖终端结果的回报。

**World-Potential Transition Signal**：r_t^WP = r_t^env + γΦ_g(x_{t+1}) − Φ_g(x_t)，由环境奖励和世界势能变化构成的步级信号。

**Parameter ρ**：过程敏感性参数（0≤ρ≤1），控制中间世界势能在回报中的时间衰减权重；ρ越小侧重近期进展，ρ越大允许远期进展影响当前评价。

**Step-level Group-Relative Advantage**：在相同任务的rollout组内对WP回报做标准化，为每个有效环境步分配差异化优势值，用于GRPO策略更新。

**Milestone Ladder**：由专家演示生成的有序里程碑集合，每个里程碑描述一个可验证的进展状态及对应的标量坐标，构成任务局部的参照框架。

## 可复现要素

- **数据集**：ALFWorld（官方train/valid_unseen split，3553训练、134评估）、ScienceWorld（官方train/test split，3589训练、1819测试）— 公开可下载
- **代码**：已开源，GitHub: https://github.com/Gabrile166/world-potential-model
- **评估器模型**：Qwen3-VL-32B-Thinking-FP8（HuggingFace公开权重，Apache 2.0）
- **Actor模型**：Qwen2.5-Instruct-3B/7B（Qwen Research License）、Qwen3-4B（Apache 2.0）
- **关键超参**：γ=0.99，ρ=0.95，β=0.9405；每步16个task group×8条rollout=128条轨迹；LR=1×10⁻⁶，100 epochs；clip PPO-style；每rollout上限30步
- **环境包**：ALFWorld官方Python包；ScienceWorld实现在仓库内
- **硬件**：8× NVIDIA H100 GPU

---
