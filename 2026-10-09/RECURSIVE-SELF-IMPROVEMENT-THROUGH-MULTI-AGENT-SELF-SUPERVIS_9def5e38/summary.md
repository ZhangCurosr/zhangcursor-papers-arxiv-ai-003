---
title: "RECURSIVE-SELF-IMPROVEMENT-THROUGH-MULTI-AGENT-SELF-SUPERVIS"
source: https://arxiv.org/pdf/2610.12176v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-09 10:19:36"
field: "多代理智能体与模型自我改进"
keywords: ["recursive self-improvement", "multi-agent", "self-supervision", "workflow optimization", "SFT", "agent coordination", "language model"]
innovations: ["提出MASS方法通过多代理工作流优化与SFT交替打破RSI同构循环的自我强化偏差", "证明多代理轨迹比单代理轨迹具有更高的训练效率（更少token达成更好性能）", "揭示工作流优化而非评估器能力是RSI速度瓶颈"]
benchmarks: ["ScienceAgentBench", "MLR-Bench", "AstaBench", "DSBench", "Terminal-Bench 2.0", "SWE-bench Verified"]
---

# 论文速读：RECURSIVE-SELF-IMPROVEMENT-THROUGH-MULTI-AGENT-SELF-SUPERVIS

## 一句话总结
论文提出 **MASS（Multi-Agent Self-Supervision）** 方法，通过多代理工作流优化与监督微调的交替循环，打破递归自我改进（RSI）中单模型同构循环的自我强化偏差；在开放研究任务上，两个MASS周期后 Qwen3.6-27B 在四项研究基准上每输出 token 得分提升 **1.2–1.6×**，并实现更高效的训练。

---

## 研究问题与动机
- **监督瓶颈**：在超越人类专家可靠评估能力的开放研究任务上，模型自身成为最好的优化器和评估器；但单模型在"生成-评估-改进"的同质循环中共享相同先验，易强化而非纠正错误。
- **手动工作流优化成本高昂**：手动调整任务分解、工具使用与验证步骤以优化工作流在不同任务和领域间需要极大努力。
- **多代理拓扑的潜力**：早期研究发现多代理工作流在开放研究任务上优于单代理执行（Lee et al., 2026），受 Minsky "Society of Mind" 启发，认为组织化的简单组件交互可激发单个实例难以稳定展现的能力。
- **两假设驱动设计**：① 优化代理间信息流（而非单纯增加代理数量）可提供有效自我监督；② 联合训练编排与有界子代理执行比训练长周期单代理轨迹具有更高的学习效率。

---

## 核心贡献（创新点）
1. **提出 MASS 算法**：通过迭代的多代理工作流优化 + SFT 内化实现递归自我改进，打破单模型同构循环的自我强化偏差。  
   *与已有工作区别*：区别于 RHI 仅修订工作流提示，MASS 进一步将多代理轨迹蒸馏为共享权重。

2. **工作流作为文本信息编码**：将多代理协调（编排器-子代理架构、信息路由、输出契约）编码为工作流提示中的文本规范，通过进化搜索约束结构护栏进行优化。  
   *与已有工作区别*：区别于 ADAS/AFlow/GPTSwarm 等自动设计代理程序/通信图的工作，MASS 在此基础上进一步联合学习编排和子代理执行并更新权重。

3. **证明多代理轨迹的训练效率优势**：在多代理学生实验中，用约 1.4× 更少 token 的单代理基准下，多代理学生达到更高胜率（68.3% vs 64.0%）。  
   *与已有工作区别*：区别于 SiriuS 等仅利用多代理交互进行训练的方法，本文系统性比较了多代理 vs 单代理轨迹的学习效率。

4. **揭示工作流优化而非评估器能力是 RSI 瓶颈**：消融实验表明更强的优化器能显著提升发现成功工作流的速度，而替换评估器的效果有限。  
   *与已有工作区别*：首次系统量化 RSI 同构循环中各角色能力对改进速度的贡献。

5. **展示跨角色能力迁移**：仅对任务解决轨迹进行微调，即可同步提升工作流优化速度（从第 0 轮到第 1 轮发现成功工作流的时间提前）和自评估准确性（73% → 93%）。  
   *与已有工作区别*：证明无需为不同角色单独设计训练目标，单一 SFT 即可带来 RSI 全角色的能力提升。

---

## 方法详解
**整体框架（Algorithm 1）**：MASS 在世代 $g = 0, \ldots, R-1$ 交替执行两个阶段：
1. **内循环：多代理工作流优化（Algorithm 2）**
   - 工作流定义 $w_t = \{\mathbf{h}_t, (\mathbf{r}_{t,j}, \mathbf{u}_{t,j}, \mathbf{c}_{t,j})_{j=1}^{m_t}\}$，包含编排顺序 $\mathbf{h}_t$、各子代理的角色 $\mathbf{r}$、指令 $\mathbf{u}$ 和输出契约 $\mathbf{c}$。
   - 在当前模型 $\mathcal{L}^{(g)}$ 统一扮演执行器/评估器/优化器角色，对每个训练任务独立维护最佳工作流 $w_t^*$，通过序列化成对比较迭代搜索。
   - 优化提示 $x_{opt}$ 显式强调信息结构 $(\mathbf{h}_t, \{\mathbf{c}_{t,j}\})$ 而非角色/指令组件，引导优化器优先改进信息流。

2. **外循环：工作流内化的 SFT（Algorithm 3）**
   - 对每个训练任务执行 $M$ 次独立 rollout，通过 Bradley–Terry 模型将成对评判转化为工作空间分数，选取 Top-$K-1$ 轨迹作为 SFT 训练集，保留第 $K$ 个作为验证集。
   - 将编排器初始工作流提示替换为裸任务提示（移除工作流文本），保留子代理分配，以助理 token loss mask 训练，得到 $\mathcal{L}^{(g+1)}$。

**关键细节**：
- SFT 损失函数：$\ell_{\text{SFT}}(\theta) = -\mathbb{E}_{z \sim q}\left[\frac{\sum_{\ell=1}^{|z|} \mu_\ell \log p_\theta(z_\ell | z_{<\ell})}{\sum_{\ell=1}^{|z|} \mu_\ell}\right]$，其中 $\mu_\ell=1$ 仅当 token 是受监督助理目标。
- 训练配置：$L_{\max}=49{,}152$ tokens，子代理:编排器采样比 2:1，LoRA rank 64，peak LR $3\times10^{-5}$。

---

## 实验与结果
**实验设置**：
- 基座模型：Qwen3.6-27B-FP8，编码代理 harness：qwen-code 0.20.0
- 8 个合成训练任务（金融/机器人/药学各若干）、3 个测试合成任务
- 6 个公开基准：ScienceAgentBench、MLR-Bench、AstaBench E2E-Bench-Hard、DSBench、Terminal-Bench 2.0、SWE-bench Verified
- 外部强模型评估：GPT-5.5 + Claude Opus 4.8（最大推理 effort），每对工作空间 6 次判定

**主要结果**：
- 公共研究基准上每输出 token 得分（Figure 2）：
  - ScienceAgentBench：28.8 → 31.7（提升约 10%）
  - MLR-Bench：1.58 → 2.40（提升约 52%）
  - AstaBench：0.053 → 0.067
  - DSBench：0.473 → 0.482
  - **综合提升 1.2–1.6×**
- 三个测试合成任务胜率：$L^{(1)}$ 对 $L^{(0)}$ 达 53.9%，$L^{(2)}$ 对 $L^{(0)}$ 达 69.9%，连续迭代增益递增（$L^{(2)}$ 对 $L^{(1)}$ 达 60.8%）
- 自评估器与外部强模型一致性：0.73 → 0.91 → 0.93
- Terminal-Bench 2.0 和 SWE-bench Verified 上性能基本持平或略降（非核心研究基准）

**多代理轨迹效率优势（Hypothesis 2）**：
- 多代理学生 $M^+$（119 轨迹，28.82M tokens）对基座胜率 68.3%
- 单代理学生 $S^{++}$（290 轨迹，42M tokens）胜率仅 64.0%
- 多代理学生在更少的 token 预算下领先 4.3 个百分点

**工作流信息流分析（Hypothesis 1）**：
- 优化过程中角色和指令的任务特异性下降，而输出契约和调用顺序的任务特异性上升
- 组件间冗余度（总相关）普遍下降，支持信息流优化假设

---

## 相关工作脉络
- **Self-Rewarding LMs（Yuan et al., 2024）**：基于自身评分的迭代偏好训练，建立生成与评估的联合改进，但无工作流搜索机制。
- **Multi-Agent Evolve（Chen et al., 2025a）**：共享骨干通过 RL 学习问题生成、求解和评判，但提议者生成问题是而非执行工作流。
- **ADAS / AFlow / GPTSwarm（Hu et al., 2025; Zhang et al., 2025; Zhuge et al., 2024）**：自动搜索代理程序/工作流图/通信边，但未联合学习共享权重。
- **RHI（Lee et al., 2026）**：基于成对工件比较修订工作流提示，本文直接沿用其工作流搜索方法与信息流动机，并新增轨迹级微调。
- **TTHE（Nie et al., 2026）**：同一冻结 LLM 承担执行、harness 提议与评判的 homogenous 角色，但权重固定不更新。
- **Sirius（Zhao et al., 2026）**：通过 bootstrapped reasoning 训练多代理交互，本文进一步验证了多代理轨迹对训练效率的增益。

---

## 局限性与未来方向
- **加速每轮性能提升**：当前 RSI 循环的速度仍有提升空间，尤其第二周期的边际收益递减。
- **工作流搜索的不稳定性**：消融显示强优化迭代后可出现明显的性能回退（Table 11 示例），文本编辑幅度与质量变化相关性弱（$|\rho| \le 0.12$）。
- **外部评估依赖**：训练过程依赖内循环自评估，但外部验证使用更强模型，存在评估鸿沟。
- **公共基准泛化**：Terminal-Bench 2.0 和 SWE-bench Verified 上未显著提升，说明方法对特定类型任务的适配性需进一步探索。
- **未来方向**：加速 RSI 循环、探索混合评估策略（强模型辅助）、扩展到更多领域（如数学证明、实验设计）、深化内化机制理解。

---

## 研究启发与可借鉴点
1. **多代理工作流作为自我监督的信号源**：将"信息流结构优化"而非"推理链长度"作为 RSI 的监督信号，为开放任务自我改进提供了新视角；可迁移至科学发现、代码生成等长周期任务。
2. **编排-执行分离的 SFT 设计**：将编排器轨迹与子代理轨迹按 2:1 比例联合训练，并通过 loss mask 区分助理 token，这一设计可直接复用于其他 multi-agent SFT 场景。
3. **能力迁移的实验验证范式**：仅对任务解决轨迹微调即证明工作流优化和评估能力同步提升，这种"跨角色能力测量"的实验设计值得借鉴。
4. **工作流信息分析的量化指标**：使用互信息和总相关度量工作流组件的任务特异性和冗余度，为分析 agent 协作演化提供了可复用的定量工具。
5. **结合本团队方向的创新机会**：可将 MASS 的思想应用于团队现有的模型自适应训练流程——用多代理工作流搜索替代手动 prompt engineering，再蒸馏为 SFT 数据，形成半自动的持续改进闭环。

---

## 关键术语表
**Recursive Self-Improvement (RSI)**：模型自身作为优化器不断改进自身的框架，核心依赖评估与改进两个组件。
**Multi-Agent Self-Supervision (MASS)**：本文提出的 RSI 方法，通过多代理工作流优化与 SFT 交替实现递归改进。
**Homogeneous RSI Loop**：执行器、评估器、优化器共享同一模型权重的 RSI 循环，易强化而非纠正错误。
**Workflow**：编排器-子代理架构的文本规范，包含角色、指令、输出契约和调用顺序，附加到任务提示指导执行。
**Output Contract**：子代理向编排器返回的信息规格，控制代理间信息流动的关键组件。
**Orchestrator-Subagent Trajectory**：多代理执行产生的完整对话轨迹，包含编排器的委托决策和各子代理的执行结果。
**Bradley–Terry Model**：将成对比较结果转化为分数ranking的统计模型，用于多代理工作空间的质量评估。
**Internalization**：通过 SFT 将多代理协调行为（如委托、代码交接）内化到共享模型权重的过程。

---

## 可复现要素
- **数据集**：合成研究任务（8 train + 3 test，来自 Lee et al., 2026 的私有 benchmark，含金融/机器人/药学各域）；公开基准包括 ScienceAgentBench、MLR-Bench、AstaBench、DSBench、Terminal-Bench 2.0、SWE-bench Verified。
- **代码/权重**：论文未明确声明开源，但使用了 qwen-code 0.20.0（Apache 2.0 许可）和 Qwen3.6-27B-FP8 基座模型。
- **关键超参**：LoRA rank 64，scaling 128，dropout 0.05；AdamW $\beta_1=0.9, \beta_2=0.95$，peak LR $3\times10^{-5}$，warmup 5 steps，cosine decay to 10%；$L_{\max}=49{,}152$ tokens，子代理:编排器采样比 2:1，1,624 optimizer steps，3 windows/step。
- **计算资源**：6× NVIDIA H100 GPU，第一周期 SFT 约 19 小时。

---
