---
title: "RECURSIVE-SELF-IMPROVEMENT-THROUGH-MULTI-AGENT-SELF-SUPERVIS"
source: https://arxiv.org/pdf/2610.12176v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-01 07:19:01"
field: "大语言模型自主改进与多智能体系统"
keywords: ["Recursive Self-Improvement", "Multi-Agent Systems", "Self-Supervision", "Workflow Optimization", "Large Language Models", "Agent Coordination"]
innovations: ["提出MASS框架，通过多智能体工作流进化搜索+ SFT蒸馏实现递归自我改进", "证明多智能体轨迹训练效率高于单智能体（同token预算下表现提升4.3pp）", "揭示工作流优化是同质RSI回路的瓶颈而非评估能力，并发现SFT可泛化至优化/评估角色"]
benchmarks: ["ScienceAgentBench", "MLR-Bench", "AstaBench E2E-Bench-Hard", "DSBench", "Terminal-Bench 2.0", "SWE-bench Verified"]
---

# 论文速读：RECURSIVE-SELF-IMPROVEMENT-THROUGH-MULTI-AGENT-SELF-SUPERVIS

## 一句话总结
论文提出 Multi-Agent Self-Supervision（MASS），通过让同一基础模型在多个角色（执行者/评估者/优化者）中以多智能体工作流形式协作，迭代优化编排结构并蒸馏为共享权重，从而突破单模型自我改进的"同质化循环"瓶颈，实现递归自我改进（RSI）。

## 研究问题与动机
1. **开放研究任务的监督瓶颈**：前沿 AI 系统在非可验证任务（如开放性科研）上，产出已超出人类专家可靠评估的能力范围，模型自身成为最优优化器与评估器。
2. **同质化 RSI 回路的缺陷**：生成、评估、修订共享同一先验，导致自反馈倾向于强化而非修正错误，单模型难以自我批判复杂推理。
3. **人工设计工作流的成本过高**：跨任务和领域手动调整任务分解、工具使用与验证步骤的工作量巨大，难以规模化。
4. **缺乏高效训练信号**：仅靠单智能体长轨迹进行 SFT，学习效率有限，需探索更密集的监督信号形式。

## 核心贡献（创新点）
1. **提出 MASS 框架**：交替执行多智能体工作流优化与 SFT 蒸馏，以多智能体协调打破同质 RSI 回路的局限性。与单模型 Self-Refine 等方法的本质区别在于引入"多个副本模型协作"来提供外部视角的监督信号。
2. **将多智能体编排编码为文本工作流并进化搜索**：用结构化 prompt（包括角色、指令、输出契约、调用顺序）定义 orchestrator-subagent 架构，通过基于 LLM 的自我评估进行精英搜索（elite size 1）。与 ADAS/AFlow 等自动设计 agent 程序的方法本质不同，本文聚焦于**信息流结构优化**而非单纯角色增加。
3. **证明多智能体轨迹比单智能体轨迹训练更高效**：在相同 token 预算下，基于多智能体轨迹训练的 student 表现优于单智能体 baseline（1.4×更多训练 token 仍不及），且 SFT 仅对任务求解轨迹进行训练却能同时提升三个角色能力（求解/优化/评估）。
4. **揭示工作流优化是 RSI 瓶颈而非评估能力**：消融实验表明，更强的 optimizer 能显著加速成功工作流的发现，而单纯更换 evaluator 效果有限。

## 方法详解
MASS 采用双层递归循环（Algorithm 1）：

**内层循环——多智能体工作流优化（Algorithm 2）**：
- 工作流定义为 $w_t = \{\mathbf{h}_t, (\mathbf{r}_{t,j}, \mathbf{u}_{t,j}, \mathbf{c}_{t,j})_{j=1}^{m_t}\}$，其中 $\mathbf{h}_t$ 为 subagent 调用顺序，$\mathbf{r}_{t,j}/\mathbf{u}_{t,j}/\mathbf{c}_{t,j}$ 分别为角色、指令和输出契约。
- 通过执行 $\mathcal{A}(x_t \oplus w_t; \mathcal{L}_{\mathrm{agent}})$ 生成轨迹与工作空间，由自评估器 $\mathcal{L}_{\mathrm{eval}}$ 与当前最优工作流输出做成对比较，产生反馈 $d_{t,i}$。
- 优化器 $\mathcal{L}_{\mathrm{opt}}$ 在优化 prompt $x_{\mathrm{opt}}$ 中**强调通信结构** $(\mathbf{h}_t, \{\mathbf{c}_{t,j}\})$ 而非角色/指令，提出下一轮工作流。
- 保留策略为精英大小为 1 的进化过程：若当前输出优于历史最优则更新 $w_t^*$。

**外层循环——工作流内部化与 SFT（Algorithm 3）**：
- 对每个训练任务，用最优工作流 $w_t^*$ 独立运行 $M$ 次生成候选轨迹。
- 使用 Bradley-Terry 模型将成对评估转化为工作空间评分，选取 Top $K-1$ 轨迹作为 $\mathcal{D}_{\mathrm{train}}$，第 $K$ 个作为 $\mathcal{D}_{\mathrm{val}}$。
- 将轨迹分段（$L_{\mathrm{max}}=49152$ tokens，前置上下文可保留但 mask 掉），构造 SFT 窗口集 $\mathcal{D}$。
- SFT 目标函数为 $\ell_{\mathrm{SFT}}(\theta) = -\mathbb{E}_{z\sim q}\left[\frac{\sum_{\ell}\mu_\ell \log p_\theta(z_\ell|z_{<\ell})}{\sum_\ell \mu_\ell}\right]$，其中 $\mu_\ell=1$ 仅对 assistant target tokens。
- 采样策略固定为 2:1 的 subagent-to-orchestrator 比例。更新后的模型 $\mathcal{L}^{(g+1)}$ 进入下一轮 RSI。

## 实验与结果
- **基座模型**：Qwen3.6-27B-FP8，agent scaffold 为 qwen-code 0.20.0。
- **训练数据**：8 个合成研究任务（金融/机器人/药学各若干），共 119 条教师轨迹，2,435 个训练窗口，17.2M supervised tokens。
- **测试任务**：3 个合成测试任务 + 6 个公开基准（ScienceAgentBench、MLR-Bench、AstaBench、DSBench、Terminal-Bench 2.0、SWE-bench Verified）。
- **主要结果**：
  - 两轮 RSI 后，$\mathcal{L}^{(2)}$ 在 4 个研究基准上的 score per output token 达到基础的 **1.2–1.6×**。
  - 测试任务胜率从 $\mathcal{L}^{(1)}$ 的 53.9% 提升至 $\mathcal{L}^{(2)}$ 的 **69.9%**。
  - 评估器与外部强模型的 agreement 从 $\mathcal{L}^{(0)}$ 的 73% 提升至 $\mathcal{L}^{(2)}$ 的 **93%**。
  - ScienceAgentBench 从 28.8% → 31.7%，MLR-Bench 从 1.58 → 2.40，AstaBench 从 0.053 → 0.067。
- **最强结果**：两轮 MASS 后，$\mathcal{L}^{(2)}$ 在研究基准上获得 1.6× 的 token 效率提升（AstaBench）。
- **多智能体效率优势**：$M^+$（28.82M tokens，68.3% win rate）vs $S^{++}$（42M tokens，64.0%），前者少用 31.4% token 却高出 4.3 个百分点。

## 相关工作脉络
1. **Self-Rewarding LMs（Yuan et al., 2024）**：基于模型自评分的迭代偏好训练，MASS 在其基础上引入工作流搜索与多智能体轨迹蒸馏。
2. **RHI（Lee et al., 2026）**：提供本文工作流搜索方法与信息流假设的动机，MASS 在此基础上增加 SFT 环节实现权重更新。
3. **TTHE（Nie et al., 2026）**：同构回路设定（单模型承担执行/提议/评判），但 TTHE 固定权重只做 prompt 层编辑，MASS 通过 SFT 实现持续权重进化。
4. **SiriuS（Zhao et al., 2026）**：在多智能体轨迹上训练并精炼失败轨迹，MASS 与之不同在于聚焦于**工作流编排结构的显式搜索**而非交互精炼。
5. **ADAS/AFlow/GPTSwarm（Hu et al., 2025; Zhang et al., 2025; Zhuge et al., 2024）**：自动设计 agent 程序/图结构，MASS 的独特定位是从优化结果中蒸馏共享权重实现 RSI。
6. **SIA（Hebbar et al., 2026）**：使用独立反馈模型编辑 scaffold 并触发权重更新，MASS 采用完全同构设定（无外部反馈模型）。

## 局限性与未来方向
1. **工作流发现的稳定性不足**：强迭代后的sharp性能回归频繁，文本编辑大小与性能变化几乎无关（$|\rho| \le 0.12$），难以预测改进方向。
2. **RSI 循环加速仍是挑战**：论文明确指出加速每轮性能提升速度是重要未来方向。
3. **工作流搜索排除部分任务**：8/9 训练任务成功找到优化工作流，1 个任务（Task 221）被排除，说明搜索并非始终可靠。
4. **评估依赖外部强模型**：最终评估使用 GPT-5.5 和 Claude Opus 4.8，与回路内自评估存在 gap（尽管内部 agreement 已达 93%）。
5. **未探索更多轮次**：实验仅完成 2 轮 MASS，循环深度对性能的影响未知。

## 研究启发与可借鉴点
1. **编排结构作为可优化变量**：将 multi-agent 的通信结构（contract + hop）显式编码为文本并在 prompt 中优先优化，为工作流搜索提供了可操作的表示形式。
2. **训练数据仅覆盖执行角色即可泛化到优化/评估**：SFT 仅用任务求解轨迹，却同时提升了 optimizer 和 evaluator 能力，说明协调行为可通过 SFT 被内部化并泛化。
3. **多智能体轨迹的信息密度更高**：Jointly learning orchestration and bounded subagent execution 比单一长轨迹提供了更结构化的监督信号，值得在其他 RSI 设定中验证。
4. **信息流分析可作为工作流质量的诊断工具**：通过 mutual information 和 total correlation 度量 workflow 组件的任务特异性与冗余度，为自动化工作流评估提供了量化方法。
5. **同质回路设计的消融策略**：固定 executor 而仅替换 optimizer/evaluator 的消融设计，清晰分离了不同角色的贡献，是可复用的实验范式。

## 关键术语表
**Recursive Self-Improvement (RSI)**：模型通过自身生成的数据和评估持续迭代改进的框架。
**Multi-Agent Self-Supervision (MASS)**：本文提出的方法，通过多智能体协作生成自我监督信号并蒸馏为共享权重。
**Orchestrator-Subagent Workflow**：由编排者（orchestrator）和多个子代理（subagent）组成的多智能体执行结构，以文本形式嵌入 prompt。
**Output Contract**：子代理向编排者返回信息的结构化规范，控制 agent 间的信息流。
**Homogeneous RSI Loop**：同一个模型同时担任执行者、评估者和优化者角色的 RSI 设定。
**Bradley-Terry Model**：将成对比较结果转化为量化评分的统计模型，用于工作空间排名。
**Information Flow Hypothesis**：优化 agent 间通信结构（而非单纯增加 agent 数量）可提供有效自我监督的假设。
**Behavioral Internalization**：SFT 后模型在无显式工作流 prompt 下仍能自发展现协调行为（如 delegation、code handoff）的现象。

## 可复现要素
- **数据集**：12 个合成研究编程任务（金融/机器人/药学），9 个训练 + 3 个测试；公开基准包括 ScienceAgentBench、MLR-Bench、AstaBench、DSBench、Terminal-Bench 2.0、SWE-bench Verified。合成任务为私有 benchmark（由 GPT-5.5 生成）。
- **代码/权重**：论文未明确声明开源。
- **关键超参**：LoRA rank=64，scaling=128，dropout=0.05；AdamW $\beta_1=0.9, \beta_2=0.95$；peak LR=$3\times10^{-5}$；5 warmup steps，cosine decay to 10%；$L_{\mathrm{max}}=49152$ tokens；subagent-to-orchestrator 采样比 2:1；每任务 M=18 候选，取 Top K=16（实际 15）；1,624 optimizer steps，3 windows/step。
- **Compute**：6 × NVIDIA H100 GPUs，约 19 小时 SFT 运行时间。
- **外部评估模型**：GPT-5.5 和 Claude Opus 4.8，最大 reasoning effort，各 3 seeds。
