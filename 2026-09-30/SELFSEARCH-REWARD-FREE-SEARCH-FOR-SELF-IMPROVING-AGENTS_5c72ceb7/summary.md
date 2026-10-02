---
title: "SELFSEARCH-REWARD-FREE-SEARCH-FOR-SELF-IMPROVING-AGENTS"
source: https://arxiv.org/pdf/2609.37968v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 14:20:16"
field: "自改进智能体"
keywords: ["Self-Improving Agents", "Reward-Free Search", "LLM Agent", "Self-Modification", "Agent Harness"]
innovations: ["提出无下游评估的自我改进搜索方法 SelfSearch，以 episode 记录替代任务分数作为改进信号", "双谱系（capability/adaptive）并行改进并共享历史记录，实现跨谱系工具复用", "以仅$4.03搜索成本在Terminal-Bench 2.1达到82.0%成功率，匹敌公开SOTA harness"]
benchmarks: ["SWE-bench Verified", "SWE-bench Multilingual", "Terminal-Bench 2.1"]
---

# 论文速读：SelfSearch - Reward-Free Search for Self-Improving Agents

## 一句话总结
论文提出 SelfSearch，一种**无需下游任务评估**的代理自我改进搜索方法：代理利用自身历史改进记录（推理轨迹、工具调用、代码变更）指导多代改进，在 SWE-bench、Terminal-Bench 等基准上提升成功率并降低执行成本。

---

## 研究问题与动机

- **现有方法依赖下游评估**：当前 LLM 代理的自我改进方法需反复在开发集上评估候选代理以指导搜索，产生显著累积成本，且改进信号与特定任务分布绑定。
- **自我改进过程本身蕴含经验**：每次改进不仅产出候选代理，还产生包含"问题诊断、修改构建、效果验证"过程的经验记录，这些记录可能为后续改进提供独立于下游任务的指导信号。
- **元认知视角的延伸**：类比人类通过反思改进过程来优化学习方式，探究代理能否利用自我改进的经验变得更善于改进自身。
- **能力迁移假设**：自我修改过程中锻炼的代码审查、失败诊断、变更实现等能力，与下游任务执行所需能力重叠，因此离线获得的改进经验可泛化至下游场景。

---

## 核心贡献（创新点）

1. **无奖励自我改进搜索框架 SelfSearch**：代理仅凭历史 episode 记录（而非下游任务评测）自主迭代改进自身指令、工具与执行逻辑；与已有工作的本质区别在于将改进过程的"过程经验"（trajectory + 变更记录）作为搜索信号，而非候选代理的任务分数。

2. **双谱系共享经验机制**：维护 capability（能力扩展）与 adaptive（自适应推理）两个并行改进谱系，两谱系共享彼此的 episode 记录，支持跨谱系的工具复用与改进（如 trajectory reader 从一个谱系引入、另一谱系精化）。

3. **发现可复用的下游工具与效率提升**：自我改进过程中开发的工具（如行范围文件查看、文本搜索、精确替换）在下游任务中被高频复用；在 SWE-bench Multilingual 上，改进后的代理成功率提升 5.0pp，同时共享任务执行成本下降 38.5%。

4. **跨模型迁移与低成本高收益**：以 DeepSeek V4 Pro 搜索的 harness 以 DeepSeek V4 Flash 在 Terminal-Bench 2.1 上达到 82.0%，与公开对比中排名第一的 Codex 持平，而搜索成本仅 \$4.03。

---

## 方法详解

### 总体框架
SelfSearch 维护初始代理 $B_0$（一个可编辑的代码仓库，包含指令、工具、执行逻辑）和初始 episode 记录 $\mathcal{E}_0$。搜索进行 $K=10$ 代，每代有两个并行谱系（capability $c$ 与 adaptive $a$），每个谱系执行一次自我改进 episode：

$$
(B_{k+1}^w, e_k^w) = \text{SelfImprove}(B_k^w; \mathcal{E}_k, d_w)
$$

其中 $d_w$ 为搜索方向（capability/adaptive），$e_k^w$ 为 episode 记录。所有谱系的 episode 记录合并为 $\mathcal{E}_{k+1}$，供下一代使用。

### 改进 Episode 流程
1. 代理从只读挂载的 $\mathcal{E}_k$ 中读取前代记录，分析推理轨迹、工具调用与结果、代码变更。
2. 在可编辑副本 `/workspace` 中定位"证据支持的显著局限性"，设计并实施单次连贯修订。
3. 修改后可通过工具进行本地验证（不依赖下游评测）。
4. 运行时捕获完整交互轨迹（reasoning、tool actions、outcomes）与代码快照，生成 $e_k$ 存入记录。
5. 修改后的代理成为下一代 $B_{k+1}$ 的起点。

### 双搜索方向
- **Capability 方向**：聚焦任务执行效率不足、失败动作或操作困难，开发可复用工具/流程。
- **Adaptive 方向**：聚焦当证据与当前计划矛盾时的策略调整能力，提升容错与自适应推理。

### 搜索环境约束
- 代理整个仓库可编辑，但模型-工具交互循环运行于**固定运行时**（不可编辑），控制推理模型、耗时上限、调用限额等。
- 搜索阶段在隔离容器中执行，无网络访问；episode 记录以只读方式挂载。

---

## 实验与结果

### 实验设置
- **模型**：GPT 配置（gpt-5.6-sol 搜索 / gpt-5.6-luna 下游）；DeepSeek 配置（deepseek-v4-pro-0813 搜索 / deepseek-v4-flash-0731 下游）。
- **基准**：SWE-bench Verified（120 tasks）、SWE-bench Multilingual（60 tasks，8种语言）、Terminal-Bench 2.1（89 tasks）。
- **基线**：初始代理 $B_0$、线性搜索（linear search）、归档搜索（archive search）。
- **生成策略**：每配置运行 10 代双谱系搜索，保留所有 checkpoint，报告最终两谱系各自结果与均值。

### 主要结果

| 基准 | 初始 $B_0$ | 最佳改进 $B^c/B^a$ | 提升 |
|---|---|---|---|
| **Terminal-Bench 2.1**（DeepSeek） | 65.2% | 73.0% | **+7.8pp** |
| **Terminal-Bench 2.1**（GPT） | 43.8% | 55.1% | **+11.3pp** |
| **SWE-bench Multilingual**（DeepSeek） | 68.3% | 73.3% | **+5.0pp** |
| **SWE-bench Multilingual**（GPT） | 53.3% | 60.0% | **+6.7pp** |
| **SWE-bench Verified**（DeepSeek） | 81.7% | 86.7% | **+5.0pp** |
| **SWE-bench Verified**（GPT） | 75.0% | 77.5% | **+2.5pp** |

- **执行效率**：DeepSeek 双谱系在 Terminal-Bench 2.1 上成功率提升的同时，平均单任务成本分别下降 12.1% / 16.9%；SWE-bench Multilingual 上共享成功任务的成本下降最高达 **38.5%**。
- **对比评估导向搜索**：SelfSearch 在 SWE-bench Verified 和 Multilingual 上匹配或超越线性/归档搜索基线，搜索成本更低（DeepSeek 配置仅 \$4.03 vs. 基线 \$7.90-\$8.59）。
- **跨模型迁移**：不同搜索模型产生的 harness 在另一模型族执行时均优于初始代理（DeepSeek harness 在 GPT Luna 上 +4.2pp，GPT harness 在 DeepSeek Flash 上 +3.3pp）。
- **与其他 harness 对比**：以 \$4.03 搜索成本，DeepSeek 配置的 capability agent 在 Terminal-Bench 2.1 达 82.0%，与最高分的 Codex 持平。

---

## 相关工作脉络

1. **Evaluation-guided agent search**（Hu et al., 2025; Zhang et al., 2026a）：通过任务评测分数指导代理变更搜索；SelfSearch 不使用下游评测，仅凭自我改进过程记录作为信号。
2. **Self-modifying agents**（Robeyns et al., 2025; Zelikman et al., 2024）：研究代理修改自身的行为；区别在于 SelfSearch 明确将 episode 记录作为跨代传递的经验载体，实现"经验的积累式利用"。
3. **Darwin Gödel Machine (DGM)**（Zhang et al., 2026a）：本文基线代理的来源；DGM 使用下游评测指导搜索，SelfSearch 在此基础上移除评测环节，改用 episode 记录。
4. **Hyperagents**（Zhang et al., 2026b）：区分 task agent 与 meta-agent，SelfSearch 则统一于同一代理，实现更简洁的角色分工。
5. **Group-evolving agents**（Weng et al., 2026）：多代理经验分享；SelfSearch 的共享机制局限于两个谱系间，但通过 episode 记录而非实时通信。
6. **Meta-Harness**（Lee et al., 2026）：端到端优化 harness；SelfSearch 更轻量，无需额外 meta-agent 架构。

---

## 局限性与未来方向

- **实验规模有限**：仅在 2 个模型配置 × 3 个基准上评估，独立搜索的重复性尚未系统验证。
- **无下游评测的潜在天花板**：完全脱离任务信号可能限制对特定任务分布的进一步优化空间。
- **仅做离线改进**：搜索过程中无法利用在线用户反馈或实时错误信号。
- **未来方向**：（1）将 SelfSearch 嵌入评估导向搜索中，作为单次评估前的多代预改进步骤；（2）探索更多模型与更大规模搜索的配置；（3）研究 episode 记录压缩/摘要机制以处理更长的历史记录。

---

## 研究启发与可借鉴点

1. **过程经验作为搜索信号**：将 agent 改进过程中的 trajectory + 变更记录作为独立于任务分数的指导信号，为"无奖励搜索"提供了可行范式，可迁移至其他自我优化场景（如代码格式化器、prompt 模板自动生成器）。

2. **双谱系互补机制**：capability 与 adaptive 两个方向的并行设计，有效促进工具跨谱系传播（如 trajectory reader 在 GPT 配置第 7 代跨谱系复用），启发了多目标/多视角并行改进的结构设计。

3. **运行时与可编辑仓库的解耦**：将推理循环固定在不可编辑的 runtime 中，代理仅修改工具和指令层，这一架构保障了搜索过程的稳定性，可作为后续工作的基础设施参考。

4. **低搜索成本的高效发现**：\$4.03 的搜索成本即达到 SOTA harness 水平，证明了经验驱动搜索的性价比，为资源受限团队提供了可行的 self-improvement 路径。

---

## 关键术语表

- **SelfSearch**：一种无需下游任务评估的代理自我改进搜索方法，通过历史 episode 记录指导多代改进。
- **Episode Record（改进记录）**：单次改进 episode 的完整轨迹，包含推理过程、工具调用、代码变更与本地验证结果。
- **Capability Lineage（能力谱系）**：搜索方向之一，聚焦识别并修复任务执行中的能力缺陷，开发可复用工具。
- **Adaptive Lineage（自适应谱系）**：搜索方向之二，聚焦提升代理在面对矛盾证据或失败时的策略调整能力。
- **Population Mean（种群均值）**：两谱系最终代理在下游基准上成功率的算术平均，作为整体改进效果的指标。
- **Shared Success Cost Reduction（共享任务成本下降率）**：比较初代与改进代理在双方均能解决的任务上的平均执行成本变化，反映效率增益。
- **Harness（代理执行框架）**：指代理的代码实现与运行时配置的整体，通常用于描述一组可替换的 agent 实现。

---

## 可复现要素

- **数据集**：SWE-bench Verified（120 tasks）、SWE-bench Multilingual（60 tasks）、Terminal-Bench 2.1（89 tasks）；论文未声明公开状态。
- **代码/权重**：论文提供了执行框架与搜索提示模板（Appendix B），**但未声明开源仓库链接**；初始代理基于 DGM 发布实现修改。
- **关键超参**：搜索模型 gpt-5.6-sol / deepseek-v4-pro-0813，下游模型 gpt-5.6-luna / deepseek-v4-flash-0731；每 episode 上限 513 次模型调用、512 次工具调用、4 小时执行；输出上限 8192（GPT）/ 16384（DeepSeek）tokens；temperature=1。
- **搜索成本**：GPT 配置 \$6.52，DeepSeek 配置 \$4.03。

---
