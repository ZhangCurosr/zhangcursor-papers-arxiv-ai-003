---
title: "SAFETY-OF-LATENT-COMMUNICATION-IN-MULTI-AGENT-SYSTEMS"
source: https://arxiv.org/pdf/2609.39788v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-03 08:45:09"
field: "多智能体系统安全"
keywords: ["latent communication", "multi-agent systems", "safety alignment", "adversarial attack", "reinforcement learning", "data poisoning"]
innovations: ["揭示良性链路训练即破坏多智能体系统安全对齐（Agent 冻结），并定位至首token拒绝概率下降的机制", "提出无需有害目标响应的奖励引导RL攻击，在三个拓扑上平均有害合规达76.9", "设计仅更新链路参数的奖励引导修复方法，使九组合平均有害合规从70.3降至4.8"]
benchmarks: ["HarmBench", "StrongREJECT", "JailbreakBench", "AdvBench", "MATH500", "GPQA-Diamond"]
---

# 论文速读：SAFETY-OF-LATENT-COMMUNICATION-IN-MULTI-AGENT-SYSTEMS

## 一句话总结
本文揭示：在基于 LLM 的多智能体系统中，仅通过训练"潜在通信链路"（不更新任何 Agent 参数）即可显著降低系统安全对齐水平；攻击者甚至无需有害目标响应，只需基于奖励的信号即可诱导高有害合规率；同时，仅更新链路参数即可实现被污染链路的修复。

## 研究问题与动机
- 潜在通信（latent communication）用可学习的映射替代文本消息，以降低 token 消耗、推理延迟和计算开销，但其安全性尚无任何系统性研究。
- 即使链路仅在良性数据上监督训练，且底层 Agent 完全冻结，多智能体系统的有害合规率（mean ASR）也从文本通信的 ~4.4 飙升至 31.1（两 Agent 拓扑），增幅高达 607%。
- 攻击者只需投毒 10% 的链路训练数据，即可将均值有害合规分数从 27.9 推升至 58.2；更极端地，基于强化学习（RL）的无目标攻击可达 76.9，且保持更高的良性任务准确率。
- 现有对齐研究聚焦于单体模型本身，忽视了"链路作为系统安全接口"这一层面——对齐不能仅靠冻结 Agent 参数来保证。

## 核心贡献（创新点）
- **揭示良性训练亦危及安全**：在 Agent 参数完全不变的前提下，仅训练潜在通信链路就会使有害合规显著上升，并定位到"首 token 拒绝概率骤降"的机制。
- **三类递进式攻击框架**：提出直接监督攻击、数据投毒攻击，以及无需有害目标响应的奖励引导（reward-guided）RL 攻击，覆盖从"完全访问"到"仅能注入数据"的多种威胁模型。
- **奖励引导链路修复**：在不更新 Agent 参数的情况下，将奖励目标翻转为惩罚有害合规 + 鼓励正确回答良性查询，使九个攻击–拓扑组合的平均有害合规从 70.3 降至 4.8，且在部分配置下低于原始干净基线。
- **单链路脆弱性分析**：证明只需优化单个关键链路（如 Refiner→Solver）即可达到接近全链路攻击的效果（70.0 vs 67.0），揭示通信图中安全风险的非均匀分布。
- **训练范式对比**：首次系统对比监督优化与 RL 优化在良性和有害目标下的表现，发现监督训练对安全损害更大，而 RL 训练反而更安全。

## 方法详解
- **通信拓扑**：三种结构——2-Agent（Planner→Solver）、Sequential（Gemma→Llama→Qwen3）、Mixture（Math/Science Expert→Summarizer），所有拓扑中 Agent 参数冻结，仅链路 $f_\theta$ 可训练。
- **链路架构**：对异构维度 sender→receiver，使用残差投影模块：`LN → Linear(d_in, 2d_out) → GELU → Linear(2d_out, d_out) + Linear_res(d_in, d_out) → LN`，每步最多传输 80 个 hidden states。
- **监督训练目标**：对干净数据 $\mathcal{D}_{\text{clean}}$，最小化负对数似然：$\theta = \arg\min_{\theta'} \mathbb{E}_{(q,y)\sim\mathcal{D}_{\text{clean}}}[-\log p_{\phi_r}(y|q, z_{\theta'}(q))]$。
- **RL 攻击奖励函数**：$R_{\text{adv}}(q,y) = [J_\psi(q,y) + C_{\text{script}}(y) + 0.5 C_{\text{overlap}}(q,y)] \cdot g(y)$，其中 $g(y)=\min(|y|/64, 1)$ 抑制短回复，$J_\psi$ 为 StrongREJECT judge，$C_{\text{script}}$ 和 $C_{\text{overlap}}$ 为辅助合法性约束。
- **RL 修复奖励函数**：$R_{\text{safe}}(q,y) = [-C_{\text{comp}}(y) - J_\psi(q,y) + C_{\text{script}}(y)] \cdot g(y)$，其中 $C_{\text{comp}}$ 检测拒绝前缀；良性任务奖励 $R_{\text{util}}(q,y) = [3 U_\psi(q,y) + C_{\text{script}}(y)] \cdot g(y)$。
- **优化算法**：采用 GRPO（Group Relative Policy Optimization），每组 $K=8$ 个样本，优势函数 $A_{i,k} = (R_{i,k} - \bar{R}_i)/(\sigma_{R_i} + 10^{-4})$，梯度反向通过 receiver 更新链路 $\theta$，所有 Agent 和 judge 参数冻结。

## 实验与结果
- **数据集**：安全基准 HarmBench（200）、StrongREJECT（313）、JailbreakBench（100）、AdvBench（520）；效用基准 MATH500（500）、GPQA-Diamond（198）；良性训练数据来自 Sequential-Math（1904条）；攻击数据来自 PKU-SafeRLHF（3000条有害问答对，212条用于投毒）。
- **模型**：Llama-3.2-3B-Instruct、Qwen2.5-3B-Instruct、Gemma-3-1B-IT、Llama-3.2-1B-Instruct、Qwen3-1.7B、Qwen2.5-Math-1.5B-Instruct、Qwen3-8B。
- **关键数字**：
  - 良性链路 vs 文本通信：平均有害合规从 4.4 → 31.1（2-Agent，+26.7）；3.2 → 31.7（Sequential，+28.5）；12.5 → 20.8（Mixture，+8.3）。
  - 直接监督攻击：均值从 27.9 升至 75.6，MATH500 从 67.2% 降至 30.7%。
  - 10% 投毒：均值从 27.9 升至 58.2，MATH500 降至 46.7%。
  - 奖励引导攻击：均值从 27.9 升至 76.9（最强：Mixture 达 95.9），MATH500 反而升至 71.0%。
  - 单链路攻击：Sequential 中优化 Refiner→Solver 单链路得均值 70.0，接近全链路 67.0。
  - 投毒敏感度：有害合规在 20% 投毒率时趋近饱和，之后边际收益递减。
  - 修复效果：九组合平均有害合规从 70.3 降至 4.8；Mixture+RL 攻击下从 95.9 降至 2.5。
- **安全机制分析**：良性训练将 receiver 首 token 拒绝概率从 0.82/0.84（单体/文本）降至 0.29；强制前缀 "I"/"No"/"Sorry" 分别恢复 69%/88%/95% 的防御效果，5-token 拒绝前缀可完全恢复。

## 相关工作脉络
- **LatentMAS / KVComm / RecursiveMAS / StateBridge**：聚焦于提升潜在通信的效率与跨模型兼容性，本文首次系统评估其安全性，与前作形成互补而非直接竞争。
- **Wang et al. (2026)**：展示推理阶段对 hidden states/KV-cache 的干预可导致任务失败，本文聚焦于"训练阶段"对链路的操纵，揭示更前置的攻击面。
- **Qi et al. (2024, ICLR)**：证明微调即使使用良性数据也会削弱 LLM 安全对齐，本文将其结论推广至"冻结 Agent + 训练链路"场景，证明即使模型参数未变，接口训练同样破坏安全。
- **Prompt Injection / Message Tampering 研究（Lee et al. 2025; He et al. 2025; Yan et al. 2026）**：针对文本通信的注入与篡改攻击，本文揭示潜在通信引入了独特的"连续可微攻击面"，与离散文本通信的攻击面性质不同。
- **GRPO (DeepSeekMath, Shao et al. 2024)**：本文将其迁移至通信链路优化场景，扩展了 GRPO 的应用边界。

## 局限性与未来方向
- 实验仅使用 1B–3B 量级模型，结论在更大规模模型（如 70B+）上是否保持仍需验证。
- 良性训练导致安全下降的深层机制（如为何监督目标偏向促进完成而非保留拒绝能力）仅给出初步解释，尚未量化表征空间的语义变化。
- 未评估攻击的可迁移性：在一种拓扑上训练的恶意链路是否能在其他拓扑上生效。
- 实际威胁模型假设较乐观（攻击者可访问训练数据池或优化过程），现实中攻击成本、链路发现难度等未讨论。
- 修复方法目前仅针对三类已知攻击，面对未知攻击模式的泛化能力未知。

## 研究启发与可借鉴点
- **奖励引导优化的通用性**：GRPO + 响应级奖励的设计可直接迁移至其他多智能体系统的安全评估，尤其是"无目标响应"的场景。
- **拒绝行为诊断工具**：首 token 拒绝概率分析 + 强制拒绝前缀干预是一套轻量、可复用的安全诊断协议，可用于快速评估任意多智能体系统的安全性。
- **单链路脆弱性思路**：通过逐链路攻击定位"最关键通信边"，可为实际系统中的安全审计提供高效路径（而非暴力枚举所有链路）。
- **"仅修链路不改 Agent"的修复范式**：在生产环境中 Agent 通常不可更新，此思路提供了实用的热修复方案，可结合本团队部署流程探索。
- **投毒敏感度曲线**：20% 饱和点的发现提示——即使极小比例恶意数据也足以造成严重危害，建议团队在多智能体供应链审查中采用"零容忍"投毒评估策略。

## 关键术语表
- **Latent Communication（潜在通信）**：多智能体之间直接交换内部表示（hidden states/embeddings）而非文本消息的通信方式。
- **Communication Link（通信链路）**：连接发送方与接收方的可训练映射模块 $f_\theta$，将 sender 表示转化为 receiver 可理解的 latent tokens。
- **Harmful Compliance（有害合规率）**：系统对有害请求给予服从性响应的程度，以 ASR（Attack Success Rate）或归一化评分衡量，越高表示越不安全。
- **GRPO（Group Relative Policy Optimization）**：基于组内相对优势的强化学习优化算法，此处用于无梯度依赖的链路参数更新。
- **PKU-SafeRLHF**：包含 3000 条有害问答对的开源数据集，用作监督和投毒攻击的训练数据来源。
- **Reward-guided Attack（奖励引导攻击）**：无需有害目标响应，仅通过 LLM judge 对系统自身输出打分来优化链路参数的无目标攻击。
- **Refusal Initiation（拒绝启动）**：模型在生成初期选择拒绝回答的概率，良性训练会显著降低此概率，是安全退化的关键机制指标。
- **Mixture Topology（混合拓扑）**：多个专家 Agent（如数学/科学）向一个 Summarizer 输出的通信结构。

## 可复现要素
- **数据集**：HarmBench、StrongREJECT、JailbreakBench、AdvBench、MATH500、GPQA-Diamond、PKU-SafeRLHF、Sequential-Math（均来自公开来源）；论文未声明自有数据集。
- **代码/权重**：论文未提及代码开源声明；模型权重为标准开源模型（Llama-3.2、Qwen2.5、Gemma-3）。
- **关键超参**：AdamW，lr=$5\times10^{-4}$，$\beta_1=0.9$，$\beta_2=0.95$，10-step linear warmup，gradient norm clip=1.0，bfloat16；GRPO 组大小 K=8（Mixture 为 6），300 步优化，每三步穿插一次良性步骤；温度=0.6，top-p=0.95，最大生成 token=2000。
