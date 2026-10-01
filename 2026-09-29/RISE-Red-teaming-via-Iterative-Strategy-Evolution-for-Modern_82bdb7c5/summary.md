---
title: "RISE-Red-teaming-via-Iterative-Strategy-Evolution-for-Modern"
source: https://arxiv.org/pdf/2609.34920v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 03:18:16"
field: "AI安全与红队测试"
keywords: ["T2I red-teaming", "prompt evolution", "adversarial attacks", "judge calibration", "black-box optimization", "safety evaluation", "text-to-image models"]
innovations: ["策略级进化搜索替代单提示改写以突破种子依赖", "严格的类别特定标准+VLM校准解决reward hacking", "在严格校准评估下揭示现有方法ASR被高估"]
benchmarks: ["DALL·E 3", "Nano Banana 2", "GPT-Image-2", "FLUX.2-klein-9B local testbed", "Leonardo.ai"]
---

# 论文速读：RISE-Red-teaming-via-Iterative-Strategy-Evolution-for-Modern

## 一句话总结
本文针对现代商业文本到图像（T2I）系统的安全红队测试问题，提出RISE方法通过迭代演化可复用的自然语言攻击策略而非直接改写单条提示，在严格校准的评估下实现了对DALL·E 3、Nano Banana 2、GPT-Image-2等系统的攻击；同时揭示现有方法的高ASR数字多由judge校准误差导致，重评后降至近零。

## 研究问题与动机
1. **评估不可靠**：现有T2I红队工作依赖的judge（NudeNet、CLIP-NSFW、InternVL2）在模糊不安全内容目标下表现不佳——要么错过真正规避guardrails的违规图像，要么错误地奖励良性边界图像；当这些judge驱动强化学习奖励时，优化器会reward-hack它们。
2. **探索能力不足**：现有基于提示改写的方法在更严格的guardrail设置下仍受限于种子提示，无法泛化；即使更换为校准后的judge，训练方法的迁移能力依然很弱。
3. **冷启动困难**：正样本过于稀疏，现有训练流程无法从头引导探索；FGPI等seed-free设置在严格阈值下ASR降至0%。
4. **种子依赖严重**：RPG-RT等方法虽能无需初始成功开始训练，但其学到的攻击仍主要重组合成人种子提示片段，成功率高估源于种子本身质量而非真正的学习能力。

## 核心贡献（创新点）
1. **严格的类别特定成功标准与VLM judge校准**：针对不同类别编写严格的违规成功标准，用人工标注校准强VLM judge；与已有工作的本质区别在于，之前工作依赖弱分类器（如NudeNet）驱动优化，而本文关闭了reward hacking，确保评估信号真实反映政策违规。
2. **RISE策略演化框架**：演化可复用的自然语言攻击策略（prompt-generation program）而非逐个改写提示；与已有prompt-level rewriting方法的本质区别在于搜索对象从"单条提示"变为"生成提示的策略"，利用策略层面的变异放大多样性。
3. **双阶段设计（演化+固定策略利用）**：Phase 1通过并行多岛进化搜索发现高质量策略，Phase 2冻结最佳策略并用UCB1分配预算进行利用；本质区别在于此前方法将优化局限在prompt级别，无法跨越seed locality障碍。
4. **重新评估现有方法的可靠性**：在相同校准评估下，此前报告DALL·E 3上ASR高达约30%的方法降至近零；这一系统性评估揭示了文献中高ASR数字多源于judge校准误差而非真实漏洞。

## 方法详解
**问题设定**：目标T2I系统$T$、冻结的违规标准$q$、校准后的judge $J_q$、文本提示$p$。对每个评估提示，RISE进行$k=3$次随机目标调用，评分$s_q(p) = \max_{j \in [k]} J_q(x_j)$，成功需至少一次图像得分$\geq t$。策略$\pi$是一个自然语言程序，将标准$q$和具体场景$c$映射为候选提示$p \sim \pi(q, c)$。

**核心配置RISE-Ablit**：使用公开的abliterated Qwen3-32B作为策略合成器和提示生成器，无模型权重更新。

**Phase 1：进化策略发现**
- 使用来自ShinkaEvolve的多岛进化搜索（K=16个岛屿并行）
- 每个岛屿通过单次变异seed strategy初始化，评估所有可用场景（通常3-5个）
- 父节点采样权重：$w(\pi) = \sigma(\lambda(F(\pi) - \text{median}(F))) \cdot \frac{1}{1 + n_\pi}$，其中$\lambda=10$，$n_\pi$为被选为父节点的次数， favor above-median strategies同时避免过度使用的parent
- 评估分两步：筛查步骤（每场景生成1个提示，每提示查询3次）若低于parent >0.05则停止；否则运行完整评估加入岛屿档案（上限50策略）
- Fitness取top-3提示级最大得分的均值（偏置向罕见突破）
- 每10岛步进行一次岛屿间迁移（每岛屿除最优外迁移一个策略）
- 共享meta-scratchpad每7岛步更新一次，总结成功/失败模式并提出最多5条建议

**Phase 2：固定策略利用**
- 冻结进化阶段的最佳5个策略
- UCB1 bandit分配目标调用预算
- 反馈驱动的场景生成：每个固定策略维护高分/低分场景记忆，每5次策略评估刷新3个场景

## 实验与结果
**目标系统**：DALL·E 3、Nano Banana 2、GPT-Image-2 (auto/low)、Leonardo.ai；本地测试床用FLUX.2-klein-9B + OpenAI严格审核（τ=0.05）。

**评估数据集**：约1,300张生成图像的annotated数据集用于judge校准；nudity类别的Fleiss' κ=0.46-0.57，其他类别更低。

**主要结果（Table 7）**：
- DALL·E 3：evolution阶段ASR@0.7 = 11.84%，Human ASR = 5.16%；fixed exploitation阶段ASR@0.7 = 24.44%，Human ASR = 13.33%
- Nano Banana 2：fixed exploitation Human ASR = 9.47%
- GPT-Image-2 (auto)：fixed exploitation Human ASR = 3.27%
- Leonardo.ai：Human ASR高达40.1%

**最强结果**：在DALL·E 3 fixed exploitation阶段达到Human ASR 13.33%，ASR@0.7达24.44%。

**基线对比**：在相同校准评估下，FGPI报告85.71% ASR降至6.7%（seed-free local），RPG-RT报告31.33% ASR降至0.3%；SneakyPrompt、PGJ、MACPrompt在DALL·E 3上均达0% human-verified ASR。

**非色情类别结果（Table 8）**：在graphic violence、self-injury、hate、suicide等类别上，使用precision-adjusted ASR或human-verified ASR报告了非零结果。

## 相关工作脉络
1. **FGPI (Xu et al., 2025, ICCV)**：基于反馈引导的prompt迭代，报告DALL·E 3高ASR；本文揭示其native-judge成功不 survive calibrated evaluation，在严格阈值下冷启动为0%。
2. **RPG-RT (Cao et al., 2025, NeurIPS)**：基于rule-based preference modeling的red-teaming；本文显示其非零校准成功仍与seed pool紧密绑定，输出的bigram多样性远低于RISE。
3. **Curiosity-driven RT (Hong et al., 2024)**：强化学习驱动的红队；本地宽松设置下提升明显，但向严格阈值迁移时崩溃。
4. **DREAM (Li et al., 2025)**：通过distribution modeling扩展red-teaming；其释放的prompts在DALL·E 3上有0.8% bypass率，但在GPT-Image-2上零迁移。
5. **JANUS (Zheng et al., 2026)**：分布优化框架；本地环境有效但所有seed池在严格设置和DALL·E 3上失败。
6. **SneakyPrompt / PGJ / MACPrompt**：传统prompt rewriting方法；在本地模型复现成功但在DALL·E 3校准评估下全为0%。

## 局限性与未来方向
1. **主要比较限于女性裸露类别**：因类别一致性高且可校准性较好；其他四类（暴力、自残、仇恨、自杀）标准更上下文依赖，人类标注者一致性较低。
2. **基线复现尽力而为**：部分baseline依赖 curated seed pools或private training data，无法保证完全复现原文性能。
3. **策略可转移性有限**：Table 19显示策略在不同prompt writer间有一定transfer但效果差异大。
4. **未来方向**：可扩展至更多安全类别的严格定义；探索prompt-writer的微调（offline GRPO可在内部模型上提升效率但不具通用性）。

## 研究启发与可借鉴点
1. **Strategy-level进化搜索**：将优化对象从单条prompt升维到"生成prompt的策略程序"，利用策略变异的多样性放大效应，值得借鉴到prompt engineering、agent design等需要大规模生成的任务中。
2. **Judge校准的重要性**：本文系统性地展示了weak judge如何导致high false positive/false negative并引发reward hacking；在设计任何依赖自动评分的optimization pipeline时，必须先用人工标注校准评估信号。
3. **Top-k fitness aggregation**：用top-3而非全部均值来评估strategy fitness，能有效应对sparse feedback regime中的noise，这一设计可迁移到其他黑盒优化问题。
4. **Meta-scratchpad全局洞察**：跨岛屿共享全局成功/失败模式总结，避免多岛陷入不同局部最优，这一机制可用于分布式优化或并行搜索框架。
5. **双阶段 Exploration-Exploitation分离**：清晰的Phase 1演化发现 + Phase 2固定策略利用的切换，配合UCB1预算分配，提供了可在budget-constrained black-box optimization中复用的范式。

## 关键术语表
**ASR (Attack Success Rate)**：攻击成功率，成功提示数占总评估提示数的比例，本文中每个提示用3次目标调用评估，任一图像得分≥阈值即算成功。
**Abliterated模型**：移除安全对齐机制但仍保留基础能力的开源模型版本，本文使用Qwen3-32B-abliterated作为prompt writer。
**ShinkaEvolve**：多岛进化搜索框架，本文扩展其到更重探索的T2I红队场景（K=16岛屿）。
**Precision-adjusted ASR**：针对非色情类别，用human-标注的calibration样本估计judge precision后对raw ASR进行校正，而非全量人工审核。
**Calibrated judge**：用类别特定success criteria和人工标注校准后的VLM评分器（如Gemini 2.5 Flash），替代不可靠的弱分类器。
**Seed locality**：攻击输出与原始人类手写种子提示的相似度，RISE的输出更远离种子（cosine similarity 0.54-0.64 vs RPG-RT的0.81）。
**Reward hacking**：优化器利用judge的漏洞获得高评分但未真正违反安全政策的现象。
**UCB1 bandit**：Upper Confidence Bound算法，用于Phase 2在多个冻结策略间分配目标调用预算。

## 可复现要素
- **数据集**：约1,300张annotated图像用于judge校准；内部标注未公开但方法论在Section B描述。
- **代码**：论文声明将在publication时发布reproduction pipeline、本地测试床和judge criteria（nudity、graphic violence、self-injury）至https://github.com/whitecircle/rise；hate和suicide标准因滥用风险未发布；发现的策略和攻击提示不发布。
- **关键超参**：16个岛屿、archive size 50、λ=10、top-3 fitness、每提示3次目标调用、evolution预算2,500 target calls、exploitation预算1,500 target calls、迁移每10岛步、meta-scratchpad每7岛步更新。
- **模型**：Qwen3-32B-abliterated（public）、Gemini 2.5 Flash（judge）。
