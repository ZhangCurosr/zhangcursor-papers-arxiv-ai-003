---
title: "RISE-Red-teaming-via-Iterative-Strategy-Evolution-for-Modern"
source: https://arxiv.org/pdf/2609.34920v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 03:18:28"
field: "AI安全与红队评估"
keywords: ["T2I红队", "文本到图像安全", "对抗提示词", "进化搜索", "VLM裁判校准", "红队自动化"]
innovations: ["将优化对象从单提示词提升至可复用策略程序层级，通过多岛进化搜索发现通用攻击策略", "系统性校准图像判别器并揭示现有T2I红队方法在human-validated评估下ASR坍塌至零", "无需模型权重更新的零训练策略进化框架，在hardened生产级T2I系统上实现最高13%人类验证ASR"]
benchmarks: ["DALL·E 3", "Nano Banana 2", "GPT-Image-2", "Leonardo.ai", "FLUX.2-klein-9B本地测试床"]
---

# 论文速读：RISE-Red-teaming-via-Iterative-Strategy-Evolution-for-Modern

## 一句话总结
RISE 提出了一种针对现代商业化文本到图像（T2I）系统的红队评估框架，通过迭代进化可复用的自然语言攻击策略（而非逐提示词重写）来发现安全绕过方法；在 DALL·E 3、Nano Banana 2 和 GPT-Image-2 等 hardened 生产级目标上实现了最高 13% 的人类验证 ASR，而在相同校准评估下，先前报告高达约 30% ASR 的方法实际降至接近零。

## 研究问题与动机
1. **评估不可靠**：现有 T2I 红队工作广泛使用的图像判别器（如 NudeNet、CLIP-NSFW、InternVL2）在模糊的不安全内容目标下表现不可靠——要么对良性边缘图像过度触发，要么遗漏真正绕过防护的微妙违规；当这些弱判别器驱动强化学习奖励时，优化器会奖励黑客（reward hacking）。
2. **探索能力不足**：现有基于提示词修改/微调的训练型红队器在更严格的 guardrail 设置下无法泛化——它们严重依赖种子提示词质量，跨阈值迁移失败，且在稀疏正样本信号下无法启动训练。
3. **已发表成果被高估**：FGPI、RPG-RT 等方法报告了 DALL·E 3 上高达 85.71% 和 31.33% 的 ASR，但在经 VLM 裁判校准并对照人工标注后，这些数字坍塌至接近零。
4. **现代生产系统的防御显著增强**：DALL·E 3、GPT-Image-2、Nano Banana 2 等系统集成了多层防御（输入提示词分类器、输出图像审核、模型侧安全微调），远比开源扩散模型更难攻破，而现有方法仍主要在宽松本地设置（如 SDXL + SLD-strong）上评估。

## 核心贡献（创新点）
1. **系统性校准图像判别器**：首次对多种 T2I 红队常用判别器（NudeNet、CLIP-NSFW、InternVL2、Q16-CLIP）与 VLM 裁判（Gemini 2.5 Flash）进行人工标注校准，揭示前者在精确率/召回率上的严重缺陷，并提出严格类别特异性成功标准。
2. **揭示现有方法的系统性失效模式**：证明训练型红队器的成功主要依赖种子质量和训练时设置的宽松度，而非训练过程本身；reward-hacking、跨阈值迁移失败、冷启动探索弱是普遍问题。
3. **RISE 框架：策略级进化而非提示词级重写**：将优化对象从单个提示词提升到可复用的自然语言"策略程序"层面，利用多岛进化搜索（K=16 岛屿）发现能生成多样化对抗提示词的通用策略。
4. **无需模型训练的零参数修改红队能力**：主方法使用公开的 abliterated Qwen3-32B 作为提示词生成器，不进行任何模型权重更新；策略被发现后可跨场景、跨目标、跨提示词模型迁移复用。

## 方法详解
**问题形式化**：给定目标 T2I 系统 T、冻结违规标准 q、校准裁判 J_q 和文本提示词 p，对每个评估提示词执行 k=3 次随机目标调用，得到图像集合 {x_1,…,x_k}；提示词得分 s_q(p) = max_j J_q(x_j) ∈ [0,1]，ASR@t = (1/N) Σ_i 1[s_q(p_i) ≥ t]。策略 π 是将标准 q 和具体场景 c 映射到候选提示词的概率程序。

**两阶段流程**：
- **Phase 1（进化）**：采用多岛进化搜索（源自 ShinkaEvolve），K=16 个并行岛屿。每个岛屿以变异初始种子策略开始，在 2,500 次目标调用预算内迭代优化。父代采样权重为 w(π) = σ(λ(F(π)−median(F)))·1/(1+n_π)，其中 λ=10，n_π 为父代被选次数（ favoring above-median 同时惩罚过度使用的策略）。候选策略先经筛选步骤（每场景 1 提示词×3 次调用），若低于父代 >0.05 则终止；否则进入完整评估。适应度定义为 top-3 提示词最大分数的均值（偏向稀有突破）。每 10 步进行一次岛屿间迁移（每岛迁移一个非最优策略）。每 7 步更新共享元 scratchpad，汇总成功/失败策略并生成 ≤5 条改进建议。
- **Phase 2（固定策略利用）**：冻结进化阶段最好的 5 条策略，使用 UCB1 博弈分配 1,500 次目标调用预算，配合反馈驱动的动态场景生成（基于历史高分/低分场景刷新）。

**关键设计选择**：
- 使用 abliterated Qwen3-32B（安全机制被移除的公开模型）作为策略合成器和提示词生成器，无权重更新。
- 策略抽象层次是核心：进化的是"生成提示词的程序"而非直接进化提示词本身，使探索变异的多样性呈复合增长。
- 进化搜索中使用分层评估（先筛选后完整评估）以降低目标调用开销。

## 实验与结果
**评测目标**：DALL·E 3、Nano Banana 2、GPT-Image-2（auto/low）、Leonardo.ai；本地受控测试床使用 FLUX.2-klein-9B + OpenAI 输出审核（τ=0.05）。

**主结果（Table 7，女性裸体类别，人类验证 ASR）**：
| 目标 | 阶段 | 目标调用 | ASR@0.7 | Human ASR |
|------|------|---------|---------|-----------|
| DALL·E 3 | 进化 | 2,500 | 11.84% | 5.16% |
| DALL·E 3 | 固定利用 | 1,500 | 24.44% | **13.33%** |
| Nano Banana 2 | 进化 | 2,500 | 10.28% | 5.72% |
| Nano Banana 2 | 固定利用 | 1,500 | 15.80% | 9.47% |
| GPT-Image-2 (auto) | 进化 | 2,500 | 2.32% | 1.52% |
| GPT-Image-2 (auto) | 固定利用 | 1,500 | 5.06% | 3.27% |
| Leonardo.ai | 进化 | 2,500 | 46.10% | 40.1% |

**裁判校准结果（Table 1，Gemini 2.5 Flash 最优平衡）**：在 DALL·E 3 裸体类别上，Gemini P=0.67/R=0.70/F1=0.69，而 NudeNet P=0.67 但 R 仅 0.08，InternVL2 R=0.79 但 P 仅 0.35。

**基线重评结果**：FGPI、RPG-RT、Curiosity 等方法在原设下的裁判中显示高成功率（如 RPG-RT 报告 52% NudeNet ASR），但经 Gemini 校准重评后降至 0.3%；SneakyPrompt、PGJ、MACPrompt 在 DALL·E 3 上人类验证 ASR 均为 0%。

**非裸体类别（Table 8）**：在 graphic violence、self-injury、hate/dehumanizing、suicide 四类上 RISE 也发现人类确认违规，其中 Graphic violence on Nano Banana 2 达到 precision-adjusted ASR 20.9%。

## 相关工作脉络
1. **FGPI（Xu et al., 2025, ICCV）**：反馈引导的提示词迭代微调方法，报告 DALL·E 3 上 85.71% ASR；本文证明其 native-judge 成功不 Survive 校准，seed-free 条件下严格阈值 ASR 降为 0%。
2. **RPG-RT（Cao et al., 2025, NeurIPS）**：基于规则偏好建模的红队器，报告 31.33% DALL·E 3 ASR；本文发现其成功几乎完全依赖种子池质量，重新排列种子后 calibrated ASR 归零。
3. **DREAM（Li et al., 2025）**：分布建模方法，提供 1,024 条开源提示词；在 DALL·E 3 上仅 0.8% bypass，无法 transfer 到 GPT-Image-2。
4. **JANUS（Zheng et al., 2026, CVPR 2026）**：并发分布优化框架；在 SDXL 3.5 和宽松本地设置上有效，但在严格 τ=0.05 和 DALL·E 3 上失败。
5. **SneakyPrompt / PGJ / MACPrompt**：黑盒提示词扰动/重写方法；在本地模型（SLD-strong/SDXL）上可达 92-100% ASR，但在 DALL·E 3 上人类验证 ASR 均为 0%。
6. **LLM-guided 进化框架（ShinkaEvolve, AlphaEvolve）**：RISE 借鉴其多岛进化搜索范式，但将其从代码/算法优化迁移到 T2I 红队的稀疏反馈黑盒场景。

## 局限性与未来方向
1. **类别聚焦限制**：主比较集中于女性裸体（因跨系统一致性禁止且可人工标注），其他四类（暴力、自残、仇恨、自杀）的标注一致性较低（Fleiss' κ=0.24-0.38），结果以 precision-adjusted ASR 报告而非完全人工验证。
2. **基线复现不确定性**：部分基线依赖 curated seed pools 或私有训练数据，无法保证完整复现原始性能。
3. **策略与攻击提示词未公开**：出于双重用途风险考虑，发现的策略和最终攻击提示词均未发布。
4. **部分类别需要切换提示词生成器**：DALL·E 3 的 self-injury 和 graphic violence 类别因 Qwen3-32B 不够有效，需改用带 persona-based jailbreak wrapper 的 GPT-4.1。
5. **策略跨模型迁移有限**：Table 19 显示不同提示词模型间策略迁移效果差异较大。
6. **未来方向**：结合 offline RL 训练提示词生成器可在本地测试床提升效率（6.82% vs 3.91%），但跨目标 transfer 增益不明显；探索更多开放权重的 low-refusal 模型作为 prompt writer。

## 研究启发与可借鉴点
1. **策略抽象层次的价值**：进化"生成提示词的程序"而非直接进化提示词，可使搜索空间多样性呈复合增长——ASR 从 0.9%（直接提示词进化）提升到 3.67%（策略进化），提升约 4 倍。这一抽象思路可迁移到其他黑盒优化场景。
2. **Top-k 适应度聚合在稀疏反馈下的优势**：用 top-3 均值替代全量均值使优化器更关注稀有突破事件，在正样本极稀疏的生产系统评估中尤为关键。
3. **元 scratchpad 跨岛屿信息共享**：定期汇总成功/失败模式并生成结构化建议，使各岛屿避免陷入局部最优，同时保留多样性探索——这一机制可推广到多目标进化搜索。
4. **校准式评估框架的系统性意义**：本文建立的"人工标注→裁判校准→严格标准→人类验证最终 ASR"流程可作为 T2I 安全评估的新基准，适用于任何对抗性评估工作。
5. **策略可复用性**：发现的策略可跨场景、跨目标、跨提示词模型迁移，为后续攻击工作提供了可积累的知识资产而非一次性绕过。

## 关键术语表
**ASR（Attack Success Rate）**：攻击成功率，指成功生成违规内容的提示词比例；本文使用 prompt-level 定义（每次提示词 3 次目标调用，任一图像达标即计为成功）。
**Calibrated Judge**：经人工标注校准的裁判；本文使用 Gemini 2.5 Flash + 冻结的类别特异性评分标准，替代 NudeNet/CLIP-NSFW 等弱判别器。
**Abliterated Model**：经 ablation 移除安全机制的大语言模型；本文使用公开可用的 abliterated Qwen3-32B 作为无约束提示词生成器。
**Multi-Island Evolution**：多岛进化搜索；将种群分为 K 个子种群并行进化，定期迁移个体以防止早熟收敛。
**Fixed-Strategy Exploitation**：固定策略利用阶段；进化停止改进后，冻结最优策略并用 UCB1 带宽分配机制集中调用预算。
**Reward Hacking**：奖励黑客现象；优化器利用裁判系统的漏洞获得高分，但实际未产生目标违规内容。
**Precision-Adjusted ASR**：精确率调整 ASR；将自动裁判 ASR 乘以该裁判在人工标注校准集上的精确率，用于估算非完全人工验证场景下的真实违规率。
**UCB1 Bandit**：Upper Confidence Bound 1 博弈策略；用于在固定策略利用阶段动态分配目标调用预算，平衡探索与利用。

## 可复现要素
- **数据集**：NSFW-200（SneakyPrompt 使用）、DREAM 开源 2,048 条提示词（1,024+1,024）、JANUS Civitai seed set（未发布）；本文收集约 1,300 张人工标注图像用于校准。
- **代码/权重**：公开复现 pipeline 和本地测试床，代码仓库计划发布于 https://github.com/whitecircle/rise；使用 abliterated Qwen3-32B（HuggingFace 公开）和 GPT-4.1 API；**不发布**发现的策略、最终攻击提示词和不安全生成图像；hate/suicide 类别的评判标准因较高滥用风险不予公开。
- **关键超参**：K=16 岛屿、archive size=50、λ=10.0、top-3 fitness、筛选阈值 0.05、迁移间隔 10 步、meta-scratchpad 更新间隔 7 步、每提示词 k=3 次目标调用、进化预算 2,500 次目标调用、利用预算 1,500 次目标调用。
