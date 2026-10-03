---
title: "YOU-CANNOT-PICK-A-PROVIDER-FROM-THE-PRICELIST-MARKET-AWARE-R"
source: https://arxiv.org/pdf/2609.37902v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-03 02:58:49"
field: "LLM推理系统与cost-aware routing"
keywords: ["LLM routing", "provider selection", "open-weight inference", "cost-aware serving", "online certification", "inverse bandit", "market-aware routing"]
innovations: ["提出provider选择作为open-weight LLM路由中被忽视的独立维度，形式化为price-taker市场的inverse bandit问题", "设计FACET在线可行性认证框架，通过certify-before-serve+per-task facet认证+CUSUM监控实现安全与成本权衡", "基于6模型×多任务×多provider×三轮测量的实证体系，揭示provider可行性是task-selective且drifting的"]
benchmarks: ["GSM8K", "MMLU", "HumanEval"]
---

# 论文速读：YOU-CANNOT-PICK-A-PROVIDER-FROM-THE-PRICELIST-MARKET-AWARE-R

## 一句话总结
本文指出开放权重LLM推理市场存在一个被忽视的决策维度：选定模型后，仍需选择由哪个provider提供推理服务；作者通过跨模型、任务、provider的多轮实测证明"provider不可从价目表推断"，并提出FACET在线可行性认证框架，可在保证质量的前提下实现最高约57%的推理成本节省。

## 研究问题与动机
1. **已有LLM路由器仅做模型选择，忽略provider选择**：现有系统（RouteLLM、FrugalGPT、CARROT、MixLLM等）只回答"选哪个模型"，但在开放权重市场中，同一模型由多个provider以不同量化、内核、批处理策略和超时行为提供服务，选完模型后仍需选provider。
2. **价格不能可靠预测quality或availability**：价格与延迟呈稳定负相关（中位Spearman ρ=−0.61），但与准确率（ρ=+0.05）和可用性（ρ=0.00）无稳定关系，高价provider未必质量更高。
3. **Feasibility是task-selective且time-varying**：同一endpoint可在知识类任务上表现正常，却在多步推理任务上"灾难性降级"；provider可行性需按(model, task, provider)三维度索引。
4. **静态测量图会漂移**：尽管标价变化稀少，但在43天内有11–22%的cell发生路由重选，驱动因素为availability恢复、限流、provider退出和静默质量衰减，而非价格变动。

## 核心贡献（创新点）
1. **提出provider选择作为独立路由维度**：将"同一模型下的provider选择"形式化为受质量、健康、延迟、成本约束的routing问题，与已有模型路由层正交且可堆叠。
2. **揭示open-weight provider市场的全方位异质性**：基于6个open模型×多任务×多provider×三轮测量的实证数据，证明价格仅能预测延迟，无法预测task-conditioned quality或availability，且存在task-selective退化。
3. **提出measured-map最廉价等效健康路由**：在每(model×task)cell中选择质量偏差≤5分、可用性≥90%的最廉价provider，在中位数上实现50–56.5%成本节省，out-of-sample下-floor率仅0–11.1%。
4. **提出FACET在线可行性认证框架**：将provider选择建模为inverse bandit（价格/延迟可观测、质量是约束而非目标），通过certify-before-serve规则防止未认证端点接现实流量，配合CUSUM持续监控检测已认证provider的质量漂移。

## 方法详解
**问题形式化**：给定上游已选定模型w和query context $\hat{c}_t$，在provider集合$\mathcal{P}(w)$中选择$p$，最小化实际成本，满足约束$q_p(w,c_t,t)\ge\theta_{w,c_t}$（质量下界）、$h_p(t)\ge h_{\min}$（健康）、$\ell_p(t)\le\ell_{\max}$（延迟预算）。质量下界定义为$\theta_{w,c}=q_{\max}(w,c)-\delta$，主实验中$\delta=0.05$。

**Measured-Map Router**：基于离线测量构建(task-conditioned)可行性地图，每(model×task)cell选取实测准确率在最佳provider 5分以内、可用性>90%的最廉价provider。

**FACET（在线可行性认证）**：
- **Fail-safe Anchoring**：新选中的廉价provider不被立即服务，先进行认证探针；流量回落到预认证的anchor。
- **Per-(provider×task) Certification**：每个(provider×task)构成独立feasibility facet，窗口内最近200条观测中准确率与同task最佳provider差距≤8分且≥$n_{\min}=12$条样本，且不低于floor $\theta-0.03$时才认证。证据不在task间转移。
- **Continuous Monitoring**：认证后的provider通过Bernoulli LLR-CUSUM持续监控，阈值$h=\log(1/0.01)$，检测已认证provider的质量漂移并隔离。
- **OOD处理**：未知任务作为uncertified facet fail-safe回落，而非映射到nearest facet。

**Cost-Safety权衡**：定义应用级损失$L=\text{cost}+\lambda\cdot N_{\text{below}}$；相对SW-UCB，FACET的break-even点约17×单次查询成本；相对冻结地图/周期刷新，break-even仅约1×查询成本。

## 实验与结果
**数据集与测量**：覆盖Llama、DeepSeek、Gemma、Mistral、Qwen等open-weight模型；light probes（数学推理、信息抽取、情感分类、代码输出）+基准规模GSM8K、MMLU、HumanEval（n=100–150/cell）；三轮测量（day 0/13/43）。

**Measured-Map路由**（Table 18）：
- In-sample：94%均值准确率，1.57×最廉价provider成本，下-floor率0%；对比：premium路由成本3.67×且17%下-floor；cheapest路由28%下-floor。
- Out-of-sample（早波选provider→晚波评估）：5分质量边界+90%健康阈值下，9/9 comparable cell均未跌破floor（0%）。
- 统计置信支撑的saving（需95% CI支持5分边界）：中位saving从55%降至46%。
- 与上游RouteLLM堆叠：MMLU下-floor率从55.9%降至0%（成本仅+$8.4）；HumanEval从57.5%降至0%（成本+$60.3）。

**FACET在线认证**：
- 周期重测量vs FACET（Table 4）：FACET总成本$1255，下-floor 0.17%；周期P=100需$2361且下-floor 3.70%。
- Ablation（Table 3）：去掉anchor则在S1下-floor暴露8.3%，去掉task-conditioned则在S1暴露6.7%。
- 不完备任务分配（Table 5）：33%标签错误时FACET暴露4.72%（vs context-blind 6.72%）；50% OOD时FACET保持0%。
- 系统评价偏差（Table 29）：70% targeted bias下，judged probes暴露4.33%，gold probes+5% audit降至0.43%。
- 稀疏反馈（Table 30）：5%标签率下FACET暴露4.57%（SW-UCB为13.32%）。

**Live部署验证**（Table 6）：36小时Llama-3.3-70B×12 provider，服务538 query，130 certification probes；从$1.04/M anchor迁移至$0.21/M认证endpoint；serving cost降低63.7%，总成本降低57.1%，准确率88.3%。

**价格-延迟分析**：21/21 cell价格-延迟Spearman均为负（中位ρ=−0.61）；10s SLA保留所有cell，成本仅+19%；1s SLA仅6/12 cell可行，成本2.62×。

## 相关工作脉络
1. **Client-side LLM路由器**（RouteLLM、FrugalGPT、CARROT、MixLLM）：解决"选哪个模型"问题，依赖静态价目表；本文在模型选定之后处理provider选择，与上述工作正交且可堆叠。
2. **Provider-side定价机制**（PriLLM等）：provider自主定价，客户端不可控；本文面向price-taker客户端视角。
3. **跨provider测量榜单**（Token Arena/Chatbot Arena）：提供静态snapshot质量排名，不闭合测量→路由的在线闭环。
4. **Constrained/Safe Bandits**（Safe LLM routing文献）：一般bandit将reward作为探索目标；本文的provider选择是inverse bandit结构——经济目标（价格）可观测，质量是待满足的约束。
5. **Change-detection Bandits**（SW-UCB、M-UCB）：适合平滑漂移，但在catastrophic cheap mine场景早期可能探索即伤害用户；FACET专攻突变式大规模质量断崖。
6. **量化/推理加速工作**（GPTQ、AWQ、Sarathi-Serve、PagedAttention）：解释provider质量差异的底层原因之一，但本文不尝试归因，强调endpoint-level行为可观测即可用于路由。

## 局限性与未来方向
1. **测量依赖单一public aggregator和单客户端位置**，延迟差异可能混入aggregator地理/路由效应，精度结论相对稳健但速度结论有confound。
2. **Light probes样本量有限**，主要区分provider间的较大差距，边际差异置信区间不足；需更大样本或confidence-aware规则。
3. **FACET并非在所有漂移形态下最优**：对渐进式退化（gradual ramp）或多并发cheap mine场景，滑动窗口bandit可能更安全廉价；FACET定位为"保险层"而非通用替代。
4. **未解决open-set任务识别问题**：OOD检测假设已知reject信号可用，本身是独立问题。
5. **质量信号存在信任边界**：当evaluator存在系统性偏差时，judged certification会被污染，需ground-truth probes或独立审计作为防护，增加了工程复杂度。
6. **未建模完全流动的token spot市场**：当前市场存在 contracted rate与peak/off-peak差异，aggregator仅暴露前者，直接API访问会面对完全不同的成本景观。

## 研究启发与可借鉴点
1. **"Price-taker + latent-constraint"的inverse bandit建模**：将经济目标（价格）作为可观测排序信号、将质量作为待满足约束，区别于经典bandit的reward-maximization范式，可用于其他有明确经济价目但服务质量不透明的服务选择场景。
2. **Certify-before-serve的安全 Admission 机制**：FACET的fail-safe anchoring + per-facet认证 + CUSUM持续监控三段式设计，是处理"cheap catastrophic mine"类风险的通用模式，可迁移至云实例选择、CDN路由等场景。
3. **Task-conditioned feasibility的granularity设计**：同一provider在不同task上可行性的显著差异提示，任何跨task共享证据的certification都会带来安全隐患；fine-grained (model×task×provider)索引值得在任务分类精度有限时设计置信边界鲁棒版本。
4. **实测驱动的routing参数设计**：以TOST等价性检验、out-of-sample跨波评估、component ablation等方式分离"saving来源"与"安全机制贡献"，这种层层剥离的实验设计对routing系统论文的评估极具参考价值。
5. **系统级live部署与stress test的结合**：本文同时提供exact-replay（控制变量）、stress-test（放宽假设）和36小时live部署三种验证层次，可作为LLM系统工程论文的可复现标杆。

## 关键术语表
**Provider Selection**：在已选定open-weight模型后，从多个 competing endpoints 中选择实际服务该模型的host/provider的决策过程。
**Feasibility Facet**：按(provider×task)细粒度划分的可行性单元，一个endpoint可能在一类任务上可行而在另一类上灾难性退化。
**Inverse Bandit**：与经典bandit相反，经济目标（价格）可预观测，需要探索的是质量是否满足约束这一latent constraint。
**Fail-safe Anchor**：预认证的高质量provider作为fallback，新候选未经certify前不接用户流量，只能落在anchor上。
**CUSUM Slip Detection**：基于Bernoulli序列的累计和控制统计量，用于检测已认证provider的准确率是否发生显著下降。
**Below-floor Exposure**：被路由至实际质量低于任务下界的provider的query比例，是核心安全指标。
**Cheapest-Equivalent-Healthy Router**：离线measured-map路由策略，选择质量等效（偏差≤5分）且健康（可用性≥90%）的最廉价provider。
**Ground-truth Audit**：用真实答案直接评分certification probe而非依赖judge model，以缓解evaluator系统性偏差带来的信任问题。

## 可复现要素
- **数据集**：GSM8K、MMLU、HumanEval；light probes（数学推理、信息抽取、情感分类、代码输出预测）；测量通过public multi-provider aggregator采集，provider列表含DeepInfra、Cloudflare、DigitalOcean、Nebius、Parasail、AtlasCloud、SiliconFlow、CoreWeave、Venice、Phala、Groq、Cerebras等。**论文未明确声明独立数据集开源**。
- **代码**：论文未明确声明开源代码仓库（截至发文）。
- **权重**：不涉及模型权重，使用现有open-weight模型的API endpoint。
- **关键超参**：质量边界$\delta=0.05$（5分margin）；认证阈值：窗口200条观测、$n_{\min}=12$、与best同task差距≤8分且不低于$\theta-0.03$；健康监测≥90%可用性；CUSUM阈值$h=\log(1/0.01)$；漂移检测post-change alternative为$\min(\mu_s-0.10,\theta-0.02)$。
