---
title: "System1-Arena-Which-Decisions-Can-Be-Left-to-System-1"
source: https://arxiv.org/pdf/2610.21244v1.pdf
model: agnes-2.5-flash
chunks: 5
summarized_at: "2026-11-04 04:01:30"
field: "大语言模型评估与校准"
keywords: ["System-1模型", "概率校准", "决策评估", "LLM基准", "闭环智能体", "推理预算"]
innovations: ["提出System1-Arena基准，首次系统性对比System-1与System-2模型在92,125个决策任务上的表现", "设计双轨Prompt体系以公平评估不同概率输出机制的模型", "发现推理提升准确性但损害概率校准的新现象"]
benchmarks: ["System1-Arena", "SWE-bench", "MMLU-Pro", "BiGGen-bench", "LiveBench"]
---

# 论文速读：System1-Arena-Which-Decisions-Can-Be-Left-to-System-1

## 一句话总结
本文提出了覆盖92,125个问题的系统性基准System1-Arena，用于对比单次前向传播的System-1模型与需多步推理的System-2 LLM在决策任务上的表现，发现System-1模型在非智能体的静态决策任务上已可匹敌甚至超越快速闭源LLM（如GPT-6 Luna），但在闭环长程规划与最高精度需求场景下仍存在明显差距。

## 研究问题与动机
- **核心问题**：单次前向传播的System-1模型能否胜任广泛决策任务，还是必须依赖System-2式生成推理的LLM？
- **现有基准不足**：已有评测（如MMLU-Pro、HELM等）多聚焦知识问答或通用能力，缺乏对"决策"这一核心交互范式的统一评估，尤其缺少对概率校准质量与延迟/成本的综合考量。
- **System-1 vs System-2机制差异**：闭源LLM以verbalized概率输出（JSON），开源/System-1模型可读取label token logprob，二者获取概率信息的机制不同，缺乏公平统一的对比框架。
- **闭环智能体评估缺失**：真实决策往往涉及多步交互与环境演化（如代码控制、购物、机器人操作），现有基准对这类闭环任务的覆盖有限。

## 核心贡献（创新点）
- **提出System1-Arena基准**：覆盖92,125个问题、105个任务、268个公开来源、16个领域的系统性决策评估基准，分为非智能体任务与单步/闭环智能体任务两大类别，与已有通用评测（MMLU-Pro、BiGGen-bench等）形成功能互补。
- **设计双轨Prompt体系**：针对闭源LLM（verbalized JSON概率输出）与开源/System-1模型（label token logprob读取）的概率获取机制差异，分别设计统一prompt格式，确保跨模型族可比性，且prompt不做逐模型调优。
- **系统性对比30个模型**：涵盖7个闭源LLM、13个开源LLM和10个System-1模型，在准确率、Brier分数、延迟、成本四个维度给出全面对比，首次量化System-1模型与顶级LLM在各类决策任务上的性能差距。
- **揭示推理提升准确性但损害校准的发现**：推理LLM（如Qwen3.8-27B经2,048 token推理后准确率从70.2%→75.6%，但Brier从0.391升至0.430），证明System-2推理改善预测准确度却未改善概率校准。
- **提供闭环环境基准**：构建138个闭环环境（含网络购物、SWE-bench代码控制、机器人、游戏、驾驶等），其中128个报告success/failure，支持实时模式评估，弥补已有基准在动态交互任务上的空白。

## 方法详解
- **问题决策请求格式**：状态s + 类型化问题Q（布尔/选择/评分），每问题附指令与选项描述，统一输入给各模型。
- **闭源LLM Prompt**：系统消息要求模型输出JSON对象，将每个问题映射到各选项的条件概率，所有问题合并为一个请求一次调用完成；缺失选项默认概率为0。
- **开源/System-1模型Prompt**：每个问题单独一个prompt，通过chat template打开assistant回复位置，直接读取label token的条件概率；选项>26个时用多位数字标签（"001"–"151"），在前缀树上评分，$z_k$为词元logprob之和。
- **概率评估指标**：accuracy = avg@k（取k=1）；Brier score衡量概率校准；ECE（10等频分箱）评估校准度；NDCG用于排序类任务；pair accuracy衡量成对请求全对比例。
- **无答案问题处理**：无答案问题对Brier score / ECE / NDCG计为均匀分布，避免遗漏偏差。
- **Thinking Budget实验**：对4个开放LLM（Qwen3-8B、Qwen3.8-27B、Gemma-4-26B-A4B、DiffusionGemma）设置预算B∈{0,128,256,512,1024,2048}词元，生成thought后闭合思维标记再读标签；Budget 0相较thinking off有惩罚（Qwen3.8-27B −2.1分，Gemma-4-26B-A4B −3.6分，DiffusionGemma −13.5分）。
- **运行框架**：所有开放模型用vLLM（Kwon et al., 2023）运行。
- **软标签**：4,425个请求携带软标签（由100名标注者选择比例构建），用于部分任务的细粒度评估。

## 实验与结果
- **数据集**：System1-Arena，共92,125个问题，105个任务，268个公开来源，16个领域；非智能体任务78,362个问题（87个任务），单步智能体任务13,763个问题（6,943个agent步骤 + 3,828个游戏 + 2,992个具身/机器人任务）；138个闭环环境。
- **评估模型（30个）**：闭源LLM 7个（Claude Opus 5.5、GPT-6 Astra、Grok 4.7、GPT-6 Sol、GPT-5.6 Sol、GPT-6 Luna、GPT-5.6 Luna）；开源LLM 13个（Qwen3/3.5/3.8系列、Gemma 4系列、Llama 3.3-70B、GPT-OSS、DiffusionGemma）；System-1模型10个（Jev、Decision-2.0系列Vega-27B/Lux-9B/Sol-2B/Nox-4B/Kai-0.6B、Microsoft-Decision-1、OpenAI Decisions API、Laya 421M、CLM-v0.1-8B）。
- **非智能体任务关键结果**：
  - Claude Opus 5.5准确率最高（81.1%），但延迟2,293ms、成本$10.38/千次；Grok 4.7延迟高达6,694ms。
  - **最佳System-1模型Decision-2.0-Vega-27B准确率达73.0%、Brier 0.343、延迟仅228ms**，超越GPT-6 Luna（71.2%）和GPT-5.6 Luna（70.0%）。
  - Jev 1.13以72.6%准确率、321ms延迟、$0.052/千次成本表现优异，成本远低于闭源LLM（约70×差距）。
  - **System-1模型概率校准优于同级别LLM**：Decision-2.0-Vega-27B Brier 0.343 vs GPT-6 Luna 0.461。
  - 推理提升准确性但不改善校准：Qwen3.8-27B经2,048 token推理后准确率70.2%→75.6%，Brier 0.391→0.430，平均置信度81.4%→97.4%。
- **按领域差距（System-1 vs 最佳LLM）**：最小差距为Filter/group/rank（0.5pp）、Retrieval/ranking（2.3pp）、Dialogue（3.9pp）；最大差距为Event forecasting（13.3pp）、Code understanding（13.1pp）、Structured data（12.2pp）。
- **最强结果**：非智能体任务中Claude Opus 5.5以81.1%准确率为最优；System-1阵营中Decision-2.0-Vega-27B以73.0%准确率为最优，领先次优Jev 0.4pp。
- **智能体任务**：Gemma-4-31B单步准确率71.3%、闭环成功率32.9%；论文第1段表格在此截断，完整智能体结果未完全提供。
- **硬件与服务**：非agentic用H100，agentic用A100；封闭LLM通过GitHub Copilot/Amazon Bedrock；Jev通过Vercel AI Gateway；OpenAI Decisions API $0.10/百万输入词元；Microsoft-Decision-1 via OpenRouter $0.042/百万输入词元。

## 相关工作脉络
- **MMLU-Pro（Wang et al., 2024b）、HELM（Liang et al., 2022）、BiGGen-bench（Kim et al., 2025）、LiveBench（White et al., 2025）**：通用LLM能力评测基准，侧重知识问答与语言理解，缺乏对决策概率输出与校准的系统评估；本文System1-Arena聚焦决策范式并统一评测概率质量。
- **AgentRewardBench（Lù et al., 2025）、τ-bench（Yao et al., 2024）、Terminal-Bench（Merrill et al., 2026）**：智能体评测基准，但本文构建的138个闭环环境（含实时模式）覆盖更广泛的交互类型（购物、代码控制、机器人、游戏、驾驶）。
- **"Language Models (Mostly) Know What They Know"（Kadavath et al., 2022）、KL Calibration（Guo et al., 2017）**：LLM校准研究，本文在此基础上将校准评估扩展到System-1模型并与System-2 LLM直接对比，发现推理改善准确度但损害校准的新发现。
- **SWE-bench（Jimenez et al., 2023）、ACPBench（Kokel et al., 2025）、PlanBench（Valmeekam et al., 2023）、ALFWorld（Shridhar et al., 2020）**：代码与规划智能体基准，本文在138个闭环环境中包含SWE-bench代码控制等任务，但额外强调实时模式与延迟成本的综合评估。
- **FActScore（Min et al., 2023）、RAGTruth（Niu et al., 2024）、TofuEval（Tang et al., 2024）**：事实核查与幻觉评测，本文覆盖Fact-checking相关任务（如LegalBench、CUAD），但以决策准确率与校准为统一度量。
- **ProcessBench（Zheng et al., 2025）、MR-Ben（Zeng et al., 2024）、Let's Verify Step by Step（Lightman et al., 2024）**：数学推理基准，本文覆盖日期算术等推理密集型任务，但核心差异在于以概率校准而非仅正确答案率作为评估维度。

## 局限性与未来方向
- **部分任务受许可证限制**：2,607个请求因版权无法重新分发，只能由loaders本地重建，限制了基准的完全可复现性。
- **智能体闭环结果不完整**：论文第1段中智能体任务的表格被截断，闭环成功率的完整排名未给出，难以全面评估System-1在长程规划中的真实能力。
- **System-1模型规模偏小**：最优System-1模型Decision-2.0-Vega-27B为27B参数，而顶级闭源LLM（Claude Opus 5.5、GPT-6 Astra）远超此规模，在最高精度需求任务上差距仍达8pp。
- **实时模式下的延迟纳入**：部分闭环环境存在实时演化，延迟直接影响性能，但当前评估对延迟的归一化方式有待进一步标准化。
- **思考预算实验仅覆盖4个模型**：Thinking Budget实验仅对Qwen3-8B、Qwen3.8-27B、Gemma-4-26B-A4B、DiffusionGemma四个模型进行，结论外推性有限。

## 研究启发与可借鉴点
- **双轨Prompt设计思路可迁移**：针对不同类型的概率输出接口（verbalized JSON vs. label token logprob）分别设计统一prompt，这一思路可应用于任何需要跨模型公平对比概率输出的评测场景。
- **校准与准确率联合评估框架**：同时报告accuracy与Brier/ECE，揭示"推理提升准确度但损害校准"的洞察，为后续研究提供了超越单一准确率指标的评估范式。
- **闭环实时环境评估设计**：138个闭环环境中纳入实时模式（环境在推理期间持续演化），将延迟纳入性能衡量，这一设计对智能体研究具有重要参考价值。
- **Thinking Budget渐进实验**：通过逐步增加推理词元预算（0–2,048）观察性能变化，可复用为分析LLM推理深度与性能关系的实验模板。
- **与团队方向结合机会**：若团队关注高效推理或边缘部署，System-1模型的低延迟低成本特性（Jev $0.052/千次 vs Claude Opus $10.38/千次）为"何时用System-1、何时用System-2"的混合架构决策提供了实证依据。

## 关键术语表
**System-1模型**：单次前向传播直接返回选项概率分布的模型，无需多步推理生成，典型代表为Jev、Decision-2.0系列。
**System-2 LLM**：需经多步生成推理（如Chain-of-Thought）才能获得答案的大语言模型，典型代表为Claude Opus、GPT-6系列。
**Brier分数**：衡量概率预测校准质量的指标，值越低表示模型输出的概率越准确反映真实发生率。
**ECE（Expected Calibration Error）**：将预测置信度分桶后计算预测准确率与平均置信度的加权偏差，用于量化概率校准度。
**Soft Label**：由多名标注者投票比例构建的标签，而非单一硬标签，用于更细粒度的任务评估（本文共4,425个请求使用）。
**Label Token Logprob**：开源模型中直接读取正确选项对应token的对数概率，区别于闭源模型的verbalized概率输出。
**Thinking Budget**：限制模型推理生成词元数量的实验设计，用于分析推理深度对性能的影响。
**闭环环境**：模型输出动作后环境状态会演化的交互场景（如购物、代码执行、机器人控制），与静态问答任务相对。

## 可复现要素
- **数据集**：System1-Arena基准，92,125个问题；2,607个请求因版权限制无法重新分发，其余可通过loaders本地重建。
- **代码/权重**：System-1模型（Jev、Decision-2.0系列、Laya、CLM-v0.1-8B等）通过作者发布代码获取请求；开源LLM通过公开权重运行；闭源LLM通过API调用。论文未明确声明System1-Arena基准代码是否开源。
- **关键超参**：vLLM运行所有开放模型；Thinking Budget取值B∈{0,128,256,512,1024,2048}词元；软标签由100名标注者构建；ECE采用10等频分箱；多选项（>26）使用多位数字标签前缀树评分。
- **硬件**：非agentic任务使用H100，agentic任务使用A100；四模型Thinking Budget实验共消耗58.8 GPU-hours on H100。
