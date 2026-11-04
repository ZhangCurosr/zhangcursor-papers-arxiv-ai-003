---
title: "Which-Knowledge-Should-LLMs-Trust-Uncovering-a-Generalizable"
source: https://arxiv.org/pdf/2610.20067v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-04 10:30:25"
field: "大语言模型机制可解释性与知识选择调控"
keywords: ["knowledge conflict", "activation steering", "residual stream", "mechanistic interpretability", "RAG", "contextual knowledge", "parameter knowledge"]
innovations: ["通过干净-冲突对比在残差流中提取可跨数据集迁移的知识选择方向", "基于角分离自动定位干预层，无需下游评分循环", "单向干预从受控事实冲突直接迁移至 RAG、多轮记忆与规则遵循任务并持续生效"]
benchmarks: ["NQ-Swap", "MacNoise", "ParaConflict", "CounterFact", "MemoTrap", "RGB", "CRAG", "LoCoMo", "LoCoMo-Plus", "LongMemEval", "IFEval", "FollowBench"]
---

# 论文速读：Which-Knowledge-Should-LLMs-Trust-Uncovering-a-Generalizable-Steering-Direction-Across-Knowledge-Conflicts

## 一句话总结
论文发现大型语言模型在参数知识（parametric）与上下文知识（contextual）之间的选择行为，可被残差流（residual stream）中一个单一方向向量线性表征；通过 clean-conflict 对比提取该方向后，在推理时施加带符号系数干预即可实现双向连续调控，且该方向可在不同冲突格式、数据集以及更复杂的 RAG、记忆、规则遵循任务间迁移。

## 研究问题与动机
- 现有大模型在参数化知识（模型内部存储的世界知识）与外部上下文知识（检索文档、多轮对话历史、显式规则）发生冲突时，常错误依赖不准确来源，产生有害输出。
- 已有干预方法分两类：输入层（提示/模板设计）可控性弱、缺乏可解释性；内部层（注意力操控、SAE 特征、解码时修改）多在简单事实数据集上评估，缺乏对 RAG、长交互历史、显式规则等复杂场景的鲁棒测试，且伴随任务特定特征搜索或解码改造成本。
- 机制可解释性研究表明，高维概念（真实性、安全性、情绪、拒绝等）在激活空间里常表现为近似线性方向，能够作为因果中介实现细粒度输出操控；本文追问：知识选择在冲突下的仲裁是否也存在一个低维、可泛化的 Steering 轴？

## 核心贡献（创新点）
- 在受控事实冲突基准上识别出残差流中单一方向，可在四个模型族上实现知识选择的双向、连续调控。
- 该方向跨不同上下文格式、知识类型与数据集保持一致有效性，不绑定于单一评估模板或数据分布。
- 最关键的是，从简单事实冲突学到的方向无需目标任务再拟合，直接迁移至 RAG（RGB、CRAG）、多轮记忆（LoCoMo、LoCoMo-Plus、LongMemEval）和规则遵循（IFEval、FollowBench）任务均取得增益，暗示不同任务族共享一条“获取外部知识”的底层方向。
- 通过匹配控制实验证明，该方向的作用超越一般性的"context sensitivity"——冲突对比产生的方向在其他非冲突对齐/无关上下文中效果显著弱于冲突对比。

## 方法详解
- **成对 prompt 构造**：对每个问题 q，构造 clean prompt（仅含 q）与 conflict prompt（注入与参数知识相矛盾的上下文段落后再接 q），保留 clean 正确、conflict 被注入答案带偏的样本，构成可靠的参数/上下文双源对照。
- **层定位（Layer Localization）**：对 clean/conflict 分别在每层末尾 token 计算平均残差激活 $\bar{h}^{(\text{clean})}_i$、$\bar{h}^{(\text{conflict})}_i$，用余弦相似度 $s^i_{T_{\text{clean}}, T_{\text{conflict}}} = \frac{\bar{h}^{(\text{clean})}_i \cdot \bar{h}^{(\text{conflict})}_i}{\|\bar{h}^{(\text{clean})}_i\| \|\bar{h}^{(\text{conflict})}_i\|}$ 度量两分布分离程度，取相似度最低层作为干预层 $\ell = \arg\min_i s^i$；排除已接近最终解码输出的末层，仅选取更早的角度极小层。
- **方向提取**：在选定层 ℓ 以差异均值构建方向 $v^{\ell} = \bar{h}^{(\text{clean})}_{\ell} - \bar{h}^{(\text{conflict})}_{\ell}$，方向从“冲突 regime"指向“更强参数依赖”。
- **推理时激活干预**：仅在冲突 prompt 的最后一 token 与选定层 ℓ 处注入，公式 $h'^{\ell} = h^{\ell} + \alpha v^{\ell}$；α 为带符号标量：α>0 偏向参数答案，α<0 偏向上下文答案，α=0 即无干预基线。
- **评估协议**：首 256 token 归一化后按字串匹配分四类（parametric-only / contextual-only / both / neither），$\operatorname{Acc}_{\text{par}}$、$\operatorname{Acc}_{\text{ctx}}$ 为两种方向的选择率；复杂任务沿用原始 benchmark 打分器，不使用 256-token 字符串匹配。
- **实现细节**：推理阶段干预，不微调模型；bf16 精度，单卡 A100 80GB；greedy decode，max_new_tokens=200；训练/验证/测试集按 70/15/15 划分用于方向估计与 α 选择。

## 实验与结果
- **模型与数据集**：四模型（Llama-3.1-8B-Instruct、Qwen3-8B、Gemma-2-9B-it、Mistral-7B-Instruct-v0.3）；主要事实冲突基线 NQ-Swap、MacNoise；跨格式/跨关系/跨数据集测试 ParaConflict、CounterFact、MemoTrap；复杂任务 RGB、CRAG、LoCoMo、LoCoMo-Plus、LongMemEval、IFEval、FollowBench。
- **主表结果（NQ-Swap + MacNoise 双向）**：
  - 参数方向（Acc_par）最优：Gemma-2-9B-IT 达到 80.9%（↑48.1 over unsteered），Llama-3.1-8B 达到 56.8%（↑23.1），均显著优于 ContextFocus、CCS、CSKS、CK-PLUG、PH3、IRCAN、JUICE、SpARE 等基线。
  - 上下文方向（Acc_ctx）最优：Qwen3-8B 达到 56.0%（↑19.3），Mistral 达到 60.2%（↑14.4）。
- **连续调控**：α 扫测显示参数/上下文选择率呈平滑反向变化，个体 sample 轨迹也呈现由单源→混合→另一单源的连续过渡。
- **鲁棒性**：
  - 外部知识强度（NQ-Swap, Llama-3.1-8B）：随上下文出现次数从 1 增至 8，参数方向始终保持 47.9%–56.8%；上下文方向随证据增强由 55.5% 升至 74.8%。
  - 题目难度（Hard 最难）：参数方向将低难度 42.6% 提升至 62.4%，难点从 12.4% 提升 37.4 个百分点至 49.8%。
- **非冲突性能保持**：同一方向系数在未冲突数据集上仍保持较高原答案准确率（ParaConflict 参数方向 98.0%；MacNoise/MemoTrap/CounterFact 约 80%–88%），上下文方向代价略高。
- **跨格式/跨关系/跨数据集迁移（方向仅在 NQ-Swap 选定）**：
  - 跨格式（ParaConflict）：参数方向由 15.8% 提升至 83.9%；上下文方向由 81.4% 提升至 90.8%。
  - 跨关系（CounterFact 六类事实）：对角/非对角均显正迁移（图 5）。
  - 跨数据集（MacNoise/ParaConflict/CounterFact/MemoTrap）：四个模型均双向显著改善（如 Llama 参数方向在 ParaConflict 达 83.9%，Gemma 在 CounterFact 达 76.8%）。
- **复杂任务迁移（Table 7）最强结果**：
  - RGB（RAG 子集）：Gemma-2-9B 达到 88.0%，较未干预基线提升约 +17 个百分点；Mistral 达 86.0%（↑5.0）。
  - LoCoMo-Plus（记忆）：Llama-3.1-8B 达 36.7%（↑4.0），Qwen3-8B 达 45.4%（↑17.2）。
  - IFEval（规则）：Qwen3-8B 达 36.2%（↑1.3）。
  - 整体趋势：所有模型在所有复杂任务上，Ours 多数指标优于或持平于 ContextFocus、JUICE、CK-PLUG 等对照；RGB 受益最大（强反事实冲突），CRAG 收益较小（噪声/陈旧片段混合）。
- **与一般 context sensitivity 的区分**：冲突上下文对比产生的方向在四个数据集上均远强于对齐上下文与无关上下文，Cosine 相似度 0.875–0.899，证明并非单纯“上下文敏感”，而是显式源竞争机制。
- **代码开源**：https://github.com/AheadOFpotato/knowledge_selection_steering

## 相关工作脉络
- **知识冲突干预路线**：JUICE（注意力剪切）、SpARE/CK-PLUG（SAE 特征）、PH3（注意力/神经元加权）、IRCAN（上下文感知神经元重加权）、ADACAD（解码时平衡）。本文定位：不依赖任务特定特征搜索/解码修改，通过残差流差异均值实现双向连续控制。
- **上下文敏感方向**：ContextFocus（按层搜索增强上下文依赖）、CCS（context-sensitivity direction）、CSKS（proxy model 连续调控）。本文定位：通过干净-冲突对比而非单纯“有/无上下文”对比得到更强方向，且证明其在更复杂 RAG/记忆/规则任务上的可迁移性。
- **线性表征假说/激活工程**：Neel 等人 truthfulness 方向（Li et al., 2024）、拒绝方向（Arditi et al., 2024）、Refusal/Safety（Wolf et al., 2025）、情绪（Sofroniew et al., 2026）、转向几何与线性假设（Turner et al., 2024; Park et al., 2024; Marks & Tegmark, 2024）。本文定位：首次将"source competition"抽象为单一线性方向并测试多任务迁移。
- **层定位策略**：传统精度扫层（sweep）、attention head/neuron 归因（Meng et al., 2023; Geva et al., 2023）。本文定位：提出基于干净/冲突余弦角分离的自动层选择，避免下游任务评分循环。
- **跨模型方向迁移**：Huang et al., 2025 跨族对齐、Tan et al., 2025 分布漂移脆弱性讨论。本文定位：在同一族内跨数据集/跨格式跨知识类型的单向迁移，避免 target-side 再拟合。
- **冲突几何视角**：Zhang & Lin, 2026 认为 in-context 冲突解决涉及残差流角位移而非幅度稀释。本文定位：借鉴角度分离定位层，以差异均值构造可加性方向。

## 局限性与未来方向
- **需白盒访问**：方向估计与注入均在残差流层面，无法直接用于 API/closed-source 模型。
- **方向是控制轴而非完整机制**：知识选择可能涉及更广电路（注意力头、MLP 特征、解码动力学），单方向仅为可观测的一维投影。
- **评估聚焦源选择准确率**：未充分刻画 hedging、部分认同、混合答案等更细腻输出形式；string-match 256 token 窗口亦可能遗漏语义等价答案。
- **复杂任务增益依赖于模型能力与证据质量**：CRAG 等噪声/陈旧片段场景增益较小，RGB 因强反事实结构增益最大；模型能力、任务难度、证据质量三者联合效应仍待建模。
- **跨模型迁移未测试**：当前所有迁移在同族内，跨架构方向的直接平移仍需研究。
- **方向稳定性依赖训练-验证-测试分离**：极端 α 可能引发 off-target 输出或重复（附录 G.3 所示失败案例）。

## 研究启发与可借鉴点
- **干净-冲突余弦角分离法做层定位**：无需下游评分循环，可推广至其他行为（诚实、偏好、指令遵循）的层识别。
- **方向由差值均值构造**：简洁、可复现、计算成本低；可作为后续“知识来源偏好”“指令依赖 vs. 内部先验”等行为的通用提取范式。
- **匹配控制（aligned/irrelevant context vs. conflicting）**：区分“上下文敏感”与“源竞争”的实证策略，值得在更多冲突类型任务中复用。
- **256-token 首段匹配 + LLM-judge 盲评双轨校验**：兼顾效率与构念效度，可沿用为生成-字符串匹配的评估规范。
- **同方向复用于 RAG/记忆/规则三类不同形式的外部依赖**：为后续“一个方向能否统一调控检索依赖、记忆依赖、指令约束依赖”提供初步证据，可作为团队在多任务统一干预方向上的起点。

## 关键术语表
- **Parametric Knowledge（参数知识）**：模型在训练过程中以权重形式内化的世界知识，不依赖 prompt 内给出的证据。
- **Contextual Knowledge（上下文知识）**：由 prompt 注入的外部信息，包括检索文档、对话历史、显式规则等。
- **Residual Stream（残差流）**：Transformer 各层之间传递的隐状态主干，是激活干预（activation steering）最常用的操作对象。
- **Knowledge Selection Direction（知识选择方向）**：由干净 prompt 与冲突 prompt 的平均残差激活之差构造出的低维向量，沿此方向移动可实现参数/上下文依赖的双向连续调控。
- **Clean-Conflict Contrast（干净-冲突对比）**：通过构造仅有问题与问题+矛盾上下文的成对样本，抽取能区分两源依赖的模式。
- **Angular Separation / Cosine Similarity（角分离/余弦相似度）**：用于在每层度量干净与冲突激活分布的差异，取最小余弦相似度层作为干预层。
- **Activation Steering（激活干预/转向）**：在推理时向选定层的残差流加上带符号的向量，改变模型输出倾向，不修改模型权重。
- **Source Match（源匹配）**：将生成首段文本与参数答案/上下文答案进行归一化子串匹配，判定模型选择了哪类来源。

## 可复现要素
- **数据集**：NQ-Swap、MacNoise、ParaConflict、CounterFact、MemoTrap、RGB、CRAG、LoCoMo、LoCoMo-Plus、LongMemEval、IFEval、FollowBench；大部分为公开 benchmark。
- **代码/权重**：代码开源（https://github.com/AheadOFpotato/knowledge_selection_steering）；使用开源指令微调模型 Llama-3.1-8B-Instruct、Qwen3-8B、Gemma-2-9B-IT、Mistral-7B-Instruct-v0.3。
- **关键超参**：bf16 精度；greedy decode（do_sample=False）；max_new_tokens=200；batch_size=64（NQ-Swap）/128；seed=42（采样）/43（复杂任务）；训练/验证/测试 70/15/15；256-token 答案窗口；左填充、pad_token=eos_token；干预层分别为 Llama 10、Qwen3 21、Gemma 16、Mistral 16。
- **计算资源**：单卡 NVIDIA A100 80GB。
