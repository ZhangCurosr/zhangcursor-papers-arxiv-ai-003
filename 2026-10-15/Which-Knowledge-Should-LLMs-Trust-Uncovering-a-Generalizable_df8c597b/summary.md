---
title: "Which-Knowledge-Should-LLMs-Trust-Uncovering-a-Generalizable"
source: https://arxiv.org/pdf/2610.20067v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-04 10:30:34"
field: "大语言模型机制解释与干预"
keywords: ["knowledge conflict", "activation steering", "residual stream", " mechanistic interpretability", "RAG", "context sensitivity"]
innovations: ["发现单一残差流方向实现知识选择的双向连续控制", "证明方向跨格式/关系/数据集及复杂任务（RAG/记忆/规则）的泛化性", "提出基于角分离的自动化干预层定位方法"]
benchmarks: ["NQ-Swap", "MacNoise", "ParaConflict", "CounterFact", "MemoTrap", "RGB", "CRAG", "LoCoMo", "LoCoMo-Plus", "LongMemEval", "IFEval", "FollowBench"]
---

# 论文速读：Which-Knowledge-Should-LLMs-Trust-Uncovering-a-Generalizable

## 一句话总结
论文发现LLM在参数化知识与外部上下文知识冲突时的选择偏好，可由残差流中**单一方向**线性表示与控制；该方向在简单事实冲突、不同格式/关系/数据集上泛化，并能迁移至RAG、多轮记忆、规则遵循等复杂任务，实现双向连续控制。

## 研究问题与动机
1. **核心问题**：LLM存储大量参数化知识，但在复杂场景中还需依赖外部上下文（检索文档、多轮记忆、显式规则），当两者冲突时模型如何选择知识来源的内在机制尚不清楚。
2. **现有输入层方法不足**：提示词设计或结构化RAG模板只能间接控制，无法揭示模型的仲裁机制，也缺乏对内部过程的可见性。
3. **现有干预方法局限**：操纵注意力头、SAE特征或解码分布的方法仅在简单事实数据集上评估，未测试检索证据、长对话历史或显式规则的复杂场景；且需任务特定的属性计算或特征搜索，成本高。
4. **动机**： mechanistic interpretability 研究表明高维概念（如真实性、安全性）在激活空间中呈线性方向；本文验证知识选择是否也能由低维steering轴控制并跨任务泛化。

## 核心贡献（创新点）
1. **发现单一残差流方向实现双向连续控制**：在受控事实冲突基准上提取一个steering方向，可在四种模型家族中平滑调节知识来源选择（参数化↔上下文）。与已有方法的区别在于不依赖任务特定设计或解码时修改，仅干预单个层的一个token位置。
2. **证明方向的跨设置泛化性**：同一方向在不同冲突格式（短替换 vs. 连贯段落）、知识类型（国籍/制造商/职业/宗教/运动/广播）和数据集（NQ-Swap→MacNoise/ParaConflict/CounterFact/MemoTrap）上均保持双向控制效果，排除过拟合或局部捷径。
3. **迁移至复杂应用任务**：从简单事实冲突学到的方向无需重新微调，即可提升RAG（RGB/CRAG）、多轮记忆（LoCoMo/LoCoMo-Plus/LongMemEval）、规则遵循（IFEval/FollowBench）任务中对外部信息的利用，表明不同任务共享底层的外部知识获取机制。
4. **提供新的层定位策略**：提出基于clean/conflict提示对最终token激活的余弦相似度最小化来自动定位干预层，替代传统的性能扫描方法，无需下游任务分数调优。

## 方法详解
1. **配对提示构造**：对每个问题q，构造clean prompt（仅q）和conflict prompt（注入冲突段落+q），保留clean正确但conflict被诱导改变答案的样本，确保可靠参数化知识和已演示的上下文获取。
2. **层定位**：对clean和conflict提示运行模型，计算各层最终token激活的余弦相似度 $s^i = \frac{\bar{h}^{clean}_i \cdot \bar{h}^{conflict}_i}{\|\bar{h}^{clean}_i\| \|\bar{h}^{conflict}_i\|}$，选择相似度最低层ℓ作为干预层（排除已解码输出的最后一层）。
3. **方向提取**：在层ℓ计算差异均值向量 $v = \bar{h}^{clean}_ℓ - \bar{h}^{conflict}_ℓ$，方向从冲突状态指向更强参数化依赖。
4. **推理时干预**：仅在层ℓ和conflict prompt的最终token处编辑残差流：$h' = h + \alpha v$，α为正 favor参数化，α为负 favor上下文，α=0为未干预基线。
5. **评估协议**：匹配前256个生成token到参数化/上下文答案，分为四类（仅参数化/仅上下文/两者/两者都不）；复杂任务使用各benchmark原生评估器。

## 实验与结果
1. **受控事实冲突基准**：在NQ-Swap和MacNoise上，使用四种模型（Llama-3.1-8B-Instruct、Qwen3-8B、Gemma-2-9B-IT、Mistral-7B-Instruct-v0.3），在正向steering下，Gemma-2在NQ-Swap上$\text{Acc}_{par}$达80.9%（↑48.1），Mistral达76.9%（↑31.4）；负向steering下，Gemma-2达65.4%（↑12.8），Mistral达60.2%（↑14.4）。
2. **跨格式迁移**：在ParaConflict上，从替换格式学到的方向在连贯格式上依然有效（Gemma-2 $\text{Acc}_{par}$ 83.9%，↑68.1）。
3. **跨关系迁移**：CounterFact上六个知识类型（国籍/制造商/职业/宗教/运动/广播）间交叉迁移，off-diagonal显著有效。
4. **跨数据集迁移**：NQ-Swap学到的方向直接在MacNoise/ParaConflict/CounterFact/MemoTrap上测试，Gemma-2在ParaConflict上$\text{Acc}_{par}$达92.8%（↑65.2），$\text{Acc}_{ctx}$达92.8%（↑22.4）。
5. **复杂任务迁移**：Table 7显示，在RGB上Llama-3.1提升+20.0（62.0→82.0），Gemma-2提升+17.0（71.0→88.0）；LoCoMo-Plus上Llama-3.1提升+5.0（32.7→36.7），Mistral提升+3.2（32.7→35.9）；IFEval上Llama-3.1提升+1.4，Qwen3提升+1.3。
6. **无冲突性能保持**：Table 5显示，相同方向在不冲突场景下保留大部分原始准确率（如ParaConflict $\text{Acc}_{par}$ 98.0%，$\text{Acc}_{ctx}$ 85.5%）。

## 相关工作脉络
1. **知识冲突干预**：Jin et al.(2024) JUICE切断注意力头、Li et al.(2025) PH3重新加权context-aware neurons、Zhao et al.(2025a) CK-PLUG/SpARE使用SAE特征；本文相比这些方法的本质区别是不依赖任务特定特征搜索，仅用单一残差流方向实现双向控制。
2. **上下文敏感性控制**：Anand et al.(2026) ContextFocus、Minder et al.(2025) CCS、Wang et al.(2025b) CSKS通过层搜索或proxy模型增加上下文依赖；本文证明所学方向捕获的是"源竞争"而非泛化上下文敏感性（matched control实验显示冲突对比产生更强效应，cosine 0.875–0.899）。
3. **激活steering几何**：Turner et al.(2024)、Panickssery et al.(2024)等证明高维概念对应近似线性方向；Zhang & Lin(2026)指出context-in冲突可能涉及残差流角位移；本文沿用几何视角但用角分离定位层而非幅度稀释。
4. **Steering方向迁移**：Marks & Tegmark(2024)证明真实性方向跨数据集泛化；Tan et al.(2025)指出directions在分布偏移下可能脆弱；本文在严格设置下（无目标任务再调优）证明跨格式/关系/数据集/任务家族的稳健迁移。
5. **指令following改善**：Stolfo et al.(2025)通过activation steering提升指令遵循；本文证明同一知识选择方向也能改善规则遵循任务（IFEval/FollowBench），扩展steering的应用场景。

## 局限性与未来方向
1. **需要白盒访问**：方法依赖残差流中间表示，不适用于黑盒/API模型。
2. **单一方向非完整机制**：知识选择可能涉及更广circuits（注意力头、MLP特征、解码动力学），单方向是可控轴但非全部机制。
3. **评估简化**：主要基于受控事实冲突和source-choice准确率，未捕捉更细粒度输出（如犹豫、部分同意、混合答案）。
4. **复杂任务增益较小**：相比受控冲突，复杂任务提升幅度小且依赖模型基础能力；未来需研究模型能力、任务难度、证据质量如何共同决定效果。
5. **未探索跨模型方向对齐**：虽有研究显示概念方向可跨模型对齐，本文未验证跨架构（如不同tokenizer）的方向兼容性。

## 研究启发与可借鉴点
1. **层定位策略可迁移**：基于角分离（余弦相似度最小化）的定位方法比性能扫描更高效，可应用于其他概念方向的自动化定位。
2. **配对提示构造范式**：clean/conflict对比构造可复用于其他需要分离内部/外部依赖的研究（如幻觉检测、事实性控制）。
3. **复杂任务通用方向验证**：证明简单任务学到的方向可迁移至RAG/记忆/规则任务，为"统一控制轴"假设提供证据，启发后续研究探索更广泛的任务迁移。
4. **bidirectional连续控制设计**：通过单方向+符号系数实现双向控制，比单独学习两个方向更高效；可应用于其他需要权衡的任务（如 truthful/helpful 权衡）。
5. **与团队方向结合机会**：若团队研究RAG系统或agent memory，可将此steering作为推理时干预手段，在检索增强或历史摘要中动态调节外部信息依赖程度。

## 关键术语表
**Parametric Knowledge**：存储在LLM参数中的世界知识，通常来自预训练数据。
**Contextual Knowledge**：由prompt提供的临时外部信息（检索文档、对话历史、规则等）。
**Knowledge Conflict**：参数化知识与外部上下文知识不一致导致的决策困境。
**Residual Stream**：Transformer中每层传递的隐藏状态残差连接，知识选择方向在此空间线性表示。
**Activation Steering**：在推理时向残差流添加定向向量以修改模型行为的干预技术。
**Source Match**：评估指标，判断生成答案与参数化答案或上下文答案的字符串匹配程度。
**Layer Localization**：通过比较clean/conflict激活的角分离自动确定最佳干预层的方法。
**Bidirectional Control**：通过正负α系数在同一方向上实现参数化↔上下文的双向切换。

## 可复现要素
- **数据集**：NQ-Swap、MacNoise、ParaConflict、CounterFact、MemoTrap、RGB、CRAG、LoCoMo、LoCoMo-Plus、LongMemEval、IFEval、FollowBench（均为公开benchmark）。
- **代码**：已开源 https://github.com/AheadOFpotato/knowledge_selection_steering。
- **模型**：Llama-3.1-8B-Instruct、Qwen3-8B、Gemma-2-9B-IT、Mistral-7B-Instruct-v0.3（开源权重）。
- **关键超参**：α系数在验证集选定（NQ-Swap 70/15/15分割），干预层位置（Llama:10, Qwen3:21, Gemma-2:16, Mistral:16），max_new_tokens=200，batch_size=128（NQ-Swap 64），precision=bfloat16，greedy decoding。
- **硬件**：单卡NVIDIA A100 80GB。
