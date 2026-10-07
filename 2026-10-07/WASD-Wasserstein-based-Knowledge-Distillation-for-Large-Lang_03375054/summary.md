---
title: "WASD-Wasserstein-based-Knowledge-Distillation-for-Large-Lang"
source: https://arxiv.org/pdf/2610.07706v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-07 22:29:42"
field: "大语言模型高效训练与压缩"
keywords: ["知识蒸馏", "Wasserstein距离", "Sinkhorn散度", "大语言模型", "词元语义对齐", "最优运输"]
innovations: ["提出WASD，利用基于词元嵌入代价的Sinkhorn散度进行语义感知的逐词元分布对齐", "推导梯度等价的stop-gradient可优化目标，无需额外网络且不经运输求解器反向传播"]
benchmarks: ["Dolly Eval", "Self-Instruct", "Vicuna", "Super-Natural Instructions", "Unnatural Instructions", "AlpacaEval", "Evol-Instruct", "UltraFeedback", "HumanEval", "MBPP", "GSM8k", "DialogSum", "Flores-200"]
---

# 论文速读：WASD-Wasserstein-based-Knowledge-Distillation-for-Large-Lang

## 一句话总结
本文提出WASD（Wasserstein-based Knowledge Distillation），通过基于词元嵌入的Wasserstein距离（Sinkhorn散度）将教师模型分布与学生分布对齐，使蒸馏过程显式利用词元语义信息，在多个LLM家族与任务上持续超越现有f-散度蒸馏方法，并改善生成质量与多样性的权衡。

## 研究问题与动机
- LLM缩放带来推理计算/内存成本上升，知识蒸馏（KD）通过对齐逐词元的离散概率分布实现知识迁移，但主流方法仅依赖概率数值比较，忽略词元间的语义关系。
- 现有基于f-散度/平滑变体的目标在词汇索引层面计算，对近义词（如sofa/couch）不作区分；损失曲面随教师概率赋值变化，而非随词义变化。
- LLM输出分布高度稀疏，依赖密度比的散度易出现数值不稳定；虽然已有辅助分布或logits对齐等稳定性改进，但仍缺乏词元空间几何信息的显式利用。
- 熵正则化最优运输虽可计算，但会在相同分布处引入非零的熵偏差，使优化最优偏离教师分布，不利于蒸馏的目标一致性。

## 核心贡献（创新点）
- 提出WASD：以教师词元嵌入构造代价矩阵，并用Sinkhorn散度对齐教师/学生逐词元分布，使语义相近词元的概率移动承担更低代价。
- 推导梯度等价的可优化WASD目标：基于对偶势差构造权重，并用stop-gradient处理势差，避免反向传播穿过运输求解器，无需额外网络。
- 理论化对比KL类目标：展示WASD以词元几何加权的学生似然梯度，区别于KL以教师概率直接加权的索引级信号。
- 系统评估多种LLM家族与尺度：在指令遵循、摘要、翻译、算术推理与代码生成上持续提升，并在质量-多样性前沿上更优。
- 消融验证语义代价的有效性：证明有意义词元几何优于均匀/置换代价，并显示Sinkhorn偏差校正进一步带来增益。

## 方法详解
- 将蒸馏目标写成逐词元发散度量求和：最小化期望下教师条件分布与学生条件分布之间的差异。
- 用Wasserstein距离刻画分布差异，代价矩阵C由教师词元嵌入之间的距离构建（如余弦距离），从而把词元语义几何引入对齐过程。
- 为可计算性引入熵正则项，得到熵正则Wasserstein距离；其可通过Sinkhorn-Knopp迭代求对偶势，并在实践中采用稀疏近邻截断（保留每词元k个最近非自邻并对称化、加自环）加速。
- 为避免熵正则导致的自相似偏差，采用去偏Sinkhorn散度：以交叉运输势减去师生各自自运输势，使其在p=q时唯一为零，适合作为蒸馏目标。
- 给出梯度等价WASD损失：以学生概率加权和的形式，权重为教师-学生对偶势与学生自对偶势之差，并对势差应用stop-gradient；理论证明该损失的梯度与Sinkhorn目标梯度一致。
- 对偶势的计算通过两类迭代完成：交叉势用Sinkhorn-Knopp算法，自势用不动点迭代；反向传播仅作用于学生概率项，不穿过迭代求解过程。
- 损失可解释为学生分布下“对数似然梯度”被势差加权；与KL相比，权重来自词元几何而非单纯教师概率。

## 实验与结果
- 通用指令遵循：在GPT-2（1.5B→0.1B/0.3B）、OpenLLaMA2（7B→3B LoRA）与Qwen2.5（7B→1.5B）上，使用Dolly/Self-Instruct/Vicuna/Super NI/UnNI等基准，以ROUGE-L为主要指标。WASD在多数设置下取得最高平均ROUGE-L；在相同训练时间对比中仍保持优势。
- GPT-2 XL→Base：WASD平均ROUGE-L达24.02（1.5B→0.1B）与25.51（1.5B→0.3B），优于AMiD、CSD、DistiLLM系列与ABKD等；在OpenLLaMA2 7B→3B setting达到29.61，整体领先。
- Qwen2.5指令遵循（结合DistiLLM-2框架并用GPT-4o-mini裁判）：WASD在AlpacaEval/Evol-Instruct/UltraFeedback三项胜率均高于AMiD基线。
- 任务特定蒸馏（Gemma-7B-IT→2B-IT）：WASD在翻译COMET、摘要ROUGE-L与算术准确率和代码生成pass@1上均优于CSD与AMiD。
- 代码生成（Qwen2.5-Coder 7B→1.5B）：WASD在HumanEval/MBPP平均pass@1达74.9，优于DistiLLM-2、CSD与AMiD。
- 消融：语义代价优于均匀与置换代价；Sinkhorn散度优于纯熵正则Wasserstein；适度小的ε更有利于保留词元几何敏感性；近邻k在较小值（如8）即有效。
- 开销：WASD训练时间约为基线的约2倍，GPU显存增加约20%，但无推理额外开销；更快收敛可在相同时长预算下获得更高性能。

## 相关工作脉络
- KL/RKL/TV/GJS/SKL/SRKL/α-β等f-散度及平滑变体：仅用词汇索引概率比较，缺少词元语义几何；WASD在此类方法的基础上引入运输语义对齐。
- 辅助分布方法（如TAID、AMiD、DistiLLM系列）：提升优化稳定性与质量，但仍主要基于概率/logit层面；WASD可与其结合，进一步利用词元空间几何。
- 基于得分匹配的对齐方法（如CSD）：避免softmax致稳效应，关注相对logit差异；与WASD不同，WASD显式建模词元间语义代价。
- 已有Wasserstein KD工作（如SinKD、WKD、WCoRD）：侧重样本/特征或类别几何；WASD聚焦自回归逐词元分布并纠熵偏。
- 跨tokenizer蒸馏（如ULD、MultiLevelOT、MCW-KD）：解决词汇不匹配；WASD假设共享词表并深耕词元内语义结构，二者互补。
- 基于Token嵌入几何的策略正则（如WPR）：用于偏好学习中的策略正则；WASD将其思想迁移到教师-学生分布匹配，并以Sinkhorn散度修正偏差。

## 局限性与未来方向
- Sinkhorn迭代带来额外计算开销：训练时间与显存占用上升，需进一步优化高效/低内存的Sinkhorn实现。
- 代价矩阵依赖教师词元嵌入的通用语义距离，可能无法充分捕获任务或领域特定的词元关系；可探索任务自适应或领域感知的代价构造。
- 实验主要在共享词表设定下进行；跨词表场景需结合投影或多代价对齐机制扩展。
- 熵正则超参ε影响几何敏感性：过大易弱化词元差异，需按任务/尺度调参或设计自适应机制。

## 研究启发与可借鉴点
- 将词元嵌入几何引入KD目标，可作为现有f-散度/辅助分布框架的正则或替换项，提升语义一致性。
- 对偶势差的stop-gradient梯度等价技巧，使得通过迭代求解器定义的目标仍可与标准训练管道兼容，便于工程落地。
- 稀疏近邻截断与对称化策略在大规模词表上显著降低计算压力，为可扩展的最优运输对齐提供参考。
- 质量-多样性权衡评估（如ROUGE-L与Self-BLEU前沿）可作为蒸馏方法的补充评测维度，揭示语义对齐的实际收益。
- 与跨tokenizer/跨架构框架结合的可能性：在不同词表空间中分别构造语义代价并做投影对齐，可能进一步推广WASD。

## 关键术语表
- **Wasserstein距离**：通过最小化将概率质量从一个分布搬运到另一分布的代价来衡量两分布差异。
- **Sinkhorn散度**：在熵正则Wasserstein基础上减去自相似项，消除熵正则偏差并保持可计算性。
- **对偶势**：熵正则最优运输对偶问题中的潜在变量，可通过矩阵缩放迭代高效求解。
- **stop-gradient**：在前向传播保留值、在反向传播切断梯度的操作，用于固定优化权重而不穿过求解器。
- **熵正则**：在运输目标中加入熵项以提升计算 tractability，但会引入自相似非零的偏差。
- **代价矩阵**：定义词元间搬运成本的矩阵，本文常用教师词元嵌入的距离构造。
- **f-散度**：基于概率比值的一族散度（如KL、JS等），仅依赖索引概率而不建模词元语义。
- **辅助分布**：用于稳定蒸馏训练的混合或中间分布，常被用于改进优化行为。

## 可复现要素
- 代码开源：是，公开于 https://github.com/aailab-kaist/WASD。
- 数据集：使用公开数据集（如databricks-dolly-15k、OpenWebText、Flores-200、DialogSum、GSM8k、WizardCoder、HumanEval、MBPP等），论文未提及自建私有数据。
- 关键超参：熵正则系数ε常设0.001；近邻数k多设8（Qwen2.5设2以提升效率）；Sinkhorn/不动点迭代步数通常10步；学习率依模型设置（如GPT-2/OpenLLaMA2为1e-4，Gemma为1e-5，Qwen为5e-5）；学生温度缩放常用2。
- 硬件：训练使用单卡NVIDIA RTX PRO 6000，评测使用单卡NVIDIA RTX 3090。
