---
title: "On-the-In-effectiveness-of-AMR-Augmentation-for-Large-Langua"
source: https://arxiv.org/pdf/2609.40121v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-03 02:59:32"
field: "大语言模型与结构化知识表示"
keywords: ["AMR", "Large Language Models", "Structured Representation", "Semantic Parsing", "Fine-tuning", "Reproducibility", "Perplexity Analysis"]
innovations: ["系统性复现并证伪AMR增强对现代LLM的有效性", "提出基于困惑度的关系知识探针方法，分离结构化格式priming效应与实际关系知识增益", "将AMR无效性结论从prompt场景扩展至fine-tuning场景"]
benchmarks: ["PAWS", "SNLI", "WMT16", "CoNLL2003", "SST-2", "PubMed45", "WiC", "SPIDER", "AG-News", "RAMS", "CNN/DailyMail", "ANLI", "LogiQA"]
---

# 论文速读：On-the-In-effectiveness-of-AMR-Augmentation-for-Large-Langua

## 一句话总结
本文系统性地重新评估了抽象意义表示（AMR）增强对现代大语言模型的效果，发现AMR增强在不同微调策略和任务类型下均无法带来一致性能提升，并通过困惑度探针证明AMR并未为LLM提供超出纯文本本身的关系知识。

## 研究问题与动机
1. **核心问题**：AMR增强能否实质性地改善现代LLM在下游任务上的表现？此前Zhang et al. (2025)报告了高达12 F1点的提升，但这一结论是否可靠存疑。
2. **先前工作存在分歧**：Jin et al. (2024)在zero-/few-shot设置下发现AMR增强对GPT-4无益甚至有害；而Zhang et al. (2025)通过SFT声称AMR对Llama-3.1-8B有效，二者结论直接矛盾，亟需澄清。
3. **复现性危机**：作者发现Zhang et al. (2025)报告的text-only基线性能（如SNLI仅35.5 F1）远低于预期，即使BERT也能达到~91%准确率，提示其实验配置可能存在问题。
4. **潜在饱和问题**：单句任务上基线模型已接近性能饱和（SST-2、SNLI、PAWS、AGNews、CoNLL2003均超过90%），需在更复杂的多句任务上验证AMR是否真正有帮助。

## 核心贡献（创新点）
1. **系统性复现与证伪**：统一超参搜索协议下重新实现了Zhang et al. (2025)的实验，发现AMR增强与纯文本baseline表现基本持平，先前报告的性能增益源于特定实验配置。
2. **复杂度扩展验证**：将实验扩展至多句事件论元提取、摘要、推理与阅读理解任务，排除单句任务饱和导致的假阴性，仍无一致提升。
3. **困惑度探针方法**：首次提出基于困惑度的关系知识探针，将AMR转化为自然语言描述（AMR-NLD），通过对比纯文本、AMR增强和节点控制条件下的模型困惑度，量化AMR是否真正提供了额外关系信息。
4. **消融与替代集成方案**：探索了GNN编码器、AMRBART编码器等多种结构化集成策略，均无法突破纯文本基线，进一步佐证AMR对LLM无效的核心结论。

## 方法详解
1. **实验框架**：使用Llama-3.1-8B-Instruct和Qwen3-8B两个模型族，通过LoRA微调（r=64, α=128, dropout=0.05，作用于q/k/v/o/gate/up/down七层），每配置五种子随机种子取平均。
2. **单句任务数据集（9个）**：PAWS、SNLI、WMT16、CoNLL2003、SST-2、PubMed45、WiC、SPIDER、AG-News；多句任务数据集（4个）：RAMS（事件论元提取）、CNN/DailyMail（摘要）、ANLI（NLI）、LogiQA（阅读理解）。
3. **三种微调策略**：
   - **Joint Fine-Tuning**：跨所有任务联合微调，训练数据为50:50的text+AMR与纯文本混合。
   - **Individual Fine-Tuning**：每个任务单独微调。
   - **AMR-to-Text Intermediate Objective**：先在AMR 3.0语料上做AMR→文本生成预训练（10 epoch, LR=1e-4），再进入上述微调，以检验模型是否能学会利用AMR结构。
4. **AMR注入方式**：线性化AMR（PENMAN格式）通过`<amr>...</amr>`特殊分隔符插入到对应句子之后；多句任务按句交错插入。
5. **困惑度探针设计**：
   - 从PAWS测试集中抽取178个样本，生成AMR-NLD（AMR自然语言描述），通过分解AMR图为关系分支后由Claude Sonnet 4.6映射为自然语言句子。
   - 计算三种输入条件下的困惑度：纯文本、文本+完整AMR、文本+AMR节点（剥离所有边关系，仅保留概念标签）。
   - 检验公式：$P_M(R_S | S, A_S) \gg P_M(R_S | S)$ 若AMR提供额外关系知识则成立；否则 $P_M(R_S | S, A_S) \approx P_M(R_S | S)$。

## 实验与结果
1. **单句任务（Table 2）**：36种配置中仅2种呈现统计显著差异（p<0.05），分别是Qwen3-8B在SNLI上纯文本显著优于AMR，以及Llama-3.1在SPIDER上AMR轻微优于纯文本。绝大多数情况下两者性能相当或纯文本更优。Zhang et al. (2025)报告的最大提升（如SNLI从35.5→54.9 F1）在复现中未出现。
2. **多句任务（Table 5）**：在RAMS、LogiQA、ANLI、CNN/DailyMail四个任务上，AMR增强与纯文本baseline在两种模型、三种微调策略下均无一致提升，且多句任务基线未饱和（如LogiQA仅~55-71%，RAMS仅~44-50%），排除了饱和解释。
3. **AMR-to-Text预训练效果（Table 13）**：Phase 1后Llama-3.1-8B和Qwen3-8B分别达到46.0和45.7 BLEU，虽低于BiBL的47.4，但证明模型确实学会了AMR格式；Phase 2后因灾难性遗忘降至43.9/43.1 BLEU，仍有相当能力保留。
4. **替代集成策略（Tables 16-17）**：GNN编码器和AMRBART编码器两种结构化集成方式同样未能超越纯文本基线。
5. **困惑度分析（Table 7）**：所有配置下+AMR确实降低了困惑度，但+AMR与+AMR-nodes（控制组）的困惑度差异极小（如Llama Text-FT: 3.06 vs 3.16；Qwen AMR-FT: 4.29 vs 4.37），说明困惑度下降源于结构化格式的 priming 效应而非真正的关系知识增益。
6. **70B模型扩展（Table 12，单种子）**：Llama-3.1-70B-Instruct上结果趋势与8B一致，无显著提升。

## 相关工作脉络
1. **Zhang et al. (2025) — SR-LLM**：报告了AMR等结构化表示对Llama-3.1-8B的SFT提升（最高12 F1点），本文通过统一超参搜索和更广实验范围直接反驳其结论，指出其text-only基线性能异常偏低（SNLI仅35.5 F1）提示实验配置问题。
2. **Jin et al. (2024)**：在zero-/few-shot提示场景下发现AMR增强平均损害GPT-3.5/4性能，本文的发现将该结论延伸至fine-tuning场景，证实AMR对现代LLM无论何种集成方式均无效。
3. **Xu et al. (2022)、Hua et al. (2023)**：在早期LM（如T5）上证明了AMR对文档级事件论元提取和长对话摘要有显著提升，本文解释了这一历史成功的背景——早期模型缺乏足够隐式关系建模能力，而现代LLM已内化此类知识。
4. **Bonial et al. (2020)、Zhang & Ji (2021)**：在QA和联合信息提取中使用AMR图编码取得进展，属于"旧LM + AMR"范式，与现代LLM直接输入纯文本的效能形成对比。
5. **Raut et al. (2025)**：系统评估了linearized AMR few-shot prompt对LLM的影响，与Jin et al. (2024)结论一致，本文在此基础上进一步探索fine-tuning setting。

## 局限性与未来方向
1. **模型规模限制**：主要实验限于8B参数模型（70B仅为单种子探索），未来需系统考察模型规模与AMR有效性之间是否存在非线性关系。
2. **仅针对AMR**：聚焦于AMR这一最流行的语义表示格式，其他格式（如UCCA、UDS、EDS）可能与LLM产生不同的交互效果。
3. **未系统评估解析器误差影响**：虽然使用AMR3-structbart-L（SOTA解析器），但parser错误对AMR增强效果的影响未深入分析；不过作者指出任何AMR集成方法都面临此固有问题。
4. **仅研究微调阶段集成**：未探索在预训练阶段直接融入结构化语义表示的潜力，这可能是更根本的改进方向。

## 研究启发与可借鉴点
1. **困惑度探针作为分析工具**：本文提出的"关系知识困惑度探针"设计优雅且可迁移，可用于评估其他结构化表示（如PRODIGY、语义依赖树、知识图谱）是否为LLM提供了超出纯文本的信息增益，值得在本团队的结构化表示研究中借鉴。
2. **控制变量的对照设计**：AMR-nodes控制组（仅保留节点不保留边关系）的设计精妙地分离了"结构化格式priming效应"与"实际关系知识增益"，这一对照思路可推广至其他结构化输入的分析。
3. **多阶段微调策略**：AMR-to-text intermediate objective的设计展示了如何通过中间任务让LLM先"学会"一种格式再应用于下游任务，该两阶段策略可复用于其他非标准输入格式的微调场景。
4. **替代编码架构的验证**：GNN encoder + Q-Former和AMRBART encoder两种替代方案虽未成功，但其完整实现细节（Appendix D）为后续研究提供了可直接复用的结构化信息注入模板。
5. **对"新瓶装旧酒"研究的批判性复现范式**：本文展示了在复现工作时如何通过统一超参搜索、扩大模型家族、扩展任务复杂度来排除实验配置偏差，为领域内的复现研究提供了方法论范例。

## 关键术语表
**Abstract Meaning Representation (AMR)**：一种以谓词-论元关系为核心的语义表示框架，将句子编码为有根有向无环图，节点代表概念，边编码概念间关系。
**PENMAN notation**：将AMR图序列化为线性文本的格式规范，使标准语言模型可直接处理AMR而无需图结构修改。
**AMR-NLD (AMR Natural Language Description)**：将AMR图中各关系分支逐一分解并映射为自然语言句子的表示形式，用于困惑度探针分析。
**Joint Fine-Tuning**：将所有任务的训练数据混合后联合微调模型的策略。
**AMR-to-Text Intermediate Objective**：先让模型学习从AMR图生成对应文本的中间预训练任务，再将其应用于下游微调的两阶段策略。
**LoRA (Low-Rank Adaptation)**：通过低秩分解对大模型部分参数进行高效微调的技术，本文设置r=64, α=128。
**Perplexity Probe**：通过比较模型在不同输入条件下的困惑度，来量化某种额外输入（如AMR）是否提供了模型原本不具备的信息。

## 可复现要素
- **代码**：GitHub公开（论文第1节末标注）
- **数据集**：PAWS、SNLI、WMT16、CoNLL2003、SST-2、PubMed45、WiC、SPIDER、AG-News、RAMS、CNN/DailyMail、ANLI、LogiQA均为公开数据集；AMR 3.0语料通过LDC获取
- **模型权重**：Llama-3.1-8B-Instruct和Qwen3-8B为开源模型
- **关键超参**：LoRA r=64, α=128, dropout=0.05；target modules为q/k/v/o/gate/up/down七层；batch size=8；10 epochs；weight decay=0.01；学习率网格搜索
- **硬件**：1 GPU（具体型号论文未详述，Appendix有提及flash_attention_2）
