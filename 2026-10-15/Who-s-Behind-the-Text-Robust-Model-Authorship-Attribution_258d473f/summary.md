---
title: "Who-s-Behind-the-Text-Robust-Model-Authorship-Attribution"
source: https://arxiv.org/pdf/2610.19583v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 17:17:15"
field: "模型溯源与AI安全"
keywords: ["Model Authorship Attribution", "Likelihood-based Attribution", "Distribution Shift", "Source-Conditioned Modeling", "Clipped Likelihood", "MAA Benchmark", "Distillation Tracing", "Robust Text Attribution"]
innovations: ["提出ATLAS似然归因框架，通过源条件文本建模替代直接分类实现鲁棒归属", "设计token级截断似然聚合机制防止极端token主导决策", "构建MAABench基准系统评估domain和version联合偏移"]
benchmarks: ["MAABench", "OpenTuringBench"]
---

# 论文速读：Who-s-Behind-the-Text-Robust-Model-Authorship-Attribution

## 一句话总结
本文提出ATLAS（Authorship Tracing via Likelihood-based Attribution of Sources），一种基于源条件文本建模的模型作者归属方法，通过在token级别聚合截断似然证据实现鲁棒的归属决策；同时构建MAABench基准，在分布偏移（领域变化、模型版本变化及二者联合）设置下显著优于现有基线。

## 研究问题与动机
- **核心问题**：给定一段文本，准确识别其生成来源的LLM模型家族（Model Authorship Attribution, MAA）。
- **现实动机**：LLM生成文本被滥用（学术不端、模型蒸馏溯源、网络攻击等），仅检测"是否机器生成"不足，需追溯具体来源以追责。
- **现有方法不足**：
  1. 现有MAA方法多依赖源判别特征，在训练/测试分布相近时表现良好，但在分布偏移（领域变化、模型版本迭代）下性能显著下降。
  2. 直接预测源标签容易学习到不稳定的捷径特征，对领域或版本变化敏感。
  3. 现有MAA基准未系统覆盖模型版本偏移（version shift），难以评估实际部署下的鲁棒性。
- **关键洞察**：生成建模比判别建模更能抵抗捷径解（Li et al., 2025），因此从生成式/似然角度建模更具鲁棒性。

## 核心贡献（创新点）
- **提出ATLAS似然归因框架**：将MAA表述为源条件文本建模问题，通过比较文本在各源条件下的token级似然进行归属，与直接分类的本质区别在于学习"源如何生成文本"而非"文本属于哪个源"。
- **设计截断似然聚合机制**：在token级别比较源条件似然并截断pairwise比值，防止极端token主导决策；与简单平均log-likelihood的本质区别在于bounded score对异常token具有鲁棒性。
- **构建MAABench分布偏移基准**：系统评估生成领域（domain）与模型版本（version）的单独及联合偏移，填补了MAA基准在version shift评估上的空白；与现有MAA基准（如OpenTuringBench）的本质区别在于同时覆盖domain和version双重偏移。
- **展示蒸馏溯源的扩展应用**：证明ATLAS学到的源条件似然可保留教师模型特征，成功识别所有6个公开蒸馏模型的真实教师，即使不同学生基于不同底座模型；与仅用于文本归属的本质区别在于扩展到模型关系推断。

## 方法详解
**ATLAS（Authorship Tracing via Likelihood-based Attribution of Sources）** 核心设计分为训练与推理两阶段：

**1. 源条件建模（Source-Conditioned Text Modeling）**
- 不再直接预测源标签，而是为每个源家族 $\mathcal{M}_k$ 学习一个源条件 $c_k$，建模该家族的文本分布。
- 训练目标：对ground-truth源 $k^*$ 的文本最大化其似然，损失函数为均值负对数似然：
  $$\mathcal{L}(y) = -\frac{1}{n}\sum_{t=1}^{n} \ell_{k^*, t}, \quad \ell_{k,t} = \log p(y_t \mid y_{<t}, c_k)$$
- 源码仅用于选择优化条件，不直接参与分类损失。

**2. 轻量源特定适配（Lightweight Source-Specific Adaptation）**
- 冻结预训练LLM主干，每个源 $k$ 学习可训练前缀嵌入 $e_k$ 和源特定的LoRA上投影 $B_k$（下投影 $A$ 共享）。
- 权重更新：$W_k = W + \Delta W_k$，其中 $\Delta W_k = B_k A$（非对称LoRA，参考HydraLoRA）。
- 7B模型可训练参数量约24.79M（占0.33%），在attention投影（q,k,v,o）上应用，rank=16, α=32。

**3. Token级截断似然聚合（Clipped Likelihood Aggregation）**
- 推理时对所有 $K$ 个源条件计算token log-likelihood $\ell_{k,t}$。
- 避免极端token主导：对每个token $y_t$ 和源 $k$，与所有竞争源比较并截断ratio至1：
  $$r_{k,t}^{(T)} = \exp\left(-\frac{1}{T}\sum_{j \neq k} [\ell_{j,t} - \ell_{k,t}]_+\right) = \prod_{j \neq k} \min\left(\frac{p(y_t|y_{<t}, c_k)}{p(y_t|y_{<t}, c_j)}, 1\right)^{1/T}$$
- 对多个温度 $T \in \{1, 2, 4, 8\}$ 平均后得最终得分 $R_k$，预测 $\hat{k} = \arg\max_k R_k$。
- 截断机制保证score在(0,1]区间，source $k$ 分配最高似然时$r_{k,t}=1$，竞争源解释更好时递减。

**4. 推断变体：蒸馏溯源**
- 对蒸馏模型，计算 $s_k(y) = \frac{1}{n}\sum_t \ell_{k,t}$，减去冻结主干似然 $s_0(y)$ 后标准化得 $z_k(y)$。
- 拒绝率定义为max_k $\tilde{s}_k(y) < 0$ 的比例，反映无源条件能优于主干解释。

## 实验与结果
**数据集**：
- **MAABench**（本文构建）：5个LLM家族（GPT、Claude、Gemini、Mistral、Llama），各4个版本；6个领域（News、Wikipedia、Academic Writing、Product Review、Patent、Stack Exchange）；共228,520条机器生成文本、35,443条人类文本。
- **OpenTuringBench**（La Cava & Tagarelli, 2025）：7个开源模型，264,321条训练文本。

**评估设置**：ID（同分布）、Domain shift（领域偏移）、Version shift（版本偏移）、Joint shift（二者联合偏移）。

**主要结果（MAABench，Table 2）**：
| 设置 | ATLAS Acc | 最强基线 | 提升 |
|------|-----------|----------|------|
| ID | 99.72% | 99.58%（DA-MTL） | +0.14pp |
| Domain shift | 97.97% | 95.31%（DA-MTL） | +2.66pp |
| Version shift | 84.30% | 71.98%（Idiosyncrasies） | +12.32pp |
| Joint shift | 74.27% | 65.27%（Idiosyncrasies） | **+9.00pp** |

**OpenTuringBench结果（Table 3）**：
- OOD设置下ATLAS达95.71%准确率，超越最强基线**33.52pp**。
- 所有ID变体（E1-E4）均取得最优或次优。

**MGT检测扩展（Table 4）**：
- ATLAS在MAABench上作为MGT检测器表现竞争力，Joint shift AUROC达99.75%。

**蒸馏溯源（Section 6, Figure 3）**：
- 6个已知教师的蒸馏模型全部被正确识别（得分63.3%-93.6%）。
- 7个部署模型中，MiniMax-M2.7 Turbo（93.8%→Claude）、Kimi-K2.6（85.2%→Claude）、GLM 5.2（84.7%→Claude）、DeepSeek-V4-Pro（90.6%→GPT）显示强归属倾向。
- 对比模型Falcon3-7B和Granite-3.0-8B拒绝率高达68.2%和74.6%，无明确归属。

**消融分析（Table 5）**：
- 截断聚合对Version shift提升4.55pp、Joint shift提升5.12pp。
- 源特定$B_k$较共享$B$提升2.18pp（Version）。
- Backbone从0.5B增至7B在OOD设置增益更大（Joint shift +7.09pp）。
- 数据效率：仅1,000文本/源即超越DA-MTL全量训练（Version: 80.10% vs 72.78%）。

## 相关工作脉络
- **MGT检测**：Fast-DetectGPT（概率曲率）、Binoculars（perplexity ratio）、GPT-who（信息密度）、Sniffer（多代理统计）、IRM/ReMoDetect（隐式奖励）。本文与它们的本质区别：不问"是否机器生成"，而问"哪个模型生成"，且基于源条件似然而非单一统计量。
- **MAA方法**：
  - 固定特征类：StatEnsemble（8维统计+MLP）、GPT-who（surprisal特征）。本文定位：生成建模比统计特征更鲁棒于分布偏移。
  - 编码器微调类：DeTeCtive（对比学习+NN）、OTBDetector（triplet loss）、DA-MTL（多任务）、Idiosyncrasies（线性头）。本文定位：直接分类易学捷径，源条件生成建模更泛化。
- **基准**：
  - TURINGBench（Uchendu et al., 2021）：固定生成器集，无偏移评估。
  - OpenTuringBench（La Cava & Tagarelli, 2025）：有domain shift但无version shift。
  - EvoBench（Yu et al., 2025）：有version shift但针对MGT检测非MAA。
  - 本文MAABench的定位：首个同时评估domain和version联合偏移的MAA基准。
- **生成vs判别鲁棒性**：Li et al. (2025)证明生成分类器更少依赖捷径特征，本文据此采用源条件生成建模。

## 局限性与未来方向
- **固定候选集假设**：仅能在预定义的$K$个源家族中归属，无法处理训练后出现的未知新源。
- **重训练需求**：若需纳入新源，需重新训练或适配，缺乏在线增量学习机制。
- **混合作者归属未探索**：人类与模型协同写作场景下，定位每部分来源的任务尚未研究。
- **跨语言泛化**：当前仅评估英语，多语言MAA未涉及（虽StatEnsemble有相关工作）。
- **API依赖限制**：部分评估依赖黑盒API（GPT/Claude/Gemini），可复现性受限于API访问权限。

## 研究启发与可借鉴点
- **生成式归因范式**：将归属问题重构为"源条件建模+似然比较"，相比直接分类更易泛化至OOD；可迁移至其他文本溯源任务（如风格归属、语言变体识别）。
- **Token级截断聚合**：bounded score设计防止极端值主导，适用于任何基于token似然的聚合场景（如检测、分类）；实现简单且效果显著（+4.55pp on Version）。
- **非对称LoRA适配**：共享下投影$A$、源特定上投影$B_k$的设计（参考HydraLoRA）兼顾参数效率与源区分度；可推广至多域/多源轻量适配。
- **蒸馏溯源应用**：证明模型输出特征在蒸馏后仍保留教师指纹，为模型血缘追踪提供技术路径；可拓展至版权保护、模型滥用检测。
- **基准设计**：MAABench的seed-level划分、prompt配对生成、长度约束过滤、domain/version联合偏移设计，为后续基准构建提供模板。

## 关键术语表
- **Model Authorship Attribution (MAA)**：识别给定文本由哪个LLM模型家族生成的细粒度溯源任务。
- **Source-conditioned text modeling**：为每个源家族学习条件文本生成模型，通过比较似然进行归属而非直接分类。
- **Clipped likelihood aggregation**：在token级别截断pairwise似然比至1，防止极端token主导最终归属决策。
- **Asymmetric LoRA**：下投影矩阵共享、上投影矩阵源特定的LoRA变体，兼顾参数效率与源区分能力。
- **Version shift**：测试时模型版本不同于训练版本（如同一家族的新版模型）。
- **Domain shift**：测试时文本领域不同于训练领域（如专利、Stack Exchange作为未见领域）。
- **Joint shift**：领域和模型版本同时发生偏移的最挑战性OOD设置。
- **Distillation teacher identification**：利用ATLAS源条件似然识别蒸馏模型的真实教师模型。

## 可复现要素
- **数据集**：MAABench已公开（官网komseok0109.github.io/ATLAS/），种子池含content-addressed identifier可验证；OpenTuringBench为公开基准。
- **代码**：实验代码开源（github.com/komseok0109/ATLAS），HuggingFace demo可用（huggingface.co/spaces/ming9999/ATLAS-demo）。
- **关键超参**：Backbone Qwen2.5-7B；LoRA rank=16, α=32, dropout=0.05；学习率1e-4，linear warmup 5%后cosine decay；batch size=128；Epoch=5（MAABench）/3（OpenTuringBench）；温度集合{T∈{1,2,4,8}}；最大长度512（MAABench）/1024（OpenTuringBench）。
- **硬件**：4×H100-80GB GPUs。
- **随机种子**：主实验mean over 3 seeds (42, 123, 456)。
