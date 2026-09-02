---
title: "Visual-Attention-Faithfulness-in-Vision-Language-Models-is-H"
source: https://arxiv.org/pdf/2609.00830v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 12:01:50"
field: "多模态大模型可解释性"
keywords: ["视觉注意力忠实性", "VLM可解释性", "因果扰动", "完备性与充分性", "视觉token评估"]
innovations: ["首次通过完备性+充分性双维度系统评估VLM视觉注意力因果忠实性", "提出三种异构处理模式分类框架揭示注意力忠实性的非均匀性", "揭示人类标注与模型视觉依赖的系统性偏差（仅~60%对齐）"]
benchmarks: ["VQAv2", "VRDU", "ChartQA"]
---

# 论文速读：Visual-Attention-Faithfulness-in-Vision-Language-Models-is-H

## 一句话总结
本文首次系统评估了VLM中视觉注意力权重的因果忠实性，通过token级零扰动实验提出完备性（comprehensiveness）与充分性缺口（sufficiency gap）度量，揭示视觉注意力忠实性呈三种异构处理模式（Faithful-Sufficient / Faithful-Distributed / Non-Focal），并发现人类标注区域与模型实际视觉依赖存在系统性偏差（仅~60%满足完备性）。

## 研究问题与动机
1. NLP领域已围绕注意力权重能否忠实反映模型推理展开长期辩论，但该问题在视觉模态（VLM）中仍属空白
2. 现有工作盲目假设低注意力视觉token可安全剪枝/合并用于推理加速，未验证注意力排名是否真实对应模型的视觉依赖
3. 可视化方法（Grad-CAM、注意力热力图）仅展示"模型看向何处"，缺乏因果验证"这些区域是否真正驱动预测"
4. 人类标注的真实区域可能无法准确代表模型的内部视觉依赖机制，需建立定量评估基准

## 核心贡献（创新点）
1. **首个VLM视觉注意力忠实性系统评估**：提出基于完备性与充分性的因果扰动度量框架，将注意力忠实性从定性辩论转向可量化评估；此前工作仅用相关性指标或单维度分析，本文同时测量必要性与充分性两个正交维度。
2. **发现三种异构处理模式**：识别出Faithful-Sufficient、Faithful-Distributed、Non-Focal三种模式，揭示注意力忠实性是非均匀的graded属性而非二元判断；与以往将注意力视为单一可靠性指标的工作形成本质区别。
3. **揭示人机视觉依赖的系统性分歧**：证明人类标注GT区域仅~60%满足完备性，模型常依赖非语义对应的上下文区域（如颜色属性从周边色块推断而非对象本身）；打破了"人类标注即模型忠实性ground truth"的默认假设。
4. **跨任务与跨架构泛化验证**：将三种模式推广至文档IE（VRDU）和图表VQA（ChartQA）等结构化任务，并验证Qwen2.5-VL与InternVL2.5在不同架构下均遵循相同模式框架但模式分配不同；证明该框架捕捉的是跨architecture的普遍规律而非数据集artifact。

## 方法详解
1. **聚合注意力评分**：对每层每头的注意力权重求平均，跨生成token求和，得到每个视觉token的聚合重要性分 $a_i = \sum_{l=1}^{L}\frac{1}{H}\sum_{h=1}^{H}\frac{1}{|y|}\sum_{s=1}^{|y|}\alpha_{l,h}(i,y_s)$，按降序排列取top-k得 $S_k$。
2. **因果扰动设计**：在语言decoder首层入口对选定token集合做zero-ablation（$h_i^{(0)} \leftarrow \mathbf{0}$），而非从序列删除，以保留空间结构不引入额外信息；扰动后比较预测概率变化。
3. **完备性度量**：$\operatorname{Comp}(S_k) = \frac{P(y|V,T) - P(y|V\setminus S_k, T)}{P(y|V,T)}$，衡量top-k token是否必要（高Comp表明注意力忠实于因果证据）。
4. **充分性缺口度量**：$\operatorname{Sgap}(S_k) = \frac{P(y|V,T) - P(y|S_k,T)}{P(y|V,T)}$，衡量仅保留top-k能否恢复预测（低/负Sgap表明注意力captured全部必要信息）。
5. **模式划分**：先对Comp做Otsu二值化分离high/low区，再对high-Comp子集对Sgap做Otsu二值化，得到三组并辅以人工边界修正。

## 实验与结果
- **数据集**：VQAv2（90样本，平衡yes/no、number、other三类），VRDU（60文档IE样本），ChartQA（100图表VQA样本）
- **基线策略**：Overall Top-k、GT Object（人类标注）、Non-GT Top-k、Random（四组对比，token数k对齐GT区域大小）
- **主要结果（VQAv2）**：Overall Top-k mean Comp=0.813、Sgap=0.469，显著优于GT（Comp=0.505）和Random（Comp=0.502）
- **三模式分布**：Faithful-Sufficient 32.9%（Comp=0.978, Sgap=-0.026）、Faithful-Distributed 47.1%（Comp=0.952, Sgap=0.818）、Non-Focal 20.0%（Comp=0.213, Sgap=0.460），三类模式下模型准确率均高（92.9%~100%）
- **最强结果**：Overall Top-k策略在所有模式下保持最高Comp最低Sgap；模式A样本中Top-k与GT高度重合时GT-Comp达0.963
- **跨模型验证**：InternVL2.5-8B同样呈现三模式（A:29.4%, B:45.9%, C:24.7%），但IE/ChartQA任务偏向模式A（Sgap@10%=0.189 vs Qwen的~0.947）
- **稳定性**：采样n=50/60/70重复1000次，三模式比例波动<5%，95% CI不含零；3B与7B规模结果一致

## 相关工作脉络
1. **Jain & Wallace (2019) "Attention is not Explanation"**：证明attention可被permutation而不改变输出，支持attention非忠实论；本文在VLM视觉模态下复现了类似结论的异质性版本——部分样本faithful（模式A/B），部分not（模式C）。
2. **Wiegreffe & Pinter (2019) "Attention is not not Explanation"**：反驳Jain，指出constrained attention训练导致行为差异；本文方法与此精神一致，通过扰动验证causal contribution而非相关性。
3. **DeYoung et al. (2020) ERASER**：提出comprehensiveness与sufficiency双维度框架评估解释忠实性；本文将其从NLP文本token迁移到VLM视觉token，是方法论的直接延伸。
4. **Jacovi & Goldberg (2020)**：形式化忠实性为graded property而非binary；本文据此将VLM注意力忠实性分类为三种模式而非简单二元判定。
5. **Grad-CAM (Selvaraju et al., 2017) 与 Chefer et al. (2021)**：生成空间heatmaps展示attention热点但未验证因果必要性；本文指出这些可视化方法无法区分"相关"与"因果驱动"。
6. **Token pruning工作（Chen et al. 2024a; Shang et al. 2025; Zhang et al. 2025）**：假设低attention token可安全剪枝；本文揭示Non-Focal模式（20%样本）及Faithful-Distributed模式（47%样本）下，简单attention阈值剪枝可能导致信息损失。

## 局限性与未来方向
1. **模型覆盖有限**：仅评估动态分辨率VLM（native-grid与dynamic-tile两类），不适用于固定分辨率模型（如LLaVA-style）
2. **未涵盖推理型VLM**：多步deliberation模型可能有不同的视觉注意力行为模式，留作未来工作
3. **样本规模较小**：VQA仅90样本，虽经bootstrap验证稳定但仍需更大规模benchmark
4. **未分析否定检测与全图理解**：排除的5个全覆盖GT样本涉及absence detection/holistic recognition，其faithfulness模式有待系统研究
5. **未对比gradient-based saliency方法**：仅评估attention本身的忠实性，未与Grad-CAM等梯度方法做causal attribution对比

## 研究启发与可借鉴点
1. **Comprehensiveness+Sgap双维度量可直接迁移**：该方法已被证明在文本域有效，本文验证其在视觉token上的适用性，可作为VLM可解释性评估的标准协议供本团队复用
2. **Zero-ablation保留空间结构的设计值得借鉴**：替代token删除方案，避免position shift混淆内容与布局信息，适用于任何基于token序列的跨模态模型扰动实验
3. **Otsu两阶段阈值划分模式的方法论**：数据驱动的无监督聚类+人工边界修正策略，可推广至其他需要类型划分的可解释性研究
4. **跨架构对比揭示design choice影响**：Qwen（Faithful-Distributed）vs InternVL（Faithful-Sufficient）的差异源于视觉编码方式，提示模型压缩/加速方法需考虑architecture-specific注意力模式
5. **人机标注偏差发现对dataset construction的启示**：人类GT区域可能遗漏模型的隐性依赖路径，未来VLM训练数据标注可考虑加入"模型关注但未标注"的region

## 关键术语表
**Comprehensiveness（完备性）**：移除attention-ranked top-k token后预测概率的归一化下降幅度，衡量这些token是否必要；高值表示注意力忠实于因果证据。
**Sufficiency Gap（充分性缺口）**：仅保留top-k token后预测概率的归一化下降幅度；低/负值表示top-k alone足以恢复预测。
**Faithful-Sufficient Mode（忠实充分模式）**：top-k attention token既必要又充分（高Comp+低/负Sgap），约占32.9%，常见于局部object-centric问题。
**Faithful-Distributed Mode（忠实分布模式）**：top-k token必要但不充分（高Comp+高Sgap），约占47.1%，模型依赖超出焦点区域的更广上下文。
**Non-Focal Mode（非焦点模式）**：top-k token几乎不必要（低Comp）但整体视觉信息仍是预测触发器，约占20.0%，visual信息作为contextual coactivation trigger而非localized evidence。
**Zero-ablation**：将选定token的hidden state替换为零向量而非删除，以保持序列长度与空间位置结构不变。
**Otsu Thresholding**：基于类间方差最大化的自动阈值选取方法，本文用于将(Comp,Sgap)二维空间划分为三簇。
**Dynamic Tile Preprocessing**：将图像切割为多个固定分辨率tile分别编码后再concatenation的视觉预处理策略（如InternVL2.5采用）。

## 可复现要素
- **数据集**：VQAv2（公开）、VRDU（公开）、ChartQA（公开）；人类GT区域标注由作者通过COCO mask+人工修正生成（附录A详述流程）
- **代码/权重**：使用开源模型Qwen2.5-VL-7B/3B与InternVL2.5-8B/4B；具体代码开源情况论文未明确声明
- **关键超参**：temperature=0（greedy decoding）；扰动层位置=语言decoder首层入口；k值=GT区域token数（VQA）或1%/3%/5%/10%比例（文档任务）；Otsu两阶段阈值；bootstrap采样1000次
- **实现细节**：附录A包含GT标注流程与visual token grid说明，附录D提供InternVL跨模型实验细节
