---
title: "TriGeoBench-A-Multimodal-Mathematical-Geometry-Benchmark-wit"
source: https://arxiv.org/pdf/2610.27626v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-06 08:45:30"
field: "多模态数学推理"
keywords: ["multimodal reasoning", "geometric benchmark", "TikZ", "image-to-code", "mathematical reasoning", "structure-aware evaluation"]
innovations: ["提出首个基于中国初高中考试的几何中心多模态基准TriGeoBench，采用{LaTeX问答, RGB图像, TikZ代码}三元组统一表示", "设计Image-to-TikZ与多模态推理双任务评测协议，揭示当前MLLMs在细粒度结构对齐上的瓶颈", "通过Oracle实验证明人工标注TikZ可提供互补几何线索，但多模态冗余输入融合仍具挑战"]
benchmarks: ["TriGeoBench", "MathVerse", "MATH-Vision", "VisioMath", "GeoEval", "SOLIDGEO"]
---

# 论文速读：TriGeoBench-A-Multimodal-Mathematical-Geometry-Benchmark-wit

## 一句话总结
本文提出了 TriGeoBench，一个源自中国初高中数学考试的几何中心多模态基准，包含 7,000 个问答对和 12,235 个带 TikZ 代码的高质量几何图形；通过 Image-to-TikZ 转换和多模态推理评测，揭示当前 MLLMs 在细粒度几何结构与图文对齐方面的持续挑战。

## 研究问题与动机
1. **现有数学基准缺乏精确的图形结构表示**：大多数基准依赖光栅化图像，无法显式保留拓扑、几何关系和符号标注，导致无法判断模型是否真正基于几何结构推理还是依赖表层视觉模式。
2. **几何知识覆盖与题型多样性不足**：现有数据集通常只涵盖平面/立体几何的子集，问答形式单一，难以全面评估几何推理能力。
3. **多模态几何推理评估的局限性**：当前 MLLM 在结合文本、符号与视觉图形的数学推导方面表现仍不充分，亟需统一的评测框架来诊断结构与对齐能力。

## 核心贡献（创新点）
1. **提出 TriGeoBench 基准**：构建了一个从真实中国初高中数学考试来源的大规模几何基准，涵盖平面几何、立体几何、解析几何、函数和向量几何等多个知识领域，以及单选、填空、证明和问答等多种题型。
2. **首创三元组统一表示**：每个实例表示为 {LaTeX 问答对, RGB 图像, TikZ 代码} 三元组，TikZ 保留了可编译的几何拓扑、符号标注和关系结构，优于单纯光栅图像。
3. **设计统一的双任务评测协议**：提出 Image-to-TikZ 和 Multilingual Mathematical Reasoning 两种评测范式，前者评估图形到可编辑代码的转换能力，后者评估图文融合的几何推理能力。
4. **揭示 MLLMs 在几何推理中的结构对齐瓶颈**：实验发现开源模型在中等/困难样本上 Human Acc 极低，且自动指标与人类评估之间存在显著差距，说明几何正确性难以被现有自动度量完全捕捉。
5. **Oracle 诊断证明结构表征的补充价值**：引入人工标注参考 TikZ 的实验表明，准确的执行代码可提供互补的几何线索，但当前模型难以有效融合冗余的视觉与符号输入。

## 方法详解
**数据构建流程**：
- 来源收集：从中国初高中数学公开考试、学校库和开放档案中提取含图的几何问题，包括题干、答案、解答和对应图形。
- 预处理：标准化数学符号、去除无关元数据、近重复过滤、与公共基准交叉比对以减少泄漏风险。
- 人工转换： annotators 将题干转录为 LaTeX，参照原图编写 TikZ 代码，保留隐藏边、辅助线、点标、相对位置等细节。
- 独立验证：双人校对机制检查数学符号正确性、语义完整性、图文一致性和 TikZ 编译性。

**数据组织维度**：
- 知识点分类：平面几何 (PG)、立体几何 (SG)、解析几何 (AG)、函数 (Fn)、向量几何 (Vec)。
- 题型分类：单选、填空、证明、解答题。
- 难度分层：基于参考解答长度和 TikZ 代码长度进行 K-means 初始化，再由教育专家校准。

**评测协议**：
- Image-to-TikZ：评估四个指标——Gen Acc（代码编译成功率）、Image Metrics（SSIM/KID/CLIP/DINO 视觉一致性）、TikZ Metrics（cBLEU/MTE/TED 结构相似性）、Human Acc（人工评判几何保真度）。
- Multimodal Reasoning：四种输入设置——Text-only、Text+Figure、Text+TikZ、Text+Figure+TikZ，使用 LLM-as-Judge 进行评分（Pearson r = 0.629, 99.0% within-one-point agreement）。

**小模型微调实验**：
- 使用 LoRA-SFT 在 Qwen3-VL 系列、GLM-4.1V、Llama-3.2、Gemma3 等小规模模型上训练，发现微调显著提升 TikZ Metrics 和 Human Acc，但对 Gen Acc 的影响不一致（部分模型下降）。

## 实验与结果
**测试集**：从 TriGeoBench 采样 1,000 实例（500 Easy、300 Medium、200 Hard），其余用于小模型训练。

**Image-to-TikZ 结果**：
- 闭源模型 Gemini-3.1-Pro 最佳：Gen Acc 96.1%，Human Acc 73.7%。
- GPT-5.4 Gen Acc 95.6% 但 Human Acc 仅 43.9%，揭示自动指标与几何正确性的差距。
- 开源模型整体表现较弱：GLM-4.5V-108B Human Acc 7.5%，Qwen2.5-VL-72B 12.8%。
- 难度梯度上，Medium/Hard 样本 Human Acc 显著下降，尤其开源模型。

**推理评测结果**：
- Seed-2.0-Pro Text-only 达 81.76%，为最高；但加入参考 TikZ 后并未普遍提升，Text+Figure+TikZ 未总能达到最优。
- Gemini-3.1-Pro 从 Text-only 49.45% 下降到 Text+Figure 43.00%，说明多图模态融合仍具挑战。
- Oracle Text+TikZ 对几何相关知识点和证明类题目增益最大。

**跨基准探索**（Table 5）：
- Seed-2.0-Pro 在 MathVerse (+13%)、MATH-Vision (+9%)、VisioMath (+3%) 上均有提升，但 VisioMath 增益有限。

## 相关工作脉络
1. **数学推理基准**：MATH (Hendrycks et al., 2021) 纯文本 → MathVista (Lu et al., 2024)、MATH-Vision (Wang et al., 2024)、MathVerse (Zhang et al., 2024) 引入视觉 → TriGeoBench 进一步强调结构化 TikZ 表示。
2. **几何专用基准**：VisioMath (Li et al., 2026)、GeoEval (Zhang et al., 2024)、SOLIDGEO (Wang et al., 2025b) 关注细粒度图示理解和空间推理，但均无 TikZ 编辑能力。
3. **结构增强几何**：Geometry3K (Lu et al., 2021) 使用形式语言描述图文；GeoQA/GeoQA+/UniGeo 引入程序/证明注解；PGPS9K 标注细粒度几何基元和子句；本文的 TikZ  triplet 统一了生成与推理。
4. **图像到图形程序合成**：DeTikZify (Belouadi et al., 2024)、AutomaTikZ (基于 DaTikZ)、TikZero (基于 DaTikZ v3)、GeoGramBench (Luo et al., 2025)、InternSVG (Wang et al., 2025) 侧重图形重建而非推理；TriGeoBench 连接了生成与结构化推理。
5. **向量图形理解**：DeepSVG (Carlier et al., 2020)、VGBench (Zou et al., 2024)、OmniSVG (Yang et al., 2025) 研究 SVG 理解；TriGeoBench 聚焦数学几何领域并兼顾推理评估。

## 局限性与未来方向
1. **语言与文化局限性**：基准聚焦中国初高中考试，评测分数混杂中文理解、数学推理与视觉能力，难以直接迁移至其他语言或课程体系。
2. **训练数据泄漏风险**：尽管进行了去重，但无法完全排除模型在预训练中接触过公开考试数据的可能性。
3. **Oracle 设定非端到端**：参考 TikZ 为人工标注的诊断性输入，并非自动转换管线，实际部署性能可能更低。
4. **等价 TikZ 编程歧义**：同一几何图形可有多种合法 TikZ 表示（不同坐标系、绘制顺序、宏），自动代码指标难以完全捕捉语义等价性。
5. **未来方向**：评估自动化 Image-to-TikZ 管线、扩展数据集覆盖范围、强化结构感知评估协议、探索多模态冗余输入融合机制。

## 研究启发与可借鉴点
1. **三元组统一表征设计**：{文本, 图像, 结构化代码} 的 triplet 范式可用于其他需要精确空间关系的领域（如物理仿真、化学分子式、电路图理解）。
2. **多层级评估指标组合**：结合可执行性 (Gen Acc)、视觉一致性 (SSIM/CLIP/DINO)、代码结构相似性 (cBLEU/TED) 和人工评判 (Human Acc) 的四层评价体系，对图形生成任务有普适参考价值。
3. **Oracle 诊断策略**：通过提供 ground-truth 结构化输入评估其对推理的增益，可有效分离"理解能力"与"表征获取能力"，适用于其他模态融合研究。
4. **难度分层与专家校准机制**：基于统计特征（代码长度、解答长度）初筛 + 教育专家人工校准的分层策略，保证了难度标签的教育学意义，可迁移至其他教育相关基准构建。
5. **小模型 LoRA-SFT 实验设计**：在控制超参（学习率、rank、epoch）下进行公平比较，发现 SFT 对格式约束任务的不一致性，提示未来需引入编译感知训练信号（如 compilation reward）。

## 关键术语表
**TikZ**：基于 LaTeX 的矢量图形描述语言，可编译生成精确的几何图示，保留拓扑与标注信息。
**Image-to-TikZ**：将光栅化几何图形转换为可编辑 TikZ 代码的任务，评估模型对图形结构的理解与重建能力。
**Figure-grounded reasoning**：指解题依赖图中非装饰性的几何信息，该信息与题干相互补充而非冗余。
**Oracle experiment**：使用人工标注的参考 TikZ 代码作为输入的诊断实验，评估结构表征对推理的上限增益。
**Gen Acc (Generation Accuracy)**：生成的 TikZ/LaTeX 代码成功编译的比例，衡量语法正确性与可执行性。
**Human Acc**：人工评判生成的图形在视觉与数学意义上与参考答案一致的比例，是几何保真度的关键指标。
**LLM-as-Judge**：使用固定 prompt 的大语言模型作为评分器，对模型答案进行结构化评分与解释。
**cBLEU / TED / MTE**：Code BLEU（词级别重叠）、Tree Edit Distance（语法树编辑距离）、Token MTE（token 级 F1）等代码结构相似性度量。

## 可复现要素
- **数据集**：TriGeoBench 包含 7,000 问答对与 12,235 图形，按原始来源许可发布；论文提供数据说明文档。
- **代码/权重**：论文未提及开源代码或预训练权重；微调实验使用 LoRA 配置，但未公开详细脚本。
- **关键超参**：LoRA fine-tuning 统一配置（learning rate、rank、epochs）论文未明确给出数值，需参考附录或补充材料。
- **评测代码**：评估指标（SSIM、KID、CLIP、DINO、cBLEU、TED 等）公式已在附录 C 给出，可独立复现。
- **Prompt**：LLM-as-Judge 评估 prompt 模板在附录 F 提供。
