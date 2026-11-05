---
title: "Staged-Continual-Adaptation-of-Multimodal-Foundation-Models"
source: https://arxiv.org/pdf/2610.27904v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 21:48:06"
---

# 论文速读：Staged-Continual-Adaptation-of-Multimodal-Foundation-Models

## 一句话总结
本文系统追踪了8.4B参数多模态基础模型在日本金融文档上的三阶段持续适配过程，发现各阶段（视觉-语言对齐、数学推理蒸馏、金融领域微调）之间存在非加性交互与能力trade-off：数学蒸馏会覆盖前期学到的金融声明性知识，而领域阶段更多是“修复”而非“新增”能力；因此最佳checkpoint高度依赖目标任务，未来设计杠杆应聚焦于阶段调度与参数隔离而非堆叠新阶段。

## 研究问题与动机
- **核心问题**：多阶段持续适配流水线中，各训练阶段的能力贡献是单调累加的，还是存在能力trade-off，导致不同任务的最佳checkpoint各不相同？
- **现有方法不足**：LLaVA式配方确立了视觉-语言对齐与多阶段微调的架构模板，但未深入刻画当目标领域要求结构化、布局依赖任务时，早期阶段与后期阶段的具体交互机制；非英语、数据稀缺的垂直领域专用多模态LLM仍极度匮乏。
- **测试床选择动机**：日本金融披露（如EDINET）同时要求监管知识、多步数值推理与日语能力，且现有SOTA LLM在纯文本配置下的表现仅略优于逻辑回归，是检验阶段交互与遗忘模式的理想场景。
- **评估设计约束**：本文统一采用纯文本输入基准（EDINET、JP Fin Harness、GSM8K）来探测主干模型（LLM backbone）的财务与数值能力演进，刻意剥离了在-modality页面阅读技能的评估（当前基准尚不支持，留待后续工作）。

## 核心贡献（创新点）
- **揭示非加性交互与不对称遗忘机制**：数学推理蒸馏（P2）会覆盖P1习得的金融声明性知识，但程序性推理技能对后续调优具有较强抗性；领域阶段（P3）的分数往往无法超越P1，表明其主要起修复作用而非能力安装。
- **提出端到端三阶段持续适配流水线**：以8.4B参数模型（SigLIP-v2 + llm-jp-4-8B-Instruct）为载体，依次执行视觉-语言对齐、XML结构化CoT蒸馏与金融领域适应，验证了流水线在日语金融场景的可行性。
- **构建可复现的政府PDF数据合成管线**：从日本内阁府、FSA、MOF公开PDF自动批量生成OCR、抽取式/推理式视觉QA与领域财务QA，全套合成数据与生成代码已开源，降低低资源垂直领域的数据门槛。
- **设计XML结构化思维链蒸馏配方**：利用Qwen3-30B-A3B-Thinking-2507作为教师，将中间推理过程封装于`<Problem>/<Thinking>/<Answer>`标签中，既定位错误传播节点，又契合金融文档多步数值演绎需求。

## 方法详解
- **基础架构**：视觉编码器SigLIP-v2 SO400M-patch14-384（~0.4B）+ 两层MLP投影器（Linear→GELU→Linear）+ 语言模型llm-jp-4-8B-Instruct（8B），总参数量约8.4B。
- **Phase 1 (P1) 视觉-语言对齐**：
  - P1-S1：仅训练投影器，冻结视觉编码器与LLM，使用STAIR Captions与政府PDF OCR数据建立图文嵌入对齐。
  - P1-S2：解冻视觉编码器与投影器，通过LoRA适配LLM；数据包含自政府PDF合成的抽取式/推理式视觉QA、JA-VG-VQA对话数据集。
- **Phase 2 (P2) 数学推理蒸馏**：动机为金融文档高频涉及跨表格数值校验与多步计算。从MGSM-ja、MathInstruct、OpenMathReasoning、OpenR1-Math-220k、orca-math-word-problems-200k、NuminaMath-1.5六个语料库随机采样约150K题目，由Qwen3-30B-A3B-Thinking-2507生成XML结构化CoT；使用QLoRA-SFT仅更新4-bit量化后的LLM，视觉部分全程冻结。
- **Phase 3 (P3) 金融领域适应**：通用数学蒸馏未覆盖金融术语与披露规范。P3-S1使用FinQA、ConvFinQA、TAT-QA进行纯文本金融基准的LoRA适配；P3-S2使用政府PDF合成的视觉领域QA，联合训练投影器与LLM的LoRA适配器；视觉编码器在P3全程冻结。
- **关键超参**：P1-S1 LR=1e-3, Epochs=2；P1-S2 LR=2e-5, LoRA(r,α)=(64,128), Epochs=1；P2 QLoRA LR=2e-4, (32,64), Epochs=1；P3 LR=5e-6, (16,32), Epochs=3；所有阶段Batch Size=2。

## 实验与结果
- **数据集与基准**：合成数据源自日本内阁府、FSA、MOF公开PDF；评测采用EDINET（fraud/earnings/industry）、JP Fin Harness（5个子任务）、GSM8K；全部为文本输入配置。
- **分阶段指标轨迹**：
  - P0→P1：JP Fin Harness (45.0→53.5)、GSM8K (58.5→64.3)、
