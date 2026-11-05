---
title: "Staged-Continual-Adaptation-of-Multimodal-Foundation-Models"
source: https://arxiv.org/pdf/2610.27904v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 21:47:39"
---

# 论文速读：Staged-Continual-Adaptation-of-Multimodal-Foundation-Models

## 一句话总结
本文以8.4B参数多模态基础模型为对象，系统追踪其在日语金融文档适配中三阶段持续训练（视觉-语言对齐、数学推理蒸馏、领域微调）的能力交互，发现各阶段并非单调叠加：数学蒸馏会覆盖早期声明式金融知识，而领域阶段主要起修复作用，因此最优checkpoint高度依赖目标任务，建议通过调度与参数隔离而非盲目增加阶段来优化流程。

## 研究问题与动机
- **核心问题**：多阶段持续适配流程中，各训练阶段是否对最终能力呈单调贡献？还是会在阶段间引发能力权衡，导致不同任务的最优checkpoint不同？
- **现有方法不足**：LLaVA-style多阶段配方已确立架构模板，但未刻画早期阶段与后期阶段在结构化、布局依赖型任务上的交互机制；财务文档理解同时要求监管知识、数值推理与领域语言能力，是检验该问题的理想测试床。
- **低资源/非英语空白**：各国披露制度差异显著（如美国SEC 10-K、日本FSA申报），日语等低资源金融领域的专用多模态LLM研究仍较匮乏，亟需高效、可复现的训练管线。
- **评测盲区**：当前多模态基准尚未能充分探测in-modality page-reading skill，本文采用纯文本输入评测，聚焦于主干网络金融与数值能力的重塑轨迹。

## 核心贡献（创新点）
1. **揭示多阶段适配的非加性交互与不对称遗忘**：通过逐阶段探针证明程序性技能（数学推理）抗覆盖性强，而声明式知识（金融术语/概念）易被后续阶段覆盖，后续领域阶段更多是“修复”而非“新建”。
2. **提出端到端页面图像训练的三阶段持续适配流水线**：从视觉对齐到CoT蒸馏再到金融微调完整串联，验证8.4B小模型在特定金融任务上可逼近70B级外部模型，效率显著优于单体巨模型训练。
3. **构建可复现的政府PDF合成数据与XML-CoT蒸馏管线**：从内阁府/FSA/MOF公开PDF自动生成OCR、抽取/推理视觉QA与领域金融QA，并设计<Problem>/<Thinking>/<Answer>标签结构以定位中间推理错误，契合金融多步计算场景。
4. **建立任务依赖型checkpoint选择策略**：证明不同子任务的最佳阶段可能截然不同（如chabsa与industry在skip-P2最优，cpa_audit在P0最优），提示实际部署应保留多阶段快照并按下游任务动态切换。

## 方法详解
- **架构基座**：视觉编码器SigLIP-v2 SO400M-patch14-384（~0.4B）+ 两层MLP投影器（Linear→GELU→Linear）+ 语言模型llm-jp-4-8B-Instruct（8B），总参数量约8.4B。
- **Phase 1（视觉-语言对齐）**：P1-S1仅训练投影器，使用STAIR Captions与政府PDF合成OCR，冻结视觉与LLM；P1-S2解冻视觉与投影器，LLM通过LoRA（r=64, α=128, LR=1e-5）适配，使用合成的抽取/推理视觉QA、JA-VG-VQA对话数据集进行多模态指令跟随训练（视觉LR=2e-6）。
- **Phase 2（数学推理蒸馏）**：从MGSM-ja、MathInstruct、OpenMathReasoning、OpenR1-Math-220k、orca-math-word-problems-200k、NuminaMath-1.5中随机采样约150K题，由Qwen3-30B-A3B-Thinking-2507生成XML结构化CoT监督；采用QLoRA（4-bit NF4量化基座，r=32, α=64, LR=2e-4）仅训练LLM，视觉部分冻结。
- **Phase 3（金融领域适配）**：P3-S1使用FinQA、ConvFinQA、TAT-QA等文本金融QA，通过LoRA（r=16, α=32, LR=5e-6, Epochs=3）适配LLM；P3-S2引入政府PDF合成的视觉金融QA，训练投影器与LLM的LoRA适配器，视觉编码器全程冻结。损失函数为标准SFT交叉熵与蒸馏匹配损失，未引入额外对抗或正则项。

## 实验与结果
- **评测设置**：EDINET（fraud/industry/earnings，ROC-AUC或Accuracy）、JP Fin Harness（5子任务）、GSM8K；均为纯文本输入，temperature=0.7/top_p=0.95或greedy decoding。
- **逐阶段轨迹**：P0→P1全面提升JP Fin Harness（45.0→53.5）、GSM8K（58.5→64.3）、EDINET fraud（0.516→0.554）、EDINET industry（4.7→8.7）；P2使GSM8K升至73.8，但JP Fin Harness骤降至31.6，EDINET industry微降至7.9；
