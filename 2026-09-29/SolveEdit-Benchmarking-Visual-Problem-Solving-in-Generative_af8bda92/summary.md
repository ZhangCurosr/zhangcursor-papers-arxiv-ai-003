---
title: "SolveEdit-Benchmarking-Visual-Problem-Solving-in-Generative"
source: https://arxiv.org/pdf/2609.35504v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 22:10:32"
field: "视觉生成与推理评测"
keywords: ["visual problem solving", "generative editing benchmark", "atomic transition contract", "SolveScore", "scene transformation", "IS/SD/RD regimes", "visual planning"]
innovations: ["提出 SolveEdit 基准，将变换信息来源显式划分为 IS/SD/RD 三类依赖 regime", "设计原子转换契约与 SolveScore，支持多解合法性并分离完成/破坏评估", "提出无参数两阶段规划器 SolveEdit-Plan，在单次生成预算下显著提升完成度与保持性"]
benchmarks: ["SolveEdit"]
---

# 论文速读：SolveEdit: Benchmarking Visual Problem Solving in Generative Models

## 一句话总结
本文提出 SolveEdit 基准，用于评估生成模型通过场景变换解决视觉问题的能力；引入基于原子转换契约的评估框架 SolveScore，并提出两阶段规划器 SolveEdit-Plan 以提升模型在 SD/RD 依赖任务上的完成度与保真度。

## 研究问题与动机
- 现有评测多孤立考察感知、生成或已明确指定的变换，缺乏对“从图像与目标中推断所需变换并执行”的目标驱动视觉问题解决能力的系统评估。
- 指令编辑类基准通常直接给出变换细节，无法分离“变换来源依赖”这一关键维度；引用基准单一参考输出也会误判多解合法性。
- 生成模型常在视觉上合理但任务语义/关系/规则层面失败，亟需能区分“解恢复错误”与“视觉执行错误”的评估机制。

## 核心贡献（创新点）
- 提出 SolveEdit 基准：将视觉问题求解形式化为场景变换任务，覆盖 2,728 案例、10 领域与 54 子领域，并按决定有效变换的信息来源划分 IS/SD/RD 三种依赖 regimes。
- 设计原子转换契约与 SolveScore：用 required/protected 条件替代单一参考输出，支持多解合法性，并将完成与附带破坏分开量化，提供六种诊断指标。
- 提出 SolveEdit-Plan 两阶段无参数规划器：在不修改编辑器的情况下，先推断并实例化隐式变换变量，再编译证据化编辑指令，显著提升完成度并降低无关内容扰动。

## 方法详解
- 任务形式化：给定输入图像 I 与请求 x，有效输出集合由Φ(x, B(I,x), K, S(I), R(I))决定；正确输出需同时满足“实现某合法规格z”与“保持无关内容”。
- 三种依赖 regime：IS（请求+实体绑定即可确定变换）、SD（需从场景状态推断缺失变量）、RD（需读取图中规则/图例/模式）。
- 原子契约评估：每个案例包含 required/protected 原子条件，每条条件判定 pass/partial/fail/abstain；按 SA/RA/VQ 三类属性与 required/protected 角色交叉形成六种诊断分数。
- SolveScore 公式：R 为加权完成分，D 为加权破坏分，最终得分为 G_quality·max(0, R - λD)，主协议取λ=0.5；质量门拒收缺失/空白/严重损坏/无关输出。
- SolveEdit-Plan：Inspect 阶段识别未解析变量并请求定向裁剪获取证据；Resolve 阶段比较候选变换并编译最终编辑指令，仅调用一次未修改编辑器。

## 实验与结果
- 评测范围：9 个图到图模型 + 2 个图到视频模型，统一在 2,728 案例上以相同契约评估；视频输出按最终帧计分。
- 最强基线：GPT-Image-2 取得最高 SolveScore 57.0%（R=67.6%, D=23.9%），Seedream 5.0 Pro 56.6%，Gemini 3.1 Flash Image 55.5%。
- 依赖 regimes 表现：IS→RD 诊断差距显著，最大下降出现在 required SA（-25.0 点）与 required RA（-13.0 点），protected VQ 近乎中性（+0.1），表明主要失分来源于解恢复而非保持能力。
- 规划干预效果：SolveEdit-Plan 使 GPT-Image-2 提升至 71.6%（Δ=+14.6），超过同预算 Generic Vision Rewrite 6.3 点；对 Qwen-Image-Edit-2509 从 20.8→26.9%，对 HunyuanVideo-1.5 从 7.4→14.0%。
- 成本：每例额外 2 次 LLM 调用（Inspect/Resolve），平均输入 3,207 tokens、输出 635 tokens。

## 相关工作脉络
- 视觉推理与图像编辑：InstructPix2Pix、ImagenHub、GIEBench、UIEBench、VIBE 等多聚焦指令遵循或单一参考对齐；SolveEdit 将“变换信息来源”显式为评估轴并允许多解。
- 生成内容评估：CLIPScore、GenEval、TIFA、V-Score 等侧重图像-文本对齐或单图质量；SolveScore 以契约化前后对比为核心，分离完成与破坏。
- 多模态推理与视觉生成：GenArtist、Visual ChatGPT、SmartEdit、UltraEdit 等采用多步工具或规划；SolveEdit-Plan 在单次生成预算内检验“前置显式推断变换”的增益。
- 视频生成评测：V-Bench、Video-MME、TiViBench 等关注时序能力；本文仅用终帧检验跨模态契约一致性，不评估时间推理。

## 局限性与未来方向
- 当前规划器无训练、依赖固定后端（GPT-5.6 Sol），跨架构泛化与效率仍有提升空间。
- 视频评测仅限终帧，未覆盖时序一致性与动态执行过程评估。
- 部分案例因证据不足会触发 abstain，覆盖差异可能影响可比性。
- 未来可扩展至更多视觉模态、引入训练型规划器，并探索自动契约生成与人类偏好对齐。

## 研究启发与可借鉴点
- 将“变换信息来源”作为正交评估轴，可迁移至其他生成任务的多解合规性评测设计。
- 原子契约+六种诊断分层的评分结构，便于定位模型在语义/关系/渲染层面的具体短板。
- SolveEdit-Plan 的“先推断后执行”范式适用于任何需要隐式规则/状态解析的视觉编辑 pipeline。
- 质量门与 abstain 透明报告策略，值得在开放基准中作为鲁棒性评估惯例。

## 关键术语表
- **SolveEdit**：面向视觉问题求解的场景变换评测基准，含 2,728 案例与三种依赖 regime。
- **IS/SD/RD regimes**：Instruction-Specified / State-Dependent / Rule-Dependent，区分有效变换的来源信息类型。
- **Atomic transition contract**：由 required 与 protected 原子条件组成的可观测评判单元集合。
- **SolveScore**：基于完成分 R、破坏分 D 与质量门 G_quality 的综合得分，λ=0.5 为主协议。
- **SA/RA/VQ**：Semantic Accuracy / Relational Accuracy / Visual Quality，三种诊断属性维度。
- **SolveEdit-Plan**：两阶段无参数视觉规划器，Inspect 收集证据、Resolve 编译指令并单次调用编辑器。
- **Quality gate G_quality**：拒绝缺失/空白/严重损坏/无关/不可用输出的前置过滤机制。
- **Solution-recovery vs. visual-execution error**：前者为选错合法解空间中的 z，后者为 z 正确但未正确渲染或破坏无关内容。

## 可复现要素
- 数据集：SolveEdit，2,728 案例；项目页面 https://wenjieshu.github.io/SolveEdit-project-page/ 提供浏览与示例。
- 代码：https://github.com/WenjieShu/SolveEdit（论文声明开源）。
- 权重/模型：评测覆盖商业与开源模型，使用各模型官方接口或发布版本；规划器使用 GPT-5.6 Sol。
- 关键超参：SolveScore 主协议λ=0.5；温度 0；重试上限 2 次；规划阶段每例最多 6 张裁剪图。
- 评测工具：VLM 为主，配合 SAM/YOLO 及 OCR 等专用检查器；分配策略对所有被测模型统一。
