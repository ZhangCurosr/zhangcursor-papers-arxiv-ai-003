---
title: "ZOOMING-WITHOUT-ZOOMING-V2-FROM-SEEING-DETAILS-TO-CONNECTING"
source: https://arxiv.org/pdf/2610.37037v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-06 09:03:30"
field: "多模态感知与视觉推理"
keywords: ["multimodal perception", "visual evidence connection", "synthetic data synthesis", "RLVR", "spatial reasoning", "long-context understanding"]
innovations: ["提出connecting evidence作为统一原子能力并用多区域单图合成数据训练", "EVIDENCECONNECT框架结合特权视角与跨模型验证+区域消融审计", "仅20.4K合成数据经RLVR即实现25基准全面泛化提升"]
benchmarks: ["ViewSpatial-Bench", "MileBench", "MMLongBench", "TreeBench", "CountQA", "MIMIC", "MMSI-Bench", "ERQA", "AgentVista", "ZoomBench", "HR-Bench", "VStar", "MME-RealWorld", "Chartography", "PerceptionBench", "MMStar", "BabyVision", "MemLens", "ScreenSpot-Pro", "UI-Vision", "O3-Bench", "SPAR-Bench", "MindCube-Tiny"]
---

# 论文速读：ZOOMING-WITHOUT-ZOOMING-V2-FROM-SEEING-DETAILS-TO-CONNECTING

## 一句话总结
论文提出 EVIDENCECONNECT 数据合成框架，通过单图像内多区域（3–7个）的组合式问题合成"连接证据"训练数据，并结合 RLVR（GRPO）训练 Qwen3.5/3.6 系列模型（ZwZ2），在 25 个基准上显著提升细粒度感知、空间推理、多模态长上下文理解及具身智能任务性能，且无需领域特定微调。

## 研究问题与动机
- **空间感知与长上下文理解的合成数据稀缺**：这两项任务是具身 AI 和多模态智能体的基石，但高质量的靶向训练数据难以低成本批量构建——空间感知需要精确多目标定位与关系标注，长上下文需要多样可回答的多图/多轮问答数据。
- **现有方法局限**：已有工作多集中于单区域细粒度感知（如 ZwZ 的 zoom-in 蒸馏）或特定格式的多图指令数据，缺乏能统一训练"跨区域/跨上下文证据连接"能力的可验证监督信号。
- **共享原子能力假设**：空间感知、长上下文理解、具身推理等任务底层共享"连接证据"（connecting evidence）能力——识别视觉事实、绑定至正确实体、恢复关系并组合为答案；针对该能力的训练数据远比对目标任务本身更容易合成与验证。
- **验证"单图多区域"监督是否能泛化**：若能在单张图像内合成高质量的 multi-region 组合问题并加以严格验证，能否让模型在无需直接接触多图/长序列训练数据的情况下，提升面向多图像和交互场景的感知能力？

## 核心贡献（创新点）
1. **提出"连接证据"作为统一原子能力概念**：首次将细粒度感知、空间感知、多模态长上下文理解统一于"识别-绑定-恢复-组合"四步证据链，为训练目标抽象提供理论依据；与以往仅针对单区域或单一任务的合成方法本质不同。
2. **设计 EVIDENCECONNECT 全自动多区域问题合成框架**：通过特权视角（privileged views）、跨模型答案锚定（cross-model answer anchoring）与自适应 drop-one 区域消融测试构建可审计的训练样本；相比 ZwZ 的单区域 zoom-in 范式，覆盖了聚合、比较、依赖链三类组合推理形式。
3. **仅凭 20.4K 合成数据即实现跨域泛化**：在 Qwen3.5-4B/9B 与 Qwen3.6-27B 上使用 RLVR（GRPO）纯强化学习微调（无 SFT），ZwZ2 在 25 个基准的 6 大领域平均提升 4.96–5.39 分，并在具身空间推理、多模态 agent 任务及长上下文任务上显著受益；与当前先进闭源模型（GPT-5.6-Luna/GLM-5.3-Flash/DeepSeek-V4.1-Flash）可比肩甚至领先细粒度感知单项。

## 方法详解

### 1. 多区域问题合成流水线
**目标问题结构**：从源图像 I 定位区域提议 $R_I$，选取一个证据组 $G = \{r_{i_1}, \ldots, r_{i_k}\} \subseteq R_I$，$3 \leq k \leq 7$。每个区域被赋予特定角色：提供数值、构成候选集成员、或作为后续步骤依赖线索。问题按三类形式组织：
- **聚合（Aggregation）**：对多个区域的计数/数值做求和或简单算术。
- **比较与匹配（Comparison & Matching）**：将区域作为封闭候选集，识别异常值、极值或配对。
- **依赖链与组合（Dependency Chains & Compositions）**：多步推理，如用线索决定下一步读取、固定视角解释其余区域等。

**证据发现**：使用强 VLM（Qwen3.5-397B-A17B）对高分辨率复杂图像进行 grounded 区域提议，优先选择可相互关联的区域，过滤过小、过大或重叠度过高（IoU ≥ 0.65）的框。

**任务条件分组**：文本-only 调用将区域按 22 种 task family（如 anomaly_detection、spatial_chain 等）和 4 种 necessity mode（answer_sensitive / set_complete / pair_complete / chain_dependency）分组。

**特权视角布局（Privileged Evidence Layout, $V_G$）**：按原始分辨率裁剪所选区域（边距 8%），在保留相对行列顺序的紧凑画布上拼接；每区域赋予不同颜色轮廓（7 色 palette）；该视图仅用于数据构建，训练时模型不可见。

**问题生成**：VLM 基于 $V_G$、区域描述及 104 条推理模板库中的 blueprints 生成候选问题与 reasoning certificate（逐步链接区域到答案的确定性程序）。通过确定性规则拒绝证书不完整、答案可绕过某区域、含主观措辞或颜色泄露答案的样本。

### 2. 两阶段验证
**跨模型答案锚定（Cross-model Answer Anchoring）**：两个不同家族 VLM（如 Qwen3.5-397B-A17B 与 Kimi-K2.6）在 $V_G$ 上各自回答≥4次，无参考答案提示；仅当每个模型的大多数响应收敛到同一答案时才保留，该答案成为参考 $a^\star$。

**自适应 Drop-one 测试（Adaptive Drop-one Testing）**：依次将每个区域在 $V_G$ 中替换为灰色占位，验证器重新回答；若多数查询仍恢复 $a^\star$ 或返回不可解析响应，则标记该区域"经验性冗余"。仅当>1个区域冗余时才拒绝整题，以容忍 crop 边界重叠噪声。

### 3. 数据终态化
- **策略视图（Policy View, $V_I$）**：对聚合类问题尝试去除颜色标记的自然语言重写（由另一家族 VLM 改写并通过独立审计确认唯一参考与任务保持）；其余保留带彩色轮廓的全图。
- **四选项多选题转换**：生成三个同格式 distractor，打乱选项顺序，标签为 A/B/C/D，适配 rule-based reward。
- **难度过滤**：轻量 VLM（Qwen3.6-35B-A3B）回答4次，全对或含不可解析响应的样本丢弃，最终保留难度分布：easy 33.75% / medium 22.04% / hard 19.07% / very hard 25.14%。

### 4. RLVR 训练
使用 GRPO（Shao et al., 2024）在 20.4K 样本上训练 2 epoch，batch size=128，lr=$10^{-6}$，每 prompt 采样 G=8 次 rollout（temperature=0.7，max length=32K），启用 decoupled clipping（$\epsilon_{low}=0.2, \epsilon_{high}=0.3$），移除 KL 惩罚（$\beta_{KL}=0$），reward 为选项字母是否匹配。生成模式关闭 thinking。

## 实验与结果

**数据集与基准**：25 个基准覆盖 6 大领域（细粒度感知、空间感知、多模态长上下文理解、具身空间推理、多模态 agent、通用视觉推理）。训练数据源自 SA-1B 图像库，经多层预筛选（长边≥1280px、熵≥6.5 bits、Canny 边缘密度≥0.04、亮度标准差≥60）。

**主要结果（Table 1）**：
- **4B**：Overall 从 49.41 → 54.79（+5.38）；细粒度 Avg +5.85；空间 Avg +6.34；具身空间 Avg +6.50；长上下文 Avg +5.61；Agent Avg +2.75；通用推理 Avg +2.62。
- **9B**：Overall 从 51.63 → 56.59（+4.96）；**27B**：Overall 从 55.52 → 60.51（+4.99）。
- 代表提升：ViewSpatial-Bench（4B）37.43→45.26（+7.83）；MileBench（4B）60.08→70.08（+10.00）；ERQA（27B）51.25→61.75（+10.50）；MIMIC（4B）55.01→66.10（+11.09）；CountQA（27B）46.73→54.32（+7.59）。
- ZwZ2-27B 在 Overall 60.51% 超越 GPT-5.6-Luna（54.02）、GLM-5.3-Flash（57.24）、DeepSeek-V4.1-Flash（55.47）。

**与现有合成数据对比（Table 2，Qwen3.5-4B）**：
- EVIDENCECONNECT（20.4K）在所有 11 个评测上优于 ZwZ（73K）、TVG（19.4K）、F²RVLM（37K）、CRIT（30K）、MMHops（31K），整体平均 53.86 最高；其余数据集在部分基准上低于 base（如 MMHops 整体 42.73）。
- 相比 ZwZ-73K（3.6× 数据量），EVIDENCECONNECT 在 10/11 基准领先，MindCube-Tiny 上 ZwZ 下降 13.8 分而本方法提升 5.4 分，印证"跨区域连接"对具身空间推理的关键性。

**消融（Table 3，7K 样本）**：
- 去除 drop-one 验证：Avg 从 52.65→51.03（-1.63）；去除 privileged views：Avg 从 52.65→51.38（-1.27）。两者均主要改善空间与具身任务。

**泛化与尺度分析（Figure 3–4）**：
- Qwen3-VL-2B/4B 同样获得 +4.81 / +2.91 提升。
- 数据规模 7K→20.4K 带来 +1.21 额外增益。
- 长上下文（Figure 5）：MemLens 在 256K token 仍保持增益；MMLongBench 在 128K 增益收窄，提示当前单图短上下文训练对超长检索帮助有限。
- 注意力对齐（Figure 6 / Table 5）：ZwZ2 在 TreeBench 405 例上，8 层中 4B 有 7 层、9B 有 6 层 mean top-k IoU 提升（最大 +0.53pp），表明训练后高注意力 token 与 ground-truth 证据空间对齐更好。

## 相关工作脉络
1. **Zooming without Zooming (ZwZ, Wei et al. 2026)**：单区域 zoom-in 蒸馏方法，将高分辨率局部视图的知识迁移至全图；本文将其扩展为多区域组合推理，覆盖聚合/比较/链式三类任务，并引入跨模型验证与区域消融审计。
2. **TVG (Zhang et al. 2026a)、F²RVLM (Bi et al. 2026)、CRIT (Sung et al. 2026)、MMHops (Zhang et al. 2026b)**：均面向特定任务（如跨模态多跳、对话片段检索）合成数据；本文强调"连接证据"这一共享能力，在单图内构建可验证的 multi-region 问题，泛化性更强。
3. **多跳空间推理 (HopChain, Wang et al. 2026; MultihopSpatial, Lee et al. 2026)**：通过人工验证或外部知识路径构建多跳 QA；本文方法完全自动化，不依赖人工标注与外部知识，并通过 region ablation 保证因果依赖。
4. **长上下文多模态理解 (MMLongBench, MileBench 等)**：已有数据集集中在多图/多轮指令格式；本文证明仅通过单图多区域训练即可迁移至此类任务，降低训练数据构建成本。
5. **空间感知与具身推理 (SpatialVLMS、ViewSpatial-Bench、MindCube)**：先前工作多依赖大规模空间标注或几何代理任务；本文以自然图像为起点，通过区域连接训练提升模型空间关系建模能力。
6. **RLVR / GRPO 多模态训练 (DAPO, DeepSeek-R1)**：本文沿用 GRPO 强化学习范式，但创新点在于合成数据的结构设计与验证机制，而非算法改进。

## 局限性与未来方向
- **训练数据未建模时序/跨帧依赖**：当前合成数据局限于单图，难以直接刻画视频或多帧间证据流转，未来可扩展至跨帧视频合成。
- **模型规模上限 27B**：更大参数模型（如百 B 级）是否同样受益尚未验证；前置阶段（预训练、SFT）是否也能从该类数据中获益仍需探索。
- **极长上下文增益收窄**：在 128K token 以上长度，MMLongBench 提升减弱，说明当前训练强化了"连接证据"能力但对"超长检索"本身改善有限，需结合更长序列训练或检索增强。
- **privileged view 与 color outline 可能引入分布偏移**：训练时模型看到的彩色轮廓是合成时的人工标记，真实场景中无此信号，尽管 unmarked 重写缓解了这一偏移。

## 研究启发与可借鉴点
1. **"特权视角+审计验证"的数据合成范式**：privileged view 仅在构建阶段使用、训练阶段隐藏的设计，兼顾了问题生成的准确性与模型推理的真实性，可迁移至其他视觉合成任务。
2. **跨模型答案锚定 + 区域消融的联合验证**：用不同家族 VLM 的多数投票确保答案正确性，再用 drop-one 测试确保因果必要性，形成闭环质量保障，减少"捷径学习"（shortcut learning）。
3. **将"共享原子能力"作为训练目标**：识别多任务底层共性（如证据连接），用易合成的数据统一训练，可避免为每个下游任务单独构建昂贵标注数据。
4. **单图多区域问题可迁移至多图/长上下文**：实验证实单一图像内的 multi-region 训练可提升多图像 benchmark，为低成本泛化训练提供新思路。
5. **注意力对齐评估（Top-k Evidence IoU）**：通过计算模型注意力与 ground-truth 证据盒的重叠度，提供可解释的训练效果诊断指标，可复用于其他感知模型分析。

## 关键术语表
- **Connecting Evidence**：VLM 的原子能力，指识别视觉事实、绑定至正确实体、恢复实体间关系并组合为答案的四步过程。
- **EVIDENCECONNECT**：论文提出的全自动多区域视觉问题合成框架，包含证据发现、分组、特权视图构建、验证与数据终态化全流程。
- **Privileged View ($V_G$)**：仅用于数据构建阶段的高清区域拼接视图，训练时模型不可见，用于生成高质量组合问题。
- **Policy View ($V_I$)**：实际送入策略模型训练的视图，通常为带彩色轮廓标记的全图或经自然语言重写的无标记图。
- **Cross-model Answer Anchoring**：用两个不同家族 VLM 分别多次回答以多数投票确定参考答案 $a^\star$ 的验证机制。
- **Adaptive Drop-one Testing**：依次掩码证据组中某一区域，验证答案是否依赖该区域，用于检测装饰性区域或捷径推理。
- **RLVR (Reinforcement Learning with Verifiable Rewards)**：利用可判定奖励（如选项字母匹配）进行强化学习训练的方法，本文采用 GRPO 算法。
- **Necessity Mode**：问题对区域依赖关系的类型标注，包括 answer_sensitive（聚合）、set/pair_complete（比较匹配）、chain_dependency（依赖链）。

## 可复现要素
- **数据集**：训练数据源自 SA-1B，合成代码与模型权重已开源（Code: inclusionAI/Zooming-without-Zooming-v2，Hugging Face collections/inclusionAI/zooming-without-zooming-v2）。
- **代码**：已开源。
- **关键超参**：GRPO 训练 2 epoch，batch size=128，lr=$10^{-6}$，G=8 rollouts，temperature=0.7，max response length=32K，decoupled clipping $\epsilon_{low}=0.2$、$\epsilon_{high}=0.3$，$\beta_{KL}=0$。
- **模型调用**：主模型 Qwen3.5-397B-A17B（thinking off），验证器含 Kimi-K2.6、GLM-5.2、Qwen3.6-35B-A3B。
