---
title: "The-More-the-Better-A-Comparison-of-Single-and-Multi-Agent-L"
source: https://arxiv.org/pdf/2610.27488v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 21:50:13"
field: "多Agent LLM系统评测"
keywords: ["LLM agents", "multi-agent systems", "single-agent baseline", "task decomposition", "defect detection", "LLM-as-a-judge", "cost-aware evaluation"]
innovations: ["最简多Agent对比设计（无角色/编排/过滤）分离任务划分与信息隔离效应", "以单次采样多次投票作为公平基线揭示重复采样优于多Agent划分", "量化Scoped Team对误报的缓解效果并揭示指令对结果的调制作用"]
benchmarks: ["UCI Credit Card Dataset (synthetic credit memos)", "300 memos / 1800 cells with known ground truth labels"]
---

# 论文速读：The-More-the-Better-A-Comparison-of-Single-and-Multi-Agent-L

## 一句话总结
本文在金融信用评估备忘录审查任务上，以受控实验比较了单 Agent 与多 Agent（同模型）在相同 prompt 条件下对缺陷的检测能力，发现多 Agent 方案以误报率为代价换取更高的漏检召回率，两者权衡后平衡准确率几乎无变化。

## 研究问题与动机
1. 业界指南提出"复杂任务将工作分配给多个 LLM 调用通常表现更好"（Schluntz & Zhang, 2024），但缺乏数据支撑，尤其缺少单 Agent 作为公平基线。
2. 既有研究多关注最终正误评分，无法同时报告漏检率（miss rate）与误报率（false-alarm rate），无法刻画两种错误的trade-off。
3. 多 Agent 研究常同时改变角色描述、编排器、过滤层或不同模型，难以分离"任务划分"本身的效应。
4. 缺少让单 Agent 获得与团队相同调用次数的公平基线（如多次采样），无法判断多 Agent 增益是否仅源于更多调用。

## 核心贡献（创新点）
1. **最简多 Agent 对比设计**：三个 Agent 在同一模型上、相同 prompt 下各检查两条要求，无角色分工、无编排器、无后续过滤阶段——与单 Agent 唯一区别是任务划分方式。
2. **分离"任务划分"与"信息隔离"效应**：通过 Scoped Team 臂（向每个 Agent 展示其余要求用于上下文但不得判定），首次在同一实验中量化"未看到其他要求"导致的误报增加。
3. **公平的单 Agent 多次采样基线**：将单 Agent 采样三次（投票规则）与三 Agent 团队在调用次数上对齐，发现单次采样的任意一票规则（any-of-three）在 Claude 模型上平衡准确率高于团队。
4. **揭示指令（instruction）对效果的调制作用**：主条件采用 "decide-once" 指令强制每次判定一次；移除该指令后，Qwen3.5 27B 的团队增益消失而误报激增，表明结果高度依赖指令设置。
5. **提供误报率的清洁基准测量**：在已知无缺陷的单元格上单独报告误报率，此前相关研究未见此类测量，使 trade-off 可量化。

## 方法详解

**任务与数据**：
- 使用 UCI Credit Card Dataset（Yeh & Lien, 2009）生成 300 份合成信用评估备忘录，每份约 1,100 词、8 个章节。
- 6 条检查要求（R1–R6），每条要求独立标注，共 1,800 个单元格（cell），其中 1,136 个干净、37% 含缺陷。
- 三种缺陷类型：裸缺陷（bare，无解释的直接矛盾）、掩蔽缺陷（masked，带解释但仍与其他事实矛盾）、算术缺陷（arithmetic，计算错误）。

**四种实验臂**：
1. **Single agent**：$f(d_i, R, \emptyset)$，一次调用处理全部 6 条要求。
2. **Team**：$f(d_i, P_m, \emptyset), \; m=1,2,3$，三个 Agent 各处理 2 条要求，不查看其余 4 条。
3. **Scoped team**：$f(d_i, P_m, R \setminus P_m)$，每个 Agent 除了自己负责的 2 条外，还看到其余 4 条（仅作为上下文，不得判定）。
4. **Three draws**：单 Agent 的同一 prompt 采样 3 次，投票规则 $t=1/2/3$（任意一票/多数票/全票）。

**评估指标**：
- 漏检率 $H_a = \frac{\sum y_{ij} v_{ij}^a}{\sum y_{ij}}$（检出缺陷的比例）
- 误报率 $F_a = \frac{\sum (1-y_{ij}) v_{ij}^a}{\sum (1-y_{ij})}$（干净单元格中被误判为缺陷的比例）
- 平衡准确率 $B_a = (H_a + 1 - F_a)/2$
- Token 消耗 $K_a$

**模型与设置**：
- Qwen3.5 9B/27B（本地 Ollama，4-bit quantization，temp=0，抽样 temp=0.7），Claude Haiku 5.5/Sonnet 5.5（Batch API）。
- 关闭模型独立思考通道，输出上限 2,048 token/要求，JSON Schema 输出。
- "decide-once" 指令：要求每个判定最多 3 句推理，不得回头修改。
- 主要预测经 Bonferroni 校正（98.75% 置信水平），次要预测 95% 水平。

## 实验与结果

**数据集**：300 份合成信用评估备忘录（1,800 个单元格），标签由生成程序、独立重算程序、Claude Opus 5.5 三方核对。

**主要结果（decide-once 指令下）**：

| 模型 | 臂 | H（总缺陷检出） | F（误报率） | B（平衡准确率） | Token 比 |
|------|-----|----------------|------------|----------------|---------|
| Qwen3.5 9B | Single | 0.613 | 0.160 | 0.726 | 1.00 |
| Qwen3.5 9B | Team | 0.733 | 0.254 | 0.740 | 2.36 |
| Qwen3.5 27B | Single | 0.792 | 0.047 | 0.873 | 1.00 |
| Qwen3.5 27B | Team | 0.834 | 0.076 | 0.879 | 2.24 |
| Haiku 5.5 | Single | 0.825 | 0.013 | 0.906 | 1.00 |
| Haiku 5.5 | Team | 0.877 | 0.025 | 0.926 | 2.16 |
| Sonnet 5.5 | Single | 0.994 | 0.000 | 0.997 | 1.00 |
| Sonnet 5.5 | Team | 0.991 | 0.013 | 0.989 | 2.17 |

**关键发现**：
- **Masked 缺陷检出提升**：在 Qwen3.5 27B 和 Haiku 5.5 上，团队比单 Agent 多检出 0.08–0.10 的掩蔽缺陷（校正水平显著）；Qwen3.5 9B 估算相同但区间更宽。
- **误报率上升**：团队在全部四个模型上均产生更多误报，校正水平在三个模型上显著。
- **平衡准确率变化≤0.02**：两种错误大致抵消。
- **Scoped Team**：在两个 Claude 模型上显著降低误报（Sonnet 5.5 校正水平显著），且仍保持对掩蔽缺陷的增益（Haiku 5.5）。
- **Three draws（any-of-three 规则）**：在全部四个模型上检出缺陷多于团队，平衡准确率高出 0.011–0.024（Claude 模型显著）。
- **指令效应**：移除 "decide-once" 后，Qwen3.5 27B 团队的缺陷检出增益消失（+0.004），误报大幅增加（+0.123），平衡准确率显著下降。

## 相关工作脉络
1. **Tran & Kiela (2026)**：单 Agent 在多跳推理上与多 Agent 对比，单 Agent 整体胜出——本文补充说明其结论取决于任务类型与评估方式。
2. **He et al. (2026)**：金融文档上全类型一次性查询 vs 逐类型查询，发现分类型提升召回但降低精确率，与本文 trade-off 方向一致。
3. **Liu et al. (2026)**：无过滤阶段的多 Agent 团队提升召回（58.7%→75.0%）但降低精确率（91.5%→74.2%）——本文在此基础上进一步分离"任务划分"与信息隔离的效应。
4. **Yu et al. (2026)**：将分任务指令给单 Agent 在同一上下文中评估，发现低于完整指令单 Agent——本文指出该设计改变了单 Agent 而非控制性地对比团队。
5. **Li et al. (2024)**："More Agents Is All You Need" 表明投票随样本数增加而提升准确率——本文通过"任意一票"规则发现重复采样可能优于多 Agent 划分。
6. **Peng et al. (2026) / Rao & Callison-Burch (2026)**：评估 rubric 数量对 judge 性能的影响——本文扩展至缺陷检测任务并补充误报率测量。

## 局限性与未来方向
1. **文档短小且缺陷率异常高**（37% vs 实际更低），结果不一定适用于长文档与低缺陷率场景。
2. **测试单一配对**（每条要求 2 个/调用），未验证 "one consideration per call" 的极限情况。
3. **合成数据无遗漏、外部事实错误和主观判断类缺陷**，仅覆盖内部一致性检查。
4. **单 Agent 与 Team 的调用顺序效应未分离**（不同 memo 顺序轮换）。
5. **Temperature 混淆**：Qwen3.5 的 Three draws（temp=0.7）与主臂（temp=0）不可直接比较，Claude 模型无此混淆。
6. **误报可能被高估**：约 10–15% 的 "false alarm" 实为"misfiled flag"（引用的是正确但在另一要求下缺陷的句子）。
7. **未来方向**：在真实文档上验证、测试单要求/调用的极限划分、将指令变量与任务划分变量正交设计。

## 研究启发与可借鉴点
1. **公平基线设计**：用"相同调用次数的多次采样"作为多 Agent 的基线，可有效区分"更多调用"与"任务划分"的真实贡献——值得迁移到任何多 Agent 系统的评测中。
2. **信息隔离的量化方法**：Scoped Team 臂（展示未分配要求给 Agent）是一个简洁的实验设计，可推广到其他需要分离多因素效应的场景。
3. **结果高度依赖推理指令**：同一任务下，"decide-once"指令使 Qwen3.5 27B 的团队从显著提升变为无显著差异——提醒我们在对比多 Agent 架构时，必须固定并报告推理约束条件。
4. **Trade-off 的可操作性框架**：基于 $w_M/w_F$ 的成本比与缺陷率 $p$ 计算盈亏平衡点（Table 4），为实际部署提供决策依据——可移植到任何检测类 LLM 应用中。
5. **与 self-consistency 范式的对比**：发现"any-of-three"规则优于多 Agent 划分，说明对某些检测任务，增加采样多样性比任务分解更高效，值得进一步探索两者的最佳组合。

## 关键术语表
- **Balanced accuracy（平衡准确率）**：$(H + 1 - F)/2$，将漏检率和误报率等权重综合为单一指标。
- **Scoped team（受限团队）**：多 Agent 团队中每个 Agent 除负责要求外，还看到其他要求作为上下文但不做判定的实验设置。
- **Masked defect（掩蔽缺陷）**：带有自我解释但仍与其他事实矛盾的缺陷类型。
- **Bare defect（裸缺陷）**：无解释的、直接与同一事实其他陈述相冲突的缺陷。
- **Misfiled flag（误归档标记）**：Agent 在干净单元格上标记的 flag，其引用句子本身无错，但属于另一要求下的缺陷句子。
- **Decide-once instruction（单次判定指令）**：要求模型每个要求只做一次判定、最多 3 句推理、不得回头的 prompt 约束。
- **Signal detection theory（信号检测理论）**：用 $d'$（辨别力）和 $c$（决策阈值）两个参数分解检测性能，本文用于分析团队是否改变判别力还是仅改变阈值。
- **Bonferroni-corrected level（Bonferroni 校正水平）**：将显著性阈值除以测试数量（4 个模型×2 个预测=8 次），本文校正后为 98.75%。

## 可复现要素
- **数据集**：合成备忘录基于 UCI Credit Card Dataset（Yeh & Lien, 2009），代码和预测已固定 committed hash。
- **代码/权重**：论文未提供开源链接，但提到 memos、prompts、predictions 和 analysis code 通过 committed hash 固化。
- **模型**：Qwen3.5 9B/27B（Ollama 0.35.1，Q4_K_M 量化），Claude Haiku 5.5/Sonnet 5.5（Batch API，effort=high）。
- **关键超参**：Context 16,384 token，输出上限 2,048 token/要求，Qwen3.5 temp=0（主臂）/0.7（抽样），Bootstrap 5,000 resamples。
