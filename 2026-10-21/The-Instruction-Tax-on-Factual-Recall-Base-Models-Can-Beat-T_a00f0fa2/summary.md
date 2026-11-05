---
title: "The-Instruction-Tax-on-Factual-Recall-Base-Models-Can-Beat-T"
source: https://arxiv.org/pdf/2610.28407v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 21:50:02"
---

# 论文速读：The-Instruction-Tax-on-Factual-Recall-Base-Models-Can-Beat-T

## 一句话总结
本文针对 AKBC LM-KBC 2026 闭卷知识抽取任务提出了一套纯提示推理方案，首次系统揭示并量化了“指令微调会损害事实性召回”的现象（Instruction Tax），证明在适当 Few-shot Completion 提示与高频采样聚合下，27B 参数的 Base 模型能以 0.6836 的官方得分大幅超越同架构指令微调版本（0.5681）与任务基线（0.2964）。

## 研究问题与动机
- **闭卷事实提取的提示敏感性**：LLM 在预训练中隐式存储了大量事实，但如何稳定、无幻觉地从参数中提取特定 subject-relation 的全量对象仍是开放难题（Petroni et al., 2019; Jiang et al., 2020）。
- **指令微调的隐性代价未被量化**：学界普遍默认 it-tuned 模型在问答类任务中更优，但 Ouyang et al. (2022) 与 Zhou et al. (2023) 已提示微调可能改变知识表征；本文聚焦其在事实召回维度的具体损耗。
- **任务要求精准拒答**：AKBC LM-KBC 2026 约 20% 的测试样本不存在合法答案，评分器对误报惩罚极重（空集行误猜得 0），迫使系统必须兼顾高召回与高置信度拒答。
- **算力与选型效率约束**：在 32B 参数预算与单次比赛机会下，需快速确定最优模型版本与交互范式，避免盲目全量试错。

## 核心贡献（创新点）
1. **提出并实证“Instruction Tax”**：通过控制变量的对照实验证明，it-tuned 模型的事实召回下降源于权重层面的知识稀释，而非提示方式差异；即使强行使用 completion 提示，其 sampling ceiling 仍大幅坍塌。
2. **构建 Base-Model Completion 抽取 pipeline**：放弃 chat 接口，采用模式匹配的文本续写提示，结合 64–256 次独立采样与频率阈值过滤，使“空集拒答”成为概率收敛的自然涌现结果。
3. **设计轻量级 Knowledge Probe 选型协议**：仅需约 1 GPU 小时即可对候选模型计算 voted accuracy 与 ceiling 双指标，预测排序与官方最终排名完全一致，大幅降低模型评测门槛。
4. **关系特化的后处理模块**：针对长尾多候选的 `awardWonBy` 引入二阶 completion 验证器（Verification Rescue），针对尾部知识占比高的 `personHasCityOfDeath` 设计生死前置检查（Existence Check），突破单纯重复采样的性能瓶颈。

## 方法详解
- **Knowledge Probe 模型筛选**：从训练集抽取 ~100 条已知样本，用多种自然语言措辞查询每个候选模型，采样 8–12 次。计算 `voted`（最高频答案的正确比例）与 `ceiling`（正确答案至少出现一次的比例），以 ceilling 衡量理论上限、以 voted 衡量稳定性，据此选定 `gemma-3-27b-pt` 作为全任务唯一推理模型。
- **Completion 式 Few-shot 提示构造**：严格禁用问句与 chat template，采用 `{subject} 的某属性值是 {value}` 的续写格式。数值示例均匀覆盖训练集各分位数以防范围偏差；空集示例约占 1/3 并标注 `none`。
- **多采样聚合与阈值决策**：每行独立采样 $n$ 次（温度固定 0.7，截断至首行换行）。数值关系按官方 ±5% 容差聚类，取最大簇中位数 $\hat{v} = \operatorname{median} C(c^*)$；集合关系按候选 $a$ 的出现频率 $s(a) = \frac{1}{m}|\{j : a \in A_j\}|$ 聚合，仅保留 $s(a) \geq \tau_r$ 的答案。阈值 $\tau_r$ 在训练集上对每个关系单独拟合，$n$ 或 $k$ 变化时重新校准。
- **Verification Rescue（`awardWonBy`）**：该关系候选集可达数百，低支持度真实候选易被噪声淹没。初筛保留 support $\geq 2/64$ 的候选后，使用 10 条正反例组成的 completion 验证器对其逐一提问（如 `Did Barack Obama get the Nobel Peace Prize?`），采样 8 次计算肯定率 $q(a)$，仅保留 $q(a) \geq 0.75$ 的候选，最后执行字符串非神经网络去重。
- **Existence Check（`personHasCityOfDeath`）**：利用“生死状态”为头部知识、“死亡地点”为尾部知识的分布差异，先用 10 条正反例（`As of 1 July 2026, {s} is alive/dead`）采样 16 次；若全部预测为 alive 则直接输出空集，否则才进入常规地点召回流程。该前置过滤使拒答 F1 提升至 0.798。

## 实验与结果
- **评测设置**：AKBC LM-KBC 2026 官方 Shared Task，6 种关系，test 集 477 样本；官方 scorer 基于归一化别名匹配计算行级 F1，空集对空集得 1.0，单边为空得 0。
- **主结果**：最终系统官方 Test F1 为 **0.6836**，较任务基线（0.2964）翻倍以上，较同架构 it-tuned chat 版本（0.5681）提升 **+0.116**。
- **分关系表现**：`hasArea` (+0.203)、`hasCapacity` (+0.156)、`personHasCityOfDeath` (+0.090)、`companyTradesAtStockExchange` (+0.088)、`awardWonBy` (+0.067)、`countryLandBordersCountry` (+0.012)。数值关系与拒答相关关系受益最大。
- **消融结论**：采样数从 8 → 24 → 64 带来连续增益，集合关系在 128–256 次趋于饱和；Verification Rescue 贡献 +0.027，Existence Check 贡献 +0.030；Probe 排序与全量提交 ranking 完全一致，验证了其代理有效性。

## 相关工作脉络
- **LAMA / LM-KBC 事实抽取线**：Petroni et al. (2019) 与 Singhania et al. (2022) 奠定闭卷 KG 补全评测范式，本文延续该设定但首次将“微调类型（base vs it）”作为核心变量显式建模。
- **指令微调的认知代价研究**：Ouyang et al. (2022) 与 Zhou et al. (2023) 指出对齐训练会重塑模型行为与知识分布，本文在 factual
