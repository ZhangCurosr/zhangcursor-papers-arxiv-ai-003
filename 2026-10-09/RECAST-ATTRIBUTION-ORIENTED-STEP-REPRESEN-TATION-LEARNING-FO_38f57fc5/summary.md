---
title: "RECAST-ATTRIBUTION-ORIENTED-STEP-REPRESEN-TATION-LEARNING-FO"
source: https://arxiv.org/pdf/2610.11334v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-09 10:19:04"
field: "多智能体系统可解释性与可靠性"
keywords: ["failure attribution", "step representation learning", "multi-agent systems", "LLM internal signals", "contrastive learning", "agent debugging"]
innovations: ["首次系统量化评估 step representation 在 MAS 失败归因中的判别分离度（ARS/GRS/SI）", "提出 ReCast 三步框架：attribution-aware 层选择 + 固定稀疏 pattern/deviation 双分支投影 + 对比/排序学习上下文 step 表示", "构建 ReCast-2K 训练数据集并实现毫秒级预计算归因，在多基准 Hit@1 达 SOTA"]
benchmarks: ["Who&When", "Who&When Pro", "AFTraj-2K", "TraceElephant"]
---

# 论文速读：RECAST: ATTRIBUTION-ORIENTED STEP REPRESENTATION LEARNING FOR LLM-BASED AGENT SYSTEMS

## 一句话总结
本文针对 LLM-based 多智能体系统（MAS）的失败归因任务，提出 ReCast 方法，通过学习 attribution-oriented 的 step representation 来定位导致任务失败的 root-cause step。ReCast 在 Who&When、Who&When Pro、AFTraj-2K、TraceElephant 四个基准上的 Hit@1 均达到最优，较最强基线 CHIEF 在 Who&When Algorithm 和 Handcrafted 上分别提升 5.65 pp 和 9.19 pp。

## 研究问题与动机
- **核心问题**：在多智能体系统执行轨迹中，早期步骤的错误往往被后续步骤掩盖，需要准确识别出首个导致失败的决定性错误步骤（root-cause step）。
- **现有表示分离度不足**：通过实证研究发现，直接使用的 Qwen3 text embeddings 与冻结 LLM 的 last-layer hidden states 在区分 root-cause step 与其他 step 时，within-trace 相似度高（ARS/GRS 偏高）、cross-trace 区分度低（SI 偏低），表明现有表示缺乏对归因任务的有效判别力。
- **推理延迟问题**：依赖 LLM 自身进行归因（如 CHIEF、Step-by-Step 协议）需要 autoregressive decoding，产生较高推理开销；而纯特征方法利用内部信号但未能充分学习适配归因的 step 表示。
- **缺乏系统性表征分析**：此前工作多聚焦于设计归因模型本身，对 step representation 的判别能力缺乏定量评估与针对性优化。

## 核心贡献（创新点）
1. **首次系统量化评估 step representation 在 MAS 失败归因中的判别能力**：定义 ARS、GRS、SI 三个度量指标，揭示 text embedding 与 hidden state 在 within-trace 和 cross-trace 层面的分离度不足问题，为后续表征学习提供动机基础。
2. **提出 ReCast 三步特征构建与学习框架**：通过 attribution-aware probe 进行层选择、固定稀疏 Johnson–Lindenstrauss 投影构造 pattern/deviation 互补特征、双向 Transformer 编码器联合对比与排序损失学习上下文 step 表示，显著提升根因可区分性。
3. **构建 ReCast-2K 大规模训练数据集**：收集 1,044 条成功轨迹与 1,496 条失败轨迹，覆盖 GAIA、AssistantBench 等真实任务与 DeepSeek V4 Flash、GPT-4o、Qwen3-235B 等多种生成模型，采用三模型投票+人工审查的根因标注协议。
4. **多基准 SOTA 验证**：在 Who&When、Who&When Pro、AFTraj-2K、TraceElephant 四个公开基准上均以 Hit@1 最优，且特征预计算后归因仅需 0.11–8.28 ms/轨迹，显著快于 LLM-based 方法（秒级）。

## 方法详解
- **Hidden-state extraction**：使用冻结 LLM（Qwen3.5-27B，隐藏维度 $d_h=5120$）对每条轨迹 prefill，对每个候选 action 的 token-level hidden states 做 mean pooling，得到第 $\ell$ 层的 step feature $\mathbf{h}_{i,t}^{(\ell)} \in \mathbb{R}^{d_h}$。
- **Attribution-aware layer selection**：对每条失败轨迹计算差值向量 $\mathbf{d}_i^{(\ell)} = \mathbf{u}_{i,t_i^*}^{(\ell)} - \frac{1}{|\mathcal{N}|}\sum_{t\in\mathcal{N}}\mathbf{u}_{i,t}^{(\ell)}$（$\mathbf{u}$ 为 LN 后的特征），在五折交叉验证中估计均值 $\pmb{\mu}^{(\ell)}$ 与方差 $\mathbf{v}^{(\ell)}$，构造评分方向 $\mathbf{w}^{(\ell)}$（带 ridge 正则化），以 Hit@1/MRR/margin 排序各层；将全部层划分为 $k=8$ 个连续组，每组取最优层，拼接得 $\mathbf{h}_{i,t} \in \mathbb{R}^{k d_h} = \mathbb{R}^{40960}$。
- **Dual-branch projection**：使用固定稀疏 Johnson–Lindenstrauss 矩阵 $\mathbf{W}_{JL} \in \mathbb{R}^{d_p \times k d_h}$（$d_p=2048$，非零密度 $\delta=1/\sqrt{L d_h}$）将拼接特征映射到两个分支：Pattern 分支对每层单独 LN 后再投影，保留相对激活模式；Deviation 分支以成功轨迹平均隐藏状态 $\mu^{succ}$ 为参考，衡量当前 step 与"正常行为"的偏离。输出拼接为 $\mathbf{x}_{i,t} \in \mathbb{R}^{2d_p}=\mathbb{R}^{4096}$，投影矩阵与参考均值全程固定。
- **View augmentation**：对每个 projected 轨迹独立生成两个 view，使用 step mask、feature mask 与高斯噪声（$\sigma=0.01$）进行数据增强，root-cause step 禁止 step masking 但允许 feature-level 扰动。
- **Bidirectional encoder + scoring head**：4 层双向 Transformer（隐藏维 $d_z=128$，head=4）将增强视图映射为上下文 step 表示 $\mathbf{z}_{i,t}^v$，再经 MLP 打分头 $g_\psi$ 输出 attribution score $\alpha_{i,t}^v$。
- **损失函数**：
  - **Root-cause contrastive loss**：以根因 step 在 view f₁ 为 anchor、f₂ 为正样本，批次内所有非根 step 为负样本，使用 InfoNCE（温度 $\kappa=0.12$）。
  - **Non-root contrastive loss**：选取根因前最近步与最高分步作为 anchor，正样本为同一步另一 view，负样本为该轨迹根因步，逐 trajectory 平均加权。
  - **Listwise ranking loss**：对每轨迹内所有候选 step 做 softmax，最大化根因步的概率，两 view 取平均。
  - 总损失 $\mathcal{L}=\lambda_{rank}\mathcal{L}_{rank}+\lambda_{root}\mathcal{L}_{root}+\lambda_{non-root}\mathcal{L}_{non-root}$（权重 $(1.0, 0.95, 0.25)$）。

## 实验与结果
- **数据集**：训练集 ReCast-2K（1,044 成功 + 1,496 失败）；评测基准 Who&When（124 AG + 58 HC）、Who&When Pro（1,251 测试）、AFTraj-2K（163 失败测试）、TraceElephant（175 有效轨迹，Captain-Agent 85 + Magentic-One 90）。
- **主要结果（Hit@1）**：
  - Who&When Algorithm：**45.43%**（+5.65 pp vs CHIEF 的 39.78%）
  - Who&When Handcrafted：**33.33%**（+9.19 pp vs CHIEF 的 24.14%）
  - TraceElephant Captain-Agent：**35.69%**（+4.71 pp vs ASCon 的 30.98%）
  - TraceElephant Magentic-One：**27.04%**（+1.11 pp vs ASCon 的 25.93%）
  - AFTraj-2K：**84.05%**（+1.02 pp vs ASCon 的 83.03%）
  - Who&When Pro：**91.42%**（+8.71 pp vs ASCon 的 82.71%）
- **Representation separation**：ReCast  learned 表示在 Handcrafted/Algorithm 上的 SI 从 0.17/0.03 提升至 0.59/0.29，ARS/GRS 显著降低，验证了表征判别力的改善。
- **效率**：特征预计算后单轨迹归因耗时 0.11–8.28 ms，远快于 LLM-based 方法的 2.88–30.50 s/轨迹。
- **消融**：移除 augmentation（−4.22 pp）、仅用 rank loss（−3.85 pp）、随机选层（−4.40 pp）等均显著降低性能，证实各组件有效性。

## 相关工作脉络
1. **Black-box 归因方法**（CHIEF、FALAT）：通过因果图或层次依赖搜索从可观测轨迹推断根因，但难以区分真实根因与传播症状；ReCast 从模型内部信号出发，提供正交的诊断路径。
2. **Learning-based 归因**（AgenTracer、AgentForesight、ASCon、StepFinder）：基于外部轨迹表征（文本/agent 身份 embedding）训练；ReCast 强调利用 LLM 内部 hidden state，并通过对比/排序学习获得更强的归因判别性。
3. **White-box 归因方法**（MASPrism、OAT）：MASPrism 直接利用 prefill 阶段的 NLL 与 attention 得分；OAT 用 neural CDE 建模成功轨迹的 hidden state 动力学并检测偏离。ReCast 不依赖单次内部信号打分，而是联合特征选择、投影与对比学习端到端学习适配归因的 step 表示。
4. **StepFinder**：使用文本 embedding 与 BiLSTM 捕获时序语义依赖；ReCast 在此基础上进一步引入内部 activation 的 pattern/deviation 双分支以及 trajectory-level 上下文建模，获得更好分离度。

## 局限性与未来方向
- **层选择依赖训练集**：attribution-aware probe 需在有标签的失败轨迹上进行五折交叉验证，跨域泛化时可能需要重新校准或在线自适应。
- **投影固定不可学习**：Johnson–Lindenstrauss 矩阵与成功参考均值全程冻结，虽提升效率但可能限制表征容量；后续可探索轻量可学习投影。
- **Backbone 规模非线性收益**：27B 骨干在多数基准最优，但在 TraceElephant 子集上 9B/0.8B 反而更强，说明更大模型并非总是更好，机制尚待厘清。
- **未探索多模态内部信号**：当前仅用文本 action 的 hidden state，图像/视频轨迹的 internal signal 如何利用未涉及（虽然 Who&When Pro 包含多模态数据）。
- **根因唯一定义**：以最早 decisive error 为唯一目标，部分轨迹可能存在多个共因（co-root causes），单一标签假设可能遗漏并发错误。

## 研究启发与可借鉴点
1. **内部信号判别性量化评估框架**：ARS/GRS/SI 三类指标为表征学习研究提供了可直接迁移的评估范式，可在其他诊断/可解释性任务中复用。
2. **固定稀疏投影 + 成功参考 deviance 的设计**：无需额外参数即可构造"正常 vs 偏离"双分支，方法简洁且对下游 encoder 规模敏感度高，可推广至其他序列异常检测任务。
3. **非根 anchor 选择策略**：选取根因前最近步与最高分步作为 non-root contrastive anchor，兼顾"易混淆步"与"位置邻近步"，为负样本挖掘提供了实用启发。
4. **预计算+轻量推理的工程范式**：冻结 backbone 提取特征、仅训练小型 encoder+head，使推理成本降至毫秒级，对需要在线监控的 agent 系统具有强工程参考价值。
5. **可迁移至 Agent 在线审计**：ReCast 可在 step 生成后即时打分，无需等待轨迹结束，与 AgentForesight 的在线检测目标一致，可作为其内部表征模块的升级替代。

## 关键术语表
**Failure Attribution（失败归因）**：在多智能体执行轨迹中定位导致任务失败的 root-cause step（最早的决定性错误步骤）。
**Decisive Error（决定性错误）**：若将该 step 的动作替换为可行修正动作后轨迹变为成功，则该 step 为决定性错误。
**ARS / GRS / SI**：相邻根因相似度（ARS）、全局根因相似度（GRS）、余弦 Silhouette Index（SI），用于量化 step representation 的 within-trace 与 cross-trace 分离度。
**Johnson–Lindenstrauss Projection**：固定稀疏随机投影，用于将高维 hidden state 降维至固定维度，保持近似距离结构。
**Pattern / Deviation Branch**：Pattern 分支保留各层相对激活模式；Deviation 分支衡量当前 step 与成功轨迹平均激活的偏离程度。
**InfoNCE Contrastive Loss**：以根因步为 anchor，同一步另一 view 为正样本，批次内所有非根 step 为负样本的对比损失。
**ReCast-2K**：本文构建的训练数据集，含 1,044 条成功轨迹与 1,496 条失败轨迹，覆盖多模型、多任务、多 MAS 架构。
**Hit@k**：根因 step 出现在模型预测 top-k 中的比例，为主要评测指标。

## 可复现要素
- **数据集**：ReCast-2K 训练集为自建；评测基准 Who&When、Who&When Pro、AFTraj-2K、TraceElephant 均为公开数据集。
- **代码**：代码已开源（https://anonymous.4open.science/r/ReCast-5FB6）。
- **关键超参**：backbone=Qwen3.5-27B；选中层 $k=8$（{7,15,19,32,33,47,49,57}）；投影维 $d_p=2048$；encoder=4 层 Transformer，$d_z=128$；温度 $\kappa=0.12$；损失权重 $(\lambda_{rank}, \lambda_{root}, \lambda_{non-root})=(1.0, 0.95, 0.25)$；优化器 AdamW，lr=$2\times10^{-4}$，batch=32，max epoch=50；dropout=0.10；augmentation step/feature mask p=0.10，高斯噪声 $\sigma=0.01$。
- **种子**：训练种子 6, 20, 42。
