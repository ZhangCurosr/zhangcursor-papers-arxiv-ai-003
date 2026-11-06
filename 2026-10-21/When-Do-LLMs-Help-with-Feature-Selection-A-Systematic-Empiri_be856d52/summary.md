---
title: "When-Do-LLMs-Help-with-Feature-Selection-A-Systematic-Empiri"
source: https://arxiv.org/pdf/2610.28556v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-06 03:00:58"
field: "特征选择与大语言模型"
keywords: ["feature selection", "large language models", "reinforcement learning", "tabular data", "semantic ablation"]
innovations: ["基数匹配的语义诊断分离语义与基数效应", "严格train-only匹配预算经典控制评估信息不对称", "GRPO适应特征选择的系统实证"]
benchmarks: ["OpenML", "UCI"]
---

# 论文速读：When-Do-LLMs-Help-with-Feature-Selection-A-Systematic-Empiri

## 一句话总结
本研究通过18数据集的受控基准，系统比较了经典统计特征选择器、冻结LLM（Qwen-7B/Llama-70B）及GRPO强化学习适应器在表格特征选择上的表现，发现经典方法在预测性能上普遍更强，而语义特征名对LLM的帮助主要在回归任务且效果依赖数据集。

## 研究问题与动机
- 核心问题：LLM能否利用特征名的语义信息做出优于经典统计方法的特征子集选择？在控制子集基数和特征预算的条件下，语义先验何时有用或误导？
- 现有方法不足：经典选择器（如mRMR、LASSO、Boruta）仅利用统计关系，不利用特征名语义；而已有LLM特征选择工作（如LMPriors、LLM-Select）未剥离子集大小混淆，难以评估纯语义贡献。
- 动机缺口：缺乏在受控信息 regime（仅摘要统计vs完整数据）和匹配基数条件下的公平对比，导致语义机制不清。

## 核心贡献（创新点）
1) **受控特征选择基准**：在18数据集（10训练/2验证/6保留）框架下系统比较经典、冻结LLM和GRPO适应选择器；本质区别在于明确划分训练/验证/保留集，避免数据泄漏，并提供跨任务排名。
2) **基数匹配的语义诊断**：提出精确k个特征的三条件实验（Original/Anonymized/Shuffled）分离语义与基数混淆；区别于以往工作仅报告平均表现，该设计直接量化语义贡献。
3) **鲁棒性与匹配预算控制**：采用5次种子划分评估不确定性、任务分离指标及严格train-only经典top-k控制；弥补已有对比中信息 regime 不对称的缺陷。
4) **GRPO适应的实证检验**：将GRPO+LoRA应用于Qwen2.5-7B以优化性能-稀疏性奖励；与冻结LLM相比，提供RL微调是否带来稳定增益的证据。

## 方法详解
- **提示构建**：PromptBuilder为每个数据集生成提示，包含数据集/任务元数据及每特征摘要统计（数值特征含均值、标准差、范围、缺失率；分类特征含顶部三类频率及缺失率）；不提供原始行或实例级目标。
- **输出解析**：LLM/RL选择器返回标记列表`SELECTED_FEATURES: ["feat_a", ...]`；OutputParser验证令牌与特征集合V的一致性，去重并丢弃幻觉名称；缺失/无效输出触发全保留回退。
- **GRPO适应**：基于Qwen2.5-7B-Instruct，应用LoRA（r=16, α=32, dropout 0.05）于投影层；GRPO生成G=4个补全，每个解析为子集S并计算奖励R(S)=Perf(S)-λ·|S|/d（λ=0.01），奖励在组内归一化形成相对优势；更新时使用KL正则化（β=0.1）。
- **训练流程**：10个数据集用于GRPO训练，2个用于checkpoint验证，6个保留评估；v1训练500步，checkpoint-100最优（验证奖励0.8282）；v2训练1500步从头开始，无验证选择。推理时每个数据集生成K=4个补全，取5次种子划分的平均。

## 实验与结果
- **数据集**：18个OpenML/UCI数据集，维度d=11~100；保留评估集为adult income、parkinsons telemonitoring、student performance、student dropout、sonar、ames housing。
- **评估基线**：No Selection、mRMR、LASSO、Boruta、XGBoost Importance、Zero-shot Qwen-7B、Few-shot Qwen-7B、Zero-shot Llama-70B、RL-GRPO v1/v2。
- **主要结果**：
  - 经典选择器整体性能更强：XGB Importance Top-k（train-only, 33.4%预算）分类ROC-AUC 0.890，回归R² 0.646；均优于RL-GRPO v1（0.799, 0.596）。
  - 7B LLM选择器产生紧凑子集（保留29.7%~33.4%特征），但排名较弱（7.00~9.17 vs 经典3.50~5.08）。
  - Llama-70B零样本整体优于v1（ROC-AUC 0.878 vs 0.799，R² 0.682 vs 0.596），但特征保留更密（60.8%）。
  - GRPO适应效果任务依赖：v1在回归上有轻微提升（+0.025 R²），但分类下降；v2未在保留集上持续改进。
  - 语义诊断：固定基数下，原始特征名使Zero-shot Qwen-7B回归R²平均提升+0.1964（Anonymized）和+0.1606（Shuffled），分类ROC-AUC提升较小（+0.0159, +0.0084）。
- **最强结果**：经典选择器（尤其是XGB Importance）在匹配预算下取得最高预测性能；语义提升在parkinsons telemonitoring最显著（R² +0.416）。

## 相关工作脉络
1) **mRMR/LASSO/Boruta/XGBoost Importance**：经典过滤器/包装器方法，利用统计关系而非语义；本文定位：作为受控基线，证明在匹配预算下仍优于LLM选择器。
2) **LMPriors/LLM-Select/CAAFE**：早期LLM特征选择工作；本文区别：引入基数匹配语义诊断，剥离子集大小混淆。
3) **TabLLM/TableLlama**：面向表格预测的LLM适配；本文聚焦特征选择子任务，强调性能-稀疏性权衡。
4) **PPO/DPO/KTO/GRPO**：RL对齐方法；本文应用GRPO+LoRA优化特征选择奖励，验证微调效果。
5) **SHAP/LIME**：后验归因方法；本文强调预训练子集选择，不依赖模型解释。

## 局限性与未来方向
- 局限性：①预测性能未超越最强经典基线；②仅使用Qwen2.5-7B一个基础架构；③语义诊断仅限Zero-shot Qwen-7B；④仅5次种子划分，统计检验力有限；⑤数据集维度≤100，高维场景未探索。
- 未来方向：扩展至更高维数据集、更丰富语义元数据、更多基础模型（如大参数LLM）及替代RL适应策略。

## 研究启发与可借鉴点
1) **可复用实验设计**：基数匹配的语义干预（Original/Anonymized/Shuffled）可迁移至任何LLM辅助特征工程研究，以隔离语义贡献。
2) **训练-测试信息 regime 控制**：严格train-only经典top-k控制解决了LLM与经典方法间的 asymmetry，可作为未来对比的标准协议。
3) **创新机会**：结合统计摘要与LLM语义生成混合选择器（如先用经典方法预选，再用LLM refine），可能兼顾性能与稀疏性。
4) **奖励函数设计**：性能-稀疏性加权奖励R(S)=Perf(S)-λ|S|/d可适用于其他RL-based特征选择任务。

## 关键术语表
- **mRMR**：最小冗余最大相关性，平衡目标相关性与特征间冗余的过滤器方法。
- **GRPO**：组相对策略优化，通过组内采样比较估计相对优势，无需独立价值 critic。
- **Cardinality-matched semantic diagnostic**：固定子集基数下的语义诊断实验，分离特征名语义与数量效应。
- **Train-only matched-budget control**：严格在训练集上拟合经典选择器，并匹配RL方法的特征保留比例。
- **Feature retention**：被选特征数占原始特征总数的比例，衡量稀疏性。
- **Parse failure**：LLM输出无法解析为有效特征列表的情况，触发了全保留回退。

## 可复现要素
- 数据集：18个OpenML/UCI数据集公开可用（链接见论文参考文献）。
- 代码/权重：使用PyTorch/HuggingFace transformers/peft/trl v0.29.1/scikit-learn/xgboost开源库；LoRA适配器与checkpoint未明确声明开源，论文未提及具体仓库。
- 关键超参：LoRA r=16, α=32, dropout=0.05；GRPO β=0.1, λ=0.01, G=4, K=4, temperature=0.7；学习率1e-6；最大补全长1024。
