---
title: "When-Do-LLMs-Help-with-Feature-Selection-A-Systematic-Empiri"
source: https://arxiv.org/pdf/2610.28556v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-06 03:01:25"
---

# 论文速读：When-Do-LLMs-Help-with-Feature-Selection-A-Systematic-Empiri

## 一句话总结
本文系统评估了大语言模型（LLMs）利用特征名称语义与摘要统计进行表格特征选择的效能，通过18数据集基准与严格控制实验揭示：语义信息在回归任务上可带来实质性增益，但经典统计选择器在预测性能与稀疏-性能权衡上仍显著优于当前7B级冻结/RL自适应LLM，且该优势无法仅由子集规模或全数据集拟合解释。

## 研究问题与动机
- **核心问题**：在严格控制子集基数与信息利用边界（train-only）的条件下，LLM特征选择器的预测性能与稀疏性如何与经典统计方法对比？有意义的特征名称语义到底贡献了多少独立价值？
- **现有方法不足**：已有LLM特征选择工作（如 LMPriors、LLM-Select）主张语义元数据有益，但语义效应常被子集大小差异、评估协议不一致及信息泄漏混杂，缺乏在严格对照下的归因分析。
- **动机**：厘清LLM特征选择器实际利用的信息类型、语义先验何时助力或误导，以及其在已知特征预算（~33%）下与目标监督统计方法的本质差异，为后续混合架构设计提供实证依据。

## 核心贡献（创新点）
- **提出受控18数据集基准**：划分10个GRPO训练集、2个验证集、6个严格保留评估集，系统对比5类经典选择器、冻结Qwen-7B/Llama-70B及两类GRPO自适应Qwen-7B在分类ROC-AUC与回归R²上的表现。
- **设计基数精确匹配的语义诊断**：引入Original/Anonymized/Shuffled三条件实验，强制所有completion的子集基数完全一致，在屏蔽数据集与目标身份的前提下精准剥离特征名称语义与卡片数混杂效应。
- **建立严格匹配预算的经典对照**：设定约33.37%特征预算并按$k_d=\max(1,\text{round}(0.3337d))$截断Top-k，经典选择器仅在60%训练集上拟合，证明其性能优势源于目标监督统计信号而非更大的子集或信息滥用。

## 方法详解
- **提示构建与输出解析管道**：`PromptBuilder`为每个数据集生成含任务元数据与逐特征摘要的提示；数值特征提供均值/标准差/范围/缺失率，类别特征提供Top-3频数/缺失率，不提供原始样本行或实例级标签。`OutputParser`提取`SELECTED_FEATURES: [...]`标签列表，校验token与真实特征集合$V$的匹配度，去重并剔除幻觉名称，解析失败时回退至保留全部特征（`parse_ok=False`）。
- **GRPO自适应微调框架**：以`Qwen2.5-7B-Instruct`为基础，对`q/k/v/o_proj`应用LoRA（$r=16,\alpha=32,\text{dropout}=0.05$）。采用TRL实现GRPO：每提示生成$G=4$个completion，由固定XGBoost下游预测器计算性能，结合稀疏惩罚定义复合奖励$R(S) = \text{Perf}(S) - 0.01 \cdot |S|/d$，经5重划分平均后归一化组内相对优势，使用KL正则化（$\beta=0.1$）更新策略，重复评估结果缓存以避免冗余XGBoost打分。
- **基数匹配语义消融**：对Zero-shot Qwen-7B施加原始名、匿名化名（随机重命名）、错排名（严格derangement映射回原列）三种条件，强制$|S^{\text{orig}}_{d,c}|=|S^{\text{anon}}_{d,c}|=|S^{\text{shuf}}_{d,c}|$，提示摘要统计不变，直接比较Original vs Anonymized（语义信息增益）与Original vs Shuffled（误导性语义敏感度）。
- **训练集严格匹配预算控制**：经典选择器（mRMR/XGB/LASSO）仅在各seed专属的60%训练集上拟合，LASSO标准化同样train-only，Boruta因无严格Top-k排序被剔除，确保信息regime与子集规模双重对齐。

## 实验与结果
- **数据集与设置**：6个保留评估数据集（Adult, Sonar, Dropout分类；Ames, Parkinsons, Student Performance回归），5重划分（seeds 0–4），固定XGBoost下游（100树、max_depth=4、lr=0.1、subsample=0.8）。7B LLM/RL方法每数据集生成K=4个completion，取5重划分内平均得分。
- **总体性能对比**：经典选择器平均排名（3.25–5.08）显著优于LLM/RL方法（5.17–9.17）。冻结Llama-3.3-70B表现最佳（分类ROC-AUC 0.878，回归R² 0.682，排名5.17），但保留60.8%特征；7B模型虽保持紧凑子集（~30–33%保留），预测性能仍落后。RL-GRPO v1分类ROC-AUC 0.799、回归R² 0.596，未见一致超越。
- **GR
