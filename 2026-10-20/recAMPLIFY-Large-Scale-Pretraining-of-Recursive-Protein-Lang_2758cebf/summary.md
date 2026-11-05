---
title: "recAMPLIFY-Large-Scale-Pretraining-of-Recursive-Protein-Lang"
source: https://arxiv.org/pdf/2610.26747v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 17:23:17"
field: "蛋白质语言模型与自适应计算"
keywords: ["protein language models", "recursive pretraining", "test-time scaling", "adaptive computation", "Gated Mixture Recurrence", "masked language modeling", "protein representation learning"]
innovations: ["首次将权重绑定递归预训练适配到掩码蛋白质语言模型，分离有效深度与唯一参数", "提出GMR统一自监督停止与残差门控，实现自适应递归计算", "揭示停止深度编码序列组成熵与疏水性等可解释生物学信号"]
benchmarks: ["ProteinGym DMS", "CASP16 Contact Recovery", "KL Calibration", "18-task TAPE/FLIP Benchmark"]
---

# 论文速读：recAMPLIFY-Large-Scale-Pretraining-of-Recursive-Protein-Lang

## 一句话总结
论文提出recAMPLIFY，一种面向掩码蛋白质语言模型的递归预训练框架，通过权重绑定Transformer核心迭代精炼隐状态，将有效模型深度与唯一参数数量分离；进一步提出GMR（Gated Mixture Recurrence）实现自适应递归计算，以155M参数在相同2T tokens上预训练，超越参数量多56%的AMPLIFY-350M。

## 研究问题与动机
1. 蛋白质语言模型进展依赖缩放参数与数据，导致计算成本高、易记忆化训练数据；ESM-2-15B虽大，但AMPLIFY-350M在同等数据下可实现相近效果。
2. 自回归LLM的测试时缩放（如Chain-of-Thought、多次采样）不直接迁移到掩码pLMs：氨基酸序列无法语音化为推理token，插入thought tokens会破坏掩码预测所需的残基-状态对齐。
3. 不同任务和检测的最优递归深度各异（见Figure 2），固定深度递归在测试时深度超出训练深度后会退化（overthinking问题），需要输入依赖的自适应计算分配。
4. pLMs的迭代精炼已在AlphaFold/ESMFold的结构预测中验证，但仅作用于结构表示而非输入依赖的测试时序列表示，输入依赖的隐状态精炼尚未被探索。

## 核心贡献（创新点）
1. **递归预训练框架**：首次将深度递归适配到掩码pLMs的预训练阶段，分离有效深度与唯一参数；与AMPLIFY等前作的本质区别在于递归发生在序列编码器内部而非结构预测 trunk 中。
2. **recAMPLIFY模型**：155M参数在2T tokens上预训练，fold准确率提升22.5%（56.8%→69.6%），超过AMPLIFY-350M；与同类紧凑模型的本质区别在于通过递归将参数效率提升了56%。
3. **GMR（Gated Mixture Recurrence）**：统一自监督停止与残差门控，解码来自递归轨迹中隐藏状态的概率加权混合；与PonderNet等前作相比，GMR将halting与gating整合到单一自适应计算目标中，无需监督验证器。
4. **机制分析发现**：残差门控（非仅停止头）将递归状态驱动至实证不动点；停止深度经长度和锐度控制后追踪序列组成熵与疏水性；这是pLMs递归动力学的首次系统性分析。

## 方法详解
**基本架构（Figure 1）**：非递归prelude（p层）→ 权重绑定的r层recurrent core → 非递归coda（c层）。预处理器将token embedding映射为残基对齐的隐状态x∈R^(T×D)，核心在L个循环中迭代精炼z_k = f_r(z_{k-1}, x)，coda将最终状态z̃=L映射到MLM输出。有效深度为p + r·L + c，但唯一参数仅对应p+r+c层。

**关键公式**：
- 基础递归：x = f_p(Embed(S)), z_k = f_r(z_{k-1}, x), ŷ_k = LMHead(f_c(z̃))
- 隐状态初始化：z_init = 1_T · u^T，u ~ N(0, 0.02²I)，冻结
- 输入注入：每循环重新注入x以保持有界迭代映射

**GMR框架**：
- 停止头：λ_k = sigmoid(HaltHead(sg(z̃_k)))，计算无条件停止质量p_k = λ_k · ∏_{j<k}(1-λ_j)
- 解码混合：z̄ = Σ_k p_k z_k，损失L = L_MLM(LMHead(f_c(z̄))) - H({p_k})，熵项鼓励更多循环
- 残差门控：g_k^ch = sigmoid(GateHead(sg(z̃_k)))，存活S_k^ch = [g_1^ch, ..., g_k^ch]，更新z_k = z_{k-1} + S_k^D ⊙ (z̃_k - z_{k-1})
- 三种门控粒度：标量（ch=1广播）、低秩（ch=16投影到D=960）、逐维度（ch=D）

**训练细节**：100K steps / 2T tokens（UniRef100 + SCOP + OAS），mask rate=0.15，max length=512，AdamW(lr=10⁻³, β₁=0.9, β₂=0.95, wd=10⁻²)，cosine schedule + 10% floor + 1K warmup，ε=0.001。

## 实验与结果
**数据集**：预训练使用UniRef100（通用蛋白质组）、SCOP（结构域多样性）、OAS（抗体序列）混合；验证集为各自held-out split。

**评估基准**：零样本——KL校准（target/non-target）、ProteinGym DMS、CASP16接触恢复；监督——18-task benchmark中4个代表性任务（fold、localization、stability、SSP）。

**主要结果（Table 1，1M步预训练）**：
| 模型 | 参数 | Fold acc | Loc acc | Stability ρ | SSP | CASP P@T | KL(T) | DMS ρ |
|------|------|----------|---------|-------------|-----|----------|-------|-------|
| AMPLIFY-350M | 354M | 0.568 | 0.716 | 0.585 | 0.742 | 0.126 | 69.31 | 0.345 |
| recAMPLIFY (fixed-L) | 155M | **0.696** | **0.741** | **0.703** | 0.742 | **0.179** | 59.32 | 0.348 |

- recAMPLIFY fold准确率提升22.5%（0.568→0.696），localization提升3.5%（0.716→0.741），stability提升20.2%（0.585→0.703）
- 100K步ablation（Table 1底部）：GMR变体将预期深度从52降至33-37（减少32-40%），CASP P@T最高达0.161（halt-only为0.117），但DMS和KL校准略有下降

**通过效率（Table 5）**：recAMPLIFY在L=3时吞吐量与AMPLIFY-350M相当，有效深度28 vs 32 blocks。

**全18任务（Table 6）**：recAMPLIFY在18个监督任务中赢下14个。

## 相关工作脉络
1. **ESM系列/Rives et al.**：建立pLMs通过无监督学习捕获结构/功能信息的范式；本文在参数效率路径上与其竞争，而非继续扩大规模。
2. **AMPLIFY-Fournier et al. 2024**：证明数据策展和架构比参数规模更重要；本文在其公开recipe和代码上直接对比，复用相同训练数据。
3. **Universal Transformer/Looped Transformer-Dehghani et al. 2019, Giannou et al. 2023**：权重绑定递归用于因果decoder；本文将其适配到双向掩码encoder，解决氨基酸不语音化的适配问题。
4. **PonderNet-Banino et al. 2021**：自监督停止头；本文GMR将其扩展为含残差门控的统一框架，适用于无verifier的MLM场景。
5. **AlphaFold/ESMFold-Jumper et al. 2021, Lin et al. 2023**：结构预测中的recycle机制；本文指出这些工作在监督结构trunk而非序列编码器内操作，且使用固定recycle次数，本文首次探索输入依赖的测试时隐状态精炼。
6. **Depth-recurrent LMs-Geiping et al. 2025**：在有限数据 regime 研究递归；本文验证其在1M步/2T tokens大规模预训练中同样有效。

## 局限性与未来方向
1. 每个蛋白质的期望停止深度E[k]在10K UniProt序列上仅变化约1%的平均值，自适应调节幅度有限。
2. 门控为per-sequence级别，无法在序列内部实现token级自适应计算；per-token存活度是自然扩展方向。
3. 未通过Jacobian分析从理论上建立递归算子的局部稳定性，稳定性仅通过实证（更新范数趋零、残差衰减）刻画。
4. 所有实验基于AMPLIFY数据recipe（UniRef100+SCOP+OAS），扩展到其他规模、架构或模态仍待验证。
5. Figure 18显示标量门控在移除输入注入后会坍缩为输入无关轨迹，提示低自由度门控可能以牺牲输入敏感性为代价换取稳定性。

## 研究启发与可借鉴点
1. **权重绑定递归作为参数效率工具**：对于资源受限的团队，可在预训练中将部分层设为recurrent core并暴露循环数为超参，以少量参数获得大模型同等的测试时计算预算。
2. **GMR框架的通用性**：统一停止+门控的设计可迁移到其他需要自适应计算的连续潜变量模型（如视觉编码器、多模态模型），无需额外verifier。
3. **停止深度编码生物学信号**：控制长度和预测锐度后，E[k]与序列组成熵/疏水性高度相关，可将halt depth作为可解释的计算分配诊断信号，探索其在其他生物大分子（RNA、多肽）建模中的应用。
4. **可变深度预训练（variable-L）增强鲁棒性**：在训练中均匀采样L∈{1,...,6}可显著缓解超出训练深度的性能退化，建议作为递归模型的标准训练策略。
5. **z̄ vs coda输出作为下游probe的选择**：Figure 8显示z̄（循环核心输出）在浅层深度优于coda，但在深层不可见深度更稳定；可根据任务深度需求灵活选择表征。

## 关键术语表
**recAMPLIFY**：基于递归预训练的蛋白质语言模型，通过权重绑定Transformer核心迭代精炼残基对齐隐状态，分离有效深度与唯一参数。
**GMR（Gated Mixture Recurrence）**：门控混合递归框架，将自监督停止与残差门控统一，解码来自递归轨迹中隐藏状态的概率加权混合。
**掩码语言建模（MLM）**：随机遮蔽序列中的token，训练模型从双向上下文中预测被遮蔽残基，是pLMs的核心预训练目标。
**残差门控（Residual Gate）**：通过sigmoid输出的存活信号控制每循环更新量的接受程度，防止递归超训练退化。
**有效深度（Effective Depth）**：D_eff = p + r·L + c，表示前向传播中Transformer block的应用次数。
**KL校准（KL Calibration）**：评估pLMs预测分布与MSA共识 profile之间的KL散度，衡量模型对进化约束的编码质量。
**ProteinGym DMS**：大规模深度突变扫描基准，包含217个assays、246万个突变变体，用于零样本突变效应预测。
**CASP Contact Recovery**：基于分类Jacobian的残基-残基接触预测，在CASP16目标上评估长程接触Precision@T。

## 可复现要素
- **数据集**：UniRef100、SCOP、OAS（均公开可用）
- **代码**：github.com/chandar-lab/AMPLIFY/tree/recamplify（开源）
- **模型权重**：huggingface.co/chandar-lab/recamplify（开源）
- **关键超参**：hidden dim D=960，attention heads=15，FFN width=2560，RoPE θ=10000，(p,r,c)=(2,10,2)，L_train=6，ε=0.001，mask rate=0.15，max length=512，AdamW(lr=10⁻³, β₁=0.9, β₂=0.95, wd=10⁻²)，gradient clip=1.0，cosine schedule + 10% floor + 1K warmup
- **训练硬件**：8× NVIDIA H100，bf16精度
- **预训练步数**：100K（ablation）/ 1M（main results）
- **FLAŞHATTENTION**：varlen buffer，per device 32768 tokens
