---
title: "WASD-Wasserstein-based-Knowledge-Distillation-for-Large-Lang"
source: https://arxiv.org/pdf/2610.07706v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-07 10:57:03"
field: "LLM知识蒸馏"
keywords: ["知识蒸馏", "最优传输", "Wasserstein距离", "Sinkhorn散度", "大语言模型", "语义对齐"]
innovations: ["提出Sinkhorn散度作为语义感知蒸馏目标，校正熵正则化bias", "推导stop-gradient对偶势损失的梯度等价性，避免穿越传输求解器反向传播"]
benchmarks: ["Dolly Eval", "Self-Instruct", "Vicuna", "Super-Natural Instructions", "AlpacaEval", "Evol-Instruct", "UltraFeedback", "HumanEval", "MBPP"]
---

# 论文速读：WASD-Wasserstein-based-Knowledge-Distillation-for-Large-Lang

## 一句话总结
本文提出 WASD（Wasserstein-based Knowledge Distillation），首次将基于最优传输的语义感知距离引入 LLM 知识蒸馏，利用 token 嵌入语义构造代价矩阵，使教师-学生分布对齐不再仅比较词汇索引上的概率值，而是考虑 token 间的语义关系。实验在 GPT-2、OpenLLaMA2、Qwen2.5、Gemma 等多个模型族上均取得一致提升。

## 研究问题与动机
- 现有 LLM 蒸馏方法（KL、RKL、GJS、SKL 等 f-散度）仅基于词汇索引处的概率值比较分布，**不利用 token 本身的语义信息**，例如 `sofa` 与 `couch` 被当作完全不同 token 处理。
- 基于概率比值的散度在高稀疏的 token 分布下存在数值不稳定问题，且无法区分"语义相近的错误 token"与"完全不相关的错误 token"。
- 直接用熵正则化 Wasserstein 距离会引入 entropic bias，使相同分布间的距离不为零，违背蒸馏目标（学生应收敛到教师分布）。
- 需要一种**同时利用 token 语义、保持数学properness、且计算可行**的蒸馏散度。

## 核心贡献（创新点）
1. **提出 Sinkhorn 散度作为蒸馏目标**：通过自相似校正消除熵正则化 bias，保证唯一最优解为教师分布；与 KL 类方法的本质区别在于代价矩阵引入了 token 嵌入语义。
2. **推导梯度等价的可优化损失（Theorem 3）**：利用 envelope theorem 证明对 dual potentials 施加 stop-gradient 后，梯度与 Sinkhorn 散度梯度完全一致，无需反向传播穿越 Sinkhorn 迭代；与传统 OT 方法需直接微分求解器的本质不同。
3. **系统性跨模型族实验验证**：在 GPT-2、OpenLLaMA2、Qwen2.5、Gemma 四大家族及代码/翻译/推理等任务上均取得一致提升，证明了语义感知蒸馏的普适性。
4. **揭示 quality-diversity trade-off 新前沿**：WASD 在维持生成质量的同时显著提升多样性，归因于语义相近 token 被赋予更低传输代价。

## 方法详解
- **代价矩阵构造**：以教师模型的 token embedding 间余弦距离（或 L2 距离）作为代价 $C_{ij}$，并采用 nearest-$k$ 截断策略（论文默认 $k=8$，Qwen2.5 用 $k=2$）构造稀疏核，对称化后加 self-loop。
- **Sinkhorn 散度定义**：
  $$D_S^\epsilon(p,q_\theta) = \tilde{D}_W^\epsilon(p,q_\theta) - \tfrac{1}{2}\tilde{D}_W^\epsilon(q_\theta,q_\theta) - \tfrac{1}{2}\tilde{D}_W^\epsilon(p,p)$$
  其中 $\tilde{D}_W^\epsilon$ 为熵正则化 Wasserstein 距离，$D_S^\epsilon$ 满足非负性且仅在 $p=q_\theta$ 时为零。
- **对偶变量与梯度**：通过 Sinkhorn-Knopp 迭代（10步）求解交叉传输的对偶变量 $\phi^{*(p,q_\theta)}$，通过不动点迭代求解自传输对偶变量 $\phi^{*(q_\theta,q_\theta)}$；WASD 损失为：
  $$\mathcal{L}_{WASD} = \mathbb{E}\left[\sum_l\sum_i \text{sg}(\phi_i^{*(p,q_\theta)} - \phi_i^{*(q_\theta,q_\theta)}) \cdot q_\theta(y_l=i|\mathbf{x},\mathbf{y}_{<l})\right]$$
- **与 KL 梯度对比**：KL 以教师概率 $p(i)$ 加权 $-\nabla_\theta\log q_\theta(i)$；WASD 以学生概率 $q_\theta(i)$ 加权对偶势差 $(\phi^{*(p,q_\theta)}_i - \phi^{*(q_\theta,q_\theta)}_i)\cdot\nabla_\theta\log q_\theta(i)$，后者由 token 嵌入几何决定。
- **超参**：熵正则化 $\epsilon=0.001$，温度缩放 $T=2$，学习率依模型族设定（GPT-2/OpenLLaMA2: $10^{-4}$，Gemma: $10^{-5}$，Qwen: $5\times10^{-5}$）。

## 实验与结果
- **GPT-2 XL(1.5B)→Base(0.1B)**（五指令基准平均 ROUGE-L）：WASD 24.02，AMiD 23.46（+0.56），CSD 22.22（+1.80）；UnNI 达 31.86 为全场最高。
- **GPT-2 XL→Medium(0.3B)**：WASD 25.51，AMiD 24.74（+0.77）；Vicuna 达 18.41 全场最高。
- **OpenLLaMA2-7B→3B**（LoRA）：WASD 29.61，AMiD 29.30（+0.31）；Super NI 达 38.83 全场最高。
- **Qwen2.5-7B→1.5B**（GPT-4o-mini 裁判 win rate）：AlpacaEval 90.04% vs AMiD 89.29%，Evol-Instruct 85.32% vs 83.94%，UltraFeedback 73.21% vs 72.81%。
- **GPT-4 评分（相对参考答案百分比）**：Dolly Eval 37.84%，Self Inst 20.90%，Vicuna 25.07%，三项均为最高。
- **任务特定蒸馏（Gemma-7B→2B）**：翻译 COMET 74.53（AMiD 73.70），摘要 ROUGE-L 35.15，算术准确率 24.87；三项均最高。
- **代码生成（Qwen2.5-Coder-7B→1.5B）**：HumanEval 74.4，MBPP 75.4，平均 74.9（AMiD 74.1）。
- **消融**：语义代价 > permuted 代价（23.47 ≈ AMiD）> uniform 代价（22.08）；Sinkhorn 散度 > 熵正则化 Wasserstein；$\epsilon$ 过大会导致性能下降。
- **开销**：训练时间约 2×基线，GPU 显存 +20%，推理无额外开销。

## 相关工作脉络
- **Hinton et al. 2015**（KL 蒸馏）：奠定基础，但仅用概率比值，无语义信息；WASD 的根本区别在于代价矩阵。
- **Gu et al. 2024 MiniLLM**（RKL）、**Ko et al. 2024 DistiLLM**（SKL/SRKL）：f-散度变体，仍为索引级比较；WASD 引入传输语义。
- **Agarwal et al. 2024 GKD**（GJS）：off-policy 蒸馏；与 WASD 正交，WASD 可替代其散度项。
- **Wang et al. 2025 ABKD**（α-β 散度）、**Kim et al. 2026 CSD**（discrete score matching）：改进优化稳定性；WASD 补充了语义维度。
- **Shin et al. 2026 AMiD**（assistant distribution）：最强 baseline，WASD 可与之结合且在同等框架下超越。
- **Na et al. 2026 WPR**：首次将 token 嵌入 OT 用于 LLM 偏好学习；WASD 继承其 nearest-k 截断与对偶势计算，但改用 Sinkhorn 散度解决 entropic bias。
- **Cuturi 2013 / Feydy et al. 2019**：Sinkhorn 散度的生成建模应用；本文首次将其适配为 LLM 蒸馏目标。
- **SinKD 2024 / WKD 2024**：OT 用于表示/特征级蒸馏；WASD 面向自回归 next-token 分布对齐，两者正交。

## 局限性与未来方向
- **计算开销**：Sinkhorn 迭代使训练时间约增 2 倍、显存增 20%，需进一步加速（如 fast Sinkhorn 算法）。
- **代价矩阵构造**：当前仅用教师 token 嵌入的通用语义距离，未融入 task-specific 或 domain-specific 信息。
- **适用边界**：假设师生共享词表，跨词表场景需结合 ULD、MCW-KD 等 cross-tokenizer 方法（论文已初步验证 DSKDv2 可扩展性，但非主线）。

## 研究启发与可借鉴点
- **语义感知 OT 作为通用蒸馏信号**：可将 Sinkhorn 散度替换任意 f-散度的位置，作为插件式改进与现有框架（AMiD、DistiLLM-2 等）兼容。
- **stop-gradient on dual potentials 的技巧**：避免穿越迭代求解器反向传播，这一设计可推广至其他基于 OT 的训练目标。
- **quality-diversity trade-off 的新优化视角**：通过语义代价保留"合理备选"，为可控多样性生成提供蒸馏层面的新思路。
- **nearest-k 稀疏核 + 对称化**：高效构造高维 token 空间的代价矩阵，可迁移至其他需要大规模最优传输的应用。
- **跨模型族蒸馏实验设计**：通过 DSKDv2 投影层复用 token 几何，展示语义传输在非共享词表场景下的潜力，可作为多架构蒸馏的新范式。

## 关键术语表
- **WASD（Wasserstein-based Knowledge Distillation）**：基于最优传输的 LLM 知识蒸馏方法，利用 token 嵌入语义构造代价矩阵。
- **Sinkhorn 散度**：经自相似校正后的熵正则化 Wasserstein 距离，保证非负且仅在两分布相同时为零，适合作为蒸馏目标。
- **对偶变量（Dual potential）**：熵正则化最优传输问题的对偶优化变量，编码了 token 间的语义传输成本信息。
- **stop-gradient（sg）**：前向传播保留值但阻断梯度回传的算子，此处用于防止梯度穿越 Sinkhorn 迭代求解器。
- **熵正则化偏差（Entropic bias）**：熵正则化使相同分布间的传输代价不为零，Sinkhorn 散度通过自传输项减法校正此偏差。
- **nearest-k 截断**：仅保留每个 token 在嵌入空间中最近的 k 个非 self 邻居构造稀疏代价核，降低计算复杂度。
- **质量-多样性权衡（Quality-diversity trade-off）**：ROUGE-L 与 Self-BLEU 之间的平衡，WASD 在此权衡上形成更优前沿。

## 可复现要素
- **代码**：开源，https://github.com/aailab-kaist/WASD
- **数据集**：databricks-dolly-15k（蒸馏）、OpenWebText（预训练）、Flores-200、DialogSum、GSM8k、WizardCoder、HumanEval、MBPP——均公开
- **模型**：GPT-2（1.5B/0.3B/0.1B）、OpenLLaMA2（7B/3B）、Qwen2.5（7B/1.5B）、Gemma（7B/2B）——均公开权重
- **关键超参**：$\epsilon=0.001$，$k=8$（Qwen2.5 用 $k=2$），Sinkhorn 迭代 10 步，温度缩放 $T=2$
- **硬件**：单卡 NVIDIA RTX PRO 6000 训练，RTX 3090 评估
