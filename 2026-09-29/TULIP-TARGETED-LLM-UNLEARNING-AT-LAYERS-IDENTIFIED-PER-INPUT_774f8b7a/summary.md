---
title: "TULIP-TARGETED-LLM-UNLEARNING-AT-LAYERS-IDENTIFIED-PER-INPUT"
source: https://arxiv.org/pdf/2609.34591v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 01:13:38"
field: "大语言模型安全与可信计算"
keywords: ["Machine Unlearning", "LLM", "Representation-level Unlearning", "Logit Lens", "Layer Selection"]
innovations: ["提出 formation-readout 分解框架，证明目标 token 在中间层已形成", "设计 TULIP 方法，通过 logit lens 为每个样本动态定位干预层并施加正交损失", "证明 per-input 层选择可显著提升现有方法的模型效用"]
benchmarks: ["TOFU", "PISTOL", "WMDP"]
---

# 论文速读：TULIP-TARGETED-LLM-UNLEARNING-AT-LAYERS-IDENTIFIED-PER-INPUT

## 一句话总结
本文提出 TULIP（Targeted Unlearning at Layers Identified Per-input），一种表示级 LLM 未学习方法。该方法为每个 forget 样本动态定位"信息形成-读取"边界层，并在该层消除 hidden state 与目标 token 的对齐，从而精准删除目标知识同时保留模型整体能力。

## 研究问题与动机
- 现有表示级未学习方法（如 RMU、LUNAR）对 entire forget set 使用单一固定干预层，但知识在 Transformer 层间分布式存储，不同样本的信息形成深度差异很大。
- 输出级方法仅压制生成结果而不真正删除知识，模型仍可通过隐式记忆复现答案（如通过 paraphrase 攻击恢复）。
- 表示级方法虽更直接干预计算过程，但固定层干预可能过早或过晚：过早则目标尚未形成、无法有效删除；过晚则侵入"读取"层，损害模型通用能力。
- 需要回答两个核心问题：（1）未学习应针对知识的"形成"还是"读取"？（2）每个样本的最佳干预层是否应不同？

## 核心贡献（创新点）
- **提出 formation-readout 分解框架**：通过 hijacking 实验证明目标 token 在中间层已形成，后续层仅执行读取操作，因此未学习应仅针对形成阶段而非整个推理链。
- **设计 TULIP 方法**：利用 logit lens 为每个 forget 样本估计形成-读取边界层，在该层施加正交损失消除 hidden state 与目标 token unembedding 向量的对齐，实现逐样本精准干预。
- **验证逐层选择的必要性**：实验表明固定层干预在 FQ 和 MU 上均逊于 per-input 动态层选择，且 TULIP 的层估计偏差（提前或延后）会导致性能下降。
- **证明 TULIP 的泛化性与兼容性**：在 TOFU、PISTOL、WMDP 三个基准上，TULIP 在 Llama、Qwen、Zephyr 系列模型（1B-8B）上 consistently 优于所有输出级和表示级基线，且可无缝嵌入现有方法作为插件提升其 MU。

## 方法详解

**Setup**：给定 Transformer 语言模型 M，含 L 层，对输入上下文 x，记第 ℓ 层末位置的 hidden state 为 h_ℓ(x) ∈ ℝ^d，目标 token t ∈ V，unembedding 矩阵 W_U ∈ ℝ^{|V|×d}，第 v 行 w_v 为 token v 的 unembedding 向量。

**Hijacking 实验与形成-读取分解**（§3.1）：
- 训练 target model M（含 forget set D_f）和 oracle model Õ（仅含 retain set D_r）。
- 将 M 在层 ℓ 的 hidden state h_ℓ(x_f) graft 入 Õ，计算 Õ_{ℓ+1:L}(h_ℓ(x_f))。若 Õ 输出目标 token t，则 hijack 成功。
- 结果：oracle 在 forget set 上的 target accuracy 从 0.63 提升至 1.00，证明目标 token 在中间层已完全形成，后续层仅需解码。

**形成-读取边界的理想定义**（§3.2）：
$$\ell^*(x_f) = \min\{\ell : \arg\max_{v \in V}[\tilde{M}_{k+1:L}(h_k(x_f))]_v = t, \forall k \geq \ell\}$$
即最早满足"此后所有层的 hijack 均成功"的层。

**Logit Lens 近似**（§3.3）：
由于 oracle 不可用，TULIP 用目标模型自身的 logit lens 近似边界：
$$\hat{t}_\ell(x_f) = \arg\max_{v \in V}[W_U \phi(h_\ell(x_f))]_v$$
$$\hat{\ell}(x_f) = \min\{\ell : \hat{t}_k(x_f) = t, \forall k \geq \ell\}$$
其中 φ(·) 为最终 layer norm，W_U 为目标模型的 unembedding 矩阵。

**遗忘损失**（§3.4）：
在边界层 ħ(x_f) 施加正交损失，使 hidden state 与目标 token unembedding 向量正交：
$$\mathcal{L}_{\text{forget}} = \mathbb{E}_{(x_f, t) \sim D_f}\left[\left(\frac{\langle h_{\hat{\ell}(x_f)}(x_f), w_t \rangle}{\|h_{\hat{\ell}(x_f)}(x_f)\|\|w_t\|}\right)^2\right]$$
梯度仅回传到 ħ(x_f) 之前的层，不影响后续读取层。

**保留损失**：
$$\mathcal{L}_{\text{retain}} = \mathbb{E}_{x_r \sim D_r}\left[\frac{1}{L}\sum_{\ell=1}^{L}\|h_\ell(x_r) - h_\ell^{\text{ref}}(x_r)\|_2^2\right]$$
保持所有层 retain 数据的 hidden state 接近冻结模型。

**总目标**：$\mathcal{L} = \lambda \mathcal{L}_{\text{forget}} + \mathcal{L}_{\text{retain}}$，λ 为平衡超参。

## 实验与结果

**数据集**：
- **TOFU**：4,000 个虚构作者 QA 对，设置 forget01/05/10（1%/5%/10% 作为 forget set），评估指标 FQ（KS-test p-value，越高越好）和 MU（ROUGE-L recall，越高越好）。
- **PISTOL**：400 个合成 QA 对，基于知识图谱中的实体间合同关系，评估同样使用 FQ 和 MU。
- **WMDP**：来自生物安全与网络安全原始文档的 3,668 道四选择题，评估 forget accuracy（越低越好，接近 25% 随机）和 retain accuracy（越高越好）。

**基线方法**：输出级（GradDiff、NPO、SimNPO）和表示级（RMU、LUNAR）。

**主要结果**（Llama-3.1-8B, TOFU forget05）：
- TULIP 取得 FQ = 0.89 ± 0.13，显著优于所有基线（RMU 0.14、LUNAR 0.54、NPO 0.24）。
- MU = 0.94 ± 0.00，与 RMU 持平且高于其他基线。
- 在 forget01/10 和 Llama-3B/1B 上均取得最佳 FQ。

**跨模型泛化**（PISTOL）：
- Llama2-7B：FQ 0.80 ± 0.17（最优），MU 0.81 ± 0.06。
- Qwen2.5-7B：FQ 0.75 ± 0.12（最优）。
- Qwen3-4B：FQ 0.75 ± 0.12（最优），MU 0.91 ± 0.00（最优）。

**WMDP**（Zephyr-7B-beta）：
- Cyber 集：F 0.267（最优，接近 0.25 随机）。
- Bio 集：F 0.370（最优），R 0.528。

**鲁棒性**（TOFU forget05, Llama-3.1-8B）：
-  paraphrase 攻击：FQ 0.81 ± 0.13（最优）。
- 8-bit 量化：FQ 0.92 ± 0.04（最优）。
- 4-bit 量化：FQ 0.30 ± 0.12，与 RMU/LUNAR 同属最稳健方法。

**插件效果**（Table 5）：将 TULIP 的 per-input 层选择嵌入各基线后，GradDiff/NPO/SimNPO 的 MU 大幅提升至 ~0.94（接近 oracle 的 0.943）。

## 相关工作脉络
- **GradDiff (Jang et al., 2023)**：输出级方法，对 forget 集做梯度上升、retain 集做梯度下降；局限在于仅改变输出概率分布，不删除底层知识。
- **NPO/SimNPO (Zhang et al., 2024; Fan et al., 2025)**：基于偏好优化的未学习方法，通过降低 forget 响应的 likelihood 实现遗忘；对 paraphrase 鲁棒性差。
- **RMU (Li et al., 2024)**：表示级方法，将 forget 表示转向随机向量；但仍使用固定干预层，且转向随机方向可能损害语义一致性。
- **LUNAR (Shen et al., 2025)**：基于 activation redirection 的表示级方法，将 forget 激活重定向到"不知晓"区域；偏向生成拒绝语料，偏离未学习的"假装没学过"目标。
- **Logit Lens (nostalgebraist, 2020)**：通过将中间 hidden state 投影到词汇空间来读取内部表征的方法；本文将其扩展用于定位形成-读取边界。
- **Tuned Lens / J-lens (Belrose et al., 2025; Gurnee et al., 2026)**：经额外训练的 lens 方法；本文验证 logit lens 在边界检测上 AUROC 最高，且无需额外训练成本。

## 局限性与未来方向
- **Logit lens 是线性近似**：假设目标 token 在边界层线性可解码，可能不适用于非线性形成的知识。
- **4-bit 量化鲁棒性下降**：TULIP 在 4-bit 量化下 FQ 降至 0.30，仍有提升空间。
- **未来方向**：设计更精确的边界估计器，以及与边界估计器协同优化的未学习目标函数。

## 研究启发与可借鉴点
- **Formation-Readout 分解视角**：将知识生产过程解耦为"形成"和"读取"两阶段，为未学习提供更细粒度的干预定位思路，可迁移到其他表示干预任务。
- **Per-input 动态层选择作为通用插件**：TULIP 的边界估计模块可独立使用，显著提升任意基线方法的 MU，为现有方法的改进提供低成本方案。
- **正交损失设计的简洁性**：通过直接消除 hidden state 与目标 token unembedding 的对齐实现遗忘，避免了转向随机向量或引入额外参考向量的复杂性，设计优雅且易于复现。
- **Hijacking 实验验证因果性**：通过 graft 实验明确证明了中间层已编码目标信息，为方法设计提供了扎实的动机支撑，这种验证范式值得借鉴。

## 关键术语表
- **Machine Unlearning**：使模型行为如同从未使用过特定训练数据，同时保留模型在其他数据上的能力。
- **Formation-Readout Boundary**：目标 token 在神经网络中层间被"形成"（encode）与后续被"读取"（decode）的分界处。
- **Logit Lens**：将任意层的 hidden state 直接投影到词汇空间，通过 argmax 预测该层已决定的 top token。
- **Forget Quality (FQ)**：基于 KS-test p-value 衡量未学习后模型与仅含 retain set 的 oracle 的分布相似度。
- **Model Utility (MU)**：以 ROUGE-L recall 衡量模型在 retain set 及通用知识上的保留程度。
- **Oracle Model**：仅用 retain set 训练的模型，作为"理想未学习"效果的参考基线。
- **Negative Preference Optimization (NPO)**：将 forget 响应视为负样本，通过偏好优化降低其 likelihood 以实现遗忘。
- **Representation-level Unlearning**：直接干预模型中间表示而非仅修改输出分布的未学习方法。

## 可复现要素
- **数据集**：TOFU、PISTOL、WMDP 均为公开基准；论文未提及自定义数据集。
- **代码/权重**：论文声明"code will be made publicly available upon publication"。
- **关键超参**：学习率 1e-5，batch size 16，训练 10 epochs；TULIP 唯一超参为 λ（forget loss 权重），其他方法超参见 Appendix D Table 6。
