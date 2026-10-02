---
title: "TULIP-TARGETED-LLM-UNLEARNING-AT-LAYERS-IDENTIFIED-PER-INPUT"
source: https://arxiv.org/pdf/2609.34591v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 01:13:48"
field: "大语言模型机器非学习"
keywords: ["machine unlearning", "LLM representation-level unlearning", "formation-readout boundary", "logit lens", "per-input layer selection", "TOFU benchmark", "WMDP"]
innovations: ["将unlearning重新定义为移除formation而非readout，通过hijacking实验验证目标token在中间层已形成", "提出TULIP用logit lens对每个输入动态估计形成-读出边界并在该层施加正交损失", "per-input层选择可作即插即用模块提升GradDiff/NPO/SimNPO/RMU等多项表示基线"]
benchmarks: ["TOFU", "PISTOL", "WMDP"]
---

# 论文速读：TULIP: TARGETED LLM UNLEARNING AT LAYERS IDENTIFIED PER-INPUT

## 一句话总结
TULIP 提出了一种按输入动态选择干预层的表示级 LLM 非学习（unlearning）方法，先通过 hijacking 实验证明答案在中间层形成后由后续层"读出"，再使用 logit lens 定位每样本的"形成-读出"边界并在该层正交化移除目标表示，从而在 TOFU、PISTOL 和 WMDP 上均显著优于已有输出级和表示级基线。

## 研究问题与动机
- **现有表示级方法固定单一干预层**：RMU、Adaptive RMU、LUNAR 等方法对全部遗忘样本使用同一固定层，未考虑知识在 LLM 不同层间分布的异质性。
- **遗忘目标是否应在"形成层"而非"读出层"干预**：输出级方法（GradDiff、NPO 等）仅压制输出，但知识仍可通过中间表示恢复；本文追问：干预点应定位在哪一层？
- **不同输入的知识形成深度存在显著差异**：hijacking 实验显示，相同 forget set 中各样本的目标最早被稳定形成的层位置跨度很大，固定层难以覆盖所有样本。
- **缺乏可复用的 per-input 层定位机制**：实际场景中 oracle（仅用 retain set 训练的模型）不可用，需要仅基于目标模型自身的无监督估计方法。

## 核心贡献（创新点）
1. **将 unlearning 重新定义为"移除形成、保留读出"**：通过 hijacking 实验（将目标模型的中间 hidden state 嫁接至仅用 retain set 训练的 oracle）证明目标 token 在中间层已完全形成，后续层仅做读出解码，且该层的稳定成功层随输入差异显著（Fig. 1(b)）。
2. **提出 TULIP 方法，实现 per-input 层选择**：用 logit lens（目标模型自身未嵌入矩阵投影）估计每个输入的"形成-读出"边界 $\hat{\ell}(x_f)$，并在该层施加正交性损失使隐藏状态与目标 token 的 unembedding 向量正交。
3. **多基准强泛化表现**：在 TOFU、PISTOL、WMDP 三个基准、Llama/Qwen/Zephyr 多模型系列、1B~8B 多规模下均超越全部已有输出级和表示级基线，并在 paraphrase 和量化攻击下保持鲁棒性；per-input 层选择还可作为即插即用模块提升其他方法的 MU 性能。

## 方法详解
- **Hijacking 实验（Section 3.1）**：给定目标模型 $M$（全数据集训练）和 oracle 模型 $\tilde{M}$（仅 retain set 训练），对遗忘样本 $x_f$ 取 $M$ 在第 $\ell$ 层的 hidden state $h_\ell(x_f)$ 嫁接至 oracle，oracle 继续运行剩余层 $\tilde{M}_{\ell+1:L}$ 产生输出。实验显示嫁接后 oracle 对 forget set 的 target accuracy 从 0.63 提升至 1.00，证明目标已在前层形成。
- **理想边界定义（Section 3.2，公式 1）**：以 oracle 为参考，稳定成功层 $\ell^\star(x_f)$ 定义为最早满足"此后所有层均稳定输出目标 token"的最小层号，作为 formation–readout boundary 的理论标准。
- **Logit lens 近似（Section 3.3，公式 2–3）**：实际中 oracle 不可得，TULIP 用目标模型自身的未嵌入矩阵 $W_U$ 做线性投影：$\hat{t}_\ell(x_f) = \arg\max_v [W_U \phi(h_\ell(x_f))]_v$，估计边界 $\hat{\ell}(x_f)$ 为从该层起后续各层 logit lens 预测始终为目标的最低层。
- **正交性遗忘损失（Section 3.4，公式 4）**：在 $\hat{\ell}(x_f)$ 层，计算隐藏状态与目标 token 未嵌入向量 $w_t$ 的余弦相似度平方：$\mathcal{L}_{\text{forget}} = \mathbb{E}[(\langle h_{\hat{\ell}}, w_t \rangle / (\|h_{\hat{\ell}}\|\|w_t\|))^2]$，梯度仅回传至边界层之前，保留读出层不变。
- **Retain 正则项（公式 5）**：在每一层最小化 retain 集上当前模型隐藏状态与冻结参考模型隐藏状态的 MSE，保持整体表征不变。
- **总损失**：$\mathcal{L} = \lambda \mathcal{L}_{\text{forget}} + \mathcal{L}_{\text{retain}}$，$\lambda > 0$ 为可调超参。

## 实验与结果
- **基准**：TOFU（1%/5%/10% forget splits）、PISTOL（合同边 A–C 遗忘）、WMDP-Cyber/Bio（恶意知识遗忘，准确率越接近随机 25% 越好）。
- **模型**：Llama-3.1（1B/3B/8B）、Llama2-7B、Qwen2.5-7B、Qwen3-4B、Zephyr-7B-beta。
- **TOFU 最强结果（Llama-3.1-8B, forget05）**：TULIP FQ=0.89±0.13，MU=0.94±0.00；次优基线 NPO FQ=0.24±0.17，RMU FQ=0.14±0.04，TULIP 的 FQ 提升约 4–6×。
- **PISTOL（Table 2）**：TULIP 在 Llama2-7B 上 FQ=0.80±0.17（次优 SimNPO 0.25±0.12）、Qwen2.5-7B 上 FQ=0.75±0.12（次优 RMU 0.20±0.11）、Qwen3-4B 上 FQ=0.75±0.12（次优 SimNPO 0.66±0.35），MU 在所有设置下均 ≥0.91。
- **WMDP（Table 3, Zephyr-7B-beta）**：Cyber 集 Forget accuracy 0.267（次优 GradDiff 0.277）、Bio 集 Forget accuracy 0.370（次优 RMU 0.452），接近随机 25%；Retain accuracy 保持 0.528。
- **鲁棒性（Table 4, TOFU forget05, Llama-3.1-8B）**：Paraphrase 攻击下 TULIP FQ=0.81±0.13（次优 LUNAR 0.34±0.20）；8-bit 量化下 FQ=0.92±0.04 为最高；4-bit 量化下 FQ=0.30±0.12 仍属较强水平。
- **即插即用效果（Table 5）**：将 per-input 层选择应用于 GradDiff/NPO/SimNPO/RMU 后，四者的 MU 均提升至 ~0.94（Oracle 为 0.943），FQ 也普遍改善。

## 相关工作脉络
- **GradDiff / NPO / SimNPO（输出级）**：直接操作输出概率分布，只能压制生成行为而不消除底层知识，受 paraphrase 攻击时易复现答案（Fig. 4–6 定性示例）。
- **RMU（表示级，固定层）**：将遗忘样本激活重定向至随机方向，对固定层敏感；本文的 per-input 层定位 + 正交损失可视为 RMU 干预思想的精细化。
- **LUNAR（表示级，固定层）**：通过学习"无知识"转向向量重定向激活，常触发拒绝行为（Fig. 4 例：LUNAR 回答"The author's name is not provided"），偏离 unlearning 语义；TULIP 仅移除形成对齐而保留正常生成能力。
- **Logit lens / Tuned lens / J-lens（表征解释工具）**：本文选用 logit lens 因其无需额外训练（Tuned/J-lens 训练耗时 0.06/14.17 GPU 小时），且 AUROC 验证其在定位 formation–readout boundary 上表现最优（Fig. 2）。
- **WMDP（安全知识削减）**：TULIP 首次在 WMDP 上实现同时达到接近随机 Forget accuracy（0.267/0.370 vs 0.25）与保持 Retain accuracy 的效果。

## 局限性与未来方向
- **Logit lens 是线性近似**：作者承认其不能精确定位所有边界，尤其当目标在边界层处非线性可分时正交损失可能失效；更精确的边界估计器是值得探索的方向。
- **4-bit 量化下鲁棒性下降**：TULIP 在 4-bit 量化时 FQ 降至 0.30，仍有较大提升空间。
- **依赖未嵌入矩阵的线性假设**：正交性损失假设目标 token 在边界层线性可解码，若该假设不成立则效果受限。
- **仅适用于 Transformer 架构**：当前方法针对 transformer hidden state 设计，对 MoE 或其他新型架构的推广性需验证。

## 研究启发与可借鉴点
- **Formation-Readout 分解范式**：将 token 生成过程解耦为"形成"和"读出"两阶段，为 unlearning 和其他表示干预任务提供了新的理论视角，可迁移至知识编辑、事实修正等场景。
- **Per-input 层选择作为通用即插模块**：TULIP 的 $\hat{\ell}(x_f)$ 估计可与任意表示级方法结合（Table 5 已验证），可作为 LLM 内部干预任务的通用增强组件。
- **Hijacking 实验设计的复用价值**：用 oracle + 隐藏状态嫁接验证知识形成深度的实验范式，可作为表征可解释性和 unlearning 机理研究的标准评估工具。
- **正交性损失替代随机重定向**：以目标 unembedding 向量为参照施加正交性（而非 RMU 的随机方向），实现了更具针对性的知识删除，该方法可与 LoRA-based 或参数高效 unlearning 结合探索。

## 关键术语表
- **Machine Unlearning**：在不重新训练的情况下修改已训练模型，使其表现得如同从未接触过指定子集训练数据。
- **Formation-Readout Boundary**：模型内部从"目标表示形成"过渡到"读出解码"的分层位置，本文提出的关键干预点概念。
- **Logit Lens**：通过直接将中间层的隐藏状态乘以未嵌入矩阵 $W_U$ 并做归一化，推断该层对 vocabulary 的最可能预测。
- **Forget Quality (FQ)**：TOFU/PISTOL 上衡量遗忘效果的指标，为未学习模型与 oracle 在 Truth Ratio 分布上的 KS 检验 p 值，越高表示越接近 oracle。
- **Model Utility (MU)**：在 retain 集（真实作者问答 + 世界知识）上的平均 ROUGE-L recall，衡量遗忘后保留能力的指标。
- **Truth Ratio**：对每个问题计算正确答案与扰动错误答案的似然比几何均值，用于构建分布以计算 FQ。
- **Hijacking Experiment**：将目标模型的中间 hidden state 嫁接至 oracle，测试 oracle 是否能直接读出目标 token，用于验证 formation 位置。
- **Representation-level Unlearning**：直接干预 LLM 中间隐藏状态而非输出概率的遗忘方法，相较输出级方法能更彻底地消除底层知识。

## 可复现要素
- **数据集**：TOFU（公开）、PISTOL（公开）、WMDP（公开）；具体 split 配置见 Appendix B。
- **代码**：论文声明"code will be made publicly available upon publication"（发表后开源）。
- **权重**：使用 Llama-3.1-8B、Llama-3.2-3B/1B、Llama2-7B、Qwen2.5-7B、Qwen3-4B、Zephyr-7B-beta 等公开基础模型。
- **关键超参**：学习率 1e-5，epoch=10，effective batch size=16（per-device=4，gradient accumulation=4），TULIP 仅调 $\lambda$；各基线超参见 Appendix D Table 6。
- **随机种子**：seed 0 选超参，seeds 0/1/2 三组运行取均值。
- **硬件**：论文未明确提及，但 J-lens 训练提到"14.17 GPU hours on 1,000 WikiText prompts"，推测为多卡 GPU 训练环境。
