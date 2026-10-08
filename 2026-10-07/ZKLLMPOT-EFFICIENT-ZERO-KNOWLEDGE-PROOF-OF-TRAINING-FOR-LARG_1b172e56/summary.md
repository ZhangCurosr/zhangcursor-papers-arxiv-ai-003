---
title: "ZKLLMPOT-EFFICIENT-ZERO-KNOWLEDGE-PROOF-OF-TRAINING-FOR-LARG"
source: https://arxiv.org/pdf/2610.08258v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-08 02:26:07"
field: "零知识机器学习审计"
keywords: ["零知识证明", "大语言模型审计", "训练证明", "zk-SNARK", "Transformer", "commit-then-challenge"]
innovations: ["将训练审计从反向传播轨迹证明转为单次前向评估，证明成本与训练步数无关", "commit-then-challenge 协议顺序消除训练者过拟合审计数据的漏洞", "算子级 sumcheck+lookup 编译覆盖 OPT/Llama/Qwen2.5/DeepSeek-Coder 四种模型族"]
benchmarks: ["OPT-1.3B", "Llama-1.1B", "Qwen2.5-1.5B", "DeepSeek-Coder-1.3B", "13B 规模扩展测试"]
---

# 论文速读：ZKLLMPOT-EFFICIENT-ZERO-KNOWLEDGE-PROOF-OF-TRAINING-FOR-LARG

## 一句话总结
zkLLMPoT 提出了一种零知识框架，通过**前向评估而非验证训练轨迹**来证明已提交的 LLM checkpoint 满足审计者指定的目标函数；其证明成本与训练迭代次数无关，在 1.1–1.5B 参数模型上仅用 41–59 秒即可生成证明，验证时间不足半秒。

## 研究问题与动机
1. **LLM 训练结果难以审计**：模型权重和训练数据均为私有，监管方/用户只能看到模型卡或 API 端点，无法独立验证所宣称的能力或安全属性是否真实成立。
2. **已有 ZK 工作聚焦推理完整性**：如 zkLLM、zkGPT 等仅证明"单次推理输出正确"，无法对 checkpoint 级别的目标函数（如 loss、公平性差距、成员推断风险）进行审计。
3. **已有 ZK 训练证明（zkPoT）成本过高**：Kaizen 在 VGG-11 上每训练迭代需约 15 分钟证明时间；Confidential-DPproof 在 CIFAR-10 上需约 100 小时；直接编码完整反向传播对 Transformer 规模完全不现实。
4. **简单方案存在过拟合漏洞**：若评估数据在 checkpoint 提交前已知，训练者可针对性修改权重以通过审计，导致证明失去意义。

## 核心贡献（创新点）
1. **结果证明（outcome-attestation）范式**：将认证目标从"训练过程正确"转为"提交的 checkpoint 在审计者数据上达到声明目标值"，从关系中完全排除反向传播与优化器，使证明成本独立于训练步数。
2. **提交-挑战（commit-then-challenge）协议顺序**：训练者先固定电路并提交权重，之后审计者再选择挑战序列 S 作为公共输入，从根本上杜绝训练者针对审计数据过拟合。
3. **面向 Transformer 的算子级 ZK 电路优化**：将线性算子编译为超立方体内积 sumcheck，将非线性算子（Norm、激活、softmax）编译为查表（logup）参数数，支持跨模型族（OPT/Llama/Qwen2.5/DeepSeek-Coder）的统一编译。
4. **审计者指定目标接口**：默认 next-token NLL，同时可表达分类交叉熵、公平性差距、成员推断检查、安全拒答分数等，只需替换终端目标 gadget，前向电路不变。
5. **与已有工作的本质区别**：相比 zkLLM/zkGPT（每查询推理证明）和 Kaizen/DPproof（每步反向传播证明），zkLLMPoT 是首个在 Transformer 尺度上实现**一次性前向评估**、且证明时间不随训练步数增长的零知识训练审计方案。

## 方法详解
- **统一架构建模**：将 OPT、Llama、Qwen2.5、DeepSeek-Coder 统一建模为 decoder-only Transformer 家族 $\mathcal{A}$，共享计算骨架：
  $$h^{(0)} = \text{Embed}_\theta(x), \quad h^{(\ell+1)} = \text{Block}_\theta^{(\ell)}(h^{(\ell)}), \quad z = h^{(L)}W_\text{out}$$
  每个 Block 使用 pre-norm 残差形式：$u = h + \text{Attn}_\theta(\text{Norm}(h); \text{Pos})$，$h' = u + \text{FFN}_\theta^\Phi(\text{Norm}(u))$。
- **目标函数**：预训练/指令微调统一使用掩码 next-token NLL：$\mathcal{L}_\text{LM}(\theta;x,m) = -\sum_{i=1}^{n-1} m_{i+1}\log p_\theta(x_{i+1}|x_{\le i})$。
- **多项式承诺方案**：采用 Hyrax 风格的 Pedersen 承诺，将张量编码为 $\mathbb{F}_p$ 上的表 $T:\{0,1\}^n\to\mathbb{F}_p$（有符号定点量化 + 模 $p$ + 补齐至 $2^n$），每条行以盲化项 $\rho_i h$ 隐藏。
- **Sumcheck 归约**：每个算子的输入-输出关系转化为形如 $\sum_{b\in\{0,1\}^n} g(b)=c$ 的 sumcheck 实例；线性层（矩阵乘）度 $\mu=2$，Norm 度 $\mu=3$，softmax 度 $\mu=4$（协议中最大）。
- **非线性查表（logup）**：GELU/SiLU/ReLU 等通过公共表 $T_\text{in}, T_\text{out}$ 的对数导数恒等式实现：$\sum_i \frac{1}{\beta+S_\text{com}[i]} = \sum_j \frac{m[j]}{\beta+T_\text{com}[j]}$。
- **Softmax 实现**：将指数分解为混合基数表示，每位对应一张公共查表 $T_k[d]$，逐位乘法由 sumcheck 验证（度 $\mu=K-L+1=4$），是整个协议中最昂贵的算子，占总证明时间 56–64%。
- **协议流程（Algorithm 1）**：①审计者固定 $\mathcal{L}$ 和接受区间；②训练者提交 $[\![\theta]\!]$；③审计者选 $S$；④训练者计算 $v$ 并提交 $[\![v]\!], [\![H]\!]$；⑤逆序逐层 sumcheck 归约；⑥审计者检查链式一致性；⑦验证公开表和打开承诺；⑧接受 iff 全部通过且 $v\in[\underline{v},\overline{v}]$。
- **安全性**：定理 1 给出 soundness 上界 $\varepsilon\le\sum_{a\in\mathcal{C}_{A,\mathcal{L}}}\varepsilon_a + \varepsilon_\text{ipa}^\text{tot}+\text{Adv}_\mathbb{G}^\text{DL}(t)$，数值评估在 T=512 时达到约 127 比特安全强度。

## 实验与结果
- **数据集/模型族**：OPT、Llama、Qwen2.5、DeepSeek-Coder，覆盖 0.125B–13B 共 14 个配置；序列长度 T=512，batch size=1。
- **硬件**：NVIDIA A100-PCIE (40GB) + Intel Xeon (80 threads)，CPU 侧验证器（mcl + BLS12-381）。
- **主要结果（Table 1）**：

| 系统 | 模型 | 参数 | Commit | Prove | Verify | Proof |
|---|---|---|---|---|---|---|
| zkPoT | logistic regression | 1K | 3690s | 600s | 30s | 350MB |
| Kaizen | VGG-11 | 10.1M | 218s | 882s/iter | 130ms | 1.63MB |
| zkLLMPoT (ours) | OPT-1.3B | 1.3B | 175s | 58.8s | 428ms | 17.0MiB |
| zkLLMPoT (ours) | Llama-1.1B | 1.1B | 138s | 53.2s | 423ms | 17.3MiB |
| zkLLMPoT (ours) | Qwen2.5-1.5B | 1.5B | 206s | 42.4s | 416ms | 19.9MiB |
| zkLLMPoT (ours) | DeepSeek-Coder-1.3B | 1.3B | 171s | 41.0s | 406ms | 16.9MiB |

- **最强结果**：DeepSeek-Coder-1.3B 证明时间最短（41.0s），验证均 < 430ms，证明大小约 17–20 MiB。
- **扩展性（Figure 3）**：参数规模扩大 104×（0.125B→13B），证明时间仅增长 8.3×（15.7s→130.6s）；commit 约 124–142s/十亿参数。
- **算子分解（Figure 2）**：线性投影+FFN+输出头合计 4.1–6.0s；Norm 2.6–3.1s；元素级激活 3.0–5.0s；softmax 为最大开销（占 56–64%）；注意力 score matmul 4.6–10.3s。
- **对比基线**：Kaizen 每迭代 882s（VGG-11），DPproof 100h（CIFAR-10），本文在参数规模大两个数量级的前提下仍远优于基线。

## 相关工作脉络
1. **zkPoT (Garg et al., 2023)**：首个实用化 zk 训练证明，但仅限逻辑回归；本文将其扩展至 Transformer 尺度且排除反向传播。
2. **Kaizen (Abbaszadeh et al., 2024)**：对深度网络证明完整梯度下降+反向传播，每迭代约 882s（VGG-11）；本文证明的是单次前向评估而非优化轨迹。
3. **Confidential-DPproof (Shamsabadi et al., 2024)**：证明差分隐私训练保证，CIFAR-10 上需 100h；本文不涉及 DP，但证明目标为更通用的 checkpoint 属性。
4. **zkLLM (Sun et al., 2024) / zkGPT (Qu et al., 2025)**：面向 LLM 推理完整性的零知识证明；本文聚焦训练/微调后 checkpoint 的审计，而非每次 serving 查询。
5. **ZKAudit (Waiwitlikhit et al., 2024)**：第三方审计框架，需审计者重执行训练或访问模型权重；本文在保证隐私的同时实现无需重跑的简洁证明。
6. **zkCNN/Mystique (Liu et al., 2021; Weng et al., 2021)**：CNN/通用神经网络推理证明；本文将其 sumcheck+lookup 技术体系推广至大规模 Transformer 算子。

## 局限性与未来方向
- **算子覆盖不全**：当前仅覆盖 decoder-only Transformer 的核心算子（线性、Norm、FFN、注意力、softmax），未包含 MoE 路由、KV cache 压缩、长序列 attention 变体等。
- **固定点量化精度**：使用有符号定点量化替代浮点，可能引入误差；接受区间的校准需依赖任务特定的标定集。
- **单审计者假设**：当前为两方协议（训练者+审计者），未支持多方联合审计或下游传递验证。
- **序列长度限制**：实验仅在 T=512 下评估，更长序列的缩放行为尚待验证（虽 figure 3 暗示良好趋势）。
- **未来方向**（论文自述）：①更快的算子设计与更紧融合以支持更大模型和更长序列；②更灵活的自定义审计目标接口；③多方（多训练者/审计者/下游验证者）协议设计。

## 研究启发与可借鉴点
1. **"结果证明"替代"过程证明"的思路**：将审计目标从训练轨迹转移到提交后的一次性前向评估，避免反向传播电路的爆炸性增长——此思路可迁移至任何需要审计模型产出属性的场景（如 RLHF 打分、扩散模型去噪质量）。
2. **commit-then-challenge 顺序设计**：训练者先绑定电路和权重，再暴露挑战数据，从协议层面消除过拟合风险——该模式可作为可信评估的标准协议模式。
3. **softmax 混合基数查表分解**：将指数函数分解为低位查表+高位乘法，是 ZK 中对昂贵非线性函数的有效编译策略，可复用于其他涉及 softmax/exp 的 ML 证明系统。
4. **跨模型族的统一电路编译**：通过 (Norm, Pos, Attn, Φ) 四个槽位参数化，一套电路框架覆盖四种主流模型族，降低了工程与维护成本，类似思路可用于其他架构族的统一证明。
5. **与团队方向的结合机会**：可将本工作的 commitment 机制与模型的隐私保护微调（如联邦学习场景下的 checkpoint 审计）结合，或将自定义目标接口扩展至 RAG 系统的检索质量审计。

## 关键术语表
- **zk-SNARK**：零知识简洁非交互式知识论证，允许证明者向验证者证明某 NP 关系成立而不泄露见证，且验证时间远小于 computation 本身。
- **Sumcheck 协议**：将高维张量求和归约为单点评估的多轮交互协议，每轮维度减半， prover 工作量线性于表大小。
- **Logup 查表论证**：利用对数导数恒等式 $\sum \frac{1}{\beta+S[i]} = \sum \frac{m[j]}{\beta+T[j]}$ 验证输入-输出表映射关系的 ZK 原语。
- **Commit-then-challenge**：协议中先提交后挑战的顺序，防止 prover 在知晓挑战数据后调整见证。
- **Hyrax 承诺方案**：将表视为 $2^{n_1}\times 2^{n_2}$ 矩阵并按行 Pedersen 承诺，Opening 通过内积论证压缩至 $O(n)$ 群元素。
- **Outcome attestation**：只证明已提交 checkpoint 满足某目标函数值，不涉及训练过程本身，从而将证明成本与训练步数解耦。
- **Next-token NLL**：语言模型标准预训练目标，即下一 token 负对数似然，本文用作默认审计目标。
- **接受区间 $[v, \overline{v}]$**：审计者声明的目标值容差范围，最终验证要求证明值落入该区间。

## 可复现要素
- **数据集**：未使用新数据集，实验基于公开模型配置生成合成定点张量；公开模型为 OPT、Llama、Qwen2.5、DeepSeek-Coder（各含自身许可证）。
- **代码/权重是否开源**：论文声明**审阅期结束后开源**，包括协议代码和复现脚本（Reproducibility Statement）。
- **关键超参**：序列长度 T=512，batch size=1，定点量化比例 $s$（公开），BLS12-381 曲线，security parameter 对应约 127 比特。
