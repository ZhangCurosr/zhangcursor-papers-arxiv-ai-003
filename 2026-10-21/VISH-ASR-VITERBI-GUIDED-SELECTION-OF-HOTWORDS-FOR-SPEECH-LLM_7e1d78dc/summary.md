---
title: "VISH-ASR-VITERBI-GUIDED-SELECTION-OF-HOTWORDS-FOR-SPEECH-LLM"
source: https://arxiv.org/pdf/2610.27867v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-06 02:57:41"
---

# 论文速读：VISH-ASR: VITERBI-GUIDED SELECTION OF HOTWORDS FOR SPEECH-LLM CONTEXTUAL ASR

## 一句话总结
本文提出 VISH（Viterbi-guided Selection of Hotwords），一种无需额外训练的即插即用热词预筛选框架，直接利用帧级 CTC 后验分布进行候选词约束搜索与声学验证。M-VISH 通过共享前缀 Trie 与 top-K 剪枝实现大规模词表的高效搜索，Bi-MVISH 进一步引入双向验证提升鲁棒性，在五个基准上显著优于现有假设过滤与学习检索基线。

## 研究问题与动机
- **核心问题**：Speech-LLM 语境化 ASR 需从数千候选热词中预筛选出紧凑子集注入 prompt，但候选词表越大，无关词干扰越强，易引发热词替换与背景词偏差。
- **离散假设过滤的缺陷**：CTC-Assistant 等方法依赖第一遍离散识别假设进行词匹配，若候选词未出现在假设中则被直接丢弃，导致真实声学证据丢失。
- **学习检索模型的负担**：GLCLAP、BR-ASR、CLAR 等方法需独立训练语音-文本检索器，增加系统复杂度、训练成本与部署门槛，难以快速适配新领域。
- **WFST/Lattice 方法的局限**：传统偏置方法依赖解码侧有限状态机或特定搜索图结构，难以直接嵌入黑盒式 Speech-LLM 流水线。

## 核心贡献（创新点）
- **提出后验空间的无训练约束搜索框架**：将热词预筛选形式化为候选词特定的 CTC 后验约束 Viterbi 搜索，无需离散假设匹配或额外模型训练，从根本上避免假设错误导致的证据丢失。与 CTC-Assistant 的本质区别在于证据来源从离散 token 序列转向连续帧级后验分布。
- **设计 M-VISH 共享前缀批量验证**：通过构建候选词前缀 Trie 并在帧同步搜索中实施 top-K 剪枝，将独立候选打分复杂度从 $O(RN\bar{L})$ 降至 $O(RK_p\bar{L})$，实现千级词表的实时声学验证。与独立逐词打分方法的本质区别在于前缀计算复用与活性状态剪枝。
- **提出 Bi-MVISH 双向互补验证**：利用 CTC collapse 与序列反转的可交换性，对后验与候选序列同步反转执行 M-VISH，取正反向平均得分缓解热词起始段声学薄弱导致的验证低估。与单向搜索的本质区别在于以极低代价引入方向互补，提升边界模糊候选的判别稳定性。

## 方法详解
- **CTC 后验获取与 PSD 过滤**：使用下游 ASR 内部编码器或冻结的外部 CTC 打分器（如 SenseVoice-Small）获取帧级后验 $p_t(k)$。通过峰值抑制解码（PSD，阈值 $\tau_b$）剔除 blank 概率过高的帧，得到保留帧序列 $\tilde{\mathbf{p}}$，压缩搜索空间。
- **候选词约束 Viterbi 打分**：对候选词 $w$ 的空白插值 CTC token 序列 $\mathbf{y}'$，在 $\tilde{\mathbf{p}}$ 上执行带根状态重置的约束 Viterbi 搜索，状态转移遵循标准 CTC stay/step/skip 规则。累积最优路径对数后验和 $m_w$，并以路径中非 blank 对齐帧数 $N_w$ 归一化得 $s_w = m_w / N_w$，消除词长时长偏差。
- **M-VISH 共享前缀 Trie 搜索**：将全部候选词组织为前缀 Trie。每帧仅对当前 top-$K_p$ 高置信度活跃前缀节点执行状态更新，共享前缀的候选词合并计算；到达终止节点时更新对应候选词的 $(m_w, N_w)$。算法伪代码见原文 Algorithm 1。
- **Bi-MVISH 双向融合**：将 $\tilde{\mathbf{p}}$ 与所有候选 CTC 序列反转，构建反向 Trie 执行相同搜索流程，获得反向得分 $s_{w,\mathrm{bwd}}$。最终得分取双向平均：$s_{w,\mathrm{bi}} = \frac{1}{2}(s_{w,\mathrm{fwd}} + s_{w,\mathrm{bwd}})$。Bi-MVISH 为段级流式（非严格逐帧流式），但验证鲁棒性显著更强。

## 实验与结果
- **数据集**：STOP1（人名）、STOP2（地名）、AISHELL-1-NE（中文命名实体）、LibriSpeech test-clean-2k、test-other-2k（各固定 2000 候选词，含真实稀有词与随机干扰）。
- **评估基线**：GLCLAP、CLAR、CTC-Assistant、BR-ASR、Oracle (GT Bias)。评估指标包括 Best-F1（预筛选）、WER/B-WER（LibriSpeech）、CER/B-CER（AISHELL-1-NE）。
- **主要结果**：
  - **预筛选精度**：M-VISH 在 5 个基准上相比重制的 GLCLAP 基线绝对提升 0.9%~11.7%（STOP2 提升最大 +11.7%，Libri-other-2k 提升 +5.3%）；Bi-MVISH 在 4/5 基准上进一步绝对提升最高 1.1%。
  - **下游语境化 ASR**：结合 Whisper-medium 时，Bi-MVISH 将 Libri-clean B-WER 从 7.6% 降至 6.9%，Libri-other 从 18.2% 降至 17.3%，AISHELL-1-NE B-CER 从 9.1% 降至 7.6%，整体 WER/CER 未恶化。结合 Prompt-SLAM-ASR 时，B-WER 同样显著下降。
  - **后验来源对比**：外部冻结 SenseVoice-Small 提供更高 Best-F1 与更低 B-WER；内部 WavLM-Large 无额外参数但精度略低，体现精度与计算开销的权衡。
  - **搜索延迟**：2000 候选词时，M-VISH 延迟 203 ms，较 CTC-Assistant（817 ms）降低 75.2%；Bi-MVISH 为 400 ms，仍较基线快 51.0
