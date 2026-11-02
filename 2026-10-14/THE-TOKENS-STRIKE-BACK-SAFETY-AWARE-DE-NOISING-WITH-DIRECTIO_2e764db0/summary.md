---
title: "THE-TOKENS-STRIKE-BACK-SAFETY-AWARE-DE-NOISING-WITH-DIRECTIO"
source: https://arxiv.org/pdf/2610.17636v1.pdf
model: agnes-2.5-flash
chunks: 4
summarized_at: "2026-11-02 09:07:41"
field: "扩散语言模型安全对齐"
keywords: ["扩散语言模型", "安全对齐", "推理时引导", "激活空间", "选择性干预", "文本扩散"]
innovations: ["激活空间位置级安全门控调制 SAD 更新", "选择性干预策略平衡安全-质量-成本", "remasking 与 warm-up 增强机制"]
benchmarks: ["RealToxicityPrompts", "BeaverTails", "JailbreakBench", "XSTest", "AdvBench", "HarmBench"]
---

# 论文速读：THE-TOKENS-STRIKE-BACK-SAFETY-AWARE-DE-NOISING-WITH-DIRECTIO

## 一句话总结
本文提出 **SADDER**（Selective Activation Steering in Diffusion Decoding for Refusal），一种面向扩散大语言模型（MDLMs）的推理时安全引导方法，通过在 embedding/activation 空间提取位置级安全信号并门控调制现有 SAD 更新，实现选择性干预，在安全性、生成质量与推理成本之间取得更优平衡。

## 研究问题与动机
- **TDMs 安全性研究滞后**：文本扩散模型（TDMs）的安全对齐研究远落后于自回归 LLMs，现有安全机制（微调、过滤、推理时解码修改）大多依赖 AR 解码假设，无法直接迁移到 TDMs。
- **SAD 的局限**：Safety-Aware Denoiser (SAD) 仅在 token 概率空间做安全引导，未利用模型内部更丰富的语义表示；且用常数 η 近似 β*，缺乏理论安全保证。
- **全序列均匀干预损害质量**：对全序列所有位置均匀施加 steer 会损害文本流畅度与质量，需要更精细的选择性策略。
- **计算开销问题**：现有方法（如 DiffuGuard 多步循环、BoN 多次重采样）计算成本高昂，缺乏轻量级插拔式解决方案。

## 核心贡献（创新点）
- **激活空间位置级安全引导**：在 embedding/activation 空间构建有害-安全对比方向向量，提取每个 token 位置的安全风险评分，实现位置级选择性干预，而非仅操作 token 概率分布。
- **门控调制 SAD 更新**：以二进制门控 $a_{k,l}^{(t)}$ 控制哪些位置需要安全修正，仅在危险信号 > 0 时才估算 unsafe denoiser，节省计算并减少不必要的干预。
- **可选 remasking 与 warm-up 机制**：支持重新屏蔽已确定 token 并再次推理以修正早期有害输出；预热阶段使用 full SAD 提升引导稳定性，后续切换至 selective 模式。
- **理论等价性分析**：推导 position-wise 目标分布 $\mathcal{D}_{\mathrm{safe}}^{(l)}$ 的精确条件，并证明当剩余位置安全权重与当前 token 条件不相关时，逐位置 steer 可近似实现全局 safe 采样（Theorem A.1）。

## 方法详解
**SADDER 两阶段设计：**

**Phase 1 — 选择风险位置（门控）**
- 对每个 prompt 构造一对有害/安全响应，计算各层隐藏表示平均值 $\bar{h}^{\mathrm{harm}}_k, \bar{h}^{\mathrm{safe}}_k$
- 构造对比方向 $\hat{v}_k = (\bar{h}^{\mathrm{harm}}_k - \bar{h}^{\mathrm{safe}}_k) / \|\cdot\|_2$ 与中点 $m_k = (\bar{h}^{\mathrm{harm}}_k + \bar{h}^{\mathrm{safe}}_k)/2$
- 中心化评分：$s_{k,l}^{(t)} = \langle h_{k,l}^{(t)} - m_k, \hat{v}_k \rangle$，正值表示位于"有害侧"
- 门控：$a_{k,l}^{(t)} = \mathbb{1}\{s_{k,l}^{(t)} > 0\}$，零分边界为有害/安全均值表示的欧氏中垂面

**Phase 2 — 选择性 SAD 修正**
$$\text{Sadder Denoiser} = \prod_{l=1}^{L}\left[\mathbb{E}_{\mathcal{D}}[x_0^l|x_t] + \eta\beta(x_t)a_{k,l}^{(t)}\left(\mathbb{E}_{\mathcal{D}}[x_0^l|x_t] - \mathbb{E}_{\mathcal{D}_{\mathrm{unsafe}}}[x_0^l|x_t]\right)\right]$$
- MDLMs 仅对 masked 位置计算门控；USDMs 每步所有 L 个位置均参与
- 仅危险信号 > 0 时才估算 unsafe denoiser，显著节省计算
- 理论保证：Theorem 3.1 推导 position-wise 目标分布，精确条件为 $\eta\beta(x_t)a_{k,l}^{(t)} = \frac{\pi_{t,l}(x_t)}{1-\pi_{t,l}(x_t)}$

**可选增强机制：**
- **Remasking**：生成过程中对先前确定的 token 重新赋予 mask 并在后续步骤再次推理，修正早期有害输出；支持 top-b 与 aligned 两种策略
- **Warm-up**：前 $w$ 步使用 full SAD（不加选择），后续步切换到 selective SADDER，提升引导稳定性

## 实验与结果
**数据集与基线模型：**
- 基线模型：LLaDA-8B-Base/Instruct、Dream-7B-Base/Instruct
- 评估基准：RTP（0–299 开发，300–899 评估）、BeaverTails、JailbreakBench、AdvBench、HarmBench、WildJailbreak、XSTest
- 攻击方法：DIJA、PAD、Zero-shot、DiffuGuard

**主要结果（MDLMs - LLaDA/RTP）：**
| 方法 | Unsafe Rate |
|---|---|
| Unmodified | 38.6% |
| Full SAD | 32.4% |
| **SADDER** | **29.4%** |
| SADDER + warm-up | 31.2% |
| CSD† | 22.4%（pPL 严重恶化） |
| BoN-8 | 19.0%（NFE=256） |

**主要结果（MDLMs - LLaDA/BeaverTails）：**
| 方法 | Unsafe Rate |
|---|---|
| Unmodified | 38.6% |
| Full SAD | 27.1% |
| **SADDER** | **29.9%** |
| SADDER + warm-up | 24.4% |

**USDMs 结果（UDLM）：** SADDER unsafe rate 45.9%，优于 unmodified 68.5%，接近 full SAD 的 47.4%，且 benign BERTScore 80.92 vs full SAD 80.49，无明显质量损失。

**Duo（USDM）结果：** full SAD 在 unsafe rate 上全面优于 SADDER，但 SADDER benign BERTScore 更高（81.68 vs 80.76）。

**Jailbreak 防御：**
- DIJA：SADDER unsafe 77.7% vs unmodified 82.0%
- XSTest over-refusal：SADDER 41.1% vs unmodified 55.5%（降低 14.4pp）
- PAD：SADDER+warm-up HB ASR 55.0% vs unmodified 64.0%（降 9pp），WG 拒绝率降至 17.9%（降 37.6pp）

**核心结论：** SADDER 在 MDLMs 上统一提升 unsafe rate；在 USDMs 上以相同 NFE 代价换取更低 unsafe rate 且无损 BERTScore；在 Duo 上虽 unsafe rate 略逊但 benign quality 显著提升；Jailbreak 实验中大幅降低 over-refusal。

## 相关工作脉络
- **SAD (Yusuf et al., 2026)**：无训练 token 概率空间引导，使用 500~1000 个 unsafe 参考样本；SADDER 的激活空间选择性与其概率空间全局引导形成对比。
- **CSD† (Lee & Han, 2026)**：对比安全方向直接修改 hidden state，配合 remasking 策略；SADDER 不改 hidden state 而改概率分布，使用中心化评分且无需 remasking。
- **ILRR (Avrahami & Nachmani, 2026)**：masked diffusion 对齐参考序列，非安全导向；SADDER 专为主流安全防御设计。
- **DiffuGuard (Li et al., 2025)**：stochastic remasking + block-level audit & repair，多步循环计算开销大；SADDER 为单步轻量插拔式方案。
- **FK steering (Singhal et al., 2025)**：多轨迹 resampling，全局重采样非局部修改；SADDER 与 FK/BoN 等正交可组合。
- **Attack 方法 (DIJA, PAD)**：Wen et al. (2026) 与 Zhang et al. (2025) 提出的 adversarial prompt 攻击；SADDER 作为防御方法与之对应。

## 局限性与未来方向
- **位置级安全保证的理论局限**：centered activation score 仅是 posterior unsafe odds 的代理，无严格理论保证（与 SAD 中 ηβ 代替 β* 同构），实际效果依赖近似质量。
- **Duo 模型上安全性让步**：在 Duo（USDM）上 unsafe rate 全面劣于 full SAD，选择性干预以一定安全性为代价换取质量提升。
- **DIJA 防御效果有限**：SADDER 对 DIJA 攻击的防御提升幅度较小，需探索更强对抗防御机制。
- **超参数敏感**：层 k、强度 η、预热步数 w 等超参数需针对不同模型和数据集调优，泛化性有待验证。

## 研究启发与可借鉴点
- **选择性干预范式**：仅对高风险位置施加修改的策略可迁移至其他序列生成模型的安全对齐任务，避免全局均匀 steer 的质量损失。
- **激活空间对比方向构建**：利用有害/安全响应的质心差构建方向向量，可推广至多模态扩散模型或其他生成任务的安全引导。
- **门控机制与计算效率**：仅在危险信号 > 0 时触发代价高昂的 unsafe denoiser 估算，为推理时安全干预提供了高效的稀疏化思路。
- **可与本团队方向结合**：SADDER 的 remasking 策略可与本团队的扩散模型质量控制工作结合，探索更多"生成-修正"循环机制；其激活空间 steer 方法可借鉴于多模态内容安全过滤。

## 关键术语表
- **SADDER**：Selective Activation Steering in Diffusion Decoding for Refusal，推理时安全引导方法，无需重新训练。
- **MDLMs**：Masked Diffusion Language Models，基于掩码扩散的语言模型，如 LLaDA、Dream。
- **USDMs**：Unidirectional Stochastic Diffusion Models，单向随机扩散模型，如 UDLM、Duo。
- **Safety-Aware Denoiser (SAD)**：在 token 概率空间做安全引导的基线方法，SADDER 在其基础上引入选择性门控。
- **Activation Steering**：在模型隐状态空间中构建方向向量并施加梯度偏移以引导生成的技术。
- **Remasking**：生成过程中对已确定 token 重新赋予 mask 并在后续步骤再次推理的策略。
- **NFE**：Number of Function Evaluations，扩散模型推理步数，衡量计算成本的指标。
- **XSTest**：评估模型 over-refusal 行为的基准，包含 250 safe + 200 unsafe prompts。

## 可复现要素
- **数据集**：RTP（0–299 开发，300–899 评估）、BeaverTails、JailbreakBench、AdvBench、HarmBench、WildJailbreak、XSTest — 均为公开数据集
- **代码/权重**：LLaDA-8B-Base/Instruct、Dream-7B-Base/Instruct 基线模型需从官方获取；SADDER 为推理时方法无需额外权重
- **关键超参**：层 k（如 LLaDA/RTP 用 k=31 或 24）、强度 η（如 100 或 3000）、预热步数 w（如 8）、干预窗口 [s,e]、remask 策略（none/top-b/aligned）、BoN/FK 候选数（2/4/8）
