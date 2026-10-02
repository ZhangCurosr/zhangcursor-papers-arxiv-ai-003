---
title: "VERIFIABLE-VISUAL-REWARDS-TRANSFER-FROM-SYNTHETIC-SCENES-TO"
source: https://arxiv.org/pdf/2609.35641v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 08:14:35"
field: "文本到图像生成与指令遵循"
keywords: ["text-to-image generation", "verifiable reward", "reinforcement learning", "instruction following", "synthetic benchmark", "programmatic verifier"]
innovations: ["首个程序化可验证图像奖励框架（VVR），无需 learned detector/VLM", "VVRBENCH 与 Challenge 数据集揭示 frontier 模型指令遵循能力缺口", "RLVVR 证明合成场景训练可泛化至自然提示并与现有奖励互补"]
benchmarks: ["VVRBENCH", "VVRBENCH-Challenge", "GenEval", "GenEval2", "OCR", "PickScore", "HPSv3"]
---

# 论文速读：VERIFIABLE-VISUAL-REWARDS-TRANSFER-FROM-SYNTHETIC-SCENES-TO

## 一句话总结
本文提出 **Verifiable Visual Rewards（VVR）**——首个面向图像生成、基于程序化可验证规则的奖励框架，能够生成任意数量与难度的合成场景任务，并将 VVR 得分作为强化学习奖励（RLVVR）训练图像生成器；实验表明，在简单合成场景上训练可显著提升模型对复杂指令及自然提示的遵循能力，并与现有后训练目标兼容互补。

## 研究问题与动机
1. **文本生成已有可验证奖励，但图像生成没有**：语言模型已能用程序化规则（如"至少出现词 X 三次"）作为确切奖励进行 RL 训练，但文本到图像领域仍依赖不可靠的 learned reward models（VLM、object detector、preference model），这些评估器本身会出错且易被策略 exploit。
2. **图像生成指令遵循精度不足**：现代 text-to-image 模型在涉及计数、空间关系、属性绑定等精确约束时错误率高，且随着提示中约束数量增加，错误率显著上升。
3. **已有评估器存在系统偏差且难以诊断**：learned evaluator 的误差难以分解到具体约束类型；模型可能通过 exploit evaluator 的已知缺陷来获得高分，而非真正学会指令遵循。
4. **缺乏可控难度、可无限扩展的训练/评测数据**：现有合成 benchmark（如 ConceptMix）依赖 VLM/detector 打分，无法像语言模型训练那样无限扩展数据规模与复杂度梯度。

## 核心贡献（创新点）
1. **VVR 框架（首个程序化可验证图像奖励框架）**：通过 Python 函数对生成像素直接做确定性判定，无需任何 learned detector/VLM/OCR/embedding model；与已有工作的本质区别在于奖励信号完全由程序 verifier 决定，消除了 learned evaluator 的误差传播和 reward hacking 风险。
2. **VVRBENCH 与 VVRBENCH-Challenge 数据集**：共 10,720 个可编程可验证任务，覆盖 32/46 种约束类型、5 个复杂度层级；最强模型 GPT-Image-2.5-Sunburst 仅解出 21.4% Challenge 题，揭示了指令遵循能力的真实缺口。
3. **RLVVR 训练范式**：将 VVR dense reward 用于 Flow-GRPO 强化学习训练扩散模型；证明（a）简单分布训练可泛化到更难任务，（b）混合 VVR 与现有奖励目标可进一步提升通用基准与人类偏好评分，推动 VVR 进入标准 post-training recipe。

## 方法详解
**任务表示**：一个 VVR 任务 $s = (\mathcal{G}, \mathcal{B}, \mathcal{A}, \mathcal{F}, p)$，其中 $\mathcal{G}$ 为对象组集合，$\mathcal{B}$ 为背景约束（固定纯色背景），$\mathcal{A}$ 为作用在对象组上的 active constraints（46 种约束类型，分 5 个 family：Grounding、Cardinality、Spatial、Size、Topology），$\mathcal{F}$ 为 forbidden-content（no unrequested objects），$p$ 为由模板生成的自然语言 prompt。

**任务生成流程**：① 采样背景和对象组（颜色、形状、位置、尺寸）→ ② 枚举所有满足 arity 的约束实例 → ③ 用 verifier 检查哪些实例在当前场景下为真，得到 satisfiable set $\mathcal{A}^*$ → ④ 从中采样子集 $\mathcal{A} \subseteq \mathcal{A}^*$ 作为 active constraints → ⑤ 用模板渲染为自然语言 prompt。整个过程保证 well-formed、jointly satisfiable、faithfully expressed。

**结构复杂度**：$C(s) = \sum_{a \in \mathcal{A}} c(a;s)$，其中每个约束类型的复杂度贡献遵循固定规则（如 $L(n) = 1 + \log_2 n$），随约束涉及的对象数量单调递增，可作为模型无关的难度度量。

**确定性 verifier（无参考图）**：
- **对象提取**：HSV 空间固定色调阈值生成 8 种颜色的二值 mask → 形态学清理（腐蚀+膨胀）→ 连通分量分析 → 基于 aspect ratio/bounding-box coverage/convexity 的几何分类器判定圆/方/三角。
- **约束 verifier**：每个约束类型对应一个 Python 函数，返回二元判定 $d_a(x,s) \in \{0,1\}$ 和 partial-credit 分数 $q_a(x,s) \in [0,1]$。
- **Exact score**：$r_{\text{exact}}(x,s) = \prod_{a \in \mathcal{B}\cup\mathcal{A}\cup\mathcal{F}} d_a(x,s)$（所有约束全通过才算正确）。
- **Dense reward**：$r_{\text{dense}}(x,s) = \psi(x,s) \sum_{a} w_a q_a(x,s)$，其中 $\psi$ 为乘性惩罚因子，防止模型只学会单一容易约束而忽略其他。

**RLVVR 训练**：在 SD3.5 Medium 上使用 Flow-GRPO，LoRA rank 32、$\alpha=64$，3000 步优化，rollout 24 次/提示，每步 768 张生成图。使用 $r_{\text{dense}}$ 作为奖励信号。

## 实验与结果
**基准评测**：
- **VVRBENCH**（10,000 任务，复杂度 3–48）：最强开源模型 FLUX.2-dev 仅 19.15%；API 模型 GPT-Image-2 达 86.86%，但在高复杂度 $C_5$ 下降至 65.72%。
- **VVRBENCH-Challenge**（720 任务，复杂度 45–80）：GPT-Image-2.5-Sunburst 最优，21.39%（复杂度 69–80 仅 7.92%），所有其他模型 ≤ 8%。
- **失败集中**：Cardinality（same_count 31%，times_as_many 33%）和 Topology（each_contains 34%）约束族错误率最高；Grounding 达 95% 通过率。

**RLVVR 训练结果（SD3.5 Medium）**：
- VVR-Easy 训练：VVRBENCH 准确率从 2.81% → **28.27%**（提升约 10×），且 $C_3$–$C_5$ 更难范围仍有显著增益（+17.16、+8.31、+1.35 百分点）。
- VVR-Matched 训练：VVRBENCH 准确率升至 **46.60%**，$C_3$–$C_5$ 分别达 45.62%/38.39%/21.82%。
- **跨域迁移**：仅用彩色几何形状训练，VVR-Easy 在 GenEval 上 +0.113、OCR 上 +0.111，8/10 非 VVR 指标改善；人类偏好 win rate 在 160 条自然 prompt 上达 **71.6%**（vs pretrained）。
- **奖励混合**：VVR-Easy + GenEval2 混合后，GenEval2 自身 +0.025，GenEval +0.030，OCR +0.030，HPSv3 +0.222，ImageReward +0.045；人类偏好 win rate 58.6%（vs GenEval2 alone）。

## 相关工作脉络
1. **Verifiable rewards in LMs**：Zhou et al. (2023)、Lambert et al. (2025)、Pyatkin et al. (2025) 在语言模型上用程序规则做可验证奖励；本文首次将此范式扩展到图像生成，且图像约束存在组合兼容冲突（如 A⊂B, B⊂C, C⊂A 两两可满足但整体不可满足），需要新的生成器保证可行性。
2. **Text-to-image rewards / post-training**：Black et al. (2024)、Fan et al. (2023)、Xu et al. (2023) 用 learned preference models 或 VLMs 作奖励；本文指出 learned evaluator 会出错且可被 exploit（Zhang et al. 2024; Hong et al. 2026），VVR 完全避免此问题。
3. **GenEval / GenEval2**：Ghosh et al. (2023) 用 object detector 打分，Kamath et al. (2025) 更新为 VLM judge；两者均为 learned 信号，VVR 提供 programmatic ground truth 作为对比和补充。
4. **CLEVR 及其后续**：Johnson et al. (2027) 从生成场景推导视觉 QA；本文从场景同时推导 prompt 和 verifier，目标是从合成到自然提示的泛化。
5. **GeoSVG-RL / TikZ 生成**：Li et al. (2026)、Rodriguez et al. (2025) 对 SVG/TikZ 程序做几何验证；VVR 直接在像素空间验证，不依赖中间程序表示。
6. **Flow-GRPO**：Liu et al. (2025a) 提出 flow matching 模型的在线 RL 算法；本文采用该算法作为 RLVVR 的优化器基座。

## 局限性与未来方向
1. **视觉域受限**：当前仅支持 8 种颜色、3 种形状、纯色背景、2D 几何对象；未来可扩展更多形状、纹理、真实物体类别及 3D 渲染。
2. **任务类型单一**：仅限 text-to-image generation，未覆盖 image editing；视频生成中的 motion/velocity/acceleration 约束尚未实现。
3. **训练方法单一**：仅在 SD3.5 Medium + Flow-GRPO 上验证；其他架构（如 SD3 Large、FLUX）和其他 RL 算法（如 PPO）的效果未知。
4. **缺少 mid-training 阶段**：当前为 RL-Zero 配方，直接在 pretrained 模型上做 RL；SFT/DPO 预训练 + RL 的两阶段策略可能进一步提升效果。
5. **部分 API 模型存在不当弃权**：Gemini 系列模型偶尔将可解任务误判为"contradictory"而拒绝生成，说明 frontier 模型的空间推理仍有缺陷。
6. **reward design 可扩展空间大**：当前使用固定权重和惩罚因子，未来可按 constraint family 动态加权或实现 adaptive curriculum。

## 研究启发与可借鉴点
1. **程序化 verifier 替代 learned reward 的可行性**：VVR 证明在图像生成领域，通过精心设计的几何/像素级规则可实现与语言模型同等精度的可验证奖励，为后续研究提供了"免 VLM/detector 依赖"的 RL 训练范式。
2. **结构复杂度作为难度控制手柄**：$C(s)$ 公式与复杂度分箱在多个模型上均呈现单调下降的准确率曲线，为合成数据生成提供了可量化、可调控的难度分布设计思路，可直接迁移至其他视觉推理任务。
3. **"Easy 训练→Hard 泛化"的 curriculum 洞察**：VVR-Easy（≤1 个 constraint family，复杂度 ≤20）训练即能显著提升更难任务的 partial score（gap closed 79%/64%），说明基础约束的可靠性训练比直接做组合训练更有效地建立泛化基础。
4. **混合奖励的互补性验证**：VVR 与 GenEval2/OCR/PickScore 等混合后各项指标普遍提升，证明 programmatic reward 与 learned reward 在不同维度上正交，为构建多源 reward mixture 提供了实证依据。
5. **failure 可诊断性**：VVR 的 per-constraint verifier 可以精确定位模型在哪类约束上失败（如 Cardinality 31% vs Grounding 95%），这种细粒度诊断能力远超现有 benchmark 的 aggregate score。

## 关键术语表
**VVR (Verifiable Visual Rewards)**：面向图像生成的程序化可验证奖励框架，通过 Python 函数对生成像素做确定性评分，无需 learned detector/VLM。
**VVRBENCH**：10,000 个可编程验证的图像生成评测任务，覆盖 32 种约束类型、复杂度 3–48。
**VVRBENCH-Challenge**：720 个高难度评测任务（复杂度 45–80），用于区分前沿模型能力。
**RLVVR**：将 VVR dense reward 作为强化学习信号训练图像生成器的方法，本文使用 Flow-GRPO 在 SD3.5 Medium 上实现。
**Dense reward ($r_{\text{dense}}$)**：VVR 的训练奖励函数，结合 partial-credit 分数与惩罚因子 $\psi$，避免模型 exploit 单一易学约束。
**Structural complexity $C(s)$**：任务难度的程序化度量，等于所有 active constraint 的复杂度贡献之和，用于控制生成数据的难度分布。
**Flow-GRPO**：针对 flow matching 模型的在线强化学习算法（Liu et al. 2025a），本文将其作为 RLVVR 的优化器。
**Constraint families**：VVR 46 种约束分为 5 个 family：Grounding（属性绑定）、Cardinality（数量关系）、Spatial（位置排列）、Size（大小比较）、Topology（接触与包含）。

## 可复现要素
- **数据集**：VVRBENCH（10,000）、VVRBENCH-Fast（820）、VVRBENCH-Challenge（720）、VVR-Easy（100K）、VVR-Matched（100K）；均在 https://huggingface.co/datasets/stellalisy/VVRBench 公开。
- **代码**：verifier、task generator、训练脚本、评估脚本均在 https://github.com/stellalisy/VVRBench 开源。
- **模型权重**：论文声明 release model checkpoints；具体链接见附录与 GitHub。
- **关键超参**：LoRA rank 32、$\alpha=64$；Flow-GRPO 1 inner epoch，advantage clip $[-5,5]$，policy ratio clip $10^{-4}$，KL coef 0.04；AdamW lr $3\times10^{-4}$，3000 steps，batch 32 prompt groups × 24 rollouts = 768 images/update；生成分辨率 512×512，25 denoising steps，CFG 4.5。
