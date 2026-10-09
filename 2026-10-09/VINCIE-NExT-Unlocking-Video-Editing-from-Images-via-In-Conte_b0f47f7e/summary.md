---
title: "VINCIE-NExT-Unlocking-Video-Editing-from-Images-via-In-Conte"
source: https://arxiv.org/pdf/2610.12104v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-09 17:15:58"
field: "视频生成与编辑"
keywords: ["video editing", "in-context learning", "diffusion model", "chain-of-editing", "position encoding", "heterogeneous training"]
innovations: ["V→I→I→V 子任务分解实现图像编辑能力向视频的 in-context 迁移", "TDF3D-RoPE 双频 3D 旋转位置编码实现跨 shot 像素级空间对应", "Chain-of-Editing 两阶段推理支持无需重训的测试时 compute-quality 扩展"]
benchmarks: ["OpenVE-Bench", "VBench"]
---

# 论文速读：VINCIE-NExT: Unlocking Video Editing from Images via In-Context Modeling

## 一句话总结
VINCIE-NExT 提出了一种将视频编辑能力从图像迁移到视频的框架，通过将视频编辑分解为 V→I→I→V 的任务链，并引入上下文视觉示范（image editing pair 作为 in-context demonstration）和 TDF3D-RoPE 位置编码，使模型能够在极少配对视频编辑数据下实现时空一致的精细编辑，同时在 OpenVE-Bench 上达到开源方法中的最优性能。

## 研究问题与动机
- **视频编辑数据极度稀缺**：视频生成有大量 video-text 对可用，但视频编辑需要 (source video, instruction, edited video) 三元组，标注成本极高，难以规模化合成。
- **现有视频编辑方法各有缺陷**：基于 pipeline 的方法（光流变形、深度估计、分割等逐帧处理）存在误差累积且编辑多样性受限；逐帧应用图像编辑器则无法建模跨帧依赖，导致时间闪烁和不一致。
- **图像编辑已高度成熟**：OmniEdit、UltraEdit 等数据集提供了百万级高质量的 (source image, instruction, edited image) 对，覆盖风格迁移、对象替换、背景修改等多种操作，这些配对不仅是训练信号，更是所需外观变换的"视觉示范"。
- **核心直觉**：若生成模型能将图像编辑对作为上下文视觉证据加以利用，即可将对单个图像的外观变换泛化到视频序列，大幅降低视频编辑对大规模配对数据的依赖。

## 核心贡献（创新点）
1. **在-context 视觉示范的视频编辑范式**：将图像编辑对 $(\mathbf{I}^s, \mathbf{I}^t)$ 作为空间外观蓝图（spatial appearance blueprint）前置到交错上下文中，使每一输出帧都能参照这一像素级空间先验；与已有工作相比，本文不是将图像编辑独立应用到每帧，而是让 DiT 统一 attends over image and video tokens，实现 richer 和更 temporal-consistent 的编辑。
2. **V→I→I→V 子任务分解与联合训练**：将视频编辑因子化为三个子任务（V→I 提取关键帧、I→I 执行图像编辑、I→V 生成完整视频），每个子任务可从异构数据源独立获取监督（I2I 用 OmniEdit、I→V 用图像条件视频生成数据、V→I 用视频-关键帧对应数据），在统一 flow-matching 目标下共享参数；与 ICVE、EditVerse 的单次视频到视频生成不同，本文有显式的已编辑关键帧作为锚点。
3. **TDF3D-RoPE（Time Dual-Frequency 3D Rotary Position Encoding）**：通过 inter-shot（单调递增的全局偏移，区分不同 shot）与 intra-video（原始空间位置，零全局偏移，使对应 patch 共享相同 $(p_h, p_w)$）两个独立频率张量的元素相加，在单次旋转中同时编码"保持因果顺序"和"像素级空间对应"两个冲突需求，无需 modality-type embeddings；这是保证外观编辑从图像示范忠实传播到所有视频帧的前提。
4. **Chain-of-Editing（CoE）两阶段推理策略与测试时扩展**：Stage 1（V→I→I）生成高质量编辑关键帧，Stage 2（V→I→I→V）以该关键帧为视觉示范引导全视频生成，两阶段均为独立扩散过程，可在测试时通过扩展链长度（如多次迭代图像编辑再进入视频生成）以额外计算换取更高保真度，无需重新训练。
5. **V↔I↔I 上下文视频驱动图像编辑数据构造**：用 GPT-4o 生成编辑指令、Seedream 4.5 逐帧应用，构建 (source video, sampled frame, edited frame) 三元组，使模型在 I→I 子步骤上获得显式监督，而非孤立训练图像编辑对；该数据源贡献了 Creative Edit 类别 +74.1% 的提升，而仅增加约 6% 的视觉训练 token。

## 方法详解
**问题形式化**：给定源视频 $\mathbf{V}^s = (f_1, \ldots, f_T)$、编辑指令 $c$、编辑视频 $\mathbf{V}^t = (f_1', \ldots, f_T')$，传统方法直接建模 $p(\mathbf{V}^t | \mathbf{V}^s, c)$，需要大量配对视频数据。

**子任务分解**：引入中间图像表示（从 $\mathbf{V}^s$ 提取关键帧 $\mathbf{I}^s$，其编辑版本 $\mathbf{I}^t$），联合分布因式分解为：
$$p(\mathbf{I}^s, \mathbf{I}^t, \mathbf{V}^t | \mathbf{V}^s, c) = \underbrace{p(\mathbf{I}^s|\mathbf{V}^s, c)}_{\text{V→I}} \cdot \underbrace{p(\mathbf{I}^t|\mathbf{I}^s, \mathbf{V}^s, c)}_{\text{I→I}} \cdot \underbrace{p(\mathbf{V}^t|\mathbf{I}^t, \mathbf{I}^s, \mathbf{V}^s, c)}_{\text{I→V}}$$

**交错序列构建**：将生成链展平为单一连续 token 序列：
$$\mathbf{x} = [c_1, \mathcal{E}(\mathbf{V}^s), c_2, \mathcal{E}(\mathbf{I}^s), c_3, \mathcal{E}(\mathbf{I}^t), c_4, \mathcal{E}(\mathbf{V}^t)]$$
其中每个视觉 segment 配有独立文本 prompt（媒体参考标签、帧采样 prompt、编辑指令、图生视频 prompt），图像编辑对 $(\mathcal{E}(\mathbf{I}^s), \mathcal{E}(\mathbf{I}^t))$ 形成 in-context 视觉示范。

**联合训练损失**：所有子任务使用统一 flow-matching 目标：
$$\mathcal{L} = \mathbb{E}\left[\|(\epsilon - y) - \epsilon_\theta(y_\tau, x, \tau)\|^2\right]$$
其中 $y_\tau = (1-\tau)y + \tau\epsilon$，$\epsilon_\theta$ 预测速度 $\epsilon - y$，训练数据 $\mathcal{D} = \mathcal{D}_{V\to I} \cup \mathcal{D}_{I\to I} \cup \mathcal{D}_{I\to V} \cup \mathcal{D}_{\text{full}}$，子任务随机采样。

**TDF3D-RoPE 核心设计**：为每个 token 分配位置三元组 $(p_t, p_h, p_w)$，inter-shot 分量赋单调递增全局偏移 $\tau_k$，intra-video 分量赋原始空间位置 $(t_i, h, w)$ 且零全局偏移，两者元素相加：
$$\mathbf{p}_{\text{inter}} = (\tau_k + t_i, \tau_k + h, \tau_k + w), \quad \mathbf{p}_{\text{intra}} = (t_i, h, w)$$
使同一空间位置在不同 shot 间共享相同 $(p_h, p_w)$，实现纯内容驱动的 attention 和像素忠实传播。

**时间位置随机化（TPR）**：为避免训练图像固定 $t_i=0$ 造成的时空位置伪相关，将图像的时间位置替换为 $\tilde{t} \sim \mathcal{U}[0, T_{\max}]$，强制模型基于空间内容而非时间索引建立对应关系。

**CoE 两阶段推理**：Stage 1（V→I→I）生成编辑关键帧 $\hat{z}^t$；Stage 2（V→I→I→V）将 $\hat{z}^t$ 直接以 latent space 插入（无 VAE decode/re-encode 循环），使用 4-shot 上下文处理，conditioning shots 接收 $\tau=0$（完全去噪），仅目标 shot 接收活跃 diffusion timestep。

**V↔I↔I 数据构造**：对每个原始视频，GPT-4o 生成跨 13 类（remove, replace, attribute, background, global, effect, camera, action, expression, dynamics, position, pose, orientation）的平衡编辑指令，Seedream 4.5 逐帧应用，形成附带源视频的 (源视频, 采样帧, 编辑帧) 链三元组。

## 实验与结果
**数据集**：OpenVE-Bench（431 个视频片段，8 类编辑：Global Style、Background Change、Local Change、Local Remove、Local Add、Subtitle Edit、Creative Edit、Camera Edit），由 Gemini 2.5 Pro 在 1-5 分制下从 Instruction Compliance、Consistency & Detail Fidelity、Visual Quality & Stability 三维评分。

**训练数据**：I2I（OmniEdit，1.20M 对）、V2V*（OpenVE，2.45M 对，重组为子链格式）、V↔I↔I（1.63M 三元组）。

**基线**：VACE、OmniVideo、InsViE、Lucy-Edit、ICVE、DITTO、OpenVE-Edit（开源）及 Runway Aleph（商业闭源）。

**主要结果**（Tab. 1，Gemini 2.5 Pro 评分，分辨率 640×480）：
- **Overall：3.08**，较 OpenVE-Edit（2.49）提升 **24% 相对增益**，缩小与 Runway Aleph（3.65）的差距。
- **Global Style：4.17**（超过 Runway Aleph 的 3.72），为最大亮点，in-context 编辑范式直接编码完整外观变换并通过 TDF3D-RoPE 时间传播。
- **Local Remove：3.24**（较 OpenVE-Edit 的 1.85 提升 75%），通过成熟图像编辑器的 coherent inpainting 实现。
- **Creative Edit：3.41**，几乎无 V2V 监督下取得显著成绩。
- **Camera Edit：2.06**，为最弱类别，需几何变换超出外观编辑范畴。

**Abation 关键发现**：
- 仅 I2I 训练无法泛化到视频；仅 V↔I↔I 即可实现有意义视频编辑。
- V2V*+CoE（2.57）已超过 OpenVE-Edit（2.49），Full data+CoE（2.65）再提升 +14.2%。
- Creative Edit 在 Full data+CoE 下达 2.89，较 Pure V2V（1.66）提升 **+74.1%**。
- TDF3D-RoPE 和 TPR 交互为乘法关系（非加性）：无 TPR 时固定时间索引造成伪相关，CoE 无法发挥；二者结合方达峰值。
- 外部编辑器（Qwen-Image-Edit、Nano Banana）始终优于 self-editing，凸显模块化 I→I 接口的可迁移优势。
- VBench 客观指标：imaging_quality 0.715（较最强基线 ICVE 的 0.688 提升 +0.027），quality_avg 0.819。
- Test-time scaling：CoE 中间图像编辑阶段 denoising steps 从 1 增至 64，overall 从 2.30 提升至 2.81（+22.2%），无需重训。

## 相关工作脉络
1. **Diffusion-based Video Editing（VACE、OmniVideo、InsViE、Lucy-Edit、DITTO、OpenVE-Edit）**：直接训练视频到视频编辑，需要大规模配对视频数据；本文通过子任务分解借用图像编辑成熟数据，大幅降低对 V2V 配对的依赖。
2. **In-context Image Editing（In-context LoRA、Generative Multimodal Models as In-Context Learners）**：证明 diverse 高质量图像编辑数据集可训练通用编辑器；本文明确将此资源用于监督视频编辑，是其核心灵感来源。
3. **Zero-shot Video Editing via Image Editors（Pix2Video、Zero-shot Video Editing using Off-the-shelf Image Diffusion Models、AnyV2V）**：逐帧应用图像编辑器或从单帧传播；存在时间闪烁问题，且无法泛化需全局时序推理的编辑；本文通过 interleaved in-context modeling 建模跨帧依赖。
4. **ICVE、EditVerse**：探索 in-context video editing，但均进行单次视频到视频生成，无显式已编辑关键帧；本文路线根本不同——jointly attends over image and video tokens 的交错上下文建模。
5. **Architecture Inflation / Frame-by-frame Approaches（ControlVideo、SparseCtrl、Align your Latents）**：在图像 U-Net 中插入时间层或逐帧应用图像级控制信号；转移了视觉质量但未转移语义编辑行为；本文不在架构层面膨胀，而在训练数据和方法论层面迁移。
6. **Training-based General Editors（Dreamix、InstructVid2Vid、VIVID-10M）**：直接在配对视频数据上训练通用编辑器；本文证明通过 V↔I↔I 数据和 CoE 推理，可用极少 V2V 数据达到更强性能。

## 局限性与未来方向
- **中间 I→I 步骤的质量瓶颈**：编辑关键帧的错误（幻觉细节、不完整编辑）会传播到所有输出帧；self-editing 在需几何变换或新内容合成的编辑上表现不佳，外部编辑器虽可改善但仍有局限。
- **单关键帧对高时序变化视频的表征不足**：对于有显著时序变化或大相机运动的视频，单个关键帧可能无法充分表征。
- **CoE 两阶段串行推理的计算开销**：虽然 53.5s（H200）快于 ICVE（118.1s）和 AnyV2V（11min 46s），但更长测试时链会增加成本。
- **评估范围有限**：仅在 OpenVE-Bench 上验证，在长视频或高动态视频上的泛化尚未验证。
- **相机运动假设**：框架假设静态或缓变相机，快速剪辑或严重遮挡视频可能挑战关键帧锚定范式。
- **未来方向**：更统一的模型同时覆盖图像/视频/音频/3D，耦合理解与生成；将 CoE 扩展为 agentic 框架，按需调用外部工具（分割、图像编辑器、质量验证器）并结合 RAG  grounding 稀有概念。

## 研究启发与可借鉴点
1. **异构数据联合训练策略**：将丰饶的图像编辑数据与稀缺的视频编辑数据通过子任务分解统一在单一扩散目标下训练，实现了"数据效率"与"能力迁移"的双重收益，此思路可迁移到其他模态间的能力迁移场景（如音频编辑、3D 编辑）。
2. **双频位置编码设计**：TDF3D-RoPE 通过 inter/intra 两个独立频率张量的相加解决"区分 shot"与"对应 patch"的冲突需求，这一技巧可推广到任何需要跨视口/跨样本空间对齐的多 shot 上下文建模任务。
3. **模块化 I→I 接口**：CoE 的 I→I 阶段作为可插拔模块，外部图像编辑器改进可直接传递到视频编辑而无需重训，为持续利用图像编辑前沿成果提供了工程范式。
4. **测试时扩展（test-time scaling）作为免费性能提升**：CoE 允许在测试时通过增加中间步骤数（denoising steps）单调提升质量，这种"compute-for-quality" trade-off 无需额外训练，可作为模型部署时的弹性策略。
5. **V↔I↔I 数据构造管线**：用 GPT-4o 生成指令+state-of-the-art 图像编辑器生成配对，是一种低成本、可扩展的合成数据构造方案，可复用于其他需要大规模指令跟随数据的视觉生成任务。

## 关键术语表
- **In-context Visual Demonstration**：将图像编辑对 $(\mathbf{I}^s, \mathbf{I}^t)$ 作为上下文视觉示范嵌入交错序列，使模型在生成视频帧时能参照已知的源→目标外观变换作为像素级空间蓝图。
- **TDF3D-RoPE（Time Dual-Frequency 3D Rotary Position Encoding）**：一种双频 3D 旋转位置编码，通过 inter-shot（全局单调偏移）和 intra-video（原始空间位置，零偏移）两个频率张量的相加，同时满足跨 shot 可区分性和同空间位置像素级对应性。
- **Chain-of-Editing（CoE）**：两阶段推理策略，Stage 1 生成编辑关键帧（V→I→I），Stage 2 以该关键帧为示范生成完整视频（V→I→I→V），支持测试时扩展。
- **V↔I↔I Data**：上下文视频驱动的图像编辑训练数据，由 GPT-4o 生成指令、Seedream 4.5 逐帧编辑构建，将视频编辑数据转化为图像编辑示范。
- **Temporal Position Randomization（TPR）**：将训练图像的时间位置从固定 $t_i=0$ 替换为 $\tilde{t} \sim \mathcal{U}[0, T_{\max}]$，消除时空位置的伪相关，是 CoE 发挥效能的前提。
- **Flow-matching Objective**：统一训练目标，预测噪声速度 $\epsilon - y$，适用于所有子任务的联合训练。
- **OpenVE-Bench**：包含 431 个视频片段的指令跟随视频编辑基准，涵盖 8 类编辑操作，由 MLLM（Gemini 2.5 Pro）在 1-5 分制下多维评分。

## 可复现要素
- **数据集**：OpenVE-Bench（431 clips，8 categories）；训练数据 I2I（OmniEdit，1.20M）、V2V*（OpenVE 子集，2.45M）、V↔I↔I（1.63M）；代码仓库中提供训练和推理 launch scripts 及完整超参配置（论文声明开源，但具体 URL 在 Project Page https://vincie-next.github.io/）。
- **代码/权重**：论文声明提供 code repository，项目页面 https://vincie-next.github.io/；DiT backbone（3B MM-DiT）初始化自 VINCIE 同一 T2V pretrained backbone；VAE 从 SD3 inflated（×8 spatial，×4 temporal，16 channels，patch size $(1,2,2)$）；text encoder 为 Flan-T5。
- **关键超参**：32×H100 GPU，≈150 小时，42k steps，lr=$5\times10^{-5}$；Stage 1: 256×256，≈80h；Stage 2: 480×640，≈70h，从 Stage 1 初始化；inference: 最多 65 frames@12fps，32 sampling steps，CFG scale=2.5；source keyframe 取 temporal mid-frame。
