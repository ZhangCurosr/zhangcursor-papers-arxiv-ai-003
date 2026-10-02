---
title: "V-JEPA-POLICY-BUILDING-EFFECTIVE-WORLD-ACTION-MODELS-ON-PRED"
source: https://arxiv.org/pdf/2609.37250v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 23:26:23"
field: "视觉-语言-动作模型与机器人学习"
keywords: ["world-action model", "V-JEPA", "predictive visual latent", "flow matching", "robot policy", "distribution shift"]
innovations: ["在冻结的V-JEPA预测性潜空间上从零联合训练未来预测器与flow-matching动作专家，无需继承完整视觉生成模型", "预测器-only在DROID视频-指令对上的无动作监督预训练可显著迁移至下游控制并提升分布外泛化"]
benchmarks: ["LIBERO", "LIBERO-Plus", "RoboCasa-GR1", "TianJi Marvin 双臂真实平台"]
---

# 论文速读：V-JEPA-POLICY-BUILDING-EFFECTIVE-WORLD-ACTION-MODELS-ON-PREDICTIVE-VISUAL-LATENTS

## 一句话总结
论文提出V-JEPA Policy框架，在冻结的V-JEPA 2.1预测性视觉编码器所定义的潜空间上，从零联合训练指令条件化的未来潜变量预测器和flow-matching动作专家，仅需0.9B参数（其中0.6B可训练）即可在LIBERO、LIBERO-Plus、RoboCasa-GR1及真实双臂平台上取得与大型生成式WAM和VLA基线有竞争力的性能，并验证了该潜空间可高效迁移在野视频上获得的未来建模知识。

## 研究问题与动机
- **核心问题**：有效的WAM学习是否必须继承完整的预训练视觉生成模型？还是仅凭大规模预测性预训练学到的视觉潜空间就足以支撑？
- **现有方法不足**：当前主流WAM方法（如FastWAM、ImageWAM、Cosmos Policy等）通过适配大规模预训练的**视频生成模型**或**图像编辑模型**来获取动作-视觉联合建模能力，继承了完整生成骨干及其庞大规模带来的高部署成本。
- **分布外泛化挑战**：在LIBERO-Plus等含相机视角、光照、语言、布局等多种扰动轴的基准上，判别式、重建式和视频理解导向的视觉表示均存在泛化瓶颈。
- **知识迁移空白**：如何在同一视觉潜空间上从广泛的无动作标签视频-指令数据中学习可迁移的未来建模知识，尚缺乏系统验证。

## 核心贡献（创新点）
1. **构建无生成骨干的WAM框架**：提出V-JEPA Policy，在冻结的V-JEPA 2.1编码器上从零联合训练未来潜变量预测器与flow-matching动作专家，证明无需继承完整预训练视觉生成模型即可实现有效WAM学习。
2. **预测性潜空间的系统对比实验**：在统一下游框架和训练预算下，与DINOv2/DINOv3（判别式）、WAN2.2 VAE（重建式）、InternVideo3（视频理解导向）进行匹配比较，验证V-JEPA预测性潜空间在分布偏移场景下显著领先。
3. **预测器预训练的知识迁移**：在DROID视频-指令对上进行零动作标签的预测器预训练，下游直接移植至V-JEPA Policy框架，使LIBERO-Plus成功率从79.25%提升至91.50%，超越延长下游训练（81.64%）的收益。

## 方法详解
- **冻结视觉基础**：使用V-JEPA 2.1 ViT-L冻结编码器（~0.3B参数），将每个相机视角的当前帧与前序帧组成两帧上下文管，将未来管位置视为掩码区域；目标潜变量由冻结编码器对完整视频clip的处理输出提供。
- **指令条件化未来潜变量预测**：预测器$P_\phi$（ViT架构，24层，隐藏维度1024）从零初始化，将上下文token $\mathbf{Z}_t^c$ 与可学习的未来查询token $\Delta_t^+$ 拼接，通过双向自注意力联合更新；采用3D RoPE编码时空坐标，并引入可学习视角嵌入区分不同相机；通过T5-XXL（冻结）提取指令特征，与 proprioceptive state token 拼接后以cross-attention注入每个预测器层。
- **未来信息接口**：预测器每层上下文位置的key-value对 $\mathcal{R}_{3D}(\mathbf{K}_c^{(j)}), \mathbf{V}_c^{(j)}$ 构成层级的"未来感知上下文接口" $\mathcal{C}_\phi = \{(\cdot,\cdot)\}_{j=1}^{L}$，保留3D RoPE变换后的keys以保持时空位置信息。
- **Flow-matching动作专家**：动作专家 $v_\psi$（ViT架构，24层，隐藏维度512）接收 $\mathcal{C}_\phi$、指令和状态条件，采用Mixture-of-Transformers (MoT) 与预测器共享注意力头维度；动作token的query在每个层jointly attend to 上下文key/value与自身key/value，而预测器token不能attend动作token；通过自适应RMS norm调制flow time $\tau$。
- **联合训练目标**：$\mathcal{L} = \mathcal{L}_{action} + \lambda_{future}\mathcal{L}_{future}$，其中 $\mathcal{L}_{future} = \mathbb{E}\|\hat{\mathbf{Z}}_t^+ - \mathbf{Z}_t^+\|_1$，$\mathcal{L}_{action}$ 为flow-matching损失；$\lambda_{future}=1$，动作损失回传通过 $\mathcal{C}_\phi$ 至预测器，使上下文状态同时受预测与动作监督塑形。
- **推理流程**：每次决策步单次前向传播生成 $\mathcal{C}_\phi$ 并缓存，随后以10步Euler积分从 $\tau=0$ 的Gaussian噪声采样至 $\tau=1$ 得到clean action chunk。

## 实验与结果
- **评估基准**：LIBERO（4套件共40任务）、LIBERO-Plus（7类分布偏移10030任务）、RoboCasa-GR1（24个类人机器人桌面任务）、真实TianJi Marvin双臂平台（Table Cleanup、Saucer Racking）。
- **LIBERO**：V-JEPA Policy（From Scratch）平均成功率97.3%，接近ImageWAM（98.4%）和PRTS（98.4%），但仅用0.9B参数且无动作监督预训练；Pretrained Predictor进一步提升至98.7%。
- **LIBERO-Plus**：From Scratch达79.25%，大幅超过FastWAM（51.5%）；Pretrained Predictor达91.50%，超越所有对比方法；扩展下游训练60k步仅提升至81.64%，不及预训练增益。
- **RoboCasa-GR1**：From Scratch 50.92%超越StarVLA-π（43.90%）和GR00T N1.6（47.60%）；Pretrained Predictor达55.58%。
- **真实世界**：From Scratch在Table Cleanup/Saucer Racking分别达55%/35%，与FastWAM相当；Pretrained Predictor大幅提升至75%/85%。
- **视觉基础对比**（Table 4）：在统一下游配方下，V-JEPA 2.1 ViT-L达LIBERO 97.25% / LIBERO-Plus 79.25%，相比最强判别基线DINOv2在分布偏移上的优势从2.35pp扩大至12.23pp；重建式WAN2.2 VAE表现最差（LIBERO-Plus仅46.43%）。
- **预测器规模效应**：V-JEPA 2.1 ViT-L在LIBERO-Plus上优于ViT-G（79.25% vs 78.37%），表明提升预测表征质量比单纯扩大容量更重要。
- **效率对比**（Table 6）：V-JEPA Policy峰值显存仅4.66 GiB，比FastWAM低63.4%，推理延迟178.17ms。

## 相关工作脉络
- **JEPA家族**：V-JEPA通过特征空间预测学习视觉表征，避免像素级重建；DINO-WM利用预训练特征进行零样本规划；VLA-JEPA将潜变量预测集成至预训练VL backbone；JEPA-WAM使用Qwen初始化预测器并增加对齐阶段。本文与之本质区别在于：从零联合学习预测器与动作专家，无需额外预训练或对齐阶段。
- **生成式WAM**：FastWAM、ImageWAM、Cosmos Policy等通过适配视频生成/图像编辑模型构建WAM，继承完整生成骨干。本文直接测试"仅预测性潜空间是否足够"，不依赖生成先验。
- **VLA基线**：π0/π0.5、OpenVLA-OFT、PRTS、ResVLA、StarVLA-π等大多依赖大规模动作监督预训练。本文在零动作预训练条件下达到可比性能。
- **预测器-only预训练**：DROID数据集包含海量无标签视频-指令对。本文首次系统性验证在该数据上预训练未来潜变量预测器后迁移至下游控制的有效性。

## 局限性与未来方向
- **预测器架构仍为Transformer**：当前预测器沿用V-JEPA 2的ViT设计，未探索更轻量的时序建模结构（如状态空间模型）。
- **单阶段下游训练**：虽然证明了联合训练的有效性，但未探索多阶段分步优化或 curriculum learning 的可能收益。
- **真实世界部署规模有限**：仅在两个双任务上验证，未覆盖更复杂的长程操作场景。
- **视觉编码器冻结**：Encoder完全冻结限制了端到端微调的可能性，未来可探索部分微调策略。
- **DROID预训练的时效性**：DROID数据截至2024年，未来可扩展至更新规模更大的在野视频数据集。

## 研究启发与可借鉴点
1. **层层级context key-value接口设计**：通过中间层而非最终输出的context states传递未来感知信息，兼顾信息丰富度与计算效率，可迁移至其他预测-决策耦合架构。
2. **预测表征质量 > 模型规模**：V-JEPA 2.1 ViT-L在LIBERO-Plus上优于ViT-G，提示在robotics downstream中应优先优化预训练目标（如dense supervision、context-token supervision）而非盲目扩容。
3. **无动作监督的预训练-迁移范式**：在DROID上仅用预测损失预训练，下游加一个随机初始化的action expert即可显著提升泛化，为低资源场景下的策略学习提供了高效范式。
4. **MoT双向交互架构**：预测器与动作专家通过bidirectional attention共享上下文，但预测器不能attend动作，形成单向信息流动的安全设计，可推广至其他多任务联合学习场景。
5. **分布偏移评估的重要性**：LIBERO-Plus结果（79.25%→91.50%）强烈依赖预训练，凸显了单一in-distribution指标可能掩盖的泛化缺陷，建议团队在评测中纳入类似扰动基准。

## 关键术语表
**V-JEPA (Joint-Embedding Predictive Architecture)**：一种通过在特征空间进行自监督预测来学习视觉表征的架构，避免像素级重建，强调对可预测时空结构的建模。
**V-JEPA 2.1**：V-JEPA的改进版本，引入context-token supervision和deep self-supervision以提升密集视觉表征质量。
**World-Action Model (WAM)**：将未来视觉状态预测与动作生成耦合的机器人控制框架，通过预测先验提升泛化能力。
**Flow Matching**：一种生成建模技术，通过学习数据分布到噪声分布的常微分方程（ODE）流来生成样本，比传统diffusion更高效。
**Mixture-of-Transformers (MoT)**：多个Transformer模块共享注意力机制但保持独立参数的架构设计，用于联合建模预测与动作。
**DROID**：大规模在野机器人操作视频-指令数据集，包含真实物理环境中的多相机视频和自然语言指令，无动作标签。
**LIBERO-Plus**：在LIBERO基础上引入7类分布偏移（视角、初始状态、语言、光照、纹理、噪声、布局）的鲁棒性评测基准。
**Context Key-Value Interface**：从预测器各层上下文位置的key-value对构成的层级接口，将未来感知信息传递给动作专家。

## 可复现要素
- **数据集**：LIBERO（官方开源）、LIBERO-Plus（官方开源）、RoboCasa-GR1（官方开源）、DROID（官方开源，https://droid-rs.github.io/）。
- **代码**：已开源，地址 https://github.com/breez3young/VJEPA-Policy。
- **权重**：V-JEPA 2.1编码器冻结权重来自官方；T5-XXL冻结权重来自官方；预测器和动作专家权重随代码发布（论文声明）。
- **关键超参**：
  - 视觉编码器：V-JEPA 2.1 ViT-L，冻结
  - 文本编码器：T5-XXL，冻结
  - 预测器：24层，隐藏维度1024，16头，head dim 64
  - 动作专家：24层，隐藏维度512，16头，head dim 64
  - 优化器：AdamW，lr=1e-4，warmup 5%，cosine decay
  - LIBERO：10 epochs，batch size 128，32-step action chunks，2×224×224视角
  - RoboCasa-GR1：50k steps，batch size 256，16-step action chunks，单目224×224
  - DROID预训练：100k steps，batch size 192，5Hz采样，BF16精度
