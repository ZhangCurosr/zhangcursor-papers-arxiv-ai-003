---
title: "V-JEPA-POLICY-BUILDING-EFFECTIVE-WORLD-ACTION-MODELS-ON-PRED"
source: https://arxiv.org/pdf/2609.37250v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 23:26:30"
field: "视觉-语言-动作（VLA）模型与世界模型"
keywords: ["世界动作模型", "预测性视觉表征", "V-JEPA", "视觉-语言-动作策略", "机器人模仿学习", "分布偏移泛化"]
innovations: ["在冻结 V-JEPA 预测性视觉潜在空间上从零联合训练预测器与流匹配动作专家，无需继承预训练视觉生成模型", "通过层级 future-informed context interface 将预测监督与动作监督联合优化，实现未来建模直接塑造动作生成", "在 DROID 视频-指令对上进行无动作标签的 predictor-only pretraining，显著迁移未来建模知识到下游控制任务"]
benchmarks: ["LIBERO", "LIBERO-Plus", "RoboCasa-GR1", "TianJi Marvin 真实双臂平台"]
---

# 论文速读：V-JEPA POLICY: BUILDING EFFECTIVE WORLD-ACTION MODELS ON PREDICTIVE VISUAL LATENTS

## 一句话总结
论文提出 V-JEPA Policy，一种构建在冻结的 V-JEPA 2.1 编码器预测性视觉潜在空间上的世界动作模型（WAM），通过单阶段联合训练指令条件化未来潜在预测器和流匹配动作专家，以仅 0.9B 参数实现了与大型生成式 WAM/VLA 基线相竞争的仿真与现实双臂控制性能。

## 研究问题与动机
1. **核心问题**：有效的 WAM 学习是否必须继承完整的预训练视觉生成模型（如视频生成器或图像编辑模型）？还是可以仅依托大规模预测性预训练所学到的预测性视觉潜在空间？
2. **现有方法不足**：主流 WAM 通过适配大规模预训练的视频生成器或图像编辑模型来利用预测先验，这类方法不可避免地继承了用于生成而非控制的知识结构，导致参数量大、推理开销高。
3. **潜在优势未验证**：V-JEPA 等 JEPA 架构已证明可学到丰富的时空动力学表征，但其作为 WAM 基础的有效性、泛化能力以及与判别式/重建式表征的对比尚无系统评估。
4. **知识迁移潜力不明**：预测性潜在空间是否能在无动作标签的视频-指令数据上习得可迁移的未来建模知识，并从中受益，是一个尚未回答的问题。

## 核心贡献（创新点）
1. **提出 V-JEPA Policy 框架**：在冻结的 V-JEPA 2.1 编码器上，从任务特定演示数据从头联合训练指令条件化未来潜在预测器与流匹配动作专家，无需继承任何完整预训练视觉生成模型即可学习有效 WAM。
2. **设计 future-informed context interface**：通过预测器层级的上下文 key–value 状态连接未来预测与动作生成，上下文与未来查询经双向注意力交互后传递时空位置编码，使未来建模直接塑造动作决策。
3. **系统验证预测性视觉潜在空间的基础有效性**：在共享下游框架与训练预算下，控制比较判别式（DINOv2/v3）、重建式（Wan VAE）、视频理解（InternVideo3）和预测式（V-JEPA）四种视觉基础，证明预测性潜在空间在分布偏移下具有最强鲁棒性。
4. **揭示 predictor-only pretraining 的知识迁移价值**：仅在 DROID 视频-指令对上预训练预测器（无动作标签），再迁移到下游 WAM 训练中，可在相同下游预算下将 LIBERO-Plus 成功率从 79.25% 提升至 91.50%，远超延长下游训练的计算收益。

## 方法详解
### 视觉固定潜在空间
- 冻结 V-JEPA 2.1 ViT-L 编码器，将当前帧与前一帧配对形成最小两帧观测上下文（tubelet size=2），将未来管位置视为 masked region。
- 对每个相机视角独立编码，拼接得到上下文表征 $\mathbf{Z}_t^c$ 和未来目标表征 $\mathbf{Z}_t^+$（经特征维度归一化）。

### 指令条件化未来潜在预测
- 预测器 $P_\phi$ 基于 V-JEPA 2 架构，接收投影后的上下文 token 与可学习未来查询 token $\Delta_t^+$（在目标位置重复），通过双向自注意力更新两者。
- 使用 3D RoPE 区分时空位置（时间/高度/宽度），叠加可学习视角嵌入区分相机。
- 指令 $\ell$ 经冻结 T5-XXL 编码后与本体状态 $\mathbf{q}_t$ 投影拼接为条件序列，在每个预测器层中通过 cross-attention 注入。
- 输出：$\hat{\mathbf{Z}}_t^+ = P_\phi(\mathbf{Z}_t^c, \Delta_t^+ | \ell, \mathbf{q}_t)$。

### 预测与动作耦合机制
- **future-informed context interface**：取预测器每层 $j$ 在上下文位置的 key $\mathbf{K}_c^{(j)}$ 和 value $\mathbf{V}_c^{(j)}$，对 key 保留 3D RoPE 变换，构成接口 $\mathcal{C}_\phi = \{(\mathcal{R}_{3D}(\mathbf{K}_c^{(j)}), \mathbf{V}_c^{(j)})\}_{j=1}^L$。
- **Flow-matching action expert**：在 MoT 架构中，动作 token 的 query 同时 attend 上下文 key/value 与自身 token，输出条件速度 $v_\psi(\mathbf{a}_t^\tau, \tau; \mathcal{C}_\phi, \ell, \mathbf{q}_t)$，目标为 $\mathbf{a}_t - \epsilon$。
- 预测器 token 不可 attend 动作 token（单向约束）。

### 联合训练与推理
- 损失函数：$\mathcal{L} = \mathcal{L}_{action} + \lambda_{future}\mathcal{L}_{future}$，其中 $\lambda_{future}=1$。
- $\mathcal{L}_{future} = \mathbb{E}[\|\hat{\mathbf{Z}}_t^+ - \mathbf{Z}_t^+\|_1]$，$\mathcal{L}_{action}$ 为标准 flow-matching MSE。
- 动作损失反向传播通过 $\mathcal{C}_\phi$ 影响预测器，实现联合优化。
- 推理时采用 imagine-then-act：单次前向构造 $\mathcal{C}_\phi$ 后缓存，用 10 步 Euler 积分从 $\tau=0$ 采样到 $\tau=1$ 得到动作 chunk。

## 实验与结果
### 基准与设置
- **LIBERO**（4 suite，1693 演示）：in-distribution 多任务模仿，10 epoch，batch=128。
- **LIBERO-Plus**：7 轴分布偏移（视角/初态/语言/光照/纹理/噪声/布局），10,030 扰动任务，无额外微调直接评估。
- **RoboCasa-GR1**：24 类人桌台任务，50k 步，batch=256。
- **真实双臂**：TianJi Marvin 平台，Table Cleanup 和 Saucer Racking，各 20 次 trial。

### 主要结果
| 模型 | 参数(B) | 有无动作预训练 | LIBERO | LIBERO-Plus | RoboCasa-GR1 |
|---|---|---|---|---|---|
| V-JEPA Policy (From Scratch) | 0.9 | ✗ | 97.3 | 79.3 | 50.9 |
| V-JEPA Policy (Pretrained Predictor) | 0.9 | ✗ | 98.7 | 91.5 | 55.6 |
| ImageWAM | 4.5 | ✗ | 98.4 | 80.5 | — |
| PRTS | 5.0 | ✓ | 98.4 | 83.1 | — |
| FastWAM | 6.0 | ✗ | 97.6 | 51.5 | — |
| π0.5 | 3.3 | ✓ | 96.9 | 85.9 | — |

- **最强结果**：Pretrained Predictor 在 LIBERO-Plus 达到 91.5%（+12.25 pp vs. From Scratch），真实任务 Table Cleanup 75%/Saucer Racking 85%（vs. 55%/35%）。
- **效率优势**：0.9B 参数 vs. ImageWAM 4.5B、PRTS 5B；峰值显存仅 4.66 GiB，比 FastWAM 低 63.4%。

### 消融关键结论
1. **视觉基础对比**：V-JEPA 2.1 ViT-L 在 LIBERO 达 97.25%、LIBERO-Plus 达 79.25%，vs. DINOv2 的 94.9%/67.02%；分布偏移下差距扩大至 12.23 pp。
2. **编码器规模**：ViT-L → ViT-G 主要提升 in-distribution 性能，对 LIBERO-Plus 增益有限甚至负向；预测表征质量改进比单纯放大容量更可靠。
3. **未来预测作用**：移除 future loss（保留 future-query）使 LIBERO 降至 91.65%、LIBERO-Plus 降至 68.81%，证明显式预测监督不可或缺。
4. **迁移收益 vs. 延长训练**：DROID 预训练（10 步下游）达到 91.50%，而 From Scratch 延长至 60k 步仅 81.64%，收益无法通过更多任务数据回收。

## 相关工作脉络
1. **JEPA 系列（LeCun et al., 2022; Bardes et al., 2024; Assran et al., 2025; Mur-Labadia et al., 2026）**：JEPA 通过在特征空间预测 masked 区域学习视觉表征，避免像素级重建；V-JEPA 2/2.1 进一步支持稠密特征输出。本文定位：首次将 JEPA 预训练潜在空间系统性地用于 WAM 学习，并验证其基础有效性。
2. **DINO-WM（Zhou et al., 2024）与 Action-Conditioned V-JEPA 2（Assran et al., 2025）**：使用预训练判别/预测特征进行机器人规划；本文区别于二者在于：不依赖动作条件化的 latent dynamics，而是联合训练预测器+动作专家的端到端 WAM。
3. **Video-Generation-based WAM（Pai et al., 2025; Kim et al., 2026; Ye et al., 2026; Yuan et al., 2026 / FastWAM）**：适配视频生成模型（Cosmos、Wan 等）进行动作生成；本文与它们的本质差异在于：不继承生成模型，仅利用其底层预测性表征，参数量大幅缩减。
4. **Image-Editing-based WAM（Zhang et al., 2026c / ImageWAM）**：利用图像编辑模型实现当前→目标状态的视觉映射；本文与之不同：预测多步未来潜在而非单步图像编辑，且无需预训练编辑模型。
5. **VLA-JEPA（Sun et al., 2026）与 JEPA-WAM（Lin et al., 2026）**：前者整合预训练 VLM 与 JEPA 预测；后者用 Qwen 初始化预测器并增加 VL 对齐阶段；本文定位为：完全从零学习，仅依赖冻结编码器，无额外 VL 对齐或动作预训练。
6. **Foundation Policies（Black et al., 2024/2025 / π0、π0.5; Zhang et al., 2026b / PRTS）**：大规模 VLA 策略通常需动作监督的 embodied pretraining；本文定位为：无需此类预训练，仅用下游演示即可达到竞争性性能，证明预测性表征本身的迁移价值。

## 局限性与未来方向
1. **动作 horizon 受限**：当前使用 16~32 步动作 chunk，尚未探索更长 horizon 下的预测一致性维护。
2. **仅评估单阶段 downstream**：未研究多阶段或分层预测（如粗到细的时间尺度）对复杂任务的增益。
3. **DROID 预训练规模有限**：仅使用 100k 步 DROID 视频-指令对，更大规模无动作标注数据的预训练效果尚未探索。
4. **真实环境未见长期零样本迁移**：虽展示双task 真实部署，但未测试跨环境/跨物体的零样本泛化。
5. **预测器-动作专家参数不对称**：预测器 0.5B vs. 动作专家 0.1B，后者容量可能成为瓶颈，尤其在长 horizon 或多视角场景。

## 研究启发与可借鉴点
1. **冻结预测性编码器 + 从零训练下游模块**是一种高性价比的 WAM 构建范式，可作为本团队在资源受限场景下替代"适配视频生成器"路线的 baseline。
2. **Future-informed context interface** 的设计（层级 KV 传递而非仅用最终预测）值得借鉴：它实现了预测监督与动作监督的梯度互通，避免了 "predict-then-condition" 两阶段的误差累积。
3. **Predictor-only pretraining on action-free video** 的迁移策略可复用到其他视觉表征空间（如 DINO、MAE），用于评估"无动作监督的未来建模知识"在不同预训练目标下的可迁移性。
4. **Matched visual-substrate 控制对比实验设计**（同一下游架构+同一训练预算比较不同视觉基础）是评估"表征基础有效性"的规范做法，建议纳入本团队未来消融实验的标准协议。
5. **MoT 单向注意力约束**（预测器不 attend 动作，动作可 attend 预测器上下文）是一种简洁的因果隔离设计，可防止动作信号泄露到预测器中，值得在类似多模态耦合架构中参考。

## 关键术语表
**World-Action Model (WAM)**：将未来视觉状态预测与动作生成耦合起来的机器人策略学习范式，通过预测先验提升泛化与任务执行能力。
**V-JEPA (Vision Joint-Embedding Predictive Architecture)**：由 Meta/Facebook AI 提出的视觉表征学习方法，通过在特征空间预测 masked 时空块来学习动力学先验，无需像素重建。
**Future-informed Context Interface**：预测器各层上下文位置的 key–value 状态集合，经双向注意力与未来查询交互后，将未来建模信息传递给动作专家。
**Flow Matching**：一种生成建模技术，学习从噪声到数据分布的连续流场，相比扩散模型在训练和推理效率上更具优势。
**Mixture-of-Transformers (MoT)**：多个 Transformer 子模块共享部分注意力机制的架构设计，本文用于联合预测器与动作专家。
**LIBERO-Plus**：LIBERO 基准的鲁棒性扩展，包含 7 类受控分布偏移（视角/初态/语言/光照/纹理/噪声/布局），共 10,030 个扰动任务。
**DROID Dataset**：大规模真实世界机器人操作视频数据集，包含约 100k+ 视频片段与对应语言指令，本文用于预测器无动作标签预训练。
**Predictive Visual Latent**：由 V-JEPA 等预测性预训练方法学到的视觉特征表示，鼓励建模可预测的时空结构与动力学，而非外观细节。

## 可复现要素
- **数据集**：LIBERO（公开）、LIBERO-Plus（公开）、RoboCasa-GR1（公开）、DROID（公开）、TianJi Marvin 真实平台数据（论文未公开，仅报告成功/失败计数）。
- **代码**：https://github.com/breez3young/VJEPA-Policy（论文声明已开源）。
- **权重**：V-JEPA 2.1 编码器权重来自官方发布；T5-XXL 官方权重；自定义预测器与动作专家权重随代码开源。
- **关键超参**：视觉编码器 V-JEPA 2.1 ViT-L（304M，冻结），文本编码器 T5-XXL（冻结）；预测器 24 层、hidden=1024、16 heads×64 dim；动作专家 24 层、hidden=512、16 heads×64 dim；AdamW lr=1e-4，5% warmup，cosine decay；LIBERO 10 epoch（21,360 步，batch=128），RoboCasa 50k 步（batch=256）；λ_future=1。
