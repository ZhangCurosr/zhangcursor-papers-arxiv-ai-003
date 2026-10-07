---
title: "Transferable-Spatial-Temporal-Coherence-Adversarial-Attack-o"
source: https://arxiv.org/pdf/2610.08331v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-07 22:28:10"
field: "多模态对抗安全"
keywords: ["Vision-Language Model", "Adversarial Attack", "Autonomous Driving", "Temporal Coherence", "Transferability", "Video Understanding"]
innovations: ["首次提出时序连贯性攻击破坏VLM的视频时序推理", "Caption引导帧选择与YOLO运动掩码结合的靶向扰动策略", "揭示领域专用VLM（Dolphin）具有更强对抗鲁棒性"]
benchmarks: ["BDD100K", "nuScenes"]
---

# 论文速读：Transferable-Spatial-Temporal-Coherence-Adversarial-Attack-o

## 一句话总结
论文提出了一种面向自动驾驶场景的**时空连贯性对抗攻击框架（STCA）**，通过三阶段攻击策略（模态扩展、空间攻击、时序攻击）对黑盒视觉语言模型（VLM）进行迁移性攻击，首次系统性揭示了VLM在视频时序维度上的脆弱性。

## 研究问题与动机
- **现有攻击方法缺失时序维度**：当前针对VLM的对抗攻击主要聚焦于单帧图像的空间扰动，忽略了视频序列中相邻帧之间的时序连贯性，无法全面评估攻击效果。
- **自动驾驶场景的特殊性**：自动驾驶VLM依赖连续帧的时序推理（如"车辆停止后行人穿越"），单纯空间扰动无法破坏模型的时序语义理解。
- **跨架构迁移性不足**：现有VLM对抗攻击在黑盒场景下跨模型迁移性能有限，需要设计更具通用性的攻击策略。
- **语义敏感区域未被充分利用**：现有方法通常对整个图像均匀施加扰动，未针对语义重要区域（如移动物体、交通参与者）进行靶向攻击。

## 核心贡献（创新点）
- **三阶段STCA攻击框架**：结合模态扩展、空间攻击与时序攻击，首次在VLM对抗攻击中显式破坏相邻帧的时序连贯性，与仅针对单帧的空间攻击形成本质区别。
- **Caption引导的帧选择策略**：利用CLIP模型计算帧与多Caption的余弦相似度，自动筛选语义最相关的帧进行扰动，避免对静态背景等无关区域的无效攻击。
- **YOLO-guided掩码扰动机制**：通过YOLOv8检测关键物体并构建二元掩码，将对抗扰动限制在语义重要区域，提升攻击有效性的同时保持高视觉相似度（SSIM≥0.79）。
- **运动引导的时序攻击损失**：设计Temporal Coherence Loss，通过LanguageBind编码器最小化相邻帧特征的余弦相似度，专门针对视频中的运动区域施加扰动。
- **黑盒迁移性验证**：在三个架构各异的VLM模型（Video LLaVA、Qwen2.5-VL、Dolphin）上验证了从白盒代理模型到黑盒目标的强迁移能力。

## 方法详解
**STCA框架包含三个阶段：**

1. **模态扩展（Modalities Expansion）**：
   - 使用Gemini 2.5和Video-LLaVA生成多个语义等价但表达不同的Caption
   - 利用语义熵（Semantic Entropy）过滤冗余Caption，保留信息最丰富的描述
   - 通过CLIP模型计算每个Caption与视频所有帧的平均余弦相似度，选取Top-K最优Caption

2. **空间攻击（Spatial Attack）**：
   - **Caption引导的帧选择**：利用公式 $s_i = v_i^\top \bar{u}$ 计算每帧与Caption集合的语义相似度，选取Top-K语义最相关帧
   - **YOLO掩码生成**：使用YOLOv8检测关键物体，构建二元掩码 $M \in \{0,1\}^{H \times W}$，仅对检测到的物体区域施加扰动
   - **PMP扰动优化**：基于Precision Mask Perturbations框架，在 $\ell_\infty$ 范数约束下最大化视觉-文本嵌入的对齐损失：
     $$\mathcal{L} = -\sum_{j=1}^{3} (E_I(i') \cdot E_T(C_j))$$
   - 应用多尺度增强策略（$S=\{0.5, 0.75, 1.0, 1.25, 1.5\}$）提升迁移性

3. **时序攻击（Temporal Attack）**：
   - **运动掩码计算**：$\mathcal{M}_{motion} = |\mathcal{M}_{t+1} - \mathcal{M}_t| > \mathcal{T}_m$（阈值0.1），仅对连续帧间发生变化的区域施加扰动
   - **Text-Visual Alignment Loss**：
     $$\mathcal{L}_{text} = \frac{1}{|\mathcal{T}|}\sum_{j=1}^{|\mathcal{T}|} \cos(E_V(i^{adv}), E_T(t_j))$$
   - **Temporal Coherence Loss**：
     $$\mathcal{L}_{temporal} = \frac{1}{K-1}\sum_{t=1}^{K-1} \mathcal{M}_t \cdot \cos(E_V(i_t^{adv}), E_V(i_{t+1}))$$
   - 使用LanguageBind视频编码器作为白盒代理模型进行攻击生成

**攻击超参数**：
- 空间攻击：$\epsilon=32/255$, $\alpha=1/255$, $T=20$步，动量衰减$\lambda=0.9$
- 时序攻击：$\epsilon=16/255$, $\alpha=2/255$, $T=20$步

## 实验与结果
**数据集**：BDD100K（800个视频）、nuScenes（85个场景，每个约40帧）

**目标模型**：Video LLaVA-7B、Qwen2.5-VL-7B、Dolphin（自动驾驶专用VLM）

**评估指标**：攻击成功率（ASR，基于Jaccard词重叠<0.5判定成功）、结构相似度（SSIM）

**主要结果（BDD100K）**：
| 模型 | 空间攻击ASR | 完整STCA ASR | SSIM |
|------|------------|-------------|------|
| Video LLaVA | 32.6% | **71%** | 0.82 |
| Qwen2.5-VL | 45% | **84.2%** | 0.82 |
| Dolphin | 46.9% | **46.2%** | 0.82 |

**主要结果（nuScenes）**：
| 模型 | 空间攻击ASR | 完整STCA ASR | SSIM |
|------|------------|-------------|------|
| Video LLaVA | 57.6% | **83%** | 0.79 |
| Qwen2.5-VL | 64.7% | **96.5%** | 0.79 |
| Dolphin | 37.6% | **47.1%** | 0.79 |

**对比基线**：在相同代理模型下，PGD（Video LLaVA 47.2%、Qwen2.5 44.8%、Dolphin 36.2%）和FGSM均显著低于STCA。

**关键发现**：
- 时序攻击使ASR较纯空间攻击提升**超过2倍**（如Video LLaVA从32.6%→71%）
- Dolphin因领域专用微调表现出更强鲁棒性（ASR仅~47%），揭示**领域专业化可提升对抗鲁棒性**
- Qwen2.5-VL最易受攻击（nuScenes ASR达96.5%），反映不同架构对时序扰动的敏感性差异

## 相关工作脉络
- **AdvCLIP**（Zhou et al.）：针对CLIP模型的下游无关对抗攻击，但仅评估分类和检索任务，未涉及视频时序维度
- **PG-Attack**（Fu et al.）：结合精确掩码扰动与欺骗性文本补丁的黑盒攻击，但聚焦静态图像而非视频时序一致性
- **ADvLM**（Zhang et al.）：首个针对自动驾驶VLM的攻击，但为白盒设置且未考虑时序动态
- **CAD**（Wang et al.）：首个自动驾驶VLM黑盒攻击，通过决策链破坏攻击，但与本文的时序连贯性攻击方向不同
- **BTC**（Kim et al.）：针对视频识别模型的时序攻击，降低相邻帧特征相似度，但目标是分类而非VLM的语义输出

## 局限性与未来方向
- **Dolphin模型的鲁棒性差异**：领域专用模型表现出显著更高的对抗鲁棒性，但其防御机制尚未被系统性分析
- **攻击计算开销**：多阶段框架涉及多个模型推理（Gemini、CLIP、YOLO、LanguageBind），实时性受限
- **迁移性边界未明确**：未系统研究不同代理模型选择对迁移性能的影响
- **物理世界部署验证缺失**：仅在数字域评估，未验证物理场景（如道路标识、车辆贴纸）的可行性
- **防御机制缺位**：论文指出防御需求迫切，但未提供有效的对抗训练或鲁棒性增强方案

## 研究启发与可借鉴点
- **时序连贯性攻击范式**：首次将时序破坏引入VLM对抗攻击，可为其他视频理解模型（如动作识别、视频问答）的安全评估提供参考
- **运动引导掩码策略**：通过连续帧掩码差异定位运动区域，可迁移至视频增强、目标追踪等领域的鲁棒性研究
- **领域专业化与鲁棒性关系**：Dolphin的发现揭示了模型专业化可能带来对抗鲁棒性增益，值得在模型设计阶段考虑
- **Caption多样性增强迁移性**：多Caption策略提升了对语义变体的覆盖，可借鉴于提升攻击对模型输入解释的泛化能力
- **开源代码与复现价值**：若代码开源，可直接作为后续对抗攻击研究的基准测试平台

## 关键术语表
- **STCA（Spatio-Temporal Coherence Adversarial Attack）**：本文提出的时空连贯性对抗攻击框架，包含模态扩展、空间攻击和时序攻击三阶段
- **VLM（Vision Language Model）**：融合视觉与语言理解的 multimodal 模型，如Video LLaVA、Qwen2.5-VL等
- **ASR（Attack Success Rate）**：攻击成功率，通过前后输出文本的词重叠相似度（Jaccard<0.5）判定
- **SSIM（Structural Similarity Index）**：结构相似度指数，衡量对抗样本与原帧的视觉相似程度
- **LanguageBind**：开源的多模态预训练模型，用于对齐视频与文本嵌入空间，本文作为时序攻击的白盒代理模型
- **Semantic Entropy**：语义熵，用于评估生成Caption的信息密度并过滤冗余描述
- **Motion-Guided Mask**：运动引导掩码，通过连续帧YOLO掩码差异识别视频中动态变化区域
- **Black-Box Transferability**：黑盒迁移性，指在仅使用白盒代理模型生成对抗样本的情况下，攻击对未知目标模型的有效性

## 可复现要素
- **数据集**：BDD100K（公开）、nuScenes（公开）
- **代码/权重**：论文未明确声明开源，但提及使用了开源模型（Video LLaVA、Qwen2.5-VL、Dolphin、LanguageBind、YOLOv8、CLIP、Gemini）
- **关键超参**：$\epsilon_{spatial}=32/255$, $\epsilon_{temporal}=16/255$, $\alpha=1/255$(空间)和$2/255$(时序), $T=20$步, 多尺度$\{0.5,0.75,1.0,1.25,1.5\}$, 运动掩码阈值$\mathcal{T}_m=0.1$, 词重叠判定阈值$0.5$
