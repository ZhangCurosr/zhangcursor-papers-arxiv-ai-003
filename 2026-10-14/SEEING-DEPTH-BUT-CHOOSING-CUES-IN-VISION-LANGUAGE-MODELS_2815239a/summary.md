---
title: "SEEING-DEPTH-BUT-CHOOSING-CUES-IN-VISION-LANGUAGE-MODELS"
source: https://arxiv.org/pdf/2610.17976v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-01 22:47:11"
field: "视觉-语言模型空间推理与可解释性"
keywords: ["vision-language models", "spatial reasoning", "depth perception", "interpretability", "intervention", "linear probe", "shortcut learning"]
innovations: ["揭示VLMs在深度判断中系统性跟随图像平面线索而非使用内部深度信息", "证明正确深度顺序可从物体表征差值中线性解码且在被误导决策时仍保留", "提出无参数对象级写回干预，利用模型自身深度读出来纠正冲突决策"]
benchmarks: ["SUN RGB-D", "SpatialRGPT", "SpatialTunnel", "CV-Bench"]
---

# 论文速读：SEEING-DEPTH-BUT-CHOOSING-CUES-IN-VISION-LANGUAGE-MODELS

## 一句话总结
该论文诊断了视觉-语言模型（VLMs）在深度判断任务中为何会犯错：研究发现，当图像平面的统计线索（视大小、垂直位置）与真实深度冲突时，VLMs 会系统性跟随误导性线索，尽管正确的深度关系仍存在于模型内部表征中；作者进一步提出一种"对象级写回"干预方法，无需更新参数即可纠正部分冲突决策。

## 研究问题与动机
1. **核心问题**：VLMs 在进行 pairwise 深度判断时，究竟使用内部深度信息，还是依赖图像平面上的统计捷径（如物体看起来更大、位置更低）？
2. **现实背景**：在自然 2D 图像中，"近物显得更大、位置更低"具有统计预测性，VLMs 在学习过程中有充分动机习得这些捷径，但这些捷径在真实 3D 场景中会失效。
3. **已有方法不足**：Prior work 仅观察到 VLMs 对透视线索敏感（Min et al., 2026），但未回答"模型是否在表征层面仍持有深度信息"以及"如何在不微调参数的情况下增强该信息的影响力"。
4. **动机来源**：理解这一 mismatch 可为空间推理增强提供新路径——若深度信息已存在但未被正确调用，则可通过干预而非重新训练来解决。

## 核心贡献（创新点）
1. **系统性诊断**：跨越 5 个家族、11 个 VLM（含不同 scale 及空间微调 checkpoint SpaceThinker），证明深度判断在 cue-conflict 条件下系统性偏向误导性线索，且模型 scaling 或空间微调均无法消除该现象。
2. **表征解耦发现**：揭示正确的深度顺序可从两物体视觉表征的差值中线性解码（probe accuracy >97% on rendered scenes, 82.9%~91.7% on real photos），且在模型给出错误答案时该深度信号依然存在。
3. **无参数干预方法**：提出 object-level write-back，利用模型自身的深度读出来反向修改两个物体的 hidden state，无需 ground-truth 深度标注、无需更新 VLM 参数即可部分纠正冲突决策（如 Qwen3-VL-8B 在 rendered size-conflict 下从 50.2% 提升至 90.8%）。
4. **因果性验证**：通过多组对照实验（random direction、answer-token write、cue-avoidant prompting、depth input）证明干预效果特异性依赖于深度方向 v、写入位置（两物体 contrast 而非 answer token）及写入符号的正确性。

## 方法详解
### 3.1 实验设置：受控深度-线索冲突
- **渲染场景（Rendered）**：在棋盘格房间中放置两个 primitive，独立采样深度（2.5~7.0m）、角大小（0.055~0.115）和垂直位置（−0.18~0.18），生成四类 split：consistent、size-conflict、vertical-conflict、decorrelated。每 split 500 场景，图像分辨率 448×448。
- **真实图像（Real）**：基于 SUN RGB-D 传感器深度，选取物体深度差 ≥0.4m 且深度比 ≥1.20 的配对，按 mask area 和 centroid row 测量 2D 线索，构建 consistent / size-conflict / vertical-conflict 子集（共 947 pairs）。
- **评估指标**：depth accuracy（选择真正更近物体的比例），50% 为随机水平。

### 3.2 深度表征线性读取
- 对每个 queried 物体，将其 segmentation mask 内的 visual tokens mean-pool 得到 $c_A$、$c_B$，构造对比向量 $d = c_A - c_B$。
- 在 decorrelated split 上拟合 logistic regression probe：
$$p(y_i=1|d_i; w, b) = \sigma(w^\top \tilde{d}_i + b)$$
其中 $\tilde{d}_i = (d_i - \mu)/s$ 为标准化后的对比向量。
- 将最优权重映射回原始隐空间并单位化得到深度方向 $v$，对新样本计算深度 trace $z = v^\top d$；$z>0$ 表示物体 A 更近。
- Probe 在 Qwen3-VL-8B 上于各语言层达到 97.3%+（rendered）和 82.9%~91.7%（real），且图像替换控制显示该读距强烈依赖当前视觉输入而非类别先验。

### 3.3 对象级写回干预（Object-level Write-back）
- 在选定层 ℓ 输出后，对两物体的 hidden states 施加反向扰动：
$$h'_s = h_s + \alpha m \cdot \text{sign}(z) \cdot (\mathbb{1}[s \in O_A] - \mathbb{1}[s \in O_B]) \cdot v$$
其中 $m$ 为该层所有 token 的平均范数，$\alpha$ 为强度。
- 更新后深度 trace 变为 $z' = z + 2\alpha m \cdot \text{sign}(z)$，即放大模型自身已读出的深度顺序，而非引入外部标注。
- 层 ℓ 和强度 $\alpha$ 在 development 集上通过最大化 $\min\{R_{\text{size}}, R_{\text{vertical}}\}$ 并约束 consistent accuracy 下降 ≤2% 来选择，之后冻结用于 evaluation。
- Qwen3-VL-8B 最佳设置：rendered $\ell=7, \alpha=1.0$；SUN RGB-D $\ell=8, \alpha=0.8$。

## 实验与结果
### 评测基线与数据集
- **模型**：5 个家族共 11 个 VLM：Qwen3-VL（4B/8B/32B）、LLaVA-OneVision（1.5-8B/2-8B）、InternVL3（8B/38B）、Gemma-4（12B/31B）、空间微调 checkpoint SpaceThinker（Qwen2.5-VL-3B）。
- **数据集**：自建 Rendered（2000 场景，4 split × 500）和 SUN RGB-D 筛选配对（947 pairs）。
- **外部基准**：SpatialRGPT、SpatialTunnel、CV-Bench。

### 主要结果
1. **Cue-conflict 下性能断崖式下跌**：
   - Qwen3-VL-8B：rendered size-conflict 42.8% → vertical-conflict 42.8%（低于 50% 随机线）；SpaceThinker 分别仅达 51.8% 和 32.4%。
   - Vertical position 是更强线索（8/11 模型中垂直冲突导致更大跌幅）。
   - 真实图像中同样存在该系统性偏移，但差距较小。
2. **深度信息在表征层面被保留**：
   - Qwen3-VL-8B probe accuracy：rendered >97.3%（所有语言层），real 82.9%~91.7%。
   - 在 99.0% 的例子中 probe 正确，但模型仅 42.8% 答对（rendered vertical-conflict）。
3. **Write-back 干预效果**（Table 1）：
   - Qwen3-VL-8B rendered：size-conflict 50.2%→90.8%（+40.6pp），vertical-conflict 42.8%→71.4%（+28.6pp）；consistent 94.4%→99.6%。
   - LLaVA-OV-1.5-8B rendered：size 42.0%→83.6%（+41.6pp），vertical 7.4%→47.0%（+39.6pp）。
   - InternVL3-38B 和 Gemma-4 系列效果较弱甚至为负。
   - 真实图像上 Qwen3-VL 和 LLaVA-OV 获得提升，InternVL3 和 Gemma-4 部分条件出现负收益。
4. **Ablation 验证特异性**（Table 2）：
   - Cue-avoidant CoT、depth input、random direction、answer-token write 均未能复现 write-back 的增益。
   - 反转深度方向产生相反行为偏移；图 10 显示 write-back 使 answer logit margin 与深度 trace 建立更强相关性。

## 相关工作脉络
1. **Min et al. (2026)**：发现 VLM 深度判断对垂直位置等透视线索敏感；本文进一步控制视大小和垂直位置，并检验错误答案下深度信息的表征保留情况。
2. **Liao et al. (2026)**：通过探针和视觉消融区分 grounded / prior-driven / inverted 深度表征；本文聚焦 image-dependent 深度信息与误导性 image-plane 线索之间的竞争关系。
3. **Kang et al. (2026)**：发现线性空间方向并可对其进行 steering；本文不从方向 steering 入手，而是从 queried-object contrast 读取深度并通过 write-back 增强其影响力。
4. **SpatialVLM (Chen et al., 2024) / SpatialRGPT (Cheng et al., 2024)**：通过额外监督微调提升空间能力；本文在 frozen model 上测试深度信息对决策的影响，无需参数更新。
5. **Causal representation analysis 脉络**（Belrose et al., 2023; Li et al., 2023a; Rimsky et al., 2024）：本文遵循该脉络，通过控制实验分离 decodability（探针可读性）与 intervention effect（写回因果影响）。

## 局限性与未来方向
1. **仅分析 pairwise 深度判断和两类线索**（视大小、垂直位置），未涵盖纹理、阴影、遮挡等其他深度线索。
2. **真实图像中其他场景属性未被独立控制**，cue-conflict 程度不如渲染场景极端，效应量较小。
3. **干预效果的架构/域依赖性显著**：InternVL3-38B 和 Gemma-4 在部分条件下效果减弱甚至负向，跨模型泛化性有限。
4. **外部基准迁移性受限**：在 CV-Bench 和部分 SpatialTunnel 设置上干预无效或有害。
5. **未来方向**：扩展至更多深度线索、更通用的空间推理任务，以及理解为何同一深度 trace 在不同架构中对决策的影响力存在差异。

## 研究启发与可借鉴点
1. **Probe + Intervention 的诊断范式**：先用线性探针确认某信息是否存于表征中，再通过写回/steering 验证其因果影响力，可用于其他"看得见但用不对"模型的诊断场景。
2. **Object-level contrast 作为信息读取接口**：两物体 hidden state 差值可有效编码 relational 信息（深度顺序），这一设计可迁移至相对属性判断（前后、大小、距离）任务。
3. **Write-back 无参数干预的价值**：仅修改特定层的 object tokens 即可改变最终 logits 分布，为测试时增强（test-time intervention）提供低成本方案，无需梯度更新或外部数据。
4. **严格控制线索-深度关系的实验设计**：渲染场景独立操控深度/大小/位置三元组，配合真实图像传感器深度筛选，为"捷径诊断"类研究树立了可复现的 benchmark 构建范式。
5. **与团队方向结合机会**：可借鉴 probe 方法诊断本团队模型在时序推理、因果判断等任务中是否存在类似"表征保留但决策被误导"的 mismatch 现象。

## 关键术语表
**Depth–cue conflict**：图像平面线索（如视大小、垂直位置）所暗示的深度顺序与真实 3D 深度顺序相违背的实验条件。
**Depth trace**：通过线性 probe 学到的深度方向 $v$ 与物体对比向量 $d$ 的点积 $z = v^\top d$，正值表示物体 A 更近，负值表示物体 B 更近。
**Object-level write-back**：在选定 Transformer 层，沿深度方向 $v$ 对两个 queried 物体的 hidden states 施加符号相反的更新，以放大模型自身已读出的深度顺序。
**Linear probe**：在固定层输出上训练的 logistic regression 分类器，用于检测某一抽象概念是否线性可分地编码在 hidden representations 中。
**Decorrelated condition**：视大小和垂直位置与真实深度统计独立的数据划分，用于无偏拟合深度方向 $v$。
**Cue-consistent accuracy**：模型在 2D 线索与真实深度一致条件下的判断准确率，作为干预不应显著降低的性能基准。
**Answer letter probe**：在 answer state 上训练的分类器，预测"正确答案对应的选项字母"（而非"哪一物体更近"），用于检验表征中是否存在但未作用于决策的信息。
**Spatial fine-tuning**：在合成空间推理轨迹数据上对 VLM 进行的后训练，本文验证其仍无法消除 cue-conflict 下的性能下降。

## 可复现要素
- **数据集**：自建 Rendered 数据集（4 split × 500，含 scene manifest）；SUN RGB-D 配对（947 pairs，含 image-level manifest 记录深度标签、选项分配、线索测量值）。数据集声明随代码一并开源（anonymous repo）。
- **代码/权重**：代码及结果 artifact 可在匿名仓库获取；模型权重为公开 VLM（Qwen3-VL、LLaVA-OV、InternVL3、Gemma-4、SpaceThinker）。
- **关键超参**：Qwen3-VL-8B rendered $\ell=7, \alpha=1.0$；real $\ell=8, \alpha=0.8$。Probe 正则化 $\lambda=2\times10^{-3}$，standardization floor $10^{-6}$，L-BFGS 最多 200 次迭代；probe split 80/20 random，seed=0。
- **图像规格**：Rendered 448×448，depth 2.5~7.0m，分离≥1.0m；mask≥60 pixels；SUN RGB-D 深度差≥0.4m，深度比≥1.20，Box IoU≤5%，mask 像素 1500~图像 1/4。
