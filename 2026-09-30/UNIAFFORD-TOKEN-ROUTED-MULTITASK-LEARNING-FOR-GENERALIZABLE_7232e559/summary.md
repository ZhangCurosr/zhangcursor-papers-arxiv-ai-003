---
title: "UNIAFFORD-TOKEN-ROUTED-MULTITASK-LEARNING-FOR-GENERALIZABLE"
source: https://arxiv.org/pdf/2609.37264v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-03 02:56:28"
field: "多模态感知与具身智能"
keywords: ["affordance perception", "2D-3D grounding", "multimodal large language model", "token routing", "zero-shot generalization"]
innovations: ["提出Token Router for Tasks范式，解耦任务路由与预定义标记生成", "构建统一2D-3D任务感知框架UniAfford，实现异构监督联合学习", "实现OOD零样本泛化和SOTA分支性能"]
benchmarks: ["AGD20K", "ReasonAff", "GEAL", "PIAD", "PIADv2"]
---

# 论文速读：UNIAFFORD: TOKEN-ROUTED MULTITASK LEARNING FOR GENERALIZABLE 2D-3D AFFORDANCE PERCEPTION

## 一句话总结
本文提出了UniAfford，一个基于MLLM的统一2D-3D任务感知框架，通过创新的Token Router机制解耦任务路由与预定义标记生成，实现了在异构像素级和点级监督下学习可迁移的对象-任务语义，在无需目标特定微调的情况下实现跨数据集零样本泛化和最先进的分支性能。

## 研究问题与动机
- **2D与3D任务感知碎片化**：现有研究中2D方法依赖大规模图像数据和语言监督但缺乏显式3D几何，3D方法提供空间定位但依赖昂贵的稀疏点级标注，两者演化出不同的任务定义、监督格式、数据集和评估协议。
- **缺乏跨模态可迁移语义学习**：尽管 multimodal方法引入了额外视觉或语言线索，但多数仍专注于单一输出空间或使用任务特定的融合模块，无法统一像素级和点级学习目标。
- **任务路由机制的局限性**：现有方法（如LISA）通过预定义文本标记和语言头部生成任务标识符来分发任务，将任务分发与预定义标记身份耦合，限制了灵活的分支分配。
- **异构监督的联合利用难题**：不同来源的图像和点云样本可能共享功能语义但不对应同一物理实例，需要统一框架连接这些信号同时保留模态特定的空间监督。

## 核心贡献（创新点）
1. **提出Token Router for Tasks范式**：直接从MLLM上下文隐藏状态预测分支分配，无需语言头部生成预定义任务标记，使密集预测损失能够塑造共享表示。
2. **引入UniAfford统一框架**：将2D和3D任务感知统一到共享对象-任务词汇表下，支持异构监督（像素级2D、点级3D、语言指令）和语义级跨模态配对。
3. **实现强OOD零样本泛化**：在AGD20K和GEAL基准上无需目标特定微调即可实现跨数据集迁移，同时保持模态隔离训练下的SOTA分支性能。
4. **构建UniAfford-Data数据集**：整合25k图像样本和69k点云样本，覆盖162个对象类别和162个任务类别，支持2D、3D及联合训练。
5. **验证设计选择的有效性**：消融实验证明token路由、联合2D-3D监督和解码器耦合的重要性，语言头部诊断揭示路由状态携带有意义的对象-任务语义。

## 方法详解
**整体架构**：
UniAfford采用MLLM作为共享语义中心，结合模态感知的Token Router和双解码器结构，支持图像、点云及多模态输入的任务感知。

**统一多模态编码**（Section 4.2）：
- 文本通过tokenization编码：$T^{\text{txt}} = E_{\text{txt}}^{\text{mllm}}(X)$
- 图像通过SigLIP编码器：$T^{\text{img}} = E_{\text{img}}^{\text{mllm}}(I)$
- 点云通过预训练SONATA编码器：$T^{\text{pc}} = E_{\text{pc}}^{\text{mllm}}(P)$
- 动态拼接可用模态构建统一prefix：$\breve{T}^{\text{in}} = [T^{\text{txt}}; T^{\text{img}}; T^{\text{pc}}]$
- MLLM自回归生成响应：$h_t = \text{MLLM}(T^{\text{in}}, U_{<t})_{\text{last}}$

**模态感知Token Router**（Section 4.3）：
- 对每个有效响应状态$h_t$，路由器预测三类分布：$z_t = g_r(h_t)$, $p_t = \text{softmax}(\tilde{z}_t)$
- 硬分配：$r_t = \arg\max_c p_{t,c}$ 其中$c \in \{\text{text, img, pc}\}$
- 分支特定投影生成查询：$q_t^{\text{img}} = g_{\text{img}}(h_t)$, $q_t^{\text{pc}} = g_{\text{pc}}(h_t)$
- 按自回归顺序拼接：$Q^{\text{img}} = [q_t^{\text{img}} | r_t = \text{img}]$, $Q^{\text{pc}} = [q_t^{\text{pc}} | r_t = \text{pc}]$
- 路由标签基于移位响应目标：若下一目标为`<img-aff>`则获图像路由标签，`<pc-aff>`获点云路由标签，其余为文本标签
- 锚点标记排除在语言建模损失外，仅提供路由监督

**任务感知解码器**（Section 4.4）：
- **2D解码器**（SAM风格）：提取密集图像特征$F^{\text{img}} = E_{\text{img}}^{\text{dec}}(I)$，通过投影相似度构建粗糙热力图：$M_{u,v}^{\text{img}} = s_{\text{img}}\langle\text{Norm}(\phi_{\text{img}}(F_{u,v}^{\text{img}})), \text{Norm}(\gamma_{\text{img}}(q^{\text{img}}))\rangle$，经SAM prompt编码器和解码器细化得到像素级预测$\widehat{Y}^{\text{2D}}$
- **3D解码器**（SONATA风格）：提取密集点特征$F^{\text{pc}} = E_{\text{pc}}^{\text{dec}}(P)$，计算点级相似度：$\widehat{Y}_i^{\text{3D}} = s_{\text{pc}}\langle\text{Norm}(\phi_{\text{pc}}(F_i^{\text{pc}})), \text{Norm}(\gamma_{\text{pc}}(q^{\text{pc}}))\rangle$

**训练目标**（Section 4.5）：
总损失函数：$\mathcal{L} = \lambda_{\text{txt}}\mathcal{L}_{\text{txt}} + m_{\text{2D}}\mathcal{L}_{\text{2D}} + m_{\text{3D}}\mathcal{L}_{\text{3D}} + \mathcal{L}_{\text{router}}$
- 语言建模损失：排除锚点的标准交叉熵
- 2D任务损失：Focal + Dice组合 $\mathcal{L}_{\text{2D}} = \lambda_f\mathcal{L}_{\text{focal}} + \lambda_d\mathcal{L}_{\text{dice}}$
- 3D任务损失：BCE + Dice组合 $\mathcal{L}_{\text{3D}} = \lambda_b\mathcal{L}_{\text{bce}} + \lambda_{pd}\mathcal{L}_{\text{dice}}$
- 路由器损失：$\mathcal{L}_{\text{router}} = \lambda_r\mathcal{L}_{\text{route}} + \lambda_e\mathcal{L}_{\text{exist}} + \lambda_s\mathcal{L}_{\text{sparse}}$
  - 路由交叉熵：监督分支分类
  - 存在损失：鼓励监督分支覆盖
  - 稀疏损失：防止冗余查询分配

**数据集构建策略**（Section 3.2）：
- 语义级伪配对：将相同对象-任务标签的图像和点云实例关联，不要求实例级空间对应
- 保留模态特定的空间标注：2D使用像素级掩码，3D使用点级标注
- 支持图像单模态、点云单模态和语义配对多模态样本

## 实验与结果
**实验设置**：
- 基座模型：Qwen3-VL，LoRA微调（rank=8, scale=16, dropout=0.05）
- 优化器：AdamW，线性warmup + 余弦学习率衰减
- GPU：NVIDIA B200
- 图像分辨率：1024×1024，点云采样：2048点

**H1：OOD零样本泛化**（Table 2）：
- **2D转移（AGD20K）**：gIoU 27.52（领先），cIoU 25.22，P50 19.88，SIM 0.37（最高），超越所有非推理方法和大多数推理方法
- **3D转移（GEAL\*）**：AUC 83.55，mIoU 14.67，SIM 0.565，MAE 0.102，超越所有zero-shot基线，甚至超过在GEAL上训练的LASO和GEAL参考模型

**H2：分支性能**（Table 3）：
- **2D分支（ReasonAff）**：gIoU 71.19，cIoU 73.63，P50 80.94，SOTA
- **3D分支（PIAD）**：AUC 77.33，mIoU 14.25（提升4.52绝对值），SIM 0.414，MAE 0.107，SOTA
- **3D分支（PIADv2）**：AUC 75.67，mIoU 9.26，超越GREAT

**H3：消融实验**（Table 4）：
- **路由 vs 固定锚点**：2D gIoU提升6.51（62.28→68.79），3D mIoU提升17.10（17.46→34.56）
- **联合学习 vs 单模态**：2D gIoU提升27.38（41.41→68.79），3D mIoU提升4.49（30.07→34.56）
- **耦合方式**：相似度耦合显著优于prompt风格耦合（2D gIoU 68.79 vs 36.97）

**效率分析**：
- 2D推理吞吐量：8.20 samples/s，约为Affordance-R1的7.8倍
- GPU内存占用：峰值约11.60 GiB

## 相关工作脉络
1. **2D/3D任务感知独立发展**：Do et al. (2018), Roy & Todorovic (2016)的2D方法依赖外观线索；Vo et al. (2023), Li et al. (2024a)的3D方法提供空间定位但标注昂贵——UniAfford统一两者于共享语义框架
2. **语言引导任务感知**：LISA (Lai et al., 2024)通过embedding-as-mask接口连接语言状态和掩码预测；AffordanceVLM配合RAGNet实现指令引导——UniAfford扩展至联合2D-3D
3. **跨模态2D-3D学习**：IAGNet (Yang et al., 2023)转移交互线索，GREAT (Yang et al., 2024)结合几何属性和交互意图——UniAfford进一步实现联合输出监督
4. **任务路由与MLLM多任务接口**：Li et al. (2024b), Xu et al. (2024)等使用预定义任务token——UniAfford解耦路由与标记生成
5. **现有数据集局限**：UMD、HANDAL、AGD20K仅2D；3D-AffordanceNet、Affogato、LASO仅3D；AGPIL和PIADv2虽支持多模态但未统一——UniAfford-Data提供统一索引和语义配对

## 局限性与未来方向
- **计算成本较高**：依赖大型预训练骨干网络（Qwen3-VL、SAM、SONATA），推理成本高于专用模型
- **缺乏实例级空间对应**：语义级配对连接不同实例但不建立几何一致性监督
- **未验证闭环机器人操作**：评估聚焦任务感知基准而非实际机器人操作
- **未来方向**：探索更高效骨干和解码器设计、结合实例级配对的大规模数据、真实世界机器人评估、扩展至更广泛的多任务密集预测

## 研究启发与可借鉴点
1. **Token Router机制的可迁移性**：解耦任务路由与预定义标记生成的思路可应用于其他多模态多任务学习场景（如VQA+分割+检测统一框架）
2. **语义级伪配对策略**：不要求实例级对齐的跨模态配对方法适用于其他2D-3D联合学习任务（如分割、检测）
3. **联合监督塑造共享表示**：异构密集预测损失通过可微路由塑形MLLM表示的设计模式可推广至多任务视觉-语言学习
4. **语言头部诊断方法**：通过语言头解码路由状态以验证语义内容的诊断技术可复用于其他MLLM多任务系统
5. **模态隔离评估协议**：分别训练评估各分支以验证架构独立能力的实验设计值得借鉴

## 关键术语表
**Affordance Perception（任务感知）**：定位支持具身交互的对象区域，如抓取杯柄、按压按钮、坐在椅子上
**Token Router for Tasks**：从MLLM上下文隐藏状态直接预测分支分配的新范式，无需语言头部生成预定义任务标记
**Semantic-level Pseudo-pairing（语义级伪配对）**：通过共享对象-任务标签关联不同物理实例的图像和点云，不要求实例级空间对应
**Modality-isolated Protocol（模态隔离协议）**：单独训练和评估每个分支，使用目标视觉模态和任务指令，无辅助观察
**OOD Zero-shot Generalization（OOD零样本泛化）**：在训练分布外基准上无需目标特定微调的直接评估，测试跨数据集迁移能力
**Similarity-based Decoder Coupling（相似度解码器耦合）**：通过投影后向量相似度连接路由语义查询与空间特征，替代prompt风格接口
**gIoU/cIoU**：图像级和聚合级交并比，衡量2D预测掩码与标注的重叠程度
**AUC/mIoU/SIM/MAE**：3D评估指标，分别衡量排序质量、阈值化重叠、分布相似性和点级误差

## 可复现要素
- **数据集**：UniAfford-Data，论文未提及是否开源（Project page: https://4dvlab.github.io/UniAfford/）
- **代码**：论文未明确说明开源状态
- **权重**：论文未提及模型权重是否开源
- **关键超参**：
  - LoRA: rank=8, scale=16, dropout=0.05
  - 图像分辨率: 1024×1024
  - 点云采样数: 2048
  - 学习率: MLLM=1e-5, 2D Decoder=5e-6/1e-5, 3D Decoder=5e-4/1e-4, Router=1e-3
  - 优化器: AdamW
  - GPU: NVIDIA B200
  - 损失权重: λ_txt=1.0, λ_route=1.0, λ_exist=0.5, λ_sparse=0.01
