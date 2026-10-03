---
title: "UNIAFFORD-TOKEN-ROUTED-MULTITASK-LEARNING-FOR-GENERALIZABLE"
source: https://arxiv.org/pdf/2609.37264v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-03 02:56:25"
field: "多模态感知与具身智能"
keywords: ["affordance perception", "multimodal large language model", "token routing", "2D-3D grounding", "zero-shot generalization", "dense prediction"]
innovations: ["Token Router for Tasks：解耦任务路由与预定义标记生成，使密集预测损失反向塑造 MLLM 共享表示", "UniAfford 统一框架：通过共享 object-affordance 分类体系与语义级伪配对，联合像素级 2D 与点级 3D 监督", "强 OOD 零样本泛化与 SOTA 分支性能：无需目标域微调即跨数据集迁移，分支独立评测达 state-of-the-art"]
benchmarks: ["AGD20K", "GEAL*", "ReasonAff", "PIAD", "PIADv2"]
---

# 论文速读：UNIAFFORD: TOKEN-ROUTED MULTITASK LEARNING FOR GENERALIZABLE 2D-3D AFFORDANCE PERCEPTION

## 一句话总结
本文提出 **Token Router for Tasks** 多任务训练范式及 **UniAfford** 统一框架，通过共享 MLLM 语义中心与模态感知 token 路由器，联合像素级 2D 与点级 3D 监督信号，实现可迁移的 2D–3D 可动作（affordance）感知，在跨数据集零样本迁移与分支独立评测中均达 SOTA。

## 研究问题与动机
- **2D/3D affordance 研究碎片化**：现有方法在数据集定义、标注格式、评估协议上完全割裂，限制跨视觉-几何空间的可迁移语义学习。
- **多模态输入融合≠统一表征**：已有跨模态方法（如 IAGNet、GREAT）侧重输入端融合，但像素级与点级监督仍各自为政，未建立共享表征接口。
- **MLLM 任务分发依赖预定义标记**：LISA-style 等方法需语言头生成指定分割 token，将任务路由与语言建模强耦合，难以让密集预测损失反向塑造共享表示。
- **跨数据集零样本泛化缺失**：缺乏能在未见 benchmark 上无需目标域微调即表现优异的统一 2D–3D affordance 系统。

## 核心贡献（创新点）
1. **Token Router for Tasks 范式**：直接从 MLLM 上下文隐状态预测分支归属，无需语言头生成预定义任务标记； routed states 由下游分支特定密集预测损失监督，与文本状态的语言建模损失解耦。
2. **UniAfford 统一框架**：以 MLLM 为共享语义中心，经模态感知路由器生成图像/点云 affordance 查询，分别驱动 SAM-style 2D 解码器与 SONATA-based 3D 解码器，支持 2D、3D 及联合推理。
3. **UniAfford-Data 数据集**：在共享 object–affordance 分类体系下整合 RAGNet、ReasonAff（2D）与 PIADv2、AGPIL（3D）数据，构建语义级伪配对（semantic-level pseudo-pair），无需实例级空间对应即可联合异构监督。
4. **强零样本与 SOTA 分支性能**：在 AGD20K、GEAL* 上实现跨数据集 zero-shot 迁移；在 ReasonAff、PIAD/PIADv2 分支独立评测中取得 SOTA，验证联合监督与路由设计的有效性。

## 方法详解
- **统一多模态编码**：文本、图像（SigLIP）、点云（预训练 SONATA 编码器）分别经投影层映射到 MLLM 共享空间，动态拼接为 prefix $\breve{T}^{in}$，送入 MLLM 生成自回归响应序列 $H=[h_1;\dots;h_L]$。
- **模态感知 Token Router**：对每个有效响应状态 $h_t$，路由头 $g_r$ 输出三类别分布 $\{text, img, pc\}$；软概率用于可微路由监督，硬分配 $r_t=\arg\max p_{t,c}$ 用于选择查询；不可用分支 logit 被 mask 为 $-\infty$。
- **分支查询投影**：被路由到 img/pc 分支的状态分别经 $g_{img}, g_{pc}$ 投影为 $q_t^{img}, q_t^{pc}$，按自回归顺序拼接为 $Q^{img}, Q^{pc}$，支持变长序列与 padding mask。
- **2D 解码器（SAM-style）**：图像特征 $F^{img}$ 与 routed query 经可学习投影后计算归一化相似度，生成粗粒度热力图 $M^{img}$，再经 SAM prompt encoder 与 mask decoder 得到像素级 logits $\widehat{Y}^{2D}$。
- **3D 解码器（SONATA-based）**：点云特征 $F^{pc}$ 与 routed query 通过缩放相似度直接输出逐点 affordance logits $\widehat{Y}^{3D}_i$。
- **联合损失函数**：$\mathcal{L}=\lambda_{txt}\mathcal{L}_{txt}+m_{2D}\mathcal{L}_{2D}+m_{3D}\mathcal{L}_{3D}+\mathcal{L}_{router}$；其中 $\mathcal{L}_{2D}$ 为 focal+Dice，$\mathcal{L}_{3D}$ 为 BCE+Dice，$\mathcal{L}_{router}$ 含 token 级 CE、存在性损失（existence）与稀疏性损失（sparsity）。
- **路由标签构造**：响应中下一目标为 `<img-aff>` 或 `<pc-aff>` 的状态分别获得 img/pc 路由标签，这些 anchor 位置从语言建模损失中排除，仅提供路由监督。

## 实验与结果
- **数据集与基线**：训练集 UniAfford-Data（162 类物体、162 类 affordance，25k 2D 样本、69k 3D 样本）；2D 评测 AGD20K、ReasonAff；3D 评测 GEAL*、PIAD/PIADv2；基线包括 Seg-Zero、Vision Reasoner、Affordance-R1、LISA-7B、SAM4MLLM、AffordanceNet、OpenAD、IAGNet、GREAT、LASO 等。
- **OOD 零样本迁移（H1）**：
  - AGD20K：UniAfford gIoU=27.52、cIoU=25.22、SIM=0.37，超越所有非推理型 MLLM 基线，与推理型方法相当。
  - GEAL*：AUC=83.55、mIoU=14.67、SIM=0.565、MAE=0.102，超越所有 zero-shot 基线，并超过在 GEAL 上训练的 LASO/GEAL 参考模型的 AUC 与 SIM。
- **分支独立评测（H2）**：
  - ReasonAff（2D）：gIoU=71.19、cIoU=73.63，SOTA；相对 Affordance-R1（gIoU 67.41）提升 +3.78。
  - PIAD（3D）：mIoU=14.25（相对 DAG 的 9.73 提升 +4.52）、AUC=77.33、MAE=0.107，SOTA。
  - PIADv2（3D）：AUC=75.67、mIoU=9.26，超越 GREAT（AUC 64.15、mIoU 8.08）。
- **消融（H3）**：
  - 固定锚点路由 vs 学习型路由：2D gIoU 62.28→68.79，3D mIoU 17.46→34.56。
  - 仅 2D 训练 vs 联合训练：2D gIoU 41.41→68.79；仅 3D vs 联合：3D mIoU 30.07→34.56。
  - 相似度耦合 vs prompt-style 耦合：2D gIoU 36.97，3D mIoU 14.31，显著下降。
- **效率**：2D 推理吞吐 8.20 samples/s，约为 Affordance-R1（1.05）的 7.8×；3D 推理 15.09 samples/s，延迟 66.27 ms/sample。

## 相关工作脉络
- **2D/3D affordance grounding**：Do et al. (AffordanceNet)、Roy & Todorovic、Vo et al. (OpenAD)、Li et al. (LASO) 等分别发展 2D 像素分割或 3D 点云标注方法，UniAfford 将其统一到共享 taxonomy 下。
- **语言引导 affordance**：LISA (Lai et al., 2024) 用 embedding-as-mask 接口连接语言与分割；AffordanceVLM (Wu et al., 2025a)、DAG (Liu et al., 2025a)、SeqAfford (Yu et al., 2025) 侧重单一模态输出，UniAfford 用语言作跨模态语义接口。
- **跨模态 2D–3D 学习**：IAGNet (Yang et al., 2023)、GREAT (Yang et al., 2024) 强调输入融合与 3D 目标，UniAfford 进一步在输出端实现像素级与点级联合监督。
- **MLLM 多任务路由**：UnifiedMLLM (Li et al., 2024b)、u-llava (Xu et al., 2024)、LLMBind (Zhu et al., 2026) 依赖预定义 task marker；Token Router 解耦路由与标记生成，使密集预测损失可直接塑造共享表示。
- **SAM/SONATA 解码范式**：SAM (Kirillov et al., 2023) 提供 prompt-driven 2D 分割；SONATA (Wu et al., 2025c) 提供自监督 3D 点表征；本文将其桥接为 affordance _dense prediction 后端。

## 局限性与未来方向
- 依赖大型预训练 backbone（Qwen3-VL、SigLIP、SONATA、SAM），训练与推理成本较高（3D 推理 FLOPs 约 1930 GFLOPs/sample）。
- 语义级伪配对不建立实例级空间对应，限制了几何一致性监督。
- 评估聚焦于 affordance grounding benchmark，未涉及闭环机器人操作验证。
- 未来方向：更高效 backbone/解码器设计、结合实例级配对的更大规模数据、真实机器人评估、扩展至更广泛的多任务密集预测。

## 研究启发与可借鉴点
1. **Token Router 范式可迁移**：将任务路由与语言标记生成解耦，允许密集预测损失反向塑造 MLLM 共享表示，适用于其他多模态密集预测任务（如 2D/3D 分割、深度估计）。
2. **语义级伪配对策略**：无需实例对齐即可联合异构监督，为跨模态数据集融合提供实用范式，可迁移至 RGB-D、text+3D 等组合场景。
3. **路由存在性/稀疏性正则**：通过 noisy-or 估计分支激活与 token 分配期望，防止路由坍塌，可复用至其他多分支 MLLM 架构。
4. **语言头诊断方法**：投影 routed states 到 vocab 空间检查语义内容，为理解 MLLM 隐状态质量提供可操作诊断工具。
5. **modality-isolated 评测协议**：分离训练/评估模态以检验分支独立能力，为统一多模态模型提供更具诊断性的评估维度。

## 关键术语表
- **Affordance Perception**：识别物体上支持特定交互的功能区域（如把手可供抓握、平面可供放置）。
- **Token Router for Tasks**：从 MLLM 上下文隐状态直接预测分支归属的路由机制，无需语言头生成预定义任务标记。
- **Semantic-level Pseudo-pair**：按相同 object–affordance 标签匹配不同来源的图像与点云实例，建立跨模态语义关联而非几何对应。
- **Modality-aware Token Router**：感知当前可用模态，将路由概率分布约束到有效分支，并投影为分支特定查询。
- **SAM-style Decoder**：基于 Segment Anything 的 prompt encoder + mask decoder，将语义查询转化为像素级分割热力图。
- **SONATA-based Decoder**：基于自监督点云表征的逐点相似度解码器，输出点级 affordance logits。
- **Existence/Sparsity Loss**：路由辅助正则，分别鼓励可用分支被激活与抑制冗余 query 分配。
- **Modality-isolated Protocol**：仅使用单一模态输入与标注训练/评估对应分支，检验统一架构的独立接地能力。

## 可复现要素
- **数据集**：UniAfford-Data，整合自 RAGNet、ReasonAff、PIADv2、AGPIL；论文未明确声明公开链接，项目主页为 https://4dvlab.github.io/UniAfford/。
- **代码/权重**：论文未明确声明开源仓库，项目主页可能包含。
- **关键超参**：MLLM 使用 Qwen3-VL + LoRA（rank=8, scale=16, dropout=0.05）；AdamW 优化；图像分辨率 1024×1024；点云采样 2048 点；学习率 MLLM 1e-5、2D 解码器 5e-6/1e-5、3D 解码器 5e-4/1e-4、Router 1e-3；损失权重 $\lambda_r=1.0, \lambda_e=0.5, \lambda_s=0.01$。
- **硬件**：NVIDIA B200 GPU。
