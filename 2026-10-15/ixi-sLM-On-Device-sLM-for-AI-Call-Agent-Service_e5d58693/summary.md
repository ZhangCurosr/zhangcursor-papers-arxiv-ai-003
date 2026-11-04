---
title: "ixi-sLM-On-Device-sLM-for-AI-Call-Agent-Service"
source: https://arxiv.org/pdf/2610.19382v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-04 03:56:48"
---

# 论文速读：ixi-sLM: On-Device sLM for AI Call Agent Service

## 一句话总结
本文提出 ixi-sLM，一款面向移动端 NPU 的 1.2B 韩语小语言模型，通过基于真实通话流量的词表裁剪与格式化 token 合并、结合 SeqMSE 敏感度与 top-k 分布对齐的混合精度量化，在约 1 GiB 内存与严格温控约束下实现了云侧 SOTA LLM 94% 的任务质量，并已稳定部署至数百万用户的 AI 通话代理服务中。

## 研究问题与动机
- 端侧部署 AI 通话代理可保障隐私、脱离网络依赖并消除按次云计费，但商用落地面临五大硬性约束（C1–C5）：~1 GiB 共享内存预算、通话中延迟与热节流限制、静态整数 NPU 预编译图执行、韩语 STT 输入（含填充词/缺标点）与预训练分布差异大、需与云端服务直接比质的用户体验。
- 现有通用端侧框架（llama.cpp、ExecuTorch、ANEMLL 等）将分词与量化视为固定配置菜单，无法针对特定服务流量与 NPU 执行模型进行联合适配，导致在真实设备上图编译失败或输出不可用。
- 传统 PTQ 校准多依赖 WikiText 等干净文本，无法准确反映 STT 真实调用分布，易造成敏感模块过度压缩或模型生成行为偏离业务需求。
- 韩语为黏着语，通用 BPE 词表会将其切分为冗长序列，在固定上下文窗口下严重浪费内存并拖慢推理；同时 tied embedding/LM head 在 INT4 权重下占比近半，是压缩瓶颈。

## 核心贡献（创新点）
- **基于调用频率的领域自适应建模**：在去标识化通话数据上重绘词频阈值将词表从 102,400 压缩至 65,536，并复用保留槽位将高频韩语表达与固定输出标记合并；与仅做语言过滤或通用继续预训练的前作不同，本文的截断与合并直接由目标服务流量驱动，实现零新增 footprint 的嵌入头缩减与序列压缩。
- **任务感知混合精度量化**：采用真实通话校准，结合 SeqMSE 逐模块评估激活敏感度，将低敏感模块（`o_proj`/`down_proj`）降至 INT4、高敏感模块保留 INT8，并以 logit lens top-k 分布对齐作为最终选型依据；区别于全局 INT4 或依赖 GPU 浮点统计的常规 Mobile Quantization，该方法在静态整数 NPU 约束下精准平衡体积与质量。
- **跨平台统一编译流水线**：构建 Opt.Studio 导出链路，一套配方同步支持 Qualcomm 与 Apple NPU，将领域适配与量化决策固化于导出阶段；与多后端需重复调参的通用框架相比，该流水线直接满足移动端静态图的确定性执行与数值一致性要求。

## 方法详解
- **词表裁剪与领域感知分词**：EXAONE4.0 1.2B 原始词表 102,400，tied embedding + LM head 在 INT4 权重下占 419 MiB（总模型 44%）。在真实通话语料上统计词频，移除推理时不出现的 token，词表降至 65,536，embedding/head 缩至 268 MiB，总模型由 954 MiB 降至 803 MiB（15.8%）。随后在保留槽位内将高频问候/确认短语及服务固定前缀（如 "One Line Summary:"）合并为单 token，使每通电话平均 token 数由 1,324 降至 1,090（-17.7%），支持最长通话全程单次定长窗口处理。
- **真实通话校准与模块级混合精度**：对比 WikiText 与真实通话校准下的 W4A16、W8A16、MW4A16（SeqMSE 初始化），以 GPU 输出与 NPU 输出的 ROUGE-L 衡量分布保持度，真实校准在 MW4A16 下增益最大（+2.76）。通过 SeqMSE 测量各线性投影激活差异，识别出 `o_proj` 与 `down_proj` 敏感度最低，仅将其降为 INT4，其余模块维持 INT8，得到 803 MiB 的 MW4A16 模型。
- **Top-k 概率对齐选型**：使用 logit lens 比较原模型与量化模型各层 top-k 分布对齐分数。全局 W4A16 虽 ROUGE-L 尚可但高层对齐严重退化，MW4A16 在最终层对齐上追平 W8A16（1,111 MiB），据此选定为部署点，确保生成行为不被过度破坏。
- **服务统一承载**：单一领域适配模型同时配置 AI Call Summary 与 AI Call Answering 两项任务，通过结构化 Prompt 模板与任务微调（Appendix B）复用同一套参数，避免多模型叠加的内存压力。

## 实验与结果
- **数据集与设备**：100 条去标识化 Held-out 通话记录（最长 4K token），测试平台为 Samsung Galaxy S25 Ultra（Snapdragon 8 Elite）与 Apple iPhone 17 Pro（A19 Pro），分别在 CPU 与 NPU 上评测延迟、吞吐量、能耗与温度。
- **质量结果**：Cloud LLM 基准（Summary 99.60 / Answering 99.00） vs Cloud sLM FP16（94.38 / 95.70，达云侧 94.8% / 96.7%） vs ixi-sLM 量化后（94.33 / 95.40），量化仅造成 0.05–0.30 分的边际损失，下降主要集中于 task suggestion 子任务（因触发频率低、样本稀疏）。
- **端侧性能**：模型体积 803 MiB（FP16 的 28%）；NPU 延迟降至 CPU 的 22%（10s vs 45s，4.5× 加速），单次通话能耗降至 23%（1.79 vs 7.79 mAh）；300 秒持续推理表面温度峰值 34.5°C，始终低于 36°C 热节流阈值，CPU 则在 60 秒内突破阈值并 plateau 于 ~45°C。Prefill 阶段 NPU 较 CPU 节能约 17×，Decode 节能约 2×。
- **框架对比**：在 Snapdragon NPU 上，ixi-sLM 预填充吞吐较 ExecuTorch 提升 ~1.5×，较 llama.cpp 提升 >30×，模型尺寸较 ExecuTorch 小
