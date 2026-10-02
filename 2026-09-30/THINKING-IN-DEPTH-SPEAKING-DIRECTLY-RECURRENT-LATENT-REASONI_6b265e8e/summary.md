---
title: "THINKING-IN-DEPTH-SPEAKING-DIRECTLY-RECURRENT-LATENT-REASONI"
source: https://arxiv.org/pdf/2609.37818v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 19:59:45"
field: "语音语言模型与副语言推理"
keywords: ["speech language models", "paralinguistic reasoning", "looped transformers", "latent reasoning", "empathetic dialogue", "covert chain-of-thought"]
innovations: ["首个将 looped Transformer 引入语音语言模型实现潜式递归推理", "两阶段 CoT-to-response 课程学习分离推理学习与回复生成", "ReCue 多尺度声学记忆重锚定机制解决递归过程中的副语言信息退化"]
benchmarks: ["EchoMind", "SD-Eval", "MMSU", "MMAU-Pro"]
---

# 论文速读：THINKING-IN-DEPTH-SPEAKING-DIRECTLY-RECURRENT-LATENT-REASONING

## 一句话总结
本文提出 **LoopSLM**，首个采用 looped Transformer 架构的语音语言模型，通过两阶段训练将语音支撑的显式 CoT 监督转化为声学锚定的潜式递归推理，在推理时无需生成 CoT 即可实现高质量的副语言情感对话生成。

## 研究问题与动机
- **感知–推理差距（perception–reasoning gap）**：显式 CoT 虽然能提升模型的副语言感知能力并使声学线索在回复中更明确，但并不能保证模型在规划回复时有效利用这些线索（Table 1 对比 CoT-SFT 后理解提升但推理准确率下降）。
- **文本 CoT 增加推理延迟**：推理时需先生成推理 token 才能生成回复，影响实时对话体验。
- **现有潜式推理方法对齐不足**：如 Coconut、CODI 等通过逐步替换 CoT 或表示蒸馏学习潜式步骤，但与原始 CoT 的对齐较弱，信用分配受限；CoLaR 虽通过稠密 CoT 目标改进但仍需自回归预测潜式步骤。
- **副语言信息在解码过程中易丢失**：音频编码器强编码的副语言信息在通过 decoder 时会退化，尤其在相同词汇不同语调的情况下，中层后的表示区分度大幅下降。

## 核心贡献（创新点）
1. **首个将 looped Transformer 引入 SLM 的工作**：将潜式回复规划从序列轴转移到递归 decoder 深度，推理时不产生自回归推理 token，本质区别于需生成文本 CoT 的方法。
2. **ReCue（声学线索重锚定）机制**：每个递归 pass 通过跨注意力从冻结音频编码器的多深度激活中重建紧凑声学记忆，确保递归过程中持续获得原始声学证据，而非仅依赖前一层状态。
3. **两阶段 CoT-to-response 课程学习**：Stage 1 用语音支撑 CoT 强化副语言推理并更新 Loop 和 ReCue；Stage 2 冻结 Loop/ReCue 仅训练 Coda 从精炼状态直接映射到回复，从本质上分离"学会推理"和"学会回复"。
4. **控制实验证明增益来自学到的递归变换而非深度本身**：消融表明仅增加通用深度（用原始块替换）效果有限，而训练深度提供渐进式 C4 提升；R=4 为最优，R=6 因格式遵循度下降导致推理准确率降低。

## 方法详解

### 架构分区
基于 Qwen2.5-Omni-7B Thinker 的 28 层 decoder，划分为四个区域：
- **Prelude（L1–8）**：冻结，输出 $H_0$
- **Loop（L9–11，$\theta_L$）**：权重共享的递归块，循环 $R=4$ 次
- **Transition（L12–14）**：冻结
- **Coda（L15–28，$\theta_C$）**：直接回复读取头

### ReCue：每 pass 声学重锚定
从冻结音频编码器的多深度激活构建声学记忆：
$$M_s = \text{CA}(Q, \Psi_s(A_s)), \quad s \in \{8, 16, 24, 32\}$$
每个 pass $r$ 混合多尺度并注入状态：
$$\widehat{M}_r = \sum_s \alpha_{r,s} M_s, \quad \widetilde{H}_{r-1} = H_{r-1} + g_r \text{CA}(H_{r-1} + e_r, \widehat{M}_r)$$
其中 $\alpha_{r,s}$ 为条件于状态池化和迭代嵌入 $e_r$ 的可学习混合权重，$g_r$ 为零初始化门控。ReCue 还通过第二 head 产生 pre-norm 调制 $\Gamma_r$，初始化时保持主干不变。

### Weight-shared looped Transformer
$$H_r = F_{9:11}(\widetilde{H}_{r-1}; \theta_L, \Gamma_r), \quad r = 1, \ldots, R$$
每个 pass 共享 $\theta_L$，维持独立的 causal attention history，递归增加深度而不破坏自回归因果性。推理时 4 次 pass 在 prompt 处理和每个 decoding step 运行，执行深度从 28 增至 37 次层调用，但不复制 Transformer 权重。

### 两阶段训练
**Stage 1（CoT 监督递归精炼）**：
- 损失函数：$\mathcal{L}_1 = -\frac{1}{|u_1|}\sum_t \log p_\Theta(u_{1,t} | a, x, u_{1,<t})$，其中 $u_1 = z$（CoT 轨迹）
- 更新参数：$\Omega_1 = \{\theta_L, \phi\}$，Coda 冻结
- 目标：将 CoT 监督引入递归路径，学习如何将声学证据转化为回复计划

**Stage 2（直接回复读取）**：
- 损失函数相同形式，$u_2 = y$（回复），$\Omega_2 = \{\theta_C\}$
- Loop 和 ReCue 冻结，仅训练 Coda 从精炼状态直接映射到回复
- 输出长度缩短至 Stage 1 的约 1/5.7

## 实验与结果

### 数据集与基线
- **主评测集**：EchoMind（副语言理解、推理、共情回复）
- **通用音频评测**：SD-Eval、MMSU、MMAU-Pro
- **训练数据**：380,540 轮对话（431.9 小时），71% 合成音频，62.2/37.8 中英比例，基于 LIME-440K 重新标注
- **基线模型**：Qwen2.5-Omni-7B、Kimi-Audio、Audio-Flamingo-3、OSUM-EChat、MiniCPM-o 4.5、MiMo-Audio-Think、Qwen3-Omni-Thinking（30B MoE）

### 主要结果（Table 1）
- **相对 Qwen2.5-Omni-7B 基准**：LoopSLM-R4 平均副语言理解准确率提升 **+9.8 点**，推理准确率提升 **+6.6 点**，四维度回复质量全面提升
- **相对 CoT-SFT 基线**：推理准确率提升 **>20 点**，生成 token 减少 **64.5%**，延迟降低 **50%**（一半）
- **相对 Qwen3-Omni-Thinking（30B MoE）**：多数共情回复指标更优，延迟降低 **34×**
- **参数规模 ≤9B 模型中**：LoopSLM 在大多数指标上达到最佳
- **泛化能力**：仅训练于对话数据，但在 MMSU（+1.28）和 MMAU-Pro（+3.16）上仍有提升

### 消融结果（Table 2）
- **移除 Stage 2**：输出仍长，回复质量差
- **Stage 1 用回复目标替代 CoT**：推理准确率显著下降，证实增益来自 CoT 监督而非简洁回复格式学习
- **联合训练 CoT+回复**：性能低于两阶段
- **Stage 2 保持 Loop 可训练**：推理准确率略增但理解和其他回复分数下降
- **移除 ReCue 记忆**：理解准确率显著下降
- **递归深度 R**：R=4 最优；R=1/2/3/6 均有不同权衡，R=6 因格式遵循度差导致推理准确率降至 47.5/23.4

## 相关工作脉络
1. **Coconut / CODI**：通过逐步 CoT 替换或自蒸馏学习潜式推理步骤，但与原始 CoT 对齐较弱，信用分配受限；LoopSLM 通过稠密语音支撑 CoT 监督 + ReCue 声学锚定实现更紧密的对齐。
2. **CoLaR**：通过稠密 CoT 目标改进潜式推理，但仍需自回归预测压缩潜式步骤；LoopSLM 完全消除推理时的自回归 token 生成。
3. **RecurTrace**：自适应潜式推理 with loop-time memory；LoopSLM 进一步引入多尺度声学记忆重锚定机制。
4. **ParaBridge**：bridging paralinguistic perception and dialogue behavior；本文聚焦于 CoT-to-latent-reasoning 的转化机制。
5. **Looped Transformers（Saunshi et al., Geiping et al.）**：理论分析与通用任务验证；本文首次将其引入 SLM 并适配副语言推理场景。
6. **Qwen3-Omni-Thinking**：原生 thinking 模式；本文证明通过递归深度而非序列扩展可实现同等甚至更优的推理能力。

## 局限性与未来方向
- **训练数据限制**：合成音频占 71%，可能影响真实场景泛化；仅训练于对话数据，虽对通用音频有正向迁移但仍有提升空间。
- **递归深度非单调**：R=6 导致格式遵循度严重下降（推理准确率从 63.9 降至 47.5），说明有效递归深度需精心调优。
- **推理计算增加**：虽无额外 token，但层调用从 28 增至 37 次，仍需一定计算开销。
- **未来方向**：探索自适应递归深度、跨语言泛化、多轮对话中的声学记忆维护、以及更低延迟的实时部署优化。

## 研究启发与可借鉴点
1. **两阶段课程学习范式可迁移**：将"学习推理"与"学习输出"解耦的思路可推广至其他需要复杂中间推理的任务（如代码生成、数学推理），避免联合训练导致的表征混淆。
2. **ReCue 的多尺度声学记忆机制**：跨层 audio encoder 激活的记忆 bank 设计可用于其他多模态推理任务，解决"模态信息在深层网络中退化"的共性问题。
3. **Looped Transformer 在 SLM 中的适配经验**：层选择（L9-11 vs L15-17）揭示早期-中层适合推理精炼、深层适合表达的风格分工，为后续 SLM 架构设计提供参考。
4. **CoT-to-Latent 转化的评估指标**：本文通过 "移除 ReCue"、"保持 Loop 可训练" 等精细消融分离各组件贡献，该评估范式可复用于其他潜式推理工作。
5. **延迟-性能权衡分析框架**：Token 数量、层调用次数、中位延迟的多维评估有助于后续工作更精准地定位优化空间。

## 关键术语表
- **Paralinguistic reasoning（副语言推理）**：根据"怎么说"（语调、节奏、音质）而非仅"说什么"推断说话者状态与沟通需求的能力。
- **Perception–reasoning gap（感知–推理差距）**：模型能识别副语言线索但并不能在回复规划中有效利用的现象。
- **Looped Transformer（循环 Transformer）**：复用少量 decoder 层多次处理同一输入以增加递归深度，而非扩展序列长度的架构。
- **ReCue（Reanchoring to acoustic Cues）**：每递归 pass 通过跨注意力从冻结音频编码器的多深度激活中重建声学记忆的机制。
- **Coda**：LoopSLM 中位于 Transition 之后、负责直接从精炼状态生成回复的最终 decoder 模块（L15-28）。
- **CoT-to-response curriculum**：两阶段训练策略，第一阶段用 CoT 训练推理，第二阶段冻结推理部分仅训练直接回复映射。
- **EchoMind**：评估 SLM 副语言理解、推理与共情回复生成的多层级基准，含合成/人类语音与多裁判评分。
- **Latent reasoning（潜式推理）**：在隐式递归深度中进行推理 computation，而非生成显式文本 CoT token 的推理方式。

## 可复现要素
- **数据集**：基于 LIME-440K 重新标注，含 Gemini-3.5-Flash 生成的 CoT 轨迹；论文未声明开源
- **代码**：论文未提及开源计划
- **权重**：基于 Qwen2.5-Omni-7B 微调，论文未声明开源
- **关键超参**：$R=4$（递归 pass 数），Loop 层 L9-11，ReCue 查询数 16，训练：Stage 1 全数据 1 epoch，Stage 2 半 epoch，8×H800 GPU
