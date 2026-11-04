---
title: "SKILL-DECOUPLED-ON-POLICY-EXPERT-DISTILLA-TION-FOR-UNIFIED-V"
source: https://arxiv.org/pdf/2610.22217v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-04 03:57:52"
field: "多模态大模型后训练"
keywords: ["Vision-Language Models", "Reinforcement Learning", "On-Policy Distillation", "Skill Extraction", "Multi-Capability Integration", "Gradient Conflict"]
innovations: ["Skill-Decoupled双阶段框架解耦能力获取与整合，规避混合RL的梯度冲突", "Reward-guided reflection从成功/失败轨迹提取显式可复用技能", "结构信息不对称蒸馏（技能仅对专家可见）使学生在推理时无需显式技能"]
benchmarks: ["CountBenchQA", "PixMo-Count", "CC-OCR V2", "ChartQA", "VisualPuzzles", "We-Math", "MUIRBench", "MMSI-Bench"]
---

# 论文速读：SKILL-DECOUPLED-ON-POLICY-EXPERT-DISTILLA-TION-FOR-UNIFIED-V

## 一句话总结
SCOPED通过独立RL训练能力特定专家、从轨迹中提取显式可复用技能，并在on-policy蒸馏中以结构性信息不对称的方式引导专家监督，实现了对统一VLM的多能力高效整合，在8个多模态基准上达到最优平均表现（SCOPED-8B均值73.13，超越Qwen3-VL-32B）。

## 研究问题与动机
1. **跨能力梯度冲突**：联合优化异构视觉能力的共享策略时，不同能力目标产生竞争梯度信号，导致"see-saw效应"（优化某能力会损害另一能力）。
2. **直接专家对齐的局限**：现有MOPD等方法直接匹配专家预测分布，无法区分可复用策略与过度拟合/奖励作弊（reward-hacking）行为，可能将噪声传给统一学生。
3. **混合能力优化损失上限**：独立RL训练的能力专家在各自目标上表现更强，证明分开获取能力、再整合的路径优于联合优化。
4. **隐式轨迹知识的利用不足**：已有技能提取工作聚焦单专家场景，多专家场景下如何将显式技能用于正则化集成过程仍待探索。

## 核心贡献（创新点）
1. **提出SCOPED解耦式框架**：先独立RL训练能力专家获取专业化行为，再通过skill-augmented on-policy distillation整合能力，与Joint RL/Mix-RL形成本质区别。
2. **Reward-guided reflection技能提取**：将专家轨迹按奖励阈值划分为成功/失败两组，通过反思生成可复用的能力特定技能 $S_c$，区别于MOPD仅对齐预测分布的做法。
3. **结构信息不对称作教-学边界**：技能 $S_c$ 仅作为专家条件输入（privileged information），学生仅接收原始多模态输入，促使学生在推理时无需显式技能即可内化程序性知识。
4. **系统的梯度冲突实证分析**：从训练奖励、专家导向更新对齐、成对梯度余弦相似度三维度量化展示Mix-RL的跨能力干扰机制，为解耦设计提供动机支撑。

## 方法详解
**阶段一：能力特定专家独立RL训练**
- 将训练语料划分为 $C$ 个能力子集 $\{B_c\}_{c=1}^C$，每个专家 $\pi_{\theta_c}$ 从共享基础策略初始化，在 $B_c$ 上独立优化：
  $\theta_c = \arg\max_\theta \mathbb{E}_{(x,a)\sim B_c, \tau\sim\pi_\theta(\cdot|x)}[r_c(a,\tau)]$
- 从每个专家收集 $K$ 条rollout轨迹，构建全局轨迹库 $\mathbb{T}_c$。

**阶段二：技能提取与蒸馏**
- **技能提取**：按奖励阈值 $\gamma$ 划分成功/失败轨迹：
  $\mathbb{T}_c^+ = \{(x,a,\tau) \mid r_c(a,\tau)\geq\gamma\}, \quad \mathbb{T}_c^- = \{(x,a,\tau) \mid r_c(a,\tau)<\gamma\}$
- 专家对两组轨迹进行reflection，生成反思结果 $\mathcal{R}_c$，再合成显式技能 $S_c = \pi_{\theta_c}(\mathrm{Instruc_{syn}}(\mathcal{R}_c))$。
- **Skill-Augmented Distillation**：对学生在步骤 $t$ 生成前缀 $y_{<t}$ 后，专家分布条件化为 $q_{\theta_c,t} = \pi_{\theta_c}(\cdot|x, S_c, y_{<t})$（含技能），学生分布为 $p_{\theta,t} = \pi_\theta(\cdot|x, y_{<t})$（不含技能）。
- 损失函数为token级reverse KL散度的跨能力均值：
  $\mathcal{L} = \frac{1}{C}\sum_{c=1}^C \mathbb{E}_{(x,a)\sim B_c, y\sim\pi_\theta(\cdot|x)}\left[\frac{1}{|y|}\sum_{t=1}^{|y|} D_{\mathrm{KL}}(p_{\theta,t} \| q_{\theta_c,t})\right]$

**奖励设计**：文本输出用token级F1；视觉定位用格式惩罚+IoU阈值的F1组合；符号推理/多图推理用format-gated accuracy（无效格式得0分）。

## 实验与结果
**数据集**：8个多模态基准，4类能力——精细感知（CountBenchQA、PixMo-Count）、OCR与图表（CC-OCR V2、ChartQA）、符号视觉推理（VisualPuzzles、We-Math）、多图推理（MUIRBench、MMSI-Bench）。训练数据共37,999样本（4类各约9,500）。

**基线**：闭源（Claude Sonnet/Opus 4.6、GPT-5.2/5.4）、开源（GLM-4.1V-9B、Qwen2.5-VL-7B、InternVL3.5-8B、Qwen3-VL-4B/8B/32B、MiMo-VL-7B-RL）。

**主要结果（Table 1）**：
- **SCOPED-8B均值73.13**，超越所有对比开源/闭源模型，较骨干Qwen3-VL-8B（67.30）提升 **+5.83点**。
- 全面超越Qwen3-VL-32B（均值72.01）约+1.12点，证明能力整合效率可替代模型规模扩展。
- SCOPED-4B均值69.96，较骨干+3.90点。
- 消融（Table 2）：SCOPED较Mix-RL均分提升3.82点；Skill Augmentation带来额外+0.74点（72.39→73.13）；Parameter Merging效果差于distillation。

**关键分析结论**：
- Mix-RL训练奖励持续低于独立RL（图2a）；共享更新与专家导向呈弱/负对齐（图2b）；跨能力梯度余弦相似度多次为负（图2c），证实梯度冲突。
- Skill augmentation提升轨迹一致性（图3b）与任务相关视觉区域注意力命中率（图3c），并增强学生保留专家诱导行为的能力（图3a）。

## 相关工作脉络
1. **MOPD (Ma et al., 2026)**：多教师on-policy蒸馏，直接对齐专家预测分布；SCOPED进一步提取显式技能作为结构化监督锚，避免传播reward-hacking噪声。
2. **Lang2Act (Xiong et al., 2026)** / **Skill-SD (Wang et al., 2026)**：从单专家轨迹提取语言工具/技能；SCOPED将其扩展至多专家解耦场景，并用技能条件化蒸馏而非仅用于自蒸馏。
3. **Trace2Skill (Ni et al., 2026)** / **SkillRL/Skill1**：构建可检索技能库；SCOPED采用单次reward-guided reflection合成能力级技能，聚焦蒸馏阶段的正则化作用而非在线检索。
4. **Mix-RL / Co-RL**：共享策略联合多能力优化；SCOPED通过独立训练+蒸馏解耦，规避梯度冲突的see-saw效应。
5. **Vision-OPD (Yuan et al., 2026)** / **On-policy context distillation**：context-conditioned变体提供辅助信息；SCOPED的"技能仅对教师可见"构成更严格的structural信息不对称。
6. **Gradient surgery (Yu et al., 2020; Liu et al., 2021)**：通过梯度正交化缓解多任务冲突；SCOPED从架构层面绕过冲突（先解耦后蒸馏），而非在联合优化中修复。

## 局限性与未来方向
1. **技能提取依赖单专家反思**：每能力仅生成一个技能 $S_c$，可能无法覆盖该能力的全部子策略变体。
2. **reward函数设计手工性强**：不同能力需定制奖励（如定位任务的IoU加权F1、格式门控accuracy），泛化到新能力时需重新设计。
3. **蒸馏阶段仍需多专家同时存在**：推理时无技能，但训练阶段需维护 $C$ 个专家及其轨迹库，显存开销较大。
4. **未探索技能的动态更新**：技能为静态提取，未考虑在蒸馏过程中根据学生表现迭代改进。
5. **仅覆盖4类视觉能力**：未涉及开放域对话、视觉生成等更广泛 multimodal 任务。

## 研究启发与可借鉴点
1. **结构性信息不对称设计**：技能仅对教师可见的学生-教师设置，可作为" privileged context distillation "的一般范式，适用于LLM Agent、多模态推理等多种场景。
2. **Reward-guided reflection的通用性**：成功/失败轨迹分组+反思合成技能流程，可迁移到纯文本RLVR（如DeepSeek-R1风格）或多Agent场景，作为离线技能构建模块。
3. **梯度冲突量化三板斧**：训练奖励曲线+专家导向更新对齐+成对梯度余弦相似度，这套诊断方案可复用于任何多任务/多能力联合训练场景的快速体检。
4. **Skill-expert matching的必要性**：消融证明匹配技能显著优于随机技能（附录A.9），提示跨能力蒸馏必须保持"一对一"结构，不可滥用任意教师的输出。
5. **与SFT/Parameter Merging对比的严谨基线**：论文同时报告SFT、Mix-RL、Parameter Merging、w/o Skill三种变体，这种多层对照可作为后续工作的标准消融模板。

## 关键术语表
**On-Policy Distillation (OPD)**：在学生学习轨迹上对齐教师预测分布的蒸馏范式，避免off-policy分布偏移。
**Skill-Decoupling**：将能力获取（独立RL）与能力整合（蒸馏）两阶段分离，避免联合优化中的梯度冲突。
**Reward-guided Reflection**：对成功/失败轨迹分组后，由专家模型反思提取可复用验证与纠错原则的过程。
**Structural Information Asymmetry**：技能仅作为专家条件输入而不暴露给学生的设计，迫使学生在无显式技能下内化知识。
**Reverse KL Divergence**：蒸馏损失形式 $D_{\mathrm{KL}}(p_\theta \| q_{\theta_c})$，鼓励学生对齐教师分布而非反向。
**Expert-guided Update Alignment**：用余弦相似度衡量实际参数位移与"趋近专家分布的负梯度方向"的一致性。
**Format-gated Accuracy**：仅当输出满足指定格式（如$\lt answer\gt$标签）时才计算答案准确率，否则奖励为0。
**See-saw Effect**：混合多能力RL中提升某一能力必然损害另一能力的梯度竞争现象。

## 可复现要素
- **数据集**：评估集为公开benchmark（CountBenchQA、PixMo-Count、CC-OCR V2、ChartQA、VisualPuzzles、We-Math、MUIRBench、MMSI-Bench）；RL训练数据来自Orsta-Data-47K、PixMo-Count、SlideVQA、We-Math 2.0 Pro、Mantis-Instruct、M4-Instruct共37,999样本（预处理后）。
- **代码/权重**：代码与数据已开源 → https://github.com/NEUIR/SCOPED。
- **关键超参**：训练epoch=3；Expert RL rollout batch=128，responses/prompt=8，max prompt/response length=8192，lr=1e-6，KL coeff=1e-2；蒸馏rollout batch=256，responses/prompt=1，max prompt length=12288，max response length=8096，lr=1e-6，distillation loss=reverse KL；图片像素min=262144，max=524288；precision=bfloat16；EasyR1 + veRL框架，NVIDIA A800 80GB。
