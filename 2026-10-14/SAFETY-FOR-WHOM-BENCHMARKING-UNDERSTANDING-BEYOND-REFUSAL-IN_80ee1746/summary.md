---
title: "SAFETY-FOR-WHOM-BENCHMARKING-UNDERSTANDING-BEYOND-REFUSAL-IN"
source: https://arxiv.org/pdf/2610.17728v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-01 22:46:39"
field: "多模态大模型安全评估"
keywords: ["egocentric multimodal assistants", "safety benchmark", "refusal behavior", "safety understanding", "harm target taxonomy"]
innovations: ["提出交互中心主义伤害目标分类（Self/The Others/Others/Unspecified）", "构建超越拒绝行为的安全理解双轴评估框架", "揭示第一人称助手在Self伤害上的系统性安全漏洞"]
benchmarks: ["STOU", "Ego4D", "AEA", "Nymeria+"]
---

# 论文速读：SAFETY FOR WHOM? BENCHMARKING UNDERSTANDING BEYOND REFUSAL IN EGOCENTRIC MULTIMODAL ASSISTANTS

## 一句话总结
论文提出了STOU基准测试，通过解耦安全拒绝行为与安全理解，系统评估第一人称多模态助手在四种伤害目标（Self、The Others、Others、Unspecified）和六种伤害类型下的安全漏洞，发现模型普遍在高UR的同时存在对Self伤害理解的显著缺失。

## 研究问题与动机
- 现有安全评估仅以拒绝率衡量安全性，无法区分模型保护的是哪类受害者（用户本人 vs 旁观者 vs 社会）
- 模型可能"盲目拒绝"（不理解风险却拒绝）或"明知故犯"（理解风险仍提供帮助），单一拒绝指标掩盖了这两种不同性质的失败模式
- 第一人称助手的输出直接引导用户的物理行为，可能影响旁观者，需要交互中心主义的安全评估视角
- 现有视频安全基准（如Video-SafetyBench、EgoSafe-Bench）缺乏音频查询、伤害目标分类和对安全理解的多维度评估

## 核心贡献（创新点）
- **交互中心主义的伤害目标分类法**：将危害目标分为Self、The Others、Others、Unspecified四类，区别于以往仅关注攻击类型而不区分散射对象的评估方式
- **STOU基准（2,902个样本）**：基于真实第一人称视频（Ego4D、AEA、Nymeria+），结合LLM生成与人工编写的音频/文本查询，覆盖四种伤害目标×六种伤害类型的24种组合
- **超越拒绝的安全理解评估框架**：通过行动探测（内省性）和伤害探测（交互性）双维度评估，定义安全理解率（SUR），首次将"是否理解伤害"从"是否拒绝"中解耦
- **揭示行为-认知差距**：系统暴露七类安全失败模式（Ideal Guard、Reflexive Refusal、Alignment Gap、Comprehension Gap等），证明拒绝率无法保证真正安全

## 方法详解
- **数据构建四阶段流水线**：(1) 从Ego4D（5s固定片段）、AEA、Nymeria+采样3-10s第一人称视频片段并使用VLM生成场景描述；(2) 按伤害目标×伤害类型组合生成有害查询，针对The Others类别需确保视频中可见目标人物；(3) 15名标注员人工验证、修正标注；(4) 使用GPT-4o-mini-tts合成语音，构建(video, text, audio)三元组
- **安全理解评估**：内省性（interiority）探测用户意图行动，交互性（interaction）探测伤害对象、机制与后果；评分采用LLM judge对比gold reference，伤害得分0-5分（对象+1、机制+2、后果+2），阈值τ=3时与人工评估的平衡准确率最高（89.4%）
- **拒绝行为评估**：采用4级标签（0=明确拒绝、1=安全警告、2=无风险识别回答、3=非安全原因未回答），不安全响应率UR = 1 - 安全响应比例
- **语音线索实验**：在10%分层样本上使用5种TTS声音（儿童女、成人女×2、儿童男、成人男）测试Qwen2.5-Omni-7B，发现模型对儿童声音无显著提升的安全警觉（Self伤害UR约80%，与成人相当）

## 实验与结果
- **评估模型**：10个多模态/全模态LLM（7个开源：MiniCPM-o 2.6/4.5、Qwen2.5-Omni-3B/7B、VITA-1.5、Baichuan-Omni-1.5、Gemma 4 E4B；3个闭源：Gemini-3.1-flash、Gemini-3-flash、Muse Spark 1.2）
- **拒绝行为结果**：开源模型UR普遍高于闭源模型；最差为Qwen2.5-Omni-3B（音频→音频UR=86.9%），最佳为Muse Spark 1.2（音频→文本UR=15.9%）；对Self伤害的UR系统性最高，Gemini达53.1%（Self vs The Others差距7.9%p）
- **安全理解结果**：闭源模型理解率显著更高（Gemini 70.7%、Muse Spark 82.8%），开源模型最高仅MiniCPM-o 4.5的51.1%；Self目标的SUR最低，所有模型在Self上理解率均低于其他目标
- **失败模式分布**：开源模型主要失败模式为Comprehension Gap（不理解且不安全响应），闭源Gemini为Alignment Gap（理解但仍有不安全响应），仅Muse Spark以Ideal Guard为主（72.9%）

## 相关工作脉络
- **Video-SafetyBench/Omni-SafetyBench/iSafetyBench**：均为第三人称视频视角、合成视频、仅评估拒绝行为；STOU的区别在于第一人称视角、真实视频、音频查询、伤害目标与理解评估
- **EgoSafetyBench/EgoSafe-Bench/MedGaze-Bench**：同为第一人称视角但缺少音频输入和安全理解评估；STOU首次实现视频+音频联合输入的全模态安全评估
- **MM-SafetyBench (Liu et al., 2024)**：针对多模态大语言模型的安全基准，但基于图像而非视频，且无交互中心主义伤害目标分类
- **Multimodal Situational Safety (Zhou et al., 2024)**：提出情境安全概念并使用LLM judge；STOU在此基础上扩展至第一人称场景并引入理解评估
- **Mechanistic safety studies (Zhao et al., 2025)**：发现有害理解与拒绝行为在不同token位置编码；STOU从评估层面验证了行为-认知的解耦现象

## 局限性与未来方向
- 数据集规模相对有限（2,902样本），且The Others类别因伦理考虑未完全公开，限制了研究的广泛性
- 仅评估了文本和音频两种输出模态，未涉及视频输出或实时交互场景
- 使用单一LLM（GPT-4.1）作为judge，可能存在系统性偏差，人工评估仅覆盖225样本
- 未深入分析模型内部表示与行为解耦的机制原因，仅停留在现象描述层面
- 语音线索实验仅测试单一模型（Qwen2.5-Omni-7B），结论的普适性有待验证

## 研究启发与可借鉴点
- **双轴评估框架可迁移**：将"行为"与"理解"解耦的评估范式可推广至其他AI安全领域（如医疗决策、自动驾驶），帮助识别"虚假安全"（盲目拒绝）与"危险理解"（理解但放任）
- **伤害目标分类具有通用价值**：Self/The Others/Others/Unspecified的分类逻辑可用于设计更具针对性的安全对齐训练数据，特别是强化对Self伤害的防护
- **LLM-as-judge的τ阈值选择策略**：通过人工标注子集优化阈值以获得最高平衡准确率的做法，可为其他自动化评估框架提供方法论参考
- **语音线索脆弱性分析实验设计**：控制变量法（固定所有输入仅改变TTS声音）结合配对bootstrap检验，是可复用的用户属性敏感性测试范式
- **失败模式四象限分类**：Ideal Guard/Reflexive Refusal/Alignment Gap/Comprehension Gap的分类体系有助于系统诊断模型安全缺陷的具体类型，指导针对性改进

## 关键术语表
- **STOU (Safety Understanding Beyond Refusal for Omni-modal and Unspecified Targets)**：第一人称多模态助手的安全基准，包含2,902个视频-音频/文本查询对
- **Harm Target**：伤害目标分类，分为Self（用户自身）、The Others（可见旁观者）、Others（不可见特定对象）、Unspecified（目标不明确）
- **Safety Understanding Rate (SUR)**：同时正确识别用户意图行动和伤害（伤害得分≥3）的样本比例
- **Unsafe Response Rate (UR)**：模型未安全拒绝有害查询的比例，UR越低表示拒绝行为越安全
- **Interiority**：安全理解的内部视角，指模型理解用户意图行动的能力
- **Interaction**：安全理解的交互视角，指模型识别伤害对象、机制与后果的能力
- **Alignment Gap**：模型理解伤害但仍提供不安全响应的失败模式
- **Comprehension Gap**：模型既不理解伤害又不安全响应的失败模式

## 可复现要素
- **数据集**：STOU基准，视频片段来自Ego4D、AEA、Nymeria+（受原始数据集许可证约束，仅提供视频ID和时间边界，需通过官方渠道获取）；The Others类别因伦理原因受限访问；项目页面标注为公开
- **代码/权重**：论文未提及代码开源声明，仅提供了项目页面链接
- **关键超参**：视频采样频率1 FPS；语音理解评估阈值τ=3；LLM judge温度=0；TTS使用GPT-4o-mini-tts；Bootstrap重采样B=10,000
