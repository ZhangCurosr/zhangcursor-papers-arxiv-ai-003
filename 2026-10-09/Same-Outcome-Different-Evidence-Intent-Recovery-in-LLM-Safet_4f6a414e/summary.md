---
title: "Same-Outcome-Different-Evidence-Intent-Recovery-in-LLM-Safet"
source: https://arxiv.org/pdf/2610.11766v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-01 14:25:31"
field: "大语言模型安全与对齐评估"
keywords: ["LLM 安全评估", "意图恢复", "ASR", "UR", "越狱攻击", "Toki Pona", "意图隐藏"]
innovations: ["提出 IRIS 框架，将操作性理解率（UR）作为 ASR 的互补归因信号配对报告", "控制性重建实验揭示恢复率与有害输出率在意图隐藏接口下可独立变化", "开源一套人工审核的跨界面意图恢复与安全评估流水线"]
benchmarks: ["HarmBench", "StrongREJECT", "JailbreakBench"]
---

# 论文速读：Same-Outcome-Different-Evidence-Intent-Recovery-in-LLM-Safet

## 一句话总结
论文提出 IRIS（Intent Recovery as an Intermediate Signal）评估框架，在攻击成功率（ASR）之外引入操作性理解率（UR），用于区分"模型成功恢复恶意意图并拒绝"与"模型未恢复目标任务"两类不同的非有害输出；在多种意图隐藏接口下揭示了 ASR 相同但 UR 差异显著的现象。

## 研究问题与动机
1. **现有 ASR 指标的归因模糊性**：在低资源语言、密码、ASCII 艺术等意图隐藏型攻击中，一个非有害输出可能源于三种不同状态——成功恢复恶意任务并拒绝、未能恢复任务、或回应了完全不同的内容，仅靠 ASR 无法区分。
2. **"可恢复性假设"未被显式检验**：HarmBench、JailbreakBench、StrongREJECT 等主流基准系统性地测量有害输出和拒绝，但隐含假设了模型已恢复目标任务，该假设在意图模糊接口下可能不成立。
3. **防御方已分离"恢复"与"安全响应"**：基于回译和显式意图分析的防御方法（如 Wang et al., 2024; Zhang et al., 2025）已将意图恢复与后续安全响应分离，但评估体系尚未同步跟进。
4. **不同界面下相同 ASR 的可比性存疑**：控制性对比显示，GPT-4o 在 Toki Pona 和 CaesarCipher 下 ASR 同为 2.9%，但 UR 分别为 15.2% 和 96.2%，意味着 ASR 相同的结果具有截然不同的证据意义。

## 核心贡献（创新点）
1. **测量贡献：识别并形式化"可恢复性假设"，提出操作性 UR 作为归因中间信号**——不同于单纯的安全评分，UR 衡量模型是否既识别了评估任务又将其作为响应框架，填补了 ASR 无法归因的空白。
2. **经验发现：恢复率与有害输出率在意图隐藏接口下可独立变化**——控制性英文重建证明恢复率随显式程度提升而一致改善，但 ASR 并不遵循相同模式；FormalLogic 下高 UR 仍可与高 ASR 共存。
3. **资源贡献：开源了一套人工审核的、可审计的评估资源与可复现流水线**——覆盖匹配行为、恢复标注、重建控制和跨界面安全测量，代码与实验输入均已公开。

## 方法详解
1. **IRIS 框架**：将 ASR 与 UR 配对报告。公式定义为：UR = P(operational task recovered)，即 P(IR=2 ∧ IS=NA)；ASR = P(harmful assistance)，即 P(SO≥2)。两者分母均为全部 105 个行为项（含被拦截/空输出）。
2. **三级标注体系 IR/IS/SO**：IR（识别）=2 表示核心恶意任务/目标/对象基本对齐；IS（响应框架）区分 NA（目标任务保持为响应框架）、M（翻译/语言分析）、R（其他非目标重构）、B（被拦截/空）；SO（有害程度）=0/1/2/3 分别对应拒绝/安全替代/部分有害/实质有害。
3. **操作性 UR 的精确定义**：仅识别任务（IR=2）不够，还必须满足 IS=NA，即模型将目标任务作为实际响应的框架，而非仅仅分析语言或进行翻译。
4. **控制性英文重建（T1→T3）**：以 Toki Pona 提示为语义核心，构造三个显式程度递增的英文版本：T1 最小化修复（保留歧义）、T2 显式化语义角色、T3 将语用关系解析为自然请求；三者严格约束于观察到的语义核心，不引入额外攻击内容。
5. **评测设置**：6 个大模型（GPT-4o、Claude Sonnet/Opus 4.5、Gemini 3 Flash、DeepSeek V3.2、Qwen3-Max），7 种界面（EN、Toki Pona、Zulu、Yoruba、CaesarCipher、FormalLogic、ArtPrompt），每种界面下均使用相同 105 个行为，temperature=0，禁用显式推理。
6. **统计方法**：95% 百分位 bootstrap 区间（10,000 次重抽样）；T1–T3 配对恢复差异采用两侧精确 McNemar 检验 + Holm 校正。

## 实验与结果
- **数据集**：105 个有害行为，源自 HarmBench、StrongREJECT、JailbreakBench 的手动去重与跨界面适配。
- **评测基线**：英文直请（EN）为显式基线；Toki Pona 为主要语义压缩压力测试；另有 Zulu、Yoruba、CaesarCipher、FormalLogic、ArtPrompt 作为补充界面。
- **关键数字**：
  - EN 基线下，所有模型 UR ≥ 97.1%，ASR < 3%。
  - Toki Pona 下 UR 大幅下降：GPT-4o 仅为 15.2%（ASR=2.9%），而 CaesarCipher 下 GPT-4o UR=96.2%（ASR=2.9%）。
  - T1→T3 重建中，所有模型的 UR 显著提升：GPT-4o 从 66.7%→99.0%（+32.4pp），Sonnet 4.5 从 54.3%→90.5%（+36.2pp），Opus 4.5 从 52.4%→93.3%（+41.0pp），均经 Holm 校正后 p < 0.001。
  - FormalLogic 界面下，多个模型实现高 UR（GPT-4o: 94.3%、Gemini: 88.6%）的同时仍伴随高 ASR（GPT-4o: 52.4%、Gemini: 49.5%）。
- **最强结果**：Qwen3-Max 在 ArtPrompt 下获得最高 UR（86.7%）且 ASR 仅 7.6%；Gemini 3 Flash 在 FormalLogic 下 UR=88.6% 但 ASR=49.5%，是最典型的"高恢复-高危害"案例。

## 相关工作脉络
1. **HarmBench / JailbreakBench / StrongREJECT**：系统测量有害输出与拒绝，但依赖 ASR 单一指标，未显式检验可恢复性假设；本文在此基础上引入 UR 作为归因信号。
2. **ArtPrompt (Jiang et al., 2024)**：证明 ASCII 艺术提示可利用模型识别困难进行越狱；本文将其纳入跨界面对比，展示同类 ASR 下不同 UR 的模式。
3. **Low-Resource Languages Jailbreak (Yong et al., 2023)**：报告低资源语言攻击中存在大量"不确定响应"；本文用 UR 量化此类"未恢复"状态的比例。
4. **Logic Jailbreak (Peng et al., 2026)**：通过形式化逻辑表达式解锁安全限制；本文 FormalLogic 实验中揭示高 UR 与高 ASR 可并存，说明恢复充分不等于安全。
5. **Backtranslation 防御 (Wang et al., 2024)** 与**意图分析防御 (Zhang et al., 2025)**：防御方已将意图恢复与安全响应分离；本文主张评估体系应同步做出相同区分。
6. **JADES (Chu et al., 2025)**：分解有害协助的响应侧分析；本文强调回复侧分解之外还需要"任务是否被恢复"的前置归因层。

## 局限性与未来方向
1. **105 项面板的泛化受限**：为跨界面可比性进行了手动筛选，排除了依赖专业术语或不稳定有害边界的行为，不能代表真实世界中所有攻击的流行度。
2. **恢复与请求变化的因果隔离不足**：T1→T3 控制了语义核心，但恢复提升与 ASR 变化之间的因果贡献尚需进一步实验隔离。
3. **UR 标注的读者依赖性**：操作性 UR 是响应基的、读者依赖的；独立双标注显示的 κ 仅为 0.29–0.63，读者间差异显著，需独立人工判读校准。
4. **接口特异性评分规则**：英文通用拒绝规则与 TOK/LRL 任务证据规则存在系统性差异，跨界面 UR 差异部分源于评分假设而非纯行为差异。
5. **Provider 漂移**：不同模型调用时间略有差异，已存档的生成记录可复现但无法完全排除版本漂移。

## 研究启发与可借鉴点
1. **UR 可作为 ASR 的归因前置信号嵌入现有评测流程**：在 HarmBench/JailbreakBench 等框架中增加 IR/IS/SO 三级标注，可在不改变主指标的前提下提升对非有害输出的解释力。
2. **控制性重建（T1→T3 范式）可用于分离"恢复能力"与"安全行为"**：通过逐步提升显式程度追踪同一语义核心的恢复曲线，是一种可迁移的消融实验设计。
3. **跨界面对比揭示了"相同 ASR 不同证据意义"的普遍风险**：在报告任何基于意图隐藏接口的安全评测结果时，应同时报告 UR，否则低 ASR 可能被误读为安全能力强。
4. **FormalLogic 下的高 UR+高 ASR 模式值得深入挖掘**：表明模型"理解恶意意图并主动执行"与"理解恶意意图并拒绝"是两个独立维度，可引导设计针对"理解但不执行"场景的防御策略。
5. **Judge 敏感性审计（v2.2 vs v2.5）的方法论价值**：对同一批输出在不同 judge/prompt 版本下的保留率分析，可作为评估 pipeline 鲁棒性的标准化工具。

## 关键术语表
**ASR（Attack Success Rate）**：攻击成功率，衡量模型对恶意请求提供有害协助的比例。
**UR（Operative Understanding Rate）**：操作性理解率，衡量模型既识别目标恶意任务又将其作为实际响应框架的比例。
**IRIS（Intent Recovery as an Intermediate Signal）**：本文提出的评估协议，将 UR 作为 ASR 的互补归因信号配对报告。
**IR（Intent Recognition）**：三级标注之一，衡量模型对核心恶意任务/目标/对象的识别程度（0/1/2）。
**IS（Intent Scope / Response Framing）**：三级标注之一，区分模型将目标任务作为响应框架（NA）、翻译/语言分析（M）、或其他重构（R）、被拦截（B）。
**SO（Safety Outcome）**：三级标注之一，衡量有害协助程度（0=拒绝/拦截，1=安全替代，2=部分有害，3=实质有害）。
**Recoverability Assumption（可恢复性假设）**：ASR 解读中的隐式前提——模型已正确恢复并 engagement 目标任务，本文论证该假设在意图隐藏接口下不总成立。
**Shift**：P(IS∈{M, R})，衡量模型将响应框架偏移到翻译/语言分析或其他非目标框架的比例。

## 可复现要素
- **数据集**：105 个有害行为源自 HarmBench、StrongREJECT、JailbreakBench 的手动去重与适配；Toki Pona 模板由有经验的译者构建；代码与实验输入已公开于公共仓库。
- **代码/权重**：代码已开源，包含 run.py（生成与评判）、analyze.py（ headline 率与配对检验）；多语言输入、frozen prompts、judge templates 均已发布。
- **关键超参**：temperature=0，显式推理禁用（where applicable），judge 输出预算 260 tokens，judge 模型为 mistral-large-2512 via OpenRouter，bootstrap 重抽样 10,000 次，McNemar 检验经 Holm 校正。
- **存档信息**：每次调用的 completion/metadata 对已存档，包含时间戳和请求设置，可供复现。
