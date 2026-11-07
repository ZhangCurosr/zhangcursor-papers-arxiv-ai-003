---
title: "UPPAAi-Towards-LLM-Assisted-Modeling-Verification-Analysis-a"
source: https://arxiv.org/pdf/2610.40042v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-07 07:58:38"
field: "AI辅助形式化验证与时系统建模"
keywords: ["模型检查", "定时自动机", "大型语言模型", "UPPAAL", "AI辅助形式化方法", "循环工程", "统计模型检查", "策略合成"]
innovations: ["提出UPPAAL Tools API，系统性暴露模型编辑/查询管理/验证引擎供LLM tool calling", "确立'LLM提议+UPPAAL验证'闭环机制并实现外部MCP与内嵌聊天双工作流", "七项跨symbolic/SMC/STRATEGO范式的case study验证工具普适性"]
benchmarks: ["Fischer互斥协议(经典demo)", "LEGO传送带分拣(对称约简)", "TSN IEEE 802.1Qbv网关调度(鲁棒性评估)", "雨水蓄水池STRATEGO控制器学习", "COVID-19 CTMC/ODE多范式一致性"]
---

# 论文速读：UPPAAi-Towards-LLM-Assisted-Modeling-Verification-Analysis-a

## 一句话总结
本文提出了 UPPAAI，一个将大语言模型（LLM）与 UPPAAL 时系统验证工具集成的框架，通过 "LLM 提议/解释/编排 + UPPAAL 验证" 的闭环机制，辅助用户完成实时系统的建模、属性指定、调试与分析。该工具已暴露 Tools API，支持外部 Agentic Client（MCP 协议）和嵌入聊天两种工作流，并在七项案例研究中验证了其有效性。

## 研究问题与动机
- **建模门槛高**：UPPAAL 等传统形式化方法工具需要专业领域知识，从非正式需求到可验证模型的整个流程涉及模板构造、时钟设计、同步机制选择、查询编写与反例解读等多个环节，对新手极不友好。
- **LLM 的正确性缺陷**：现有 LLM 在代码生成方面表现突出，但其"看似合理但不一定正确"的本质使其无法独立承担安全关键系统的验证任务；而人工验证又缺乏可扩展性。
- **反馈闭环缺失**：当前 LLM-assisted formal methods 多停留在"生成-自修正"阶段，缺少外部可信执行器（如模型检验器）作为 critic 驱动迭代修复的闭环机制。
- **跨范式统一分析困难**：UPPAAL 本身支持符号模型检查、统计模型检查（SMC）、策略合成（STRATEGO）等多种分析模式，但如何以统一接口暴露给 LLM 仍无定解。

## 核心贡献（创新点）
1. **UPPAAL Tools API**：首次系统性地将 UPPAAL 的模型编辑、查询管理与验证引擎暴露为 LLM 可调用的函数集合（包括 getContext、verifyQueries、insertQuery 等），使 LLM 能够动态构造与审查模型。
   - *本质区别*：不同于仅访问文档级接口的通用 RAG 方案，本 API 直接绑定 UPPAAL 内部数据结构，支持细粒度语法检查、增量修改与多引擎联合调用。

2. **"LLM proposes / UPPAAL verifies" 闭环机制**：确立用户保留需求控制权、LLM 负责提议与解释、形式化分析器充当最终裁判的设计哲学。
   - *本质区别*：与纯文本生成的 prompt engineering 方案不同，该方法以 executable semantics 作为 truth source，所有 claims 必须有可执行分析支撑。

3. **双工作流架构（MCP vs langchain4j）**：提供外部 Agentic Client 接入（Workflow A）与 UPPAAL 内嵌聊天（Workflow B）两套集成路径，并分析其上下文管理、权限隔离与交互流畅度的 trade-off。
   - *本质区别*：前者复用 Claude Code 等已有 AI 编程助手生态，后者由 UPPAAL 主导会话历史与 GUI 状态感知，避免外部 client 的 context-polling 开销。

4. **七项多样化案例研究**：覆盖时序自动机、随机混合自动机、连续时间马尔可夫链（CTMC）、常微分方程（ODE）、Timed Games（STRATEGO 策略学习）等 UPPAAL 全部主要子系统，证明工具在不同正式化程度与用户经验水平下的普适性。
   - *本质区别*：多数 prior work 仅验证单一范式或单一场景，本文实证展示了同一 LLM+checker loop 可无缝切换 symbolic / stochastic / strategic analysis 三种模式。

## 方法详解
**Tools API 设计**：
UPPAAL 暴露一组 Java 函数作为 LLM tool calling 接口，按功能分为五类：
- **模型与模板操作**：`getContext`（获取 GUI 当前选中/变更状态）、`get/replaceModel`、`checkSyntax`、`getTemplateList`、`add/deleteTemplate`、`get/replaceTemplate`
- **位置与边操作**：`getLocations`、`add/modify/deleteLocation`、`add/modify/deleteEdge`、`addBranchPoint`（概率边分支点）、`deleteByXPath`（XPath 表达式删除任意元素）
- **声明区操作**：`get/setGlobalDeclarations`、`get/setLocalDeclarations`、`get/setSystemDeclarations`
- **查询管理**：`getQueries`、`insert/removeQuery`、`setQueryOption`、`getQueryTrace`、`getQueryPlot`
- **验证执行**：`verifyQueries`、`stop`（取消进行中的验证）、`getResults`、`getOptions`

**双工作流对比**：
- **Workflow A（外部 Agentic Client + MCP）**：用户通过 Claude Code 等外部客户端输入提示，客户端维护聊天历史并通过 Model Context Protocol (MCP) 将 tool call 转发至 UPPAAL 本地 HTTP server（langchain4j）。UPPAAL 在此模式下处于被动响应状态，无法主动推送用户手动编辑事件，需依赖额外 `getContext` 轮询检测变更。
- **Workflow B（嵌入 UPPAAL 聊天）**：用户在 UPPAAL GUI 内直接与 LLM 对话，UPPAAL 通过 langchain4j 的 LCMessages Java API 向 LLM Service 发送包含初始指令、Tools API 描述、当前会话历史、GUI 上下文（选中的元素/视图）及用户 prompt 的请求。UPPAAL 主动掌控上下文注入，无需额外 polling。

**核心循环原理**：
```
用户自然语言需求 → LLM 提议模型/查询/解释
→ UPPAAL 解析语法并运行 verifyta/verifyta-smc/STRATEGO
→ 返回 verdict + counterexample / trace / probability estimate
→ LLM 用通俗语言解释原因并建议下一步修改
→ 用户确认接受或拒绝
```
该循环可延伸至定量问题（SMC 概率估计）与控制器合成（STRATEGO reinforcement learning）。

**损失/评估函数**：
本文未引入新的 loss function，而是依托 UPPAAL 已有的 TCTL 公式判定（reachability/safety/liveness/deadlock-freedom）、SMC 假设检验与置信区间、以及 STRATEGO 的奖励函数（如 `minE(dist + 10^4 excess + 100 deficit)`）。

## 实验与结果
**数据集与基线**：
共七项 case study，分别使用 UPPAAL 的经典示例（Fischer 协议、LEGO 分拣机、车道变换模型）与真实应用域（COVID-19 流行病、雨水蓄水池、TSN 网关调度、ATM 网络）。基线为手工建模与逐项调试的传统工作流，并与作者先前发表的工作（Jensen et al., ISoLA 2020 COVID-19 fluid model-checking [9]）对照。

**主要结果数字**：
| 案例 | 用户目标 | UPPAAL MCP 调用次数 | 关键成果 |
|---|---|---|---|
| 传送带分拣 | 验证黑/红盒分类正确性 | 59（4 提示词） | 发现单定时器设计的 bug，改为每盒独立定时器后 3s 内验证；规模扩展至 8 盒使用 scalar sets 对称约简 |
| 雨水蓄水池 | 从零构建 STRATEGO 模型并学习控制器 | 39（12 提示词） | 自动生成 ODE  Pond 动力学 + CTMC Rain 场景，学到周期控制策略；平均水位 1.32m，溢出概率 ≤5%；将离散步长从 0.01 调至 0.5 min 使训练加速 ≈28× |
| TSN 网关调度 | 评估时钟漂移鲁棒性 | 13（8 提示词） | 枚举 29 种配置，发现 SMC 预测的 miss 概率 1/64 实际应为 2/64；精确最坏情况延迟 WCRT=46+Δ |
| Fischer 协议 | 向新手解释协议并展示漏洞触发条件 | 50（11 提示词） | 成功生成 req⇝cs starvation 环路（15 步 lasso），演示不等式放宽导致的 race condition（6 steps @ 0.007s） |

**最强结果与提升幅度**：
- **传送带案例**：LLM 在不被要求的情况下主动提出 scalar set symmetry reduction，使原本会超时的大规模实例（5 黑盒命名实例）能在 3 秒内完成验证——作者承认"我们从未将该模型推进至此深度"。
- **TSN 案例**：精确引擎纠正了 LLM 的 probabilistic prediction，并找到第二条违反 case（延迟 49 > 48），体现 verifier-as-critic 的价值。
- **教学价值**：Fischer 案例中学习者只需少量 effort 即可获得一套交互式 mutation 实验菜单（四个编号实验）与模拟器 walkthrough，显著降低入门门槛。

## 相关工作脉络
1. **Baldur (First et al., ESEC/FSE 2023)**：用 LLM 生成并修复完整形式化证明， specification 给定且不变；UPPAAI 差异在于 LLM 同时写入模型与属性，需检验抽象保真度。
2. **Lemur (Wu et al., ICLR 2024)**：将 LLM 嵌入 automated program verification 循环，同样依赖外部 prover 作 critic；UPPAAI 进一步扩展至统计检查与控制器合成，支持 hybrid stochastic dynamics。
3. **Loop Engineering (Lulla et al., under review JAWs@ASE 2026)**：定义"generation ↔ external evaluator ↔ repair"的软件工程范式；UPPAAI 是该范式在实时系统建模领域的具体落地实例。
4. **Fluid model-checking in Uppaal for Covid-19 (Jensen et al., ISoLA 2020)**：作者先前的 COVID-19 流体建模工作；UPPAAI 案例显示 LLM 首次提案即达到该工作的最终产出，且自动衍生出 CTMC 与 ODE 变体并验证三者一致性。
5. **UPPAAL STRATEGO (David et al., TACAS 2015)**：时系统上的强化学习控制器合成框架；本文将其工具化暴露给 LLM，实现"需求→随机场景→学习→策略评估"的端到端自动化。
6. **Self-refine (Madaan et al., NeurIPS 2023)**：LLM 自我迭代修正；UPPAAI 明确反对此类 self-evaluation 路线，转而采用 sound external verifier，呼应 Kambhampati et al. (ICML 2024) 的立场声明。

## 局限性与未来方向
- **单次会话评估**：每个 case study 仅进行一次 LLM 会话，未重复测量输出的方差与稳定性；未来需跨 provider (OpenAI / Mistral / Gemini) 与多 LLM 做系统性比较。
- **信息泄露风险**：本地 MCP 服务器让 AI agent 具备访问操作系统 CLI、文件系统的能力（如 TSN 案例中 Claude Code 自行搜索 verifyta），存在敏感数据外泄隐患；远程 API 工作流更安全但对付费 plan 有依赖。
- **接口限制导致体验瓶颈**：
  - 长时间验证（>1 min）会阻塞 editor 和所有 MCP 调用，即使已点击 GUI stop 按钮
  - 无法从 API 访问 engine options 与 simulator
  - `replaceModel` 清空 undo history
  - Learned strategies 仅驻留 engine memory，未持久化进模型文件（后通过新增 `saveStrategy`/`loadStrategy` 查询解决）
  - 默认不启用 trace 输出，诊断需显式请求
- **真实性问题仍未根治**：LLM 可能提出正确的抽象级别错误、空洞性质或不断弱化查询直至通过；verifier 只能判对错，无法保证"模型与现实系统的映射保真度"。
- **未见基准评测集**：目前以 qualitative case studies 为主，缺乏标准 benchmark suite 与自动化评分管线。

## 研究启发与可借鉴点
1. **"Verifier as oracle, not just post-hoc check"**：将模型检查器的输出（含 timeout、busy、counterexample length）作为第一 class citizen 注入 LLM 的下一轮 prompt，而非仅在最终报告中使用——该 design choice 在 belt 案例的 symmetric reduction 建议、TSN 的案例枚举 sweep 中体现出显著提升效率的潜力。
2. **Tools API 粒度设计值得移植**：按"模型/位置/边/声明/查询/验证"六个正交维度切分函数，既满足 LLM function calling 的 schema 简洁性要求，又保证原子操作的组合灵活性；可推广至其他 formal methods 工具（Isabelle/HOL, Coq, Z3）。
3. **Dual workflow architecture 为生态集成铺路**：Workflow A 利用 Claude Code 等成熟 agentic environment 的低 adoption friction，Workflow B 提供 context-awareness 与 sandbox 安全优势；这一 dichotomy 可作为后续"embedded vs plug-in"formal-methods-AI 集成的通用 taxonomy。
4. **Case-study 报告模板标准化**：每项研究都包含 prompt 数、MCP 调用总数、失败率（timeout/busy/error）、查询数量、模型变体数、wall-clock 时间、assistant 时间等量化指标，形成可复用的 reproducibility audit trail；适合纳入 formal methods 论文的 open science practice。
5. **主动 feature suggestion 现象**：LLM 在被问"是否满足某性质"时额外建议 scalar sets 对称约简、雨景 stress test、guard band 实验等未经明示的分析路径——说明 prompt design 应鼓励exploratory assistance 而非仅回答性问答。

## 关键术语表
- **UPPAAL**：丹麦奥尔胡斯大学开发的开源时系统与 cyber-physical 系统建模与验证平台，支持定时自动机、随机杂交自动机、时游戏与强化学习控制器合成等多种分析模式。
- **Model Context Protocol (MCP)**：Anthropic 提出的标准化协议，允许 AI agent（如 Claude Code）通过工具调用与外部程序（此处为 UPPAAL 本地 server）进行结构化交互，支持上下文推送与沙箱化权限控制。
- **TCTL (Timed Computation Tree Logic)**：UPPAAL 用于符号模型检查的属性语言子集，包含 E<>（存在可达）、A[]（全局安全）、A<>（活性）、−−>（leads-to）等算子，不允许嵌套时序算符。
- **Statistical Model Checking (SMC)**：基于大量蒙特卡洛模拟轨迹统计估计时序属性的满足概率，适用于带有随机延时或 ODE 动力学的非确定性混合系统；给出带置信区间的概率 bound 而非精确判定。
- **UPPAAL STRATEGO**：集成于 UPPAAL 的强化学习控制器合成模块，可将时系统视为 timed MDP，通过学习近优策略在满足硬约束（如不溢出）的同时优化期望代价（如平均水位最高化）。
- **Scalar Set**：UPPAAL 特有的对称约简技术，通过将同质对象（如多个相同类型的盒子）编码为标量集合上的取值而非逐个命名实例，大幅压缩状态空间以实现大规模验证。
- **Counterexample Trace**：当安全性或活性属性被违反时，符号引擎（verifyta）生成的最短反例执行轨迹（步骤序列 + 变量赋值），用于定位 bug 根因并指导修复。
- **Tool Calling / Function Calling**：LLM 在推理过程中根据 prompt 生成结构化 JSON 调用特定函数的能力；UPPAAI 借助此机制让 LLM 动态读写模型、插入查询、触发验证并获得结果回灌。

## 可复现要素
- **数据集**：使用 UPPAAL 官方 demo 与公开文献中的经典案例（Fischer 协议、LEGO 分拣机、车道变换、IEEE 802.1Qbv TSN 调度论文 [2,7]、ATM 网络等）；COVID-19 模型参考作者先前工作 [9]。案例相关文件已公开于 https://deis-tools.github.io/uppaal-models/CaseStudies/AI-IFS2027
- **代码/权重开源**：UPPAAI 作为 UPPAAL 的扩展实现，论文附录提供了 MCP 配置指引与 langchain4j 集成细节；但完整源代码仓库未在正文标明，待作者发布。
- **LLM 环境**：主要使用 Anthropic Claude（Claude Desktop / Claude Code）；API 兼容层亦支持 OpenAI、Mistral、DeepSeek、Gemini、Grok、Groq、Ollama 等（Table 2）。
- **关键超参**：论文未显式列出学习率、episode 数、discretisation step（雨水案例 0.01→0.5 min 为经验调优）、SMC 置信水平（95% CI）、STRATEGO 训练 budget 等细节——均需阅读原 UPPAAL STRATEGO/SMC 文档获知。
- **硬件要求**：未指定；STRATEGO 训练耗时约 7 min（0.01 min 步长）至 <2 min（0.5 min 步长），verifyta 最大状态数达 7·10⁶（TSN 扫描），内存占用约 1.3 GB。
