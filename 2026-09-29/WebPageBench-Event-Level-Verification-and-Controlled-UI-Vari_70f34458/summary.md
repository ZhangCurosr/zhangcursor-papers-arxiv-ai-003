---
title: "WebPageBench-Event-Level-Verification-and-Controlled-UI-Vari"
source: https://arxiv.org/pdf/2609.35026v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 08:15:36"
field: "Web Agent Benchmarking"
keywords: ["web agent", "benchmark", "event-level verification", "UI variant", "agent evaluation"]
innovations: ["事件级验证合约：通过 typed events 匹配判定成功，无需 judge 模型且与界面语言无关", "受控 UI 变体生成：9 个配置键 33 种实现，87 个变体隔离控件敏感性", "Completion–Verification Gap 量化：独立记录 agent 完成声明与客观验证，揭示 over/under-claim"]
benchmarks: ["WebPageBench"]
---

# 论文速读：WebPageBench-Event-Level-Verification-and-Controlled-UI-Vari

## 一句话总结
本文提出 **WebPageBench**，一个基于事件级验证的开源 web agent 评测框架，通过在模拟网站上埋点记录 typed events，实现无 judge 模型的任务成功率判定，并利用同一 instrumentation 机制生成受控 UI 变体，以量化 agent 对界面形式实现的敏感性。

## 研究问题与动机
- 现有 web agent benchmark 在**可验证性与规模**之间难以兼得：离线轨迹匹配（如 Mind2Web）可复现但评估的是路径而非结果；真实网站基准（如 WebVoyager、InSTA）依赖 judge 模型，且受反爬虫机制和内容漂移影响。
- 自托管环境（如 WebArena、REAL）仅检查 episode 结束时的应用状态，**无法区分 agent 的执行步骤与最终留下的状态**，也缺乏对中间过程的细粒度验证。
- 现有 benchmark 无法**隔离 agent 对单个 UI 控件实现的敏感性**：同一任务通过 popup 日历与原生 `<input type="date">` 渲染时，聚合分数无法区分是任务理解错误还是控件交互失败。
- Agent 的"完成声明"（completion signal）与真实成功率之间存在显著差距（最大达 41 个百分点），现有协议缺乏对这种**over-claim / under-claim** 的测量能力。

## 核心贡献（创新点）
1. **事件级验证合约（Event-Level Verification Contract）**：定义 19 种带类型参数的事件及 per-task 条件匹配逻辑，通过 EMS（Event-Match Score）判定任务成功，无需任何 judge 模型，且与界面语言无关。
2. **受控 UI 变体生成机制（Controlled UI-Variant Generator）**：通过 9 个 UI 配置键（8 个控件 + 1 个主题）与 33 种实现，从 25 个规范任务生成 87 个变体，其中 82 个变体仅改变单一控件，prompt 与成功条件完全不变，使界面敏感性可测量。
3. **统一 Runner 与排行榜（Runner & Leaderboard）**：一套 eval pipeline 同时支持 DOM 型 harness（6 种）与 screenshot-only GUI agent（5 种），独立记录 agent 的 completion 信号与事件验证结果，揭示 over-claim/under-claim 差距。

## 方法详解
**整体架构**：由 Docker Compose 部署的 mock 站点构成，包含后端事件存储、前端页面与评估容器。每个 task 是一个 JSON 文件，声明 prompt 与成功条件；domain 配置文件决定控件实现；两者组合为 isolated track。

**事件系统（Events）**：
- 界面在用户/agent 操作时向 track log 写入 typed event，格式为 `{"type": "<event_name>", "<param>": <value>, ...}`。
- 19 种语义事件类型（如 `basket_add`、`select_tariff`、`bench_hotel_select_guests`），低级别 DOM 事件（click/keypress/scroll）也记录但不用于判定。
- 条件匹配：字符串精确/通配符匹配、数值比较运算符、group 任意满足即可；除数量类事件外，不强制事件顺序。

**验证合约（Verification Contract）**：
- 主要指标：EMS（Event-Match Score），每任务二元值（1 = 所有条件满足，0 = 否则）。
- 额外信号：`Completion`（harness 自身声明完成）、`Done&pass`（agent 声明完成且验证通过）。
- Completion – EMS = over-claim 率；EMS – Done&pass = under-claim 率。

**UI 配置键（UI Configuration Keys）**：
- 日期选择：6 种实现（split popup / inline calendar / typed string / native `<input>` / single-field range / grid popup）。
- 城市选择：autocomplete vs native select。
- 客人计数：per-room popup / inline stepper / compact select / pill buttons。
- 共 9 个键、33 种实现，同一控件的不同实现发出相同事件与参数，条件跨实现有效。

**变体生成**：
- Canonical task（手写）→ 应用 profile（UI key 赋值）→ 生成 variant（仅改变控件实现，prompt 与条件不变）。
- 87 个变体来自 25 个 donor，82 个只变一个控件，5 个文档柜变体同时改变 layout/year/button 三个共享屏幕的控件。
- 敏感性度量：$\Delta_{h,w} = \text{EMS}_h(D_w) - \text{EMS}_h(V_w)$。

**时间相关任务修复**：将所有日期条件向前偏移一个共享 offset，确保最早日期 ≥ 今天，并同步改写 prompt 中的日期表达。

## 实验与结果
**数据集**：
- 6 个 mock 站点：marketplace（32 任务）、books（21）、document cabinet（18）、rail travel（26）、grocery delivery（10）、hotel search（45）。
- 总计 152 任务：65 canonical + 87 variants，521 个条件，平均每个任务 3.4 个条件。
- 全俄语文本，但验证合约与语言无关。

**评估基线**：
- 24 个 model–harness 对：GPT-5.6-luna、Gemini-3.8-flash、DeepSeek-V4.1-Flash、Qwen3.8-27B、Fara-1.5、GLM-5.2、OpenCUA、EvoCUA、UI-TARS 等 × browser-use、OpenManus、OpenHands、Ouroboros、Fara、OpenCUA 等 harness。

**主要结果**：
| 最佳 pair | EMS | Completion | Gap |
|---|---|---|---|
| Gemini-3.8-flash × OpenManus | **0.822** | 1.000 | +0.178（over-claim） |
| GPT-5.6-luna × Ouroboros-full-iso | **0.796** | 0.868 | +0.072 |
| DeepSeek-V4.1-Flash × Browser-Use | 0.789 | 0.993 | +0.204 |
| Qwen3.8-27B × OpenManus | 0.763 | 1.000 | +0.237 |

- **Completion–Verification Gap 最大达 41 个百分点**：GPT-5.6-luna × OpenManus（Completion=1.000，EMS=0.592），该模型在所有任务上均声明完成但仅 59% 真正达标。
- **Gap 方向随 harness 翻转**：Gemini × OpenHands（Completion=0.461，EMS=0.572），存在 under-claim（18/87 验证成功未触发完成信号）。
- **GUI 代理显著落后 DOM 代理**：Qwen3.8-27B 通过 DOM harness 得分 0.737–0.763，通过截图 harness 仅 0.480；UI-TARS-1.5-7B（0.336）、EvoCUA（<0.10）。
- **Cost ≠ Performance**：GLM-5.2 × OpenHands 花费 $76.08 但仅得 0.500；GPT-5.6-luna × Browser-Use 仅 $3.23 达 0.704。

## 相关工作脉络
1. **Mind2Web / AssistantBench / WebVoyager**：基于 reference match 或 live-web + judge model，不检查环境内部状态，依赖外部 LLM 判断结果。
2. **WebArena / VisualWebArena / REAL**：自托管环境通过应用状态 diff 验证，但仅能检查 episode 结束时的最终状态，无法捕获中间过程。
3. **WorkArena++**：确定性验证，但仅支持换品牌/换颜色，不改变控件实现，无法隔离 UI 形式敏感性。
4. **WebShop / AutoWebWorld / WebForge**：通过属性 reward、state-machine replay 或 final-state comparison 判定，均不记录中间事件流。
5. **BrowserGym / AgentLab 生态**：提供共享 agent 接口封装多个 benchmark，本文的 runner 可与其对接；本文与它们是互补关系而非竞争。

## 局限性与未来方向
- **Mock 环境而非生产网站**：不包含反爬虫机制、真实支付链路、内容漂移等现实因素。
- **跨 session 场景不可表达**：应用状态（购物车、收藏、登录）存储在浏览器本地，无法测试多 session 任务。
- **仅俄语界面**：未测试代码切换（codeswitching）或转写（transliteration）。
- **未报告敏感性指标**：由于提交数据未分离 donor 与 variant，当前快照无法报告 $\Delta_{h,w}$。
- **事件匹配不证明理解**：条件独立检查且无顺序约束，agent 理论上可在不理解任务的情况下触发所需事件；论文未进行对抗审计验证。

## 研究启发与可借鉴点
1. **事件级验证替代 judge model**：在可植入 instrumentation 的仿真环境中，通过 typed events 匹配判定成功，彻底消除 judge 模型的偏差与语言依赖，可直接迁移至 robotics/simulation 评测。
2. **UI 变体生成机制**：将"控件实现"作为可控实验变量，同一 prompt+条件渲染多种 UI 形态，为 agent UI robustness 评测提供了标准化范式。
3. **Completion–Verification Gap 作为独立指标**：将 agent 的 self-reported completion 与客观验证分离，量化 over/under-claim，值得作为 agent benchmark 的标准报告字段。
4. **自动时间偏移修复**：对含日期的任务在运行前统一偏移，保持任务文件不变的同时确保长期可复现，技巧简洁实用。
5. **DOM 与 screenshot action space 统一评测**：同一 verifier 服务两种异构 agent，消除了 hessian 差异，可在公平比较中暴露模型 vs harness 的贡献。

## 关键术语表
**Event-Level Verification**：通过匹配界面埋点记录的 typed events 而非 judge 模型或最终页面状态来判定任务成功。
**Event-Match Score (EMS)**：任务级二元指标（0/1），所有条件满足则为 1，取均值作为主要评测分数。
**UI Configuration Key**：一个可被配置替换的 UI 控件抽象（如日期选择器），同一 key 下挂载不同实现发出相同事件。
**Canonical Task**：手写编制的规范任务，prompt 与条件均为人工撰写，与变体形成 donor-variant 配对。
**Controlled UI Variant**：从 canonical task 复制而来，仅改变一个或多个控件实现，prompt 与条件完全保持不变的任务实例。
**Completion–Verification Gap**：Agent 声明完成的比例与事件验证通过比例的差值，正值为 over-claim，负值为 under-claim。
**Harness**：连接 agent 模型与浏览器/DOM 的执行层，本文评估 6 种 browser/DOM harness 与 5 种 screenshot-only GUI adapter。
**Interaction Class**：19 类 UI 交互 taxonomy（如 BASKET、DATE、COUNTER、PAY），每任务标注一类 primary 用于分层报告。

## 可复现要素
- **数据集**：152 个任务（65 canonical + 87 variants），发布在 GitHub 仓库；任务与界面为俄语。
- **代码**：已开源（论文中引用 repository link，Apache License 2.0）。
- **权重**：评测涉及 24 个 model–harness 对，各模型权重来自对应官方发布。
- **关键超参**：未特别强调；各 harness 使用默认 decode settings；时间限制为共享 timebound。
- **部署**：Docker Compose 一键启动，包含 backend + frontend + evaluation container。
