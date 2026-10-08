---
title: "nanoMuse-An-Open-Source-Personal-Agent-for-Every-Device-You"
source: https://arxiv.org/pdf/2610.08699v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-08 02:26:13"
field: "个人AI代理与GUI自动化"
keywords: ["personal agent", "GUI agent", "screen manipulation", "open-source AI", "agent safety", "cross-device orchestration", "mobile automation"]
innovations: ["进程内Sentinel策略边界与taint单向传播机制", "多设备peer架构下scope-based权限传递", "Hands模块以自然语言意图作为确定性策略判层"]
benchmarks: ["AndroidWorld", "OSWorld", "MemGUI-Bench", "OS-Harm", "MAS-Bench"]
---

# 论文速读：nanoMuse-An-Open-Source-Personal-Agent-for-Every-Device-You

## 一句话总结
nanoMuse 是一个开源的个人 AI Agent 系统（GPL-3.0），将完整的个人 Agent 运行在用户自有的多个设备上（Android、桌面、iOS、Web），通过可选的开源中继实现跨设备协同，填补了 Meta Muse 等闭源产品中缺乏"屏幕操控能力"和"用户可控性"的空白。

## 研究问题与动机
- **闭源个人 Agent 无法被用户审查**：Meta Muse（2026.9）等产品虽展示了个人 Agent 的完整形态，但运行在厂商云中，代码、模型、服务均不开放，用户无法审计其记忆文件或行为日志。
- **现有开源方案缺少"屏幕之手"**：Hermes Agent、OpenClaw 等运行在自有设备上但无屏幕操控能力；OpenMinis 有手机屏幕操控但缺少跨设备协同和长期记忆体系；OpenMuse 仅覆盖浏览器/桌面，无法操作原生手机 App。
- **厂商方案"逐设备一台 VM"的成本与锁定不可持续**：Muse 为每人分配一台 Linux VM，运行成本由订阅制或厂商其他业务承担，且数据存储在单一国家/厂商云内，形成供应商锁定。
- **屏幕操控是个人 Agent 覆盖真实生活的关键缺口**：许多日常应用（银行、政府服务）无 API 或 Web 界面，只有拥有真正"屏幕之手"的 Agent 才能操作。

## 核心贡献（创新点）
1. **首个端到端可运行的开源个人 Agent 全系统**：涵盖 Android App、iOS App（TestFlight）、桌面 App、Web App 及中继服务器（nanoMuse Cloud），GPL-3.0 许可，每段代码均可读、可审计。
2. **在个人 Agent 中首次将"屏幕操控"置于核心架构**：Android 设备内置完整的 Hands 模块（基于 MemGUI-Bench 方言的单截图-单动作循环），桌面端移植自 UI-TARS-desktop，打通了从 skill/命令行/MCP 到直接操作屏幕的四级层级。
3. **每台设备各自运行完整 Agent 实例，通过中继共享一次会话**：设备间互为工具，"在手机查订单号再填入 Mac 表格"是一条跨设备对话，权限通过有作用域的授权（scope grant）传递，防止越权。
4. **Sentinel 策略边界设计（进程内而非进程外）**：每次工具调用经过固定决策顺序（拒绝列表→用户规则→脏标记→风险分级→不可绕过的警告），每条决策记录在用户可审计的日志中，与 Muse 的进程外权限边界形成对照。
5. **定义"个人 Agent"的五问三视界框架**：从记忆可读可编辑、一处 Agent 多只手、审批即信任语法、在同意下学习、低成本自治运行五个维度提出评估标准，为后续研究提供概念基线。

## 方法详解
- **整体架构**：每个设备（Android / macOS / Windows / Linux / Web）搭载一个完整的 nanoMuse Agent 实例，包含自身 Sentinel、记忆文件、Hands 模块和模型调用；所有设备通过可选的中继（relay，nanoMuse Cloud）对齐身份与会话，同一人在各设备上的 Agent 互为工具。
- **Hands 屏幕操控**：四级爬梯策略——skill → 命令行工具 / MCP 服务器 → 带登录态的页面 → 设备屏幕。最后一级 Hands 采用 MemGUI-Bench 方言的单截图-单动作循环：模型输入当前截图、目标指令、每步一条历史摘要、设备无障碍树（如有），输出思考、一句自然语言意图说明、一个手势。自然语言句子作为"可被策略判断的中间层"：确定性策略无法直接判图，但可以判模型将要点击的文字。
- **敏感步骤的确定性拦截**：标签含承诺词（confirm payment / transfer / place order / send / delete）的步骤每次都需审批；一次性提交的输入（如 messenger 回车即发送）同样敏感；无标签步骤携带警告。Android 同时读取无障碍文本，取两者更严格的判定。操作时屏幕上显示"胶囊提示+红 Stop 按钮+下一步触点环形标记"；桌面端以边缘辉光+路径轨迹呈现，每条轨迹含截图与动作标注。
- **Sentinel 策略边界**：五步决策顺序——① 拒绝列表工具；② 用户自定义规则；③ always-allow / always-ask 列表；④ 脏标记（读取私有数据后，任何向外部发数据的调用都转成 ask）；⑤ 用户设定的风险模式；最后是不可绕过的若干警告（破坏性 shell 命令、管道下载就地执行、含承诺词的标签）。审批粒度为 scope-based：once / this conversation / always（限于特定收件人、站点、文件夹、设备或应用）。记忆类审批（付款、密码）永不持久。
- **记忆文件系统**：手机端以 Markdown 文件存储：SOUL.md（Agent 自我描述）、USER.md（用户画像）、GLOBAL.md（跨会话信息）、日记文件、HEARTbeat.md（例行任务）；桌面端每个记忆行为独立一行，带变更日志和 undo，支持关键词召回或 embedding 语义召回， tidy-up 自动合并但绝不覆盖用户写入内容。Feed / Ideas / Goals 由 Agent 在后台隐式对话中生成，用户在"房间"中阅读。
- **模型选择**：内置 18 个提供商目录（含国内与海外），支持用户自带 API Key（只暴露模型调用，不暴露其他数据）、ChatGPT Plan（经 OpenAI Codex 授权，覆盖聊天与 Hands）、本地模型服务器三种接入方式。
- **中继（Relay）数据保留**：仅保存账户哈希、发出的密钥、分配的设备清单、模型调用账本（model/token/cost，不含内容）、Agent 配置、心跳、以及用户选择同步的对话文本（默认主对话）。不存储文件、图片或工具返回值。账户删除后 90 天内仅保留密钥哈希以便离线设备感知下线；"帮助改进模型"开关默认开启（社区中继）/ 由自建中继自行决定。

## 实验与结果
- **规模与成本估算**（论文未做大规模 benchmark 评测，以工程指标为主）：Android App 下载 38 MB（arm64）；桌面 App 下载 256–498 MB，安装后 Linux 占 928 MB，空闲内存约 0.5 GB；中继仅需 1 个 Python 进程 + 1 个 SQLite，空闲内存约 85 MB，建议配置 1 vCPU + 1 GB 内存，月租 ¥30–60 或 $4–6；模型调用按百川 2026.10 定价：对话模型 ¥2/百万输入 token、¥8/百万输出 token（高峰时段），Hands 模型 ¥3/¥12，日常聊天每日仅几 fen。
- **对比基线**：与 Muse（Meta）、Today、Manus Cue、OpenAI dots 等闭源个人 Agent 对照；与 Hermes Agent、OpenClaw、OpenMinis、OpenMuse 等开源/可自托管项目对照。Table 1 汇总各方"运行位置 / 手机屏幕 / 电脑屏幕 / 是否开源"。
- **最强定位结论**：nanoMuse 是目前唯一同时具备"开源全栈 + 跨设备 peer 架构 + Android 屏幕操控 Hands"的完整个人 Agent 系统；在手头公开可用的方案中，它在"设备覆盖度"和"可审计性"上领先。
- **缺失的定量结果**：论文明确承认"尚无成功率与人工接管次数等 Hands 评测数字"，并指出 Linux 端 Hands 仅支持 X11 会话；未来计划在 AndroidWorld、OSWorld、MemGUI-Bench、OS-Harm 四个基准上补充数字。

## 相关工作脉络
1. **Android 端起点 OpenMinis**（Jun 2026, GPL-3.0）：nanoMuse 在其之上叠加了跨设备中继、持久记忆、审批流和桌面/移动端多客户端；OpenMinis 仅有手机端无跨设备协同。
2. **桌面端基座 DeepSeek Harness**：nanoMuse 桌面 App 围绕 DeepSeek 插件化运行时构建，承接了 Python 运行时与 MCP 生态，区别于 CopilotKit 的 OpenMuse（OpenMuse 以服务器+浏览器为主，无原生桌面 Hands）。
3. **Hands 模块来源 UI-TARS-desktop / MemGUI-Bench**：屏幕操控采用 MemGUI-Bench 方言的单动作循环（手机端）与 UI-TARS-desktop（桌面端）的移植；二者共同对应 MAS-Bench 的"shortcut + 屏幕"混合评测思路 [46]。
4. **长程手机 GUI Agent 系列工作**：包括 CogAgent [7]、AppAgent [45]、Mobile-Agent / Mobile-Agent-v3 [38, 44]、AutoGLM [16]、UI-TARS [32]、Aguvis [43]、OS-Atlas [41]、OpenCUA [39] 等，nanoMuse 将其整合为可落地的产品级流程并补充了审批/记忆/跨设备治理层。
5. **闭源对标 Muse (Meta, Sep 2026)**：Muse 的安全设计（Sentinel 进程外权限边界、surrogate token、kernel taint tracking、 accessibility tree 隔离）启发 nanoMuse 在进程内 Sentinel 上做简化复刻；Muse 因无原生手机屏幕操控而成为 nanoMuse 的核心参照与差距来源。
6. **个人 Agent 先行开源项目 Hermes Agent / OpenClaw**：二者已实现"在用户自有设备上运行 + 跨会话记忆"，但均无屏幕操控；nanoMuse 在此基础上补全 Hands 与跨设备协同。

## 局限性与未来方向
- **进程内策略边界而非进程外特权边界**：Sentinel 与 Agent 同域运行，设备被攻陷即 Agent 被攻陷；虽有 taint 规则与 Linux sandbox 缓解，但无法达到 Muse 那种"真实凭证由宿主服务持有"的安全性。
- **Hands 缺少量化评测**：尚未报告成功率与人工接管次数；对图标-only 按钮、高 DPI 专业屏幕等场景的处理仍是未解问题；Linux 端仅支持 X11。
- **记忆缺乏元数据（provenance）**：当前记忆文件未记录每行由哪个模型、何时、以何种置信度写入，也没有对旧记忆的自动复核机制；强模型可能将弱模型的猜测行当作事实。
- **中继为共享且存文本**：社区中继仅由项目组一台机器运行，用户同步的对话文本存储于非自持服务器；"改进模型"开关默认开启，项目组自认"对此最不放心"。
- **Muse 描述基于公开资料与一份未经验证的 prompt 副本**：prompt 随每次部署变化，文中 October 2026 版本的推断可能与现行生产版本存在偏差。
- **未来方向**：近端补齐 Muse 体验（时机敏感的记忆召回、任务间主动性、更少的 Hands 失败率、一键自托管）；中期建立开源评测套件 + 小规模自训 Hands 模型；远期延伸至物理世界（摄像头/机器人臂/家电）与"跨模型与设备存活"的记忆/身份体系。

## 研究启发与可借鉴点
1. **"用文字作为屏幕动作的中间可判层"**：让模型先输出一句"将要做什么"的自然语言意图，再交由确定性策略基于关键词/标签进行审批与拦截——这一设计绕开了纯图像识别难以形式化验证的问题，可直接迁移到桌面 GUI Agent 或任何视觉-动作闭环系统中。
2. **脏标记（taint）驱动的风险放大策略**：读取私有数据后，所有出站调用自动升级审批强度，反之不反向降级。该单向传播模型简单且鲁棒，适合通用 Agent 安全框架。
3. **跨设备 peer 架构 + scope-based 权限传递**：每台设备持有自己的 Sentinel，跨设备任务只在发起端获得一次授权，执行端再经本地 Sentinel 二次审查；这种"最小权限 + 本地策略"的组合可直接推广到多智能体协同与家庭 IoT 场景。
4. **记忆以可读写文件呈现 + change log/undo**：将 Agent 记忆落地为普通文件而非黑盒向量库，既便于用户审计，又天然兼容迁移与替换模型；该思路对 Long-context / episodic memory 研究有参考价值。
5. **四档 Hands 层级（skill → CLI/MCP → 浏览器 → 屏幕）**：用层次化的"能力阶梯"避免过早进入屏幕操控；对研究"何时该用快捷方式、何时该用视觉操控"的问题提供了明确的工程框架，可与 MAS-Bench 等混合评测结合进一步量化。

## 关键术语表
**Personal Agent**：长期服务于单一用户、跨会话运行、可主动通知并在用户缺席时执行任务的 AI 程序，区别于一次性完成任务的助手。
**Hands**：nanoMuse 的屏幕操控模块，采用单截图-单动作循环，通过模型输出的自然语言意图接受确定性策略的审批。
**Sentinel**：运行于 Agent 进程内的策略边界，所有工具调用必须经其五步决策顺序审批，记录可审计日志。
**Taint 规则**：Agent 一旦读取私有数据，后续所有向外部发送数据的调用自动触发审批，且该传播不可逆。
**Scope Grant**：按"一次性 / 本次对话 / 针对特定对象永久"等粒度授予的权限，付款和密码类权限永不被记住。
**Relay（中继）**：连接多设备 Agent 实例的共享服务，仅保存账户哈希、调用账本和可选同步的对话文本，不存文件与工具返回值。
**MemGUI-Bench 方言**：手机端 Hands 采用的单动作交互协议，源自 MemGUI-Bench 基准中的手机操作规范。
**UI-TARS-desktop**：字节跳动开源的桌面 GUI Agent 应用，nanoMuse 桌面端的手部操作移植自此。

## 可复现要素
- **数据集**：本文未提出新数据集；Hands 规划将在 AndroidWorld、OSWorld、MemGUI-Bench、OS-Harm 上评估（均未在本次发布中给出数字）。
- **代码**：已开源，GitHub: https://github.com/nano-muse/nanoMuse（GPL-3.0）。
- **权重/模型**：模型由用户自选，未绑定自有模型；社区中继默认开启的"改进模型"开关会收集部分对话文本用于训练开源 Hands 模型（规划中，未公布具体数据规模与训练细节）。
- **关键超参**：论文未提及模型侧超参；运行时参数包括 Hands 的 screenshot→action 循环频率、标签关键词列表（confirm payment/transfer/place order/send/delete 等）、taint 触发条件（读取私有数据即置位）、决策顺序的五步优先级等，具体数值以仓库为准。
- **演示/项目主页**：https://nanomuse.cn / https://nanomuse.cn/web/
- **依赖基座**：Android 端基于 OpenMinis（GPL-3.0）修改；桌面端基于 DeepSeek Harness；Hands 移植自 UI-TARS-desktop 与 MemGUI-Bench。
