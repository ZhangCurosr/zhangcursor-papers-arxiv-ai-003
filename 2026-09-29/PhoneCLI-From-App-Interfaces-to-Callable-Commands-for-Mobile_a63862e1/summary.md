---
title: "PhoneCLI-From-App-Interfaces-to-Callable-Commands-for-Mobile"
source: https://arxiv.org/pdf/2609.35671v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 03:16:24"
field: "低资源 GUI 控制与 Agent 增强"
keywords: ["Mobile GUI Agent", "App Map Compilation", "Deterministic Replay", "Two-Phase Routing", "Zero-Shot App Integration", "VLM Fallback"]
innovations: ["离线编译 App 导航为确定性可调用命令，在线子秒零 VLM 重放", "两阶段保守路由（选择+验证）+ 事后着陆检查保障成功率下界", "分层消融揭示地图注入与确定性重放的正交互补贡献"]
benchmarks: ["AndroidLab", "AndroidWorld"]
---

# 论文速读：PhoneCLI-From-App-Interfaces-to-Callable-Commands-for-Mobile

## 一句话总结
PhoneCLI 提出"编译+解释"双层架构，将移动应用的静态 GUI 导航离线编译为确定性可调用命令（每屏一条），在线执行时以子秒级零 VLM 成本完成导航；开放式交互与编译失败路径则优雅回退至嵌入式 VLM 解释器。该方法无需任何应用内部 API、运行时插装或模型训练，即可在 AndroidLab 和 AndroidWorld 双基准上同时提升任务成功率并降低步骤/Token 消耗。

## 研究问题与动机
1. **闭源困境（Closedness）**：移动端应用无可用的程序化接口（区别于桌面端 CLI/API），agent 只能依赖截图-感知循环，速度慢（每步数秒）、成本高（每步一次 API 调用）、易幻觉坐标与导航失败。
2. **重复性高但效率低（Repetitiveness）**：日常 App 使用高度重复、遵循稳定有序模式，导航骨架变化远慢于应用内容，却仍被 VLM 逐步感知与决策，造成浪费。
3. **泛化瓶颈（Generalization）**：基于训练的方法每接入新 App 需重新采集数据或微调，百万级 App 规模下不可行；未见过的 App 没有直接捷径。
4. **现有方案未能消除重复导航瓶颈**：要么从内部增强 VLM 循环（更强模型+grounding+memory），要么承受训练成本——均未将"可预枚举的重复导航"与"不可枚举的开放交互"分离。

## 核心贡献（创新点）
1. **GUI-to-CLI 编译流水线**：将黑盒 App 离线编译为结构化地图与确定性命令目录，无需 API/插装/训练，新 App 分钟级集成；与 PreAct 等在线轨迹编译工作本质不同——它编译的是"App 是什么"而非"Agent 做过什么"。
2. **两阶段路由 + 确定性重放**：路由在子秒内完成，执行前选命+验证，执行后着陆检查，任何失败都优雅回退至纯 VLM 解释器；保证成功率下界不低于纯 VLM 基线，同时导航部分零 VLM 开销。
3. **双基准实证**：在 AndroidLab（138 任务/9 个 App）达到 SOTA（零训练，Qwen3.7-Plus 35B 从 50.7% 提升至 63.0%，超越所有 GUI 专用微调模型）；在 AndroidWorld 迁移至官方 M3A 也一致带来效率增益。
4. **分层消融揭示组件互补**：地图注入（+4.4 点）与确定性重放（+7.9 点）各贡献显著，任一单机制无法替代；STABLE/DYNAMIC 元素分类与语义富化是最重要的单组件。

## 方法详解
PhoneCLI 由三阶段构成，对应"编译 + 解释"的混合执行范式：

**Stage 1 — 离线编译（构建 App Map）**
- 输入：App 包名，干净启动状态。
- 结构探索：广度优先遍历（BFS），深度 ≤ 3、屏幕预算 ≤ 50、每屏最多滚动 3 页；记录每个屏幕的交互元素、归一化坐标、导航边与可达屏幕。
- 屏幕去重：基于原始元素签名（而非语义）去重，避免共享底部导航栏将不同 Tab 折叠为同一屏幕。
- 语义富化（LLM 驱动）：每个元素标注 STABLE/DYNAMIC 类型、生成语义别名与类型（button/input/label 等）；每个屏幕生成自然语言描述。
- 命令编译：以干净启动为起点，为每个屏幕生成确定性的操作序列 `c = [force_stop, launch, tap..., swipe...]`，坐标基于屏幕尺寸换算。

**Stage 2 — 在线调用（路由 + 重放）**
- 两阶段路由 `r(T, C) → (d, id, ans)`：
  - Phase I（选择）：将任务 T 与压缩后的符号目录（前缀树格式，仅含导航路径/语义标签/稳定 id，不含坐标/动作步骤）送入轻量 LLM，输出四类决策之一：
    - `FINISH`：无需设备交互，直接返回答案。
    - `NEED_VLM`：无匹配命令，交由 Stage 3。
    - `OP`：命令可独立完成，完成还需回答性检查。
    - `MACRO_VLM`：命令完成导航后仍需 VLM 继续交互。
  - Phase II（验证）：检查选中命令目标屏幕与自然语言描述是否与任务语义一致，不一致则拒绝，回退 Stage 3。
- 确定性重放：按命令序列执行 ADB 操作，全程零 VLM 调用，子秒完成。
- 着陆检查（Landing Check）：重放后匹配当前 UI hierarchy 与目标屏幕签名，不符则重启 App 并回退 VLM。
- 命令完成策略：OP 路径下检查目标屏幕是否可直接回答问题；若不能则降级为 MACRO_VLM，并向 VLM 注入着陆提示（landing hint：目标屏描述 + 交互元素列表），避免冷启动。

**Stage 3 — 运行时解释（VLM 回退）**
- 纯 VLM 感知-行动循环（INTERPRET），接收可选着陆提示 h。
- 显式记忆：每轮 VLM 响应附带 `STATE_ASSESSMENT` 字段，历史注入下一轮提示。
- 自我修正：检测无效行为（反复滑动/等待但无屏幕进展）并注入针对性提示；执行失败时记录错误并在下轮附上纠正提示。
- 优雅降级规则：任何编译路径失败均进入 Stage 3，且回退状态与纯 VLM agent 一致，确保成功率下界不退化。

**形式化**：App Map $\mathcal{G}=(S,E,\lambda,M)$，其中 $S$ 屏幕集合，$E$ 交互元素集合，$\lambda:E\to S\times\text{Alias}^*$ 记录导航边与别名，$M$ 命令集（每屏一条确定性序列）。

## 实验与结果
**数据集**
- **AndroidLab**：138 任务，9 个 App，程序化 XML 判定（94 任务）+ LLM judge（44 任务）；统一使用 Qwen3.7-Plus 重新评判以消除 judge 偏差。
- **AndroidWorld**：116 任务，20 个 App，跨应用工作流，比较对象为官方 M3A 基线。

**模型与实现**
- 三骨干：Qwen3.7-Plus（35B）、GLM-4.6V（107B-A12B）、Kimi-K3（2.8T-A104B）；均 off-the-shelf，temperature=0，每调用最多 512 tokens。
- AndroidWorld 统一使用 Qwen3.7-Plus。
- Set-of-Mark 观察约定；Android 13 模拟器（Pixel 7 Pro AVD，1440×3120），每任务前恢复干净快照，上限 25 轮。

**主要结果（AndroidLab，Table 1）**

| 模型 | VLM-only | PhoneCLI | Δ SR | Avg Steps | Avg Tokens |
|---|---|---|---|---|---|
| Kimi-K3 | 68.8% | 69.6% | +0.8 | 6.34 → 6.02↓5% | 53.3k → 48.1k↓10% |
| GLM-4.6V | 41.3% | 44.5% | +3.2 | 7.07 → 6.52↓8% | 61.4k → 54.4k↓11% |
| Qwen3.7-Plus | 50.7% | **63.0%** | **+12.3** | 7.52 → 6.72↓11% | 40.1k → 34.4k↓14% |

- 最强结果：Qwen3.7-Plus + PhoneCLI 达到 **63.0%**，超越 GPT-4o（31.2%）、Claude-Sonnet-4（40.6%）、GLM-4.6V（41.3%）、Gemini-2.5-Pro（56.5%），并超过全部 GUI 专用微调模型（最高 AutoGLM-Phone 47.7%）。
- 提升幅度：比 Kimi-K3 基线差距缩小至 5.8 点；相比 W/o Replay 变体（55.1%），确定性重放贡献更大增量（+7.9 vs +4.4）。

**AndroidWorld 迁移（Table 4）**
- M3A 官方：61.2% / 27.58M tokens / 8.4 steps
- M3A w/ PhoneCLI：65.5% / 25.04M tokens / 7.7 steps
- 收益：+4.3 点成功率，-9% tokens，-0.7 steps，证明不依赖 AndroidLab 专属设计。

**消融（Table 3，Qwen3.7-Plus）**
- w/o 元素分类（STABLE/DYNAMIC）：-15.9 点
- w/o 语义富化：-15.2 点
- w/o 命令完成：-14.4 点
- w/o 着陆提示：-13.0 点
- w/o 着陆检查：-12.3 点
- w/o Replay（仅地图注入）：55.1%（+4.4 vs 50.7%）
- PhoneCLI 完整：63.0%（+7.9 vs 55.1%）

**地图规模分析（Figure 3）**
- Setting/Clock 均呈倒 U 型：预算 10→50 提升，50→100 下降；默认 50 为甜点。
- 原因：小地图覆盖不足；过大地图膨胀路由候选集（10 屏约 130 tokens，100 屏约 3.5k tokens），增加误路由风险。

**错误分析（Figure 4）**
- 总失败从 68 降至 51（-17），其中 27 被 saved、10 被 lost（主要为路由代价）。
- 编译层修复的四大失败模式均被覆盖：循环（looping 16）、premature finish（1）、execution（6）、grounding（4）；grounding 失败全部被挽救。

## 相关工作脉络
1. **PreAct [15]**：在线编译 agent 成功轨迹为状态机程序，仅复现同一任务；本文离线编译 App 本身，支持未见任务。
2. **UI-KOBE [16]**：离线构建 UI 知识图，agent 仍需在环内逐跳导航；本文重放命令完全绕过 VLM 感知。
3. **GUI-explorer [17] / GUI-Xplore [18] / RAG-GUI [36]**：将探索结果以文本/视频/教程形式回灌 VLM，仍产生感知开销；本文编译为可执行命令，零感知成本。
4. **AutoDroid [35] / AppAgent [34]**：生成脚本/文档供 VLM 阅读后自行执行；本文让程序直接执行，VLM 只负责决策。
5. **GUI 专用微调模型（AutoGLM [1]、MobileUse [3]、UI-Tars [29]、V-Droid [30] 等）**：能力存在于权重，新 App 需重新训练；本文无需训练，分钟级集成新 App。
6. **混合 GUI+CLI 方案（PhoneHarness [8]、MCPCWorld [6]）**：依赖 OS 已存在的程序化表面；本文从无到有合成该表面，无需任何 API 或插装。

## 局限性与未来方向
- **地图规模与覆盖的权衡难以统一**：倒 U 型曲线表明不同 App 最优预算不同，目前需经验选择 50 屏。
- **动态内容无法编译**：Feed、推荐流、每次加载内容不同的页面仍依赖 VLM；编译对象限于导航骨架。
- **App 版本更新导致地图过期**：元素位置/语义变化后地图失效，需定期重建；论文未讨论增量更新策略。
- **跨应用工作流支持有限**：当前设计针对单 App 内导航，AndroidWorld 的跨 App 任务仍主要依赖 VLM。
- **深度嵌套与复杂表单**：多层级菜单、多步表单的长命令重放仍可能因轻微 UI 漂移导致着陆失败。
- **未来方向**：自动化增量地图更新、地图压缩与路由效率优化、结合 RL 微调路由模块、扩展到跨 App 场景与桌面 GUI。

## 研究启发与可借鉴点
1. **"编译+解释"双层架构**：将静态可预枚举部分预编译为确定性执行单元，动态部分保留 VLM 解释——该范式可迁移至桌面 GUI agent、网页自动化、RPA 等场景。
2. **两阶段保守路由设计**：Phase I 选择 + Phase II 验证 + 事后着陆检查，形成"事前-事后"双重 guard，保证不会因误判引入新失败模式；这种"宁可回退也不冒险"原则对安全敏感场景极具参考价值。
3. **符号化目录压缩**：目录仅携带路径前缀/语义标签/稳定 id，剔除坐标与动作步骤，使路由 LLM 调用 Token 开销可控（压缩至 1.9k tokens），且支持前缀树共享前缀——该压缩策略可复用于其他需要 LLM 做路由/规划的 GUI agent。
4. **着陆提示（Landing Hint）机制**：将"冷启动 teleport"转化为"热启动"，用地图本身的信息为 VLM 提供目标屏幕描述与元素列表，无需额外训练即改善 VLM 首步决策质量。
5. **消融揭示组件独立贡献**：地图注入（信息）与确定性重放（执行）分开评估并量化各自贡献，证明"知道去哪"和"真的去那"是正交且互补的两个维度——此设计对后续研究拆分"规划"与"执行"模块有借鉴意义。

## 关键术语表
- **PhoneCLI**：将移动 App GUI 导航离线编译为确定性可调用命令的框架，采用"编译+解释"混合架构。
- **App Map $\mathcal{G}=(S,E,\lambda,M)$**：结构化地图，包含屏幕集合、交互元素、导航边（元素→目标屏+别名）、以及每屏对应的确定性命令序列。
- **确定性重放（Deterministic Replay）**：离线编译的命令序列在运行时直接执行，全程零 VLM 调用，子秒完成导航。
- **两阶段路由（Two-Phase Routing）**：Phase I 从命令目录选择目标命令，Phase II 验证目标屏语义与任务一致性，二者均由轻量 LLM 完成。
- **OP / MACRO_VLM 决策**：OP 表示命令本身可完成任务（后续需回答性检查），MACRO_VLM 表示命令仅完成导航后仍需 VLM 继续交互。
- **着陆检查（Landing Check）**：重放后比对当前 UI hierarchy 与目标屏签名，不匹配则重启 App 并回退 VLM。
- **着陆提示（Landing Hint）**：向 VLM 注入目标屏自然语言描述与交互元素列表，使其避免冷启动。
- **STABLE / DYNAMIC 元素分类**：LLM 标注元素是否跨 App 版本保持稳定的语义标识，供路由模块筛选可靠条目。

## 可复现要素
- **数据集**：AndroidLab（公开）、AndroidWorld（公开）；论文未提供自建数据集。
- **代码**：Github Repo `https://github.com/HKUDS/OpenPhone`（论文已声明）。
- **权重**： backbone 模型均为 off-the-shelf（Qwen3.7-Plus、GLM-4.6V、Kimi-K3），无需微调。
- **关键超参**：地图构建 BFS 深度 ≤ 3、屏幕预算 ≤ 50、每屏最多滚动 3 页；VLM temperature=0，每调用最多 512 tokens；任务上限 25 轮。
- **环境**：Android 13 模拟器（Pixel 7 Pro AVD，1440×3120）。
- **评判**：94 任务用程序化 XML checker，44 任务用 LLM judge（统一 Qwen3.7-Plus 重评）；统计显著性 $p<0.05$。
