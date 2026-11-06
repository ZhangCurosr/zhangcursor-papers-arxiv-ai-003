---
title: "What-Does-Now-Mean-to-Your-Computer-Use-Agent-Stale-screensh"
source: https://arxiv.org/pdf/2610.39282v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-06 17:15:44"
field: "Computer-use agent 感知与时序一致性"
keywords: ["computer-use agents", "GUI agents", "vision-language models", "preflight check", "stale screenshots", "browser automation", "benchmark"]
innovations: ["首次在大样本真实网页上量化 stale click 并定位风险在页面加载而非规划延迟", "像素比对与小/前沿 VLM 在三类点击前问题上的直接公平对比", "基于 C/δ/r/f/k 的检查盈亏公式及其 94.7% held-out 一致率验证"]
benchmarks: ["31-page real-page stale-click trace", "HTTP Archive desktop crawl (booking/gov/ecommerce)", "6-synthetic-page benchmark (4608 click pairs + 3600 action-outcome pairs)"]
---

# 论文速读：What-Does-Now-Mean-to-Your-Computer-Use-Agent-Stale-screensh

## 一句话总结
本文首次在大规模真实网页上量化了 computer-use agent "截图规划→坐标点击"流程中 stale click 的发生频率与根源（主要是页面仍在加载，而非规划延迟本身），并系统对比了像素比对、OCR 与小/前沿 VLM 三种"点击前校验"感知方式的代价-收益，最终给出三条可落地的 agent 开发者规则。

## 研究问题与动机
- **核心问题**：agent 在截屏后到实际点击之间通常经历 1–13 s 的规划期，期间若页面发生变化，原坐标点击会落到错误元素上；这一风险的真实来源究竟是规划延迟，还是页面加载过程中的持续变化，此前从未实测。
- **现有方法不足**：PUSV 等安全型方案在任意变化时都直接 halt，未评估普通网页变化的基线频率；AOI / 预编译策略树等只关注缩短规划间隔，未对"等待多久再截屏"的系统规则做对比；浏览器测试框架（如 Playwright）有 actionability 校验但未被 agent 社区引用。
- **三个 RQ**：
  1. RQ1：截屏在何种时点之后失效（规划期内？加载期内？）
  2. RQ2：哪类感知手段适合回答"目标是否还在原位 / 是否被遮挡 / 上一步是否成功"
  3. RQ3：一次检查的代价何时能被其拦截的收益覆盖

## 核心贡献（创新点）
1. **首次在大样本真实网页上测量 stale click 基线**：31 个公开页面上，完全加载后 30 s 内仅 0.4% 的点击失效；而在页面打开首秒内高达 17%——风险集中在页面加载，而非规划延迟。
2. **十种开源 agent 的"导航后等待规则"对照评测**：重构其在官方仓库中的默认规则并应用到同一批页面，发现 8/10 留下 >1% 的 stale 点击，只有 Skyvern 接近网络空闲规则。
3. **像素比对 vs. VLM 的直接公平对比**：针对"目标是否仍在位"这一精确位置问题，312 ms 的像素比对召回 82%、误报 4%，而 Claude Opus 5.5 在 2.4 s 下仅 83% 召回、12% 误报；但针对"上一步是否成功"这一内容问题，4B 模型召回 100% 且零误报，与 Opus 持平。
4. **给出"检查是否值得"的盈亏公式与实证验证**：Equation (1) 基于规划时长 C、检查延迟 δ、召回 r、误报率 f 给出临界概率 p*，在 3000 个 held-out 决策上 94.7% 的 check/skip 判断与回放一致。
5. **开源 benchmark generator 与 break-even 公式**：六张可人工注入事件的合成页面、4608 个 click pair + 3600 个 action-outcome pair，所有阈值在看标签前冻结。

## 方法详解
- **预飞行检查（preflight）的三类问题**：① 目标是否仍在计划位置；② 是否有弹窗/对话框遮挡；③ 上一步动作是否成功。前两类是"少量像素"的位置判断，第三类是"屏幕现在报告什么"的内容判断，因此需要不同感知手段。
- **像素检查（pixel check）**：在规划时刻连续采集两张相隔 200 ms 的截图，把独立于 agent 的动态像素（spinner、闪烁光标、ticker）排除；在当前屏幕上以目标位置为中心、±40 px 范围内匹配计划 patch，要求最佳匹配与次佳 lookalike 之间留有清晰裕度；额外检测目标被覆盖 / 出现对话框。输出 proceed 或 re-plan，全程 312 ms，无模型调用。
- **模型检查**：将问题发给 Qwen3-VL 2B/4B、SmolVLM2 2.2B、Florence-2 OCR、Claude Haiku 5.5、Claude Opus 5.5；输入为"计划截图的目标附近 crop + 当前全图 + 问题"，第三类问题额外提供动作前后两张截图与一句预期效果描述。
- **PUSV 复现**：按原文实现并对齐到 2% 误报率，评估其在普通页面上的表现。
- **盈亏公式（Equation 1）**：设 p 为目标在点击前变化的先验概率，C 为规划时长，δ 为检查延迟，r 为召回，f 为误报率，k 为一次 stale click 的成本（普通点击 k=2，支付/删除 k 更高），则检查更优当且仅当
  \[
  p > p^* = \frac{\delta/C + f}{r(k-1) + f}.
  \]
  推论：规划越快 p* 越高（因为浪费更少）；操作风险越高 p* 越低；过慢的检查反而延长间隔。Equation (2) 用于在同类问题的两种检查间做选择。
- **实验设置**：31 个公开页面（时钟、地图、天气、状态页等）每页最多跟踪 20 个可见链接/按钮共 30 s，从"完全加载 / 滚动后 / DOM-ready 瞬间"三种初始态出发，三分轮次共约 1300 个 tracked target/态；HTTP Archive 10 月桌面爬行数据覆盖 65 个预订/购物主站、4972 个电商软件站、3028 个 .gov 首页；合成 benchmark 6 张页面 × 4 种布局，注入 7 类事件（无变化 / 目标移动 / 相似物移动 / 目标被覆盖 / 目标消失 / 弹窗 / 慢加载）。

## 实验与结果
- **RQ1 页加载风险集中**：完全加载后 0.5–30 s 内 stale 率稳定在 0.4%；DOM-ready 后 0.5 s 内达 14.5%，1 s 后达到 17% 并封顶（布局在 ~1 s 内一次性完成）。
- **等待规则对照（Table 1）**：唯一低于 1% 的是 Playwright 的 `networkidle`（500 ms 无请求），中位等待 1.24 s，stale 率 0.46%；固定 0.5/1 s 延迟、`load` 事件、布局稳定 0.5 s 均停留在 2.5–3.1%。十种 agent 中仅 Skyvern（load→1s→animation）接近；OpenAI CUA / Stagehand 无等待，stale 率高达 15.4%/12.5%。
- **商业页加载更久**：经 3 s 封顶的 networkidle 等待后，52.3% 的预订/购物首页在随后 6.5 s 内仍发生布局偏移；.gov 页仅 7.8%（差值 44.5 个百分点，95% CI 32.4–56.9）。15.4% 的预订页在 3 s 内未达网络空闲，.gov 页为 63.4%。
- **RQ2 感知手段分工（Table 3）**：
  - "目标仍在位"：pixel 82%/4% / 312 ms；Opus 83%/12% / 2366 ms；Haiku 召回 98% 但误报 95%；4B 66%/37%。
  - "弹窗遮挡"：pixel 72%/1%；Opus 100%/7%；4B 95%/14%。
  - "上一步失败"：4B 100%/0%；Opus 100%/0%；Florence-2 OCR 89%/0%；pixel change score 90%/0%（但只能给出分数，需阈值判定）。
- **RQ3 盈亏公式**：在 3000 个 held-out 决策上 check/skip 判断一致率达 94.7%；Equation (2) 在 93% 情况下选出成本更低的检查。对于 6.5 s 规划器，pixel 检查在 stale 率 >≈10% 时划算，Opus 在任何 ≤50% 的率下均不划算。
- **活体 agent loop**：Haiku 规划器 + pixel 检查，stale click 从 65–69% 降至 9–11%；581/606 次 re-plan 均为合理触发。若按 6.5 s 计费，每步开销增加 28–105%（超 10% 阈值）；若按实测 1.1 s 计费，每步净节省 0.2–0.3 s。
- **失败后不应调整原坐标**：任何像素设置都无法把错误调整率压到 1% 以下（最好 11%）；让 Haiku 重定位元素的错误率为 34%；应直接 re-plan。

## 相关工作脉络
- **PUSV [1]**：测量到 6.5 s 的规划间隔并提出像素再校验，但面向对抗式威胁模型，在普通页面上误报 21%，本文定位其"安全型 check"在常规场景的代价。
- **AOI [2] / 预编译策略树 [5]**：通过更快执行缩短间隔；本文指出即使间隔缩短，加载期风险仍占主导。
- **BlindTest [8] / DiffSpot [9] / VisCritic [13]**：记录 VLM 在精细空间判断上的局限；本文首次把 VLM 与 pixel 方法在同一组 click 问题上并排对比。
- **Belcak et al. [14]**：主张 agent 用小模型；本文从"位置 vs. 内容"的问题分层给出小模型的适用边界。
- **机器人延迟行动 [10, 15–17]**：40–400 ms 尺度上用运动预测补偿推理延迟；网页无等效运动模型，因此"调整原坐标"在此失效。
- **浏览器测试框架 [4]**：Playwright/MCP 的 actionability/wait-for 机制；本文把它们的等待语义移植到 agent 截屏时机上做了量化评测。
- **鲁棒性基准 WAREX [18] / StressWeb [19] / DynamicGUIBench [20] / WebDecept [21]**：分别注入网络延迟、布局偏移、界面变化、欺骗弹窗；但未测量"截图到点击"的真实 stale 基线，本文填补该空白。

## 局限性与未来方向
- 真实页面 stale 率极低（0.4%），合成页约 70% 用作 check 压力测试，两者互补但不可相互替换。
- 真实页仅 31 个公开只读页、单一浏览器、三天内访问；商业页来自 HTTP Archive 静态爬行，未实际登录或进入购物车/结算流程。
- 商业页的 crawl shift 信号与实测 stale 率呈负相关（ρ=−0.12），无法跨站点推算 stale 率，需后续标定。
- 移动端数据仅在 41% 预订页可达网络空闲，差值缩窄至 12.9 个百分点，结论偏桌面端。
- 4B 模型"100% 召回、0 误报"可能是失败集相对容易所致，待更大基准验证。
- 仅用单一规划器（Haiku），且其自身 ~70% 错误源于任务生成器产出的相似标签，preflight 无法覆盖此类根因。
- 未来方向：测量"action 前帧"由固定规则采样的训练数据中有多少落在加载期内；扩展到需要鉴权的连续更新应用；研究多模态训练 pipeline 的时序对齐。

## 研究启发与可借鉴点
1. **问题分层决定感知手段**：位置精确定位用像素、内容判断用模型——这一二分思路可直接迁移到其他 agent 子任务（如 anchor 拾取 vs. 表单验证）。
2. **盈亏公式的工程价值**：Equation (1)(2) 仅依赖 C、δ、r、f、k 五个可测参数即可做 check/skip 决策，可在 agent 运行时动态评估，无需重新训练。
3. **合成 stress-test + 真实 base-rate 双轨实验设计**：用可精确注入事件的合成页测 recall/false-alarm，用真实页测先验 p，两者合在一起给出更稳健的结论，值得在 GUI agent 评测中推广。
4. **"预注册 + 全量报告（含被证伪假设）"**：14 条假设 6 支持 8 被拒全部公开，增强结论可信度；该方法学本身是benchmark 社区的可借鉴实践。
5. **可复用的 benchmark generator 与 Three Rules 模板**：开源代码可作为 agent 框架的参考实现；三条规则（等网络空闲、按问题选检查、失败后 re-plan 不调整）可直接写入 agent 开发 checklist。

## 关键术语表
- **Stale click**：agent 按计划坐标点击时，目标元素已移动、消失或被遮挡，导致点击落到错误元素上。
- **Preflight check（预飞行检查）**：在 agent 即将执行点击前运行的一次感知校验，回答"目标是否仍在 / 是否被遮挡 / 上一步是否成功"三问之一。
- **Network idle**：网页在最近 500 ms（或作者阈值）内未发出任何网络请求；本文证明这是唯一能把 stale 率压到 <1% 的截屏等待规则。
- **DOM-ready**：浏览器解析完 HTML 的时刻，通常早于图片、字体、异步数据的到达；也是 stale click 风险最高的起始点。
- **PUSV**：一种针对对抗式 TOCTOU 攻击的像素再校验方案，普通网页上误报率高达 21%。
- **Break-even formula**：判断 preflight 检查是否划算的临界概率公式（Equation 1），仅依赖规划延迟、检查延迟、召回与误报率。
- **Live loop**：用真实规划器（Haiku 5.5）在合成移动页面上跑 400 步、有/无检查对照，用于评估端到端效果。
- **Shifted page**：截屏后 6.5 s 内累计布局偏移 >0.0001 的页面，用作商业页不稳定性的 proxy 指标。

## 可复现要素
- **数据集**：31 个公开只读网页（仅用于测量 stale 基线，截图因含第三方内容未公开）；HTTP Archive 10 月桌面爬行（public，BigQuery 可用）；6 张合成页面 + 4608 个 click pair + 3600 个 action-outcome pair（全部开源）。
- **代码/权重**：benchmark generator、所有检查实现、爬取查询、预注册原始数据与分析报告均开源 → https://github.com/multimodal-ai-engineering/preflight-bench；VLM 权重（Qwen3-VL、SmolVLM2、Florence-2）为公开模型；云端模型用 Claude Haiku/Opus 5.5 按 API 调用。
- **关键超参**：像素匹配范围 ±40 px；动态像素排除窗口 200 ms；网络空闲阈值 500 ms、封顶 3 s；布局偏移判定阈值 0.0001；合成页规划-点击间隔固定 6.5 s；Equation (1) 中 k=2（普通点击）/ k=10（ irreversible 点击）；阈值在未见 held-out 标签前冻结。
