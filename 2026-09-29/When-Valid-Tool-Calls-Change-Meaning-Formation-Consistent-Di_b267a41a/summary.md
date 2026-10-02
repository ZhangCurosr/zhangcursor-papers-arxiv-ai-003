---
title: "When-Valid-Tool-Calls-Change-Meaning-Formation-Consistent-Di"
source: https://arxiv.org/pdf/2609.35088v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 08:16:43"
---

# 论文速读：When-Valid-Tool-Calls-Change-Meaning-Formation-Consistent-Di

## 一句话总结
论文提出 Formation-Consistent Dispatch (FCD) 机制，解决 LLM Agent 在 MCP 协议下因“描述符–处理器”绑定关系在审批到执行间隙发生漂移（schema-epoch drift），导致相同且合法的 tool call 获得相反安全语义的问题；通过形成期目录快照绑定、调用级效应证明与生命周期原子准入，确保 pending call 仅在授权解释器上以原始契约执行。

## 研究问题与动机
- **核心问题**：MCP host 向模型暴露工具名称、描述与输入 schema，模型返回 name+arguments 后由 host 在稍后阶段选择具体二进制/配置/副本执行。两次决策之间存在审批延迟、重试、重连、滚动更新等间隙，标准 dispatch 未携带 descriptor–handler 绑定信息，使“不变且 schema 合法”的调用在 rollout 或 reconnect 后可被不同解释器赋予不同安全效应。
- **现有方法不足**：
  1. **Exact pinning** 虽能阻断漂移但过于保守，会强制终止所有 pending call 并要求重新审批，破坏长生命周期 Agent 工作流。
  2. **ETDI** 仅锁定定义版本与 hash，未覆盖同一调用在具体 runtime 被不同实现解释出的 effect 差异。
  3. **语义差分/变更影响分析** 只能识别跨版本行为差异，无法将其转化为单次调用的执行授权并贯穿部署生命周期。
  4. **既有 TOCTOU 工作** 多聚焦分离调用间的环境变化，未解决单次 formation-to-execution 区间内解释器内部语义漂移问题。

## 核心贡献（创新点）
1. **形式化定义 schema-epoch drift 并跨官方 release 与重连路径实证其普遍性**：通过 GitHub、Azure、Reference Git、DBHub 等 32 个官方发版与编排回滚实验，首次系统展示即使 descriptor 未变，相同省略参数调用也可在 v1.3→v1.4 或 beta.31→beta.32 中从 private 翻转为 public。
2. **提出 FCD 框架，将效应兼容性、调用授权与生命周期可用性解耦**：三者仅在交集中允许执行；严格遵循 closed-target 策略，禁止后续 rollout 或策略变更扩展现有 pending call 的授权集合。
3. **设计 profile-guided call-specific evidence 管线**：基于官方源码分析生成带 provenance 的 over-approximation effect summary，仅当具体调用的安全契约 $C_i$ 完全包含目标摘要时才颁发证书并纳入 $A_i^0$，实现调用粒度而非 release 粒度的授权。
4. **实现原子准入事务与终跳围栏（atomic admission & final-hop fence）**：结合 SQLite WAL/etcd 事务与状态机（ADVERTISED→HIDDEN_BUT_VALID→DRAINING→STALE），保证准入决策与退役顺序强一致，防止未退役源处理器触发错误回退。

## 方法详解
- **权限不变式与 closed-target 策略**：对调用 $i$，固定初始授权集 $A_i^0 = \{e_s\} \cup S_i$（$e_s$ 为源处理器，$S_i$ 为 formation 时捕获的至多一个 successor）。任意时刻 $t$ 的剩余权限单调收缩：$t_2 \geq t_1 \Rightarrow A_i(t_2) \subseteq A_i(t_1) \subseteq A_i^0$，且所有受保护 effect 必须满足 $\mathsf{effect}(i, e, t) \Rightarrow e \in A_i(t)$；非源 effect 仅在源达到 STALE 后才允许：$\mathsf{effect}(i, e, t) \wedge e \neq e_s \Rightarrow e \in S_i \wedge \mathsf{state}(e_s, t) = \mathrm{STALE}$。
- **目录快照与认证上下文**：host 从通告 epoch 获取 descriptors 并签发绑定 service scope、route、tool name、descriptor digest、artifact identity、expiry 的凭证；模型可见侧剥离私有扩展字段，但 `tools/call.params._meta` 携带凭证引用，gateway 在分发前验证而非查找最新 credential，避免 relist 替换 provenance。
- **Profile-guided 调用级证据生成**：四大语言前端（Python AST、JavaScript SQL、Go HTTP、C# Cloud）针对特定 handler 路径提取 callee 身份并生成效应规则集；formation 时以契约 $C_i$ 校验目标摘要，满足 $\mathsf{ActualInScopeEffects}(e_t, i) \subseteq S(e_t, i) \subseteq C_i$ 才颁发 certificate 并加入 $A_i^0$，否则进入 UNKNOWN/escalation 或需 manual review。
- **原子准入与生命周期管控**：etcd/SQLite WAL 事务将凭证验证、生命周期检查、租约创建原子化；`DRAINING` 禁止新租约，`STALE` 要求租约前缀为空；并发准入要么在 drain 线性化点前提交，要么观察到转换后拒绝，保证 lease 优先权。
- **终跳围栏（Final-hop fence）**：在 effect 产生前再次校验 artifact、configuration 与 single-use generation token，防止注册后的配置漂移或镜像替换绕过审核，支持与 backend-native guard 或 authenticated sidecar 组合。

## 实验与结果
- **数据集与基线**：4 个 reviewed profiles 覆盖 32 个官方 release（GitHub MCP Server v1.3.0/v1.4.0、Azure MCP Server beta.31/beta.32、Reference Git 2025.11.25/2025.12.18、DBHub 0.22.2/0.22.6）；基线对比 exact pinning、release-wide deny/allow、execution-time adoption。
- **主要结果**：
  - 29/32 个 release 无需修改 profile 自动复用，仅 3 个因 classifier 跨模块边界而 escalation；冻结扩展中 16 个 holdout release 全部匹配独立 source oracle。
  - 预注册对比：FCD 完成 3/3 个安全 pending call（保持 private / disabled-network 意图），拦截 3/3 个语义反转 call；exact pinning 全拦截（0/6）；release-wide allow 放行全部 6/6 但产生 3 次公开效应违规。
  - 模型侧透明性：GPT、Nova Lite、Gemini Flash Lite 在 FCD/plain 下的 accepted-call 数分别为 64/63、65/64、74/74，McNemar 检验 p=1.00，证明 hidden metadata 不干扰模型决策。
  - 运行时开销：本地绑定 +0.030 ms，终跳围栏 +0.321/0.431 ms（p50/p95）；高并发（64）优化版 etcd 吞吐 256.72 req/s，p95=248.42 ms，goodput 达 TLS 基准 62.7%。
-
