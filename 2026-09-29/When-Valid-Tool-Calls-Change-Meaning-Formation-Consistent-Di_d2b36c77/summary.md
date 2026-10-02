---
title: "When-Valid-Tool-Calls-Change-Meaning-Formation-Consistent-Di"
source: https://arxiv.org/pdf/2609.35088v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 08:16:48"
---

# 论文速读：When-Valid-Tool-Calls-Change-Meaning-Formation-Consistent-Di

## 一句话总结
本文针对MCP Agent中工具调用在形成到执行期间因部署更新、重连或延迟审批导致安全语义发生漂移（schema-epoch drift）的问题，提出Formation-Consistent Dispatch (FCD) 机制。FCD通过将描述符、处理程序与安全契约在形成期绑定，并在生命周期内原子化强制执行，确保待执行调用只能由形成时被授权的实现完成，从而保证调用意图的时序一致性。

## 研究问题与动机
1. **核心问题**：MCP协议中LLM生成工具调用（`tools/call`）与主机选择具体实现（binary/config/replica）是分离的。在此间隔中，滚动发布、版本回滚、重连或路由调整可能导致格式合法、参数未变的调用被路由到不同版本的处理程序，产生与形成时预期截然不同的安全效果（如私有仓库变公开、网络访问由关闭变开启）。
2. **现有方法不足**：现有Agent防御（ETDI、Attested admission、Tool Forge等）主要验证定义版本、参数Schema、服务器身份或执行胶囊，但未将“描述符–实现–安全效果”的绑定关系延伸到实际执行阶段；语义差分与变更影响分析也仅识别行为差异，无法转化为单次调用的执行授权。
3. **协议缺口**：MCP协议协商仅标识通信修订版本（communication revision），而非具体工具实现。模型无法验证实际执行使用的是否为赋予其调用含义的实现。
4. **威胁模型**：部署控制器或敌手可通过控制可用性、未认证服务加入、路由选择与重连时序，在模型固定调用字节后篡改解释器，且无需伪造主机凭证或篡改认证
