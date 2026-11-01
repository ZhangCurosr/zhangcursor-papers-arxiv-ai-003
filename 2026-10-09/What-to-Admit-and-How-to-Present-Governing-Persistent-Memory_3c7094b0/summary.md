---
title: "What-to-Admit-and-How-to-Present-Governing-Persistent-Memory"
source: https://arxiv.org/pdf/2610.11188v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-01 18:59:14"
---

# 论文速读：What-to-Admit-and-How-to-Present-Governing-Persistent-Memory

## 一句话总结
本文在推理时阶段将持久化记忆的治理解耦为“准入（Admission）”与“呈现（Presentation）”两个独立杠杆，提出无需重训练的因子编译准入（FC）与权限语义准入（PS）设计，并通过预注册控制实验证明：严格管控哪些记忆物理进入工作上下文可显著降低阿谀奉承与跨域泄露风险，但现有治理措辞尚无法有效化解记忆-证据冲突，且会附带增加个性化记忆漏用率。

## 研究问题与动机
1. **核心问题**：LLM Agent 的持久化记忆在提升个性化服务的同时，极易诱发阿谀奉承（sycophancy）与跨域泄露（cross-domain leakage）。当前管线通常将
