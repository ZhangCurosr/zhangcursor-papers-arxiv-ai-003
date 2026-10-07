---
title: "ScienceClaw-Benchmarking-Continual-Self-Evolution-of-AI-for"
source: https://arxiv.org/pdf/2610.08691v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-07 22:24:58"
---

# 论文速读：ScienceClaw: Benchmarking Continual Self-Evolution of AI-for-Science Agents Across the Natural and Social Sciences

## 一句话总结
ScienceClaw 提出了一种固定大模型参数下的科学智能体程序级持续自演进框架，将多轮交互中经验证的失败-成功轨迹转化为可复用的 Skill 与 Operator 更新；配套的 ScienceClaw-Eval 基准覆盖 23 个自然与社会科学学科，系统评测了科学正确性、演进增益、知识保留与跨数据集迁移能力。

## 研究问题与动机
1. **修复难以持久化**：现有科学智能体虽能完成单任务求解，但运行时的错误修复策略多为瞬时上下文，极少能沉淀为部署后可复用的程序级改进。
2. **评测缺乏连续性**：现有基准多聚焦静态问答或单次工具调用，未构建跨学科的序列任务流，无法衡量能力是否随任务推进持续累积。
3. **演进粒度偏碎片化**：训练免（training-free）类方法多局限于单一工具、Skill 或 Agent Harness 的优化，缺乏对“策略 Skill ↔ 可执行 Operator”联动更新的系统评测。
4. **核心问题**：如何在冻结基础模型参数（$\Theta_0$ 不变）的前提下，让经过科学验证的执行证据驱动持续、可迁移、防过拟合的程序级自演进？

## 核心贡献（创新点）
1. **形式化 ScienceClaw 任务**：首次将 AI
