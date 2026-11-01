---
title: "Using-Weisfeiler-Leman-Features-for-Algorithm-Selection-in-C"
source: https://arxiv.org/pdf/2610.12119v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-01 18:57:29"
---

# 论文速读：Using-Weisfeiler-Leman-Features-for-Algorithm-Selection-in-C

## 一句话总结
本文提出了一种基于 1-WL 颜色精炼的轻量级图特征提取方法，将 FlatZinc 约束模型转换为有向图后自动生成结构直方图特征，填补了传统算法选择中“结构感知”与“经典 ML 兼容性”之间的空白，并在 MiniZinc Challenge 数据集上验证了其对 SVM 基线的显著提升。

## 研究问题与动机
1. **传统特征的结构盲区**：CP 算法选择主流工具 `fzn2feat` 仅依赖手工统计量（变量/约束计数、域大小、度数分布等），无法捕获实例的底层图结构与复杂子式关联。
2. **GNN 的工程代价过高**：图神经网络虽能学习结构嵌入，但需要大量标注数据、训练开销大且推理延迟高，不满足算法选择场景对“特征提取与预测必须轻量”的硬性要求。
3. **标准 WL 的度数敏感缺陷**：直接将 WL 应用于约束图时，相同类型的全局约束（如 `all_different([v1,v2])` 与 `all_different([v1,v2,v3])`）因入度不同会被染成完全无关的颜色，丢失了语义相似性。
4. **缺乏面向经典 ML 的结构特征桥梁**：现有工作要么依赖手工特征，要么耦合深度模型，缺少一种可直接对接 SVM/RF/MLP 且理论表达力受 1-WL 有界保证的自动化特征管线。

## 核心贡献（创新点）
1. **FlatZinc 到有向约束图的标准化转换流水线**：将约束模型解构为 Variable/Parameter/Operator/Constraint/Solution 五类节点及 Lead/Trailing/Global 三种有向边，实现与经典 ML 及深度学习范式的无缝对接。本质上区别于仅输出平面统计量的 `fzn2feat`，保留了约束间的逻辑拓扑与运算依赖。
2. **Cut-based 结构表征（WLc）**：针对变长全局约束设计“切分节点”机制，通过集合聚合（set）替代多重集聚合，使同构约束在不同参数量下获得一致颜色，并用配对直方图补偿度数信息。本质上突破了标准 1-WL 对节点度数的敏感性，实现了“结构同构感知”与“规模计数”的信号解耦。
3. **无训练的 WL 图特征提取协议**：利用 1-WL 迭代颜色精炼生成固定维度直方图向量，规避了 GNN 的训练与推理开销，理论表达力受 Xu et al. (2018) 证明的 MP-GNN 上界保护。本质上与深度图嵌入方法形成互补，专为低资源/低延迟的算法选择场景设计。
4. **双任务系统级实证验证**：在 2023–2025 MiniZinc Challenge 的 300 个实例上构建 Borda 积分最大化与预测准确率最大化任务，完整覆盖 SVM/RF/MLP 三类范式与嵌套交叉验证协议。本质上为 CP 算法选择的特征对比提供了高可信度、可复现的评估基线。

## 方法详解
- **图转换设计**：从 FlatZinc 抽取 5 类节点与 3 种有向边。边类型区分操作数顺序（Lead 用于无序或左侧操作数，Trailing 用于右侧操作数，Global 专连全局约束）。主动剔除值域节点（Domain Values），避免如 `cgt 12h27` 实例单变量域跨度达 360 万导致的图规模爆炸。
- **基础 WL 特征提取**：初始化颜色 $c^{(0)}(v) = \text{HASH}(\text{type}(v) \parallel \text{subtype}(v))$。对每个非切分节点 $v$，第 $t$ 轮聚合入边邻居颜色多重集 $M(v) \leftarrow \Downarrow\{c^{(t-1)}(w) \mid (w,v)\in E\}$，更新 $c^{(t)}(v) \leftarrow \text{HASH}(c^{(t-1)}(v), \text{sort}(M(v)))$。本文仅取最后一轮直方图（附录消融实验证明concat全部轮次无统计显著提升）。
- **Cut 机制（WLc）**：
  1. 切分节点在迭代阶段跳过颜色更新（$c^{(t)}(v) \leftarrow c^{(t-1)}(v)$），但继续向后传色。
  2. 迭代结束后，切分节点颜色改用**集合**聚合初始邻居类型：$\bar{c}^{(\text{final})}(v) \leftarrow \text{HASH}(c^{(T)}(v), \text{sort}(S(v)))$，其中 $S(v) = \{c^{(0)}(w)
