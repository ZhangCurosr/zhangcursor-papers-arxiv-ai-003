---
title: "XferAgent-Cross-Agent-Experience-Transfer-for-Self-Improving"
source: https://arxiv.org/pdf/2610.19581v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-04 03:55:55"
field: "多智能体经验学习与知识共享"
keywords: ["LLM Agents", "Experience Transfer", "Cross-Agent Learning", "Memory-Augmented Reasoning", "Graph-Based Retrieval", "Self-Improving Agents"]
innovations: ["跨智能体经验调和机制，通过语义相似度与类型匹配对齐异构贡献并避免不可靠合并", "任务条件 PPR 路由，结合图传播与跨智能体历史结果统计构建连通经验子图", "本地-共享经验融合排序，贝叶斯平滑融合本地观察与跨 Agent 先验结果"]
benchmarks: ["GAIA", "ALFWorld", "MBPP"]
---

# 论文速读：XferAgent-Cross-Agent-Experience-Transfer-for-Self-Improving

## 一句话总结
XferAgent 提出了一种跨智能体经验迁移框架，通过本地与共享经验图的结构化组织、经验对齐融合以及任务条件路由机制，实现异构 LLM Agent 间有效复用成功经验与失败教训，在多任务基准上显著优于现有记忆基线方法。

## 研究问题与动机
- **现有方法的局限性**：Reflexion、A-MEM、ExpGraph 等仅支持单智能体本地经验复用，无法利用其他智能体已获取的相关策略或失败教训。
- **跨智能体迁移困难**：Agent KB、FoT 等跨智能体共享方法中，不同 Agent 的推理协议、工具集和执行历史存在差异，导致共享经验难以直接对齐和有效利用。
- **核心挑战**：如何在异构 Agent 贡献的经验中识别兼容内容、避免不可靠合并，并为新任务筛选出与目标 Agent 本地经验互补的共享经验。

## 核心贡献（创新点）
1. **形式化跨智能体经验迁移问题并提出 XferAgent 框架**——将经验组织为本地与共享两张图，通过经验对齐、路由和融合实现异构 Agent 间的协作式经验复用。
2. **跨智能体经验调和（Cross-Agent Experience Reconciliation）**——基于语义相似度与类型匹配将贡献经验并入共享图，引入相似度余量 $\delta_{\mathrm{rec}}$ 避免低置信度合并。
3. **任务条件经验路由（Task-Conditioned Experience Routing）**——结合 Personalized PageRank 图传播与跨智能体历史结果统计，构建连通的经验子图，兼顾语义相关性与实证有效性。
4. **本地-共享经验融合与最终排序**——构建临时融合图并执行 PPR，利用贝叶斯式平滑融合本地观察与跨智能体先验结果，以探索奖励增强选择得分。

## 方法详解
- **经验表示**：每个节点包含经验文本 $e_v$、类型 $k_v \in \{\text{SUCCESS, FAILURE}\}$、嵌入 $h_v = \phi(e_v)$、来源及结果统计；边由余弦相似度决定。
- **跨智能体经验调和**：对贡献经验 $v$，候选集 $\mathcal{C}(v) = \{u \in \mathcal{V}^S : k_u = k_v, \cos(h_u, h_v) \geq \theta_{\mathrm{rec}}\}$；若存在唯一候选则合并，多个候选时仅当最高分超出次高分至少 $\delta_{\mathrm{rec}}$ 才合并，否则新建节点。
- **任务条件路由**：Agent 本地计算任务与共享节点相似度作为 PPR 初始分布 $\rho_i(x_t)$，迭代更新 $s_i^{\mathrm{graph}, (\ell+1)} = \alpha \rho_i(x_t) + (1-\alpha) A^\top s_i^{\mathrm{graph}, (\ell)}$；同时计算留一法跨智能体结果分 $s_{-i,v}^{\mathrm{outcome}} = \frac{z_{-i,v}^S + \kappa u_0}{n_{-i,v}^S + \kappa}$，结合探索奖励与贡献者支持分得到路由分 $s_{i,v}^{\mathrm{route}}$。
- **本地-共享融合**：构建临时融合图，合并匹配节点（保留本地 richer 内容），应用 PPR 后融合得分 $s_{i,v}^{\mathrm{fused}} = \frac{z_{iv}^{L,t-1} + \kappa s_{-i,v}^{\mathrm{outcome}}}{n_{iv}^{L,t-1} + \kappa}$，最终选择分 $s_{i,v}^{\mathrm{select}} = (1-\lambda_u)\mathrm{Norm}(a(x_t,v)) + \lambda_u \mathrm{Norm}(s_{i,v}^{\mathrm{exp-fused}})$。
- **经验更新**：任务评估后更新被选中经验的使用计数与累积结果分，并将新执行转化为 SUCCESS/FAILURE 经验插入本地与共享图。

## 实验与结果
- **数据集**：GAIA（301 测试任务，QA）、ALFWorld（274 任务，交互执行）、MBPP（60 代码生成任务）。
- **基线**：No Memory、AgentKB、ReasoningBank、ExpGraph、Mem0、LightMem。
- **主要结果**：GAIA 上 XferAgent 用 GPT-5.6 Luna 达 52.16%（较 No Memory +3.99pp，较最强基线 ExpGraph +4.65pp）；ALFWorld 达 88.7%（+4.0pp）；MBPP 达 53.3%（+3.3pp）。
- **渐进自改进**：GAIA 验证集三轮后累计准确率从 56.36% 升至 74.55%。
- **跨框架迁移**：SmolAgents 与 OWL 异构 Agent 协作，8 Agent 配置下平均准确率 55.0%（Level 3 从 15.4% 提升至 30.8%）。
- **消融**：移除本地经验 (-5.32pp)、移除结果排序 (-3.99pp)、移除连通约束 (-5.32pp)。
- **敏感性**：$K=5$ 时验证集准确率最高 62.22%，$\lambda_u=0.40$ 为最优平衡点。

## 相关工作脉络
1. **单智能体经验记忆**：Reflexion、ExpL、ReasoningBank、A-MEM、ExpGraph——聚焦单 Agent 内部经验存储与检索，无跨 Agent 共享机制。
2. **跨智能体知识共享**：Agent KB、FICAL、FoT——聚合多 Agent 经验，但未解决异构经验对齐与任务条件筛选问题。
3. **XferAgent 的定位差异**：引入图结构经验调和、任务条件 PPR 路由、本地-共享融合排序，实现可复用、可扩展的跨 Agent 协作学习。

## 局限性与未来方向
- 隐私保护有限：虽去除任务特异性信息，但未提供形式化隐私保证。
- 仅支持文本经验：未扩展至多模态 Agent 场景。
- 源任务与评估任务可能存在预训练语义重叠，未完全排除数据泄露风险。
- 未来方向包括：扩展至多模态环境、引入形式化隐私机制、探索更大规模异构 Agent 协作生态。

## 研究启发与可借鉴点
1. **图结构经验组织**：用带权语义边连接相关经验，配合 PPR 传播任务相关性，可迁移至任何需要结构化记忆检索的 Agent 系统。
2. **留一法跨智能体结果估计**：用其他 Agent 的历史表现作为本地经验不足的平滑先验，有效缓解冷启动问题。
3. **经验调和的相似度余量策略**：通过 $\delta_{\mathrm{rec}}$ 避免低置信合并，可在分布式知识融合场景中通用。
4. **探索奖励与结果加权融合**：结合使用频率的 UCB 式探索项与贝叶斯结果平滑，可用于在线经验选择优化。

## 关键术语表
- **Cross-Agent Experience Reconciliation**：将异构 Agent 贡献的经验通过类型匹配与语义相似度对齐，避免不可靠合并并并入共享经验图。
- **Task-Conditioned Experience Routing**：基于任务相似度锚点与 PPR 图传播，从共享图中筛选与当前任务相关的连通经验子图。
- **Local-Shared Experience Integration**：将目标 Agent 本地经验与路由得到的共享经验在临时融合图中对齐合并，并结合任务相似度和历史结果进行最终排序选择。
- **Personalized PageRank (PPR)**：以任务锚点为初始分布，在经验图上进行随机游走传播，量化各节点与任务的图相关分数。
- **Leave-One-Agent-Out (LOAO) Outcome Score**：排除目标 Agent 自身历史，用其他 Agent 的使用计数与结果累积值估计跨智能体经验有效性。
- **Experience Graph**：由经验节点（SUCCESS/FAILURE）与语义边构成的图结构，用于组织可复用的执行经验。
- **Derived-From Relation**：从新贡献经验指向任务提示中已使用的共享经验的有向关系，标记经验的来源链路。
- **Exploration Bonus**：基于经验使用频次的 UCB 式奖励项，鼓励探索使用较少但可能有效的经验节点。

## 可复现要素
- **数据集**：GAIA、ALFWorld、MBPP（均为公开基准）。
- **代码/权重**：论文未明确声明开源；附录提供详细算法伪代码与超参设置。
- **关键超参**：子图预算 $B=24$（GAIA）、最终经验数 $K=3$、路由权重 $\lambda_u=0.40$、调和余量 $\delta_{\mathrm{rec}}=0.03$、PPR 重启概率 $\alpha=0.35$、结果先验强度 $\kappa=2$、探索权重 $c=0.25$。
