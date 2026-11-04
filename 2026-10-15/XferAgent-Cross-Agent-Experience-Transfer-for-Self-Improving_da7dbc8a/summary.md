---
title: "XferAgent-Cross-Agent-Experience-Transfer-for-Self-Improving"
source: https://arxiv.org/pdf/2610.19581v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-04 03:55:59"
field: "多智能体经验驱动自我改进"
keywords: ["cross-agent experience transfer", "LLM agents", "self-improving agents", "experience graph", "task-conditioned routing", "graph-based memory"]
innovations: ["跨代理经验对齐机制，通过相似度边际阈值避免不可靠合并", "任务条件经验路由结合个性化PageRank与跨代理结果统计构建连通子图", "本地-共享经验临时融合与探索增强排名实现任务相关经验选择"]
benchmarks: ["GAIA", "ALFWorld", "MBPP"]
---

# 论文速读：XferAgent-Cross-Agent-Experience-Transfer-for-Self-Improving

## 一句话总结
论文提出 XferAgent 框架，通过构建本地与共享经验图，实现异构 LLM 代理之间可重用的经验对齐、路由与整合，从而在重复任务中实现累积性的自我改进。

## 研究问题与动机
- 现有基于记忆的方法（如 Reflexion、ExpGraph 等）主要支持单个代理的本地经验复用，无法利用其他代理已习得的策略或失败教训。
- 跨代理经验共享面临异构执行历史、推理协议和工具使用的差异，导致经验难以直接对齐与合并，且并非所有共享经验都对新任务有用。
- 需要一种机制来协调来自不同代理的经验贡献，并在任务执行前根据任务条件选取最相关的经验子集。
- 经验驱动的自我改进应在多次任务迭代中体现累积性能提升，而不仅是一次性检索增强。

## 核心贡献（创新点）
- **跨代理经验转移的形式化与框架设计**：首次将跨代理经验转移系统化为本地/共享经验图协同运作的流程，区别于仅依赖单代理记忆的现有方法。
- **跨代理经验对齐（Reconciliation）**：通过稳定标识符与相似度比较合并兼容贡献，并在多候选时引入相似度边际阈值以避免不可靠合并，与 AgentKB 等直接聚合的方法形成对比。
- **任务条件经验路由与连通子图构建**：结合个性化 PageRank 图传播与跨代理结果统计、贡献者支持度，选择与任务相关且保持连通性的经验子图，区别于仅基于相似度排序的基线。
- **本地‑共享经验融合与探索增强排名**：在临时融合图中结合任务相似度和贝叶斯平滑的结果得分，并加入基于使用频率的探索奖励，区别于静态或仅依赖历史成功率的选择策略。
- **多基准实证与渐进自我改进证据**：在 GAIA、ALFWorld、MBPP 上均优于代表性记忆基线，并在 GAIA 验证集三轮迭代中展示累积成功率持续提升。

## 方法详解
- **问题设定**：N 个异构代理，每个代理维护本地经验图 $\mathcal{G}_i^{L,t}$，所有代理贡献的经验组织在共享经验图 $\mathcal{G}^{S,t}$ 中；节点包含经验文本 $e_v$、类型 $k_v\in\{\text{SUCCESS},\text{FAILURE}\}$、嵌入 $h_v=\phi(e_v)$、来源与结果统计，边表示语义相似度与类型关系。
- **跨代理经验对齐**：将执行结果经 schema-constrained mapper 转换为 SUCCESS/FAILURE 经验，贡献到共享图时剥离任务特定与执行敏感信息；若存在相同类型的候选且余弦相似度 $\ge \theta_{\text{rec}}$，则合并，多重候选时仅当最高相似度比次高值至少大 $\delta_{\text{rec}}$ 才合并，否则新建节点；合并时保留 DERIVED‑FROM 关系（固定权 1）。
- **任务条件经验路由**：目标代理本地计算任务与共享节点相似度 $a(x_t,v)=\cos(\phi(x_t),h_v)$，筛选高于 $\theta_{\text{match}}$ 的锚点；使用个性化 PageRank 传播相关性：$\mathbf{s}_i^{\text{graph},(\ell+1)}=\alpha\boldsymbol{\rho}_i(x_t)+(1-\alpha)\boldsymbol{A}^\top\mathbf{s}_i^{\text{graph},(\ell)}$；同时计算留一法跨代理结果得分 $s_{-i,v}^{\text{outcome}}=\frac{z_{-i,v}^S+\kappa u_0}{n_{-i,v}^S+\kappa}$ 与贡献者支持度，融合为路由得分 $s_{i,v}^{\text{route}}$ 并构建连通子图（预算 B）。
- **本地‑共享经验整合**：分别从本地图与路由子图中用 PPR 选取候选，构建临时融合图；匹配节点优先保留 richer 本地内容并保留共享标识；再次 PPR 后按 $s_{i,v}^{\text{select}}=(1-\lambda_u)\text{Norm}(a(x_t,v))+\lambda_u\text{Norm}(s_{i,v}^{\text{exp-fused}})$ 排序选择最终经验条目。
- **经验更新**：任务完成后，仅对被选经验更新使用次数 $n_v$ 与累积得分 $z_v$，并将本次执行的新经验写入本地图与共享图，建立 DERIVED‑FROM 关系。

## 实验与结果
- **数据集与骨干**：GAIA（301 任务，Qwen3‑Next‑80B‑A3B‑Instruct 与 GPT‑5.6 Luna）、ALFWorld（274 任务，Qwen3.8‑27B）、MBPP（60 代码任务，Llama‑3.1‑8B‑Instruct）；代理框架主要为 SmolAgents，跨框架实验引入 OWL。
- **基线**：No Memory、AgentKB、ReasoningBank、ExpGraph、Mem0、LightMem。
- **主要结果**：
  - GAIA（GPT‑5.6 Luna）：XferAgent 整体准确率 **52.16%**，较 No Memory（48.17%）提升 **+3.99pp**，较最强基线 ExpGraph（47.51%）提升 **+4.65pp**；Qwen 骨干下达 29.57%。
  - ALFWorld：成功率 **88.7%**，较 No Memory（84.7%）提升 **+4.0pp**，略超 AgentKB/ExpGraph（88.3%）。
  - MBPP：通过率 **53.3%**，较 No Memory/AgentKB（50.0%）提升 **+3.3pp**。
- **渐进自我改进**：GAIA 验证集 165 任务三轮迭代，累计准确率从 56.36% 升至 67.88% 再到 **74.55%**（Level 3 从 42.31% 升至 69.23%）。
- **跨框架转移**：2/4/8 个异构代理（SmolAgents/OWL）在 60 道保留题上宏平均准确率从 40.4% 升至 47.2% 再到 **55.0%**（Level 3 从 15.4% 升至 30.8%）。
- **消融**：移除共享经验下降至 46.84%（−5.32pp），移除结果排序至 48.17%（−3.99pp），移除连通约束至 46.84%（−5.32pp）。
- **超参敏感**：默认 B=24、K=3、$\lambda_u=0.40$、$\delta_{\text{rec}}=0.03$；将 K 增至 5 可达 62.22%（+4.44pp）。

## 相关工作脉络
- **经验驱动学习/结构化记忆**：Reflexion、ExpeL、ReasoningBank、A‑MEM、AriGraph、ExpGraph 等聚焦单代理经验组织与检索；XferAgent 在结构化管理基础上引入跨代理图协同与对齐。
- **跨代理经验共享**：AgentKB、FoT、FICAL 等允许经验共享，但缺乏对异构经验的可靠对齐机制与任务条件路由；本文以相似度边际阈值与连通子图解决该缺口。
- **图检索与传播**：ExpGraph 使用图检索与效用排名；本文进一步采用个性化 PageRank 与锚点传播构建任务相关的连通子图。
- **联邦/分布式经验演化**：如 FedAgentKE 侧重联邦语义知识演化；XferAgent 侧重执行轨迹抽取、经验对齐与局部‑共享融合的可重用架构。
- **多代理工作流**：MetaGPT 等关注多代理协作编排；本文关注异构代理在独立执行后的经验沉淀与复用。
- **内存管理**：Mem0、LightMem 侧重内存持久化与压缩；本文以图结构承载结果统计与DERIVED‑FROM 关系，支持更细粒度的任务条件选择。

## 局限性与未来方向
- 经验共享去除了任务特定信息，但仅限制信息而非提供形式化隐私保证。
- 当前仅支持文本型可重用经验，尚未扩展到多模态代理场景。
- 源任务与评估任务之间可能存在语义重叠或预训练暴露，影响泛化评估的纯净性。
- 未来可拓展至多模态执行经验、引入隐私保护机制，并在更动态开放环境中验证。

## 研究启发与可借鉴点
- **图结构组织异构经验**：以节点/边统一表示成功/失败经验及其语义与派生关系，便于跨代理对齐与传播，可迁移至其他多智能体经验库建设。
- **PPR 传播 + 连通子图路由**：在经验图上进行锚点启动的个性化 PageRank，并以连通性约束构建子图，保证所选经验在语义与结构上的连贯性。
- **结果统计的贝叶斯平滑与探索奖励**：结合使用计数与累积得分进行留一法平滑，并加入 $\sqrt{\log T/n}$ 形式的探索 bonus，平衡利用已知成功经验与探索低频经验。
- **本地‑共享临时融合图**：在不修改持久图结构的前提下构建临时融合图进行再排序，降低跨代理集成的工程复杂度。
- **可复用的对齐与派生关系**：稳定标识符匹配、相似度边际阈值与 DERIVED‑FROM 关系可被其他跨代理经验系统借鉴，提升合并可靠性与因果可追溯性。

## 关键术语表
- **Cross‑Agent Experience Transfer**：不同 LLM 代理之间沉淀与复用执行经验的过程。
- **Local/Shared Experience Graph**：分别存储单个代理自有经验与跨代理聚合经验的图结构，节点含类型、嵌入、统计，边含语义与派生关系。
- **Task‑Conditioned Experience Routing**：基于任务相似度、图传播与结果统计选择并构建连通经验子图的机制。
- **Local‑Shared Experience Integration**：将代理本地候选与跨代理路由候选在临时融合图中合并并再排序的过程。
- **Personalized PageRank (PPR)**：从指定锚点出发在加权图上传播 relevance 的迭代算法。
- **Cross‑Agent Outcome Score**：聚合其他代理对某经验的成败使用统计并经先验平滑得到的可用性分数。
- **Exploration Bonus**：依据经验使用次数给予的奖励项，鼓励对低使用经验的尝试。
- **DERIVED‑FROM Relation**：记录新经验由哪些已有共享经验派生的有向边，固定权值为 1。

## 可复现要素
- **数据集**：GAIA（公开）、ALFWorld（公开）、MBPP（公开）；论文使用了 GAIA 测试集与验证集子集。
- **代码/权重**：论文未明确声明开源；需以 arXiv 页面为准。
- **关键超参**：默认 B=24、K=3、$\lambda_u=0.40$、$\delta_{\text{rec}}=0.03$、PPR 重启概率 $\alpha=0.35$、结果先验强度 $\kappa=2$、探索权重 $c=0.25$、融合阈值 0.92、锚点匹配阈值 0.08 等（详见附录 C/D）。
