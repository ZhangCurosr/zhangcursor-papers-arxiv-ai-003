---
title: "TAEC-TRAJECTORY-AWARE-EVIDENCE-COORDINA-TION-FOR-MULTI-STEP"
source: https://arxiv.org/pdf/2609.37349v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 19:58:40"
field: "多模态检索增强生成"
keywords: ["视觉RAG", "多步推理", "证据管理", "轨迹协调", "无需训练", "多模态"]
innovations: ["提出轨迹级证据利用率退化现象并实证验证", "通过共享未解决需求状态统一协调证据准入/记忆暴露/视觉细节分配", "无需训练的轻量协调层在多个商业VLM上取得最佳性能"]
benchmarks: ["ViDoSeek", "SlideVQA", "MMLongBench-Doc"]
---

# 论文速读：TAEC-TRAJECTORY-AWARE-EVIDENCE-COORDINA-TION-FOR-MULTI-STEP

## 一句话总结
本文提出 TAEC（Trajectory-Aware Evidence Coordination），一种无需训练的协调层框架，通过维护共享的"未解决答案需求状态"，在多步视觉 RAG 中统一协调证据准入、记忆暴露和视觉细节分配三个决策，有效缓解了长推理轨迹中的证据利用率退化问题。在 ViDoSeek、SlideVQA 和 MMLongBench-Doc 三个基准上，TAEC 在多个商业视觉语言模型上取得了最佳平均准确率。

## 研究问题与动机
- **多步视觉 RAG 的证据可用性≠有效性**：论文分析 ReAct 在 ViDoSeek 上的轨迹发现，虽在涉及多次检索的轨迹中 74.6% 都检索到了标注证据页，但随着搜索次数增加，准确率从两次搜索的 91.7% 降至九次以上的 52.0%，错误率从 7.9% 升至 28.3%。
- **冗余来源占用上下文容量**：随着推理进行，重复或冗余的视觉源占据有限上下文，挤压了补充关键证据的空间。
- **已解决问题相关的观察残留在上下文中**：与已解决子问题或无效搜索分支相关的信息持续存在于上下文，与后续推理所需信息竞争注意力。
- **视觉源的细粒度阅读不足**：被保留的视觉源可能以不够细致的分辨率呈现，无法满足模型当前信息需求。
- 作者将这一现象统称为**轨迹级证据利用率退化（trajectory-level evidence utilization degradation）**。

## 核心贡献（创新点）
1. **首次在多步视觉 RAG 中识别并实证验证了轨迹级证据利用率退化现象**——与已有工作仅关注单次检索相关性不同，本文揭示了"证据被检索到但未被有效使用"的系统性衰退问题。
2. **提出 TAEC，一个无需训练的三组件协调框架**——通过共享的未解决答案需求状态统一协调证据准入、自适应记忆暴露和视觉细节分配；与 VISOR、VimRAG 等前作依赖语义优先级/图依赖/时间衰减不同，TAEC 的决策依据是"当前仍有哪些答案需求尚未满足"。
3. **在统一评估协议下实现了多个商业 VLM 上的最佳性能**——与 ViDoRAG 相比，在 Gemini-3.5-Flash 上分别提升 7.8、11.1、18.6 个百分点；与重实现的 M3RAG 相比提升 5.0、2.1、2.5 个百分点，且在更长轨迹上增益更大。
4. **揭示了轨迹长度与准确率下降的因果关系**——通过冻结检索仅添加中性文本的对照实验，证明仅增加上下文长度本身就会使准确率下降 5.2 个百分点，说明问题本质是上下文污染而非单纯检索不足。

## 方法详解
TAEC 是一个附着在 DAG agent 基座之上的无训练协调层，维护一个共享的轨迹状态 $\mathcal{U}_t$（未解决答案需求集合，每个需求 $u$ 带权重 $\bar{w}_{u,t}$ 表示已被证据支持的程度）。在每个步骤 $t$，由三个组件协同决定配置 $\mathcal{C}_t = (S_t, \{E_{m,t}\}, \{p_{i,t}\})$：

**（1）证据准入（Evidence Admission）**：从过采样池（25个候选）中greedy选择 $S_t$（最多5页），目标函数：
$$S_t^* = \arg\max_{S \subseteq \mathcal{P}_t}\left[\alpha_A \sum_{(u,\bar{w}_{u,t})\in\mathcal{U}_t}\bar{w}_{u,t}C_u(S) + \beta_A\sum_{i\in S}b_i - \eta_A\sum_{i<j\in S}\text{sim}(i,j)\right]$$
三项分别奖励未满足需求的覆盖度、检索相关性和惩罚候选间冗余（caption Jaccard）。检索 top-1 始终作为锚点保留。

**（2）自适应记忆暴露（Adaptive Memory Exposure）**：对历史记忆 $m$，其暴露形式为：
$$E_{m,t} = \left(e^{-\lambda_{g_m}\Delta t_m},\ \mathcal{R}(h_m, s_{m,t})\right)$$
第一项为粒度依赖的指数衰减控制记忆持久性；第二项渲染函数 $\mathcal{R}$ 根据当前轨迹角色 $s_{m,t}$（active/resolved/failed）分别以完整、精简陈述或短终端轨迹的形式呈现文本观察，不修改存储的轨迹图。

**（3）视觉细节分配（Visual Detail Allocation）**：为每个保留图像分配像素预算：
$$p_{i,t} = B_t\left[\kappa\frac{r_{i,t}}{\sum_j r_{j,t}} + (1-\kappa)\frac{\exp(\tau_V r_{i,t}d_{i,t}c_{i,t})}{\sum_j\exp(\tau_V r_{j,t}d_{j,t}c_{j,t})}\right]$$
第一项按历史能量值比例分配，第二项结合图像相关性、细粒度收益估计和当前需求支持度做 softmax 加权，总预算 $B_t$ 随上下文压力动态调整。

## 实验与结果
- **数据集**：ViDoSeek（1,142题，PDF页面图像，单跳645/多跳497）、SlideVQA（2,215题，幻灯片）、MMLongBench-Doc（847道可答题，长文档）。
- **基线**：Vanilla、ReAct、ViDoRAG、DAG agent（去TAEC基座）、重实现M3RAG。
- **执行模型**：Gemini-3.5-Flash、Kimi-K3、GPT-5.6-Sol、GPT-4o-mini（均冻结，无训练）。
- **主要结果（Gemini-3.5-Flash，Table 1）**：

| 系统 | ViDoSeek | SlideVQA | MMLongBench-Doc | 平均 |
|------|---------|---------|-----------------|------|
| Vanilla | 76.7 | 79.6 | 40.1 | 60.3 |
| ReAct | 79.5 | 77.2 | 42.4 | 61.0 |
| ViDoRAG | 79.7 | 72.5 | 31.5 | 59.4 |
| DAG agent | 80.1 | 76.7 | 44.5 | 62.3 |
| **TAEC** | **87.5** | **83.6** | **50.1** | **66.8** |

- TAEC 在12组对比中赢10组；相比 ViDoRAG 提升 7.8/11.1/18.6 个百分点；相比 M3RAG 提升 5.0/2.1/2.5 个百分点。
- **消融**：三个组件各自带来约 41%–88% 的最终增益；Admission 贡献最大（ViDoSeek +6.2/总+7.4；MMLongBench-Doc +3.7/总+5.6）。
- **轨迹深度分析**：在 ReAct 执行≥4次搜索的子集上，TAEC 比 ReAct 分别高出 28.1（ViDoSeek）、26.6（SlideVQA）、13.6（MMLongBench-Doc）个百分点。
- **覆盖率分解**：TAEC 在 115 道仅自身检索到 gold page 的问题上准确率达 86%，而 DAG agent 仅 22%。
- **效率**：TAEC 每问题平均传输 0.61 MB 上下文（ReAct 为 3.73 MB）、14.7 张图像（ReAct 为 92.8 张），模型调用 2.91 次（ReAct 5.34 次）。

## 相关工作脉络
- **ReAct (Yao et al., 2023)**：基础的多步 think-search-observe 循环，无记忆管理，所有检索图像保留在上下文中；TAEC 在其之上引入轨迹状态协调，解决其证据利用率退化问题。
- **ViDoRAG (Wang et al., 2025a)**：多智能体训练-free 框架，有公开代码；TAEC 作为轻量层可叠加于类似 DAG agent 基座上，无需多智能体架构。
- **VISOR (Shen et al., 2026)**：通过滑动窗口和查询提醒管理记忆；TAEC 的区别在于以"未解决需求"为核心驱动所有证据决策，而非位置/时间衰减。
- **VimRAG (Wang et al., 2026a)**：用语义优先级、图依赖和时间衰减选择视觉记忆；TAEC 以更简洁的统一需求状态替代多重独立信号。
- **M3RAG (Du & Li, 2026)**：多智能体规划-检索-验证循环；TAEC 对比实验显示在相同协议下平均超越重实现 M3RAG。
- **Self-RAG (Asai et al., 2024) / Adaptive-RAG (Jeong et al., 2024)**：基于置信度/反思的自适应检索；TAEC 关注的是检索后"如何使用证据"，而非"何时检索"。

## 局限性与未来方向
- **依赖底层模型的智能体能力**：TAEC 无法弥补 acting model 在轨迹规划和停止决策上的根本缺陷；当模型不具备多步推理能力时（如 Qwen2.5-VL-7B），单轮检索仍优于多步配置。
- **策略训练是互补方向**：当前 TAEC 只协调"展示给模型什么"，不改变"模型决定做什么"，与 RL 训练搜索策略（如 VRAG-RL）正交。
- **评估仅限单一检索器族**：主实验使用同一 dense 检索器，替代检索器（BM25、ColPali）只在附录中验证。
- **基准均为英文文档集合**：跨语言泛化未验证。
- **未处理"不可回答"问题**：MMLongBench-Doc 中 244 道题标注为不可回答，三个组件均不鼓励拒绝回答，导致在该子集上表现弱于有 verifier 的 M3RAG。

## 研究启发与可借鉴点
- **"未解决需求状态"作为统一协调信号的设计范式**：将分散的证据管理决策（准入/记忆/细节）统一到一个共享的状态变量下，避免各组件各自为政；此思路可迁移至文本 RAG、代码 RAG 等多步 agent 系统。
- **粒度依赖的记忆衰减**：不同信息粒度（页面级 vs. 区域级）用不同衰减率 $\lambda_g$，比全局统一衰减更精细；可借鉴到任何需要分层记忆的 agent 系统。
- **轨迹级诊断指标的引入**：Coverage（gold page 是否进入上下文）与 Use（命中后的准确率）分解为评估证据利用效率提供了新工具，优于单一最终准确率。
- **冻结检索的对照实验设计**：Section G 通过固定检索结果仅增加中性文本，精确分离"上下文增长"与"检索不足"两个混淆因素，是评估上下文管理方法价值的干净范式。
- **与 RL 方法的正交性**：TAEC 与 VRAG-RL 等 RL 训练方法作用于不同维度（证据呈现 vs. 策略学习），提示未来工作可探索两者的组合。

## 关键术语表
- **Trajectory-level evidence utilization degradation（轨迹级证据利用率退化）**：在多步视觉 RAG 推理过程中，即使相关证据被检索到，其随推理步数增加而逐渐失效、无法支撑正确回答的现象。
- **Requirement state $\mathcal{U}_t$（需求状态）**：由未解决答案需求 $(u, \bar{w}_{u,t})$ 构成的共享状态，$\bar{w}_{u,t}$ 反映该需求在当前轨迹中被证据支持的程度，是三个 TAEC 组件的共同决策依据。
- **DAG agent**：基于轨迹有向无环图的 agent 基座，每个节点记录一次搜索及其摘要，边表示依赖关系，是 TAEC 的底层实现载体。
- **Evidence Admission（证据准入）**：TAEC 第一个组件，从候选检索结果中greedy选择进入上下文的源，平衡需求覆盖、检索相关性和冗余惩罚。
- **Adaptive Memory Exposure（自适应记忆暴露）**：TAEC 第二个组件，按记忆节点的轨迹角色（active/resolved/failed）以不同粒度呈现历史观察，并结合粒度依赖的时间衰减。
- **Visual Detail Allocation（视觉细节分配）**：TAEC 第三个组件，在总像素预算约束下，按图像相关性、细粒度收益和需求支持度为保留图像分配不同分辨率。
- **Post-hit accuracy（命中率后准确率）**：在 gold page 已进入上下文的子集上计算的准确率，反映模型"用好证据"的能力。
- **Coverage（覆盖率）**：至少有一页 gold page 进入过轨迹上下文的题目比例，反映证据获取能力。

## 可复现要素
- **数据集**：ViDoSeek、SlideVQA、MMLongBench-Doc（均为公开数据集）
- **代码**：论文未明确声明开源仓库链接，但附录提到"released code"（已发布的代码）
- **权重**：使用商业 VLM API（Gemini-3.5-Flash、Kimi-K3、GPT-5.6-Sol、GPT-4o-mini），均冻结不训练
- **关键超参**：附录 A 声明所有系数/衰减率/阈值为 released implementation 默认值，全实验保持一致，具体数值见代码；主要参数包括 $\alpha_A, \beta_A, \eta_A, \lambda_g, \kappa, \tau_V$ 等
- **检索器**：Qwen3-VL-Embedding-2B FAISS index，36,233 页的跨基准混合索引
- **每步最多5页 admissions，最多20次模型调用，图像 JPEG ≤300K pixels/≤38KB**
