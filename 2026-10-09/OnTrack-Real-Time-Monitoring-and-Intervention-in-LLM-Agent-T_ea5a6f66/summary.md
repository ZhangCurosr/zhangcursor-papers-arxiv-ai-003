---
title: "OnTrack-Real-Time-Monitoring-and-Intervention-in-LLM-Agent-T"
source: https://arxiv.org/pdf/2610.12375v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-01 14:24:11"
---

# 论文速读：OnTrack-Real-Time-Monitoring-and-Intervention-in-LLM-Agent-T

## 一句话总结
OnTrack 是一种流式实时监控系统，将 LLM Agent 的执行过程建模为动态生长 DAG，并通过流式结构感知最优传输（Streaming Structure-Aware Optimal Transport）将实时轨迹与历史参考对齐，在每步约 1ms 内检测循环、停滞、幻觉或因果反转，并可对不可逆操作进行前置拦截。

## 研究问题与动机
- **核心问题**：LLM Agent 高度自主执行任务时，易因幻觉、工具乱序调用、无限循环等问题导致算力浪费或不可逆安全风险，亟需事中监控与干预机制。
- **现有方法不足**：
  1. **离线评估/事后日志分析**：问题发现时 token 已消耗、损害已造成，无法阻止错误扩散。
  2. **基于 LLM Agent 的在线守卫**：额外调用增加延迟与成本，小模型难以有效判决大模型错误，或依赖黑盒打分模型，且仍主要输出分数而非可执行动作。
  3. **忽视执行图拓扑**：现有方法均未关注步骤间的依赖关系，而循环、乱序调用、因果反转等故障本质上反映在 DAG 结构异常中。
  4. **流式对齐的结构性挑战**：前缀比较会惩罚“进展过早”的 Agent；新步骤初期依赖未显现易致假阳性；逐事件重算全图对齐超出预算；聚合分数无法直接映射为连续干预动作。

## 核心贡献（创新点）
1. **首个流式结构感知最优传输监控框架**：将原本仅用于事后评估的 UFGW 对齐机制改造为可在线流式运行的实时监测器，打破离线评估的时间壁垒。
2. **三层解耦架构（L1/L2/L3）**：L1 无参考基础体征检测（循环/停滞）、L2 流式 UFGW 传输对齐、L3 基于工具 Schema 的事前不可逆操作拦截，三者按需开关，适配全参考/仅 Schema/无先验三种信息 regime。
3. **前沿掩码边际（Frontier-Masked Marginals）**：根据当前已覆盖的参考子 DAG 动态缩放参考质量分布，确保 Agent 按拓扑并行的合法提前执行不被错误惩罚，同时将“未完成即惩罚”转化为覆盖率进度仪表。
4. **临时结构处理与热启动增量求解器**：引入 age weights 线性爬坡新节点结构可信度，结合回溯边发现机制抑制假阳性；每步仅执行一次条件梯度（CG）迭代与 Sinkhorn 热启动，以 p50 0.85ms 的极低延迟支撑实时流式对齐。
5. **基于单步信号的决策引擎**：设计探索生命周期（Exploration Lifecycle）与严重标志滑动窗口密度阈值，输出 CONTINUE/WARN/BLOCK 动作，彻底规避“固定聚合分数阈值在变长轨迹中语义漂移”的失效问题。

## 方法详解
- **执行建模**：Agent 轨迹视为流式事件序列 $s_t = (a_t, \tau_t, c_t, e_t, \zeta_t)$，构建前缀图 $P_t$（节点为步骤，边为工具产出/消费依赖）。依赖元数据 $\zeta_t$ 支持三档：全声明、部分产物日志、纯无元数据（依赖 Schema 规则与账本匹配推断）。
- **代价矩阵**：$C_{ij}$ 综合动作文本余弦距离、参数余弦距离与工具替换代价，权重默认 $(\alpha, \beta, \delta) = (0.45, 0.20, 0.35)$，各项归一化至 $[0,1]$。
- **目标函数**：采用非平衡 Fused Gromov-Wasserstein（UFGW）形式，联合优化内容匹配、结构距离一致性（二次项，权重 $\theta=0.35$）、质量 KL 惩罚与熵正则化（$\varepsilon=0.05$）。
- **前沿掩码（Sec 4.2）**：依据上一时刻耦合矩阵计算参考节点覆盖率 $\mathrm{cov}_t(j)$，满足阈值 $\kappa=0.3$ 的节点保持全质量，前景节点降权为 $\beta_{look}\nu_j$，未激活节点降权为 $\varepsilon_\nu\nu_j$（$\varepsilon_\nu=0.01$）。
- **流式求解器（Sec 4.3）**：继承 Otap 批量 UFGW 思路，改为每步追加一行代价矩阵与结构矩阵后，计算 GW 梯度线性化近似，执行一次热启动非平衡 Sinkhorn（默认 $n_{inner}=5$），耦合按 $\gamma=0.7$ 指数移动平均。仅在触发干预前升级为收敛求解（≤10 外层 CG，≤50 内层 Sinkhorn）。
- **临时结构与 Age Weights（Sec 4.4）**：新节点结构行权重 $\omega_i(t) = \min(1, (t-t_i)/h_{age})$（默认 $h_{age}=3$），未被子节点消费前结构成本线性爬坡；依赖边可回溯添加，下一次 CG 迭代自动吸收变化。
- **信号与决策（Sec 4.5）**：节点泄漏率 $\ell_i = [1 - \frac{1}{\mu_i}\sum_j T_{ij}]_{[0,1]}$ 作为核心实时信号；新节点进入 EXPLORING 宽限期（默认 3 步），超期后分类为 premature-valid / CAUSAL INVERSION / unresolved orphan；累积未解析泄漏超预算 $B=25\%$ 时升级为 OFF TRACK。L3 门控在动作执行前仅校验 Schema 前置产物与单行 proposal alignment。

## 实验与结果
- **数据集**：`sailplane/swe-agent-trajs`（SWE-bench 轨迹），共 2,288 条可用轨迹（430 resolved, 1,858 unresolved）。按实例 ID 划分校准集（228）、参考集（5）与评估集，杜绝信息泄露。
- **基线**：余弦相似度（Cosine）、轨迹长度（n steps）、线性非平衡 OT（LinUOT, $\theta=0$）、图编辑距离代理（GED）、Markov surprise、grep frac。
- **主要结果**：
  - **早期
