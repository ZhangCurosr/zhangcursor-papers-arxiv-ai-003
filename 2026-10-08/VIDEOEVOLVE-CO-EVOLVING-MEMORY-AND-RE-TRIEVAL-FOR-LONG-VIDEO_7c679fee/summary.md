---
title: "VIDEOEVOLVE-CO-EVOLVING-MEMORY-AND-RE-TRIEVAL-FOR-LONG-VIDEO"
source: https://arxiv.org/pdf/2610.10183v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-08 10:43:43"
---

# 论文速读：VIDEOEVOLVE-CO-EVOLVING-MEMORY-AND-RE-TRIEVAL-FOR-LONG-VIDEO

## 一句话总结
本文提出 VideoEvolve，一种通过交替式 Agentic RL 协同进化记忆构建与检索策略的自进化框架，将长视频理解中的“记什么”与“如何取”从割裂优化转变为双向反馈闭环。

## 研究问题与动机
- **记忆-检索失配（Memory-Retrieval Misalignment）**：现有基于记忆的方法多采用单向适应范式，记忆构造策略固定且不解耦于下游推理，导致缺失细节难以恢复，检索效率受限于静态记忆的覆盖面。
- **构造与检索优化割裂**：构造与检索通常独立训练或依赖预定义规则，缺乏显式对齐机制，使系统无法利用“检索不到什么”来反哺“应记住什么”。
- **单一侧优化易过拟合特定问题分布**：仅训练记忆或仅训练检索均只能在部分基准上获益，且重复优化固定问题集易导致能力狭窄化，缺乏对薄弱环节的系统性引导。

## 核心贡献（创新点）
1. **交替式 Agentic RL 协同进化框架**：同时优化 Memory Evolver 与 Retrieval Evolver，前者学习选择性增补记忆，后者学习自适应检索与重访，两者通过冻结-更新交替实现双向反馈。与以往仅调检索策略或冻结记忆构造的工作本质不同，本文首次将构造与检索视为耦合系统联合进化。
2. **瓶颈感知进化反馈（BEF）**：通过对比纯记忆与允许重访原始视频的推理表现，量化当前系统在记忆侧还是检索侧存在性能瓶颈，并据此动态分配下一轮的训练预算比例。与固定比例训练或单一奖励信号相比，BEF 实现了训练算力的自适应流转。
3. **能力感知进化反馈（CEF）**：将下游诊断信号映射至 9 类视频理解能力，结合 EMA 平滑与均匀覆盖率混合生成自适应采样权重，引导模型向未充分发展但可学习的方向进化。与纯随机采样或固定课程相比，CEF 在保障能力均衡的同时避免了对特定问题分布的过度拟合。
4. **质量优先的群体相对优化目标（Quality-First Group-Relative Optimization）**：在群体内优先比较任务正确性，仅在任务奖励相等的轨迹中用归一化资源成本作平局决胜加分，配套 clip 策略梯度与 KL 正则。与直接减法惩罚资源消耗的目标相比，该设计确保任务正确性永远优先，效率仅作为次要排序信号。

## 方法详解
- **固定分级 Base Memory 与 Delta 增量**：以 0.5 fps 均匀采样构建 Root→Super→Macro 三层基础记忆，包含事件、实体、OCR 记录与时间锚点，进化期间 Base 永不重写。Memory Evolver 每次从相同 Base 出发，按需生成结构化 Delta 记录，两者合并为可检索记忆。
- **Memory Evolver（Observe–Decide–Augment）**：决策包含宏观选择（选 Macro 或 STOP）、时间区域（头/中/尾/边界）、帧数（2/4/8/16）与信息焦点（动作状态/外观空间/文本对齐/通用）。冻结的 Writer 将观测转为带来源链接的记录，接受条件通过结构校验与去重。
- **Retrieval Evolver（Search–Inspect & Revisit–Reason & Answer）**：支持层级导航、Lexical-Semantic 混合搜索、时间区间定位与存储帧检查；开启重访时可按需获取原始视频帧（每问题上限 64 帧，单次 ≤16 帧）。策略在 12 个探索槽后进入最终决策，输出带证据引用的答案或弃权。
- **交替优化流程**：第 k 轮先冻结 R^k，对每个视频采样 8 条构造轨迹，由 R^k 在纯记忆模式下打分并更新 C^k→C^{k+1}；重建记忆后固定 M^{k+1}，采样 4 条检索轨迹更新 R^k→R^{k+1}。梯度不经过 Writer、记忆文件或索引。
- **奖励设计**：构造侧奖励 = 救援项（参考未解但候选已解的问题比例）− λ_reg·回归项（参考已解但候选失败的比例）− λ_inv·无效终止惩罚；检索侧奖励 = 答案正确性 − λ_inv·无效惩罚。任务奖励经群体均值/方差归一化为 A_task，仅在 A_task 相等时用 A_eff = β(平均成本 − 当前成本) 作决胜加分，最终策略损失采用 clipped group-relative PPO 形式加 KL 正则。
- **BEF/CEF 调度**：BEF 聚合 paired 诊断信号得到 d_C/d_R，计算目标份额 s_target 并经滑动窗口平滑限制单轮变化幅度；CEF 按 9 类能力聚合需求 d_{p,c}，经 EMA（α=0.5）平滑后与均匀分布混合（ρ=0.3），投影至概率单纯形得到下轮采样权重，并按 0.6/0.2/0.2 混合 Frontier/Exploration/Retention 样本。

## 实验与结果
- **数据集与评测**：Video-MME（w/ sub, w/o sub, Long）、LongVideoBench（Overall, Long）、LVBench、MLVU、MMVU。训练使用 Video-MME-v2 去音频依赖后的 514 视频/2056 问题，诊断集 128 视频/512 问题。
- **主要结果**：VideoEvolve-8B（训练版）在 5 个基准的 6 项指标中均位列开源 Video-MLLM 第一：LongVideoBench 70.2%（超 ParaVT 9.8pp）、LVBench 58.9%（超 VideoZoomer 17.4pp）、Video-MME (Long) 67.1%、MMVU 75.1%。相较工具启用版 Qwen3-VL-8B baseline 提升 14.7–28.6pp。在记忆基线中，训练版 LVBench 58.9% 超 MemVid 14.5pp；免训练 Qwen3.8-27B 版本 LVBench 达 76.7%，超 MERIT-GPT 4.9pp。
- **消融结论**：协同进化优于单侧优化（LVBench 56.6% vs 记忆仅 51.1% / 检索仅 52.7%）；去掉交替更新降 LongVideoBench (Long) 4.0pp；BEF+CEF 组合将 MMVU 从 68.4% 提升至 75.1%；关闭视频重访仅微降 0.8–1.6pp；两阶段 SFT 冷启动不可或缺（无 SFT 直接跌至 LVBench 32.8%）。

## 相关工作脉络
- **Video-MLLMs / Video-Revisiting 方法**：如 VideoAgent、LongVT、VideoZoomer、Ego-R1 等侧重“何时/何地巡检源视频”。本文与其定位互补，聚焦“持久记住什么”，并通过记忆-检索双轴进化弥补单纯巡检范式的累积成本问题。
- **基于记忆的长视频理解**：如 EgoRAG、HippoMM、WorldMM、MERIT、MemVid 等多采用预定义构造或冻结检索器，构造与检索解耦。本文通过交替 RL 与配对诊断信号实现双向反馈，突破静态记忆只能服务固定检索的局限。
- **Agentic RL 与自进化 Agent**：如 EvolveR、SkillRL、Agent0、Evolving-RL 主要改进推理策略或技能库。本文将该范式迁移至长视频 Agent 的核心耦合轴（记忆构造 × 检索策略），并引入瓶颈/能力双维调度机制，区别于单一策略递归进化。
- **多模态推理优化目标设计**：与直接惩罚资源消耗或纯 accuracy 导向的 RL 不同，本文 Quality-First 设计将效率仅作为同质量组的 tie-breaker，与现有 group-relative RL 方法形成明确目标函数差异。

## 局限性与未来方向
- **评测范围受限**：当前仅验证离线多选题理解，未覆盖流式视频、开放对话或直接音频理解场景。
- **感知与预算瓶颈**：高度依赖冻结 Writer 的提取质量与有限观测预算，仍可能遗漏短暂事件或细微视觉细节；来源链接记录不保证事实绝对正确。
- **信号依赖与泛化边界**：策略优化依赖 GT 答案，BEF/CEF 基于固定诊断池与预设 9 类能力 taxonomy，泛化至无标签数据或新型能力时信号覆盖不足。
- **训练与推理开销**：交替训练需候选 rollout、冻结打分与重复记忆/索引重建，计算成本高；推理时虽可复用记忆，但多步检索与重访仍存在延迟。
- **未来方向**：扩展至流式增量记忆与多模态证据；
