---
title: "VisualEvolve-A-Skill-Self-Evolution-Framework-for-Visual-Rea"
source: https://arxiv.org/pdf/2610.28386v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-06 02:59:26"
---

# 论文速读：VisualEvolve: A Skill Self-Evolution Framework for Visual Reasoning

## 一句话总结
本文提出 VisualEvolve，一种无需更新模型参数即可自我进化视觉语言模型（VLM）可执行视觉操作与工具使用策略的框架；通过快照选择机制防止性能回退传播，并借助渐进式证据访问克服多轮进化中的证据过载问题，在多个基准上取得显著且高效的提升。

## 研究问题与动机
- **核心问题**：如何让 VLM 在不更新骨干参数的前提下，通过任务执行经验持续精炼外部的视觉技能与工具调用策略？
- **现有训练方法的局限**：DeepEyes、Mini‑o3 等方法依赖预定义视觉操作并通过 SFT/RL 学习调用策略，难以适应任务执行中暴露的新需求；且所需 RL 轨迹规模极大（数十万条），成本高昂。
- **技能进化面临的两大挑战**：
  1. **性能回退**：针对观察到的失败修改技能可能引入新的性能下降，且错误会随迭代累积。
  2. **进化证据过载**：随着回合增加，需同时检查累积的轨迹历史、技能修改记录和评估结果，上下文负担加重，能力缺口更难诊断。
- **动机**：将视觉操作和策略表示为可重用的外部技能（存储于模型参数之外），通过结构化反馈循环实现技能自我进化，同时设计机制缓解上述两个挑战。

## 核心贡献（创新点）
1. **提出 VisualEvolve 自进化框架**：通过“选择–执行–进化–评估”循环迭代更新外部技能库，全程保持 VLM 骨干冻结。与训练型方法本质区别在于不更新模型参数，仅通过外部技能实现性能提升。
2. **技能库快照选择（Snapshot Selection）**：每轮从已归档的历史快照中选取固定验证集上准确率最高者作为本轮起点，切断性能回退的传播链。与直接基于最新快照进化的方法本质不同，有效防止错误累积。
3. **渐进式证据访问（PEA）**：构建从粗到细的证据访问层次（技能性能报告→进化日志→轨迹摘要→原始轨迹片段），使智能体按需获取证据，大幅降低上下文负担。与一次性加载所有历史记录的方法本质不同，提升诊断效率。
4. **系统性实验与高效性验证**：在四个冻结骨干和六个评估集上验证有效性，最大提升 15.1 个百分点；消融与泛化实验证实各组件贡献；与基于 RL 的训练方法相比，所需任务轨迹少一个数量级以上（仅 3,912 条）。

## 方法详解
VisualEvolve 运行 R 轮迭代，每轮包含四个阶段：

### 1. 选择阶段（Select）
- 维护历史快照集合 $\mathcal{C}_{r-1} = \{\mathcal{L}_0, \ldots, \mathcal{L}_{r-1}\}$ 及对应验证准确率 $a_t^{\text{val}}$。
- 本轮起点快照为 $\mathcal{L}_{p_r} = \arg\max_{\mathcal{L}_t \in \mathcal{C}_{r-1}} a_t^{\text{val}}$（第 1 轮取 $\mathcal{L}_0$）。

### 2. 执行阶段（Execute）
- 使用冻结骨干 $A_\theta$ 和当前技能库 $\mathcal{L}_{p_r}$ 实例化 Worker，在轮特定执行集 $\mathcal{D}_r^{\text{exe}}$（从进化数据中采样，与固定验证集 $\mathcal{D}_{\text{val}}$ 不交）上求解视觉任务。
- 生成完整轨迹 $T_{r,j}$，记录工具调用、视觉操作、中间推理与最终预测；通过比对 ground‑truth 标注正确性标签。
- 由 Memory Agent 将每条轨迹压缩为 **轨迹摘要** $\mathcal{M}_r = \{m_{r,j}\}$，每个摘要包含多条结构化发现，并附带行级源引用（如 `[[trajectory.json|L3-L16]]`）以便回溯。

### 3. 进化阶段（Evolve）
- 由 Self‑Evolving Agent 通过 **渐进式证据访问（PEA）** 诊断能力缺口，并更新技能库 $\mathcal{L}_{p_r} \rightarrow \mathcal{L}_r$。
- PEA 层次：
  1. **技能性能报告（SPR）**：计算每个技能的 **技能性能分数（SPS）**，定义为在技能相关验证子集 $\mathcal{D}_s^{a,b}$ 上两快照平均正确率之差：
     $$\text{SPS}_{a \to b}(s) = \frac{1}{|\mathcal{D}_s^{a,b}|}\sum_{d_j \in \mathcal{D}_s^{a,b}} c^{(b)}(d_j) - \frac{1}{|\mathcal{D}_s^{a,b}|}\sum_{d_j \in \mathcal{D}_s^{a,b}} c^{(a)}(d_j)$$
     其中 $c^{(t)}(d_j)\in\{0,1\}$ 表示在快照 $\mathcal{L}_t$ 下答案是否正确；$\mathcal{D}_s^{a,b}$ 为快照 $a$ 或 $b$ 下至少加载技能 $s$ 的验证样本集合。SPS 绝对值越大表明该技能与性能变化关联越强。
  2. **进化日志（EL）**：记录每轮验证准确率 $a_r^{\text{val}}$ 及相邻快照间每个技能的差异文件（`.dif`），形成性能‑修改映射历史。
  3. **轨迹摘要（TS）**：优先分析错误预测的摘要，识别重复失败模式（如重复访问相同图像区域、过早终止搜索）并抽象为能力缺口。
  4. **原始轨迹**：当摘要证据不足时，按源引用访问对应原始轨迹片段。
- 更新后的技能库需经过结构性和可执行性检查（由 Orchestrator 自动完成），确保目录命名、YAML frontmatter、脚本路径与接口符合规范。

### 4. 评估阶段（Evaluate）
- Worker 在固定验证集 $\mathcal{D}_{\text{val}}$ 上评估 $\mathcal{L}_r$，得到准确率 $a_r^{\text{val}}$ 及技能暴露轮廓 $\mathcal{V}_r$。
- 更新 SPR（基于 $\mathcal{V}_0$、$\mathcal{V}_{p_r}$、$\mathcal{V}_r$ 计算三个快照对的 SPS）和 EL（追加记录 $\rho_r = (r, a_r^{\text{val}}, \text{Ref}(\Delta\mathcal{L}_r))$）。
- 将 $\mathcal{L}_r$ 归档为新快照，进入下一轮。

全部 R 轮完成后，返回验证准确率最高的快照 $\mathcal{L}^\star = \mathcal{L}_{t^\star}$ 作为最终技能库。

## 实验与结果
- **进化数据**：默认混合 VisualProbe（2.9K）与 DeepEyes Visual Search 组件（~22K）；保留原图、问题、答案。另测试 Mini‑o3 语料（VisualProbe_train + DeepEyes‑train‑4K）与 CodeVision 语料（CodeVision‑SFT + CodeVision‑RL）。
- **评估基准**：V*Bench（191）、HRBench4K（800）、HRBench8K（800）、VisualProbe Easy（141）/Medium（268）/Hard（106）。
- **骨干网络**：Qwen2.5‑VL‑7B、Qwen3‑VL‑32B、Qwen3.5‑Plus、Kimi‑K2.5、Doubao‑Seed‑2.0‑Pro（实验重点汇报后四者）。
- **基线**：Single‑Pass（无工具单次推理）、Initial Skill Library（仅含初始 crop 技能）、训练型代理（Pixel Reasoner、DeepEyes、Mini‑o3 等）。
- **主要结果**：
  - VisualEvolve 在 24 个骨干‑评估集组合中 **23 个** 优于 Initial Skill Library，全部优于 Single‑Pass。
  - **最大提升**：VisualProbe Hard + Qwen3.5‑Plus，准确率从 39.6% 升至 54.7%，**提升 15.1 个百分点**（相对 38.1%）。
  - 宏平均提升：Qwen3.5‑Plus +8.8、Qwen3.6‑Plus +5.4、Kimi‑K2.5 +4.6、Doubao‑Seed‑2.0‑Pro +3.9 个百分点。
  - 与训练型代理相比，Qwen3.5‑Plus 上 VisualEvolve 达 77.6%，高于 Mini‑o3（67.4%）。
- **消融**：移除 Select / Execute / Evaluate 分别下降 5.4 / 7.7 / 6.4 个百分点；PEA 各组件（SPR / EL / TS）移除各降 2.6–4.6 个百分点，整体移除 PEA 降 6.3 个百分点。
- **泛化**：
  - 跨数据来源（Default / Mini‑o3 / CodeVision）均稳定提升（+8.8 / +7.1 / +5.0 个百分点）。
  - 将 Kimi‑K2.5 上获得的 $\mathcal{L}^\star_{\text{Kimi}}$ 直接迁移至其他三个骨干，宏平均仅比专属进化低 1.1–2.7 个百分点。
- **成本**：总轨迹预算 3,912 条；训练型方法需 63×–209× 更多 RL 轨迹；每轮平均 API 成本 $2.55–$4.12；VisualProbe Hard 上每例额外成本仅 $0.0268 即获 15.1 个百分点增益。

## 相关工作脉络
