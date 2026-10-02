---
title: "VISUAL-PARALLEL-SEARCH-LEARNING-TO-SEARCH-HIGH-RESOLUTION-IM"
source: https://arxiv.org/pdf/2609.37002v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 23:28:28"
field: "多模态大模型视觉推理与 Agent 搜索"
keywords: ["visual parallel search", "high-resolution VQA", "agent orchestration", "reinforcement learning", "tiling inspection", "adaptive zoom", "GRPO"]
innovations: ["并行规则网格检查+自适应缩放的最小化视觉搜索框架", "无提示污染的子智能体ROI监督数据管线", "配对角色专属GRPO低方差半梯度训练代理"]
benchmarks: ["TIR", "VSTAR", "ZoomBench", "HR-Bench 4K", "HR-Bench 8K"]
---

# 论文速读：VISUAL-PARALLEL-SEARCH-LEARNING-TO-SEARCH-HIGH-RESOLUTION-IM

## 一句话总结
VPS（Visual Parallel Search）提出了一种"并行网格检查 + 自适应缩放"的视觉搜索框架：主智能体首先调度多个问题条件化的子智能体并行读取图像瓷砖以获取全覆盖概览，随后对选定的精确区域或合并邻域进行缩放以恢复细节；该框架同时支持免训练推理与主/子智能体的后置训练（SFT + 配对GRPO）。

## 研究问题与动机
- **高分辨率视觉证据获取难题**：多模态模型回答小目标/局部文字/关系型问题时，需要小块、空间局部的视觉证据；全局缩放会抹除细节，而缺乏先验的顺序缩放则使感知变成昂贵的串行搜索。
- **既有 crop-and-zoom 方法的结构缺陷**：现有方案（如 V*、DC²、ZoomEye）多基于固定几何分区或单智能体顺序规划，固定分区易跨边界割裂对象/表格，单顺序智能体在覆盖率与局部细节之间被迫取舍。
- **训练接口弱、角色耦合**：已有设计将局部感知、区域选择、证据聚合与最终作答纠缠在一条长轨迹中，难以对"主控制器"与"局部阅读器"分别监督与优化。
- **学习可控的任务分解需求**：作者希望把高分辨率视觉搜索构建为可训练的解耦架构——主智能体负责搜索控制与答案合成，子智能体专注局部证据读取，并给出两套后训练方案（SFT + 配对 role-specific GRPO）。

## 核心贡献（创新点）
1. **并行栅格搜索 + 自适应缩放的最小化并行搜索 harness**：以规则空间分解 + 并行同构 tile reader 替代单智能体串行探索，本质区别在于用"全量并行召回 + 集中式证据聚合 + 精确缩放精修"取代"单点逐次试探"。
2. **跨 3 种模型尺寸、5 个基准分割的系统性对比**：与专用 zoom-only 搜索在相同模型下逐项对比（15 个单元格中 14 个获胜），本质区别在于提供了一套可直接复用的"并行 vs. 串行"搜索基线对照框架。
3. **无提示污染的 sub-agent ROI 监督数据管线**：教师模型标注时携带 ROI hint，但学生 prompt 只保留 tile + question + ID + 坐标，并由 hint-free verifier 重审可疑样本；本质区别在于切断了标注泄露路径，使子智能体学会"仅凭局部像素回答问题"。
4. **配对角色专属 GRPO 代理（paired role-specific GRPO surrogate）**：主、子分别按相同 prompt 采样成组并独立归一化，主损失经最终答案反馈、子损失经局部证据反馈，显式省略跨角色的全局长距信用路径；本质区别在于把完整联合策略梯度近似为一组低方差、角色解耦的半梯度，避免把一个终端回报广播到多个并行子报告引发的Credit assignment 噪声。

## 方法详解
- **问题设定与角色划分**：策略分解为主智能体 π_main（控制轨迹、调用工具、综合答案）与子智能体 π_sub（评估局部 tile 证据）；两者可共享参数但使用不同 prompt 与 response mask。
- **Parallel Tile Inspection（grid_search）**：主智能体调用 `grid_search(I, n)`（n ∈ {3,…,8}），将当前图像划分为 n×n 规则网格，并行派发至各 tile reader。每个子智能体返回结构化紧凑记录：
  > o_j = (useful, priority, evidence, candidate_answer, bbox, confidence)
  子智能体可见当前 tile、问题、tile ID 与坐标，不可见 gold ROI 与其他 tile；框架只做聚合，不替主智能体投票。
- **Adaptive Zoom & Region Aggregation（zoom_in）**：主智能体调用 `zoom_in(I, b)`，b 可为单个归一化矩形框，也可为多个邻接框的列表（后者返回最小包围 crop，用于修复网格边界碎片化与关系型证据的多点聚合）。
- **Agent Loop**：从全局缩略图出发，主智能体每步可选：(i) 作答；(ii) 发起一次 grid scan；(iii) 对原图某 region 执行 zoom。观测追加至轨迹直至作答或耗尽 turn budget（默认 15 步）；超预算时触发最终作答指令。
- **Supervised Post-Training（SFT）**：
  - 主智能体：学习何时搜索、如何解读 grid 报告、何时合并 region、何时停止；
  - 子智能体：学习结构化输出的 schema 有效性、usefulness/priority 判断、evidence 描述、candidate answer 与 bbox 定位。
  - SFT 作为 orchestrator RL 的冷启动基线。
- **Paired Role-Specific GRPO（核心训练公式）**：
  - 对每个主 prompt 采样 K 条轨迹构成 G^main_i；子智能体 prompt 取自固定外部 tile 数据集，对每个 tile prompt 采样 K 条响应构成 G^sub_j；
  - 两组独立归一化得到 advantage A^main_{i,k} 与 A^sub_{j,k}；
  - 联合损失：L_pair = L^G_main + λ_sub · L^L_sub,dir；
  - 主损失仅作用于主生成 token，子损失仅作用于固定 tile prompt 下重新采样的新响应；被拷贝的工具观测与原嵌套输出不参与策略梯度；
  - 在共享权重下该目标等价于对完整联合返回梯度的有偏、低方差半梯度近似。

## 实验与结果
- **数据集**：TIR（120）、VSTAR（191 金标）、ZoomBench（845）、HR-Bench 4K（800 行）与 HR-Bench 8K（800 行）；CircularEval 在每子集 200 题上测试"四旋转选项全对"的严格一致性。
- **模型**：Qwen3.5-4B/9B、Qwen3.6-27B；子智能体默认同规模，部分设定替换为 Gemini 3.5 Flash 以考察读者选择的影响。
- **核心结果（训练免调 Grid+Zoom vs. Zoom-only）**：
  - **14/15** 相同模型单元格 mean accuracy 更高；4B 在 TIR / VSTAR / HR-8K 分别 +6.9 / +8.0 / +7.1，9B 在 VSTAR +8.0，ZoomBench 在三档模型均稳定 +3.2；27B 仅在 ZoomBench (+3.2) 与 HR-4K (+1.3) 超出重复波动范围。
  - 用 Gemini 3.5 Flash 作 27B 的 tile reader，相对自 tile 在 TIR +1.7、VSTAR +2.3。
- **CircularEval**：4B HR-4K/8K 从 48.2/40.2% 提升至 59.0/52.8%，说明优势在"选项顺序无关的一致正确性"上依然成立。
- **SFT**：五分割均提升；匹配未打种子 HR-Bench 4K 比较给出 **+4.17 点**增益（95% 区间 [+1.54, +6.75]）。
- **Main-only RL**：内部 4-response/100-prompt 评估，mean tool calls 从 2.65 降至 2.11（Δ=-0.54，区间 [-0.87,-0.21]），pass@1 72.50% vs. 71.25%（区间跨零）；跨基准外部提升不稳健。
- **Joint RL 角色非对称**：Joint RL 作为 sub（SFT main）HR-4K +1.38（区间含零）；Joint RL 作为 main（SFT sub）-2.38（区间含零），但 Joint RL main + Joint RL sub 组合 -3.88（[-6.88,-1.12]，显著）。Joint RL main 轨迹更长（mean calls 2.22→4.35）、完成率下降（789→720/800），失败集中在 iteration/context 上限（32768 token）。
- **数据质量审计**：VisualProbe 框与人工框 median IoU=0.243、28.9% 不相交；ZwZ 框 median IoU=0.887；Nine-pool 重叠审计在 TIR/VSTAR/HR-Bench 未检出有效污染，ZoomBench 少量 exact/perceptual 命中但不影响结论。

## 相关工作脉络
1. **V* (Wu & Xie, 2024)**：LLM-guided 引导视觉搜索 + V*Bench；本文在"并行全量 tile 采集 + 主智能体合并决策"上与 V* 的单点逐步探索形成分工差异。
2. **DC² + HR-Bench (Wang et al., 2025)**：递归划分图像并把 patch 转文本证据；本文与之区别在于不递归，而以规则 n×n 单层网格 + 主智能体的可选 merge 缩放替代深层树划分。
3. **ZoomEye (Shen et al., 2025)**：层次树结构的 region 探索；本文固定几何分区 + 并行读者 + 主智能体集中决策，更利于可审计的子任务监督与 RL 解耦。
4. **ToolOrchestra (Su et al., 2025) / AOrchestra (Ruan et al., 2026) / ParaManager (Yuan et al., 2026) / Orchestra-o1 (Zhang et al., 2026)**：学习型编排器与并行调度；本文将这些思想约束到"同质 tile reader + 固定空间分解 + 可审计 region"这一最小测试床，以显式分离主/子学习信号。
5. **Mini-o3 / VisualProbe (Lai et al., 2026) & ZwZ-RL-VQA (Wei et al., 2026)**：提供部分 tile 监督图像与 ROI box；本文在此基础上构造 hint-free 验证与双角色 GRPO，而非仅使用现成数据。
6. **LEMON (Chen et al., 2026) / MAS-Orchestra (Ke et al., 2026)**：counterfactual 与功能调用 RL 优化编排；本文与它们定位不同——聚焦高分辨率视觉证据获取这一具体子任务，并通过角色解耦的 loss 构造与 ROI 监督实现可解释的局部/全局分工。

## 局限性与未来方向
- 当前仅一层 grid search，递归或异步搜索可能提升 recall 但会加剧上下文管理与 credit assignment 复杂度。
- Joint RL 角色非对称：本地证据阅读与全局搜索控制并未同向优化，deploy joint RL main 反而降低完成率与准确率。
- 部分实验配置（historical joint reward code、exact tool/judge prompt、早期视觉 token 与 resize 策略）已无法恢复，限制完全复现。
- 数据质量存在来源异质性（VisualProbe 框质量偏低），验证集受 SFT 暴露影响；外部 transfer 结论多为 descriptive 而非配对显著。
- 评估集偏向高分辨率 VQA，尚未在同等的覆盖度下隔离计数、关系、OCR、文档排版失败模式。

## 研究启发与可借鉴点
1. **主/子角色解耦训练范式**：把"搜索控制（主）"与"局部证据读取（子）"视为两个可分别监督的分布，用固定外部 tile prompt 集合 + 配对 GRPO 获得低方差、角色隔离的训练信号，可迁移至任何"规划者 + 专用专家"的多步工具调用系统。
2. **Hint-free verification + 结构化输出 gate**：对 ROI 监督数据做"教师带提示标注 → 学生无提示推理 → 自动重审可疑样本 → 人工重写 artifact 痕迹"的数据管线，有效切断 hint 泄露；这对任何依赖空间标注的训练场景均有参考价值。
3. **CircularEval 式严格一致性评估**：用多选项旋转的全对率衡量"证据获取 + 答案合成"的稳定性，比单行准确率更能刻画搜索骨架的实际效用。
4. **角色矩阵（role matrix）部署实验**：交叉部署 SFT / RL 的主、子角色以识别"局部提升是否能传递到全局"，是诊断联合训练非对称性的必要手段。
5. **低成本并行召回 + 按需精修的组合策略**：先并行收集低分辨率 tile 概览、再由主智能体决策 merge 与 zoom，可在多数任务上以较少总计算换取更高的 coverage，适用于图像/视频/文档的 agentic 感知系统。

## 关键术语表
- **VPS (Visual Parallel Search)**：论文提出的并行视觉搜索框架，核心为 grid_search + adaptive zoom_in 的组合与主/子角色分工。
- **Main Agent / Sub-Agent (Tile Reader)**：分别负责搜索控制/答案综合与局部 tile 证据评估的两个角色，可共享参数但 prompt 与训练信号分离。
- **grid_search(I, n)**：将图像划分为 n×n 规则网格、并行分发给子智能体返回结构化报告的函数调用。
- **zoom_in(I, b)**：对单个归一化矩形框或相邻框列表执行高精度裁剪，用于修复网格边界碎片与关系证据聚合。
- **Paired Role-Specific GRPO**：主、子智能体分别以相同 prompt 成组采样并独立归一化，主用最终答案、子用局部证据奖励，组合为低方差半梯度代理的目标函数。
- **CircularEval**：要求同一问题的全部四个循环选项旋转均回答正确才算正确的严格评估协议，用于检验答案一致性。
- **Hint-free verification**：剔除标注过程中 ROI hint 泄露的监督流程，确保子智能体仅依靠 tile 图像与问题作答。
- **Role Asymmetry（角色非对称）**：论文观察到的现象——局部阅读器与全局控制器在联合 RL 下优化方向不一致，提升一方未必带动另一方。

## 可复现要素
- **数据集**：TIR、VSTAR、ZoomBench、HR-Bench 4K/8K、VisualProbe、ZwZ-RL-VQA；外部 benchmark 多为公开发布，内部 RL 训练池来自这些来源转换并做 overlap 排除。
- **代码/权重**：项目主页 https://xijia-tao.github.io/vps/（论文未直接在正文给出 GitHub 仓库链接）；部分 historical joint reward code、tool/judge prompt 与早期服务修订已无法恢复。
- **关键超参**：SFT 用 Qwen3.5-4B + LoRA rank/scaling 32/64、lr 1e-4 cosine、bf16、max_len 32768、2 epoch；Main-only RL 用 8 prompts × 8 rollouts/update、LoRA rank/scale 32/32、AdamW lr 1e-5、weight_decay 0.01、KL 系数 0.001、entropy -0.01；Joint RL 用 16 prompts/batch、8 rollouts/prompt、mini-batch 2、LoRA 作用于 7 个投影模块。grid n ∈ {3,…,8}，主迭代预算 15 步，子样本数 K=8。
