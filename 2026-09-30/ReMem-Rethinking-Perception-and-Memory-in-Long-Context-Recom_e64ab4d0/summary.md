---
title: "ReMem-Rethinking-Perception-and-Memory-in-Long-Context-Recom"
source: https://arxiv.org/pdf/2609.37311v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 14:19:08"
field: "推荐系统中的长上下文推理与Agent"
keywords: ["Recommendation Agents", "Long-Context Reasoning", "OCR Perception", "Dynamic Memory", "RL for LLM", "Multi-modal Recommendation"]
innovations: ["首次将OCR多模态感知与动态记忆联合应用于推荐Agent", "提出时间演化记忆TEM实现线性复杂度无限上下文推理", "设计Multi-Mem GRPO将最终答案奖励传播至中间记忆更新"]
benchmarks: ["Amazon Reviews 2023 (Games/Books/MovieTV)", "InstructRec", "WebWalkerQA", "HotpotQA"]
---

# 论文速读：ReMem: Rethinking Perception and Memory in Long-Context Recommendation Agents

## 一句话总结
ReMem 提出了一种新型推荐智能体框架，通过 OCR 多模态感知替代脆性的 HTML 解析，并结合时间演化动态记忆机制，以线性计算复杂度高效处理任意长度用户历史，实现跨平台的长时间上下文偏好建模。

## 研究问题与动机
1. **HTML 感知脆弱性**：现有 RecAgent 普遍直接读取原始 HTML 内容，面对异构平台布局差异大、广告噪声多、无法捕捉图像等多模态信息，导致生成质量下降。
2. **长上下文推理低效**：用户交互历史和多步 Agent 轨迹产生大量 token，超出 LLM 上下文窗口限制；现有方法（Token 级压缩、外部记忆插件）存在分布外泛化困难、破坏标准自回归流程等问题。
3. **动态偏好建模缺失**：用户偏好随时间演化，现有方法多依赖静态记忆或外部模块，无法在无额外硬件/架构修改的前提下支持线性复杂度推理。

## 核心贡献（创新点）
1. **首个 OCR + 动态记忆联合框架**：首次将 OCR 多模态感知与 RL 增强动态记忆结合于 RecAgent，替代传统 HTML 解析。
2. **时间演化记忆（TEM）**：通过逐块（chunk-wise）顺序更新固定长度记忆缓存，使模型能在有界上下文窗口内处理任意长度交互历史，推理复杂度为 O(N)（线性），无需外部 KV cache 操作或注意力修改。
3. **Multi-Mem GRPO 训练策略**：将最终答案的组相对优势（advantage）传播至所有贡献于最终回答的中间记忆更新，解决独立生成的多轮对话难以分配信用的问题。
4. **系统级实验验证**：在三个 InstructRec 数据集的搜索、排序、判定任务上，平均超越最强基线 5.16%，且在所有长上下文长度设置下均保持一致性优势。

## 方法详解
**整体架构**（Figure 3）：由三大模块构成：
- **OCR 感知模块（§3.2）**：使用 DeepSeek-OCR-2 对商品页面截图进行解析，提取结构化多模态文本描述（自然语言 + 图片信息），嵌入 item card 模板后供 LLM 使用，绕过噪声 HTML 解析。
- **时间演化记忆（TEM，§3.3）**：将交互历史 $\mathcal{H} = \{v_1, \dots, v_J\}$ 划分为 $T$ 个连续块 $\mathbf{c}^t$，记忆更新公式为：
  $$\mathbf{m}^t \sim p_\theta(\mathbf{m}^t \mid \mathbf{m}^{t-1}, \mathbf{c}^t, Q)$$
  其中记忆长度 $|\mathbf{m}^t| = L$（论文中设为 5,000 token），初始 $\mathbf{m}^0 = \varnothing$，新记忆覆盖旧记忆。最终推荐基于 $\mathbf{m}^T$ 生成：$y \sim p_\theta(y \mid \hat{\mathbf{c}}, \mathbf{m}^T, Q)$。
  - **复杂度分析**：每步上下文字长为 $O(C + L)$，总复杂度为 $O(T(C+L)^2) = O(N)$，其中 $N = TW$。
- **Multi-Memory GRPO（§3.4）**：
  - 对每组查询采样 $G$ 条完整推理实例 $\{\mathcal{M}_g, y_g\}_{g=1}^G$，温度 $= 1.5$。
  - 组相对优势计算：$A_g = (r_g - \mu_r) / (\sigma_r + \epsilon)$。
  - 最终答案损失（Clip 目标）：$\mathcal{I}_{\text{Ans}}(\theta)$（式14）。
  - 记忆级损失将相同优势 $A_g$ 传播至每个中间记忆 token：$\mathcal{I}_{\text{Mem}}(\theta)$（式15）。
  - 总目标：$\mathcal{I}(\theta) = \mathcal{I}_{\text{Ans}}(\theta) + \lambda \mathcal{I}_{\text{Mem}}(\theta) - \beta \mathcal{D}_{\text{KL}}^m$，其中 $\lambda = 0.7$，$\beta = 0.04$。

## 实验与结果
- **数据集**：基于 Amazon Reviews 2023 构建的三个数据集（Games：94K 用户/25K 物品；Books：7K 用户/120K 物品；MovieTV：5.6K 用户/29K 物品），含广告噪声模拟真实场景。
- **基线类别**：DeepRec（SASRec、BERT4Rec）、LLMRec（P5、TokenRec）、RecAgent（ToolRec、iAgent）、LongAgent（Qwen-L1、Mem0）。
- **最强结果**：平均提升 **5.16%**。具体亮点：
  - Games 搜索任务 HR@1：**0.4292**（vs 基线 iAgent 0.3824，+12.2%）
  - Games 搜索任务 HR@3：**0.5590**（vs 基线 iAgent 0.5096，+9.7%）
  - Books 排序任务 NDCG@3：**0.4180**（vs 基线 iAgent 0.3839，+8.87%）
  - Books 判定任务 Accuracy：**0.7701**（vs 基线 Qwen-L1 0.7466，+3.14%）
- **长上下文分析（Figure 4）**：随上下文从短（0–14K）到中（14–112K）到长（>112K）增长，ReMem 在所有长度区间均保持最优；在 Books 长上下文上，搜索 HR@1 较 Qwen-L1 提升 **+18.6%**，排序 NDCG@3 提升 **+34.1%**。
- **消融实验**：移除 TEM 导致最大性能下降（Games HR@1 从 0.4292 降至 0.3301，-23%），验证动态记忆的核心作用。

## 相关工作脉络
1. **MemAgent [48] / MEM1 [61]**：聚焦通用长文档理解的增量记忆覆盖方法；ReMem 将其迁移至推荐领域，但额外引入 OCR 感知模块，并通过 Multi-Mem GRPO 将任务级奖励传播至每个中间记忆。
2. **ToolRec [58] / iAgent [44]**：RecAgent 类工作，依赖结构化/文本输入和外部静态记忆；ReMem 通过 OCR 实现平台无关的视觉感知，且记忆为可解释的自然语言 token。
3. **QwenLong-L1 [34] / Mem0 [3]**：长上下文 LLM 方法；ReMem 在不修改模型架构的前提下，通过内部记忆更新实现等价效果，且支持线性复杂度推理。
4. **PaddleOCR [4] / DeepSeek-OCR-2 [40]**：OCR 工具基础；ReMem 首次将其引入推荐 Agent 的感知层，验证了"像人一样看图阅读"的有效性。
5. **GRPO [29]**：Group Relative Policy Optimization 标准形式；ReMem 扩展为 Multi-Mem GRPO，解决多轮独立记忆生成的信用分配问题。

## 局限性与未来方向
1. **推理延迟增加**：相比直接使用 LLM，迭代式记忆构建和多步推理引入额外延迟（Books 数据集平均 224.60s/样本），不适合严格实时场景。
2. **用户行为建模有限**：当前仅模拟浏览商品页行为，未涵盖展开描述、搜索更多信息、完成结账等细粒度交互。
3. **OCR 模型依赖**：不同 OCR 模型的失败率和对各平台的泛化能力尚未系统评估。
4. **未来方向**：扩充交互环境以支持更丰富的用户行为信号；评估不同 OCR 模型和 chunk 大小；对比 MemAgent 等最新记忆 Agent。

## 研究启发与可借鉴点
1. **"看"代替"读"的感知范式**：用 OCR 替代 HTML 解析的思路可迁移至任何面向网页信息的 Agent 任务（如 WebSearch、Financial Agents）。
2. **奖励传播到中间记忆**：Multi-Mem GRPO 的信度分配技巧（将最终答案优势传播至所有中间推理步骤）可用于任何多阶段、独立生成-聚合的 Agent 系统。
3. **人脑式记忆抽象作为长期记忆方案**：TEM 的"固定长度 + 逐块覆盖"设计无需外部存储即可实现无限上下文处理，适用于资源受限的边缘 Agent 部署。
4. **人类可读的记忆表示**：记忆以自然语言 token 显式存储，便于调试和人工干预，为可解释推荐 Agent 提供了新思路。

## 关键术语表
**RecAgent（Recommendation Agent）**：基于 LLM 的自主推荐智能体，能够主动感知商品信息、推理用户偏好并执行推荐决策。
**TEM（Time-Evolving Memory）**：时间演化记忆，一种逐块顺序更新、固定长度的记忆机制，用于在有限上下文中建模长期用户偏好。
**Multi-Mem GRPO**：多记忆版组相对策略优化，将最终答案的优势值传播至所有中间记忆更新步骤的强化学习训练方法。
**OCR（Optical Character Recognition）**：光学字符识别，将商品页面截图转换为结构化文本描述，用于替代 HTML 解析。
**GRPO（Group Relative Policy Optimization）**：不依赖价值函数、通过组内奖励归一化估计优势的无基线强化学习策略梯度方法。
**InstructRec**：基于指令遵循的推荐框架，提供推荐任务的标准化指令格式和评测基准。

## 可复现要素
- **数据集**：基于 Amazon Reviews 2023 构建，实验配置见 Appendix C.2；官方代码已开源（https://github.com/Quhaoh233/ReMem）。
- **代码**：已公开。
- **权重**：论文未提及自训练权重，使用 Qwen3.5-9B 作为 backbone；OCR 使用 DeepSeek-OCR-2。
- **关键超参**：记忆长度 $L = 5,000$ token；Chunk 窗口 $W = 3$；最大 chunk 长度 $C = 100,000$ token；Rollout batch size = 64；Group size = 8；$\lambda = 0.5$（warm-up 后）/ 0（前 2 epoch）；$\beta = 0.04$；Inference temperature = 0.2；Training temperature = 1.5。
