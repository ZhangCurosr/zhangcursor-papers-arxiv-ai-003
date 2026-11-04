---
title: "When-Should-Enterprise-Policy-Decisions-Be-Learned-and-When"
source: https://arxiv.org/pdf/2610.17643v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-04 03:55:21"
field: "企业级Agent决策优化"
keywords: ["enterprise agent", "policy decision", "constrained decoding", "test-time reasoning", "structured generation", "counterfactual benchmark"]
innovations: ["分离输出格式与推理时间的2×2因子实验设计", "构建无人工标注的密封可执行基准", "发现监督训练与推理时间的部分替代关系"]
benchmarks: ["EnterpriseOps-Blind-A", "sealed 300-item benchmark", "open re-rendering 248-item benchmark"]
---

# 论文速读：When-Should-Enterprise-Policy-Decisions-Be-Learned-and-When

## 一句话总结
该研究在企业代理决策场景下，通过受控实验分离了"动作输出格式"（生成JSON对象 vs 约束单token读取）与"推理时间"两个因素，发现：对已训练策略二者无精度差异但成本悬殊，对未见策略边界推理可带来+17.3pp提升，且监督训练与推理时间呈部分替代关系而非独立增益来源。

## 研究问题与动机
1. **核心问题**：企业助手在回答前需选择7种动作之一（answer/ask clarification/ask confirmation/call tool/call multiple tools/escalate/refuse），这一决策应通过自回归生成JSON还是单次token读取实现，是否需要在决策前进行推理？
2. **现有方法不足**：当前实践通常将输出格式与推理时间捆绑优化，当质量下降时损失被归因于其中一个因素，导致无法准确识别真正有价值的成分。
3. **实际意义**：部署路径（推理+JSON生成）产生中位数542个token，而廉价路径仅1个token，延迟相差约两个数量级，但精度差异尚未量化。
4. **分布外泛化缺口**：模型在已见策略上表现良好，但对未见策略边界的决策能力不明，需要密封基准进行测试。

## 核心贡献（创新点）
1. **分离输出格式与推理时间的受控实验**：在In-distribution策略上检测到廉价单token路径与部署推理+JSON路径之间无显著精度差异（94.14% vs 93.46%），但廉价路径启动到启动动作翻转率低约90%（0.29% vs 3.00%）——这是对两种操作路径的比较，而非输出接口的纯净估计。
2. **构建无人工标注的密封基准**：隐藏的可执行策略将潜在因子映射为唯一正确动作并发放证明，自然语言渲染由外部模型完成且从不接触标签，300项基准的标签由确定性执行程序生成而非人类或LLM提供。
3. **推理驱动分布外泛化缺口**：在未见过策略边界上，推理带来+17.3pp [12.3, 23.0]的精确动作提升；在无推理条件下，输出格式对精确动作无显著影响（0.0pp [-4.3, +4.0]），反转了In-distribution评估暗示的顺序。
4. **监督与推理时间的部分替代关系**：边界目标监督在推理关闭时比随机监督带来+11.7pp vs +16.6pp的推理增益，交互效应为-4.9pp [-8.1, -1.7]，表明监督在推理不可用时创造价值而非叠加。
5. **零样本跨语言迁移验证**：无英语训练数据下，英语渲染比韩语低-0.82pp [-2.68, +0.93]，与重新渲染韩语的变异量级相同，未检测到显著英语退化。

## 方法详解
1. **三接口设计**：所有接口接收相同任务内容，输出契约不同——（a）JSON生成：自回归输出包含action/tool/arguments/needs_confirmation的JSON对象；（b）约束单token：动作词汇映射为7个单token，决策为受限集上的argmax；（c）LoRA指针头：rank-16适配器加小头直接从隐藏状态评分7种动作。
2. **密封基准构建**：隐藏策略表示为一组潜在因子（是否需要实时状态、是否缺少必需事实、是否需要确认副作用、是否被禁止、是否需要人工审查、涉及多少工具等），确定性执行器按固定优先级映射为唯一动作并发放证明；自然语言渲染由评估家族外的模型完成，通过恢复门（LLM提取器恢复决策因子）和挑战门（对抗模型寻找替代解读）验证保真度。
3. **两两因子实验设计**：推理（开/关）× 输出格式（JSON/单token）交叉，在同一密封300项上评估，评分harness声明三个主要终点（ pooled boundary accuracy、counterfactual pairs both-correct、invariance pairs both-correct）和两个次要终点（exact action、call tool recall）。
4. **边界目标监督训练**：预处理比较中，治疗臂添加边界对比数据（answer vs call tool、call tool vs ask confirmation、missing information三个边界），对照组添加等token量的随机域内数据，保持配方、调度、种子完全相同。
5. **训练×推理因子设计**：在所有四种配置（随机监督+推理关、随机监督+推理开、目标监督+推理关、目标监督+推理开）上评估同一harness，测量交互效应。

## 实验与结果
1. **In-distribution性能（2800项，严格过滤后2414项）**：推理+JSON路径96.86%（严格）/93.46%（完整），单token无推理路径96.75%/94.14%，未检测到精度差异；稳定性上JSON路径3.00%动作翻转 vs 单token 0.29%；延迟上JSON路径p50=8.89s vs 单token 85ms。
2. **密封基准两两实验（300项）**：推理开关对精确动作的影响——单token列+17.3pp [12.3, 23.0]，JSON列+14.7pp [10.0, 19.7]；格式效应在无推理时为0.0pp [-4.3, +4.0]，有推理时为+2.7pp [-0.7, +6.0]（受confound限制）。
3. **Counterfactual pairs验证**：推理带来+43.3pp [23.3, 63.3]提升；无推理时格式效应为-10.0pp [-20.0, -0.0]（偏好JSON），有推理时为-6.7pp。
4. **边界目标监督vs随机监督（3 seeds）**：任意工具召回率从60.2%提升至70.1%（+9.9pp，满足预注册阈值）；ANSWER/CALL_TOOL边界+17.5pp [6.3, 29.4]，CALL_TOOL/CONFIRM边界+20.0pp [5.7, 34.3]；未覆盖的REFUSE/ESCALATE边界无显著移动（+4.4pp [-3.7, +12.6]）。
5. **训练×推理交互（4单元格）**：推理在目标监督臂上值+11.7pp，在随机臂上值+16.6pp，交互效应-4.9pp [-8.1, -1.7]（所有3对种子符号一致），支持部分替代假设。
6. **跨语言迁移**：英语渲染比韩语低-0.82pp [-2.68, +0.93]，在预注册保留阈值（不劣于-5pp）内；目标监督增益在两种语言中点估计相似（韩语+3.03pp，英语+2.91pp）但不稳定。
7. **开放重渲染验证**：推理效应（+21.8pp）和英语迁移结论复制；训练×推理交互保持负号但不稳定（-3.4pp [-6.8, +0.1]）。

## 相关工作脉络
1. **结构化生成 vs 约束解码**：Willard & Louf (2023)、Geng et al. (2023)、Dong et al. (2025) 研究如何高效生成有效结构，本文在上游提问：当结构是7种已知动作之一时，生成步骤是否可移除及其成本。
2. **任务导向对话**：Hosseini-Asl et al. (2020)、Peng et al. (2021) 显示单一生成模型优于模块化pipeline，本文声称更窄：固定模型和训练，仅移除动作符号的自回归发射。
3. **推理前置动作**：Wei et al. (2022) 的Chain-of-Thought、Yao et al. (2023) 的ReAct建立显式中间计算可改善决策，本文的因子设计与这些工作不同在于将监督训练和推理时间视为可分别测量的资源。
4. **工具使用与策略代理**：BFCL (Patil et al. 2025)、Tool-Sandbox (Lu et al. 2025)、τ-bench (Yao et al. 2025) 评估工具调用，本文的7动作决策隔离了这些系统通常嵌入更长轨迹中的动作选择边界。
5. **标签由模拟器计算的基准**：WebArena (Zhou et al. 2024)、SWE-bench (Jimenez et al. 2024) 通过环境状态或可执行测试判定成功，本文应用相同原则到策略决策并添加证明和counterfactual pairs。
6. **决策后推理而非生成决策**：Jeeves-9B (Waltz 2026) 将指针头附加到9B模型并报告类似形状的推理增益，本文的指针头臂是诊断性而非方法性贡献。

## 局限性与未来方向
1. **单一模型与构造策略家族**：所有结果来自单一27B韩语企业模型和一组构造策略，密封基准虽标签可辩护但生态效度有限，需要在真实企业策略上验证。
2. **推理开时的输出格式比较受confound**：单token臂和JSON臂的推理prompt不同，后验控制（Section 13）未能分离，因为输出指令本身改变了推理轨迹（中位数从964token降至280token，3.4倍差异）。
3. **目标vs随机比较移动类别先验**：治疗臂接收3.4倍的CALL_TOOL比例，虽后验标签匹配控制（C1）反对类别先验作为充分解释，但未能彻底排除。
4. **种子与重训练不确定性未在区间内**：边界目标比较的区间固定3对种子重采样项目，未携带重训练不确定性；推理+JSON单元格为单次运行，携带运行级噪声。
5. **公开重渲染是后验且被筛选**：重渲染在已知原始结果后生成， gates不均匀拒绝了52项（特别是多工具场景仅38%保留），公开基准比主要声明依赖的基准容易3.9pp。
6. **未来方向**：在真实企业策略上评估、构建匹配推理的输出格式实验、探索自适应控制器（论文提及AUC 0.922的不一致信号）。

## 研究启发与可借鉴点
1. **受控因子实验设计**：将通常捆绑的优化变量（输出格式、推理时间）分离为2×2因子，可准确识别各成分的真实价值，避免归因错误。
2. **密封可执行基准构建**：通过隐藏策略+确定性执行器+外部渲染+恢复/挑战门的流水线，构建无人工标注污染、标签可辩护的测试集，适合高价值决策场景评估。
3. **监督与推理的替代关系发现**：监督训练和推理时间在未见边界上呈部分替代而非互补，指导资源分配：当推理不可用时优先边界目标数据，推理可用时监督边际收益递减。
4. **零样本跨语言迁移审计设计**：保持潜在因子不变仅改变表面语言，分离语言效应与任务理解，为本团队多语言Agent评估提供参考。
5. **操作路径比较而非接口比较**：明确区分"两个操作路径的比较"与"两个输出接口的纯净估计"，避免过度解读混杂实验的结果。

## 关键术语表
**Sealed benchmark**：密封基准，标签由隐藏可执行策略计算而非人工标注，内容哈希封存，自然语言渲染由外部模型完成且不接触标签。
**Reasoning**：推理时间，模型在决策前生成的中间token序列（中位数542个），不展示给下游系统，通过chat template的enable_thinking开关控制。
**Action-output generation**：动作输出生成，自回归发射动作对象本身（如JSON），区别于推理生成。
**Constrained single token**：约束单token，动作词汇映射为7个单token，决策为第一步logits上的argmax，输出1个token。
**Boundary-targeted supervision**：边界目标监督，包含contrastive pairs的训练数据，专门覆盖answer vs call tool、call tool vs confirm等决策边界。
**Counterfactual pairs**：counterfactual对，单个生成条件翻转的成对项，用于验证模型是否响应正确变量而非表面特征。
**Partial substitutes**：部分替代关系，监督训练和推理时间对精度的贡献非独立可加，监督的边际价值在推理可用时降低。
**Item-clustered bootstrap**：项目聚簇引导 Bootstrap，重采样时同项目的3个种子一起移动，估计seed-averaged paired difference的置信区间。

## 可复现要素
- **数据集**：密封基准原始渲染（内容哈希090fedfed3660b34...）未公开，公开开放重渲染（248项）及所有300项潜在场景定义、可执行金标签和证明；HuggingFace: https://huggingface.co/datasets/ThakiCloud/EnterpriseOps-Blind-A
- **代码/权重**：基础模型Qwen3.8-27B-Human-KO-Enterprise-v0.1及边界目标checkpoint（合并完整权重）开源；https://huggingface.co/ThakiCloud/Qwen3.8-27B-Human-KO-Enterprise-Boundary
- **训练语料**：未公开（内部和许可来源）
- **关键超参**：LoRA rank=16, alpha=32, dropout=0.05；学习率LoRA 5e-5/head 1e-3；bf16精度；vLLM v0.28.0引擎；max-model-len=8192
